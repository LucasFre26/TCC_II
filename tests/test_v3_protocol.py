"""V3, grupo 2: seleção aninhada, divisão interna inviável, auditoria da população e alinhamento das origens."""

import numpy as np
import pandas as pd
import pytest

from energy_prediction.evaluation.nested import (
    AlignmentError,
    InfeasibleInnerSplitError,
    build_inner_splits,
    check_alignment,
    eligible_for_window,
    evaluate_grid,
    inner_split,
)
from energy_prediction.evaluation.metrics import mae
from energy_prediction.models.neural.windows import window_tensor
from energy_prediction.preprocessing.cluster import resolve_machine_info
from energy_prediction.preprocessing.population_audit import PopulationAuditError, audit_population
from energy_prediction.validation.rolling_origin import describe_splits, make_splits
from synthetic import make_meta_frame
from v3_helpers import constant_fit_predict

# Mesma geometria da análise principal da V2: origens 12..2297, 5 folds de 12 h, teste de 24 h.
ORIGINS = np.arange(12, 2298)


def _v2_splits(origins=ORIGINS):
    return make_splits(origins, n_folds=5, validation_steps=144, test_steps=288, min_train_steps=576, purge_steps=18, max_horizon_steps=6, max_window_steps=12)


def _inner(splits, origins=ORIGINS):
    inner, dropped = build_inner_splits(origins, splits, validation_steps=144, purge_steps=18, min_train_steps=576)
    assert dropped == []
    return inner


def test_v2_geometry_is_preserved():
    splits = _v2_splits()
    assert splits.test_range == (2010, 2297)
    assert [f.validation_range for f in splits.folds][0] == (1272, 1415)
    inner = _inner(splits)
    assert inner["fold_0"].validation_range == (1110, 1253)
    assert inner["fold_0"].train_range == (12, 1091)


def test_outer_validation_labels_do_not_select_but_inner_labels_do():
    splits = _v2_splits()
    inner = _inner(splits)
    y = np.full(len(ORIGINS), 10.0)
    constants = [10.0, 20.0]
    base = evaluate_grid(2, constant_fit_predict(constants), y, splits, inner, metric=mae)
    assert base.best == 0 and base.rule == "inner_validation"

    # Rótulos só da validação externa do último fold (nunca usados em validação interna) mudam.
    outer_only = y.copy()
    outer_only[splits.folds[-1].validation] = 20.0
    assert evaluate_grid(2, constant_fit_predict(constants), outer_only, splits, inner, metric=mae).best == 0

    inner_changed = y.copy()
    for split in inner.values():
        inner_changed[split.validation] = 20.0
    assert evaluate_grid(2, constant_fit_predict(constants), inner_changed, splits, inner, metric=mae).best == 1


def test_tie_goes_to_first_grid_point_and_refits_happen_once():
    splits = _v2_splits()
    inner = _inner(splits)
    y = np.full(len(ORIGINS), 15.0)
    calls = []

    def fit_predict(g, train, predict, **_):
        calls.append((g, len(train), len(predict)))
        return np.full(len(predict), [10.0, 20.0][g]), {}

    result = evaluate_grid(2, fit_predict, y, splits, inner, metric=mae)
    assert result.best == 0
    assert len(calls) == 2 * 5 + 5 + 1  # seleção + validação externa + teste
    assert calls[-1][2] == len(splits.test)


def test_nested_off_uses_v2_rule():
    splits = _v2_splits()
    result = evaluate_grid(2, constant_fit_predict([1.0, 2.0]), np.ones(len(ORIGINS)), splits, None, metric=mae)
    assert result.rule == "outer_fold_validation_v2" and result.best == 0


def test_infeasible_inner_split_reports_both_sizes():
    origins = np.arange(0, 300)
    with pytest.raises(InfeasibleInnerSplitError) as error:
        inner_split(origins, np.arange(200), 200, validation_steps=144, purge_steps=18, min_train_steps=576, label="fold_0")
    message = str(error.value)
    assert "38" in message and "576" in message
    assert "teste" not in message.lower()


def test_drop_fold_policy_only_when_declared():
    origins = np.arange(12, 1300)
    splits = make_splits(origins, n_folds=3, validation_steps=144, test_steps=144, min_train_steps=100, purge_steps=18, max_horizon_steps=6, max_window_steps=12)
    with pytest.raises(InfeasibleInnerSplitError):
        build_inner_splits(origins, splits, validation_steps=144, purge_steps=18, min_train_steps=576)
    inner, dropped = build_inner_splits(origins, splits, validation_steps=144, purge_steps=18, min_train_steps=576, infeasible_policy="drop_fold")
    assert dropped and all(d["fold"] not in inner for d in dropped)


def test_backward_window_of_144_steps_does_not_move_the_test():
    splits = _v2_splits()
    valid = np.ones(2304, dtype=bool)
    eligible = eligible_for_window(ORIGINS, valid, 144)
    evaluation = np.concatenate([*(f.validation for f in splits.folds), splits.test])
    assert eligible[evaluation].all()
    assert int((~eligible).sum()) == 143 - 12
    assert _v2_splits().test_range == (2010, 2297)
    inner = build_inner_splits(ORIGINS, splits, validation_steps=144, purge_steps=18, min_train_steps=576, eligible=eligible)[0]
    assert inner["fold_0"].validation_range == (1110, 1253)
    assert ORIGINS[inner["fold_0"].train].min() == 143

    matrix = np.arange(2304 * 2, dtype=float).reshape(2304, 2)
    before = window_tensor(matrix, np.array([2010]), 144)
    matrix[2011:] = -1.0  # observação posterior à origem
    assert np.array_equal(before, window_tensor(matrix, np.array([2010]), 144))


def test_alignment_against_preserved_folds():
    splits = _v2_splits()
    table = pd.DataFrame({"origin_index": ORIGINS, "y_h1": np.ones(len(ORIGINS))})
    reference = describe_splits(table, splits, "y_h1")
    assert check_alignment(splits, reference, [2010, 2297])["test_range"] == [2010, 2297]
    with pytest.raises(AlignmentError):
        check_alignment(splits, reference, [2009, 2297])
    shifted = [dict(r) for r in reference]
    shifted[0]["evaluation_range"] = [1273, 1416]
    with pytest.raises(AlignmentError):
        check_alignment(splits, shifted, None)


def _audit_fixture(inactive: str = "m_3"):
    machines = {f"m_{i}": 96 for i in range(6)}
    meta = make_meta_frame(machines, status={inactive: "OFFLINE"})
    meta = pd.concat([make_meta_frame(machines), meta.loc[meta["machine_id"] == inactive]], ignore_index=True)
    rng = np.random.default_rng(1)
    grid = pd.DataFrame(
        [{"machine_id": m, "bin": b, "cpu": 30 + 10 * i + rng.normal(), "mem": 50.0, "mem_gps": np.nan, "net_in": 1.0, "net_out": 1.0, "disk": 1.0} for i, m in enumerate(machines) for b in range(40)]
    )
    info = resolve_machine_info(meta, active_statuses=["USING"])
    return grid, meta, info


def test_audit_keeps_inactive_machine_in_main_and_out_of_cohort():
    grid, meta, info = _audit_fixture()
    audit = audit_population(grid, meta, info, step_seconds=300.0, cohort_cpu_num=None, cohort_min_machines=2)
    assert audit["n_machines_main"] == 6 and audit["n_machines_cohort"] == 5
    assert audit["weight_main"] == 576 and audit["weight_cohort"] == 480
    [only] = audit["only_in_main"]
    assert only["machine_id"] == "m_3" and only["active"] is False
    assert only["status_counts"] == {"USING": 1, "OFFLINE": 1}
    assert audit["series_comparison"]["series_identical"] is False
    assert audit["n_inactive_in_main"] == 1


def test_expected_weights_must_match_or_be_explained():
    grid, meta, info = _audit_fixture()
    kwargs = dict(step_seconds=300.0, cohort_cpu_num=None, cohort_min_machines=2)
    check = audit_population(grid, meta, info, expected_weights={"main": 576, "cohort": 480}, **kwargs)["expected_weights_check"]
    assert check["main_matches"] and check["cohort_matches"]
    explained = audit_population(grid, meta, info, expected_weights={"main": 1000, "cohort": 904}, **kwargs)["expected_weights_check"]
    assert explained["difference_explained_by_listed_ids"] and not explained["main_matches"]
    with pytest.raises(PopulationAuditError):
        audit_population(grid, meta, info, expected_weights={"main": 1000, "cohort": 990}, **kwargs)
