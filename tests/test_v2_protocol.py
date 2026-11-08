"""Modelos, seleção, ablação e métricas da V2 sobre séries sintéticas."""

import numpy as np
import pandas as pd
import pytest

from energy_prediction.evaluation.protocol import PERSISTENCE_CONFIG, run_protocol, score
from energy_prediction.features.cpu_features import (
    PERSISTENCE_INPUT,
    ablation_columns,
    assert_equal_information,
    build_feature_table,
    build_group_specs,
    drop_incomplete_origins,
    horizon_steps,
    max_window_steps,
    target_column,
)
from energy_prediction.models.registry import (
    ALLOWED_MODELS,
    PersistenceModel,
    RidgeModel,
    UnknownModelError,
    make_model,
    param_grid,
    validate_model_names,
)
from energy_prediction.validation.rolling_origin import make_splits, required_purge_steps
from synthetic import make_series

STEP = 300.0
GROUPS = {
    "cpu": {"columns": ["cpu"], "lags": ["5min", "10min", "30min"], "windows": ["15min", "30min"]},
    "mem": {"columns": ["mem", "mem_gps"], "lags": ["5min"], "windows": ["15min"]},
    "net": {"columns": ["net_in", "net_out"], "lags": ["5min"], "windows": []},
    "disk": {"columns": ["disk"], "lags": ["5min"], "windows": []},
}
ABLATION = {"A": ["cpu"], "B": ["cpu", "mem"], "C": ["cpu", "mem", "net", "disk"]}
HORIZONS = ["5min", "15min", "30min"]
RIDGE_ONLY = {"names": ["persistence", "ridge"], "ridge": {"alphas": [1.0]}}
BOTH = {
    "names": ["persistence", "ridge", "xgboost"],
    "ridge": {"alphas": [0.1, 10.0, 1000.0]},
    "xgboost": {
        "grid": {"max_depth": [2, 3], "n_estimators": [20]},
        "fixed": {"learning_rate": 0.1},
        "n_jobs": 1,
    },
}


def _prepare(series=None, n=600):
    series = make_series(n, step_seconds=STEP, seed=3) if series is None else series
    specs = build_group_specs(GROUPS, STEP)
    steps = horizon_steps(HORIZONS, STEP)
    columns = ablation_columns(specs, ABLATION)
    table = drop_incomplete_origins(build_feature_table(series, specs, steps), columns["C"], steps)
    purge = required_purge_steps(max(steps), max_window_steps(specs))
    splits = make_splits(
        table["origin_index"].to_numpy(),
        n_folds=3,
        validation_steps=40,
        test_steps=60,
        min_train_steps=80,
        purge_steps=purge,
        max_horizon_steps=max(steps),
        max_window_steps=max_window_steps(specs),
    )
    return table, columns, steps, splits


def _run(table, columns, steps, splits, cfg=RIDGE_ONLY, **kw):
    return run_protocol(table, columns, steps, splits, cfg, seed=42, step_seconds=STEP, **kw)


def test_only_persistence_ridge_and_xgboost_are_allowed():
    assert ALLOWED_MODELS == ("persistence", "ridge", "xgboost")
    for name in ("lstm", "tcn", "random_forest", "transformer", "hist_gradient_boosting"):
        with pytest.raises(UnknownModelError):
            validate_model_names(["persistence", name])
        with pytest.raises(UnknownModelError):
            make_model(name, {}, feature_names=[PERSISTENCE_INPUT], seed=0)


def test_disallowed_model_fails_before_any_fit():
    table, columns, steps, splits = _prepare()
    with pytest.raises(UnknownModelError):
        _run(table, columns, steps, splits, cfg={"names": ["persistence", "lstm"], "ridge": {"alphas": [1.0]}})


def test_persistence_predicts_cpu_at_origin():
    table, columns, _, _ = _prepare()
    cols = columns["A"]
    pred = PersistenceModel(cols).predict(table[cols].to_numpy())
    assert np.array_equal(pred, table["cpu_t"].to_numpy())
    with pytest.raises(ValueError):
        PersistenceModel(["cpu_lag_1"])


def test_ridge_scaler_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    train = rng.normal(10, 2, size=(80, 3))
    validation = rng.normal(500, 50, size=(30, 3))
    model = RidgeModel(1.0).fit(train, train.sum(axis=1))
    assert model.scaler_.mean_ == pytest.approx(train.mean(axis=0))
    assert model.scaler_.scale_ == pytest.approx(train.std(axis=0))
    model.predict(validation)
    assert model.scaler_.mean_ == pytest.approx(train.mean(axis=0))


def test_ridge_alpha_belongs_to_grid_and_is_the_validation_minimum():
    table, columns, steps, splits = _prepare()
    cfg = {"names": ["ridge"], "ridge": {"alphas": [1000.0, 0.1, 10.0]}}
    result = _run(table, columns, steps, splits, cfg)
    grid = [p["alpha"] for p in param_grid("ridge", cfg["ridge"])]
    assert grid == [0.1, 10.0, 1000.0]
    for selection in result.selections:
        assert selection["best_params"]["alpha"] in grid
        errors = selection["grid_validation_mae"]
        assert errors[grid.index(selection["best_params"]["alpha"])] == min(errors)


def test_xgboost_is_deterministic_and_seeded():
    pytest.importorskip("xgboost")
    table, columns, steps, splits = _prepare()
    cols = columns["B"]
    x, y = table[cols].to_numpy(), table[target_column(steps[0])].to_numpy()
    params = {"max_depth": 3, "n_estimators": 30}
    cfg = {"n_jobs": 1}
    first = make_model("xgboost", params, feature_names=cols, seed=7, model_cfg=cfg).fit(x[:300], y[:300]).predict(x[300:])
    second = make_model("xgboost", params, feature_names=cols, seed=7, model_cfg=cfg).fit(x[:300], y[:300]).predict(x[300:])
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-9)


def test_xgboost_grid_selection_uses_only_validation_folds():
    pytest.importorskip("xgboost")
    table, columns, steps, splits = _prepare()
    result = _run(table, columns, steps, splits, BOTH)
    xgb = [s for s in result.selections if s["model"] == "xgboost"]
    assert len(xgb) == len(steps) * len(columns)
    for selection in xgb:
        assert len(selection["grid_validation_mae"]) == 2
        assert selection["best_params"]["max_depth"] in (2, 3)
    final_fits = [f for f in result.fit_log if f["phase"] == "final"]
    assert all(f["phase"] in {"final"} or f["phase"].startswith("fold_") for f in result.fit_log)
    assert len(final_fits) == len(steps) * len(columns) * 2  # ridge + xgboost


def test_independent_fit_per_horizon_and_expected_fit_count():
    table, columns, steps, splits = _prepare()
    result = _run(table, columns, steps, splits)
    folds = len(splits.folds)
    assert len(result.fit_log) == len(steps) * 1 * len(columns) * (folds + 1)
    for h in steps:
        per_horizon = [f for f in result.fit_log if f["horizon_steps"] == h]
        assert len(per_horizon) == len(columns) * (folds + 1)


def test_ablation_configurations_share_origins_and_persistence_runs_once():
    table, columns, steps, splits = _prepare()
    result = _run(table, columns, steps, splits)
    predictions = result.predictions
    for h in steps:
        by_config = {}
        for config in columns:
            part = predictions[(predictions["model"] == "ridge") & (predictions["config"] == config) & (predictions["horizon_steps"] == h)]
            by_config[config] = {s: sorted(g["origin_index"]) for s, g in part.groupby("split")}
        assert by_config["A"] == by_config["B"] == by_config["C"]
        persistence = predictions[(predictions["model"] == "persistence") & (predictions["horizon_steps"] == h)]
        assert set(persistence["config"]) == {PERSISTENCE_CONFIG}
        assert persistence["split"].nunique() == len(splits.folds) + 1
        assert persistence.groupby("split")["origin_index"].apply(lambda s: s.is_unique).all()
        ridge_a = predictions[(predictions["model"] == "ridge") & (predictions["config"] == "A") & (predictions["horizon_steps"] == h)]
        merged = ridge_a.merge(persistence, on=["split", "origin_index"], suffixes=("_m", "_p"))
        assert len(merged) == len(ridge_a)


def test_metric_grain_rows_by_model_horizon_config_fold():
    table, columns, steps, splits = _prepare()
    result = _run(table, columns, steps, splits)
    metrics = result.metrics
    per_key = len(splits.folds) + 3  # folds + média + desvio + teste final
    ridge = metrics[metrics["model"] == "ridge"]
    assert len(ridge) == len(steps) * len(columns) * per_key
    assert {"aggregate_mean", "aggregate_std", "test_final", "fold_0"} <= set(metrics["split"])
    persistence = metrics[metrics["model"] == "persistence"]
    non_std = persistence[persistence["split"] != "aggregate_std"]
    assert non_std["mase"].to_numpy() == pytest.approx(1.0)
    assert non_std["skill_score"].to_numpy() == pytest.approx(0.0)
    assert {"mae", "rmse", "mase", "skill_score", "n", "horizon_seconds"} <= set(metrics.columns)


def test_score_marks_undefined_when_reference_error_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    result = score(y, y + 1, y)
    assert np.isnan(result["mase"]) and np.isnan(result["skill_score"])
    assert result["mae"] == pytest.approx(1.0)
    same = score(y, y + 1, y + 1)
    assert same["mase"] == pytest.approx(1.0) and same["skill_score"] == pytest.approx(0.0)


def test_ridge_on_cpu_at_origin_only_is_close_to_persistence():
    series = make_series(700, step_seconds=STEP, seed=0, random_walk=True)
    table, _, steps, splits = _prepare(series)
    columns = {"S": [PERSISTENCE_INPUT]}
    assert_equal_information([PERSISTENCE_INPUT], columns)
    result = _run(table, columns, steps, splits, {"names": ["persistence", "ridge"], "ridge": {"alphas": [0.1]}})
    predictions = result.predictions
    for h in steps:
        part = predictions[(predictions["split"] == "test_final") & (predictions["horizon_steps"] == h)]
        ridge = part[part["model"] == "ridge"].sort_values("origin_index")["y_pred"].to_numpy()
        naive = part[part["model"] == "persistence"].sort_values("origin_index")["y_pred"].to_numpy()
        # Em passeio aleatório o Ridge aprende inclinação ~1; a diferença fica pequena frente ao desvio da série.
        assert np.sqrt(np.mean((ridge - naive) ** 2)) / series["cpu"].std() < 0.15
    test = result.metrics[(result.metrics["model"] == "ridge") & (result.metrics["split"] == "test_final")]
    assert test["skill_score"].abs().max() < 0.1
    assert (test["mase"] - 1).abs().max() < 0.1


def test_final_test_block_does_not_influence_selection_or_development_metrics():
    table, columns, steps, splits = _prepare()
    baseline = _run(table, columns, steps, splits, BOTH if pytest.importorskip("xgboost") else RIDGE_ONLY)
    altered = table.copy()
    rows = splits.test
    feature_cols = columns["C"]
    altered.loc[rows, feature_cols] = altered.loc[rows, feature_cols] * 50.0 + 1000.0
    for h in steps:
        altered.loc[rows, target_column(h)] = altered.loc[rows, target_column(h)] * -7.0 + 500.0
    other = _run(altered, columns, steps, splits, BOTH)
    keys = ["model", "config", "horizon_steps", "split"]
    left = baseline.metrics[baseline.metrics["split"] != "test_final"].sort_values(keys).reset_index(drop=True)
    right = other.metrics[other.metrics["split"] != "test_final"].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert [s["best_params"] for s in baseline.selections] == [s["best_params"] for s in other.selections]
    test_left = baseline.metrics[baseline.metrics["split"] == "test_final"]["mae"].to_numpy()
    test_right = other.metrics[other.metrics["split"] == "test_final"]["mae"].to_numpy()
    assert not np.allclose(test_left, test_right)
