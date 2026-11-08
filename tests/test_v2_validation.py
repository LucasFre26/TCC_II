"""Folds rolling-origin, purga e teste final isolado."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from energy_prediction.validation.rolling_origin import (
    InfeasibleFoldsError,
    PurgeTooSmallError,
    describe_splits,
    make_splits,
    required_purge_steps,
)

BASE = dict(
    n_folds=4,
    validation_steps=20,
    test_steps=30,
    min_train_steps=40,
    purge_steps=9,
    max_horizon_steps=6,
    max_window_steps=3,
)


def _splits(n=300, **overrides):
    params = {**BASE, **overrides}
    return make_splits(np.arange(n), **params)


def test_required_purge_is_horizon_plus_window():
    assert required_purge_steps(6, 12) == 18


def test_purge_below_minimum_is_an_error():
    with pytest.raises(PurgeTooSmallError, match="mínimo 9"):
        _splits(purge_steps=8)


def test_layout_sizes_and_final_test_at_the_end():
    splits = _splits(300)
    assert splits.test.tolist() == list(range(270, 300))
    assert splits.purge_steps == 9
    assert splits.development_range == (0, 270 - 9 - 1)
    assert len(splits.folds) == 4
    assert all(len(f.validation) == 20 for f in splits.folds)
    assert splits.folds[-1].validation_range[1] == splits.development_range[1]


def test_chronological_order_no_shuffle_and_expanding_training():
    splits = _splits(300)
    previous = None
    for fold in splits.folds:
        assert np.all(np.diff(fold.train) > 0) and np.all(np.diff(fold.validation) > 0)
        assert fold.train.max() < fold.validation.min()
        if previous is not None:
            assert set(previous.train) < set(fold.train)
            assert fold.validation.min() > previous.validation.max()
        previous = fold


def test_target_reach_stays_before_validation_and_test_in_every_fold():
    splits = _splits(300)
    horizon = BASE["max_horizon_steps"]
    for fold in splits.folds:
        assert fold.train.max() + horizon < fold.validation.min()
        assert fold.train.max() + horizon + BASE["max_window_steps"] < fold.validation.min()
        gap = fold.validation.min() - fold.train.max() - 1
        assert gap >= BASE["purge_steps"]
    dev_last = splits.development.max()
    assert dev_last + horizon < splits.test.min()
    assert splits.test.min() - dev_last - 1 >= BASE["purge_steps"]


def test_development_region_never_touches_the_test_block_or_its_purge():
    splits = _splits(300)
    forbidden_from = splits.test.min() - splits.purge_steps
    assert splits.development.max() < forbidden_from
    for fold in splits.folds:
        assert fold.validation.max() < forbidden_from


def test_matches_sklearn_time_series_split_as_oracle():
    splits = _splits(300)
    dev = splits.development
    tss = TimeSeriesSplit(n_splits=4, test_size=20, gap=BASE["purge_steps"])
    oracle = list(tss.split(dev))
    assert len(oracle) == len(splits.folds)
    for (train, test), fold in zip(oracle, splits.folds):
        assert dev[train].tolist() == fold.train.tolist()
        assert dev[test].tolist() == fold.validation.tolist()


def test_infeasible_series_reports_required_and_available():
    with pytest.raises(InfeasibleFoldsError, match=r"exige pelo menos \d+"):
        _splits(100)
    try:
        _splits(100)
    except InfeasibleFoldsError as error:
        assert "100 passos" in str(error)


def test_gaps_in_origin_index_keep_purge_in_time():
    index = np.array([i for i in range(300) if not 120 <= i < 130])
    splits = make_splits(index, **BASE)
    for fold in splits.folds:
        last_train_time = index[fold.train].max()
        first_val_time = index[fold.validation].min()
        assert first_val_time - last_train_time - 1 >= BASE["purge_steps"]


def test_unknown_strategy_and_bad_input_are_rejected():
    with pytest.raises(ValueError, match="não suportada"):
        _splits(strategy="sliding")
    with pytest.raises(ValueError):
        make_splits(np.array([3, 2, 1]), **BASE)


def test_descriptors_record_bounds_counts_and_target_stats():
    splits = _splits(300)
    table = pd.DataFrame({"origin_index": np.arange(300), "y_h1": np.linspace(0, 100, 300)})
    rows = describe_splits(table, splits, "y_h1")
    assert [r["split"] for r in rows] == [f"fold_{i}" for i in range(4)] + ["test_final"]
    required = {
        "train_range",
        "evaluation_range",
        "n_train",
        "n_evaluation",
        "train_target_mean",
        "train_target_std",
        "evaluation_target_mean",
        "evaluation_target_std",
    }
    for row in rows:
        assert required <= set(row)
    assert rows[-1]["n_evaluation"] == 30
    assert rows[-1]["evaluation_target_mean"] > rows[-1]["train_target_mean"]
