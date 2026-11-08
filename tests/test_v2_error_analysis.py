"""Erros extremos, erro nos picos, block bootstrap e antecedência das exógenas."""

import numpy as np
import pandas as pd
import pytest

from energy_prediction.evaluation.error_analysis import (
    block_indices,
    bootstrap_table,
    default_pairs,
    error_extremes_table,
    extreme_errors,
    moving_block_bootstrap_diff,
    peak_errors,
)
from energy_prediction.evaluation.exogenous_lead import lead_correlations


def test_extreme_errors_on_known_vector():
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_pred = np.array([12.0, 7.0, 10.0, 9.0])  # erros: -2, +3, 0, +1
    out = extreme_errors(y_true, y_pred)
    assert out["max_abs_error"] == 3.0
    assert out["max_underestimation"] == 3.0
    assert out["underestimation_fraction"] == 0.5
    assert out["p95_abs_error"] == pytest.approx(np.quantile([2, 3, 0, 1], 0.95))


def test_max_underestimation_is_zero_when_model_only_overestimates():
    assert extreme_errors(np.zeros(3), np.ones(3))["max_underestimation"] == 0.0


def test_peak_bias_is_positive_when_peaks_are_underestimated():
    y_true = np.arange(100, dtype=float)
    y_pred = np.where(y_true >= 90, y_true - 5, y_true)
    out = peak_errors(y_true, y_pred, quantile=0.9)
    assert out["peak_n"] == 10
    assert out["peak_bias"] == pytest.approx(5.0)
    assert out["peak_mae"] == pytest.approx(5.0)


def test_block_indices_are_contiguous_within_blocks():
    idx = block_indices(50, 5, np.random.default_rng(0), n_boot=3)
    assert idx.shape == (3, 50)
    for row in idx:
        for start in range(0, 50, 5):
            assert np.all(np.diff(row[start:start + 5]) == 1)


def test_block_larger_than_series_fails():
    with pytest.raises(ValueError):
        block_indices(4, 5, np.random.default_rng(0))


def test_bootstrap_of_identical_losses_is_zero_and_not_significant():
    loss = np.random.default_rng(1).random(200)
    out = moving_block_bootstrap_diff(loss, loss, block=10, n_boot=200, seed=42)
    assert out["mean_diff"] == out["ci_low"] == out["ci_high"] == 0.0
    assert not out["significant"]


def test_bootstrap_is_reproducible_and_detects_clear_difference():
    rng = np.random.default_rng(2)
    a, b = rng.random(300), rng.random(300) + 1.0
    first = moving_block_bootstrap_diff(a, b, block=12, n_boot=500, seed=42)
    second = moving_block_bootstrap_diff(a, b, block=12, n_boot=500, seed=42)
    assert first == second
    assert first["ci_high"] < 0 and first["significant"]


def _predictions(n=96):
    rng = np.random.default_rng(3)
    y = 40 + rng.normal(0, 5, n)
    rows = []
    for model, config, noise in [("persistence", "-", 3.0), ("ridge", "A", 2.0), ("ridge", "B", 2.0), ("xgboost", "A", 1.0)]:
        for steps in (1, 3):
            rows.append(pd.DataFrame({
                "split": "test_final", "model": model, "config": config, "horizon_steps": steps,
                "origin_index": np.arange(n), "y_true": y, "y_pred": y + rng.normal(0, noise, n),
            }))
    return pd.concat(rows, ignore_index=True)


def test_default_pairs_compare_with_persistence_and_with_config_a():
    pairs = default_pairs(_predictions())
    assert (("ridge", "A"), ("persistence", "-")) in pairs
    assert (("xgboost", "A"), ("persistence", "-")) in pairs
    assert (("ridge", "B"), ("ridge", "A")) in pairs
    assert len(pairs) == 4


def test_tables_have_one_row_per_horizon_and_pair_or_model():
    predictions = _predictions()
    intervals = bootstrap_table(predictions, default_pairs(predictions), block=6, n_boot=100, seed=42)
    assert len(intervals) == 2 * 4
    assert (intervals["ci_low"] <= intervals["mean_diff"]).all()
    extremes = error_extremes_table(predictions, peak_quantile=0.9)
    assert len(extremes) == 2 * 4
    assert {"max_abs_error", "p95_abs_error", "peak_mae", "peak_bias"} <= set(extremes.columns)


def test_lead_correlation_detects_exogenous_that_anticipates_cpu():
    rng = np.random.default_rng(4)
    driver = np.cumsum(rng.normal(0, 1, 2000))
    series = pd.DataFrame({
        "cpu": np.roll(driver, 3),           # a CPU repete o driver 3 passos depois
        "lead": driver,                      # exógena que antecede a CPU
        "noise": rng.normal(0, 1, 2000),     # exógena sem relação
    }).iloc[10:].reset_index(drop=True)
    out = lead_correlations(series, target="cpu", exogenous=["lead", "noise"], horizons=[3])
    assert out["exogenous"]["lead"]["lead_correlation"]["3"] > 0.5
    assert abs(out["exogenous"]["noise"]["lead_correlation"]["3"]) < 0.1
    assert out["target_autocorrelation"]["3"] > 0.9
