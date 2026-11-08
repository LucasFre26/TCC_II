"""V3, grupos 4 e 6: incremento, espaço de estados filtrado, híbrido, quantis e intervalo conformal."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_prediction.evaluation.conformal import adjusted_upper, conformal_offset
from energy_prediction.evaluation.probabilistic import interval_metrics, pinball_loss, quantile_mean_metrics, quantile_peak_metrics
from energy_prediction.experiments.v3_context import build_context
from energy_prediction.features.cpu_features import PERSISTENCE_INPUT, target_column
from energy_prediction.models.delta import DeltaXGBoost, reconstruct
from energy_prediction.models.hybrid import CausalityError, HybridCorrector, assert_filtered
from v3_helpers import ROOT, fake_v2_reference, smoke_config

statsmodels = pytest.importorskip("statsmodels")

from energy_prediction.models.statespace import FilteredForecasts, StateSpaceForecaster, fit_parameters, filtered_forecasts  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    config = smoke_config()
    config["families"]["quantile"]["configs"] = ["A", "B"]
    return build_context(config, ROOT)


def test_delta_reconstruction_closed_case():
    assert reconstruct(40.0, 1.5) == pytest.approx(41.5)
    X = np.array([[40.0, 1.0], [50.0, 2.0], [60.0, 3.0], [70.0, 4.0]] * 10)
    y = X[:, 0] + 1.5
    model = DeltaXGBoost({"max_depth": 2, "n_estimators": 50, "learning_rate": 0.3}, feature_names=["cpu_t", "x"], seed=42, n_jobs=1)
    model.fit(X, y)
    assert np.allclose(model.predict(X), X[:, 0] + model.predict_increment(X))
    assert np.allclose(model.predict(X), y, atol=0.05)
    with pytest.raises(ValueError, match="cpu_t"):
        DeltaXGBoost({}, feature_names=["x"], seed=1, n_jobs=1)


def test_cpu_at_origin_in_every_configuration(ctx):
    assert all(PERSISTENCE_INPUT in columns for columns in ctx.columns.values())


def test_filtered_forecast_ignores_future_observations():
    rng = np.random.default_rng(0)
    y = 40 + np.cumsum(rng.normal(0, 1, 400))
    for spec in ("local_level", "local_linear_trend"):
        params = fit_parameters(y[:300], spec)
        base = filtered_forecasts(y, spec, params, [1, 3, 6])
        changed = y.copy()
        changed[251:] += 25.0  # observações posteriores à origem 250
        after = filtered_forecasts(changed, spec, params, [1, 3, 6])
        for h in (1, 3, 6):
            assert base.by_horizon[h][250] == pytest.approx(after.by_horizon[h][250])
            assert base.by_horizon[h][251] != pytest.approx(after.by_horizon[h][251])


def test_smoothed_state_is_never_computed(monkeypatch):
    from statsmodels.tsa.statespace.mlemodel import MLEModel

    def forbidden(*_, **__):
        raise AssertionError("estado suavizado não pode ser usado")

    monkeypatch.setattr(MLEModel, "smooth", forbidden)
    y = 40 + np.cumsum(np.random.default_rng(1).normal(0, 1, 200))
    forecaster = StateSpaceForecaster(y, horizons=[1, 6])
    assert np.isfinite(forecaster.forecasts("local_linear_trend", 150).by_horizon[6][150])


def test_hybrid_rejects_smoothed_residuals_and_uses_only_past():
    smoothed = FilteredForecasts("local_level", np.ones(2), {1: np.zeros(5)}, kind="smoothed")
    with pytest.raises(CausalityError):
        assert_filtered(smoothed)
    corrector = HybridCorrector({"max_depth": 2, "n_estimators": 5, "learning_rate": 0.1}, seed=42, n_jobs=1)
    with pytest.raises(CausalityError):
        corrector.fit(np.zeros((5, 1)), np.zeros(5), np.zeros(5), smoothed)

    y = 40 + np.cumsum(np.random.default_rng(2).normal(0, 1, 300))
    params = fit_parameters(y[:200], "local_level")
    residual = y[1:201] - filtered_forecasts(y, "local_level", params, [1]).by_horizon[1][:200]
    changed = y.copy()
    changed[202:] -= 30.0  # depois do alvo da última origem de treino (199 + 1)
    residual_after = changed[1:201] - filtered_forecasts(changed, "local_level", params, [1]).by_horizon[1][:200]
    assert np.allclose(residual, residual_after)


def test_residual_family_loses_to_perfect_v2_and_still_writes(ctx, tmp_path: Path):
    from energy_prediction.experiments.v3_families import residual

    fake_v2_reference(ctx, tmp_path / "v2", perfect=True)
    ctx.cfg["reference"]["v2_dir"] = str(tmp_path / "v2")
    out = tmp_path / "residual"
    out.mkdir()
    try:
        summary = residual.run(ctx, out)
    finally:
        ctx.cfg["reference"]["v2_dir"] = None
    metrics = pd.read_csv(out / "metrics.csv")
    test = metrics[metrics["split"] == "test_final"]
    grain = test.groupby(["model", "config", "horizon_steps"]).size()
    assert (grain == 1).all()
    assert set(test["model"]) == {"persistence", "xgboost_delta", "statespace", "hybrid"}
    assert set(test.loc[test["model"] == "statespace", "config"]) == {"A"}
    assert set(test.loc[test["model"] == "hybrid", "config"]) == {"A", "B", "C"}
    assert (test["mae"] > 0).all()  # todos perdem para a V2 perfeita
    boot = pd.read_csv(out / "bootstrap_ci.csv")
    versus = boot[boot["comparison"] == "new_vs_v2_xgboost_A"]
    assert set(versus["block_steps"]) == {6, 12, 24} and (versus["mean_diff"] > 0).all()
    assert {"hybrid_vs_statespace", "ablation_vs_A", "d_vs_a", "new_vs_persistence_A"} <= set(boot["comparison"])
    assert (boot.groupby(["comparison", "horizon_steps", "model_a", "config_a", "model_b"])["primary_block"].sum() == 1).all()
    d = pd.read_csv(out / "d_contrast_metrics.csv")
    assert set(d["config"]) == {"A", "D"}
    selections = json.loads((out / "selections.json").read_text())
    assert all(s["rule"] == "inner_validation" for s in selections)
    assert summary["n_metric_rows"] == len(metrics)


def test_pinball_and_peak_metrics_on_known_vector():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    q = np.array([12.0, 18.0, 30.0, 35.0])
    # erros y - q: -2, 2, 0, 5
    assert pinball_loss(y, q, 0.9) == pytest.approx((0.1 * 2 + 0.9 * 2 + 0 + 0.9 * 5) / 4)
    mean = quantile_mean_metrics(y, q, 0.9)
    assert mean["underestimation_frequency"] == pytest.approx(0.5)
    assert mean["underestimation_magnitude"] == pytest.approx(3.5)
    assert mean["coverage_below_quantile"] == pytest.approx(0.5)
    peak = quantile_peak_metrics(y, q, 0.9)
    assert peak["peak_n"] == 1 and peak["peak_mae"] == pytest.approx(5.0) and peak["peak_bias"] == pytest.approx(5.0)
    assert peak["max_abs_error"] == pytest.approx(5.0)
    interval = interval_metrics(y, np.array([9.0, 21.0, 29.0, 36.0]), np.array([11.0, 25.0, 31.0, 39.0]))
    assert interval["interval_coverage"] == pytest.approx(0.5) and interval["mean_width"] == pytest.approx(2.75)


def test_conformal_needs_minimum_residuals_and_uses_finite_sample_rank():
    few = conformal_offset(np.arange(150.0), 0.05, min_residuals=200)
    assert few["applied"] is False and few["offset"] is None and adjusted_upper(np.zeros(3), few) is None
    scores = np.arange(1.0, 201.0)
    calibration = conformal_offset(scores, 0.05, min_residuals=200)
    assert calibration["rank"] == 191 and calibration["offset"] == pytest.approx(191.0)


def test_quantile_family_interval_ignores_test_labels_and_q50_stays_out_of_point_table(ctx, tmp_path: Path):
    from energy_prediction.experiments.v3_families import quantile

    first = tmp_path / "q1"
    first.mkdir()
    quantile.run(ctx, first)
    original = ctx.table.copy()
    try:
        for steps in ctx.steps:
            column = target_column(steps)
            ctx.table.loc[ctx.table.index[ctx.splits.test], column] += 50.0
        second = tmp_path / "q2"
        second.mkdir()
        quantile.run(ctx, second)
    finally:
        ctx.table = original
    p1 = pd.read_csv(first / "predictions.csv")
    p2 = pd.read_csv(second / "predictions.csv")
    upper = lambda p: p[(p["mode"] == "conformal_upper") & (p["split"] == "test_final")]["y_pred"].to_numpy()  # noqa: E731
    assert len(upper(p1)) > 0 and np.allclose(upper(p1), upper(p2))
    metrics = pd.read_csv(first / "metrics.csv")
    assert set(metrics["model"]) == {"xgboost_quantile"}
    assert {"pinball", "coverage_below_quantile"} <= set(metrics.columns)
    peaks = pd.read_csv(first / "peak_metrics.csv")
    assert {"peak_threshold", "peak_mae", "peak_bias", "max_abs_error", "p95_abs_error"} <= set(peaks.columns)
    assert "peak_mae" not in metrics.columns
    selections = json.loads((first / "selections.json").read_text())
    assert {s["selection_metric"] for s in selections} == {"pinball"}
    calibration = json.loads((first / "calibration.json").read_text())
    assert "garantia" in calibration["statement"]
