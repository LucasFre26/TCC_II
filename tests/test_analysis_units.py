"""Análise expandida: unidades com dados sintéticos (sem corrida real e sem treino)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_prediction.analysis import config as C
from energy_prediction.analysis import diagnostics, energy, export, integrity, latex, peaks, performance, statistics, style
from energy_prediction.energy.fan import FanModel
from energy_prediction.evaluation.error_analysis import moving_block_bootstrap_diff

ROOT = Path(__file__).resolve().parents[1]


def test_config_requires_keys_and_guards_protected_output(tmp_path):
    cfg = C.load_config(ROOT / "config/v3-analysis.yaml")
    broken = {k: v for k, v in cfg.items() if k != "statistics"}
    with pytest.raises(C.AnalysisConfigError, match="statistics"):
        C.validate(broken)
    bad = dict(cfg, run_dir="results/v2-full", output_subdir="x")
    with pytest.raises(C.ProtectedPathError):
        C.resolve_paths(bad, ROOT)
    inside = dict(cfg, output_subdir="residual/novo")
    with pytest.raises(C.ProtectedPathError):
        C.resolve_paths(inside, ROOT)


def test_point_metrics_on_known_vector():
    m = performance.point_metrics(np.array([1.0, -1.0, 2.0, -2.0]))
    assert m["mae_pp"] == pytest.approx(1.5) and m["rmse_pp"] == pytest.approx(np.sqrt(2.5))
    assert m["bias_obs_minus_pred_pp"] == pytest.approx(0.0) and m["underestimation_fraction"] == pytest.approx(0.5)
    assert m["max_abs_error_pp"] == pytest.approx(2.0) and m["median_abs_error_pp"] == pytest.approx(1.5)


def _long(models: dict[str, np.ndarray], y: np.ndarray, split="test_final", config="A") -> pd.DataFrame:
    rows = []
    for model, pred in models.items():
        for i, (yt, yp) in enumerate(zip(y, pred)):
            rows.append(
                {
                    "version": "V3", "family": "x", "model": model, "config": "-" if model == "persistence" else config, "mode": "direct", "population": "main", "analysis_set": "main",
                    "split": split, "horizon_steps": 1, "horizon_min": 5.0, "origin_index": 100 + i, "origin_time_s": (100 + i) * 300.0, "target_index": 101 + i,
                    "target_time_s": (101 + i) * 300.0, "target_time_h": (101 + i) / 12, "y_obs_cpu_pct": yt, "y_pred_cpu_pct": yp, "error_obs_minus_pred_pp": yt - yp, "abs_error_pp": abs(yt - yp),
                }
            )
    return pd.DataFrame(rows)


def test_persistence_reference_has_unit_mase_and_zero_skill():
    rng = np.random.default_rng(0)
    y = 40 + rng.normal(0, 3, 50)
    frame = pd.concat([_long({"persistence": y + rng.normal(0, 1, 50), "xgboost_v2": y + rng.normal(0, 0.5, 50)}, y, split=s) for s in ("fold_0", "test_final")])
    metrics, by_fold = performance.metrics_tables(frame)
    p = metrics[(metrics["model"] == "persistence") & (metrics["split"] == "test_final")].iloc[0]
    assert p["mase_vs_persistence"] == pytest.approx(1.0) and p["skill_score_rmse"] == pytest.approx(0.0) and p["improvement_vs_persistence_pct"] == pytest.approx(0.0)
    x = metrics[(metrics["model"] == "xgboost_v2") & (metrics["split"] == "test_final")].iloc[0]
    assert x["improvement_vs_xgboost_v2_pct"] == pytest.approx(0.0) and x["mase_vs_persistence"] < 1
    assert set(metrics["split"]) == {"validation_pooled", "test_final"} and set(by_fold["split"]) == {"fold_0"}


def test_bootstrap_matches_v2_function_and_holm_is_monotone():
    rng = np.random.default_rng(1)
    a, b = rng.normal(1, 1, 288), rng.normal(1.2, 1, 288)
    ours = statistics.bootstrap(a, b, block=12, n_boot=500, seed=42)
    ref = moving_block_bootstrap_diff(a, b, block=12, n_boot=500, seed=42)
    for key in ("mean_diff", "ci_low", "ci_high", "significant"):
        assert ours[key] == ref[key]
    assert 0 < ours["p_value_bootstrap"] <= 1
    p = np.array([0.01, 0.04, 0.03, 0.5])
    adj = statistics.holm(p)
    assert adj[0] == pytest.approx(0.04) and np.all(adj >= p) and adj[np.argsort(p)].tolist() == sorted(adj.tolist())


def test_target_time_alignment_and_duplicate_detection():
    frame = pd.DataFrame({"model": ["m", "m"], "config": ["A", "A"], "mode": ["direct", "direct"], "split": ["test_final", "test_final"], "horizon_steps": [6, 6], "origin_index": [2010, 2010]})
    assert not integrity.uniqueness(frame, "x")["passed"]
    from energy_prediction.analysis.normalize import _frame

    raw = pd.DataFrame({"model": ["m"], "config": ["A"], "split": ["test_final"], "horizon_steps": [6], "origin_index": [2010], "timestamp": [2010 * 300.0], "y_true": [40.0], "y_pred": [38.0]})
    out = _frame(raw, version="V3", family="", population="main", analysis_set="main", step=300.0)
    assert out["target_index"].iloc[0] == 2016 and out["target_time_s"].iloc[0] == 2016 * 300.0
    assert out["error_obs_minus_pred_pp"].iloc[0] == pytest.approx(2.0)


def test_metric_tampering_is_detected():
    preds = pd.DataFrame({"model": "m", "config": "A", "split": "test_final", "horizon_steps": 1, "origin_index": range(10), "y_true": np.arange(10.0), "y_pred": np.arange(10.0) + 1})
    metrics = pd.DataFrame({"model": ["m"], "config": ["A"], "horizon_steps": [1], "split": ["test_final"], "n": [10], "mae": [1.0], "rmse": [1.0]})
    assert integrity.metric_reproduction(metrics, preds, "ok", 1e-9)["passed"]
    assert not integrity.metric_reproduction(metrics.assign(mae=1.01), preds, "x", 1e-9)["passed"]


def test_windows_depend_only_on_observed_series():
    rng = np.random.default_rng(2)
    cpu = pd.Series(40 + np.cumsum(rng.normal(0, 1, 400)), index=np.arange(400))
    w = diagnostics.objective_windows(cpu, 50, 350, length=36, change_steps=6, step_seconds=300.0)
    assert list(w["window"]) == ["volatilidade", "subida", "queda", "estabilidade"]
    spans = sorted(zip(w["start_index"], w["end_index"]))
    assert all(b0 > a1 for (a0, a1), (b0, b1) in zip(spans, spans[1:]))
    again = diagnostics.objective_windows(cpu.copy(), 50, 350, length=36, change_steps=6, step_seconds=300.0)
    pd.testing.assert_frame_equal(w, again)


def test_regime_labels_are_shared_by_models():
    cpu = pd.Series(np.r_[np.full(100, 40.0), 40 + np.arange(20) * 3.0], index=np.arange(120))
    y = cpu.shift(-1).loc[100:115].to_numpy()
    frame = _long({"a": y - 1, "b": y + 1}, y)
    frame["origin_index"] = np.tile(np.arange(100, 116), 2)
    out, info = diagnostics.annotate(frame, cpu, dev_end=99, horizons=[1], n_bands=2, change_bins=[-1e9, 0, 1e9])
    assert (out.groupby("origin_index")["regime"].nunique() == 1).all()


def test_peak_events_alerts_and_false_alarms():
    cpu = pd.Series(np.full(60, 40.0), index=np.arange(60))
    cpu.loc[20] = 70.0
    cpu.loc[40:42] = 70.0
    ev = peaks.events(cpu, 5, 55, 60.0, merge_gap=1)
    assert len(ev) == 2 and ev["onset_observed"].all() and list(ev["duration_steps"]) == [1, 3]
    persistence = cpu.shift(1).loc[5:55]
    s = peaks.alert_scores(persistence, 1, ev, low=5, high=55, threshold=60.0, tolerance=1)
    assert s["n_anticipated"] == 0
    perfect = cpu.loc[5:55].copy()
    perfect.loc[30] = 65.0  # alerta sem evento
    s = peaks.alert_scores(perfect, 1, ev, low=5, high=55, threshold=60.0, tolerance=1)
    assert s["n_anticipated"] == 2 and s["n_false_alarm_episodes"] == 1


def test_jensen_zero_for_linear_and_incomplete_trajectory_excluded():
    agg = pd.DataFrame({"bin": [0, 0], "r": [1.0, 1.4], "n_present": [2, 2], "coverage": [1.0, 1.0], "u_mean": [0.5, 0.5], "u_var": [0.04, 0.04], "p_of_mean_w": [150.0, 160.0], "mean_of_p_w": [150.0, 157.0], "sum_of_p_w": [300.0, 314.0], "second_order_mean_of_p_w_diagnostic": [150.0, 157.5], "unstable_approximation": [False, False]})
    table, summary = energy.jensen(agg, 300.0)
    assert table.loc[table["r"] == 1.0, "jensen_gap_w_per_server"].iloc[0] == 0.0
    assert table.loc[table["r"] == 1.4, "jensen_gap_fleet_w"].iloc[0] == pytest.approx(-6.0)
    traj = pd.DataFrame([{"model": m, "origin_index": o, "step": s, "y_true": 50.0, "y_pred": 50.0 if not (m == "x" and s == 4) else np.nan} for m in ("persistence", "x") for o in (0, 1) for s in range(1, 7)])
    wide = energy.trajectories_wide(traj)
    origin = energy.origin_errors(FanModel(100, 200), wide, [1.0], [1, 6], 300.0, 60.0)
    x6 = origin[(origin["model"] == "x") & (origin["horizon_steps"] == 6)]
    assert not x6["complete_trajectory"].any() and x6["reference_energy_wh"].notna().all()
    metrics = energy.energy_metrics(origin, energy.power_timeseries(FanModel(100, 200), wide, [1.0], 300.0))
    assert not ((metrics["model"] == "x") & (metrics["horizon_steps"] == 6)).any()


def test_export_refuses_empty_and_undocumented(tmp_path):
    with pytest.raises(export.ExportError):
        export.write_csv(pd.DataFrame(), tmp_path / "x.csv")
    with pytest.raises(export.ExportError, match="sem dicionário"):
        export.write_tables({"predictions": pd.DataFrame({"coluna_inventada": [1]})}, tmp_path)
    frame = pd.DataFrame({"model": ["a"], "mae_pp": [1.23456789012345]})
    export.write_tables({"model_metrics": frame}, tmp_path)
    assert (tmp_path / "model_metrics.csv").read_text().splitlines()[1] == "a,1.23456789012"


def test_latex_table_columns_are_consistent():
    text = latex.table_env("x", "Legenda", ["A", "B", "C"], [["1", "2{,}0", "3"], ["4", "5", "6"]], "nota")
    assert "\\caption{Legenda}" in text and "\\label{tab:x}" in text
    rows = [line for line in text.splitlines() if line.endswith("\\\\") and "&" in line]
    assert all(line.count("&") == 2 for line in rows)
    assert latex.num(-3.16) == "$-$3{,}16" and latex.esc("50%_a") == "50\\%\\_a"


def test_decimal_comma_keeps_thousands():
    assert style._comma("MAE 3.25 e 4.023 máquinas, r = 1.4") == "MAE 3,25 e 4.023 máquinas, r = 1,4"
    assert style._tick(-0.5) == "−0,5"


def test_tree_shap_is_additive_and_divergent_refit_is_refused():
    from energy_prediction.analysis import importance
    from energy_prediction.models.registry import XGBoostModel

    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    y = 2 * X[:, 0] + X[:, 1] + rng.normal(0, 0.1, 200)
    model = XGBoostModel({"max_depth": 3, "n_estimators": 50, "learning_rate": 0.1}, seed=42, n_jobs=1).fit(X, y)
    pred = model.predict(X)
    refit = importance.Refit("xgboost_v2", "A", 1, pred, pred.copy(), 0.0, 0.0, model._model.get_booster(), X, ["cpu_t", "cpu_lag_1", "sin_day"], {})
    shap = importance.tree_shap(refit)
    assert shap["additivity_max_abs_diff"].iloc[0] < 1e-4 and shap.loc[shap["rank"] == 1, "feature"].iloc[0] == "cpu_t"
    assert importance.reproduces(refit, 1e-6)
    shifted = importance.Refit("xgboost_v2", "A", 1, pred + 1e-3, pred, 0.0, 0.0, None, None, [], {})
    assert not importance.reproduces(shifted, 1e-6)
