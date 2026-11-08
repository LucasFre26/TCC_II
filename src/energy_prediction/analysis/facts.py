"""Números-chave extraídos das tabelas da análise, usados nos relatórios e no LaTeX.

Nenhum número é digitado à mão: tudo sai das tabelas calculadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def br(value, digits: int = 2) -> str:
    """Número com vírgula decimal; traço para ausente."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def thousands(value) -> str:
    """Inteiro com ponto de milhar (padrão brasileiro)."""
    return f"{int(round(float(value))):,}".replace(",", ".")


def pct(value, digits: int = 1) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{br(100 * value, digits)}%"


def signed(value, digits: int = 2) -> str:
    return "—" if value is None or not np.isfinite(value) else (f"+{br(value, digits)}" if value > 0 else br(value, digits))


def ci(row) -> str:
    return f"[{br(row['ci_low'])}; {br(row['ci_high'])}]"


def test_table(t: dict, split: str = "test_final", config: str = "A") -> pd.DataFrame:
    m = t["model_metrics"]
    return m[(m["split"] == split) & (m["population"] == "main") & (m["analysis_set"] == "main") & (m["mode"] == "direct") & m["config"].isin([config, "-"])]


def prereg(t: dict, comparison: str, model_a: str, h: int, config_a: str = "A") -> pd.Series | None:
    s = t["statistical_comparisons"]
    part = s[(s["analysis_type"] == "pre_registrada") & (s["comparison"] == comparison) & (s["model_a"] == model_a) & (s["horizon_steps"] == h) & (s["config_a"] == config_a) & (s["loss"] == "erro_absoluto")]
    return part.iloc[0] if len(part) else None


def compute(t: dict, info: dict) -> dict:
    f: dict = {}
    test = test_table(t)
    val = test_table(t, "validation_pooled")
    f["test_mae"] = test.pivot(index="model", columns="horizon_steps", values="mae_pp")
    f["val_mae"] = val.pivot(index="model", columns="horizon_steps", values="mae_pp")
    f["best_test"] = {h: f["test_mae"][h].idxmin() for h in f["test_mae"].columns}
    f["best_val"] = {h: f["val_mae"][h].idxmin() for h in f["val_mae"].columns}
    f["bias"] = test.pivot(index="model", columns="horizon_steps", values="bias_obs_minus_pred_pp")
    f["imp_xgb_val"] = val.pivot(index="model", columns="horizon_steps", values="improvement_vs_xgboost_v2_pct")
    f["imp_xgb_test"] = test.pivot(index="model", columns="horizon_steps", values="improvement_vs_xgboost_v2_pct")
    f["imp_pers_test"] = test.pivot(index="model", columns="horizon_steps", values="improvement_vs_persistence_pct")
    s = t["statistical_comparisons"]
    pre = s[(s["analysis_type"] == "pre_registrada")]
    f["n_prereg"] = int(len(pre))
    f["n_prereg_sig"] = int(pre["significant"].sum())
    f["n_prereg_holm"] = int(pre["holm_significant"].sum())
    f["min_p"] = float(pre["p_value_bootstrap"].min())
    exp = s[s["analysis_type"] == "exploratoria"]
    f["n_exp"] = int(len(exp))
    f["n_exp_sig"] = int(exp["significant"].sum())
    f["n_exp_holm"] = int(exp["holm_significant"].sum())
    neural = ["dlinear", "tcn", "nbeatsx"]
    f["neural_vs_xgb_sig"] = all(bool(prereg(t, "new_vs_v2_xgboost_A", m, h)["significant"]) for m in neural for h in (1, 3, 6) if prereg(t, "new_vs_v2_xgboost_A", m, h) is not None)
    f["neural_vs_xgb_sig_count"] = sum(bool(prereg(t, "new_vs_v2_xgboost_A", m, h)["significant"]) for m in neural for h in (1, 3, 6))
    f["neural_vs_pers_sig_count"] = sum(bool(prereg(t, "new_vs_persistence_A", m, h)["significant"]) for m in neural for h in (1, 3, 6))
    f["residual_vs_xgb_sig_count"] = sum(bool(prereg(t, "new_vs_v2_xgboost_A", m, h)["significant"]) for m in ("xgboost_delta", "statespace", "hybrid") for h in (1, 3, 6))
    v2 = pre[(pre["comparison"] == "new_vs_v2_xgboost_A") & (pre["loss"] == "erro_absoluto")]
    f["v3_better_than_xgb"] = v2[v2["significant"] & (v2["mean_diff"] < 0)][["model_a", "horizon_steps"]].values.tolist()
    g = t["generalization"].set_index(["model", "horizon_steps"])
    f["gen"] = g
    prof = t.get("dataset_profile")
    if prof is not None:
        cpu = prof[prof["variable"] == "cpu"].set_index("period")
        f["cpu_mean_dev"], f["cpu_mean_test"] = float(cpu.loc["desenvolvimento", "mean"]), float(cpu.loc["teste", "mean"])
        f["cpu_std_dev"] = float(cpu.loc["desenvolvimento", "std"])
    e = t["errors_by_origin"]
    pers = e[(e["model"] == "persistence") & (e["horizon_steps"] == 6)]
    labels = info.get("diagnostics", {}).get("cpu_band_labels", [])
    if labels:
        f["share_upper_bands"] = float(pers["cpu_band"].isin(labels[-2:]).mean())
    r = t["errors_by_regime"]
    stable = r[(r["grouping"] == "regime") & (r["group"] == "estável") & r["config"].isin(["A", "-"])]
    f["stable_bias"] = stable.pivot(index="model", columns="horizon_steps", values="bias_obs_minus_pred_pp")
    reg = r[(r["grouping"] == "regime") & r["config"].isin(["A", "-"]) & (r["horizon_steps"] == 6)]
    f["regime_mae_h6"] = reg.pivot(index="model", columns="group", values="mae_pp")
    f["regime_n_h6"] = reg[reg["model"] == "persistence"].set_index("group")["n"]
    thr = t["peak_thresholds"].set_index("threshold_kind")
    f["dev_p90"], f["dev_p95"] = float(thr.loc["desenvolvimento_p90", "threshold_cpu_pct"]), float(thr.loc["desenvolvimento_p95", "threshold_cpu_pct"])
    ev = t["peak_events"]
    ev90 = ev[ev["threshold_kind"] == "desenvolvimento_p90"]
    f["n_events"] = int(len(ev90))
    f["n_single_step_events"] = int((ev90["duration_steps"] == 1).sum())
    f["longest_event"] = ev90.sort_values("duration_steps").iloc[-1]
    pk = t["peak_analysis"]
    pk = pk[pk["config"].isin(["A", "-"]) & (pk["threshold_kind"] == "desenvolvimento_p90")]
    f["peak"] = pk.set_index(["model", "horizon_steps"])
    al = t["peak_alerts"]
    f["alerts"] = al[al["threshold_kind"] == "desenvolvimento_p90"].set_index(["signal", "horizon_steps"])
    q = t["quantile_calibration"]
    f["quant"] = q[(q["config"] == "A")].set_index(["split", "horizon_steps", "quantile", "variant"])
    em = t["energy_metrics"]
    f["energy"] = em.set_index(["model", "r", "horizon_steps"])
    cum = t["energy_cumulative"]
    r_ref = 1.4
    last5 = cum[(cum["r"] == r_ref) & (cum["window_min"] == 5.0)].groupby("model").last()
    last30 = cum[(cum["r"] == r_ref) & (cum["window_min"] == 30.0)].groupby("model").last()
    f["cum5"], f["cum30"] = last5, last30
    f["energy_ref_24h"] = float(last5["cumulative_reference_wh"].iloc[0])
    f["jensen"] = t["jensen_summary"].set_index("r")
    cc = t.get("cohort_comparison")
    if cc is not None:
        test_cc = cc[cc["split"] == "test_final"]
        f["cohort_max_mae_diff"] = float(test_cc["mae_difference_main_minus_cohort_pp"].abs().max())
        f["cohort_series_test"] = float(test_cc["series_max_abs_difference_test_period_pp"].iloc[0]) if "series_max_abs_difference_test_period_pp" in test_cc else np.nan
        f["cohort_series_full"] = float(test_cc["series_max_abs_difference_full_pp"].iloc[0]) if "series_max_abs_difference_full_pp" in test_cc else np.nan
    pdiff = t.get("population_difference")
    if pdiff is not None:
        f["m2739_last_h"] = float(pdiff.loc[pdiff["m_2739_present"], "time_h"].max())
        f["m2739_n"] = int(pdiff["m_2739_present"].sum())
    fi = t.get("feature_importance")
    if fi is not None:
        f["shap_groups"] = fi.groupby(["model", "config", "horizon_steps", "group"])["share"].sum()
    ab = t["ablation_results"]
    f["ablation"] = ab[ab["population"] == "main"].set_index(["model", "config", "horizon_steps"])
    cp = t["computational_performance"]
    neural_perf = cp[cp["device"] == "cuda"]
    f["vram_range"] = (float(neural_perf["peak_cuda_memory_mib"].min()), float(neural_perf["peak_cuda_memory_mib"].max())) if len(neural_perf) else (np.nan, np.nan)
    comp = t["model_complexity"]
    nn = comp[comp["complexity_kind"] == "parâmetros treináveis"]
    f["params_range"] = (int(nn["n_trainable_parameters"].min()), int(nn["n_trainable_parameters"].max())) if len(nn) else (0, 0)
    fit = t.get("neural_fit_summary")
    if fit is not None:
        f["fit"] = fit[(fit["config"] == "A") & (fit["mode"] == "direct")].set_index(["model", "horizon"])
    win = t.get("neural_window_selection")
    if win is not None:
        f["window"] = win.groupby("window_steps")["inner_mae_sum_pp"].first()
    f["best_neural"] = info.get("best_neural")
    return f
