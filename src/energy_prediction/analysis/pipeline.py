"""Cálculo de todas as tabelas da análise expandida a partir dos artefatos (sem gravar nada)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from energy_prediction.analysis import ablation, computational, diagnostics, energy, integrity, neural, normalize, peaks, performance, probabilistic, statistics
from energy_prediction.analysis import dataset as ds
from energy_prediction.analysis.config import AnalysisPaths
from energy_prediction.analysis.loader import RunArtifacts, load_run
from energy_prediction.energy.fan import FanModel


@dataclass
class AnalysisResult:
    cfg: dict
    paths: AnalysisPaths
    art: RunArtifacts
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: object | None = None  # DatasetContext quando a camada 2 está disponível

    def add(self, name: str, frame: pd.DataFrame | None, reason: str = "") -> None:
        if frame is None or frame.empty:
            self.warnings.append(f"{name}.csv não gerado: {reason or 'sem dados'}.")
            return
        self.tables[name] = frame.reset_index(drop=True)


def _best_neural(metrics: pd.DataFrame, models: list[str]) -> str:
    """Melhor rede pelo MAE interno de seleção (A, 5 min), nunca pelo teste."""
    inner = metrics[(metrics["split"] == "test_final") & (metrics["config"] == "A") & (metrics["horizon_steps"] == 1) & (metrics["mode"] == "direct") & metrics["model"].isin(models) & (metrics["analysis_set"] == "main")]
    return str(inner.sort_values("inner_selection_mae_pp").iloc[0]["model"])


def _load_dataset(result: AnalysisResult) -> None:
    cfg, art = result.cfg, result.art
    if not cfg["dataset"]["enabled"]:
        result.warnings.append("Camada do cache de grade desligada na configuração; dataset, coorte reconstruída, TreeSHAP e tempos clássicos não foram calculados.")
        return
    try:
        from energy_prediction.analysis.dataset import load_dataset

        result.data = load_dataset(art.manifest, result.paths.project_root)
    except (FileNotFoundError, ImportError, OSError) as error:
        result.warnings.append(f"Cache de grade indisponível ({error}); seções de dataset, coorte reconstruída, TreeSHAP e tempos clássicos puladas.")


def compute(cfg: dict, paths: AnalysisPaths) -> AnalysisResult:
    art = load_run(paths)
    result = AnalysisResult(cfg=cfg, paths=paths, art=art, warnings=list(art.warnings))
    substitute = yaml.safe_load(paths.substitute.read_text(encoding="utf-8"))
    tol = cfg["tolerances"]

    # 1. Integridade (falha interrompe).
    result.checks += integrity.run_checks(art, tol, substitute)
    integrity.raise_on_failure(result.checks)
    preregistered, check = statistics.reproduce_preregistered(art, tolerance=float(tol["bootstrap"]))
    result.checks.append(check)
    integrity.raise_on_failure(result.checks)

    # 2. Tabela longa e métricas.
    step = normalize.step_seconds(art)
    preds = normalize.predictions_table(art, cfg["models"]["families"])
    inner = performance.inner_selection([art.frame("residual/metrics"), art.frame("neural/metrics")])
    metrics, by_fold = performance.metrics_tables(preds, inner)
    result.add("predictions", preds)
    result.add("model_metrics", metrics)
    result.add("model_metrics_by_fold", by_fold)
    result.add("model_rankings", performance.rankings(metrics))
    result.add("fold_wins", performance.fold_wins(by_fold))
    test_range = tuple(art.doc("integrity/alignment")["test_range"])
    test_start, test_end = int(test_range[0]), int(test_range[1])
    horizons = sorted(int(h) for h in preds.loc[preds["mode"] == "direct", "horizon_steps"].unique())
    result.info.update(step_seconds=step, test_range=test_range, horizons=horizons, run_id=art.run_id)

    # 3. Camada 2 e série observada.
    _load_dataset(result)
    data = result.data
    if data is not None:
        result.checks.append(ds.series_check(data, preds, float(tol["series_pp"])))
        integrity.raise_on_failure(result.checks)
        cpu = data.cpu
    else:
        cpu = normalize.observed_series(preds)
    dev_end = test_start - 1
    result.info["dev_end_index"] = dev_end

    # 4. Diagnóstico temporal.
    dcfg = cfg["diagnostics"]
    main_test = preds[(preds["split"] == "test_final") & (preds["population"] == "main") & (preds["analysis_set"] == "main") & (preds["mode"] == "direct")]
    annotated, diag_info = diagnostics.annotate(main_test, cpu, dev_end=dev_end, horizons=horizons, n_bands=int(dcfg["cpu_bands"]), change_bins=dcfg["recent_change_bins"])
    result.info["diagnostics"] = diag_info
    result.add("errors_by_origin", annotated)
    result.add("errors_by_regime", diagnostics.errors_by_group(annotated, ["regime", "cpu_band", "recent_change_1_step_bin", "realized_change_bin"]))
    result.add("error_distribution", diagnostics.error_distribution(preds[(preds["analysis_set"] == "main") & preds["split"].isin(["test_final"])]))
    result.add("residual_autocorrelation", diagnostics.residual_acf(main_test, int(dcfg["acf_max_lag"])))
    low, high = test_start + max(horizons), test_end + min(horizons)
    windows = diagnostics.objective_windows(cpu, low, high, length=int(dcfg["window_steps"]), change_steps=int(dcfg["change_steps"]), step_seconds=step)
    result.add("diagnostic_windows", windows)
    result.info["best_neural"] = _best_neural(metrics, cfg["models"]["neural"])
    result.info["event_range"] = (low, high)

    # 5. Picos.
    pcfg = cfg["peaks"]
    thr = peaks.thresholds(cpu, dev_end, main_test["y_obs_cpu_pct"], pcfg["quantiles"])
    result.add("peak_thresholds", thr)
    result.add("peak_analysis", peaks.peak_metrics(main_test, thr))
    signals = {}
    for (model, config, h), part in main_test[main_test["config"].isin(["A", "-"])].groupby(["model", "config", "horizon_steps"]):
        signals[(model, "previsão pontual", int(h))] = part.set_index("target_index")["y_pred_cpu_pct"].sort_index()
    qt = normalize.quantile_table(art)
    if qt is not None:
        q_test = qt[(qt["split"] == "test_final") & (qt["config"] == "A")]
        for column, label in (("q90_cpu_pct", "Q90 bruto"), ("q95_cpu_pct", "Q95 bruto"), ("q90_adjusted_cpu_pct", "Q90 ajustado"), ("q95_adjusted_cpu_pct", "Q95 ajustado")):
            for h, part in q_test.groupby("horizon_steps"):
                signals[(f"xgboost_quantile {label}", "quantil", int(h))] = part.set_index("target_index")[column].sort_index()
    alerts, events = peaks.alert_table(signals, cpu, thr, low=low, high=high, merge_gap=int(pcfg["merge_gap_steps"]), tolerance=int(pcfg["tolerance_steps"]))
    result.add("peak_alerts", alerts)
    result.add("peak_events", events)
    dev_p90 = float(thr.loc[thr["threshold_kind"] == "desenvolvimento_p90", "threshold_cpu_pct"].iloc[0])
    result.info["dev_p90"] = dev_p90

    # 6. Estatística.
    scfg = cfg["statistics"]
    info = art.doc("energy/energy_info")["substitute"]
    watts_per_pp = (float(info["p_busy_w"]) - float(info["p_idle_w"])) / 100.0
    models_a = [m for m in cfg["models"]["order"] if m in set(main_test["model"])]
    exploratory = statistics.exploratory(preds, models_a, config=scfg["exploratory_config"], blocks=scfg["blocks"], n_boot=int(scfg["n_boot"]), seed=int(scfg["seed"]))
    combined = statistics.decorate(pd.concat([preregistered, exploratory], ignore_index=True), primary_block=int(scfg["primary_block"]), practical=float(scfg["practical_relative_threshold"]), watts_per_pp=watts_per_pp)
    result.add("bootstrap_intervals", combined)
    result.add("statistical_comparisons", statistics.summary_table(combined))
    result.info["watts_per_pp"] = watts_per_pp

    # 7. Redes neurais.
    history = art.frame("neural/epoch_history")
    if history is not None:
        hist = neural.history_table(history, art.doc("neural/selections"))
        result.add("neural_training_history", hist)
        units = neural.unit_diagnostics(hist)
        result.add("neural_fit_diagnostics", units)
        result.add("neural_fit_summary", neural.unit_summary(units))
    persistence_inner = {}
    if data is not None and data.ctx.inner is not None:
        from energy_prediction.experiments.v3_common import persistence_inner_mae

        persistence_inner = {h: persistence_inner_mae(data.ctx.table, data.ctx.inner, h) for h in horizons}
        gate = (art.doc("neural/optional_architectures") or {}).get("gate", {})
        if "persistence_inner_mae_h1" in gate:
            diff = abs(persistence_inner[1] - gate["persistence_inner_mae_h1"])
            result.checks.append({"check": "persistencia_interna_h1", "passed": diff <= 1e-9, "detail": "MAE interno da persistência recalculado contra optional_architectures.json", "max_abs_diff": diff, "tolerance": 1e-9})
    result.add("generalization", neural.generalization(metrics, persistence_inner), "MAE interno da persistência indisponível")
    if art.doc("neural/window_selection"):
        result.add("neural_window_selection", neural.window_table(art.doc("neural/window_selection")))

    # 8. Complexidade, tempo e importância.
    profile = art.frame("neural/training_profile")
    device_name = (art.doc("neural/family_info") or {}).get("device", {}).get("device_name")
    comp_n, perf_n = computational.neural_rows(profile, device_name) if profile is not None else (pd.DataFrame(), pd.DataFrame())
    comp_c, perf_c = pd.DataFrame(), pd.DataFrame()
    if data is not None:
        from energy_prediction.analysis import importance

        refits, records = importance.run_refits(data.ctx, art, horizons=horizons, tolerance=float(tol["refit_pp"]), repeats=int(cfg["computational"]["timing_repeats"]))
        result.info["refits"] = records
        refused = [r for r in records if not r["reproduced"]]
        for r in refused:
            result.warnings.append(f"Reajuste de {r['model']}/{r['config']}/h{r['horizon_steps']} não reproduziu as previsões gravadas (diferença {r['max_abs_diff_pp']:.3g}); importância e tempo não usados.")
        result.checks.append({"check": "reproducao_reajustes", "passed": not refused, "detail": f"{len(records)} reajustes clássicos comparados com as previsões gravadas", "max_abs_diff": max(r["max_abs_diff_pp"] for r in records), "tolerance": float(tol["refit_pp"])})
        shap = [importance.tree_shap(r) for r in refits if r.booster is not None]
        if shap:
            result.add("feature_importance", pd.concat(shap, ignore_index=True))
        comp_c, perf_c = computational.classical_rows(refits, records)
    complexity = pd.concat([comp_c, comp_n], ignore_index=True)
    performance_table = pd.concat([perf_c, perf_n], ignore_index=True)
    result.add("model_complexity", complexity)
    result.add("computational_performance", performance_table)
    if not complexity.empty and not performance_table.empty:
        result.add("cost_tradeoff", computational.tradeoff(complexity, performance_table, metrics))

    # 9. Ablação.
    ab = ablation.ablation_table(metrics, statistics.summary_table(combined))
    result.add("ablation_results", ab)
    result.add("ablation_grid", ablation.ablation_grid(ab, [m for m in cfg["models"]["order"] if m != "persistence"]))

    # 10. Quantis.
    if qt is not None:
        result.add("quantile_predictions", qt)
        result.add("quantile_calibration", probabilistic.calibration_table(qt, dev_p90))
        result.checks.append(probabilistic.pinball_check(qt, art.frame("quantile/metrics"), float(tol["metric"])))

    # 11. Energia.
    ecfg = cfg["energy"]
    trajectories = art.frame("energy/trajectories")
    if trajectories is not None:
        fan = FanModel(float(info["p_idle_w"]), float(info["p_busy_w"]))
        exps = [float(r) for r in ecfg["exponents"]]
        wide = energy.trajectories_wide(trajectories)
        result.checks.append(energy.reproduction_check(fan, wide, art.require("energy/energy_by_origin"), exps, horizons, step, float(tol["energy_j"])))
        power = energy.power_timeseries(fan, wide, exps, step)
        origin = energy.origin_errors(fan, wide, exps, horizons, step, dev_p90)
        result.add("energy_timeseries", power)
        result.add("energy_origin_errors", origin)
        result.add("energy_metrics", energy.energy_metrics(origin, power))
        result.add("energy_cumulative", energy.cumulative(origin, test_start, step))
        result.add("energy_sensitivity", energy.sensitivity(fan, wide, exps, ecfg["sensitivity"], step, float(ecfg["reference_exponent"])))
        result.info["fan"] = {"p_idle_w": fan.p_idle, "p_busy_w": fan.p_busy}
        result.info["substitute"] = info
    aggregate = art.frame("energy/aggregate_effect")
    if aggregate is not None:
        jt, js = energy.jensen(aggregate, step)
        result.add("jensen_analysis", jt)
        result.add("jensen_summary", js)

    # 12. Dataset e populações.
    if data is not None:
        period_table = ds.periods(art.doc("integrity/alignment"), None, len(data.series))
        result.info["periods"] = period_table.to_dict("records")
        result.add("dataset_series", ds.aggregated_series(data))
        result.add("dataset_profile", ds.profile(data, period_table))
        result.add("dataset_changes", ds.changes(data, horizons, period_table))
        result.add("dataset_dependence", ds.dependence(data, 72, dev_end))
        result.add("fleet_distribution", ds.fleet_distribution(data))
        result.add("machine_coverage", ds.machine_coverage(data))
        diff = ds.population_difference(data)
        result.add("population_difference", diff)
        result.add("machine_m_2739", ds.machine_records(data))
        audit = art.doc("integrity/population_audit") or {}
        stored = audit.get("series_comparison", {}).get("max_abs_cpu_difference")
        if stored is not None:
            got = float(diff["difference_main_minus_cohort_pp"].abs().max())
            result.checks.append({"check": "auditoria_populacao", "passed": abs(got - stored) <= 1e-12, "detail": "diferença máxima principal − coorte contra population_audit.json", "max_abs_diff": abs(got - stored), "tolerance": 1e-12})
        result.add("cohort_comparison", ds.cohort_metrics(metrics, diff, test_range))
    else:
        result.add("cohort_comparison", ds.cohort_metrics(metrics, None, test_range))
    integrity.raise_on_failure(result.checks)
    return result
