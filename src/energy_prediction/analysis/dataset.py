"""Caracterização da série agregada, da frota por máquina e da diferença entre principal e coorte.

Lê o cache de grade com a configuração gravada no manifesto da corrida (mesma série da V3).
Correlação entre exógenas e CPU futura descreve associação; não implica antecipação nem causa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VARIABLES = ["cpu", "mem", "net_in", "net_out", "disk"]


@dataclass
class DatasetContext:
    ctx: object  # V3Context
    series: pd.DataFrame
    cohort_series: pd.DataFrame
    step_seconds: float

    @property
    def cpu(self) -> pd.Series:
        return self.series.set_index("bin")["cpu"]


def load_dataset(manifest: dict, project_root: Path) -> DatasetContext:
    from energy_prediction.experiments.v3_context import build_context
    from energy_prediction.preprocessing.cluster import aggregate

    ctx = build_context(manifest["config"], Path(project_root))
    cohort, _ = aggregate("homogeneous_cohort", ctx.grid, ctx.machine_info, step_seconds=ctx.step_seconds)
    return DatasetContext(ctx=ctx, series=ctx.series.copy(), cohort_series=cohort, step_seconds=float(ctx.step_seconds))


def series_check(data: DatasetContext, predictions: pd.DataFrame, tolerance: float) -> dict:
    """A série reconstruída precisa coincidir com o observado das previsões da população principal."""
    main = predictions[(predictions["population"] == "main") & (predictions["analysis_set"] == "main")]
    observed = main.groupby("target_index")["y_obs_cpu_pct"].first()
    rebuilt = data.cpu.reindex(observed.index)
    diff = float(np.nanmax(np.abs(rebuilt.to_numpy() - observed.to_numpy())))
    passed = bool(np.isfinite(diff) and diff <= tolerance and rebuilt.notna().all())
    return {"check": "serie_reconstruida", "passed": passed, "detail": f"{len(observed)} instantes previstos comparados com o cache de grade", "max_abs_diff": diff, "tolerance": tolerance}


def periods(alignment: dict, folds: list | dict | None, n_bins: int) -> pd.DataFrame:
    """Intervalos de origem de cada período (treino até o fold 0, validação dos folds, purga, teste)."""
    test = alignment["test_range"]
    validation = alignment["fold_validation_ranges"]
    rows = [{"period": "treino_inicial", "start_index": 0, "end_index": int(validation[0][0]) - 1}]
    for k, (a, b) in enumerate(validation):
        rows.append({"period": f"validacao_fold_{k}", "start_index": int(a), "end_index": int(b)})
    rows.append({"period": "desenvolvimento", "start_index": 0, "end_index": int(test[0]) - 1})
    rows.append({"period": "teste", "start_index": int(test[0]), "end_index": int(test[1])})
    rows.append({"period": "serie_completa", "start_index": 0, "end_index": n_bins - 1})
    return pd.DataFrame(rows)


def profile(data: DatasetContext, period_table: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas por período e variável (valores observados)."""
    s = data.series.set_index("bin")
    rows = []
    for _, p in period_table.iterrows():
        part = s.loc[p["start_index"] : p["end_index"]]
        for variable in VARIABLES + ["coverage", "n_machines"]:
            values = part[variable].astype(float)
            finite = values.dropna()
            rows.append(
                {
                    "period": p["period"], "start_index": p["start_index"], "end_index": p["end_index"],
                    "start_time_h": p["start_index"] * data.step_seconds / 3600.0, "end_time_h": p["end_index"] * data.step_seconds / 3600.0,
                    "variable": variable, "n_bins": int(len(values)), "n_missing": int(values.isna().sum()),
                    "mean": finite.mean(), "std": finite.std(), "min": finite.min(), "p05": finite.quantile(0.05), "p25": finite.quantile(0.25),
                    "p50": finite.median(), "p75": finite.quantile(0.75), "p95": finite.quantile(0.95), "max": finite.max(),
                }
            )
    return pd.DataFrame(rows)


def changes(data: DatasetContext, horizons: list[int], period_table: pd.DataFrame) -> pd.DataFrame:
    """Distribuição da variação de CPU em h passos, Δ = cpu(t+h) − cpu(t), por período."""
    cpu = data.cpu
    rows = []
    for _, p in period_table[period_table["period"].isin(["desenvolvimento", "teste"])].iterrows():
        for h in horizons:
            delta = (cpu.shift(-h) - cpu).loc[p["start_index"] : p["end_index"]].dropna()
            q = delta.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            rows.append(
                {
                    "period": p["period"], "horizon_steps": h, "horizon_min": h * data.step_seconds / 60.0, "n": int(len(delta)),
                    "mean_change_pp": delta.mean(), "std_change_pp": delta.std(), "mean_abs_change_pp": delta.abs().mean(),
                    **{f"p{int(k * 100):02d}_change_pp": v for k, v in q.items()},
                }
            )
    return pd.DataFrame(rows)


def _corr(a: pd.Series, b: pd.Series) -> float:
    frame = pd.concat([a, b], axis=1).dropna()
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1])) if len(frame) > 2 else np.nan


def dependence(data: DatasetContext, max_lag: int, end_index: int) -> pd.DataFrame:
    """Autocorrelação da CPU e correlação entre cpu(t+k) e cada exógena em t, no desenvolvimento.

    Também a correlação entre a variação futura da CPU e a variação recente da exógena,
    que descreve associação defasada, não antecipação causal.
    """
    s = data.series.set_index("bin").loc[:end_index]
    cpu = s["cpu"]
    rows = []
    for lag in range(0, max_lag + 1):
        rows.append({"kind": "autocorrelacao_cpu", "variable": "cpu", "lag_steps": lag, "lag_min": lag * data.step_seconds / 60.0, "correlation": _corr(cpu, cpu.shift(lag)), "n": int(cpu.notna().sum())})
    for variable in VARIABLES[1:]:
        x = s[variable]
        for lead in range(0, 13):
            rows.append({"kind": "nivel_exogena_vs_cpu_futura", "variable": variable, "lag_steps": lead, "lag_min": lead * data.step_seconds / 60.0, "correlation": _corr(x, cpu.shift(-lead)), "n": int(x.notna().sum())})
            if lead > 0:
                rows.append(
                    {
                        "kind": "variacao_exogena_vs_variacao_futura_cpu", "variable": variable, "lag_steps": lead, "lag_min": lead * data.step_seconds / 60.0,
                        "correlation": _corr(x - x.shift(1), cpu.shift(-lead) - cpu), "n": int(x.notna().sum()),
                    }
                )
    return pd.DataFrame(rows)


def aggregated_series(data: DatasetContext) -> pd.DataFrame:
    """Série agregada da população principal (a mesma das previsões), uma linha por intervalo."""
    s = data.series
    out = pd.DataFrame(
        {
            "bin_index": s["bin"], "time_h": s["bin"] * data.step_seconds / 3600.0, "aggregated_cpu_weighted_pct": s["cpu"], "aggregated_mem_pct": s["mem"],
            "aggregated_net_in": s["net_in"], "aggregated_net_out": s["net_out"], "aggregated_disk_pct": s["disk"], "weighted_coverage": s["coverage"], "n_machines_present": s["n_machines"],
        }
    )
    return out.reset_index(drop=True)


def fleet_distribution(data: DatasetContext) -> pd.DataFrame:
    """Por intervalo: máquinas presentes, percentis da CPU individual e CPU agregada ponderada."""
    grid = data.ctx.grid
    population = set(data.ctx.population)
    present = grid[grid["machine_id"].isin(population) & grid["cpu"].notna()]
    q = present.groupby("bin")["cpu"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
    q.columns = [f"machine_cpu_p{int(c * 100):02d}_pct" for c in q.columns]
    out = q.join(present.groupby("bin")["cpu"].agg(machine_cpu_mean_pct="mean", machine_cpu_std_pct="std", n_present="size"))
    out = out.join(data.series.set_index("bin")[["cpu", "coverage"]].rename(columns={"cpu": "aggregated_cpu_weighted_pct", "coverage": "weighted_coverage"}))
    out = out.reset_index().rename(columns={"bin": "bin_index"})
    out.insert(1, "time_h", out["bin_index"] * data.step_seconds / 3600.0)
    return out


def machine_coverage(data: DatasetContext) -> pd.DataFrame:
    grid = data.ctx.grid
    n_bins = int(data.series["bin"].max()) + 1
    info = data.ctx.machine_info.set_index("machine_id")
    counts = grid[grid["cpu"].notna()].groupby("machine_id")["bin"].agg(n_bins_present="size", first_bin="min", last_bin="max")
    counts["coverage_fraction"] = counts["n_bins_present"] / n_bins
    counts = counts.join(info[["cpu_num", "active"]], how="left")
    cpu = grid.groupby("machine_id")["cpu"].agg(machine_mean_cpu_pct="mean", machine_max_cpu_pct="max")
    return counts.join(cpu).reset_index()


def population_difference(data: DatasetContext) -> pd.DataFrame:
    main = data.series.set_index("bin")
    cohort = data.cohort_series.set_index("bin")
    out = pd.DataFrame(
        {
            "main_cpu_pct": main["cpu"], "cohort_cpu_pct": cohort["cpu"].reindex(main.index),
            "main_n_machines": main["n_machines"], "cohort_n_machines": cohort["n_machines"].reindex(main.index),
        }
    )
    out["difference_main_minus_cohort_pp"] = out["main_cpu_pct"] - out["cohort_cpu_pct"]
    machine = data.ctx.grid[data.ctx.grid["machine_id"] == "m_2739"].set_index("bin")
    out["m_2739_present"] = out.index.isin(machine.index)
    out["m_2739_cpu_pct"] = machine["cpu"].reindex(out.index)
    out = out.reset_index().rename(columns={"bin": "bin_index"})
    out.insert(1, "time_h", out["bin_index"] * data.step_seconds / 3600.0)
    return out


def machine_records(data: DatasetContext, machine_id: str = "m_2739") -> pd.DataFrame:
    grid = data.ctx.grid
    part = grid[grid["machine_id"] == machine_id].copy()
    part.insert(2, "time_h", part["bin"] * data.step_seconds / 3600.0)
    return part.rename(columns={"bin": "bin_index", "cpu": "cpu_pct", "mem": "mem_pct", "disk": "disk_pct"}).reset_index(drop=True)


def cohort_metrics(metrics: pd.DataFrame, difference: pd.DataFrame | None, test_range: tuple[int, int]) -> pd.DataFrame:
    """MAE de teste da V2 na principal e na coorte, por modelo, configuração e horizonte."""
    base = metrics[(metrics["version"].isin(["V2", "V3"])) & (metrics["analysis_set"] == "main") & (metrics["mode"] == "direct")]
    base = base[base["model"].isin(["persistence", "ridge_v2", "xgboost_v2"])]
    keys = ["model", "config", "horizon_steps", "split"]
    main = base[base["population"] == "main"][keys + ["mae_pp", "rmse_pp"]]
    cohort = base[base["population"] == "cohort"][keys + ["mae_pp", "rmse_pp"]]
    out = main.merge(cohort, on=keys, suffixes=("_main", "_cohort"))
    out["mae_difference_main_minus_cohort_pp"] = out["mae_pp_main"] - out["mae_pp_cohort"]
    out["rmse_difference_main_minus_cohort_pp"] = out["rmse_pp_main"] - out["rmse_pp_cohort"]
    if difference is not None:
        test = difference[(difference["bin_index"] >= test_range[0]) & (difference["bin_index"] <= test_range[1] + 6)]
        out["series_max_abs_difference_test_period_pp"] = float(test["difference_main_minus_cohort_pp"].abs().max())
        out["series_max_abs_difference_full_pp"] = float(difference["difference_main_minus_cohort_pp"].abs().max())
    return out.sort_values(["split", "model", "config", "horizon_steps"]).reset_index(drop=True)
