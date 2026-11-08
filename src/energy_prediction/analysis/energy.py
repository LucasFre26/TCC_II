"""Potência e energia de TI modeladas no servidor substituto, sensibilidade e efeito de Jensen.

Grandeza: energia de TI modelada de um servidor substituto do SPECpower_ssj2008 à utilização
agregada da frota (estimativa paramétrica, não consumo medido). PUE não é calculado.
Convenção de sinal desta seção (a mesma de energy_by_origin.csv): diferença = projetada −
referência; negativa quando a previsão subestima a potência ou a energia.
Jensen: gap por servidor = média de P(u_i) − P(média de u); gap da frota = soma de P(u_i) −
n · P(média de u), em watts de uma frota hipotética de servidores substitutos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_prediction.energy.fan import FanModel
from energy_prediction.energy.integrate import JOULES_PER_WH, horizon_energies

QUANTITY = "energia de TI modelada (estimativa paramétrica, servidor substituto SPECpower_ssj2008)"


def trajectories_wide(trajectories: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Por modelo: origens, CPU na origem, observado e previsto (n, 6)."""
    persistence = trajectories[trajectories["model"] == "persistence"]
    cpu_origin = persistence[persistence["step"] == 1].set_index("origin_index")["y_pred"].sort_index()
    out = {}
    for model, part in trajectories.groupby("model"):
        obs = part.pivot(index="origin_index", columns="step", values="y_true").sort_index()
        pred = part.pivot(index="origin_index", columns="step", values="y_pred").sort_index()
        out[model] = {"origins": obs.index.to_numpy(), "cpu_origin": cpu_origin.reindex(obs.index).to_numpy(), "obs": obs.to_numpy(), "pred": pred.to_numpy(), "steps": obs.columns.to_numpy()}
    return out


def energies(fan: FanModel, wide: dict, r: float, horizons: list[int], step_seconds: float) -> dict[str, dict[int, np.ndarray]]:
    p0, _ = fan.power(wide["cpu_origin"], r)
    p_obs, _ = fan.power(wide["obs"], r)
    p_pred, _ = fan.power(wide["pred"], r)
    return {"reference": horizon_energies(p0, p_obs, horizons, step_seconds), "projected": horizon_energies(p0, p_pred, horizons, step_seconds)}


def reproduction_check(fan: FanModel, wide_all: dict, by_origin: pd.DataFrame, exponents: list[float], horizons: list[int], step_seconds: float, tolerance: float) -> dict:
    diff = 0.0
    n = 0
    for (model, r, h), part in by_origin.groupby(["model", "r", "horizon_steps"]):
        wide = wide_all[model]
        e = energies(fan, wide, float(r), [int(h)], step_seconds)
        ref = pd.Series(e["reference"][int(h)], index=wide["origins"]).reindex(part["origin_index"]).to_numpy()
        proj = pd.Series(e["projected"][int(h)], index=wide["origins"]).reindex(part["origin_index"]).to_numpy()
        diff = max(diff, float(np.nanmax(np.abs(ref - part["reference_energy_j"].to_numpy()))), float(np.nanmax(np.abs(proj - part["projected_energy_j"].to_numpy()))))
        n += len(part)
    return {"check": "reproducao_energia", "passed": diff <= tolerance, "detail": f"{n} energias por origem recalculadas das trajetórias", "max_abs_diff": diff, "tolerance": tolerance}


def power_timeseries(fan: FanModel, wide_all: dict, exponents: list[float], step_seconds: float) -> pd.DataFrame:
    rows = []
    for model, wide in wide_all.items():
        for r in exponents:
            p_obs, _ = fan.power(wide["obs"], r)
            p_pred, _ = fan.power(wide["pred"], r)
            for j, step in enumerate(wide["steps"]):
                target = wide["origins"] + int(step)
                rows.append(
                    pd.DataFrame(
                        {
                            "model": model, "r": r, "origin_index": wide["origins"], "step": int(step), "target_index": target, "target_time_h": target * step_seconds / 3600.0,
                            "cpu_obs_pct": wide["obs"][:, j], "cpu_pred_pct": wide["pred"][:, j], "cpu_pred_minus_obs_pp": wide["pred"][:, j] - wide["obs"][:, j],
                            "power_reference_w": p_obs[:, j], "power_projected_w": p_pred[:, j], "power_projected_minus_reference_w": p_pred[:, j] - p_obs[:, j],
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def origin_errors(fan: FanModel, wide_all: dict, exponents: list[float], horizons: list[int], step_seconds: float, peak_threshold: float) -> pd.DataFrame:
    rows = []
    for model, wide in wide_all.items():
        for r in exponents:
            e = energies(fan, wide, r, horizons, step_seconds)
            for h in horizons:
                ref, proj = e["reference"][h] / JOULES_PER_WH, e["projected"][h] / JOULES_PER_WH
                cpu_err = (wide["pred"][:, :h] - wide["obs"][:, :h]).mean(axis=1)
                rows.append(
                    pd.DataFrame(
                        {
                            "model": model, "r": r, "horizon_steps": h, "horizon_min": h * step_seconds / 60.0, "origin_index": wide["origins"],
                            "reference_energy_wh": ref, "projected_energy_wh": proj, "energy_projected_minus_reference_wh": proj - ref,
                            "energy_abs_error_wh": np.abs(proj - ref), "energy_relative_error": (proj - ref) / ref,
                            "mean_cpu_pred_minus_obs_pp": cpu_err, "terminal_cpu_pred_minus_obs_pp": wide["pred"][:, h - 1] - wide["obs"][:, h - 1],
                            "trajectory_reaches_peak": wide["obs"][:, :h].max(axis=1) >= peak_threshold, "complete_trajectory": np.isfinite(ref) & np.isfinite(proj),
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def energy_metrics(origin: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, r, h), part in origin[origin["complete_trajectory"]].groupby(["model", "r", "horizon_steps"]):
        diff = part["energy_projected_minus_reference_wh"].to_numpy()
        peak = part["trajectory_reaches_peak"].to_numpy()
        step = power[(power["model"] == model) & (power["r"] == r) & (power["step"] == h)]["power_projected_minus_reference_w"].to_numpy()
        rows.append(
            {
                "quantity": QUANTITY, "model": model, "r": r, "horizon_steps": int(h), "horizon_min": part["horizon_min"].iloc[0], "n_origins": len(part),
                "total_reference_wh": part["reference_energy_wh"].sum(), "total_projected_wh": part["projected_energy_wh"].sum(),
                "energy_mae_wh": float(np.mean(np.abs(diff))), "energy_bias_projected_minus_reference_wh": float(diff.mean()),
                "energy_p95_abs_error_wh": float(np.quantile(np.abs(diff), 0.95)), "energy_mean_abs_relative_error": float(np.mean(np.abs(part["energy_relative_error"]))),
                "energy_underestimation_fraction": float(np.mean(diff < 0)),
                "n_peak_origins": int(peak.sum()), "energy_mae_peak_origins_wh": float(np.mean(np.abs(diff[peak]))) if peak.any() else np.nan,
                "energy_bias_peak_origins_wh": float(diff[peak].mean()) if peak.any() else np.nan,
                "energy_mae_nonpeak_origins_wh": float(np.mean(np.abs(diff[~peak]))) if (~peak).any() else np.nan,
                "power_mae_at_horizon_w": float(np.mean(np.abs(step))), "power_bias_at_horizon_projected_minus_reference_w": float(step.mean()),
            }
        )
    return pd.DataFrame(rows)


def cumulative(origin: pd.DataFrame, test_start: int, step_seconds: float) -> pd.DataFrame:
    """Energia acumulada no teste sem sobrepor janelas: 288 janelas de 5 min e 48 de 30 min."""
    parts = []
    for h in (1, 6):
        part = origin[(origin["horizon_steps"] == h) & ((origin["origin_index"] - test_start) % h == 0)].sort_values("origin_index").copy()
        part["window_min"] = h * step_seconds / 60.0
        part["window_end_time_h"] = (part["origin_index"] + h) * step_seconds / 3600.0
        grouped = part.groupby(["model", "r"], sort=False)
        part["cumulative_reference_wh"] = grouped["reference_energy_wh"].cumsum()
        part["cumulative_projected_wh"] = grouped["projected_energy_wh"].cumsum()
        part["cumulative_projected_minus_reference_wh"] = part["cumulative_projected_wh"] - part["cumulative_reference_wh"]
        parts.append(part[["model", "r", "window_min", "origin_index", "window_end_time_h", "reference_energy_wh", "projected_energy_wh", "cumulative_reference_wh", "cumulative_projected_wh", "cumulative_projected_minus_reference_wh"]])
    return pd.concat(parts, ignore_index=True)


def sensitivity(fan: FanModel, wide_all: dict, exponents: list[float], cfg: dict, step_seconds: float, reference_r: float) -> pd.DataFrame:
    """Sensibilidade do erro de energia a r e a variações multiplicativas de P_idle e P_busy."""
    h = int(cfg["horizon_steps"])
    settings = [("r", r, fan.p_idle, fan.p_busy, r) for r in exponents]
    settings += [("p_idle", f, fan.p_idle * f, fan.p_busy, reference_r) for f in cfg["p_idle_factors"]]
    settings += [("p_busy", f, fan.p_idle, fan.p_busy * f, reference_r) for f in cfg["p_busy_factors"]]
    rows = []
    for kind, factor, p_idle, p_busy, r in settings:
        model_fan = FanModel(p_idle, p_busy)
        for model, wide in wide_all.items():
            e = energies(model_fan, wide, r, [h], step_seconds)
            ref, proj = e["reference"][h] / JOULES_PER_WH, e["projected"][h] / JOULES_PER_WH
            diff = proj - ref
            rows.append(
                {
                    "parameter": kind, "factor_or_value": factor, "p_idle_w": p_idle, "p_busy_w": p_busy, "r": r, "model": model, "horizon_steps": h,
                    "energy_mae_wh": float(np.mean(np.abs(diff))), "energy_bias_projected_minus_reference_wh": float(diff.mean()),
                    "energy_mean_abs_relative_error": float(np.mean(np.abs(diff / ref))), "total_reference_wh": float(ref.sum()), "total_projected_wh": float(proj.sum()),
                }
            )
    return pd.DataFrame(rows)


def jensen(aggregate: pd.DataFrame, step_seconds: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = aggregate.copy()
    out = pd.DataFrame(
        {
            "bin_index": a["bin"], "time_h": a["bin"] * step_seconds / 3600.0, "r": a["r"], "n_present_machines": a["n_present"], "coverage": a["coverage"],
            "u_mean": a["u_mean"], "u_variance": a["u_var"], "power_of_mean_u_w_per_server": a["p_of_mean_w"], "mean_of_power_w_per_server": a["mean_of_p_w"],
            "jensen_gap_w_per_server": a["mean_of_p_w"] - a["p_of_mean_w"], "fleet_sum_of_power_w": a["sum_of_p_w"],
            "fleet_n_times_power_of_mean_w": a["n_present"] * a["p_of_mean_w"], "jensen_gap_fleet_w": a["sum_of_p_w"] - a["n_present"] * a["p_of_mean_w"],
            "second_order_mean_of_power_w_per_server": a["second_order_mean_of_p_w_diagnostic"],
            "second_order_minus_exact_w_per_server": a["second_order_mean_of_p_w_diagnostic"] - a["mean_of_p_w"], "unstable_approximation": a["unstable_approximation"],
        }
    )
    summary = (
        out.groupby("r")
        .agg(
            n_bins=("bin_index", "size"), mean_gap_w_per_server=("jensen_gap_w_per_server", "mean"), min_gap_w_per_server=("jensen_gap_w_per_server", "min"),
            max_gap_w_per_server=("jensen_gap_w_per_server", "max"), mean_power_of_mean_w=("power_of_mean_u_w_per_server", "mean"),
            mean_relative_gap=("jensen_gap_w_per_server", lambda s: float((s / out.loc[s.index, "power_of_mean_u_w_per_server"]).mean())),
            mean_gap_fleet_w=("jensen_gap_fleet_w", "mean"), mean_fleet_sum_w=("fleet_sum_of_power_w", "mean"), mean_n_present=("n_present_machines", "mean"),
            mean_abs_second_order_error_w=("second_order_minus_exact_w_per_server", lambda s: float(s.abs().mean())),
            max_abs_second_order_error_w=("second_order_minus_exact_w_per_server", lambda s: float(s.abs().max())), unstable_bins=("unstable_approximation", "sum"),
        )
        .reset_index()
    )
    return out, summary
