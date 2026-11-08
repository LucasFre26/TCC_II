"""Efeito de agregar a utilização antes de aplicar a curva de potência (desigualdade de Jensen).

Em cada intervalo, nas máquinas da população principal com CPU naquele intervalo e com os pesos
cpu_num da agregação principal:
  - P(média): potência da utilização média ponderada;
  - média de P: média ponderada das potências individuais;
  - soma de P: soma das potências individuais (uma máquina = um servidor substituto). É a coluna
    de frota no servidor substituto; P(média) vezes o número de máquinas não recebe esse nome.
A aproximação de segunda ordem P(ū) + ½ ΔP g''(ū) Var(u) é só diagnóstico; o resultado é a
diferença numérica. Intervalos com ū abaixo do limite e r != 1 são contados como instáveis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_prediction.energy.fan import FanModel, shape_second_derivative, utilization


def aggregate_effect(
    grid: pd.DataFrame,
    machine_info: pd.DataFrame,
    fan: FanModel,
    exponents: list[float],
    *,
    unstable_below: float,
    population: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    weights = machine_info.set_index("machine_id")["cpu_num"].astype(float)
    known = grid["machine_id"].isin(weights.index)
    if population is not None:
        known &= grid["machine_id"].isin(set(population))
    frame = grid.loc[known, ["machine_id", "bin", "cpu"]].copy()
    u, invalid = utilization(frame["cpu"].to_numpy())
    frame["u"] = u
    frame = frame.loc[frame["u"].notna()]
    frame["w"] = frame["machine_id"].map(weights).to_numpy()
    total_weight = float(weights.loc[weights.index.isin(grid.loc[known, "machine_id"].unique())].sum())

    by_bin = frame.groupby("bin")
    w_sum = by_bin["w"].sum()
    u_mean = (frame["u"] * frame["w"]).groupby(frame["bin"]).sum() / w_sum
    deviation = frame["u"].to_numpy() - frame["bin"].map(u_mean).to_numpy()
    u_var = pd.Series(frame["w"].to_numpy() * deviation**2, index=frame.index).groupby(frame["bin"]).sum() / w_sum
    base = pd.DataFrame(
        {
            "n_present": by_bin.size(),
            "weight_present": w_sum,
            "coverage": w_sum / total_weight if total_weight > 0 else np.nan,
            "u_mean": u_mean,
            "u_var": u_var,
        }
    )
    rows = []
    for r in exponents:
        r = float(r)
        power = fan.power_from_u(frame["u"].to_numpy(), r)
        mean_p = pd.Series(power * frame["w"].to_numpy(), index=frame.index).groupby(frame["bin"]).sum() / w_sum
        sum_p = pd.Series(power, index=frame.index).groupby(frame["bin"]).sum()
        part = base.copy()
        part["r"] = r
        part["p_of_mean_w"] = fan.power_from_u(part["u_mean"].to_numpy(), r)
        part["mean_of_p_w"] = mean_p
        part["sum_of_p_w"] = sum_p
        part["jensen_gap_w"] = part["mean_of_p_w"] - part["p_of_mean_w"]
        curvature = shape_second_derivative(part["u_mean"].to_numpy(), r)
        part["second_order_mean_of_p_w_diagnostic"] = part["p_of_mean_w"] + 0.5 * fan.dynamic_range * curvature * part["u_var"]
        part["unstable_approximation"] = (part["u_mean"] < float(unstable_below)) & (r != 1.0)
        rows.append(part.reset_index().rename(columns={"index": "bin"}))
    table = pd.concat(rows, ignore_index=True)
    summary = {
        "quantity": "potência de TI modelada no servidor substituto (estimativa paramétrica)",
        "invalid_cpu_values": int(invalid),
        "n_bins": int(base.shape[0]),
        "population_weight": total_weight,
        "per_exponent": [],
    }
    for r, part in table.groupby("r"):
        summary["per_exponent"].append(
            {
                "r": float(r),
                "mean_p_of_mean_w": float(part["p_of_mean_w"].mean()),
                "mean_mean_of_p_w": float(part["mean_of_p_w"].mean()),
                "mean_jensen_gap_w": float(part["jensen_gap_w"].mean()),
                "max_abs_jensen_gap_w": float(part["jensen_gap_w"].abs().max()),
                "mean_sum_of_p_w": float(part["sum_of_p_w"].mean()),
                "unstable_bins": int(part["unstable_approximation"].sum()),
            }
        )
    return table, summary
