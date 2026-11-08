"""Como o erro de CPU se traduz em diferença de potência e de energia modeladas.

A diferença é entre duas grandezas modeladas (energia projetada pela trajetória prevista menos
energia de referência pela trajetória observada). Não é erro contra um medidor e não vira
intervalo de confiança da energia elétrica do cluster. O mapa local ΔP·g'(u)·erro é diagnóstico.
"""

from __future__ import annotations

import numpy as np

from energy_prediction.energy.fan import FanModel, shape_derivative

STATEMENT = (
    "Diferença entre energia de TI modelada com a trajetória prevista e com a trajetória observada, "
    "no mesmo servidor substituto. Não é erro contra medidor."
)


def linearized_power_difference(fan: FanModel, u_observed, u_predicted, r: float) -> np.ndarray:
    """(P_busy - P_idle) · g'(u_obs) · (u_prev - u_obs), por passo."""
    u_obs = np.asarray(u_observed, dtype=float)
    return fan.dynamic_range * shape_derivative(u_obs, r) * (np.asarray(u_predicted, dtype=float) - u_obs)


def linearized_energy_difference(power_differences, horizons: list[int], step_seconds: float) -> dict[int, np.ndarray]:
    """Integral pelo trapézio da diferença linearizada; na origem a diferença é zero (CPU observada)."""
    diff = np.asarray(power_differences, dtype=float)
    out = {}
    for h in horizons:
        path = diff[:, :h]
        complete = np.all(np.isfinite(path), axis=1)
        energy = step_seconds * (path[:, :-1].sum(axis=1) + path[:, -1] / 2.0)
        out[int(h)] = np.where(complete, energy, np.nan)
    return out
