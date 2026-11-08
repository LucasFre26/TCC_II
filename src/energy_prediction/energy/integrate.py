"""Energia de um horizonte pela regra do trapézio sobre a trajetória de potência.

    E(t, h) = Δt · [ P(u(t))/2 + P(t+1) + ... + P(t+h-1) + P(t+h)/2 ]

O extremo esquerdo é sempre a potência da CPU observada na origem. Sem algum dos passos
1..h, a energia do horizonte não é calculada: o valor terminal não é repetido no lugar dos
passos ausentes.
"""

from __future__ import annotations

import numpy as np

JOULES_PER_WH = 3600.0


class IncompleteTrajectoryError(ValueError):
    """Faltam passos intermediários da trajetória até o horizonte."""


def trapezoid_energy(power_path, step_seconds: float) -> float:
    """Energia (J) de uma trajetória [P(t), P(t+1), ..., P(t+h)]."""
    path = np.asarray(power_path, dtype=float)
    if path.ndim != 1 or len(path) < 2:
        raise IncompleteTrajectoryError("A trajetória precisa da origem e de pelo menos um passo.")
    if not np.all(np.isfinite(path)):
        raise IncompleteTrajectoryError("Trajetória com passo ausente; a energia não é preenchida com o valor terminal.")
    return float(step_seconds * (path[0] / 2.0 + path[1:-1].sum() + path[-1] / 2.0))


def horizon_energies(p_origin, p_steps, horizons: list[int], step_seconds: float) -> dict[int, np.ndarray]:
    """Energia (J) por origem para cada horizonte; NaN onde algum passo 1..h está ausente.

    `p_origin` tem forma (n,), `p_steps` tem forma (n, H) com os passos 1..H.
    """
    origin = np.asarray(p_origin, dtype=float)
    steps = np.asarray(p_steps, dtype=float)
    if steps.ndim != 2 or steps.shape[0] != origin.shape[0]:
        raise ValueError("p_steps precisa ter forma (n_origens, n_passos).")
    out = {}
    for h in horizons:
        if h > steps.shape[1]:
            raise IncompleteTrajectoryError(f"Horizonte {h} pede mais passos que os {steps.shape[1]} disponíveis.")
        path = np.column_stack([origin, steps[:, :h]])
        complete = np.all(np.isfinite(path), axis=1)
        energy = step_seconds * (path[:, 0] / 2.0 + path[:, 1:-1].sum(axis=1) + path[:, -1] / 2.0)
        out[int(h)] = np.where(complete, energy, np.nan)
    return out
