"""Rotas de potência de TI modelada da frota e decomposição do erro de energia (H6).

Para cada origem t, o conjunto fixo F_t contém as máquinas presentes em t com CPU observada nos
passos 1..H. Uma rota substitui a utilização de cada máquina i ∈ F_t por:
- a própria CPU (rota por máquina; com CPU observada, é a referência);
- a média ponderada do seu grupo em F_t (rota por grupo);
- a média ponderada de F_t (rota da média).
Potência da rota = Σ_i P_{s(i)}(x_i), com s(i) o substituto da máquina (homogêneo ou cenário
sintético). Diferença total = rota prevista − referência = aproximação por agregação (rota
observada − referência) + erro de previsão (rota prevista − rota observada); a soma fecha por
construção. A diferença entre a média de F_t e o alvo agregado oficial é o termo de população.
Estimativas paramétricas num servidor substituto; não são medições e não há PUE.
"""

from __future__ import annotations

import numpy as np

from energy_prediction.energy.fan import FanModel


def fixed_sets(cpu: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    """Máscara (origens, máquinas): presentes em t e com CPU observada em t+1..t+H."""
    o = np.asarray(origins, dtype=int)
    rows = o[:, None] + np.arange(int(horizon) + 1)[None, :]
    return np.isfinite(cpu[rows]).all(axis=1)


def observed_paths(cpu: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    """(origens, H+1, máquinas) da CPU observada em t..t+H."""
    o = np.asarray(origins, dtype=int)
    rows = o[:, None] + np.arange(int(horizon) + 1)[None, :]
    return cpu[rows].astype(float)


def group_means(U: np.ndarray, mask: np.ndarray, weights: np.ndarray, groups: np.ndarray, labels: list) -> np.ndarray:
    """(origens, H+1, grupos): média ponderada de U nas máquinas de F_t de cada grupo."""
    out = np.full((U.shape[0], U.shape[1], len(labels)), np.nan)
    for j, g in enumerate(labels):
        m = mask & (groups[None, :] == g)
        w = m * weights[None, :]
        tot = w.sum(axis=1)
        num = np.einsum("okm,om->ok", np.where(np.isfinite(U), U, 0.0), w)
        out[:, :, j] = np.where(tot[:, None] > 0, num / np.where(tot[:, None] > 0, tot[:, None], 1.0), np.nan)
    return out


def _u(x: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=float) / 100.0
    clipped = int(np.sum((x < 0) | (x > 1)))
    return np.clip(x, 0.0, 1.0), clipped


def machine_route_power(U: np.ndarray, mask: np.ndarray, substitute_of: np.ndarray, fans: dict[str, FanModel], r: float) -> tuple[np.ndarray, int]:
    """(origens, H+1) potência somada das máquinas de F_t com a utilização própria."""
    u, clipped = _u(U)
    total = np.zeros(U.shape[:2])
    for sid, fan in fans.items():
        m = mask & (substitute_of[None, :] == sid)
        if m.any():
            p = fan.power_from_u(np.where(np.isfinite(u), u, 0.0), r)
            total += np.einsum("okm,om->ok", p, m.astype(float))
    return total, clipped


def grouped_route_power(X: np.ndarray, mask: np.ndarray, groups: np.ndarray, labels: list, substitute_of: np.ndarray, fans: dict[str, FanModel], r: float) -> tuple[np.ndarray, int]:
    """(origens, H+1) potência com a utilização de cada máquina trocada pela do seu grupo (X: origens, H+1, grupos)."""
    u, clipped = _u(X)
    total = np.zeros(X.shape[:2])
    for sid, fan in fans.items():
        for j, g in enumerate(labels):
            count = (mask & (groups[None, :] == g) & (substitute_of[None, :] == sid)).sum(axis=1)
            if count.any():
                p = fan.power_from_u(np.where(np.isfinite(u[:, :, j]), u[:, :, j], 0.0), r)
                total += p * count[:, None]
    return total, clipped


def horizon_energy_wh(power: np.ndarray, horizon: int, step_seconds: float) -> np.ndarray:
    """Trapézio de t a t+H com o extremo esquerdo na potência da origem (Wh)."""
    p = np.asarray(power, dtype=float)
    h = int(horizon)
    joules = step_seconds * (p[:, 0] / 2.0 + p[:, 1:h].sum(axis=1) + p[:, h] / 2.0)
    return joules / 3600.0


def decompose(reference: np.ndarray, route_observed: np.ndarray, route_predicted: np.ndarray) -> dict[str, np.ndarray]:
    total = route_predicted - reference
    aggregation = route_observed - reference
    forecast = route_predicted - route_observed
    return {"total": total, "aggregation": aggregation, "forecast": forecast, "closure": total - aggregation - forecast}
