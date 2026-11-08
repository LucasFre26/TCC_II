"""Modelo de potência de Fan: P(u) = P_idle + (P_busy - P_idle) * (2u - u^r).

u é a utilização de CPU dividida por 100. Valores fora de [0, 100] ficam ausentes e são contados;
a restrição a [0, 1] só é aplicada depois dessa rejeição (protege contra erro de arredondamento).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QUANTITY = "potência de TI modelada (estimativa paramétrica; não é potência medida)"


class FanParameterError(ValueError):
    """Potências inválidas ou expoente fora do domínio do modelo."""


@dataclass(frozen=True)
class FanModel:
    p_idle: float
    p_busy: float

    def __post_init__(self):
        if not (np.isfinite(self.p_idle) and np.isfinite(self.p_busy)) or self.p_idle < 0 or self.p_busy <= self.p_idle:
            raise FanParameterError(f"Exige 0 <= P_idle < P_busy; recebido P_idle={self.p_idle}, P_busy={self.p_busy}.")

    @property
    def dynamic_range(self) -> float:
        return float(self.p_busy - self.p_idle)

    def power_from_u(self, u: np.ndarray, r: float) -> np.ndarray:
        return self.p_idle + self.dynamic_range * shape(u, r)

    def power(self, cpu_percent, r: float) -> tuple[np.ndarray, int]:
        u, invalid = utilization(cpu_percent)
        return self.power_from_u(u, r), invalid


def _check_r(r: float) -> float:
    r = float(r)
    if r < 1.0:
        raise FanParameterError(f"Expoente r={r} fora do domínio usado (r >= 1).")
    return r


def utilization(cpu_percent) -> tuple[np.ndarray, int]:
    """u = cpu/100; fora de [0, 100] vira NaN e entra na contagem de inválidos."""
    cpu = np.asarray(cpu_percent, dtype=float)
    invalid = np.isfinite(cpu) & ((cpu < 0.0) | (cpu > 100.0))
    u = np.where(invalid, np.nan, cpu / 100.0)
    return np.clip(u, 0.0, 1.0), int(invalid.sum())


def shape(u: np.ndarray, r: float) -> np.ndarray:
    """g(u) = 2u - u^r."""
    r = _check_r(r)
    u = np.asarray(u, dtype=float)
    return 2.0 * u - np.power(u, r)


def shape_derivative(u: np.ndarray, r: float) -> np.ndarray:
    """g'(u) = 2 - r u^(r-1)."""
    r = _check_r(r)
    u = np.asarray(u, dtype=float)
    return 2.0 - r * np.power(u, r - 1.0) if r != 1.0 else np.ones_like(u)


def shape_second_derivative(u: np.ndarray, r: float) -> np.ndarray:
    """g''(u) = -r(r-1) u^(r-2) para r != 1; zero para r = 1 (diverge em u -> 0 quando 1 < r < 2)."""
    r = _check_r(r)
    u = np.asarray(u, dtype=float)
    if r == 1.0:
        return np.zeros_like(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -r * (r - 1.0) * np.power(u, r - 2.0)
