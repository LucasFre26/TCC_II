"""Nível local e nível local com tendência (statsmodels), previsão só pelo estado filtrado.

As variâncias são estimadas por máxima verossimilhança apenas no segmento de treino (observações
até a última origem de treino). Com esses parâmetros fixos, o filtro de Kalman percorre a série;
a previsão de horizonte h na origem t é Z·T^h·a(t|t), o estado filtrado em t propagado h passos
sem observação nova. O estado suavizado nunca é calculado: `fit` pede só os parâmetros
(`return_params=True`), e a previsão usa `filter`, não `smooth`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

SPECIFICATIONS = {"local_level": "local level", "local_linear_trend": "local linear trend"}


class StateSpaceError(ValueError):
    """Especificação desconhecida ou segmento de treino curto demais."""


def _model(endog: np.ndarray, specification: str):
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
    except ImportError as error:  # pragma: no cover - depende do extra v3
        raise ImportError("statsmodels não está instalado; instale o extra `v3`.") from error
    if specification not in SPECIFICATIONS:
        raise StateSpaceError(f"Especificação desconhecida: {specification!r}. Use {list(SPECIFICATIONS)}.")
    return UnobservedComponents(np.asarray(endog, dtype=float), level=SPECIFICATIONS[specification])


def fit_parameters(segment: np.ndarray, specification: str, *, maxiter: int = 200) -> np.ndarray:
    """Máxima verossimilhança só no segmento de treino; devolve os parâmetros, sem suavizar."""
    segment = np.asarray(segment, dtype=float)
    if np.isfinite(segment).sum() < 10:
        raise StateSpaceError("Segmento de treino com menos de 10 observações válidas.")
    model = _model(segment, specification)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.fit(disp=False, maxiter=int(maxiter), return_params=True), dtype=float)


def _time_invariant(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 3:
        if matrix.shape[2] != 1:
            raise StateSpaceError("Matriz do sistema variante no tempo não é suportada.")
        matrix = matrix[:, :, 0]
    return matrix


@dataclass
class FilteredForecasts:
    """Previsões por horizonte, uma por instante da série, feitas com o estado filtrado em t."""

    specification: str
    params: np.ndarray
    by_horizon: dict[int, np.ndarray]
    kind: str = "filtered"


def filtered_forecasts(series: np.ndarray, specification: str, params: np.ndarray, horizons: list[int]) -> FilteredForecasts:
    model = _model(series, specification)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.filter(np.asarray(params, dtype=float))
    state = np.asarray(result.filtered_state)  # (k, n): a(t|t)
    transition = _time_invariant(model.ssm["transition"])
    design = _time_invariant(model.ssm["design"])
    out = {}
    for h in horizons:
        out[int(h)] = (design @ np.linalg.matrix_power(transition, int(h)) @ state).ravel()
    return FilteredForecasts(specification=specification, params=np.asarray(params, dtype=float), by_horizon=out)


class StateSpaceForecaster:
    """Ajusta parâmetros por fim de segmento e guarda as previsões filtradas (reuso entre horizontes)."""

    def __init__(self, series: np.ndarray, *, horizons: list[int], maxiter: int = 200):
        self.series = np.asarray(series, dtype=float)
        self.horizons = sorted({int(h) for h in horizons})
        self.maxiter = int(maxiter)
        self._cache: dict[tuple[str, int], FilteredForecasts] = {}

    def forecasts(self, specification: str, segment_end: int) -> FilteredForecasts:
        """Parâmetros estimados em series[:segment_end + 1]; filtro aplicado na série inteira."""
        key = (specification, int(segment_end))
        if key not in self._cache:
            params = fit_parameters(self.series[: int(segment_end) + 1], specification, maxiter=self.maxiter)
            self._cache[key] = filtered_forecasts(self.series, specification, params, self.horizons)
        return self._cache[key]
