"""Híbrido estados + XGBoost: o corretor aprende o resíduo da previsão filtrada fora da amostra.

Resíduo de treino da origem s: y(s+h) - previsão filtrada feita em s (estado atualizado só até s).
A previsão final é a previsão filtrada mais a correção. Previsões feitas com estado suavizado
(que usa observações posteriores à origem) são recusadas.
"""

from __future__ import annotations

import numpy as np

from energy_prediction.models.registry import XGBoostModel


class CausalityError(ValueError):
    """O resíduo oferecido ao corretor não vem de uma previsão filtrada fora da amostra."""


def assert_filtered(forecasts) -> None:
    kind = getattr(forecasts, "kind", None)
    if kind != "filtered":
        raise CausalityError(
            f"O corretor só aceita previsões do estado filtrado até a origem; recebido kind={kind!r}."
        )


def causal_residuals(y_target: np.ndarray, base_forecast: np.ndarray) -> np.ndarray:
    return np.asarray(y_target, dtype=float) - np.asarray(base_forecast, dtype=float)


class HybridCorrector:
    def __init__(self, params: dict, *, seed: int, n_jobs: int, fixed: dict | None = None):
        self._model = XGBoostModel(params, seed=seed, n_jobs=n_jobs, fixed=fixed)

    def fit(self, X: np.ndarray, y: np.ndarray, base: np.ndarray, forecasts) -> "HybridCorrector":
        assert_filtered(forecasts)
        residual = causal_residuals(y, base)
        keep = np.isfinite(residual)
        self._model.fit(np.asarray(X, dtype=float)[keep], residual[keep])
        return self

    def predict(self, X: np.ndarray, base: np.ndarray) -> np.ndarray:
        return np.asarray(base, dtype=float) + self._model.predict(np.asarray(X, dtype=float))
