"""XGBoost com objetivo quantílico (um modelo por quantil, horizonte e configuração)."""

from __future__ import annotations

import numpy as np


class QuantileXGBoost:
    def __init__(self, quantile: float, params: dict, *, seed: int, n_jobs: int, fixed: dict | None = None):
        from xgboost import XGBRegressor

        if not 0.0 < float(quantile) < 1.0:
            raise ValueError(f"Quantil fora de (0, 1): {quantile}")
        settings = {
            "tree_method": "hist",
            "verbosity": 0,
            **(fixed or {}),
            **params,
            "objective": "reg:quantileerror",
            "quantile_alpha": float(quantile),
            "random_state": int(seed),
            "n_jobs": int(n_jobs),
        }
        self.quantile = float(quantile)
        self._model = XGBRegressor(**settings)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileXGBoost":
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(np.asarray(X, dtype=float))
