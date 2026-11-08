"""XGBoost no incremento de CPU: aprende cpu(t+h) - cpu(t) e reconstrói cpu(t) + incremento.

`cpu_t` entra no vetor de atributos e na reconstrução, em todas as configurações.
"""

from __future__ import annotations

import numpy as np

from energy_prediction.features.cpu_features import PERSISTENCE_INPUT
from energy_prediction.models.registry import XGBoostModel


def reconstruct(cpu_at_origin, increment) -> np.ndarray:
    return np.asarray(cpu_at_origin, dtype=float) + np.asarray(increment, dtype=float)


class DeltaXGBoost:
    def __init__(self, params: dict, *, feature_names: list[str], seed: int, n_jobs: int, fixed: dict | None = None):
        if PERSISTENCE_INPUT not in feature_names:
            raise ValueError(f"O modelo de incremento exige {PERSISTENCE_INPUT} entre os atributos.")
        self._origin = list(feature_names).index(PERSISTENCE_INPUT)
        self._model = XGBoostModel(params, seed=seed, n_jobs=n_jobs, fixed=fixed)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "DeltaXGBoost":
        X = np.asarray(X, dtype=float)
        self._model.fit(X, np.asarray(y, dtype=float) - X[:, self._origin], sample_weight=sample_weight)
        return self

    def predict_increment(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return reconstruct(X[:, self._origin], self.predict_increment(X))
