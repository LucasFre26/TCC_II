"""Registro dos modelos da V2: persistência, Ridge e XGBoost. Nenhum outro é permitido."""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from energy_prediction.features.cpu_features import PERSISTENCE_INPUT

ALLOWED_MODELS = ("persistence", "ridge", "xgboost")
FITTED_MODELS = ("ridge", "xgboost")


class UnknownModelError(ValueError):
    """Nome de modelo fora do conjunto permitido da V2."""


def validate_model_names(names: list[str]) -> list[str]:
    unknown = [name for name in names if name not in ALLOWED_MODELS]
    if unknown:
        raise UnknownModelError(
            f"Modelo(s) fora do conjunto permitido na V2: {unknown}. Permitidos: {list(ALLOWED_MODELS)}."
        )
    return list(names)


class PersistenceModel:
    """u_hat(t+h) = u(t), lido da coluna `cpu_t`. Sem parâmetros ajustáveis."""

    def __init__(self, feature_names: list[str]):
        if PERSISTENCE_INPUT not in feature_names:
            raise ValueError(f"A persistência exige a coluna {PERSISTENCE_INPUT} entre as features.")
        self._index = list(feature_names).index(PERSISTENCE_INPUT)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceModel":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, self._index].copy()


class RidgeModel:
    """Ridge sobre features padronizadas; o scaler é ajustado só nos dados de `fit`."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.scaler_: StandardScaler | None = None
        self.model_: Ridge | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        self.scaler_ = StandardScaler().fit(np.asarray(X, dtype=float))
        self.model_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(np.asarray(X, dtype=float)), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler_ is not None and self.model_ is not None
        return self.model_.predict(self.scaler_.transform(np.asarray(X, dtype=float)))


class XGBoostModel:
    """XGBRegressor com semente e n_jobs da configuração. Sem scaler."""

    def __init__(self, params: dict, *, seed: int, n_jobs: int, fixed: dict | None = None):
        try:
            from xgboost import XGBRegressor
        except ImportError as error:  # pragma: no cover - depende do ambiente
            raise ImportError("xgboost não está instalado; instale com `pip install -e .`.") from error
        settings = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "verbosity": 0,
            **(fixed or {}),
            **params,
            "random_state": int(seed),
            "n_jobs": int(n_jobs),
        }
        self.params = settings
        self._model = XGBRegressor(**settings)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(np.asarray(X, dtype=float))


def param_grid(name: str, model_cfg: dict) -> list[dict]:
    """Pontos da grade em ordem determinística; o primeiro melhor vence empates."""
    if name == "ridge":
        return [{"alpha": float(a)} for a in sorted(model_cfg["alphas"])]
    if name == "xgboost":
        grid = model_cfg["grid"]
        keys = sorted(grid)
        return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    raise UnknownModelError(f"Modelo sem grade de hiperparâmetros: {name!r}")


def make_model(name: str, params: dict, *, feature_names: list[str], seed: int, model_cfg: dict | None = None):
    if name == "persistence":
        return PersistenceModel(feature_names)
    if name == "ridge":
        return RidgeModel(params["alpha"])
    if name == "xgboost":
        cfg = model_cfg or {}
        return XGBoostModel(params, seed=seed, n_jobs=int(cfg.get("n_jobs", 1)), fixed=cfg.get("fixed"))
    raise UnknownModelError(f"Modelo desconhecido: {name!r}. Permitidos: {list(ALLOWED_MODELS)}.")
