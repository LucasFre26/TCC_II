"""XGBoost global por máquina (H2): um modelo compartilhado por horizonte, agregação ponderada.

A previsão agregada na origem t é a média, ponderada por `cpu_num`, das previsões das máquinas
presentes em t com todos os atributos; máquina sem CPU em t não recebe previsão e não entra no
denominador. Com `device: cuda` o ajuste usa o XGBoost em GPU (o build instalado tem CUDA).
"""

from __future__ import annotations

import time

import numpy as np

from energy_prediction.features.machine_features import MachineFeatures


def xgb_device(requested: str, effective_torch: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import xgboost

        has_cuda = bool(xgboost.build_info().get("USE_CUDA"))
    except Exception:  # pragma: no cover - depende do ambiente
        has_cuda = False
    if requested == "cuda" and not has_cuda:
        raise RuntimeError("XGBoost sem CUDA neste ambiente.")
    return "cuda" if has_cuda and (requested == "cuda" or effective_torch == "cuda") else "cpu"


class GlobalXGBoost:
    def __init__(self, params: dict, *, seed: int, n_jobs: int, device: str = "cpu"):
        from xgboost import XGBRegressor

        self.device = device
        self.params = {"objective": "reg:squarederror", "tree_method": "hist", "verbosity": 0, **params, "random_state": int(seed), "n_jobs": int(n_jobs), "device": device}
        self._model = XGBRegressor(**self.params)
        self.fit_seconds = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GlobalXGBoost":
        started = time.perf_counter()
        self._model.fit(X, y)
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.device == "cuda":
            self._model.set_params(device="cpu")  # dados em CPU: evita a cópia implícita e o aviso de dispositivo
        out = self._model.predict(X)
        if self.device == "cuda":
            self._model.set_params(device=self.device)
        return out


def present_columns(features: MachineFeatures, origin: int, columns: np.ndarray) -> np.ndarray:
    return columns[np.isfinite(features.cpu[int(origin), columns])]


def aggregate_predictions(features: MachineFeatures, model, origins: np.ndarray, columns: np.ndarray, weights: np.ndarray, steps: int, *, keep_machine: bool = False, chunk: int = 32):
    """Previsão agregada por origem, máquinas usadas e, opcionalmente, a matriz (origens, máquinas) por máquina."""
    origins = np.asarray(origins, dtype=int)
    columns = np.asarray(columns, dtype=int)
    agg = np.full(len(origins), np.nan)
    used = np.zeros(len(origins), dtype=int)
    dropped = np.zeros(len(origins), dtype=int)
    observed_same_set = np.full(len(origins), np.nan)
    per_machine = np.full((len(origins), len(columns)), np.nan, dtype=np.float32) if keep_machine else None
    for start in range(0, len(origins), int(chunk)):
        part = origins[start : start + int(chunk)]
        o = np.repeat(part, len(columns))
        c = np.tile(columns, len(part))
        X = features.design(o, c)
        ok = np.isfinite(features.cpu[o, c]) & np.isfinite(X).all(axis=1)
        pred = np.full(len(o), np.nan)
        if ok.any():
            pred[ok] = model.predict(X[ok])
        y = features.target(o, c, steps)
        for i, origin in enumerate(part):
            sl = slice(i * len(columns), (i + 1) * len(columns))
            mask = ok[sl]
            present = np.isfinite(features.cpu[origin, columns])
            dropped[start + i] = int((present & ~mask).sum())
            used[start + i] = int(mask.sum())
            if mask.any():
                w = weights[columns][mask]
                agg[start + i] = float(np.sum(pred[sl][mask] * w) / np.sum(w))
                ym = np.isfinite(y[sl]) & mask
                if ym.any():
                    observed_same_set[start + i] = float(np.sum(y[sl][ym] * weights[columns][ym]) / np.sum(weights[columns][ym]))
            if keep_machine:
                per_machine[start + i, mask] = pred[sl][mask]
    return agg, used, dropped, observed_same_set, per_machine
