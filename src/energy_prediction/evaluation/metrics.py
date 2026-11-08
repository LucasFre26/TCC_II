"""MAE, RMSE, MASE e skill score contra a persistência."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(truth - pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def mase(model_mae: float, persistence_mae: float) -> float:
    if persistence_mae == 0:
        raise ZeroDivisionError("MAE da persistência é zero; MASE indefinido.")
    return float(model_mae / persistence_mae)


def skill_score(model_rmse: float, persistence_rmse: float) -> float:
    if persistence_rmse == 0:
        raise ZeroDivisionError("RMSE da persistência é zero; skill score indefinido.")
    return float(1.0 - (model_rmse / persistence_rmse))
