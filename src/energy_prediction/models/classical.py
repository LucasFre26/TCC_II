"""Ridge e histogram gradient boosting, um ajuste por horizonte."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from energy_prediction.evaluation.metrics import mae
from energy_prediction.preprocessing.scale import apply_scaler, fit_scaler

ALLOWED_MODELS = (
    "persistence",
    "moving_average",
    "ridge",
    "hist_gradient_boosting",
)


def select_ridge_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    alphas: list[float],
) -> tuple[float, np.ndarray]:
    """Escolhe o alpha de menor MAE de validação. Empate fica com o menor alpha."""
    best_alpha = None
    best_mae = None
    best_test_placeholder = None
    ordered = sorted(alphas)
    for alpha in ordered:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        validation_mae = mae(y_validation, model.predict(x_validation))
        if best_mae is None or validation_mae < best_mae - 1e-12:
            best_mae = validation_mae
            best_alpha = alpha
            best_test_placeholder = model
    assert best_alpha is not None and best_test_placeholder is not None
    return best_alpha, best_test_placeholder.predict(x_validation)


def predict_ridge(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    target: str,
    alphas: list[float],
) -> tuple[np.ndarray, float]:
    scaler = fit_scaler(train.loc[:, columns])
    x_train = apply_scaler(scaler, train.loc[:, columns])
    x_validation = apply_scaler(scaler, validation.loc[:, columns])
    x_test = apply_scaler(scaler, test.loc[:, columns])
    y_train = train[target].to_numpy(dtype=float)
    y_validation = validation[target].to_numpy(dtype=float)
    best_alpha, _ = select_ridge_alpha(x_train, y_train, x_validation, y_validation, alphas)
    model = Ridge(alpha=best_alpha)
    model.fit(x_train, y_train)
    return model.predict(x_test), best_alpha


def predict_hist_gradient_boosting(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    target: str,
    seed: int,
) -> np.ndarray:
    scaler = fit_scaler(train.loc[:, columns])
    model = HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=100,
        learning_rate=0.1,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(apply_scaler(scaler, train.loc[:, columns]), train[target].to_numpy(dtype=float))
    return model.predict(apply_scaler(scaler, test.loc[:, columns]))
