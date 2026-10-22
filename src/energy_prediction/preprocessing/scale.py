"""Padronização ajustada só no treino."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(train_features: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_features.to_numpy(dtype=float))
    return scaler


def apply_scaler(scaler: StandardScaler, features: pd.DataFrame) -> np.ndarray:
    return scaler.transform(features.to_numpy(dtype=float))
