"""Baselines univariados na série de GPU."""

from __future__ import annotations

import numpy as np


def persistence_forecast(gpu_at_origin: np.ndarray) -> np.ndarray:
    """Previsão em qualquer horizonte: valor observado na origem."""
    return np.asarray(gpu_at_origin, dtype=float).copy()


def moving_average_forecast(gpu: np.ndarray, origin_indices: np.ndarray, window: int) -> np.ndarray:
    """Média dos `window` valores estritamente anteriores a cada origem."""
    values = np.asarray(gpu, dtype=float)
    origins = np.asarray(origin_indices, dtype=int)
    forecasts = np.empty(len(origins), dtype=float)
    for position, origin in enumerate(origins):
        if origin < window:
            raise ValueError("Origem sem histórico suficiente para a média móvel.")
        forecasts[position] = values[origin - window : origin].mean()
    return forecasts
