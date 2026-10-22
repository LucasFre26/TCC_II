"""Tabela supervisionada, atributos causais e corte com purga."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

GPU_FEATURE_PREFIXES = ("gpu_lag_", "gpu_roll_mean_", "gpu_roll_std_")
CALENDAR_FEATURES = ("sin_pos", "cos_pos")


def feature_columns(gpu_lags: list[int], rolling_windows: list[int], memory_lags: list[int]) -> dict[str, list[str]]:
    gpu_columns = [f"gpu_lag_{lag}" for lag in gpu_lags]
    for window in rolling_windows:
        gpu_columns.append(f"gpu_roll_mean_{window}")
        gpu_columns.append(f"gpu_roll_std_{window}")
    gpu_columns.extend(CALENDAR_FEATURES)
    memory_columns = [f"memory_lag_{lag}" for lag in memory_lags]
    return {
        "gpu_only": gpu_columns,
        "gpu_memory": gpu_columns + memory_columns,
    }


def build_supervised(
    frame: pd.DataFrame,
    *,
    gpu_lags: list[int],
    memory_lags: list[int],
    rolling_windows: list[int],
    horizons: list[int],
) -> pd.DataFrame:
    """Uma linha por origem com lookback e com o maior horizonte disponíveis.

    Defasagens e janelas de GPU usam só instantes anteriores à origem.
    A memória em t é contemporânea e permitida; memória futura não entra.
    """
    gpu = frame["gpu"].to_numpy(dtype=float)
    memory = frame["memory"].to_numpy(dtype=float)
    timestamps = frame["timestamp"].to_numpy(dtype=float)
    n = len(frame)
    max_lag = max([*gpu_lags, *memory_lags, *rolling_windows])
    max_horizon = max(horizons)
    if n <= max_lag + max_horizon:
        raise ValueError("Série curta demais para lags e horizontes configurados.")

    origins = np.arange(max_lag, n - max_horizon)
    rows: dict[str, np.ndarray] = {
        "origin_index": origins,
        "timestamp": timestamps[origins],
        "gpu_at_origin": gpu[origins],
    }
    for lag in gpu_lags:
        rows[f"gpu_lag_{lag}"] = gpu[origins - lag]
    for lag in memory_lags:
        rows[f"memory_lag_{lag}"] = memory[origins - lag]
    for window in rolling_windows:
        means = np.empty(len(origins), dtype=float)
        stds = np.empty(len(origins), dtype=float)
        for position, origin in enumerate(origins):
            past = gpu[origin - window : origin]
            means[position] = past.mean()
            stds[position] = past.std(ddof=0)
        rows[f"gpu_roll_mean_{window}"] = means
        rows[f"gpu_roll_std_{window}"] = stds
    phase = origins / n
    rows["sin_pos"] = np.sin(2.0 * math.pi * phase)
    rows["cos_pos"] = np.cos(2.0 * math.pi * phase)
    for horizon in horizons:
        rows[f"y_{horizon}"] = gpu[origins + horizon]
    return pd.DataFrame(rows)


def split_origins(
    n_origins: int,
    *,
    test_fraction: float,
    validation_fraction: float,
    purge_steps: int,
) -> dict[str, np.ndarray]:
    """Corta a lista de origens já restrita ao horizonte mais longo."""
    test_count = int(n_origins * test_fraction)
    if test_count < 1:
        raise ValueError("Bloco de teste vazio.")
    remaining = n_origins - test_count - purge_steps
    if remaining < 2:
        raise ValueError("Série curta demais para treino, validação e purga.")
    validation_count = int(remaining * validation_fraction)
    if validation_count < 1:
        raise ValueError("Bloco de validação vazio.")
    train_count = remaining - validation_count
    if train_count < 1:
        raise ValueError("Bloco de treino vazio.")

    positions = np.arange(n_origins)
    train_end = train_count
    validation_end = train_count + validation_count
    test_start = n_origins - test_count
    if validation_end != test_start - purge_steps:
        raise ValueError("A purga não ficou imediatamente antes do teste.")
    return {
        "train": positions[:train_end],
        "validation": positions[train_end:validation_end],
        "purge": positions[validation_end:test_start],
        "test": positions[test_start:],
    }
