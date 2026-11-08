"""Estatísticas da série agregada e juntada."""

from __future__ import annotations

import pandas as pd

from energy_prediction.preprocessing.checks import median_step_seconds


def profile_joined(frame: pd.DataFrame, excluded_timestamps: int) -> dict:
    gpu = frame["gpu"]
    memory = frame["memory"]
    return {
        "n_timestamps": int(len(frame)),
        "median_step_seconds": median_step_seconds(frame["timestamp"]),
        "gpu_zero_fraction": float((gpu == 0).mean()),
        "gpu_min": float(gpu.min()),
        "gpu_median": float(gpu.median()),
        "gpu_mean": float(gpu.mean()),
        "gpu_max": float(gpu.max()),
        "memory_min": float(memory.min()),
        "memory_median": float(memory.median()),
        "memory_mean": float(memory.mean()),
        "memory_max": float(memory.max()),
        "contemporaneous_correlation": float(gpu.corr(memory)),
        "gpu_outside_0_100": int(((gpu < 0) | (gpu > 100)).sum()),
        "excluded_timestamps": int(excluded_timestamps),
        "mean_containers_gpu": float(frame["n_containers_gpu"].mean()),
        "mean_containers_memory": float(frame["n_containers_memory"].mean()),
    }
