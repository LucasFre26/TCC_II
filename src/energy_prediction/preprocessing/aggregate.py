"""Agregação por instante e junção das duas séries."""

from __future__ import annotations

import pandas as pd


def aggregate_by_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Média aritmética do valor e contagem de contêineres em cada instante."""
    grouped = (
        frame.groupby("timestamp", sort=True)
        .agg(value=("value", "mean"), n_containers=("value", "size"))
        .reset_index()
    )
    return grouped


def join_series(
    gpu: pd.DataFrame, memory: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Inner join. Devolve também quantos instantes existem em só uma série."""
    gpu_times = set(gpu["timestamp"].tolist())
    memory_times = set(memory["timestamp"].tolist())
    excluded = len(gpu_times.symmetric_difference(memory_times))

    merged = gpu.merge(memory, on="timestamp", how="inner", suffixes=("_gpu", "_memory"))
    joined = pd.DataFrame(
        {
            "timestamp": merged["timestamp"],
            "gpu": merged["value_gpu"],
            "memory": merged["value_memory"],
            "n_containers_gpu": merged["n_containers_gpu"],
            "n_containers_memory": merged["n_containers_memory"],
        }
    )
    return joined, excluded


class DuplicateTimestampError(ValueError):
    """O agregado ainda tem mais de uma linha para o mesmo instante."""


def sort_unique(frame: pd.DataFrame) -> pd.DataFrame:
    if frame["timestamp"].duplicated().any():
        raise DuplicateTimestampError("Timestamp duplicado depois da agregação.")
    ordered = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if not ordered["timestamp"].is_monotonic_increasing:
        raise DuplicateTimestampError("A série ordenada não é estritamente crescente.")
    return ordered
