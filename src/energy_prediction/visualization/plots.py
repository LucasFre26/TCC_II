"""Figuras gravadas em arquivo, sem interface gráfica."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_gpu_series(frame: pd.DataFrame, destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(frame["timestamp"], frame["gpu"], color="#1f4e79", linewidth=1.0)
    axis.set_title("Duty cycle agregado de GPU (resultados preliminares)")
    axis.set_xlabel("timestamp anonimizado")
    axis.set_ylabel("média do duty cycle")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def plot_horizon_forecasts(
    timestamps: pd.Series,
    observed: pd.Series,
    forecasts: dict[str, pd.Series],
    destination: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(timestamps, observed, color="black", linewidth=1.4, label="observado")
    for name, values in forecasts.items():
        axis.plot(timestamps, values, linewidth=1.0, label=name)
    axis.set_title("Horizonte 1, teste: observado e previsto (resultados preliminares)")
    axis.set_xlabel("timestamp anonimizado")
    axis.set_ylabel("duty cycle agregado")
    axis.legend(fontsize=8)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120)
    plt.close(fig)
