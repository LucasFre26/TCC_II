"""Figuras da V2: real vs previsto, comparação por horizonte e comparação da ablação."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MODEL_COLORS = {"persistence": "#7f7f7f", "ridge": "#1f77b4", "xgboost": "#d62728"}


def _finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_real_vs_predicted(predictions: pd.DataFrame, path: Path, *, config: str, horizon_steps: int, step_seconds: float) -> None:
    test = predictions[(predictions["split"] == "test_final") & (predictions["horizon_steps"] == horizon_steps)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    reference = test[test["model"] == "persistence"].sort_values("origin_index")
    ax.plot(reference["timestamp"] / 3600.0, reference["y_true"], color="black", lw=1.4, label="observado")
    for model in ("persistence", "ridge", "xgboost"):
        part = test[(test["model"] == model) & (test["config"].isin([config, "-"]))].sort_values("origin_index")
        if not part.empty:
            ax.plot(part["timestamp"] / 3600.0, part["y_pred"], color=MODEL_COLORS[model], lw=1.0, label=model)
    ax.set_xlabel("tempo do trace (h desde o início; não é hora civil)")
    ax.set_ylabel("CPU (%)")
    ax.set_title(f"Teste final, horizonte {horizon_steps * step_seconds / 60:g} min, configuração {config}")
    ax.legend(ncol=4, fontsize=8)
    _finish(fig, path)


def plot_horizon_comparison(metrics: pd.DataFrame, path: Path, *, config: str) -> None:
    test = metrics[metrics["split"] == "test_final"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for model in ("persistence", "ridge", "xgboost"):
        part = test[(test["model"] == model) & (test["config"].isin([config, "-"]))].sort_values("horizon_seconds")
        if not part.empty:
            ax.plot(part["horizon_seconds"] / 60.0, part["mae"], marker="o", color=MODEL_COLORS[model], label=model)
    ax.set_xlabel("horizonte (min)")
    ax.set_ylabel("MAE no teste final (pontos percentuais de CPU)")
    ax.set_title(f"Erro por horizonte, configuração {config}")
    ax.legend(fontsize=8)
    _finish(fig, path)


def plot_ablation_comparison(metrics: pd.DataFrame, path: Path) -> None:
    test = metrics[metrics["split"] == "test_final"]
    horizons = sorted(test["horizon_seconds"].unique())
    configs = sorted(c for c in test["config"].unique() if c != "-")
    models = [m for m in ("ridge", "xgboost") if m in set(test["model"])]
    fig, axes = plt.subplots(1, len(horizons), figsize=(3.6 * len(horizons), 3.6), sharey=True, squeeze=False)
    width = 0.8 / max(len(models), 1)
    for ax, horizon in zip(axes[0], horizons):
        block = test[test["horizon_seconds"] == horizon]
        for j, model in enumerate(models):
            values = [block[(block["model"] == model) & (block["config"] == c)]["mae"].mean() for c in configs]
            ax.bar([i + j * width for i in range(len(configs))], values, width=width, color=MODEL_COLORS[model], label=model)
        naive = block[block["model"] == "persistence"]["mae"]
        if not naive.empty:
            ax.axhline(float(naive.iloc[0]), color=MODEL_COLORS["persistence"], ls="--", lw=1, label="persistence")
        ax.set_xticks([i + width * (len(models) - 1) / 2 for i in range(len(configs))])
        ax.set_xticklabels(configs)
        ax.set_title(f"{horizon / 60:g} min")
        ax.set_xlabel("configuração de ablação")
    axes[0][0].set_ylabel("MAE no teste final")
    axes[0][0].legend(fontsize=8)
    _finish(fig, path)
