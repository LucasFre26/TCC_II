"""Linhas de métricas e de previsões no formato da V2, comuns às famílias da V3."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from energy_prediction.evaluation.metrics import mae
from energy_prediction.evaluation.nested import GridResult
from energy_prediction.evaluation.protocol import METRIC_COLUMNS, score
from energy_prediction.features.cpu_features import PERSISTENCE_INPUT, target_column
from energy_prediction.validation.rolling_origin import Splits

PERSISTENCE = "persistence"
NO_CONFIG = "-"
METRIC_ORDER = ["model", "config", "mode", "horizon_steps", "horizon_seconds", "split", "n", *METRIC_COLUMNS, "best_params"]


def predictions_frame(table: pd.DataFrame, positions, *, split: str, model: str, config: str, steps: int, y_pred, mode: str = "direct", **extra) -> pd.DataFrame:
    part = table.iloc[np.asarray(positions, dtype=int)]
    frame = pd.DataFrame(
        {
            "split": split,
            "model": model,
            "config": config,
            "mode": mode,
            "horizon_steps": int(steps),
            "origin_index": part["origin_index"].to_numpy(),
            "timestamp": part["timestamp"].to_numpy(),
            "y_true": part[target_column(steps)].to_numpy(dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    for key, value in extra.items():
        frame[key] = value
    return frame


def _aggregate(rows: list[dict], keys: dict) -> list[dict]:
    frame = pd.DataFrame(rows)
    out = []
    for label, fn in (("aggregate_mean", np.nanmean), ("aggregate_std", np.nanstd)):
        row = {**keys, "split": label, "n": int(frame["n"].sum())}
        for metric in METRIC_COLUMNS:
            values = frame[metric].to_numpy(dtype=float)
            row[metric] = float(fn(values)) if not np.all(np.isnan(values)) else float("nan")
        out.append(row)
    return out


def persistence_prediction(table: pd.DataFrame) -> np.ndarray:
    return table[PERSISTENCE_INPUT].to_numpy(dtype=float)


def scored_rows(
    table: pd.DataFrame,
    splits: Splits,
    *,
    steps: int,
    step_seconds: float,
    keys: dict,
    outer: dict[int, np.ndarray],
    test: np.ndarray,
    inner_score: float | None = None,
    inner_metric: str = "mae",
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Linhas por fold (validação externa), médias, teste final e a métrica interna da seleção."""
    y = table[target_column(steps)].to_numpy(dtype=float)
    reference = persistence_prediction(table)
    base = {**keys, "horizon_steps": int(steps), "horizon_seconds": float(steps * step_seconds)}
    rows, frames = [], []
    fold_rows = []
    for fold in splits.folds:
        pred = outer[fold.index]
        fold_rows.append({**base, "split": f"fold_{fold.index}", **score(y[fold.validation], pred, reference[fold.validation])})
        frames.append(predictions_frame(table, fold.validation, split=f"fold_{fold.index}", model=keys["model"], config=keys["config"], steps=steps, y_pred=pred, mode=keys.get("mode", "direct")))
    rows.extend(fold_rows)
    rows.extend(_aggregate(fold_rows, base))
    rows.append({**base, "split": "test_final", **score(y[splits.test], test, reference[splits.test])})
    frames.append(predictions_frame(table, splits.test, split="test_final", model=keys["model"], config=keys["config"], steps=steps, y_pred=test, mode=keys.get("mode", "direct")))
    if inner_score is not None:
        rows.append({**base, "split": f"inner_selection_{inner_metric}", "n": 0, "mae": inner_score if inner_metric == "mae" else float("nan"), "rmse": float("nan"), "mase": float("nan"), "skill_score": float("nan")})
    return rows, frames


def persistence_rows(table: pd.DataFrame, splits: Splits, *, steps: int, step_seconds: float) -> tuple[list[dict], list[pd.DataFrame]]:
    reference = persistence_prediction(table)
    outer = {fold.index: reference[fold.validation] for fold in splits.folds}
    keys = {"model": PERSISTENCE, "config": NO_CONFIG, "mode": "direct", "best_params": "{}"}
    return scored_rows(table, splits, steps=steps, step_seconds=step_seconds, keys=keys, outer=outer, test=reference[splits.test])


def persistence_inner_mae(table: pd.DataFrame, inner: dict, steps: int) -> float:
    """MAE médio da persistência nas validações internas (mesmas origens da seleção)."""
    y = table[target_column(steps)].to_numpy(dtype=float)
    reference = persistence_prediction(table)
    return float(np.mean([mae(y[s.validation], reference[s.validation]) for s in inner.values()]))


def selection_record(result: GridResult, grid: list[dict], **keys) -> dict:
    return {
        **keys,
        "rule": result.rule,
        "best_index": int(result.best),
        "best_params": grid[result.best],
        "selection_score": float(result.grid_scores[result.best]),
        "grid_scores": [float(v) for v in result.grid_scores],
        "fold_scores": [[float(v) for v in scores] for scores in result.fold_scores],
        "best_epochs_by_fold": {str(k): v for k, v in result.best_epochs.items()},
        "final_epochs": result.final_epochs,
    }


def ordered_metrics(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in METRIC_ORDER:
        if column not in frame:
            frame[column] = np.nan
    extra = [c for c in frame.columns if c not in METRIC_ORDER]
    return frame[METRIC_ORDER + extra]


def params_text(params: dict) -> str:
    return json.dumps(params, sort_keys=True)
