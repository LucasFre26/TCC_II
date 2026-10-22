"""Protocolo de avaliação da V2: folds de desenvolvimento, seleção e teste final único.

Para cada horizonte, modelo ajustado e configuração de ablação:
  1. cada ponto da grade é ajustado em cada fold e avaliado só na validação do fold;
  2. o ponto de menor MAE médio de validação vence (empate: o primeiro da grade);
  3. o modelo escolhido é reajustado na região de desenvolvimento inteira e avaliado uma vez no teste final.
A persistência é avaliada uma vez por horizonte, nas mesmas origens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from energy_prediction.evaluation.metrics import mae, mase, rmse, skill_score
from energy_prediction.features.cpu_features import target_column
from energy_prediction.models.registry import (
    FITTED_MODELS,
    make_model,
    param_grid,
    validate_model_names,
)
from energy_prediction.validation.rolling_origin import Splits

METRIC_COLUMNS = ("mae", "rmse", "mase", "skill_score")
PERSISTENCE_CONFIG = "-"


@dataclass
class ProtocolResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selections: list[dict]
    fit_log: list[dict] = field(default_factory=list)


def score(y_true: np.ndarray, y_pred: np.ndarray, y_reference: np.ndarray) -> dict:
    """MAE, RMSE, MASE e skill contra a persistência das mesmas origens. Referência zero => indefinido."""
    model_mae, model_rmse = mae(y_true, y_pred), rmse(y_true, y_pred)
    reference_mae, reference_rmse = mae(y_true, y_reference), rmse(y_true, y_reference)
    try:
        relative_mae = mase(model_mae, reference_mae)
    except ZeroDivisionError:
        relative_mae = float("nan")
    try:
        skill = skill_score(model_rmse, reference_rmse)
    except ZeroDivisionError:
        skill = float("nan")
    return {"n": int(len(y_true)), "mae": model_mae, "rmse": model_rmse, "mase": relative_mae, "skill_score": skill}


def _aggregate_rows(rows: list[dict], keys: dict) -> list[dict]:
    frame = pd.DataFrame(rows)
    out = []
    for label, fn in (("aggregate_mean", np.nanmean), ("aggregate_std", np.nanstd)):
        row = {**keys, "split": label, "n": int(frame["n"].sum())}
        for metric in METRIC_COLUMNS:
            values = frame[metric].to_numpy(dtype=float)
            row[metric] = float(fn(values)) if not np.all(np.isnan(values)) else float("nan")
        out.append(row)
    return out


def _prediction_rows(table, positions, split, model, config, steps, y_pred, step_label):
    part = table.iloc[positions]
    return pd.DataFrame(
        {
            "split": split,
            "model": model,
            "config": config,
            "horizon_steps": steps,
            "origin_index": part["origin_index"].to_numpy(),
            "timestamp": part["timestamp"].to_numpy(),
            "y_true": part[step_label].to_numpy(dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )


def run_protocol(
    table: pd.DataFrame,
    columns_by_config: dict[str, list[str]],
    horizon_step_list: list[int],
    splits: Splits,
    models_cfg: dict,
    *,
    seed: int,
    step_seconds: float,
) -> ProtocolResult:
    names = validate_model_names(list(models_cfg["names"]))
    fitted = [name for name in names if name in FITTED_MODELS]
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    selections: list[dict] = []
    fit_log: list[dict] = []
    reference_columns = columns_by_config[next(iter(columns_by_config))]

    for steps in horizon_step_list:
        label = target_column(steps)
        y = table[label].to_numpy(dtype=float)
        horizon_meta = {"horizon_steps": int(steps), "horizon_seconds": float(steps * step_seconds)}

        # Persistência: sem ajuste, uma vez por horizonte, nas mesmas origens.
        persistence = make_model("persistence", {}, feature_names=reference_columns, seed=seed)
        x_ref = table[reference_columns].to_numpy(dtype=float)
        ref_pred = persistence.predict(x_ref)
        if "persistence" in names:
            base = {"model": "persistence", "config": "-", **horizon_meta, "best_params": "{}"}
            fold_rows = []
            for fold in splits.folds:
                row = {**base, "split": f"fold_{fold.index}", **score(y[fold.validation], ref_pred[fold.validation], ref_pred[fold.validation])}
                fold_rows.append(row)
                prediction_frames.append(_prediction_rows(table, fold.validation, f"fold_{fold.index}", "persistence", PERSISTENCE_CONFIG, steps, ref_pred[fold.validation], label))
            metric_rows.extend(fold_rows)
            metric_rows.extend(_aggregate_rows(fold_rows, base))
            metric_rows.append({**base, "split": "test_final", **score(y[splits.test], ref_pred[splits.test], ref_pred[splits.test])})
            prediction_frames.append(_prediction_rows(table, splits.test, "test_final", "persistence", PERSISTENCE_CONFIG, steps, ref_pred[splits.test], label))

        for model_name in fitted:
            model_cfg = models_cfg[model_name]
            grid = param_grid(model_name, model_cfg)
            for config, columns in columns_by_config.items():
                x = table[columns].to_numpy(dtype=float)
                cache: dict[int, dict[int, np.ndarray]] = {}
                mean_mae: list[float] = []
                for g, params in enumerate(grid):
                    cache[g] = {}
                    fold_mae = []
                    for fold in splits.folds:
                        model = make_model(model_name, params, feature_names=columns, seed=seed, model_cfg=model_cfg)
                        model.fit(x[fold.train], y[fold.train])
                        fit_log.append({"horizon_steps": steps, "model": model_name, "config": config, "phase": f"fold_{fold.index}", "params": g})
                        pred = model.predict(x[fold.validation])
                        cache[g][fold.index] = pred
                        fold_mae.append(mae(y[fold.validation], pred))
                    mean_mae.append(float(np.mean(fold_mae)))
                best = 0
                for g in range(1, len(grid)):
                    if mean_mae[g] < mean_mae[best] - 1e-12:
                        best = g
                selections.append(
                    {
                        "model": model_name,
                        "config": config,
                        **horizon_meta,
                        "best_params": grid[best],
                        "validation_mae_mean": mean_mae[best],
                        "grid_validation_mae": mean_mae,
                    }
                )
                keys = {"model": model_name, "config": config, **horizon_meta, "best_params": json.dumps(grid[best], sort_keys=True)}
                fold_rows = []
                for fold in splits.folds:
                    pred = cache[best][fold.index]
                    fold_rows.append({**keys, "split": f"fold_{fold.index}", **score(y[fold.validation], pred, ref_pred[fold.validation])})
                    prediction_frames.append(_prediction_rows(table, fold.validation, f"fold_{fold.index}", model_name, config, steps, pred, label))
                metric_rows.extend(fold_rows)
                metric_rows.extend(_aggregate_rows(fold_rows, keys))
                final = make_model(model_name, grid[best], feature_names=columns, seed=seed, model_cfg=model_cfg)
                final.fit(x[splits.development], y[splits.development])
                fit_log.append({"horizon_steps": steps, "model": model_name, "config": config, "phase": "final", "params": best})
                test_pred = final.predict(x[splits.test])
                metric_rows.append({**keys, "split": "test_final", **score(y[splits.test], test_pred, ref_pred[splits.test])})
                prediction_frames.append(_prediction_rows(table, splits.test, "test_final", model_name, config, steps, test_pred, label))

    metrics = pd.DataFrame(metric_rows)
    ordered = ["model", "config", "horizon_steps", "horizon_seconds", "split", "n", *METRIC_COLUMNS, "best_params"]
    return ProtocolResult(
        metrics=metrics[ordered],
        predictions=pd.concat(prediction_frames, ignore_index=True),
        selections=selections,
        fit_log=fit_log,
    )
