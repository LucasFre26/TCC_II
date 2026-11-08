"""Ajuste por bloco de modelos tabulares (XGBoost, incremento, Kalman, híbrido) da V3 Parte 2.

Pontos de grade podem trazer chaves auxiliares iniciadas por "_" (ex.: `_half_life`, `_lambda`),
que não vão para o XGBoost: são entregues à função de peso junto com as posições de treino.
"""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd

from energy_prediction.evaluation.day_forward import evaluate_block
from energy_prediction.evaluation.metrics import mae
from energy_prediction.experiments.v3p2_families.common import BLOCK, INNER, block_frame
from energy_prediction.features.cpu_features import target_column
from energy_prediction.models.delta import DeltaXGBoost
from energy_prediction.models.hybrid import HybridCorrector
from energy_prediction.models.registry import XGBoostModel, param_grid
from energy_prediction.models.statespace import StateSpaceForecaster

WeightFn = Callable[[np.ndarray, dict], np.ndarray]


def split_params(point: dict) -> tuple[dict, dict]:
    model = {k: v for k, v in point.items() if not str(k).startswith("_")}
    extra = {k: v for k, v in point.items() if str(k).startswith("_")}
    return model, extra


def expand_grid(base: list[dict], **extra_axes) -> list[dict]:
    """Produto da grade do XGBoost por eixos auxiliares (chaves com "_")."""
    if not extra_axes:
        return list(base)
    keys = sorted(extra_axes)
    out = []
    for point in base:
        for values in itertools.product(*(extra_axes[k] for k in keys)):
            out.append({**point, **{f"_{k}": v for k, v in zip(keys, values)}})
    return out


def xgb_grid(xgb_cfg: dict) -> list[dict]:
    return param_grid("xgboost", xgb_cfg)


def xgb_fit_predict(table: pd.DataFrame, columns: list[str], steps: int, grid: list[dict], xgb_cfg: dict, seed: int, *, delta: bool = False, weight_fn: WeightFn | None = None):
    X = table[columns].to_numpy(dtype=float)
    y = table[target_column(steps)].to_numpy(dtype=float)

    def fp(g, train, predict, **_):
        params, extra = split_params(grid[g])
        weights = None if weight_fn is None else weight_fn(train, extra)
        if delta:
            model = DeltaXGBoost(params, feature_names=columns, seed=seed, n_jobs=int(xgb_cfg["n_jobs"]), fixed=xgb_cfg.get("fixed"))
        else:
            model = XGBoostModel(params, seed=seed, n_jobs=int(xgb_cfg["n_jobs"]), fixed=xgb_cfg.get("fixed"))
        model.fit(X[train], y[train], sample_weight=weights)
        return model.predict(X[predict]), {}

    return fp


def statespace_fit_predict(table: pd.DataFrame, forecaster: StateSpaceForecaster, specification: str, steps: int):
    origins = table["origin_index"].to_numpy()

    def fp(g, train, predict, **_):
        segment_end = int(origins[train].max())
        return forecaster.forecasts(specification, segment_end).by_horizon[steps][origins[predict]], {"segment_end": segment_end}

    return fp


def hybrid_fit_predict(table: pd.DataFrame, columns: list[str], steps: int, grid: list[dict], xgb_cfg: dict, seed: int, forecaster: StateSpaceForecaster, specification: str):
    X = table[columns].to_numpy(dtype=float)
    y = table[target_column(steps)].to_numpy(dtype=float)
    origins = table["origin_index"].to_numpy()

    def fp(g, train, predict, **_):
        segment_end = int(origins[train].max())
        forecasts = forecaster.forecasts(specification, segment_end)
        base = forecasts.by_horizon[steps][origins]
        corrector = HybridCorrector(grid[g], seed=seed, n_jobs=int(xgb_cfg["n_jobs"]), fixed=xgb_cfg.get("fixed"))
        corrector.fit(X[train], y[train], base[train], forecasts)
        return corrector.predict(X[predict], base[predict]), {"segment_end": segment_end}

    return fp


def run_blocks(ctx, *, model: str, variant: str, steps: int, fit_predict, grid: list[dict], table: pd.DataFrame | None = None, blocks=None, extra: dict | None = None) -> tuple[list[pd.DataFrame], list[dict]]:
    """Seleção interna, reajuste e previsão em cada bloco; devolve previsões (inner + block) e seleções."""
    table = ctx.table if table is None else table
    y = table[target_column(steps)].to_numpy(dtype=float)
    frames, selections = [], []
    for block in ctx.blocks if blocks is None else blocks:
        result = evaluate_block(len(grid), fit_predict, y, block, metric=mae)
        keys = {"model": model, "variant": variant, "steps": steps, **(extra or {})}
        frames.append(block_frame(table, block, block.inner.validation, phase=INNER, model=model, variant=variant, steps=steps, y_pred=result.inner_prediction, **(extra or {})))
        frames.append(block_frame(table, block, block.evaluation, phase=BLOCK, model=model, variant=variant, steps=steps, y_pred=result.block_prediction, **(extra or {})))
        selections.append(
            {
                "model": model,
                "variant": variant,
                "horizon_steps": int(steps),
                "block": block.label,
                "rule": "inner_validation_mae",
                "best_index": int(result.best),
                "best_params": grid[result.best],
                "grid_scores": result.grid_scores,
                "best_epoch": result.best_epoch,
                **{k: v for k, v in keys.items() if k not in ("model", "variant", "steps")},
            }
        )
    return frames, selections


def inherited_step(step: int, scored: list[int]) -> int:
    candidates = [h for h in sorted(scored) if h >= step]
    if not candidates:
        raise ValueError(f"Passo {step} além do maior horizonte pontuado.")
    return int(candidates[0])


def best_params(selections: list[dict], *, model: str, variant: str, steps: int, block: str) -> dict:
    for record in selections:
        if record["model"] == model and record["variant"] == variant and int(record["horizon_steps"]) == int(steps) and record["block"] == block:
            return record["best_params"]
    raise KeyError(f"Sem seleção de {model}/{variant}/h{steps}/{block}.")


def path_predictions(ctx, *, model: str, variant: str, selections: list[dict], factory, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trajetória de 1 a 6 passos por bloco; passos não pontuados herdam o hiperparâmetro do horizonte que os cobre.

    `factory(step, params)` devolve um fit_predict de um único ponto de grade (índice 0).
    """
    table = ctx.table if table is None else table
    origins = table["origin_index"].to_numpy()
    rows = []
    for block in ctx.blocks:
        for step in ctx.path_steps:
            params = best_params(selections, model=model, variant=variant, steps=inherited_step(step, ctx.steps), block=block.label)
            pred, _ = factory(step, params)(0, block.train, block.evaluation)
            y = table[target_column(step)].to_numpy(dtype=float)[block.evaluation]
            rows.append(pd.DataFrame({"model": model, "variant": variant, "block": block.label, "day": block.day, "role": block.role, "step": int(step), "origin_index": origins[block.evaluation], "y_true": y, "y_pred": np.asarray(pred, dtype=float)}))
    return pd.concat(rows, ignore_index=True)
