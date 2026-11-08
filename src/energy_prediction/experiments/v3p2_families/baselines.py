"""Família de referências congeladas reajustadas no protocolo por dia.

Persistência, XGBoost com os atributos e a grade da V2 (configuração A), XGBoost no incremento,
Kalman de nível local (estado filtrado) e híbrido Kalman + XGBoost, com as especificações da V3.
Grava as previsões da validação interna (usadas por combinadores e calibração) e do bloco, e as
trajetórias de 1 a 6 passos de persistência, XGBoost e híbrido (usadas pela energia).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.experiments.v3p2_families.common import BLOCK, INNER, block_frame, level_shift_by_block, metrics_table, write_tables
from energy_prediction.experiments.v3p2_families.tabular import (
    hybrid_fit_predict,
    path_predictions,
    run_blocks,
    statespace_fit_predict,
    xgb_fit_predict,
    xgb_grid,
)
from energy_prediction.features.cpu_features import PERSISTENCE_INPUT
from energy_prediction.models.statespace import StateSpaceForecaster

FAMILY = "baselines"
VARIANT = "A"
MODELS = ("persistence", "xgboost_v2", "xgboost_delta", "statespace", "hybrid")


def forecaster_for(ctx) -> StateSpaceForecaster:
    fam = ctx.cfg["families"]["baselines"]
    return StateSpaceForecaster(ctx.series["cpu"].to_numpy(), horizons=ctx.path_steps, maxiter=int(fam["statespace"]["maxiter"]))


def persistence_frames(ctx, steps: int) -> list[pd.DataFrame]:
    cpu = ctx.table[PERSISTENCE_INPUT].to_numpy(dtype=float)
    frames = []
    for block in ctx.blocks:
        frames.append(block_frame(ctx.table, block, block.inner.validation, phase=INNER, model="persistence", variant="-", steps=steps, y_pred=cpu[block.inner.validation]))
        frames.append(block_frame(ctx.table, block, block.evaluation, phase=BLOCK, model="persistence", variant="-", steps=steps, y_pred=cpu[block.evaluation]))
    return frames


def run(ctx, out: Path, **_runtime) -> dict:
    fam = ctx.cfg["families"]["baselines"]
    xgb_cfg = fam["xgboost"]
    grid = xgb_grid(xgb_cfg)
    spec = str(fam["statespace"]["specification"])
    columns = ctx.columns["A"]
    forecaster = forecaster_for(ctx)
    frames, selections = [], []
    for steps in ctx.steps:
        frames.extend(persistence_frames(ctx, steps))
        for model, fp in (
            ("xgboost_v2", xgb_fit_predict(ctx.table, columns, steps, grid, xgb_cfg, ctx.seed)),
            ("xgboost_delta", xgb_fit_predict(ctx.table, columns, steps, grid, xgb_cfg, ctx.seed, delta=True)),
        ):
            f, s = run_blocks(ctx, model=model, variant=VARIANT, steps=steps, fit_predict=fp, grid=grid)
            frames.extend(f)
            selections.extend(s)
        f, s = run_blocks(ctx, model="statespace", variant=VARIANT, steps=steps, fit_predict=statespace_fit_predict(ctx.table, forecaster, spec, steps), grid=[{"specification": spec}])
        frames.extend(f)
        selections.extend(s)
        f, s = run_blocks(ctx, model="hybrid", variant=VARIANT, steps=steps, fit_predict=hybrid_fit_predict(ctx.table, columns, steps, grid, xgb_cfg, ctx.seed, forecaster, spec), grid=grid)
        frames.extend(f)
        selections.extend(s)
    predictions = pd.concat(frames, ignore_index=True)
    shift = level_shift_by_block(ctx)
    predictions["level_shift"] = predictions["block"].map(shift)

    paths = [
        path_predictions(ctx, model="persistence", variant="-", selections=[{"model": "persistence", "variant": "-", "horizon_steps": h, "block": b.label, "best_params": {}} for h in ctx.steps for b in ctx.blocks], factory=lambda step, p: (lambda g, train, predict, **_: (ctx.table[PERSISTENCE_INPUT].to_numpy(dtype=float)[predict], {}))),
        path_predictions(ctx, model="xgboost_v2", variant=VARIANT, selections=selections, factory=lambda step, p: xgb_fit_predict(ctx.table, columns, step, [p], xgb_cfg, ctx.seed)),
        path_predictions(ctx, model="hybrid", variant=VARIANT, selections=selections, factory=lambda step, p: hybrid_fit_predict(ctx.table, columns, step, [p], xgb_cfg, ctx.seed, forecaster, spec)),
    ]
    paths = pd.concat(paths, ignore_index=True)
    metrics = metrics_table(predictions)
    write_tables(
        out,
        predictions=predictions,
        paths=paths,
        metrics=metrics,
        selections=selections,
        family_info={
            "models": list(MODELS),
            "variant": VARIANT,
            "statespace_specification": spec,
            "selection_rule": "MAE da validação interna de cada bloco",
            "level_shift_by_block": shift,
            "statement": "Referências reajustadas no protocolo por dia com as especificações da V2 e da V3.",
        },
    )
    dev = metrics[(metrics["block"] == "development_pooled")]
    return {"development_mae": {f"{r.model}/h{r.horizon_steps}": float(r.mae) for r in dev.itertuples()}, "n_predictions": int(len(predictions))}
