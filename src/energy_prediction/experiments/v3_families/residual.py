"""Família residual: XGBoost no incremento, espaço de estados filtrado e híbrido estados + XGBoost.

Mesmas origens, horizontes e folds da V2; seleção pela MAE interna. O modelo de estados é
univariado (configuração A); incremento e híbrido percorrem A, B e C. O contraste D contra A do
XGBoost de incremento usa as origens em que D está completa, com A recalculado nelas. Nenhum
modelo é descartado por perder para o XGBoost da V2 no teste.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.evaluation.comparisons import bootstrap_blocks, compare_losses, load_v2_reference, loss_series
from energy_prediction.evaluation.error_analysis import error_extremes_table
from energy_prediction.evaluation.metrics import mae
from energy_prediction.evaluation.nested import AlignmentError, evaluate_grid
from energy_prediction.experiments.v3_common import (
    NO_CONFIG,
    PERSISTENCE,
    ordered_metrics,
    params_text,
    persistence_rows,
    scored_rows,
    selection_record,
)
from energy_prediction.experiments.v3_context import V3Context
from energy_prediction.features.cpu_features import target_column
from energy_prediction.models.delta import DeltaXGBoost
from energy_prediction.models.hybrid import HybridCorrector
from energy_prediction.models.registry import param_grid
from energy_prediction.models.statespace import StateSpaceForecaster
from energy_prediction.models.v3_registry import validate_family_models
from energy_prediction.runtime.output import dump_json

FAMILY = "residual"


def delta_fit_predict(table: pd.DataFrame, columns: list[str], steps: int, grid: list[dict], xgb_cfg: dict, seed: int):
    X = table[columns].to_numpy(dtype=float)
    y = table[target_column(steps)].to_numpy(dtype=float)

    def fit_predict(g, train, predict, **_):
        model = DeltaXGBoost(grid[g], feature_names=columns, seed=seed, n_jobs=int(xgb_cfg["n_jobs"]), fixed=xgb_cfg.get("fixed"))
        model.fit(X[train], y[train])
        return model.predict(X[predict]), {}

    return fit_predict


def statespace_fit_predict(table: pd.DataFrame, forecaster: StateSpaceForecaster, specifications: list[str], steps: int):
    origins = table["origin_index"].to_numpy()

    def fit_predict(g, train, predict, **_):
        segment_end = int(origins[train].max())
        forecasts = forecaster.forecasts(specifications[g], segment_end)
        return forecasts.by_horizon[steps][origins[predict]], {"segment_end": segment_end}

    return fit_predict


def hybrid_fit_predict(table, columns, steps, grid, xgb_cfg, seed, forecaster: StateSpaceForecaster, specification: str):
    X = table[columns].to_numpy(dtype=float)
    y = table[target_column(steps)].to_numpy(dtype=float)
    origins = table["origin_index"].to_numpy()

    def fit_predict(g, train, predict, **_):
        segment_end = int(origins[train].max())
        forecasts = forecaster.forecasts(specification, segment_end)
        base = forecasts.by_horizon[steps][origins]
        corrector = HybridCorrector(grid[g], seed=seed, n_jobs=int(xgb_cfg["n_jobs"]), fixed=xgb_cfg.get("fixed"))
        corrector.fit(X[train], y[train], base[train], forecasts)
        return corrector.predict(X[predict], base[predict]), {"segment_end": segment_end, "specification": specification}

    return fit_predict


def _evaluate(name, config, steps, fit_predict, grid, table, splits, inner, ctx, rows, frames, selections, grid_labels=None):
    y = table[target_column(steps)].to_numpy(dtype=float)
    result = evaluate_grid(len(grid), fit_predict, y, splits, inner, metric=mae)
    keys = {"model": name, "config": config, "mode": "direct", "best_params": params_text(grid[result.best])}
    new_rows, new_frames = scored_rows(
        table, splits, steps=steps, step_seconds=ctx.step_seconds, keys=keys, outer=result.outer_predictions,
        test=result.test_predictions, inner_score=result.grid_scores[result.best] if inner is not None else None,
    )
    rows.extend(new_rows)
    frames.extend(new_frames)
    selections.append(selection_record(result, grid_labels or grid, model=name, config=config, horizon_steps=int(steps)))
    return result


def _pairs(predictions: pd.DataFrame, d_predictions: pd.DataFrame | None, ctx: V3Context, models: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    cfg = ctx.cfg
    pairs = cfg["comparisons"]["pairs"]
    blocks, primary, n_boot = bootstrap_blocks(cfg)
    seed = ctx.seed
    v2 = load_v2_reference(ctx.v2_dir)
    skipped = []
    rows = []
    point = [m for m in ("xgboost_delta", "statespace", "hybrid") if m in models]
    for steps in ctx.steps:
        def add(label, a, b, la=None, lb=None):
            la = la if la is not None else loss_series(predictions, model=a[0], config=a[1], horizon_steps=steps)
            lb = lb if lb is not None else loss_series(predictions, model=b[0], config=b[1], horizon_steps=steps)
            meta = {"comparison": label, "horizon_steps": int(steps), "model_a": a[0], "config_a": a[1], "model_b": b[0], "config_b": b[1]}
            rows.extend(compare_losses(la, lb, blocks=blocks, primary_block=primary, n_boot=n_boot, seed=seed, meta=meta))

        if "new_vs_persistence_A" in pairs:
            for model in point:
                add("new_vs_persistence_A", (model, "A"), (PERSISTENCE, NO_CONFIG))
        if "new_vs_v2_xgboost_A" in pairs:
            if steps not in v2:
                skipped.append({"item": f"new_vs_v2_xgboost_A h{steps}", "reason": "previsões da V2 não configuradas"})
            else:
                reference = v2[steps]
                lb = pd.Series((reference["y_true"] - reference["y_pred"]).abs().to_numpy(), index=reference["origin_index"].to_numpy()).sort_index()
                for model in point:
                    la = loss_series(predictions, model=model, config="A", horizon_steps=steps)
                    if not la.index.equals(lb.index):
                        raise AlignmentError(f"Origens do teste da V3 ({model}, h{steps}) diferem das da V2 preservada.")
                    add("new_vs_v2_xgboost_A", (model, "A"), ("xgboost_v2", "A"), la=la, lb=lb)
        if "ablation_vs_A" in pairs:
            for model in ("xgboost_delta", "hybrid"):
                if model in models:
                    for config in ctx.cfg["families"]["residual"]["configs"]:
                        if config != "A":
                            add("ablation_vs_A", (model, config), (model, "A"))
        if "hybrid_vs_statespace" in pairs and {"hybrid", "statespace"} <= set(models):
            add("hybrid_vs_statespace", ("hybrid", "A"), ("statespace", "A"))
        if "d_vs_a" in pairs and d_predictions is not None:
            la = loss_series(d_predictions, model="xgboost_delta", config="D", horizon_steps=steps)
            lb = loss_series(d_predictions, model="xgboost_delta", config="A", horizon_steps=steps)
            add("d_vs_a", ("xgboost_delta", "D"), ("xgboost_delta", "A"), la=la, lb=lb)
    return pd.DataFrame(rows), skipped


def run(ctx: V3Context, out: Path, **_runtime) -> dict:
    fam = ctx.cfg["families"]["residual"]
    models = validate_family_models(FAMILY, list(fam["models"]))
    configs = list(fam["configs"])
    xgb_cfg = fam["xgboost"]
    grid = param_grid("xgboost", xgb_cfg)
    specifications = list(fam["statespace"]["specifications"])
    spec_grid = [{"specification": s} for s in specifications]
    table, splits, inner = ctx.table, ctx.splits, ctx.inner
    forecaster = StateSpaceForecaster(ctx.series["cpu"].to_numpy(), horizons=ctx.path_steps, maxiter=int(fam["statespace"]["maxiter"]))

    rows, frames, selections = [], [], []
    chosen_spec: dict[int, str] = {}
    for steps in ctx.steps:
        if PERSISTENCE in models:
            new_rows, new_frames = persistence_rows(table, splits, steps=steps, step_seconds=ctx.step_seconds)
            rows.extend(new_rows)
            frames.extend(new_frames)
        if "statespace" in models or "hybrid" in models:
            result = _evaluate(
                "statespace", "A", steps, statespace_fit_predict(table, forecaster, specifications, steps),
                spec_grid, table, splits, inner, ctx, rows, frames, selections,
            )
            chosen_spec[steps] = specifications[result.best]
        for config in configs:
            columns = ctx.columns[config]
            if "xgboost_delta" in models:
                _evaluate("xgboost_delta", config, steps, delta_fit_predict(table, columns, steps, grid, xgb_cfg, ctx.seed), grid, table, splits, inner, ctx, rows, frames, selections)
            if "hybrid" in models:
                fp = hybrid_fit_predict(table, columns, steps, grid, xgb_cfg, ctx.seed, forecaster, chosen_spec[steps])
                _evaluate("hybrid", config, steps, fp, grid, table, splits, inner, ctx, rows, frames, selections)
    if "statespace" not in models:
        rows = [r for r in rows if r["model"] != "statespace"]
        frames = [f for f in frames if f["model"].iloc[0] != "statespace"]

    predictions = pd.concat(frames, ignore_index=True)
    metrics = ordered_metrics(rows)

    d_predictions = None
    d_metrics = None
    excluded = []
    if "xgboost_delta" in ctx.cfg["comparisons"]["d_contrast_models"] and "xgboost_delta" in models:
        if ctx.d_table is None:
            excluded.append({"item": "contraste D do xgboost_delta", "reason": "atributos de distribuição desligados"})
        else:
            d_rows, d_frames, d_selections = [], [], []
            for steps in ctx.steps:
                for config in ("A", "D"):
                    fp = delta_fit_predict(ctx.d_table, ctx.columns[config], steps, grid, xgb_cfg, ctx.seed)
                    _evaluate("xgboost_delta", config, steps, fp, grid, ctx.d_table, ctx.d_splits, ctx.d_inner, ctx, d_rows, d_frames, d_selections)
            d_predictions = pd.concat(d_frames, ignore_index=True)
            d_metrics = ordered_metrics(d_rows)
            for record in d_selections:
                record["table"] = "origens com D completa"
            selections.extend({**r, "contrast": "d_vs_a"} for r in d_selections)

    intervals, skipped = _pairs(predictions, d_predictions, ctx, models)
    metrics.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    intervals.to_csv(out / "bootstrap_ci.csv", index=False)
    error_extremes_table(predictions, peak_quantile=float(ctx.cfg["comparisons"]["peak_quantile"])).to_csv(out / "error_extremes.csv", index=False)
    if d_metrics is not None:
        d_metrics.to_csv(out / "d_contrast_metrics.csv", index=False)
        d_predictions.to_csv(out / "d_contrast_predictions.csv", index=False)
    dump_json(out / "selections.json", selections)
    dump_json(
        out / "family_info.json",
        {
            "models": models,
            "configs": configs,
            "statespace_specification_by_horizon": {str(k): v for k, v in chosen_spec.items()},
            "selection_rule": "inner_validation" if ctx.nested else "outer_fold_validation_v2",
            "skipped_comparisons": skipped,
            "statement": "Perder para o XGBoost da V2 é resultado válido; nenhum modelo foi descartado pelo teste.",
            "limitations": [
                "Variâncias do modelo de estados estimadas por máxima verossimilhança no segmento de treino; "
                "o estado na origem é sempre filtrado (sem suavização).",
            ],
        },
    )
    test = metrics[(metrics["split"] == "test_final")]
    return {
        "n_metric_rows": int(len(metrics)),
        "test_mae": {f"{r.model}/{r.config}/h{r.horizon_steps}": float(r.mae) for r in test.itertuples()},
        "excluded": excluded + [{"item": s["item"], "reason": s["reason"]} for s in skipped],
    }
