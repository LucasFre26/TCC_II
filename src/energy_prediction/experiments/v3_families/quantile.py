"""Família quantílica: XGBoost para Q50, Q90 e Q95, pinball, picos e intervalo conformal.

Seleção pela pinball interna de cada quantil. O Q50 não entra na tabela pontual principal (fica
só aqui). O intervalo reportado tem piso Q50 e teto Q95 ajustado pelos escores da validação
externa; o teste não entra na calibração. Cobertura é empírica.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.evaluation.comparisons import bootstrap_blocks, compare_losses, loss_series
from energy_prediction.evaluation.conformal import adjusted_upper, conformal_offset
from energy_prediction.evaluation.nested import evaluate_grid
from energy_prediction.evaluation.probabilistic import (
    interval_metrics,
    pinball_loss,
    quantile_mean_metrics,
    quantile_peak_metrics,
)
from energy_prediction.experiments.v3_common import params_text, predictions_frame, selection_record
from energy_prediction.experiments.v3_context import V3Context
from energy_prediction.features.cpu_features import target_column
from energy_prediction.models.quantile_xgb import QuantileXGBoost
from energy_prediction.models.registry import param_grid
from energy_prediction.runtime.output import dump_json

FAMILY = "quantile"
MODEL = "xgboost_quantile"
STATEMENT = (
    "Cobertura e largura dos intervalos são empíricas no teste; sob dependência temporal não há "
    "garantia teórica de cobertura."
)


def _fit_predict(table, columns, steps, grid, xgb_cfg, seed, quantile):
    X = table[columns].to_numpy(dtype=float)
    y = table[target_column(steps)].to_numpy(dtype=float)

    def fit_predict(g, train, predict, **_):
        model = QuantileXGBoost(quantile, grid[g], seed=seed, n_jobs=int(xgb_cfg["n_jobs"]), fixed=xgb_cfg.get("fixed"))
        model.fit(X[train], y[train])
        return model.predict(X[predict]), {}

    return fit_predict


def calibration_scores(table: pd.DataFrame, splits, outer: dict[int, np.ndarray], steps: int) -> np.ndarray:
    """y - q̂ na validação externa dos folds, em ordem cronológica de origem."""
    y = table[target_column(steps)].to_numpy(dtype=float)
    positions = np.concatenate([fold.validation for fold in splits.folds])
    predicted = np.concatenate([outer[fold.index] for fold in splits.folds])
    order = np.argsort(table["origin_index"].to_numpy()[positions], kind="mergesort")
    return (y[positions] - predicted)[order]


def run(ctx: V3Context, out: Path, **_runtime) -> dict:
    fam = ctx.cfg["families"]["quantile"]
    quantiles = [float(q) for q in fam["quantiles"]]
    conformal = fam["conformal"]
    upper_q, aux_q = float(conformal["upper_quantile"]), float(conformal["auxiliary_quantile"])
    if 0.5 not in quantiles or upper_q not in quantiles or aux_q not in quantiles:
        raise ValueError("A família quantílica precisa de 0,5, do quantil superior e do auxiliar entre os quantis.")
    xgb_cfg = fam["xgboost"]
    grid = param_grid("xgboost", xgb_cfg)
    table, splits, inner = ctx.table, ctx.splits, ctx.inner
    peak_q = float(ctx.cfg["comparisons"]["peak_quantile"])

    mean_rows, peak_rows, interval_rows, calibrations, selections, frames = [], [], [], [], [], []
    for steps in ctx.steps:
        y = table[target_column(steps)].to_numpy(dtype=float)
        for config in fam["configs"]:
            columns = ctx.columns[config]
            test_pred: dict[float, np.ndarray] = {}
            outer_pred: dict[float, dict[int, np.ndarray]] = {}
            for q in quantiles:
                metric = lambda truth, pred, q=q: pinball_loss(truth, pred, q)  # noqa: E731
                result = evaluate_grid(len(grid), _fit_predict(table, columns, steps, grid, xgb_cfg, ctx.seed, q), y, splits, inner, metric=metric)
                selections.append(selection_record(result, grid, model=MODEL, config=config, horizon_steps=int(steps), quantile=q, selection_metric="pinball"))
                test_pred[q], outer_pred[q] = result.test_predictions, result.outer_predictions
                keys = {"model": MODEL, "config": config, "horizon_steps": int(steps), "quantile": q, "best_params": params_text(grid[result.best])}
                for fold in splits.folds:
                    mean_rows.append({**keys, "split": f"fold_{fold.index}", **quantile_mean_metrics(y[fold.validation], result.outer_predictions[fold.index], q)})
                    frames.append(predictions_frame(table, fold.validation, split=f"fold_{fold.index}", model=MODEL, config=config, steps=steps, y_pred=result.outer_predictions[fold.index], mode="quantile", quantile=q))
                mean_rows.append({**keys, "split": "test_final", **quantile_mean_metrics(y[splits.test], test_pred[q], q)})
                if inner is not None:
                    mean_rows.append({**keys, "split": "inner_selection_pinball", "n": 0, "pinball": result.grid_scores[result.best]})
                peak_rows.append({**keys, "split": "test_final", **quantile_peak_metrics(y[splits.test], test_pred[q], peak_q)})
                frames.append(predictions_frame(table, splits.test, split="test_final", model=MODEL, config=config, steps=steps, y_pred=test_pred[q], mode="quantile", quantile=q))

            truth = y[splits.test]
            base = {"model": MODEL, "config": config, "horizon_steps": int(steps), "split": "test_final", "lower": "Q50"}
            interval_rows.append({**base, "upper": f"Q{upper_q:g} cru", "conformal_applied": False, **interval_metrics(truth, test_pred[0.5], test_pred[upper_q])})
            for q, alpha, tag in ((upper_q, 1.0 - upper_q, "principal"), (aux_q, 1.0 - aux_q, "auxiliar")):
                calibration = conformal_offset(calibration_scores(table, splits, outer_pred[q], steps), alpha, min_residuals=int(conformal["min_residuals"]))
                calibrations.append({"config": config, "horizon_steps": int(steps), "quantile": q, "role": tag, **calibration})
                upper = adjusted_upper(test_pred[q], calibration)
                if upper is None:
                    interval_rows.append({**base, "upper": f"Q{q:g} ajustado", "conformal_applied": False, "note": "calibração não aplicada; cobertura teórica não publicada"})
                else:
                    interval_rows.append({**base, "upper": f"Q{q:g} ajustado", "conformal_applied": True, "offset": calibration["offset"], **interval_metrics(truth, test_pred[0.5], upper)})
                    frames.append(predictions_frame(table, splits.test, split="test_final", model=MODEL, config=config, steps=steps, y_pred=upper, mode="conformal_upper", quantile=q))

    predictions = pd.concat(frames, ignore_index=True)
    blocks, primary, n_boot = bootstrap_blocks(ctx.cfg)
    boot_rows = []
    if "ablation_vs_A" in ctx.cfg["comparisons"]["pairs"]:
        quantile_part = predictions[predictions["mode"] == "quantile"]
        for steps in ctx.steps:
            for q in quantiles:
                def pinball(frame, q=q):
                    error = frame["y_true"] - frame["y_pred"]
                    return np.maximum(q * error, (q - 1.0) * error)

                reference = loss_series(quantile_part, model=MODEL, config="A", horizon_steps=steps, loss=pinball, quantile=q)
                for config in fam["configs"]:
                    if config == "A":
                        continue
                    other = loss_series(quantile_part, model=MODEL, config=config, horizon_steps=steps, loss=pinball, quantile=q)
                    meta = {"comparison": "ablation_vs_A", "loss": "pinball", "quantile": q, "horizon_steps": int(steps), "model_a": MODEL, "config_a": config, "model_b": MODEL, "config_b": "A"}
                    boot_rows.extend(compare_losses(other, reference, blocks=blocks, primary_block=primary, n_boot=n_boot, seed=ctx.seed, meta=meta))

    pd.DataFrame(mean_rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(peak_rows).to_csv(out / "peak_metrics.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(out / "interval_metrics.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(out / "bootstrap_ci.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    dump_json(out / "selections.json", selections)
    dump_json(out / "calibration.json", {"statement": STATEMENT, "calibrations": calibrations})
    return {
        "quantiles": quantiles,
        "conformal_applied": sum(1 for c in calibrations if c["applied"]),
        "conformal_not_applied": sum(1 for c in calibrations if not c["applied"]),
        "excluded": [
            {"item": f"conformal {c['config']} h{c['horizon_steps']} Q{c['quantile']:g}", "reason": c["reason"]}
            for c in calibrations
            if not c["applied"]
        ],
    }
