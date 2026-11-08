"""Família de treino orientado a picos (H4): XGBoost da V2 com peso nas subidas bruscas.

Peso por exemplo: w = 1 + λ quando y(s+h) − y(s) ≥ σ_h (desvio dessas variações no treino do
bloco), λ ∈ grade pré-registrada; λ = 0 reproduz o XGBoost de referência. Para cada λ, a grade
do XGBoost é escolhida pelo MAE da validação interna. O λ do bloco e horizonte é o de menor
subestimação média nos picos da validação interna entre os que não pioram o MAE interno em mais
que `max_mae_increase` em relação a λ = 0. Picos: alvos acima do quantil `peak_quantile` da CPU
do treino do bloco. Eventos, antecipação e falso alarme seguem a análise expandida da V3.

Variantes não treinadas: perda assimétrica (equivale, no limite, à regressão quantílica já
avaliada); Huber (reduz a influência dos picos, direção oposta ao objetivo); multitarefa
(exigiria nova arquitetura para eventos que duram, em geral, um passo de 5 min).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.analysis.peaks import alert_scores, events
from energy_prediction.evaluation.day_forward import evaluate_block
from energy_prediction.evaluation.metrics import mae
from energy_prediction.experiments.v3p2_families.common import (
    BLOCK,
    INNER,
    block_frame,
    compare,
    finalize_comparisons,
    inherited_rows,
    loss_frame,
    metrics_table,
    point_metrics,
    read_family_csv,
    write_tables,
)
from energy_prediction.experiments.v3p2_families.tabular import xgb_fit_predict, xgb_grid
from energy_prediction.features.cpu_features import PERSISTENCE_INPUT, target_column
from energy_prediction.models.sample_weights import rise_sigma, rise_weights

FAMILY = "peaks"
EXCLUDED_VARIANTS = [
    {"variant": "perda assimétrica", "reason": "equivale, no limite, à regressão quantílica (pinball com τ > 0,5) já avaliada na família quantílica da V3"},
    {"variant": "Huber", "reason": "reduz a influência dos picos, direção oposta ao objetivo de H4; o descompasso MSE/MAE é outra pergunta"},
    {"variant": "multitarefa (nível e variação)", "reason": "exigiria nova arquitetura neural para eventos que, em 11 de 12 casos na V3, duraram um passo de 5 min"},
]


def peak_threshold(ctx, block, quantile: float) -> float:
    return float(np.nanquantile(ctx.series["cpu"].to_numpy()[: block.train_range[1] + 1], quantile))


def underestimation_at_peaks(y, pred, threshold) -> float:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    peak = y >= threshold
    return float(np.mean(y[peak] - pred[peak])) if peak.any() else np.nan


def choose_lambda(scores: dict[float, tuple[float, float]], max_mae_increase: float) -> tuple[float, list[float]]:
    """Regra pré-registrada: menor subestimação nos picos entre os λ com MAE interno até (1 + limite) × MAE de λ = 0."""
    base_mae = scores[0.0][0]
    allowed = [lam for lam, (m, _) in scores.items() if m <= base_mae * (1.0 + float(max_mae_increase))]
    finite = [lam for lam in allowed if np.isfinite(scores[lam][1])]
    chosen = min(finite, key=lambda lam: (scores[lam][1], lam)) if finite else 0.0
    return float(chosen), sorted(allowed)


def peak_row(y, pred, threshold) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    e = y - pred
    peak = y >= threshold
    return {
        **point_metrics(y, pred),
        "peak_threshold": threshold,
        "n_peak": int(peak.sum()),
        "peak_bias_obs_minus_pred": float(np.mean(e[peak])) if peak.any() else np.nan,
        "peak_underestimation_fraction": float(np.mean(e[peak] > 0)) if peak.any() else np.nan,
        "peak_mae": float(np.mean(np.abs(e[peak]))) if peak.any() else np.nan,
    }


def run(ctx, out: Path, *, run_dir=None, **_runtime) -> dict:
    fam = ctx.cfg["families"]["peaks"]
    xgb_cfg = ctx.cfg["families"]["baselines"]["xgboost"]
    grid = xgb_grid(xgb_cfg)
    columns = ctx.columns["A"]
    cpu_t = ctx.table[PERSISTENCE_INPUT].to_numpy(dtype=float)
    lambdas = [float(v) for v in fam["lambdas"]]
    frames, choices, tradeoff, alerts, timing = [], [], [], [], []
    cpu_series = pd.Series(ctx.series["cpu"].to_numpy(dtype=float))
    for block in ctx.blocks:
        threshold = peak_threshold(ctx, block, float(fam["peak_quantile"]))
        for steps in ctx.steps:
            y = ctx.table[target_column(steps)].to_numpy(dtype=float)
            sigma = rise_sigma(y[block.train], cpu_t[block.train])
            per_lambda = {}
            for lam in lambdas:
                t0 = time.perf_counter()
                fp = xgb_fit_predict(ctx.table, columns, steps, grid, xgb_cfg, ctx.seed, weight_fn=lambda train, extra, lam=lam: rise_weights(y[train], cpu_t[train], lam=lam, sigma=sigma))
                result = evaluate_block(len(grid), fp, y, block, metric=mae)
                inner_y = y[block.inner.validation]
                per_lambda[lam] = {
                    "result": result,
                    "inner_mae": mae(inner_y, result.inner_prediction),
                    "inner_peak_underestimation": underestimation_at_peaks(inner_y, result.inner_prediction, threshold),
                    "seconds": time.perf_counter() - t0,
                }
                timing.append({"block": block.label, "horizon_steps": int(steps), "lambda": lam, "train_seconds": per_lambda[lam]["seconds"]})
            chosen, allowed = choose_lambda({lam: (v["inner_mae"], v["inner_peak_underestimation"]) for lam, v in per_lambda.items()}, float(fam["max_mae_increase"]))
            choices.append({"block": block.label, "horizon_steps": int(steps), "chosen_lambda": chosen, "sigma_h": sigma, "peak_threshold": threshold, **{f"inner_mae_l{lam:g}": per_lambda[lam]["inner_mae"] for lam in lambdas}, **{f"inner_peak_under_l{lam:g}": per_lambda[lam]["inner_peak_underestimation"] for lam in lambdas}, "allowed_lambdas": allowed})
            for lam in lambdas:
                r = per_lambda[lam]["result"]
                variants = [f"lambda_{lam:g}"] + (["candidate"] if lam == chosen else [])
                for variant in variants:
                    frames.append(block_frame(ctx.table, block, block.inner.validation, phase=INNER, model="xgboost_rise", variant=variant, steps=steps, y_pred=r.inner_prediction, lam=lam, peak_threshold=threshold))
                    frames.append(block_frame(ctx.table, block, block.evaluation, phase=BLOCK, model="xgboost_rise", variant=variant, steps=steps, y_pred=r.block_prediction, lam=lam, peak_threshold=threshold))
                tradeoff.append({"block": block.label, "role": block.role, "horizon_steps": int(steps), "lambda": lam, "chosen": lam == chosen, "train_seconds": per_lambda[lam]["seconds"], **peak_row(y[block.evaluation], r.block_prediction, threshold)})
                low, high = int(ctx.table["origin_index"].to_numpy()[block.evaluation].min()) + steps, int(ctx.table["origin_index"].to_numpy()[block.evaluation].max()) + steps
                table = events(cpu_series, low, high, threshold, int(fam["merge_gap"]))
                signal = pd.Series(r.block_prediction, index=ctx.table["origin_index"].to_numpy()[block.evaluation] + steps)
                alerts.append({"block": block.label, "role": block.role, "horizon_steps": int(steps), "lambda": lam, "threshold": threshold, **alert_scores(signal, steps, table, low=low, high=high, threshold=threshold, tolerance=int(fam["tolerance"]))})
    predictions = pd.concat(frames, ignore_index=True)
    reference = predictions[predictions["variant"] == "lambda_0"].assign(model="xgboost_v2", variant="A")

    def peak_loss(part):
        return part["y_true"] - part["y_pred"]

    rows, per_day = [], []
    for steps in ctx.steps:
        peak = lambda f: f["y_true"] >= f["peak_threshold"]  # noqa: E731
        la = loss_frame(predictions, model="xgboost_rise", variant="candidate", steps=steps, loss=peak_loss, extra_filter=peak)
        lb = loss_frame(reference, model="xgboost_v2", variant="A", steps=steps, loss=peak_loss, extra_filter=peak)
        r, d = compare(la, lb, ctx.cfg, {"family": FAMILY, "hypothesis": "H4", "scope": "primary", "horizon_steps": int(steps), "model_a": "xgboost_rise", "variant_a": "candidate", "model_b": "xgboost_v2", "variant_b": "A", "metric": "erro (obs − prev) nos instantes de pico; negativo = menos subestimação"})
        rows += r
        per_day += d
        r, d = compare(loss_frame(predictions, model="xgboost_rise", variant="candidate", steps=steps), loss_frame(reference, model="xgboost_v2", variant="A", steps=steps), ctx.cfg, {"family": FAMILY, "hypothesis": "H4-mae", "scope": "exploratory", "horizon_steps": int(steps), "model_a": "xgboost_rise", "variant_a": "candidate", "model_b": "xgboost_v2", "variant_b": "A", "metric": "erro absoluto (custo no erro médio)"})
        rows += r
    comparisons = finalize_comparisons(rows, alpha=float(ctx.cfg["statistics"]["alpha"]))
    baseline = read_family_csv(run_dir, "baselines", "predictions")
    base_v2 = baseline[(baseline["model"] == "xgboost_v2") & (baseline["phase"] == BLOCK)]
    check = predictions[(predictions["variant"] == "lambda_0") & (predictions["phase"] == BLOCK)].merge(base_v2[["block", "horizon_steps", "origin_index", "y_pred"]], on=["block", "horizon_steps", "origin_index"], suffixes=("", "_baseline"))
    reproduces = float(np.max(np.abs(check["y_pred"] - check["y_pred_baseline"]))) if len(check) else np.nan
    write_tables(
        out,
        predictions=predictions,
        metrics=metrics_table(predictions),
        tradeoff=pd.DataFrame(tradeoff),
        alerts=pd.DataFrame(alerts),
        choices=pd.DataFrame(choices),
        timing=pd.DataFrame(timing),
        comparisons=comparisons,
        comparisons_by_day=pd.DataFrame(per_day),
        inherited_test=inherited_rows(pd.concat([predictions, reference], ignore_index=True), [(("xgboost_rise", "candidate"), ("xgboost_v2", "A"))], ctx.steps),
        family_info={
            "lambdas": lambdas,
            "selection_rule": f"menor subestimação média nos picos da validação interna com MAE interno até {float(fam['max_mae_increase']):.0%} acima de λ = 0",
            "peak_definition": f"alvo ≥ quantil {fam['peak_quantile']} da CPU do treino do bloco",
            "lambda0_reproduces_baseline_max_abs_diff": reproduces,
            "excluded_variants": EXCLUDED_VARIANTS,
        },
    )
    primary = comparisons[(comparisons["scope"] == "primary") & comparisons["primary_block"]]
    return {"primary_decisions": {f"H4/h{r.horizon_steps}": r.decision for r in primary.itertuples()}, "lambda0_reproduces_baseline_max_abs_diff": reproduces}
