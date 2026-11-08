"""Família de combinação condicionada a regime (H3), sobre as previsões da família `baselines`.

Pesos e seleções vêm só da validação interna de cada bloco. A escolha entre os dois combinadores
condicionados (seleção e inverso do MAE) é feita com divisão da validação interna em metades
cronológicas: pesos na primeira metade, MAE na segunda; o escolhido é reajustado na validação
interna inteira. O regime realizado (variação até o alvo) é só diagnóstico.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.experiments.v3p2_families.common import (
    BLOCK,
    INNER,
    compare,
    finalize_comparisons,
    inherited_rows,
    loss_frame,
    metrics_table,
    point_metrics,
    read_family_csv,
    write_tables,
)
from energy_prediction.models.regime import REGIMES, classify, combine, fit_weights, regime_signals, thresholds

FAMILY = "regimes"
CONDITIONED = ("select", "inverse_mae")
UNCONDITIONED = ("mean", "global_inverse_mae")


def realized_regime(y_true, cpu_t, sigma) -> np.ndarray:
    change = np.asarray(y_true) - np.asarray(cpu_t)
    out = np.full(len(change), "estavel", dtype=object)
    out[change >= sigma] = "subida"
    out[change <= -sigma] = "queda"
    return out


def run(ctx, out: Path, *, run_dir=None, **_runtime) -> dict:
    fam = ctx.cfg["families"]["regimes"]
    experts = list(fam["experts"])
    base = read_family_csv(run_dir, "baselines", "predictions")
    t0 = time.perf_counter()
    slope, vol = regime_signals(ctx.series["cpu"].to_numpy(), slope_steps=int(fam["slope_steps"]), volatility_steps=int(fam["volatility_steps"]))
    detector_seconds = time.perf_counter() - t0
    origins_all = ctx.table["origin_index"].to_numpy()
    frames, choices, regime_rows, weight_rows, confusion = [], [], [], [], []
    t1 = time.perf_counter()
    for block in ctx.blocks:
        limits = thresholds(slope, vol, origins_all[block.train], volatility_quantile=float(fam["volatility_quantile"]), rise_quantile=float(fam["rise_quantile"]), fall_quantile=float(fam["fall_quantile"]))
        for steps in ctx.steps:
            def matrix(phase):
                part = base[(base["block"] == block.label) & (base["horizon_steps"] == steps) & (base["phase"] == phase)]
                wide = part.pivot_table(index="origin_index", columns="model", values="y_pred")
                y = part.drop_duplicates("origin_index").set_index("origin_index")["y_true"].reindex(wide.index)
                cpu_t = part.drop_duplicates("origin_index").set_index("origin_index")["cpu_t"].reindex(wide.index)
                return wide.index.to_numpy(), wide[experts].to_numpy(dtype=float), y.to_numpy(dtype=float), cpu_t.to_numpy(dtype=float)

            o_in, P_in, y_in, _ = matrix(INNER)
            o_bl, P_bl, y_bl, cpu_bl = matrix(BLOCK)
            r_in = classify(slope, vol, o_in, limits)
            r_bl = classify(slope, vol, o_bl, limits)
            half = len(o_in) // 2
            scores = {}
            for method in CONDITIONED:
                w, _ = fit_weights(y_in[:half], P_in[:half], r_in[:half], method=method, min_origins=int(fam["min_regime_origins"]) // 2)
                scores[method] = float(np.mean(np.abs(y_in[half:] - combine(P_in[half:], r_in[half:], w))))
            chosen = min(CONDITIONED, key=lambda m: (scores[m], CONDITIONED.index(m)))
            choices.append({"block": block.label, "horizon_steps": int(steps), "chosen": chosen, **{f"half_split_mae_{m}": v for m, v in scores.items()}, **{f"threshold_{k}": v for k, v in limits.items()}})
            for method in (*CONDITIONED, *UNCONDITIONED):
                w, fallback = fit_weights(y_in, P_in, r_in, method=method, min_origins=int(fam["min_regime_origins"]))
                pred = combine(P_bl, r_bl, w)
                variants = [method] + (["candidate"] if method == chosen else [])
                for variant in variants:
                    frames.append(pd.DataFrame({"model": "combination", "variant": variant, "horizon_steps": int(steps), "block": block.label, "day": block.day, "role": block.role, "phase": BLOCK, "origin_index": o_bl, "timestamp": np.nan, "cpu_t": cpu_bl, "y_true": y_bl, "y_pred": pred, "regime": r_bl}))
                for regime in REGIMES:
                    weight_rows.append({"block": block.label, "horizon_steps": int(steps), "method": method, "regime": regime, "fallback_to_global": regime in fallback, **dict(zip(experts, np.round(w[regime], 6)))})
            for regime in REGIMES:
                regime_rows.append({"block": block.label, "horizon_steps": int(steps), "regime": regime, "n_inner": int((r_in == regime).sum()), "n_block": int((r_bl == regime).sum())})
            sigma = float(np.nanstd(ctx.table[f"y_h{steps}"].to_numpy()[block.train] - ctx.table["cpu_t"].to_numpy()[block.train]))
            real = realized_regime(y_bl, cpu_bl, sigma)
            confusion.append(pd.crosstab(pd.Series(r_bl, name="detected"), pd.Series(real, name="realized")).stack().rename("count").reset_index().assign(block=block.label, horizon_steps=int(steps)))
    combine_seconds = time.perf_counter() - t1
    predictions = pd.concat(frames, ignore_index=True)
    experts_frame = base[base["model"].isin(experts) & (base["phase"] == BLOCK)]
    reference = base[base["model"] == "xgboost_v2"]

    rows, per_day = [], []
    for steps in ctx.steps:
        ref = loss_frame(reference, model="xgboost_v2", variant="A", steps=steps)
        r, d = compare(loss_frame(predictions, model="combination", variant="candidate", steps=steps), ref, ctx.cfg, {"family": FAMILY, "hypothesis": "H3", "scope": "primary", "horizon_steps": int(steps), "model_a": "combination", "variant_a": "candidate", "model_b": "xgboost_v2", "variant_b": "A", "metric": "erro absoluto"})
        rows += r
        per_day += d
        for method in (*CONDITIONED, *UNCONDITIONED):
            r, _ = compare(loss_frame(predictions, model="combination", variant=method, steps=steps), ref, ctx.cfg, {"family": FAMILY, "hypothesis": "H3", "scope": "exploratory", "horizon_steps": int(steps), "model_a": "combination", "variant_a": method, "model_b": "xgboost_v2", "variant_b": "A", "metric": "erro absoluto"})
            rows += r
    comparisons = finalize_comparisons(rows, alpha=float(ctx.cfg["statistics"]["alpha"]))

    by_regime = []
    allp = pd.concat([predictions, experts_frame.merge(predictions[["block", "horizon_steps", "origin_index", "regime"]].drop_duplicates(), on=["block", "horizon_steps", "origin_index"], how="left")], ignore_index=True)
    for key, part in allp.groupby(["model", "variant", "horizon_steps", "role", "regime"]):
        by_regime.append({**dict(zip(["model", "variant", "horizon_steps", "role", "regime"], key)), **point_metrics(part["y_true"], part["y_pred"])})
    write_tables(
        out,
        predictions=predictions,
        metrics=metrics_table(pd.concat([predictions, experts_frame], ignore_index=True)),
        metrics_by_regime=pd.DataFrame(by_regime),
        regime_counts=pd.DataFrame(regime_rows),
        weights=pd.DataFrame(weight_rows),
        choices=pd.DataFrame(choices),
        confusion_detected_vs_realized=pd.concat(confusion, ignore_index=True),
        comparisons=comparisons,
        comparisons_by_day=pd.DataFrame(per_day),
        inherited_test=inherited_rows(pd.concat([predictions, reference], ignore_index=True), [(("combination", "candidate"), ("xgboost_v2", "A"))], ctx.steps),
        family_info={
            "experts": experts,
            "detector": {k: fam[k] for k in ("slope_steps", "volatility_steps", "volatility_quantile", "rise_quantile", "fall_quantile")},
            "cost_seconds": {"detector": detector_seconds, "combination": combine_seconds, "note": "sem o custo de ajuste dos especialistas (família baselines)"},
            "statement": "O regime realizado (com o alvo) aparece só na matriz de confusão; nenhum combinador o usa.",
        },
    )
    primary = comparisons[(comparisons["scope"] == "primary") & comparisons["primary_block"]]
    return {"primary_decisions": {f"H3/h{r.horizon_steps}": r.decision for r in primary.itertuples()}, "cost_seconds": detector_seconds + combine_seconds}
