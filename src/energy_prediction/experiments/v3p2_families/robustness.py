"""Família de robustez à mudança de nível (H8).

- H8a: redes (DLinear primária; TCN e PatchTST exploratórias) na formulação da V3 e com
  normalização por janela, com várias sementes; a comparação usa a perda por origem média entre
  as sementes.
- H8b: XGBoost da V2 com peso exponencial de recência (meia-vida escolhida na validação interna).
- H8c: XGBoost da V2 sem os atributos de calendário.
A referência do XGBoost vem da família `baselines`. A tabela de deslocamento de nível × viés por
bloco é o diagnóstico principal desta família.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.experiments.v3p2_families.common import (
    BLOCK,
    compare,
    finalize_comparisons,
    inherited_rows,
    level_shift_by_block,
    loss_frame,
    metrics_table,
    read_family_csv,
    write_tables,
)
from energy_prediction.experiments.v3p2_families.networks import NetworkData, run_network
from energy_prediction.experiments.v3p2_families.tabular import expand_grid, run_blocks, xgb_fit_predict, xgb_grid
from energy_prediction.features.cpu_features import CALENDAR_FEATURES
from energy_prediction.models.sample_weights import recency_weights
from energy_prediction.runtime.device import torch_installed

FAMILY = "robustness"


def seed_mean_loss(predictions: pd.DataFrame, *, model: str, variant: str, steps: int) -> pd.DataFrame:
    """Erro absoluto por origem médio entre as sementes (perda esperada de uma semente sorteada)."""
    part = predictions[(predictions["model"] == model) & (predictions["variant"] == variant) & (predictions["horizon_steps"] == steps) & (predictions["phase"] == BLOCK) & (predictions["role"] == "development")]
    if part.empty:
        raise KeyError(f"Sem previsões de {model}/{variant}/h{steps}.")
    part = part.assign(loss=(part["y_true"] - part["y_pred"]).abs())
    grouped = part.groupby(["day", "origin_index"], as_index=False)["loss"].mean()
    return grouped[["day", "origin_index", "loss"]]


def seed_mean_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "variant", "horizon_steps", "block", "day", "role", "phase", "origin_index"]
    agg = predictions.groupby(keys, as_index=False).agg(timestamp=("timestamp", "first"), cpu_t=("cpu_t", "first"), y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    return agg


def shift_bias_table(predictions: pd.DataFrame, shift: dict[str, float]) -> pd.DataFrame:
    block = predictions[predictions["phase"] == BLOCK]
    rows = []
    for (model, variant, steps), part in block.groupby(["model", "variant", "horizon_steps"]):
        per = []
        for name, b in part.groupby("block"):
            bias = float((b["y_true"] - b["y_pred"]).mean())
            per.append((shift[name], bias))
            rows.append({"model": model, "variant": variant, "horizon_steps": int(steps), "block": name, "role": b["role"].iloc[0], "level_shift_vs_train": shift[name], "bias_obs_minus_pred": bias})
        x, y = np.array(per).T
        corr = float(np.corrcoef(x, y)[0, 1]) if len(per) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan
        for r in rows[-len(per):]:
            r["corr_shift_bias_all_blocks"] = corr
    return pd.DataFrame(rows)


def run(ctx, out: Path, *, device: dict | None = None, no_amp: bool = False, **runtime) -> dict:
    rob = ctx.cfg["families"]["robustness"]
    xgb_cfg = ctx.cfg["families"]["baselines"]["xgboost"]
    base_grid = xgb_grid(xgb_cfg)
    columns = ctx.columns["A"]
    origins = ctx.table["origin_index"].to_numpy()
    frames, selections = [], []
    excluded = []

    recency_grid = expand_grid(base_grid, half_life=[int(h) for h in rob["recency_half_lives"]])
    no_calendar = [c for c in columns if c not in CALENDAR_FEATURES]
    for steps in ctx.steps:
        fp = xgb_fit_predict(ctx.table, columns, steps, recency_grid, xgb_cfg, ctx.seed, weight_fn=lambda train, extra: recency_weights(origins[train], extra["_half_life"]))
        f, s = run_blocks(ctx, model="xgboost_v2", variant="recency", steps=steps, fit_predict=fp, grid=recency_grid)
        frames += f
        selections += s
        fp = xgb_fit_predict(ctx.table, no_calendar, steps, base_grid, xgb_cfg, ctx.seed)
        f, s = run_blocks(ctx, model="xgboost_v2", variant="no_calendar", steps=steps, fit_predict=fp, grid=base_grid)
        frames += f
        selections += s
    tabular = pd.concat(frames, ignore_index=True).assign(seed=-1)

    loop = dict(rob["loop"])
    if no_amp:
        loop["mixed_precision"] = "off"
    net_frames, profile = [], []
    effective = (device or {}).get("effective", "cpu")
    if torch_installed():
        data = NetworkData(ctx, int(rob["window"]))
        for arch in rob["networks"]:
            for formulation in ("v3", "window"):
                f, s, p = run_network(ctx, data, arch=arch, formulation=formulation, seeds=[int(x) for x in rob["seeds"]], rob={**rob, "loop": loop}, device=effective)
                net_frames += f
                selections += s
                profile += p
    else:
        excluded.append({"item": "redes (H8a)", "reason": "PyTorch não instalado (extra v3)"})
    networks = pd.concat(net_frames, ignore_index=True) if net_frames else pd.DataFrame(columns=tabular.columns)

    shift = level_shift_by_block(ctx)
    reference = read_family_csv(runtime["run_dir"], "baselines", "predictions")
    reference = reference[reference["model"] == "xgboost_v2"].assign(seed=-1)
    nets_mean = seed_mean_predictions(networks) if not networks.empty else networks
    combined = pd.concat([reference, tabular, nets_mean.assign(seed=-2)], ignore_index=True)
    combined["level_shift"] = combined["block"].map(shift)

    rows, per_day = [], []
    cfg = ctx.cfg
    for steps in ctx.steps:
        def add(scope, hypothesis, a, b, la, lb):
            meta = {"family": FAMILY, "hypothesis": hypothesis, "scope": scope, "horizon_steps": int(steps), "model_a": a[0], "variant_a": a[1], "model_b": b[0], "variant_b": b[1], "metric": "erro absoluto"}
            r, d = compare(la, lb, cfg, meta)
            rows.extend(r)
            per_day.extend(d)

        add("primary", "H8b", ("xgboost_v2", "recency"), ("xgboost_v2", "A"), loss_frame(tabular, model="xgboost_v2", variant="recency", steps=steps), loss_frame(reference, model="xgboost_v2", variant="A", steps=steps))
        add("primary", "H8c", ("xgboost_v2", "no_calendar"), ("xgboost_v2", "A"), loss_frame(tabular, model="xgboost_v2", variant="no_calendar", steps=steps), loss_frame(reference, model="xgboost_v2", variant="A", steps=steps))
        if not networks.empty:
            for arch in rob["networks"]:
                scope = "primary" if arch == rob["primary_network"] else "exploratory"
                add(scope, "H8a", (arch, "window"), (arch, "v3"), seed_mean_loss(networks, model=arch, variant="window", steps=steps), seed_mean_loss(networks, model=arch, variant="v3", steps=steps))
                add("exploratory", "H8a-vs-xgboost", (arch, "window"), ("xgboost_v2", "A"), seed_mean_loss(networks, model=arch, variant="window", steps=steps), loss_frame(reference, model="xgboost_v2", variant="A", steps=steps))
    comparisons = finalize_comparisons(rows, alpha=float(cfg["statistics"]["alpha"]))

    seed_metrics = metrics_table(networks, keys=("model", "variant", "horizon_steps", "seed")) if not networks.empty else pd.DataFrame()
    spread = pd.DataFrame()
    if not seed_metrics.empty:
        spread = seed_metrics.groupby(["model", "variant", "horizon_steps", "block", "role"], as_index=False).agg(mae_mean=("mae", "mean"), mae_std_between_seeds=("mae", "std"), bias_mean=("bias_obs_minus_pred", "mean"), n_seeds=("seed", "nunique"))
    pairs = [(("xgboost_v2", "recency"), ("xgboost_v2", "A")), (("xgboost_v2", "no_calendar"), ("xgboost_v2", "A"))] + [((a, "window"), (a, "v3")) for a in rob["networks"]]
    write_tables(
        out,
        predictions=pd.concat([tabular, networks], ignore_index=True),
        metrics=metrics_table(combined),
        seed_metrics=seed_metrics,
        seed_spread=spread,
        shift_bias=shift_bias_table(combined, shift),
        comparisons=comparisons,
        comparisons_by_day=pd.DataFrame(per_day),
        inherited_test=inherited_rows(combined, pairs, ctx.steps),
        training_profile=pd.DataFrame(profile),
        selections=selections,
        family_info={
            "networks": list(rob["networks"]),
            "primary_network": rob["primary_network"],
            "seeds": list(rob["seeds"]),
            "window": int(rob["window"]),
            "mixed_precision": loop["mixed_precision"],
            "device": effective,
            "recency_half_lives": list(rob["recency_half_lives"]),
            "statement": "Comparações primárias nos dias de desenvolvimento (exploratórias); o dia 7 é teste herdado.",
        },
    )
    primary = comparisons[(comparisons["scope"] == "primary") & comparisons["primary_block"]] if not comparisons.empty else comparisons
    return {"primary_decisions": {f"{r.hypothesis}/h{r.horizon_steps}": r.decision for r in primary.itertuples()}, "excluded": excluded}
