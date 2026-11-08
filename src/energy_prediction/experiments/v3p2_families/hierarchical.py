"""Família hierárquica (H1): grupos de máquinas, XGBoost por grupo, reconstrução e reconciliação.

Em cada bloco, os grupos são formados só com o treino do bloco; a série de cada grupo é a média
ponderada das máquinas presentes do grupo. Um XGBoost (atributos A e grade da V2) por grupo e
horizonte é escolhido na validação interna e reajustado no treino do bloco. A previsão agregada é
reconstruída pelos pesos presentes na origem (de baixo para cima) ou reconciliada (OLS e MinT com
encolhimento, com o XGBoost direto da família `baselines` como previsão base do total). O
candidato primário (critério × K × método) é escolhido por horizonte no MAE agregado da validação
interna; a matriz completa é exploratória. A reconstrução com os pesos do instante do alvo é só
diagnóstico do efeito da variação de presença.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.evaluation.day_forward import DayBlock, evaluate_block
from energy_prediction.evaluation.reconciliation import origin_shares, reconcile, shrunk_covariance
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
from energy_prediction.experiments.v3p2_families.tabular import inherited_step, xgb_grid
from energy_prediction.features.cpu_features import build_feature_table, target_column
from energy_prediction.models.registry import XGBoostModel
from energy_prediction.preprocessing.grouping import assign_groups, group_series
from energy_prediction.preprocessing.panel import train_statistics

FAMILY = "hierarchical"
NATURAL = "natural"


def nan_mae(y_true, y_pred) -> float:
    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.mean(np.abs(e))) if len(e) else np.inf


def group_design(ctx, values: np.ndarray, steps_list: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Atributos A da série do grupo (preenchimento causal pelo último valor) e alvos sem preenchimento."""
    raw = pd.Series(values)
    series = pd.DataFrame({"bin": ctx.series["bin"].to_numpy(), "timestamp": ctx.series["timestamp"].to_numpy(), "cpu": raw.ffill().to_numpy()})
    table = build_feature_table(series, {"cpu": ctx.specs["cpu"]}, list(steps_list))
    X = table[ctx.columns["A"]].to_numpy(dtype=float)
    targets = {s: raw.shift(-s).to_numpy()[: len(table)] for s in steps_list}
    return X, targets


def _fit_group(X, y, positions_origin, train_o, inner_train_o, inner_val_o, eval_o, grid, xgb_cfg, seed):
    """Seleção interna, reajuste e previsões de um grupo; trabalho isolado para paralelizar."""

    def fp(g, train, predict, **_):
        rows = train[np.isfinite(y[train]) & np.isfinite(X[train]).all(axis=1)]
        model = XGBoostModel(grid[g], seed=seed, n_jobs=int(xgb_cfg["n_jobs"]), fixed=xgb_cfg.get("fixed"))
        model.fit(X[rows], y[rows])
        out = np.full(len(predict), np.nan)
        ok = np.isfinite(X[predict]).all(axis=1)
        if ok.any():
            out[ok] = model.predict(X[predict][ok])
        return out, {}

    class _Inner:
        train, validation = inner_train_o, inner_val_o

    block = type("B", (), {"inner": _Inner, "train": train_o, "evaluation": eval_o, "label": "group"})
    result = evaluate_block(len(grid), fp, y, block, metric=nan_mae)
    return result.best, result.grid_scores, result.inner_prediction, result.block_prediction


def _parallel(jobs, n_jobs: int):
    if int(n_jobs) <= 1:
        return [fn(*args) for fn, args in jobs]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=int(n_jobs), backend="loky")(delayed(fn)(*args) for fn, args in jobs)


def combos(fam: dict) -> list[tuple[str, int | str]]:
    out = []
    for criterion in fam["criteria"]:
        if criterion == "failure_domain":
            out.append((criterion, NATURAL))
        else:
            out.extend((criterion, int(k)) for k in fam["k_values"])
    return out


def block_groups(ctx, block: DayBlock, criterion: str, k, fam: dict, stats_cache: dict) -> pd.DataFrame:
    if block.label not in stats_cache:
        last_bin = int(ctx.series["bin"].iloc[block.train_range[1]])
        stats_cache[block.label] = train_statistics(ctx.panel(), last_bin=last_bin, step_seconds=ctx.step_seconds, min_coverage=float(ctx.cfg["panel"]["min_train_coverage"]))
    return assign_groups(
        stats_cache[block.label], criterion, None if k == NATURAL else int(k), domains=ctx.domains(),
        min_group_size=int(fam["min_group_size"]), n_init=int(fam["profile_n_init"]), seed=ctx.seed,
    )


def fleet_structure(ctx, block: DayBlock, values: np.ndarray, weights: np.ndarray) -> dict:
    """Variância explicada pela 1ª componente (máquinas) e correlação média entre grupos no treino."""
    stop = block.train_range[1] + 1
    cpu = ctx.panel().values["cpu"][:stop].astype(float)
    keep = np.isfinite(cpu).mean(axis=0) > 0.8
    Z = pd.DataFrame(cpu[:, keep]).ffill().bfill().to_numpy()
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) < 1e-9, 1.0, Z.std(0))
    s = np.linalg.svd(Z - Z.mean(0), full_matrices=False, compute_uv=False)
    share = float((s[0] ** 2) / np.sum(s**2)) if len(s) else np.nan
    g = pd.DataFrame(values[:stop]).dropna(axis=1, how="all")
    corr = g.corr().to_numpy()
    off = corr[~np.eye(len(corr), dtype=bool)]
    return {"block": block.label, "pc1_variance_share_machines": share, "n_machines_pca": int(keep.sum()), "mean_corr_between_groups": float(np.nanmean(off)) if len(off) else np.nan}


def run(ctx, out: Path, *, run_dir=None, **_runtime) -> dict:
    fam = ctx.cfg["families"]["hierarchical"]
    xgb_cfg = ctx.cfg["families"]["baselines"]["xgboost"]
    grid = xgb_grid(xgb_cfg)
    started = time.perf_counter()
    weights = ctx.panel_weights()
    base = read_family_csv(run_dir, "baselines", "predictions")
    base = base[base["model"] == "xgboost_v2"]
    base_paths = read_family_csv(run_dir, "baselines", "paths")
    base_paths = base_paths[base_paths["model"] == "xgboost_v2"]
    fleet_y = {s: ctx.table[target_column(s)].to_numpy(dtype=float) for s in ctx.path_steps}
    origins = ctx.table["origin_index"].to_numpy()
    stats_cache: dict = {}
    assignments, group_rows, agg_frames, structure, drift_rows, selections, timing = [], [], [], [], [], [], []
    candidate_rows, group_path_rows, path_rows = [], [], []
    methods = list(fam["reconciliation"])
    for block in ctx.blocks:
        o_train, o_itrain, o_ival, o_eval = (origins[block.train], origins[block.inner.train], origins[block.inner.validation], origins[block.evaluation])
        per_combo: dict = {}
        for criterion, k in combos(fam):
            t0 = time.perf_counter()
            assignment = block_groups(ctx, block, criterion, k, fam, stats_cache)
            assignments.append(assignment.assign(block=block.label, criterion=criterion, k=str(k)))
            labels, values, present = group_series(ctx.panel(), weights, assignment)
            if criterion == fam["criteria"][0] and k == combos(fam)[0][1]:
                structure.append(fleet_structure(ctx, block, values, present))
            designs = [group_design(ctx, values[:, j], ctx.path_steps) for j in range(len(labels))]
            jobs = []
            for steps in ctx.steps:
                for j in range(len(labels)):
                    X, targets = designs[j]
                    jobs.append((_fit_group, (X, targets[steps], None, o_train, o_itrain, o_ival, o_eval, grid, xgb_cfg, ctx.seed)))
            results = _parallel(jobs, int(fam["n_jobs"]))
            it = iter(results)
            fitted = {steps: [next(it) for _ in labels] for steps in ctx.steps}
            per_combo[(criterion, k)] = (labels, values, present, fitted, designs)
            timing.append({"block": block.label, "criterion": criterion, "k": str(k), "n_groups": len(labels), "seconds": time.perf_counter() - t0})
            for steps in ctx.steps:
                for j, label in enumerate(labels):
                    best, scores, inner_pred, block_pred = fitted[steps][j]
                    selections.append({"block": block.label, "criterion": criterion, "k": str(k), "group": label, "horizon_steps": int(steps), "best_params": grid[best], "grid_scores": scores})
                    y_g = designs[j][1][steps]
                    share = origin_shares(present[o_eval])[:, j]
                    for phase, o, pred in ((INNER, o_ival, inner_pred), (BLOCK, o_eval, block_pred)):
                        group_rows.append(pd.DataFrame({"block": block.label, "day": block.day, "role": block.role, "criterion": criterion, "k": str(k), "group": label, "horizon_steps": int(steps), "phase": phase, "origin_index": o, "y_true_group": y_g[o], "y_pred_group": pred, "weight_share_origin": origin_shares(present[o])[:, j], "n_machines_group": int((assignment["group"] == label).sum())}))
        # Reconciliação por combinação e escolha do candidato na validação interna.
        for steps in ctx.steps:
            b_inner = base[(base["block"] == block.label) & (base["horizon_steps"] == steps) & (base["phase"] == INNER)].set_index("origin_index")["y_pred"].reindex(o_ival).to_numpy()
            b_block = base[(base["block"] == block.label) & (base["horizon_steps"] == steps) & (base["phase"] == BLOCK)].set_index("origin_index")["y_pred"].reindex(o_eval).to_numpy()
            best_variant, best_score = None, np.inf
            for criterion, k in combos(fam):
                labels, values, present, fitted, designs = per_combo[(criterion, k)]
                g_inner = np.column_stack([fitted[steps][j][2] for j in range(len(labels))])
                g_block = np.column_stack([fitted[steps][j][3] for j in range(len(labels))])
                y_inner_groups = np.column_stack([designs[j][1][steps][o_ival] for j in range(len(labels))])
                resid = np.column_stack([fleet_y[steps][block.inner.validation] - b_inner, y_inner_groups - g_inner])
                W, lam = shrunk_covariance(resid)
                for method in methods:
                    tot_inner, _ = reconcile(b_inner, g_inner, origin_shares(present[o_ival]), method, W)
                    tot_block, rec_groups = reconcile(b_block, g_block, origin_shares(present[o_eval]), method, W)
                    variant = f"{criterion}_k{k}_{method}"
                    score = nan_mae(fleet_y[steps][block.inner.validation], tot_inner)
                    selections.append({"block": block.label, "variant": variant, "horizon_steps": int(steps), "inner_aggregate_mae": score, "mint_shrinkage": lam if method == "mint_shrink" else None})
                    for phase, pos, pred in ((INNER, block.inner.validation, tot_inner), (BLOCK, block.evaluation, tot_block)):
                        agg_frames.append(pd.DataFrame({"model": "hierarchical", "variant": variant, "horizon_steps": int(steps), "block": block.label, "day": block.day, "role": block.role, "phase": phase, "origin_index": origins[pos], "timestamp": ctx.table["timestamp"].to_numpy()[pos], "cpu_t": ctx.table["cpu_t"].to_numpy()[pos], "y_true": fleet_y[steps][pos], "y_pred": pred, "criterion": criterion, "k": str(k), "method": method}))
                    if score < best_score - 1e-12:
                        best_variant, best_score = (criterion, k, method), score
                    if method == "bottom_up":
                        drift_target, _ = reconcile(b_block, g_block, origin_shares(present[o_eval + steps]), "bottom_up")
                        drift_rows.append(pd.DataFrame({"block": block.label, "variant": variant, "horizon_steps": int(steps), "origin_index": o_eval, "bottom_up_origin_weights": tot_block, "bottom_up_target_weights": drift_target, "difference": tot_block - drift_target}))
            criterion, k, method = best_variant
            candidate_rows.append({"block": block.label, "day": block.day, "role": block.role, "horizon_steps": int(steps), "criterion": criterion, "k": str(k), "method": method, "inner_aggregate_mae": best_score})
        # Trajetória do candidato do maior horizonte, para a rota de energia por grupo.
        chosen = next(r for r in candidate_rows if r["block"] == block.label and r["horizon_steps"] == max(ctx.steps))
        key = (chosen["criterion"], NATURAL if chosen["k"] == NATURAL else int(chosen["k"]))
        labels, values, present, fitted, designs = per_combo[key]
        missing = [step for step in ctx.path_steps if step not in fitted]
        jobs = []
        for step in missing:
            scored = inherited_step(step, ctx.steps)
            for j in range(len(labels)):
                params = [grid[fitted[scored][j][0]]]  # passo não pontuado herda o hiperparâmetro do horizonte que o cobre
                jobs.append((_fit_group, (designs[j][0], designs[j][1][step], None, o_train, o_itrain, o_ival, o_eval, params, xgb_cfg, ctx.seed)))
        results = iter(_parallel(jobs, int(fam["n_jobs"])))
        for step in missing:
            fitted[step] = [next(results) for _ in labels]
        b_inner_by_step = {}
        for step in ctx.path_steps:
            scored = inherited_step(step, ctx.steps)
            g_block = np.column_stack([fitted[step][j][3] for j in range(len(labels))])
            b_path = base_paths[(base_paths["block"] == block.label) & (base_paths["step"] == step)].set_index("origin_index")["y_pred"].reindex(o_eval).to_numpy()
            g_inner = np.column_stack([fitted[step][j][2] for j in range(len(labels))])
            y_inner_groups = np.column_stack([designs[j][1][step][o_ival] for j in range(len(labels))])
            b_inner = base[(base["block"] == block.label) & (base["horizon_steps"] == scored) & (base["phase"] == INNER)].set_index("origin_index")["y_pred"].reindex(o_ival).to_numpy()
            b_inner_by_step[step] = scored
            W, _ = shrunk_covariance(np.column_stack([fleet_y[scored][block.inner.validation] - b_inner, y_inner_groups - g_inner]))
            total, rec = reconcile(b_path, g_block, origin_shares(present[o_eval]), chosen["method"], W)
            path_rows.append(pd.DataFrame({"model": "hierarchical", "variant": "candidate", "block": block.label, "day": block.day, "role": block.role, "step": int(step), "origin_index": o_eval, "y_true": fleet_y[step][block.evaluation], "y_pred": total}))
            for j, label in enumerate(labels):
                group_path_rows.append(pd.DataFrame({"block": block.label, "criterion": chosen["criterion"], "k": chosen["k"], "group": label, "step": int(step), "origin_index": o_eval, "y_pred_group": rec[:, j], "weight_share_origin": origin_shares(present[o_eval])[:, j]}))

    predictions = pd.concat(agg_frames, ignore_index=True)
    cand = pd.DataFrame(candidate_rows)
    pieces = []
    for r in cand.itertuples():
        part = predictions[(predictions["block"] == r.block) & (predictions["horizon_steps"] == r.horizon_steps) & (predictions["variant"] == f"{r.criterion}_k{r.k}_{r.method}")]
        pieces.append(part.assign(variant="candidate", chosen=f"{r.criterion}_k{r.k}_{r.method}"))
    predictions = pd.concat([predictions, *pieces], ignore_index=True)
    reference = base.copy()

    rows, per_day = [], []
    for steps in ctx.steps:
        ref = loss_frame(reference, model="xgboost_v2", variant="A", steps=steps)
        r, d = compare(loss_frame(predictions, model="hierarchical", variant="candidate", steps=steps), ref, ctx.cfg, {"family": FAMILY, "hypothesis": "H1", "scope": "primary", "horizon_steps": int(steps), "model_a": "hierarchical", "variant_a": "candidate", "model_b": "xgboost_v2", "variant_b": "A", "metric": "erro absoluto"})
        rows += r
        per_day += d
        for variant in sorted(v for v in predictions["variant"].unique() if v != "candidate"):
            r, d = compare(loss_frame(predictions, model="hierarchical", variant=variant, steps=steps), ref, ctx.cfg, {"family": FAMILY, "hypothesis": "H1", "scope": "exploratory", "horizon_steps": int(steps), "model_a": "hierarchical", "variant_a": variant, "model_b": "xgboost_v2", "variant_b": "A", "metric": "erro absoluto"})
            rows += r
    comparisons = finalize_comparisons(rows, alpha=float(ctx.cfg["statistics"]["alpha"]))

    groups = pd.concat(group_rows, ignore_index=True)
    gm = []
    for key, part in groups[groups["phase"] == BLOCK].groupby(["block", "role", "criterion", "k", "group", "horizon_steps"]):
        ok = np.isfinite(part["y_true_group"]) & np.isfinite(part["y_pred_group"])
        gm.append({**dict(zip(["block", "role", "criterion", "k", "group", "horizon_steps"], key)), **point_metrics(part.loc[ok, "y_true_group"], part.loc[ok, "y_pred_group"]), "mean_weight_share": float(part["weight_share_origin"].mean()), "n_machines_group": int(part["n_machines_group"].iloc[0])})
    gm = pd.DataFrame(gm)
    weighted = gm.groupby(["block", "role", "criterion", "k", "horizon_steps"]).apply(lambda p: pd.Series({"weighted_group_mae": float(np.average(p["mae"], weights=p["mean_weight_share"].fillna(0) + 1e-12)), "weighted_group_bias": float(np.average(p["bias_obs_minus_pred"], weights=p["mean_weight_share"].fillna(0) + 1e-12)), "n_groups": int(len(p))}), include_groups=False).reset_index()
    write_tables(
        out,
        predictions=predictions,
        metrics=metrics_table(pd.concat([predictions, reference], ignore_index=True)),
        group_predictions=groups,
        group_metrics=gm,
        group_metrics_weighted=weighted,
        candidates=cand,
        assignments=pd.concat(assignments, ignore_index=True),
        weight_drift=pd.concat(drift_rows, ignore_index=True),
        paths=pd.concat(path_rows, ignore_index=True),
        group_paths=pd.concat(group_path_rows, ignore_index=True),
        comparisons=comparisons,
        comparisons_by_day=pd.DataFrame(per_day),
        inherited_test=inherited_rows(pd.concat([predictions, reference], ignore_index=True), [(("hierarchical", "candidate"), ("xgboost_v2", "A"))], ctx.steps),
        structure=pd.DataFrame(structure),
        timing=pd.DataFrame(timing),
        selections=selections,
        family_info={
            "criteria": list(fam["criteria"]),
            "k_values": list(fam["k_values"]),
            "reconciliation": methods,
            "candidate_rule": "menor MAE agregado na validação interna, por bloco e horizonte",
            "limitations": [
                "A covariância da MinT e a escolha do candidato usam a mesma validação interna; isso favorece a MinT na escolha.",
                "Grupos e estatísticas são recalculados em cada bloco só com o treino do bloco.",
            ],
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    primary = comparisons[(comparisons["scope"] == "primary") & comparisons["primary_block"]]
    return {"primary_decisions": {f"H1/h{r.horizon_steps}": r.decision for r in primary.itertuples()}, "candidates": candidate_rows, "elapsed_seconds": time.perf_counter() - started}
