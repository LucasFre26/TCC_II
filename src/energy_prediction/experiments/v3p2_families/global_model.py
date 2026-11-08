"""Família do modelo global por máquina (H2).

Em cada bloco: estatísticas de treino por máquina, amostra estratificada de M máquinas, pares
(origem, máquina) do treino com passo fixo e um XGBoost compartilhado por horizonte, escolhido
pelo MAE agregado na validação interna (previsões de todas as máquinas presentes, agregadas com
`cpu_num`). Reajuste no treino do bloco e previsão agregada do bloco; as previsões por máquina
(passos 1 a 6) vão para a energia. O diagnóstico de escala repete o ajuste com outros M num único
bloco. O tamanho efetivo por instante é estimado pelo efeito de desenho e não se confunde com o
número de janelas. A TCN global só roda se o portão pré-registrado for cumprido.
"""

from __future__ import annotations

import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.evaluation.metrics import mae
from energy_prediction.experiments.v3p2_families.common import (
    BLOCK,
    INNER,
    compare,
    finalize_comparisons,
    inherited_rows,
    loss_frame,
    metrics_table,
    read_family_csv,
    write_tables,
)
from energy_prediction.experiments.v3p2_families.tabular import inherited_step
from energy_prediction.features.cpu_features import target_column
from energy_prediction.features.machine_features import build_machine_features, design_effect, sample_machines, training_pairs
from energy_prediction.models.global_xgb import GlobalXGBoost, aggregate_predictions, xgb_device
from energy_prediction.preprocessing.panel import train_statistics
from energy_prediction.runtime.device import torch_installed

FAMILY = "global_model"


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _fit(features, o, c, steps, params, *, seed, n_jobs, device):
    X = features.design(o, c)
    y = features.target(o, c, steps)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    model = GlobalXGBoost(params, seed=seed, n_jobs=n_jobs, device=device).fit(X[ok], y[ok])
    return model, {"n_examples": int(ok.sum()), "n_dropped_missing": int((~ok).sum()), "fit_seconds": model.fit_seconds}


def _frame(ctx, block, positions, *, phase, model, variant, steps, y_pred, **extra):
    from energy_prediction.experiments.v3p2_families.common import block_frame

    return block_frame(ctx.table, block, positions, phase=phase, model=model, variant=variant, steps=steps, y_pred=y_pred, **extra)


class BlockFit:
    """Ajuste do XGBoost global num bloco para um conjunto de máquinas amostradas."""

    def __init__(self, ctx, block, features, sampled_cols, all_cols, weights, gm, device):
        self.ctx, self.block, self.features = ctx, block, features
        self.sampled, self.all, self.weights, self.gm, self.device = sampled_cols, all_cols, weights, gm, device
        self.origins = ctx.table["origin_index"].to_numpy()
        self.grid = list(gm["grid"])

    def run(self, steps: int, *, keep_machine: bool) -> dict:
        gm, ctx, block = self.gm, self.ctx, self.block
        y_fleet = ctx.table[target_column(steps)].to_numpy(dtype=float)
        o_in, c_in = training_pairs(self.origins[block.inner.train], self.sampled, int(gm["origin_stride"]))
        scores, infos = [], []
        inner_preds = []
        t0 = time.perf_counter()
        for params in self.grid:
            model, info = _fit(self.features, o_in, c_in, steps, params, seed=ctx.seed, n_jobs=int(gm["n_jobs"]), device=self.device)
            agg, *_ = aggregate_predictions(self.features, model, self.origins[block.inner.validation], self.all, self.weights, steps)
            ok = np.isfinite(agg)
            scores.append(mae(y_fleet[block.inner.validation][ok], agg[ok]) if ok.any() else np.inf)
            infos.append(info)
            inner_preds.append(agg)
        best = int(np.argmin(scores))
        o_tr, c_tr = training_pairs(self.origins[block.train], self.sampled, int(gm["origin_stride"]))
        model, info = _fit(self.features, o_tr, c_tr, steps, self.grid[best], seed=ctx.seed, n_jobs=int(gm["n_jobs"]), device=self.device)
        t1 = time.perf_counter()
        agg, used, dropped, same_set, per_machine = aggregate_predictions(self.features, model, self.origins[block.evaluation], self.all, self.weights, steps, keep_machine=keep_machine)
        return {
            "best": best,
            "best_params": self.grid[best],
            "grid_scores": scores,
            "inner_prediction": inner_preds[best],
            "block_prediction": agg,
            "used": used,
            "dropped": dropped,
            "same_set_observed": same_set,
            "per_machine": per_machine,
            "n_examples_train": info["n_examples"],
            "n_examples_dropped": info["n_dropped_missing"],
            "n_distinct_origins_train": int(len(np.unique(o_tr))),
            "fit_seconds_total": float(sum(i["fit_seconds"] for i in infos) + info["fit_seconds"]),
            "inference_seconds": time.perf_counter() - t1,
            "wall_seconds": time.perf_counter() - t0,
        }

    def path(self, step: int, params: dict) -> np.ndarray:
        """Previsão por máquina de um passo não pontuado, com o hiperparâmetro herdado."""
        o_tr, c_tr = training_pairs(self.origins[self.block.train], self.sampled, int(self.gm["origin_stride"]))
        model, _ = _fit(self.features, o_tr, c_tr, step, params, seed=self.ctx.seed, n_jobs=int(self.gm["n_jobs"]), device=self.device)
        _, _, _, _, per_machine = aggregate_predictions(self.features, model, self.origins[self.block.evaluation], self.all, self.weights, step, keep_machine=True)
        return per_machine


def _block_setup(ctx, block):
    panel = ctx.panel()
    last_bin = int(ctx.series["bin"].iloc[block.train_range[1]])
    stats = train_statistics(panel, last_bin=last_bin, step_seconds=ctx.step_seconds, min_coverage=float(ctx.cfg["panel"]["min_train_coverage"]))
    features = build_machine_features(panel, ctx.specs["cpu"], stats, ctx.domains(), ctx.series["cpu"].to_numpy(), step_seconds=ctx.step_seconds)
    return stats, features


def run_tcn(ctx, block, features, sampled_cols, all_cols, weights, gm, device, steps, seed):
    from energy_prediction.models.neural.global_tcn import WindowSource, predict_global_tcn, train_global_tcn

    tcn = gm["tcn"]
    origins = ctx.table["origin_index"].to_numpy()
    cpu = features.cpu
    scale = float(np.nanstd(cpu[: block.train_range[1] + 1][:, sampled_cols]))
    source = WindowSource(cpu, ctx.panel().bins, ctx.step_seconds, int(tcn["window"]), scale)

    def pairs(positions):
        o, c = training_pairs(origins[positions], sampled_cols, 1)
        y = features.target(o, c, steps)
        ok = source.eligible(o, c) & np.isfinite(y)
        return o[ok], c[ok], y[ok]

    inner = train_global_tcn(source, pairs(block.inner.train), pairs(block.inner.validation), tcn, device=device, seed=seed)
    final = train_global_tcn(source, pairs(block.train), None, tcn, device=device, seed=seed, epochs=inner["best_epoch"])
    eval_o = origins[block.evaluation]
    o = np.repeat(eval_o, len(all_cols))
    c = np.tile(all_cols, len(eval_o))
    ok = source.eligible(o, c)
    pred = np.full(len(o), np.nan)
    pred[ok] = predict_global_tcn(final["model"], source, o[ok], c[ok], device=device)
    pred = pred.reshape(len(eval_o), len(all_cols))
    w = weights[all_cols][None, :] * np.isfinite(pred)
    agg = np.where(w.sum(axis=1) > 0, np.nansum(pred * weights[all_cols][None, :], axis=1) / np.where(w.sum(axis=1) > 0, w.sum(axis=1), 1), np.nan)
    return agg, {"best_epoch": inner["best_epoch"], "train_seconds": inner["train_seconds"] + final["train_seconds"], "peak_cuda_bytes": max(inner["peak_cuda_bytes"], final["peak_cuda_bytes"]), "history": inner["history"]}


def run(ctx, out: Path, *, run_dir=None, device: dict | None = None, **_runtime) -> dict:
    gm = ctx.cfg["families"]["global_model"]
    if not gm["enabled"]:
        return {"executed": False, "excluded": [{"item": "modelo global por máquina", "reason": "desligado na configuração"}]}
    started = time.perf_counter()
    effective = (device or {}).get("effective", "cpu")
    xdev = xgb_device(str(gm["device"]), effective)
    panel = ctx.panel()
    weights = ctx.panel_weights()
    all_cols = np.arange(panel.n_machines)
    column_of = panel.column_of()
    frames, cost, samples, selections, scale_rows, pop_rows = [], [], [], [], [], []
    machine_dir = out / "machine_predictions"
    machine_dir.mkdir()
    fits: dict = {}
    for block in ctx.blocks:
        stats, features = _block_setup(ctx, block)
        sampled_ids = sample_machines(stats, int(gm["n_machines"]), seed=ctx.seed)
        sampled = np.array([column_of[m] for m in sampled_ids])
        de = design_effect(features.cpu[: block.train_range[1] + 1][:, sampled])
        fitter = BlockFit(ctx, block, features, sampled, all_cols, weights, gm, xdev)
        per_step = {}
        for steps in ctx.steps:
            r = fitter.run(steps, keep_machine=True)
            per_step[steps] = r["per_machine"]
            fits[(block.label, steps)] = r
            frames.append(_frame(ctx, block, block.inner.validation, phase=INNER, model="global_xgb", variant="main", steps=steps, y_pred=r["inner_prediction"]))
            frames.append(_frame(ctx, block, block.evaluation, phase=BLOCK, model="global_xgb", variant="main", steps=steps, y_pred=r["block_prediction"], machines_used=r["used"], machines_dropped=r["dropped"]))
            pop_rows.append(pd.DataFrame({"block": block.label, "horizon_steps": steps, "origin_index": ctx.table["origin_index"].to_numpy()[block.evaluation], "official_target": ctx.table[target_column(steps)].to_numpy()[block.evaluation], "same_set_observed": r["same_set_observed"]}))
            selections.append({"block": block.label, "horizon_steps": int(steps), "best_params": r["best_params"], "grid_scores": r["grid_scores"], "n_machines": len(sampled)})
            cost.append({"model": "global_xgb", "variant": "main", "block": block.label, "horizon_steps": int(steps), "n_machines_sampled": len(sampled), "n_examples_train": r["n_examples_train"], "n_examples_dropped_missing": r["n_examples_dropped"], "fit_seconds_total": r["fit_seconds_total"], "inference_seconds": r["inference_seconds"], "wall_seconds": r["wall_seconds"], "xgb_device": xdev, "peak_rss_mb": _rss_mb()})
        samples.append({"block": block.label, "n_machines_sampled": int(len(sampled)), "n_examples_train_h1": fits[(block.label, ctx.steps[0])]["n_examples_train"], "n_distinct_origins_train": fits[(block.label, ctx.steps[0])]["n_distinct_origins_train"], "origin_stride": int(gm["origin_stride"]), **{f"design_{k}": v for k, v in de.items()}, "effective_sample_estimate": de["effective_per_instant"] * fits[(block.label, ctx.steps[0])]["n_distinct_origins_train"], "note": "n efetivo = efeito de desenho por instante × origens distintas; sem correção da autocorrelação temporal"})
        for step in ctx.path_steps:
            if step not in per_step:
                per_step[step] = fitter.path(step, fits[(block.label, inherited_step(step, ctx.steps))]["best_params"])
        stacked = np.stack([per_step[s] for s in ctx.path_steps], axis=2)
        np.savez_compressed(machine_dir / f"{block.label}.npz", origins=ctx.table["origin_index"].to_numpy()[block.evaluation], machines=np.array(panel.machines), steps=np.array(ctx.path_steps), predictions=stacked)
        if block.day == int(gm["scale_day"]):
            for m in gm["scale_machines"]:
                m = int(m)
                ids = sample_machines(stats, m, seed=ctx.seed)
                cols = np.array([column_of[x] for x in ids])
                for steps in ctx.steps:
                    if len(cols) == len(sampled) and set(cols) == set(sampled):
                        r = fits[(block.label, steps)]
                    else:
                        r = BlockFit(ctx, block, features, cols, all_cols, weights, gm, xdev).run(steps, keep_machine=False)
                    y = ctx.table[target_column(steps)].to_numpy()[block.evaluation]
                    ok = np.isfinite(r["block_prediction"])
                    scale_rows.append({"block": block.label, "n_machines": len(cols), "horizon_steps": int(steps), "mae": mae(y[ok], r["block_prediction"][ok]), "inner_mae": float(min(r["grid_scores"])), "fit_seconds_total": r["fit_seconds_total"], "n_examples_train": r["n_examples_train"], "peak_rss_mb": _rss_mb(), "note": "diagnóstico de escala num único bloco; descritivo"})
    predictions = pd.concat(frames, ignore_index=True)

    # Portão da TCN global: MAE interno agregado médio dos dias de desenvolvimento.
    base = read_family_csv(run_dir, "baselines", "predictions")
    base = base[base["model"] == "xgboost_v2"]
    gate_rows = []
    for steps in ctx.steps:
        g = predictions[(predictions["phase"] == INNER) & (predictions["role"] == "development") & (predictions["horizon_steps"] == steps)]
        b = base[(base["phase"] == INNER) & (base["role"] == "development") & (base["horizon_steps"] == steps)]
        gate_rows.append({"horizon_steps": int(steps), "global_inner_mae": float((g["y_true"] - g["y_pred"]).abs().mean()), "direct_inner_mae": float((b["y_true"] - b["y_pred"]).abs().mean())})
    gate = pd.DataFrame(gate_rows)
    gate["ratio"] = gate["global_inner_mae"] / gate["direct_inner_mae"]
    passed = bool((gate["ratio"] <= float(gm["tcn"]["gate_ratio"])).any())
    gates = [{"gate": "tcn_global", "threshold_ratio": float(gm["tcn"]["gate_ratio"]), "observed_ratios": gate["ratio"].round(6).tolist(), "passed": passed}]
    excluded, tcn_profile = [], []
    if not gm["tcn"]["enabled"]:
        excluded.append({"item": "TCN global", "reason": "desligada na configuração"})
    elif not passed:
        excluded.append({"item": "TCN global", "reason": f"portão não cumprido: razões {gate['ratio'].round(4).tolist()} > {gm['tcn']['gate_ratio']}"})
    elif not torch_installed():
        excluded.append({"item": "TCN global", "reason": "PyTorch não instalado"})
    else:
        tcn_frames = []
        for block in ctx.blocks:
            stats, features = _block_setup(ctx, block)
            sampled = np.array([column_of[m] for m in sample_machines(stats, int(gm["n_machines"]), seed=ctx.seed)])
            for steps in ctx.steps:
                for seed in gm["tcn"]["seeds"]:
                    agg, info = run_tcn(ctx, block, features, sampled, all_cols, weights, gm, effective, steps, int(seed))
                    tcn_frames.append(_frame(ctx, block, block.evaluation, phase=BLOCK, model="global_tcn", variant="window", steps=steps, y_pred=agg, seed=int(seed)))
                    tcn_profile.append({"block": block.label, "horizon_steps": int(steps), "seed": int(seed), "device": effective, **{k: v for k, v in info.items() if k != "history"}})
        tcn = pd.concat(tcn_frames, ignore_index=True)
        tcn = tcn.groupby(["model", "variant", "horizon_steps", "block", "day", "role", "phase", "origin_index"], as_index=False).agg(timestamp=("timestamp", "first"), cpu_t=("cpu_t", "first"), y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        predictions = pd.concat([predictions, tcn], ignore_index=True)

    rows, per_day = [], []
    for steps in ctx.steps:
        ref = loss_frame(base, model="xgboost_v2", variant="A", steps=steps)
        r, d = compare(loss_frame(predictions, model="global_xgb", variant="main", steps=steps), ref, ctx.cfg, {"family": FAMILY, "hypothesis": "H2", "scope": "primary", "horizon_steps": int(steps), "model_a": "global_xgb", "variant_a": "main", "model_b": "xgboost_v2", "variant_b": "A", "metric": "erro absoluto"})
        rows += r
        per_day += d
        if (predictions["model"] == "global_tcn").any():
            r, d = compare(loss_frame(predictions, model="global_tcn", variant="window", steps=steps), ref, ctx.cfg, {"family": FAMILY, "hypothesis": "H2", "scope": "exploratory", "horizon_steps": int(steps), "model_a": "global_tcn", "variant_a": "window", "model_b": "xgboost_v2", "variant_b": "A", "metric": "erro absoluto (média das sementes)"})
            rows += r
            per_day += d
    comparisons = finalize_comparisons(rows, alpha=float(ctx.cfg["statistics"]["alpha"]))
    cost_table = pd.DataFrame(cost)
    write_tables(
        out,
        predictions=predictions,
        metrics=metrics_table(pd.concat([predictions, base], ignore_index=True)),
        comparisons=comparisons,
        comparisons_by_day=pd.DataFrame(per_day),
        inherited_test=inherited_rows(pd.concat([predictions, base], ignore_index=True), [(("global_xgb", "main"), ("xgboost_v2", "A")), (("global_tcn", "window"), ("xgboost_v2", "A"))], ctx.steps),
        cost=cost_table,
        scale=pd.DataFrame(scale_rows),
        sample_summary=pd.DataFrame(samples),
        population_term=pd.concat(pop_rows, ignore_index=True).assign(difference=lambda f: f["official_target"] - f["same_set_observed"]),
        gate=gate,
        tcn_profile=pd.DataFrame(tcn_profile),
        selections=selections,
        family_info={
            "n_machines": int(gm["n_machines"]),
            "origin_stride": int(gm["origin_stride"]),
            "xgb_device": xdev,
            "aggregation": "média ponderada por cpu_num das máquinas presentes na origem com todos os atributos",
            "machine_predictions": "machine_predictions/<bloco>.npz (origens × máquinas × passos 1–6)",
            "statement": "O número de janelas não é o número de observações independentes; ver sample_summary.csv.",
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    primary = comparisons[(comparisons["scope"] == "primary") & comparisons["primary_block"]]
    return {"executed": True, "primary_decisions": {f"H2/h{r.horizon_steps}": r.decision for r in primary.itertuples()}, "tcn_gate_passed": passed, "gates": gates, "excluded": excluded, "elapsed_seconds": time.perf_counter() - started}
