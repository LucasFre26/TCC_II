"""Família de energia refinada (H6): rotas média, por grupo e por máquina, e decomposição do erro.

Energia de TI modelada (estimativa paramétrica em servidores substitutos do SPECpower_ssj2008),
sem medição elétrica e sem PUE. Em cada bloco e origem, o conjunto fixo F_t são as máquinas
presentes em t com CPU observada em t+1..t+6. Referência = rota por máquina com CPU observada.
Rotas previstas: média (todos os modelos com trajetória), por grupo (candidato hierárquico, com as
previsões reconciliadas dos grupos) e por máquina (modelo global, se executado; máquina de F_t sem
previsão usa a própria CPU na origem, regra declarada e contada). A diferença total é decomposta em
aproximação por agregação e erro de previsão. Cenários: cada substituto homogêneo e um cenário
heterogêneo sintético; r ∈ {1,0; 1,4; 1,8}. Os arquivos de energia da V3 são lidos só por hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.energy.heterogeneous import SCENARIO, assign_substitutes
from energy_prediction.energy.routes import decompose, fixed_sets, group_means, grouped_route_power, horizon_energy_wh, machine_route_power, observed_paths
from energy_prediction.energy.substitute import load_substitutes
from energy_prediction.evaluation.error_analysis import block_indices
from energy_prediction.experiments.v3_context import resolve_path
from energy_prediction.experiments.v3p2_families.common import compare, finalize_comparisons, read_family_csv, write_tables
from energy_prediction.features.cpu_features import target_column

FAMILY = "energy"
QUANTITY = "energia de TI modelada (Wh), estimativa paramétrica em servidor substituto; não é consumo medido"
PRIMARY_R = 1.4
BOUND_NOTE = "Limites por passo P(Q05) e P(Q95) valem por passo (P é monotônica em [0, 1] para r ≤ 2); somar limites por passo não produz um limite da energia somada."


def _tree_hash(directory: Path | None) -> str | None:
    if directory is None or not directory.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _path_matrix(paths: pd.DataFrame, block: str, origins: np.ndarray, steps: list[int], value: str = "y_pred") -> np.ndarray:
    part = paths[paths["block"] == block].pivot_table(index="origin_index", columns="step", values=value)
    return part.reindex(index=origins, columns=steps).to_numpy(dtype=float)


def run(ctx, out: Path, *, run_dir=None, **_runtime) -> dict:
    fam = ctx.cfg["families"]["energy"]
    record, entries, fans = load_substitutes(resolve_path(ctx.root, fam["substitutes"]))
    primary_sub = str(record["primary"])
    v3_energy = resolve_path(ctx.root, ctx.cfg["reference"]["v3_run"])
    v3_energy = None if v3_energy is None else v3_energy / "energy"
    v3_hash_before = _tree_hash(v3_energy)
    panel = ctx.panel()
    cpu = panel.values["cpu"]
    weights = ctx.panel_weights()
    machines = panel.machines
    H = max(ctx.path_steps)
    steps_axis = list(ctx.path_steps)
    exponents = [float(r) for r in fam["exponents"]]
    scenarios = {sid: np.full(len(machines), sid, dtype=object) for sid in fans}
    excluded = []
    if fam["heterogeneous"]:
        mapping = assign_substitutes(machines, list(fans))
        scenarios[SCENARIO] = np.array([mapping[m] for m in machines], dtype=object)
    proportions = {name: {sid: float(np.mean(arr == sid)) for sid in fans} for name, arr in scenarios.items()}

    base_paths = read_family_csv(run_dir, "baselines", "paths")
    hier_paths = read_family_csv(run_dir, "hierarchical", "paths")
    group_paths = read_family_csv(run_dir, "hierarchical", "group_paths")
    assignments = read_family_csv(run_dir, "hierarchical", "assignments")
    cal_paths = read_family_csv(run_dir, "calibration", "paths")
    global_dir = run_dir.family_path("global_model") / "machine_predictions"
    models = [m for m in fam["models"]]
    if "global" in models and not global_dir.is_dir():
        models.remove("global")
        excluded.append({"item": "rota por máquina prevista (modelo global)", "reason": "família global_model não executada ou desligada nesta corrida"})
    aggregate_paths = {m: base_paths[base_paths["model"] == {"xgboost_v2": "xgboost_v2"}.get(m, m)] for m in models if m in ("persistence", "xgboost_v2", "hybrid")}
    if "hierarchical" in models:
        aggregate_paths["hierarchical"] = hier_paths

    rows, decomposition_frames, bound_rows, population_rows, clip_rows, fallback_rows = [], [], [], [], [], []
    origins_all = ctx.table["origin_index"].to_numpy()
    for block in ctx.blocks:
        O = origins_all[block.evaluation]
        F = fixed_sets(cpu, O, H)
        U = observed_paths(cpu, O, H)
        n_F = F.sum(axis=1)
        zero = np.zeros(len(machines), dtype=object)
        mean_obs = group_means(U, F, weights, zero, [0])
        official = np.column_stack([ctx.series["cpu"].to_numpy()[O + k] for k in range(H + 1)])
        population_rows.append(pd.DataFrame({"block": block.label, "origin_index": O, "n_fixed_set": n_F, "fraction_fixed_of_present": n_F / np.maximum(np.isfinite(cpu[O]).sum(axis=1), 1), "mean_population_term_pp": np.nanmean(mean_obs[:, :, 0] - official, axis=1)}))
        gp = group_paths[group_paths["block"] == block.label]
        crit, kk = gp["criterion"].iloc[0], str(gp["k"].iloc[0])
        assign = assignments[(assignments["block"] == block.label) & (assignments["criterion"] == crit) & (assignments["k"].astype(str) == kk)].set_index("machine_id")["group"].reindex(machines).to_numpy(dtype=object)
        labels = sorted(set(assign))
        groupings = {"group_candidate": (assign, labels)}
        for k in fam["group_k_sensitivity"]:
            sel = assignments[(assignments["block"] == block.label) & (assignments["criterion"] == "mean") & (assignments["k"].astype(str) == str(k))]
            if len(sel):
                a = sel.set_index("machine_id")["group"].reindex(machines).to_numpy(dtype=object)
                groupings[f"group_mean_k{k}"] = (a, sorted(set(a)))
        group_obs = {name: group_means(U, F, weights, a, lab) for name, (a, lab) in groupings.items()}

        predicted_mean = {}
        for model, paths in aggregate_paths.items():
            Y = _path_matrix(paths, block.label, O, steps_axis)
            X = np.concatenate([mean_obs[:, :1, 0], Y], axis=1)[:, :, None]
            predicted_mean[model] = X
        X_group = None
        if "hierarchical" in models:
            X_group = np.full((len(O), H + 1, len(labels)), np.nan)
            X_group[:, 0, :] = group_obs["group_candidate"][:, 0, :]
            for j, g in enumerate(labels):
                part = gp[gp["group"] == g]
                X_group[:, 1:, j] = part.pivot_table(index="origin_index", columns="step", values="y_pred_group").reindex(index=O, columns=steps_axis).to_numpy()
        U_global = None
        if "global" in models:
            data = np.load(global_dir / f"{block.label}.npz", allow_pickle=False)
            if list(data["origins"]) != list(O) or list(data["machines"]) != list(machines):
                raise ValueError(f"Previsões por máquina de {block.label} não alinhadas com o bloco.")
            pred = data["predictions"].astype(float)  # (origens, máquinas, passos)
            U_global = np.concatenate([U[:, :1, :], np.transpose(pred, (0, 2, 1))], axis=1)
            missing = F[:, None, :] & ~np.isfinite(U_global)
            fallback_rows.append({"block": block.label, "machine_step_cells_fallback": int(missing.sum()), "fraction": float(missing.sum() / max(F.sum() * (H + 1), 1))})
            U_global = np.where(missing, U[:, :1, :], U_global)

        for scenario, subs in scenarios.items():
            scenario_fans = fans if scenario == SCENARIO else {scenario: fans[scenario]}
            for r in exponents:
                ref_p, _ = machine_route_power(U, F, subs, scenario_fans, r)
                mean_obs_p, _ = grouped_route_power(mean_obs, F, zero, [0], subs, scenario_fans, r)
                obs_routes = {"mean": mean_obs_p}
                for name, (a, lab) in groupings.items():
                    obs_routes[name], _ = grouped_route_power(group_obs[name], F, a, lab, subs, scenario_fans, r)
                pred_routes = {}
                for model, X in predicted_mean.items():
                    pred_routes[(model, "mean")], c = grouped_route_power(X, F, zero, [0], subs, scenario_fans, r)
                    clip_rows.append({"block": block.label, "scenario": scenario, "r": r, "model": model, "route": "mean", "clipped_values": c})
                if X_group is not None:
                    pred_routes[("hierarchical", "group_candidate")], c = grouped_route_power(X_group, F, assign, labels, subs, scenario_fans, r)
                if U_global is not None:
                    pred_routes[("global", "machine")], c = machine_route_power(U_global, F, subs, scenario_fans, r)
                for h in ctx.steps:
                    ref = horizon_energy_wh(ref_p, h, ctx.step_seconds)
                    for route, p in obs_routes.items():
                        e = horizon_energy_wh(p, h, ctx.step_seconds)
                        rows.append(pd.DataFrame({"block": block.label, "day": block.day, "role": block.role, "scenario": scenario, "r": r, "horizon_steps": h, "route": route, "group_candidate": f"{crit}_k{kk}", "origin_index": O, "n_machines": n_F, "reference_wh": ref, "route_observed_wh": e, "aggregation_wh": e - ref}))
                    for (model, route), p in pred_routes.items():
                        obs_key = "mean" if route == "mean" else route
                        e_pred = horizon_energy_wh(p, h, ctx.step_seconds)
                        e_obs = horizon_energy_wh(obs_routes[obs_key] if obs_key in obs_routes else ref_p, h, ctx.step_seconds)
                        d = decompose(ref, e_obs, e_pred)
                        decomposition_frames.append(pd.DataFrame({"block": block.label, "day": block.day, "role": block.role, "scenario": scenario, "r": r, "horizon_steps": h, "model": model, "route": route, "group_candidate": f"{crit}_k{kk}", "origin_index": O, "n_machines": n_F, "reference_wh": ref, "route_observed_wh": e_obs, "route_predicted_wh": e_pred, "total_wh": d["total"], "aggregation_wh": d["aggregation"], "forecast_wh": d["forecast"], "closure_wh": d["closure"], "reference_wh_per_server": ref / np.maximum(n_F, 1), "total_wh_per_server": d["total"] / np.maximum(n_F, 1)}))
        # Limites de potência por passo (servidor substituto primário, por servidor, agregado).
        cp = cal_paths[cal_paths["block"] == block.label]
        L = cp.pivot_table(index="origin_index", columns="step", values="lower").reindex(index=O, columns=steps_axis).to_numpy()
        Uq = cp.pivot_table(index="origin_index", columns="step", values="upper").reindex(index=O, columns=steps_axis).to_numpy()
        fan = fans[primary_sub]
        for r in exponents:
            p_low = fan.power_from_u(np.clip(L / 100.0, 0, 1), r)
            p_high = fan.power_from_u(np.clip(Uq / 100.0, 0, 1), r)
            p_mean_obs = fan.power_from_u(np.clip(mean_obs[:, 1:, 0] / 100.0, 0, 1), r)
            ref_server = machine_route_power(U, F, scenarios[primary_sub], {primary_sub: fan}, r)[0][:, 1:] / np.maximum(n_F, 1)[:, None]
            ok = np.isfinite(p_low) & np.isfinite(p_high)
            for j, step in enumerate(steps_axis):
                m = ok[:, j]
                bound_rows.append({"block": block.label, "role": block.role, "r": r, "step": int(step), "substitute": primary_sub, "n": int(m.sum()), "ordered": bool(np.all(p_low[m, j] <= p_high[m, j])), "coverage_power_of_mean": float(np.mean((p_mean_obs[m, j] >= p_low[m, j]) & (p_mean_obs[m, j] <= p_high[m, j]))), "coverage_reference_per_server": float(np.mean((ref_server[m, j] >= p_low[m, j]) & (ref_server[m, j] <= p_high[m, j]))), "mean_width_w": float(np.mean(p_high[m, j] - p_low[m, j])), "note": BOUND_NOTE})

    decomposition = pd.concat(decomposition_frames, ignore_index=True)
    aggregation = pd.concat(rows, ignore_index=True)
    closure = float(np.nanmax(np.abs(decomposition["closure_wh"]))) if len(decomposition) else 0.0

    # Energia acumulada em 24 h: janelas de 30 min sem sobreposição e bootstrap por blocos de janelas.
    st = ctx.cfg["statistics"]
    hmax = max(ctx.steps)
    cumulative = []
    rng_seed = int(ctx.seed)
    for key, part in decomposition[decomposition["horizon_steps"] == hmax].groupby(["block", "role", "scenario", "r", "model", "route"]):
        start = part["origin_index"].min()
        windows = part[(part["origin_index"] - start) % hmax == 0].sort_values("origin_index")
        d = windows["total_wh"].to_numpy(dtype=float)
        if len(d) < 2:
            continue
        rng = np.random.default_rng(rng_seed)
        size = min(int(fam["bootstrap_window_steps"]) // hmax * 2 or 2, len(d))
        sums = d[block_indices(len(d), size, rng, int(st["n_boot"]))].sum(axis=1)
        lo, hi = np.percentile(sums, [2.5, 97.5])
        cumulative.append({**dict(zip(["block", "role", "scenario", "r", "model", "route"], key)), "n_windows": int(len(d)), "reference_wh": float(windows["reference_wh"].sum()), "predicted_wh": float(windows["route_predicted_wh"].sum()), "cumulative_total_wh": float(d.sum()), "cumulative_aggregation_wh": float(windows["aggregation_wh"].sum()), "cumulative_forecast_wh": float(windows["forecast_wh"].sum()), "ci_low_wh": float(lo), "ci_high_wh": float(hi), "bootstrap_block_windows": int(size)})

    comp_rows, per_day = [], []
    primary = decomposition[(decomposition["scenario"] == primary_sub) & (decomposition["r"] == PRIMARY_R) & (decomposition["role"] == "development")]

    def lf(model, route, h):
        p = primary[(primary["model"] == model) & (primary["route"] == route) & (primary["horizon_steps"] == h)]
        return pd.DataFrame({"day": p["day"].to_numpy(), "origin_index": p["origin_index"].to_numpy(), "loss": p["total_wh"].abs().to_numpy()})

    group_route = "group_candidate" if ((decomposition["model"] == "hierarchical") & (decomposition["route"] == "group_candidate")).any() else None
    for h in ctx.steps:
        if not lf("xgboost_v2", "mean", h).empty:
            ref = lf("xgboost_v2", "mean", h)
            if group_route is not None:
                r_, d_ = compare(lf("hierarchical", group_route, h), ref, ctx.cfg, {"family": FAMILY, "hypothesis": "H6", "scope": "primary", "horizon_steps": int(h), "model_a": "hierarchical", "variant_a": "group_route", "model_b": "xgboost_v2", "variant_b": "mean_route", "metric": f"|diferença total de energia| (Wh), r = {PRIMARY_R}, substituto {primary_sub}"})
                comp_rows += r_
                per_day += d_
            for (model, route) in decomposition[["model", "route"]].drop_duplicates().itertuples(index=False):
                if (model, route) == ("xgboost_v2", "mean") or (model == "hierarchical" and route == group_route):
                    continue
                cand = lf(model, route, h)
                if cand.empty:
                    continue
                r_, _ = compare(cand, ref, ctx.cfg, {"family": FAMILY, "hypothesis": "H6", "scope": "exploratory", "horizon_steps": int(h), "model_a": model, "variant_a": f"{route}_route", "model_b": "xgboost_v2", "variant_b": "mean_route", "metric": "|diferença total de energia| (Wh)"})
                comp_rows += r_
    comparisons = finalize_comparisons(comp_rows, alpha=float(st["alpha"]))

    summary = decomposition.groupby(["role", "scenario", "r", "horizon_steps", "model", "route"]).agg(
        n=("total_wh", "size"), mean_reference_wh=("reference_wh", "mean"), mean_abs_total_wh=("total_wh", lambda s: float(np.mean(np.abs(s)))), mean_total_wh=("total_wh", "mean"),
        mean_aggregation_wh=("aggregation_wh", "mean"), mean_abs_forecast_wh=("forecast_wh", lambda s: float(np.mean(np.abs(s)))), mean_forecast_wh=("forecast_wh", "mean"),
        mean_relative_abs_total=("total_wh", "size"),
    ).reset_index()
    rel = decomposition.assign(rel=np.abs(decomposition["total_wh"]) / decomposition["reference_wh"]).groupby(["role", "scenario", "r", "horizon_steps", "model", "route"])["rel"].mean().reset_index(drop=True)
    summary["mean_relative_abs_total"] = rel.to_numpy()
    agg_summary = aggregation.groupby(["role", "scenario", "r", "horizon_steps", "route"]).agg(mean_reference_wh=("reference_wh", "mean"), mean_aggregation_wh=("aggregation_wh", "mean"), mean_relative_aggregation=("aggregation_wh", "mean")).reset_index()
    agg_summary["mean_relative_aggregation"] = agg_summary["mean_aggregation_wh"] / agg_summary["mean_reference_wh"]
    v3_hash_after = _tree_hash(v3_energy)
    if v3_hash_before != v3_hash_after:
        raise RuntimeError("Os arquivos de energia da V3 mudaram durante a família de energia.")
    synthetic_label = {"scenario": SCENARIO, "label": "hardware sintético: atribuição por hash; não é o hardware do cluster do Alibaba", "proportions": proportions.get(SCENARIO)}
    write_tables(
        out,
        decomposition=decomposition.drop(columns=["closure_wh"]),
        aggregation_error=aggregation,
        summary=summary,
        aggregation_summary=agg_summary,
        cumulative_24h=pd.DataFrame(cumulative),
        power_bounds=pd.DataFrame(bound_rows),
        population_term=pd.concat(population_rows, ignore_index=True),
        clipped_predictions=pd.DataFrame(clip_rows),
        machine_fallback=pd.DataFrame(fallback_rows),
        comparisons=comparisons,
        comparisons_by_day=pd.DataFrame(per_day),
        energy_info={
            "quantity": QUANTITY,
            "substitutes": entries,
            "primary_substitute": primary_sub,
            "primary_r": PRIMARY_R,
            "exponents": exponents,
            "scenarios": list(scenarios),
            "synthetic_scenario": synthetic_label,
            "max_abs_decomposition_closure_wh": closure,
            "fixed_set_rule": "máquinas presentes na origem com CPU observada nos passos 1 a 6",
            "machine_route_fallback": "máquina de F_t sem previsão do modelo global usa a própria CPU na origem",
            "power_bounds_note": BOUND_NOTE,
            "v3_energy_sha256_before": v3_hash_before,
            "v3_energy_sha256_after": v3_hash_after,
            "statement": "Estimativa paramétrica em servidores substitutos; sem medição elétrica; a eficiência da instalação não foi calculada.",
        },
    )
    if SCENARIO in scenarios:
        decomposition[decomposition["scenario"] == SCENARIO].assign(hardware="sintetico").to_csv(out / f"decomposition_{SCENARIO}.csv", index=False)
    prim = comparisons[(comparisons["scope"] == "primary") & comparisons["primary_block"]] if not comparisons.empty else comparisons
    return {"primary_decisions": {f"H6/h{r.horizon_steps}": r.decision for r in prim.itertuples()}, "max_abs_closure_wh": closure, "excluded": excluded}
