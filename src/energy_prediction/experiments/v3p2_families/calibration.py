"""Família de calibração adaptativa (H5): quantis de CPU e limites conformais cronológicos.

XGBoost quantílico (configuração A) para cada quantil, escolhido pela pinball da validação
interna e reajustado no treino do bloco; os quantis previstos são ordenados por origem (cruzamentos
contados). Limites: superior sobre Q95 e inferior sobre Q05 (bilateral nominal de 90%), com os
métodos estático, janela móvel, ACI com retorno atrasado e, se couber no orçamento, SPCI. O
método da comparação primária é fixado no pré-registro. Trajetórias de Q05 e Q95 (passos 1–6,
herdando hiperparâmetros) calibradas pelo método primário alimentam os limites de potência.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.evaluation.adaptive_conformal import aci_offsets, interval_score, rolling_offsets, sort_quantiles, spci_offsets, static_offsets
from energy_prediction.evaluation.day_forward import evaluate_block
from energy_prediction.evaluation.probabilistic import pinball_loss
from energy_prediction.experiments.v3p2_families.common import compare, finalize_comparisons, write_tables
from energy_prediction.experiments.v3p2_families.tabular import inherited_step, xgb_grid
from energy_prediction.features.cpu_features import target_column
from energy_prediction.models.quantile_xgb import QuantileXGBoost
from energy_prediction.models.regime import classify, regime_signals, thresholds

FAMILY = "calibration"
STATEMENT = "Coberturas empíricas no protocolo por dia; sob dependência temporal nenhum método tem garantia teórica de cobertura."


def quantile_fit_predict(ctx, columns, steps, grid, xgb_cfg, quantile):
    X = ctx.table[columns].to_numpy(dtype=float)
    y = ctx.table[target_column(steps)].to_numpy(dtype=float)

    def fp(g, train, predict, **_):
        model = QuantileXGBoost(quantile, grid[g], seed=ctx.seed, n_jobs=int(xgb_cfg["n_jobs"]), fixed=xgb_cfg.get("fixed"))
        model.fit(X[train], y[train])
        return model.predict(X[predict]), {}

    return fp


def method_specs(cal: dict) -> list[dict]:
    specs = [{"method": "static", "name": "static"}]
    if "rolling" in cal["methods"]:
        specs += [{"method": "rolling", "name": f"rolling_n{int(n)}", "window": int(n)} for n in cal["rolling_windows"]]
    if "aci" in cal["methods"]:
        specs += [{"method": "aci", "name": f"aci_g{float(g):g}", "gamma": float(g)} for g in cal["aci_gammas"]]
    if "spci" in cal["methods"] and cal["spci"]["enabled"]:
        specs.append({"method": "spci", "name": "spci"})
    return specs


def primary_name(cal: dict) -> str:
    method = cal["primary_method"]
    if method == "aci":
        return f"aci_g{float(cal['aci_gammas'][0]):g}"
    if method == "rolling":
        return f"rolling_n{int(cal['rolling_windows'][0])}"
    return method


def offsets_for(spec, initial, scores, origins, steps, alpha, cal, seed):
    if spec["method"] == "static":
        return static_offsets(initial, len(scores), alpha), None
    if spec["method"] == "rolling":
        return rolling_offsets(initial, scores, origins, steps, alpha, spec["window"]), None
    if spec["method"] == "aci":
        return aci_offsets(initial, scores, origins, steps, alpha, spec["gamma"])
    sp = cal["spci"]
    return spci_offsets(initial, scores, origins, steps, alpha, lags=int(sp["lags"]), n_estimators=int(sp["n_estimators"]), refit_every=12, seed=seed), None


def run(ctx, out: Path, **_runtime) -> dict:
    cal = ctx.cfg["families"]["calibration"]
    xgb_cfg = ctx.cfg["families"]["baselines"]["xgboost"]
    grid = xgb_grid(xgb_cfg)
    columns = ctx.columns["A"]
    quantiles = [float(q) for q in cal["quantiles"]]
    low_q, high_q = quantiles[0], quantiles[-1]
    alpha_low, alpha_high = low_q, 1.0 - high_q
    specs = method_specs(cal)
    primary = primary_name(cal)
    reg = ctx.cfg["families"]["regimes"]
    slope, vol = regime_signals(ctx.series["cpu"].to_numpy(), slope_steps=int(reg["slope_steps"]), volatility_steps=int(reg["volatility_steps"]))
    origins_all = ctx.table["origin_index"].to_numpy()
    rows, base_rows, crossing, timing, selections, path_rows = [], [], [], [], [], []
    excluded, gates = [], []
    spci_budget = float(cal["spci"]["budget_minutes"]) * 60.0
    spci_spent = 0.0
    for block in ctx.blocks:
        limits = thresholds(slope, vol, origins_all[block.train], volatility_quantile=float(reg["volatility_quantile"]), rise_quantile=float(reg["rise_quantile"]), fall_quantile=float(reg["fall_quantile"]))
        regime = classify(slope, vol, origins_all[block.evaluation], limits)
        peak_thr = float(np.nanquantile(ctx.series["cpu"].to_numpy()[: block.train_range[1] + 1], float(ctx.cfg["families"]["peaks"]["peak_quantile"])))
        best_by: dict = {}
        for steps in [*ctx.steps, *[s for s in ctx.path_steps if s not in ctx.steps]]:  # pontuados primeiro: os demais herdam
            scored = steps in ctx.steps
            y = ctx.table[target_column(steps)].to_numpy(dtype=float)
            fit_qs = quantiles if scored else [low_q, high_q]
            inner_pred, block_pred = {}, {}
            for q in fit_qs:
                if scored:
                    result = evaluate_block(len(grid), quantile_fit_predict(ctx, columns, steps, grid, xgb_cfg, q), y, block, metric=lambda yt, yp, q=q: pinball_loss(yt, yp, q))
                    best_by[(q, steps)] = result.best
                    inner_pred[q], block_pred[q] = result.inner_prediction, result.block_prediction
                    selections.append({"block": block.label, "horizon_steps": int(steps), "quantile": q, "best_params": grid[result.best], "grid_scores": result.grid_scores})
                else:
                    params = [grid[best_by[(q, inherited_step(steps, ctx.steps))]]]
                    fp = quantile_fit_predict(ctx, columns, steps, params, xgb_cfg, q)
                    inner_pred[q], _ = fp(0, block.inner.train, block.inner.validation)
                    block_pred[q], _ = fp(0, block.train, block.evaluation)
            order = [q for q in quantiles if q in fit_qs]
            Qi, ci = sort_quantiles(np.column_stack([inner_pred[q] for q in order]))
            Qb, cb = sort_quantiles(np.column_stack([block_pred[q] for q in order]))
            crossing.append({"block": block.label, "horizon_steps": int(steps), "crossed_inner": ci, "crossed_block": cb, "n_block": int(len(Qb))})
            col = {q: j for j, q in enumerate(order)}
            y_in, y_bl = y[block.inner.validation], y[block.evaluation]
            o_bl = origins_all[block.evaluation]
            up_init, up_scores = y_in - Qi[:, col[high_q]], y_bl - Qb[:, col[high_q]]
            lo_init, lo_scores = Qi[:, col[low_q]] - y_in, Qb[:, col[low_q]] - y_bl
            if scored:
                for q in order:
                    base_rows.append({"block": block.label, "role": block.role, "horizon_steps": int(steps), "quantile": q, "pinball": pinball_loss(y_bl, Qb[:, col[q]], q), "raw_coverage": float(np.mean(y_bl <= Qb[:, col[q]]))})
            for spec in specs:
                if not scored and spec["name"] != primary:
                    continue
                if spec["method"] == "spci":
                    if spci_budget <= 0 or spci_spent > spci_budget:
                        if not any(e["item"] == "SPCI" for e in excluded):
                            excluded.append({"item": "SPCI", "reason": f"orçamento de {cal['spci']['budget_minutes']} min esgotado ({spci_spent / 60:.1f} min)"})
                        continue
                t0 = time.perf_counter()
                d_up, level_up = offsets_for(spec, up_init, up_scores, o_bl, steps, alpha_high, cal, ctx.seed)
                d_lo, level_lo = offsets_for(spec, lo_init, lo_scores, o_bl, steps, alpha_low, cal, ctx.seed)
                elapsed = time.perf_counter() - t0
                if spec["method"] == "spci":
                    spci_spent += elapsed
                timing.append({"block": block.label, "horizon_steps": int(steps), "method": spec["name"], "seconds": elapsed})
                upper, lower = Qb[:, col[high_q]] + d_up, Qb[:, col[low_q]] - d_lo
                if not scored:
                    path_rows.append(pd.DataFrame({"block": block.label, "day": block.day, "role": block.role, "step": int(steps), "origin_index": o_bl, "lower": lower, "upper": upper, "method": spec["name"]}))
                    continue
                if spec["name"] == primary:
                    path_rows.append(pd.DataFrame({"block": block.label, "day": block.day, "role": block.role, "step": int(steps), "origin_index": o_bl, "lower": lower, "upper": upper, "method": spec["name"]}))
                rows.append(pd.DataFrame({
                    "block": block.label, "day": block.day, "role": block.role, "horizon_steps": int(steps), "method": spec["name"], "origin_index": o_bl,
                    "y_true": y_bl, "q_low_raw": Qb[:, col[low_q]], "q50": Qb[:, col[0.5]] if 0.5 in col else np.nan, "q_high_raw": Qb[:, col[high_q]],
                    "lower": lower, "upper": upper, "covered_upper": y_bl <= upper, "covered_bilateral": (y_bl >= lower) & (y_bl <= upper),
                    "width": upper - lower, "interval_score": interval_score(y_bl, lower, upper, alpha_low + alpha_high),
                    "infinite_bound": ~np.isfinite(upper) | ~np.isfinite(lower), "aci_level_upper": level_up if level_up is not None else np.nan,
                    "is_peak": y_bl >= peak_thr, "regime": regime,
                }))
    if spci_budget <= 0 or spci_spent > spci_budget:
        gates.append({"gate": "spci_budget", "threshold_seconds": spci_budget, "observed_seconds": spci_spent, "passed": False})
    else:
        gates.append({"gate": "spci_budget", "threshold_seconds": spci_budget, "observed_seconds": spci_spent, "passed": True})
    limits_frame = pd.concat(rows, ignore_index=True)
    window = int(round(float(cal["window_hours"]) * 3600.0 / ctx.step_seconds))

    def summarize(part):
        finite = np.isfinite(part["width"])
        return pd.Series({
            "n": len(part), "coverage_upper": part["covered_upper"].mean(), "coverage_bilateral": part["covered_bilateral"].mean(),
            "mean_width": part.loc[finite, "width"].mean(), "mean_interval_score": part.loc[finite, "interval_score"].mean(),
            "pinball_q50": pinball_loss(part["y_true"], part["q50"], 0.5) if part["q50"].notna().all() else np.nan,
            "coverage_upper_peaks": part.loc[part["is_peak"], "covered_upper"].mean() if part["is_peak"].any() else np.nan,
            "n_peaks": int(part["is_peak"].sum()), "n_infinite": int(part["infinite_bound"].sum()),
        })

    summary = limits_frame.groupby(["method", "horizon_steps", "block", "role"]).apply(summarize, include_groups=False).reset_index()
    pooled = limits_frame[limits_frame["role"] == "development"].groupby(["method", "horizon_steps"]).apply(summarize, include_groups=False).reset_index().assign(block="development_pooled", role="development")
    by_regime = limits_frame.groupby(["method", "horizon_steps", "role", "regime"]).apply(summarize, include_groups=False).reset_index()
    win_rows = []
    for key, part in limits_frame.groupby(["method", "horizon_steps", "block", "role"]):
        part = part.sort_values("origin_index")
        cov = [part["covered_upper"].iloc[i : i + window].mean() for i in range(0, len(part), window)]
        win_rows.append({**dict(zip(["method", "horizon_steps", "block", "role"], key)), "window_steps": window, "min_window_coverage_upper": float(np.min(cov)), "max_window_coverage_upper": float(np.max(cov)), "std_window_coverage_upper": float(np.std(cov)), "n_windows": len(cov)})

    comp_rows, per_day = [], []
    alpha = float(ctx.cfg["statistics"]["alpha"])
    dev = limits_frame[limits_frame["role"] == "development"]
    for steps in ctx.steps:
        def lf(name):
            p = dev[(dev["method"] == name) & (dev["horizon_steps"] == steps)]
            return pd.DataFrame({"day": p["day"].to_numpy(), "origin_index": p["origin_index"].to_numpy(), "loss": p["interval_score"].to_numpy()})

        if "static" not in set(dev["method"]):
            continue
        for spec in specs:
            if spec["name"] == "static" or spec["name"] not in set(dev["method"]):
                continue
            a, b = lf(spec["name"]), lf("static")
            if not (np.isfinite(a["loss"]).all() and np.isfinite(b["loss"]).all()):
                excluded.append({"item": f"comparação {spec['name']} × static h{steps}", "reason": "limite infinito em parte das origens (nível efetivo α_t ≤ 0); interval score indefinido"})
                continue
            scope = "primary" if spec["name"] == primary else "exploratory"
            r, d = compare(a, b, ctx.cfg, {"family": FAMILY, "hypothesis": "H5", "scope": scope, "horizon_steps": int(steps), "model_a": "xgboost_quantile", "variant_a": spec["name"], "model_b": "xgboost_quantile", "variant_b": "static", "metric": "interval score [Q05, Q95] calibrado"})
            comp_rows += r
            per_day += d
    comparisons = finalize_comparisons(comp_rows, alpha=alpha)
    write_tables(
        out,
        limits=limits_frame,
        summary=pd.concat([summary, pooled], ignore_index=True),
        coverage_by_regime=by_regime,
        coverage_windows=pd.DataFrame(win_rows),
        base_quantiles=pd.DataFrame(base_rows),
        crossing=pd.DataFrame(crossing),
        paths=pd.concat(path_rows, ignore_index=True),
        timing=pd.DataFrame(timing),
        comparisons=comparisons,
        comparisons_by_day=pd.DataFrame(per_day),
        selections=selections,
        family_info={"quantiles": quantiles, "methods": [s["name"] for s in specs], "primary_method": primary, "nominal": {"upper": high_q, "bilateral": 1.0 - alpha_low - alpha_high}, "statement": STATEMENT},
    )
    prim = comparisons[(comparisons["scope"] == "primary") & comparisons["primary_block"]] if not comparisons.empty else comparisons
    return {"primary_decisions": {f"H5/h{r.horizon_steps}": r.decision for r in prim.itertuples()}, "gates": gates, "excluded": excluded}
