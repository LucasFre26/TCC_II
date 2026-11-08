"""Figuras e tabelas complementares da V3 Parte 2, lidas só dos arquivos de uma corrida já executada.

Não treina nem altera nada nas pastas das famílias: grava em `<corrida>/atlas/` as figuras
(PNG, SVG, PDF e o CSV de cada uma), as tabelas de apoio e `figures.json`. Uso:

    python -m energy_prediction.analysis.v3p2_atlas --run results/v3-part2/v3p2-full

Convenções: erro = observado − previsto (positivo = subestimação); diferença de MAE =
candidato − XGBoost da V2 (negativo favorece o candidato); dias 3 a 6 = desenvolvimento
(exploratório); dia 7 = teste herdado, só continuidade. Potência e energia são estimativas
paramétricas em servidores substitutos; não há medição elétrica.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.analysis import style

H = {1: "5 min", 3: "15 min", 6: "30 min"}
STEPS = (1, 3, 6)
DEV_DAYS = (3, 4, 5, 6)

# Cor por entidade, igual em todas as figuras (paleta categórica validada para daltonismo).
ENTITY = {
    "referência": style.INK,
    "linha de base": style.MUTED,
    "hierárquico": "#eb6834",
    "global": "#1baf7a",
    "regimes": "#6250d6",
    "robustez": "#2a78d6",
    "picos": "#eda100",
}

# (modelo, variante) → (rótulo, entidade, marcador, variante nova da Parte 2?)
MODELS = {
    ("persistence", "-"): ("persistência", "linha de base", "o", False),
    ("statespace", "A"): ("Kalman (nível local)", "linha de base", "v", False),
    ("xgboost_delta", "A"): ("XGBoost incremento", "linha de base", "^", False),
    ("hybrid", "A"): ("híbrido Kalman + XGBoost", "linha de base", "s", False),
    ("xgboost_v2", "A"): ("XGBoost da V2 (referência)", "referência", "D", False),
    ("xgboost_v2", "recency"): ("XGBoost + peso de recência", "robustez", "<", True),
    ("xgboost_v2", "no_calendar"): ("XGBoost sem calendário", "robustez", ">", True),
    ("dlinear", "v3"): ("DLinear (formulação V3)", "robustez", "o", False),
    ("dlinear", "window"): ("DLinear normalizado", "robustez", "o", True),
    ("tcn", "v3"): ("TCN (formulação V3)", "robustez", "s", False),
    ("tcn", "window"): ("TCN normalizada", "robustez", "s", True),
    ("patchtst", "v3"): ("PatchTST (formulação V3)", "robustez", "P", False),
    ("patchtst", "window"): ("PatchTST normalizado", "robustez", "P", True),
    ("hierarchical", "candidate"): ("hierárquico (candidato)", "hierárquico", "D", True),
    ("global_xgb", "main"): ("XGBoost global (1.024 máq.)", "global", "D", True),
    ("global_tcn", "window"): ("TCN global normalizada", "global", "s", True),
    ("combination", "candidate"): ("combinação por regime (candidato)", "regimes", "D", True),
    ("combination", "mean"): ("média simples dos especialistas", "regimes", "o", True),
    ("xgboost_rise", "candidate"): ("XGBoost com peso nas subidas", "picos", "D", True),
}
FAMILY_OF = {
    ("persistence", "-"): "baselines", ("statespace", "A"): "baselines", ("xgboost_delta", "A"): "baselines",
    ("hybrid", "A"): "baselines", ("xgboost_v2", "A"): "baselines", ("xgboost_v2", "recency"): "robustness",
    ("xgboost_v2", "no_calendar"): "robustness", ("dlinear", "v3"): "robustness", ("dlinear", "window"): "robustness",
    ("tcn", "v3"): "robustness", ("tcn", "window"): "robustness", ("patchtst", "v3"): "robustness",
    ("patchtst", "window"): "robustness", ("hierarchical", "candidate"): "hierarchical",
    ("global_xgb", "main"): "global_model", ("global_tcn", "window"): "global_model",
    ("combination", "candidate"): "regimes", ("combination", "mean"): "regimes", ("xgboost_rise", "candidate"): "peaks",
}
NETWORKS = {"dlinear", "tcn", "patchtst"}


def _read(run: Path, family: str, name: str) -> pd.DataFrame:
    path = run / family / f"{name}.csv"
    if not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path)


def _save(fig, data: pd.DataFrame, figdir: Path, name: str, records: list, caption: str, sources: list[str]) -> None:
    style.localize(fig)
    for ext in ("png", "svg", "pdf"):
        (figdir / ext).mkdir(parents=True, exist_ok=True)
        meta = {} if ext == "png" else {"Date": None}
        fig.savefig(figdir / ext / f"{name}.{ext}", dpi=150, bbox_inches="tight", metadata=meta)
    (figdir / "data").mkdir(parents=True, exist_ok=True)
    data.to_csv(figdir / "data" / f"{name}.csv", index=False)
    records.append({"name": name, "caption": caption, "data": f"figures/data/{name}.csv", "sources": sources})
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------- tabelas


def pooled_metrics(run: Path) -> pd.DataFrame:
    """Métricas dos dias de desenvolvimento (linha `development_pooled`) e do dia 7, por modelo."""
    rows = []
    seed_metrics = _read(run, "robustness", "seed_metrics")
    for (model, variant), family in FAMILY_OF.items():
        m = _read(run, family, "metrics")
        if m.empty:
            continue
        part = m[(m["model"] == model) & (m["variant"] == variant)]
        for _, r in part[part["block"].isin(["development_pooled", "day_7"])].iterrows():
            row = {"model": model, "variant": variant, "label": MODELS[(model, variant)][0], "family": family, **r.drop(["model", "variant"]).to_dict()}
            row["skill_vs_persistence_pct"] = 100.0 * (1.0 - r["mase_vs_persistence"])
            row["prediction"] = "média das previsões de 3 sementes" if model in NETWORKS | {"global_tcn"} else "única"
            rows.append(row)
    out = pd.DataFrame(rows)
    if not seed_metrics.empty:
        # MAE médio entre sementes, que é a perda usada nas comparações das redes.
        dev = seed_metrics[seed_metrics["role"] == "development"]
        per_seed = dev.groupby(["model", "variant", "horizon_steps", "seed"]).apply(lambda g: np.average(g["mae"], weights=g["n"]), include_groups=False)
        mean_seed = per_seed.groupby(["model", "variant", "horizon_steps"]).agg(["mean", "std"]).reset_index().rename(columns={"mean": "mae_seed_mean", "std": "mae_seed_std"})
        out = out.merge(mean_seed, on=["model", "variant", "horizon_steps"], how="left")
        out.loc[out["block"] != "development_pooled", ["mae_seed_mean", "mae_seed_std"]] = np.nan
    ref = out[(out["model"] == "xgboost_v2") & (out["variant"] == "A")][["block", "horizon_steps", "mae"]].rename(columns={"mae": "mae_xgboost_v2"})
    out = out.merge(ref, on=["block", "horizon_steps"], how="left")
    out["gain_vs_xgboost_v2_pct"] = 100.0 * (1.0 - out["mae"] / out["mae_xgboost_v2"])
    return out


def error_quantiles(run: Path) -> pd.DataFrame:
    """P50, P95, P99 e máximo do erro absoluto e viés, por modelo e horizonte (desenvolvimento)."""
    rows = []
    for (model, variant), family in FAMILY_OF.items():
        p = _read(run, family, "predictions")
        if p.empty:
            continue
        part = p[(p["model"] == model) & (p["variant"] == variant) & (p["role"] == "development")]
        if "phase" in part:
            part = part[part["phase"] != "inner"]
        if model in NETWORKS:
            # Mesma previsão das métricas: média das previsões das sementes em cada origem.
            part = part.groupby(["day", "horizon_steps", "origin_index"], as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        for steps, g in part.groupby("horizon_steps"):
            err = (g["y_true"] - g["y_pred"]).to_numpy(dtype=float)
            a = np.abs(err)
            rows.append({"model": model, "variant": variant, "label": MODELS[(model, variant)][0], "horizon_steps": int(steps), "n": int(len(a)),
                         "abs_p50": float(np.quantile(a, 0.5)), "abs_p95": float(np.quantile(a, 0.95)), "abs_p99": float(np.quantile(a, 0.99)),
                         "abs_max": float(a.max()), "bias": float(err.mean())})
    return pd.DataFrame(rows)


def machine_level(run: Path, project_root: Path) -> pd.DataFrame:
    """MAE por máquina do XGBoost global contra a persistência da própria máquina (desenvolvimento)."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    key = ((manifest.get("data") or {}).get("cache") or {}).get("key")
    if not key:
        return pd.DataFrame()  # dados sintéticos: sem painel em disco
    panels = sorted((project_root / "data" / "processed").glob(f"panel_{key}-*"))
    if not panels:
        return pd.DataFrame()
    cpu = np.load(panels[0] / "cpu.npy")
    rows = []
    for day in DEV_DAYS:
        path = run / "global_model" / "machine_predictions" / f"day_{day}.npz"
        if not path.is_file():
            continue
        z = np.load(path)
        origins, steps, pred = z["origins"], list(z["steps"]), z["predictions"]
        for s in STEPS:
            target = origins + s
            ok_rows = target < cpu.shape[0]
            y = cpu[target[ok_rows]]
            base = cpu[origins[ok_rows]]
            yhat = pred[ok_rows, :, steps.index(s)].astype(float)
            valid = np.isfinite(y) & np.isfinite(base) & np.isfinite(yhat)
            n = valid.sum(axis=0)
            err_m = np.where(valid, np.abs(y - yhat), 0.0).sum(axis=0)
            err_p = np.where(valid, np.abs(y - base), 0.0).sum(axis=0)
            keep = n >= 144
            for j in np.flatnonzero(keep):
                rows.append({"day": day, "horizon_steps": s, "machine": str(z["machines"][j]), "n": int(n[j]),
                             "mae_global": err_m[j] / n[j], "mae_persistence": err_p[j] / n[j]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ratio"] = out["mae_global"] / out["mae_persistence"].replace(0, np.nan)
    return out


# ---------------------------------------------------------------- figuras


def fig_ranking(metrics: pd.DataFrame, figdir: Path, records: list) -> None:
    dev = metrics[metrics["block"] == "development_pooled"].copy()
    dev["mae_plot"] = np.where(dev["mae_seed_mean"].notna(), dev["mae_seed_mean"], dev["mae"])
    order = dev.groupby("label")["mae_plot"].mean().sort_values(ascending=False).index.tolist()
    fig, axes = style.figure(9.0, 5.6, ncols=3, sharey=True)
    for ax, s in zip(axes, STEPS):
        part = dev[dev["horizon_steps"] == s].set_index("label").reindex(order)
        y = np.arange(len(order))
        ref = float(part.loc[MODELS[("xgboost_v2", "A")][0], "mae_plot"])
        ax.axvline(ref, color=style.AXIS, lw=1, ls="--")
        for i, (label, r) in enumerate(part.iterrows()):
            key = (r["model"], r["variant"])
            _, entity, marker, new = MODELS[key]
            c = ENTITY[entity]
            ax.plot(r["mae_plot"], i, marker=marker, ms=6, color=c, mfc=c if new else "white", mec=c, mew=1.4, ls="none")
        ax.set_title(H[s])
        ax.set_xlabel("MAE nos dias 3–6 (p.p.)")
        ax.set_yticks(y, order)
    _save(fig, dev[["label", "model", "variant", "horizon_steps", "mae", "mae_seed_mean", "rmse", "bias_obs_minus_pred", "mase_vs_persistence", "gain_vs_xgboost_v2_pct"]],
          figdir, "07_ranking_desenvolvimento", records,
          "MAE nos dias de desenvolvimento por modelo e horizonte; tracejado = XGBoost da V2; marcador cheio = variante nova da Parte 2, vazio = especificação herdada. Redes: MAE médio entre 3 sementes.",
          ["baselines/metrics.csv", "robustness/metrics.csv", "robustness/seed_metrics.csv", "hierarchical/metrics.csv", "global_model/metrics.csv", "regimes/metrics.csv", "peaks/metrics.csv"])


def fig_by_day(run: Path, metrics_all: dict, figdir: Path, records: list) -> None:
    series = [(("hierarchical", "candidate"), "hierarchical"), (("global_xgb", "main"), "global_model"),
              (("combination", "candidate"), "regimes"), (("xgboost_rise", "candidate"), "peaks")]
    ref = metrics_all["baselines"]
    ref = ref[(ref["model"] == "xgboost_v2") & (ref["variant"] == "A") & (ref["day"] > 0)][["horizon_steps", "day", "mae"]].rename(columns={"mae": "mae_ref"})
    rows = []
    for key, family in series:
        m = metrics_all[family]
        part = m[(m["model"] == key[0]) & (m["variant"] == key[1]) & (m["day"] > 0)][["horizon_steps", "day", "mae"]]
        part = part.merge(ref, on=["horizon_steps", "day"])
        part["label"] = MODELS[key][0]
        part["entity"] = MODELS[key][1]
        part["diff"] = part["mae"] - part["mae_ref"]
        rows.append(part)
    data = pd.concat(rows, ignore_index=True)
    fig, axes = style.figure(9.0, 3.4, ncols=3, sharey=True)
    for ax, s in zip(axes, STEPS):
        ax.axhline(0, color=style.AXIS, lw=1)
        ax.axvspan(6.5, 7.5, color=style.GRID, alpha=0.6, lw=0)
        for (label, entity), g in data[data["horizon_steps"] == s].groupby(["label", "entity"]):
            g = g.sort_values("day")
            c = ENTITY[entity]
            dev = g[g["day"] <= 6]
            ax.plot(dev["day"], dev["diff"], color=c, lw=1.2, marker="o", ms=5, label=label)
            last = g[g["day"] == 7]
            ax.plot(last["day"], last["diff"], color=c, marker="o", ms=6, mfc="white", mec=c, ls="none")
        ax.set_title(H[s])
        ax.set_xticks([3, 4, 5, 6, 7], ["dia 3", "dia 4", "dia 5", "dia 6", "dia 7\n(herdado)"])
    axes[0].set_ylabel("MAE candidato − XGBoost da V2 (p.p.)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, data, figdir, "08_diferenca_por_dia", records,
          "Diferença de MAE contra o XGBoost da V2 em cada dia; abaixo de zero favorece o candidato. Dia 7 (vazio, faixa cinza) é o teste herdado, só continuidade.",
          ["baselines/metrics.csv", "hierarchical/metrics.csv", "global_model/metrics.csv", "regimes/metrics.csv", "peaks/metrics.csv"])


def fig_hierarchical_grid(run: Path, figdir: Path, records: list) -> None:
    m = _read(run, "hierarchical", "metrics")
    dev = m[m["block"] == "development_pooled"]
    ref = dev[dev["variant"] == "A"].set_index("horizon_steps")["mae"]
    grid = dev[(dev["model"] == "hierarchical") & (dev["variant"] != "candidate")].copy()
    grid["diff"] = grid["mae"] - grid["horizon_steps"].map(ref)
    grid["method"] = grid["variant"].str.extract(r"_(bottom_up|ols|mint_shrink)$")[0]
    grid["structure"] = grid["variant"].str.replace(r"_(bottom_up|ols|mint_shrink)$", "", regex=True)
    rows = ["mean_k4", "mean_k8", "mean_k16", "volatility_k4", "volatility_k8", "volatility_k16", "profile_k4", "profile_k8", "profile_k16", "failure_domain_knatural"]
    names = {"mean": "média", "volatility": "volatilidade", "profile": "perfil", "failure_domain": "domínio de falha"}
    cols = [(s, meth) for s in STEPS for meth in ("bottom_up", "ols", "mint_shrink")]
    mat = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, (s, meth) in enumerate(cols):
            v = grid[(grid["structure"] == r) & (grid["method"] == meth) & (grid["horizon_steps"] == s)]["diff"]
            if len(v):
                mat[i, j] = float(v.iloc[0])
    lim = float(np.nanmax(np.abs(mat)))
    fig, ax = style.figure(8.2, 4.6)
    im = ax.imshow(mat, cmap=style.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.3f}".replace("-", "−"), ha="center", va="center", fontsize=6.5, color=style.INK)
    def row_label(r):
        crit, k = r.rsplit("_k", 1)
        return f"{names[crit]} · K = {'natural (20)' if k == 'natural' else k}"
    ax.set_yticks(range(len(rows)), [row_label(r) for r in rows])
    meth_names = {"bottom_up": "ascendente", "ols": "OLS", "mint_shrink": "MinT"}
    ax.set_xticks(range(len(cols)), [f"{meth_names[m]}\n{H[s]}" for s, m in cols], fontsize=7)
    for x in (2.5, 5.5):
        ax.axvline(x, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("MAE variante − XGBoost da V2 (p.p.)")
    _save(fig, grid[["structure", "method", "horizon_steps", "mae", "diff"]], figdir, "09_grade_hierarquica", records,
          "MAE de cada configuração hierárquica fixa contra o XGBoost da V2 nos dias 3–6 (azul favorece o hierárquico). Leitura descritiva: a grade inteira foi vista depois da avaliação.",
          ["hierarchical/metrics.csv"])


def fig_series(run: Path, figdir: Path, records: list, day: int = 6, steps: int = 6) -> None:
    sources = [(("xgboost_v2", "A"), "baselines"), (("hierarchical", "candidate"), "hierarchical"),
               (("global_xgb", "main"), "global_model"), (("combination", "candidate"), "regimes")]
    frames = []
    for key, family in sources:
        p = _read(run, family, "predictions")
        part = p[(p["model"] == key[0]) & (p["variant"] == key[1]) & (p["day"] == day) & (p["horizon_steps"] == steps)]
        if "phase" in part:
            part = part[part["phase"] != "inner"]
        part = part[["origin_index", "y_true", "y_pred"]].assign(label=MODELS[key][0], entity=MODELS[key][1])
        frames.append(part)
    data = pd.concat(frames, ignore_index=True)
    data["hour"] = (data["origin_index"] + steps) * 300 / 3600
    fig, ax = style.figure(9.0, 3.8)
    obs = data[data["label"] == MODELS[("xgboost_v2", "A")][0]].sort_values("hour")
    ax.plot(obs["hour"], obs["y_true"], color=style.OBSERVED, lw=1.8, label="CPU observada")
    for (label, entity), g in data.groupby(["label", "entity"], sort=False):
        g = g.sort_values("hour")
        c = ENTITY[entity] if entity != "referência" else "#2a78d6"
        ax.plot(g["hour"], g["y_pred"], color=c, lw=1.0, label=label, alpha=0.95)
    ax.set_xlabel("horas desde o início do trace (instante previsto)")
    ax.set_ylabel("CPU agregada (%)")
    ax.legend(ncol=2, loc="upper right")
    _save(fig, data, figdir, "10_serie_real_prevista", records,
          f"CPU observada e prevista no dia {day} (desenvolvimento), horizonte de {H[steps]}, no instante previsto.",
          ["baselines/predictions.csv", "hierarchical/predictions.csv", "global_model/predictions.csv", "regimes/predictions.csv"])


def fig_regimes(run: Path, figdir: Path, records: list, steps: int = 3) -> None:
    m = _read(run, "regimes", "metrics_by_regime")
    part = m[(m["role"] == "development") & (m["horizon_steps"] == steps)]
    keys = [("persistence", "-"), ("xgboost_v2", "A"), ("hybrid", "A"), ("combination", "candidate"), ("combination", "mean")]
    regimes = ["estavel", "subida", "queda", "alta_volatilidade"]
    names = {"estavel": "estável", "subida": "subida", "queda": "queda", "alta_volatilidade": "alta volatilidade"}
    fig, ax = style.figure(7.6, 3.8)
    width = 0.15
    rows = []
    for k, key in enumerate(keys):
        g = part[(part["model"] == key[0]) & (part["variant"] == key[1])].set_index("regime").reindex(regimes)
        label, entity, marker, new = MODELS[key]
        c = ENTITY[entity]
        x = np.arange(len(regimes)) + (k - 2) * width
        ax.plot(x, g["mae"], marker=marker, ms=6, color=c, mfc=c if new else "white", mec=c, mew=1.4, ls="none", label=label)
        rows.append(g.reset_index().assign(label=label))
    n = part.drop_duplicates("regime").set_index("regime").reindex(regimes)["n"]
    ax.set_xticks(range(len(regimes)), [f"{names[r]}\n(n = {int(n[r])})" for r in regimes])
    ax.set_ylabel(f"MAE nos dias 3–6, {H[steps]} (p.p.)")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, pd.concat(rows, ignore_index=True), figdir, "11_mae_por_regime", records,
          f"MAE por regime detectado na origem (sem usar o alvo), dias 3–6, {H[steps]}.", ["regimes/metrics_by_regime.csv"])


def fig_peaks_by_day(run: Path, figdir: Path, records: list) -> None:
    c = _read(run, "peaks", "comparisons_by_day")
    c = c[(c["scope"] == "primary") & (c["hypothesis"] == "H4")]
    ch = _read(run, "peaks", "choices")
    ch["day"] = ch["block"].str.replace("day_", "").astype(int)
    data = c.merge(ch[["day", "horizon_steps", "chosen_lambda"]], on=["day", "horizon_steps"], how="left")
    fig, axes = style.figure(9.0, 3.4, ncols=3, sharey=True)
    for ax, s in zip(axes, STEPS):
        g = data[data["horizon_steps"] == s].sort_values("day")
        x = np.arange(len(g))
        ax.bar(x - 0.19, g["loss_b"], 0.36, color=style.MUTED, label="XGBoost da V2", edgecolor="white", linewidth=2)
        ax.bar(x + 0.19, g["loss_a"], 0.36, color=ENTITY["picos"], label="peso nas subidas", edgecolor="white", linewidth=2)
        for xi, (_, r) in zip(x, g.iterrows()):
            ax.text(xi + 0.19, r["loss_a"] + 0.15, f"λ = {r['chosen_lambda']:g}", ha="center", fontsize=6.5, color=style.INK_2)
        ax.set_xticks(x, [f"dia {d}\n(n = {n})" for d, n in zip(g["day"], g["n"])], fontsize=7)
        ax.set_title(H[s])
    axes[0].set_ylabel("viés nos picos, obs − prev (p.p.)")
    axes[-1].legend(loc="upper right")
    _save(fig, data, figdir, "12_picos_por_dia", records,
          "Viés de subestimação nos instantes de pico (alvo ≥ P90 do treino) por dia, com o λ escolhido na validação interna; λ = 0 reproduz o XGBoost da V2.",
          ["peaks/comparisons_by_day.csv", "peaks/choices.csv"])


def fig_calibration(run: Path, figdir: Path, records: list, steps: int = 3) -> None:
    reg = _read(run, "calibration", "coverage_by_regime")
    summ = _read(run, "calibration", "summary")
    methods = {"static": ("conformal estático (V3)", style.MUTED, "o"), "rolling_n288": ("janela móvel 288", "#2a78d6", "s"),
               "aci_g0.005": ("ACI γ = 0,005", ENTITY["regimes"], "D"), "spci": ("SPCI", ENTITY["picos"], "^")}
    regimes = ["estavel", "subida", "queda", "alta_volatilidade"]
    names = {"estavel": "estável", "subida": "subida", "queda": "queda", "alta_volatilidade": "alta\nvolatilidade"}
    fig, (a, b) = style.figure(9.0, 3.8, ncols=2)
    part = reg[(reg["role"] == "development") & (reg["horizon_steps"] == steps)]
    for k, (meth, (label, c, mk)) in enumerate(methods.items()):
        g = part[part["method"] == meth].set_index("regime").reindex(regimes)
        a.plot(np.arange(len(regimes)) + (k - 1.5) * 0.14, g["coverage_upper"], marker=mk, color=c, ms=6, ls="none", label=label)
    a.axhline(0.95, color=style.AXIS, ls="--", lw=1)
    a.set_xticks(range(len(regimes)), [names[r] for r in regimes])
    a.set_ylabel(f"cobertura do limite Q95, {H[steps]}")
    a.set_title("(a) por regime, dias 3–6")
    pooled = summ[summ["block"] == "development_pooled"]
    for meth, (label, c, mk) in methods.items():
        g = pooled[pooled["method"] == meth].sort_values("horizon_steps")
        b.plot(g["mean_width"], g["coverage_upper"], marker=mk, color=c, ms=6, lw=0.8, label=label)
        for _, r in g.iterrows():
            b.annotate(H[int(r["horizon_steps"])], (r["mean_width"], r["coverage_upper"]), textcoords="offset points", xytext=(4, -9), fontsize=6, color=style.INK_2)
    b.axhline(0.95, color=style.AXIS, ls="--", lw=1)
    b.set_xlabel("largura média do intervalo Q05–Q95 (p.p.)")
    b.set_ylabel("cobertura do limite Q95")
    b.set_title("(b) cobertura × largura")
    b.legend(loc="lower right")
    data = pd.concat([part.assign(panel="a"), pooled.assign(panel="b")], ignore_index=True)
    _save(fig, data, figdir, "13_calibracao_regime_largura", records,
          "Cobertura empírica do limite superior por regime (15 min) e compromisso cobertura × largura por método; nominal 0,95. Coberturas empíricas, sem garantia sob dependência temporal.",
          ["calibration/coverage_by_regime.csv", "calibration/summary.csv"])


def fig_energy(run: Path, figdir: Path, records: list) -> None:
    info = json.loads((run / "energy" / "energy_info.json").read_text(encoding="utf-8"))
    agg = _read(run, "energy", "aggregation_summary")
    summ = _read(run, "energy", "summary")
    primary = info["primary_substitute"]
    fig, (a, b) = style.figure(9.4, 3.9, ncols=2)
    part = agg[(agg["role"] == "development") & (agg["scenario"] == primary) & (agg["horizon_steps"] == 6)]
    routes = {"mean": ("rota da média", style.INK, "o", "-"), "group_mean_k4": ("grupos K = 4", ENTITY["hierárquico"], "v", ":"),
              "group_mean_k8": ("grupos K = 8", ENTITY["hierárquico"], "s", "--"), "group_mean_k16": ("grupos K = 16", ENTITY["hierárquico"], "^", "-"),
              "group_candidate": ("grupos do candidato", ENTITY["hierárquico"], "D", "-.")}
    for route, (label, c, mk, ls) in routes.items():
        g = part[part["route"] == route].sort_values("r")
        a.plot(g["r"], 100 * g["mean_relative_aggregation"], marker=mk, color=c, ls=ls, lw=1.1, ms=5, label=label)
    a.axhline(0, color=style.AXIS, lw=1)
    a.set_xticks([1.0, 1.4, 1.8])
    a.set_xlabel("expoente r do modelo de Fan")
    a.set_ylabel("aproximação por agregação (% da referência)")
    a.set_title("(a) rota observada − referência por máquina, 30 min")
    a.legend(loc="upper left")
    scen = {primary: "2288H V5\n(V3, 112 thr.)", "power_ssj2008-20170807-00775": "NF5280 M4\n(88 thr.)",
            "power_ssj2008-20190730-00995": "TX2550 M5\n(96 thr., 2019)", "heterogeneo_sintetico": "heterogêneo\nsintético"}
    keys = [("persistence", "mean", "persistência · média"), ("xgboost_v2", "mean", "XGBoost V2 · média"),
            ("hierarchical", "group_candidate", "hierárquico · grupos"), ("global", "machine", "global · por máquina")]
    ents = {"persistence": "linha de base", "xgboost_v2": "referência", "hierarchical": "hierárquico", "global": "global"}
    marks = {"persistence": "o", "xgboost_v2": "D", "hierarchical": "D", "global": "s"}
    pb = summ[(summ["role"] == "development") & (summ["r"] == info["primary_r"]) & (summ["horizon_steps"] == 6)]
    rows = []
    for k, (model, route, label) in enumerate(keys):
        g = pb[(pb["model"] == model) & (pb["route"] == route)].set_index("scenario").reindex(list(scen))
        c = ENTITY[ents[model]]
        b.plot(np.arange(len(scen)) + (k - 1.5) * 0.15, 100 * g["mean_relative_abs_total"], marker=marks[model], color=c, ms=6, ls="none", label=label)
        rows.append(g.reset_index().assign(label=label))
    b.set_xticks(range(len(scen)), list(scen.values()), fontsize=7)
    b.set_ylabel("|energia projetada − referência| (% da referência)")
    b.set_title(f"(b) erro total por origem, 30 min, r = {info['primary_r']:g}")
    b.legend(loc="upper right", fontsize=6.5)
    data = pd.concat([part.assign(panel="a"), pd.concat(rows, ignore_index=True).assign(panel="b")], ignore_index=True)
    _save(fig, data, figdir, "14_energia_sensibilidade", records,
          "Energia de TI modelada (estimativa paramétrica, não medição): (a) parcela de agregação por rota e expoente r; (b) erro total por origem por modelo, rota e servidor substituto.",
          ["energy/aggregation_summary.csv", "energy/summary.csv", "energy/energy_info.json"])


def fig_inherited(run: Path, figdir: Path, records: list) -> pd.DataFrame:
    rows = []
    for family in ("robustness", "hierarchical", "global_model", "regimes", "peaks"):
        t = _read(run, family, "inherited_test")
        if t.empty:
            continue
        rows.append(t[(t["model_b"] == "xgboost_v2") & (t["variant_b"] == "A")])
    data = pd.concat(rows, ignore_index=True).drop_duplicates(["model_a", "variant_a", "horizon_steps"])
    data["label"] = [MODELS[(m, v)][0] for m, v in zip(data["model_a"], data["variant_a"])]
    data["entity"] = [MODELS[(m, v)][1] for m, v in zip(data["model_a"], data["variant_a"])]
    order = data.groupby("label")["mae_diff"].mean().sort_values(ascending=False).index.tolist()
    fig, axes = style.figure(9.0, 3.6, ncols=3, sharey=True)
    for ax, s in zip(axes, STEPS):
        g = data[data["horizon_steps"] == s].set_index("label").reindex(order)
        ax.axvline(0, color=style.AXIS, lw=1)
        for i, (label, r) in enumerate(g.iterrows()):
            c = ENTITY[r["entity"]]
            ax.plot(r["mae_diff"], i, marker="o", mfc="white", mec=c, mew=1.4, ms=6, ls="none")
        ref = float(data[data["horizon_steps"] == s]["mae_b"].iloc[0])
        ax.set_title(f"{H[s]} · referência = {ref:.3f}")
        ax.set_yticks(range(len(order)), order)
        ax.set_xlabel("MAE − XGBoost da V2 no dia 7 (p.p.)")
    _save(fig, data, figdir, "15_teste_herdado", records,
          "Continuidade no dia 7 (teste herdado, quarta inspeção): diferença de MAE contra o XGBoost da V2 reajustado no protocolo por dia, com hiperparâmetros escolhidos de novo na validação interna (MAE 3,040 / 3,856 / 4,347). A previsão preservada da V2, reproduzida com diferença de 7 × 10⁻¹⁵ p.p., tem MAE 3,064 / 3,856 / 4,334. Sem inferência e sem valor de confirmação.",
          ["robustness/inherited_test.csv", "hierarchical/inherited_test.csv", "global_model/inherited_test.csv", "regimes/inherited_test.csv", "peaks/inherited_test.csv"])
    return data


def fig_machine(machine: pd.DataFrame, figdir: Path, records: list) -> pd.DataFrame:
    summary = machine.groupby("horizon_steps").agg(
        n_machine_days=("ratio", "size"), median_mae_global=("mae_global", "median"), median_mae_persistence=("mae_persistence", "median"),
        median_ratio=("ratio", "median"), share_better=("ratio", lambda r: float((r < 1).mean())),
        p10_ratio=("ratio", lambda r: float(r.quantile(0.1))), p90_ratio=("ratio", lambda r: float(r.quantile(0.9)))).reset_index()
    fig, ax = style.figure(6.6, 3.6)
    for s in STEPS:
        r = np.sort(machine.loc[machine["horizon_steps"] == s, "ratio"].dropna().to_numpy())
        ax.plot(r, np.arange(1, len(r) + 1) / len(r), lw=1.4, color={1: "#2a78d6", 3: ENTITY["global"], 6: ENTITY["regimes"]}[s], label=H[s])
    ax.axvline(1, color=style.AXIS, ls="--", lw=1)
    ax.set_xlim(0.6, 1.6)
    ax.set_xlabel("MAE do XGBoost global ÷ MAE da persistência da própria máquina")
    ax.set_ylabel("fração acumulada de máquina-dias")
    ax.legend(loc="lower right")
    _save(fig, machine, figdir, "16_global_por_maquina", records,
          "Distribuição, entre máquinas e dias 3–6, da razão entre o MAE do XGBoost global e o da persistência por máquina (≥ 144 origens válidas por máquina-dia). A cauda acima de 1,6 (cerca de 5%) reúne máquinas quase constantes, em que a persistência erra perto de zero.",
          ["global_model/machine_predictions/day_<d>.npz", "data/processed/panel_<chave>/cpu.npy"])
    return summary


def generate(run: Path, project_root: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    style.apply()
    run = Path(run)
    out = run / "atlas"
    figdir = out / "figures"
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    metrics = pooled_metrics(run)
    metrics.to_csv(tables / "metricas_consolidadas.csv", index=False)
    error_quantiles(run).to_csv(tables / "quantis_do_erro.csv", index=False)
    metrics_all = {f: _read(run, f, "metrics") for f in ("baselines", "hierarchical", "global_model", "regimes", "peaks")}
    fig_ranking(metrics, figdir, records)
    fig_by_day(run, metrics_all, figdir, records)
    fig_hierarchical_grid(run, figdir, records)
    fig_series(run, figdir, records)
    fig_regimes(run, figdir, records)
    fig_peaks_by_day(run, figdir, records)
    fig_calibration(run, figdir, records)
    fig_energy(run, figdir, records)
    fig_inherited(run, figdir, records)
    machine = machine_level(run, project_root)
    if not machine.empty:
        fig_machine(machine, figdir, records).to_csv(tables / "global_por_maquina_resumo.csv", index=False)
    (out / "figures.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"figures": [r["name"] for r in records], "out": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Figuras e tabelas complementares da V3 Parte 2 (só leitura dos resultados).")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(generate(args.run, args.project_root))


if __name__ == "__main__":
    main()
