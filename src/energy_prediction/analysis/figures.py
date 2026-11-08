"""Catálogo das figuras da análise expandida.

Cada figura é uma função que recebe as tabelas da análise e devolve (figura, dados desenhados).
O catálogo registra a pergunta que a figura responde, os CSVs de origem e a legenda sugerida.
Tempo nos eixos: horas desde o início do trace (não é horário civil).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from energy_prediction.analysis import style
from energy_prediction.analysis.style import HORIZON_LABEL, HORIZON_MARKERS, INK_2, MUTED, OBSERVED

TIME_LABEL = "tempo desde o início do trace (h; não é horário civil)"
CPU_LABEL = "CPU agregada (%)"


@dataclass(frozen=True)
class FigureSpec:
    name: str
    section: str
    title: str
    question: str
    sources: tuple[str, ...]
    caption: str
    build: Callable


class Figures:
    def __init__(self, tables: dict[str, pd.DataFrame], info: dict, cfg: dict):
        self.t = tables
        self.info = info
        self.cfg = cfg
        self.labels = cfg["models"]["labels"]
        self.families = cfg["models"]["families"]
        self.order = cfg["models"]["order"]
        self.best_neural = info.get("best_neural", "dlinear")
        self.highlight = [*cfg["models"]["highlight"], self.best_neural]

    # utilitários --------------------------------------------------------------------------
    def lab(self, model: str) -> str:
        return self.labels.get(model, model)

    def col(self, model: str) -> str:
        return style.color(model, self.families)

    def metrics(self, split: str = "test_final", config: str = "A") -> pd.DataFrame:
        m = self.t["model_metrics"]
        return m[(m["split"] == split) & (m["population"] == "main") & (m["analysis_set"] == "main") & (m["mode"] == "direct") & m["config"].isin([config, "-"])]

    def models_in(self, frame: pd.DataFrame) -> list[str]:
        present = set(frame["model"])
        return [m for m in self.order if m in present]

    def family_legend(self, ax, loc="lower right", extra=()):
        handles = [Patch(color=c, label=f) for f, c in style.FAMILY_COLORS.items() if f != "V3 quantílico"]
        ax.legend(handles=[*handles, *extra], loc=loc, fontsize=7)

    def test_errors(self) -> pd.DataFrame:
        e = self.t["errors_by_origin"]
        return e[e["config"].isin(["A", "-"])]

    def shade_windows(self, ax):
        windows = self.t.get("diagnostic_windows")
        if windows is None:
            return
        for _, w in windows.iterrows():
            ax.axvspan(w["start_time_h"], w["end_time_h"] + 5 / 60, color="#f0efec", zorder=0)
            ax.text((w["start_time_h"] + w["end_time_h"]) / 2, 0.98, w["window"], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color=INK_2)

    # 1. dataset ---------------------------------------------------------------------------
    def f01(self):
        s = self.t["dataset_series"]
        fig, axes = style.figure(6.6, 4.8, nrows=3, sharex=True, height_ratios=[2.2, 1, 1])
        for ax in axes:
            for p in self.info.get("periods", []):
                if p["period"].startswith("validacao_fold"):
                    ax.axvspan(p["start_index"] / 12, (p["end_index"] + 1) / 12, color="#cde2fb", zorder=0, lw=0)
                if p["period"] == "teste":
                    ax.axvspan(p["start_index"] / 12, (p["end_index"] + 1) / 12, color="#fbd9ca", zorder=0, lw=0)
        axes[0].plot(s["time_h"], s["aggregated_cpu_weighted_pct"], color=OBSERVED, lw=0.8)
        axes[0].set_ylabel(CPU_LABEL)
        axes[0].set_title("Série agregada da população principal (4.023 máquinas) e partições temporais", loc="left")
        axes[0].legend(handles=[Patch(color="#cde2fb", label="validação externa (5 folds de 12 h)"), Patch(color="#fbd9ca", label="teste final (24 h, já inspecionado na V2)")], loc="upper left", ncols=2)
        axes[1].plot(s["time_h"], s["aggregated_mem_pct"], color=INK_2, lw=0.8)
        axes[1].set_ylabel("memória (%)")
        axes[2].plot(s["time_h"], s["n_machines_present"], color=INK_2, lw=0.8)
        axes[2].set_ylabel("máquinas\npresentes")
        axes[2].set_xlabel(TIME_LABEL)
        return fig, s

    def f02(self):
        s = self.t["dataset_series"].copy()
        test_start = self.info["test_range"][0]
        s["period"] = np.where(s["bin_index"] < test_start, "desenvolvimento", "teste")
        dep = self.t["dataset_dependence"]
        fig, axes = style.figure(6.6, 4.6, nrows=2, ncols=2)
        ax = axes[0, 0]
        bins = np.linspace(15, 80, 40)
        for period, c in (("desenvolvimento", "#2a78d6"), ("teste", "#eb6834")):
            v = s.loc[s["period"] == period, "aggregated_cpu_weighted_pct"].dropna()
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.5, color=c, label=f"{period} (média {v.mean():.1f}%)")
        ax.set_xlabel(CPU_LABEL)
        ax.set_ylabel("densidade")
        ax.set_title("(a) Nível da CPU", loc="left")
        ax.legend()
        ax = axes[0, 1]
        cpu = s.set_index("bin_index")["aggregated_cpu_weighted_pct"]
        rows = []
        for h in (1, 3, 6):
            d = (cpu.shift(-h) - cpu).dropna()
            rows.append(d.to_numpy())
        ax.boxplot(rows, tick_labels=[HORIZON_LABEL[h] for h in (1, 3, 6)], whis=(5, 95), showfliers=False, medianprops={"color": OBSERVED})
        ax.set_ylabel("cpu(t+h) − cpu(t) (p.p.)")
        ax.set_title("(b) Variação em h (P25–P75; hastes P5–P95)", loc="left")
        ax = axes[1, 0]
        acf = dep[dep["kind"] == "autocorrelacao_cpu"]
        ax.plot(acf["lag_min"] / 60, acf["correlation"], color=OBSERVED)
        for h in (1, 3, 6):
            r = acf[acf["lag_steps"] == h]["correlation"].iloc[0]
            ax.plot(h * 5 / 60, r, "o", color="#eb6834", ms=5)
            ax.annotate(f"{HORIZON_LABEL[h]}: {r:.2f}", (h * 5 / 60, r), xytext=(6, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("defasagem (h)")
        ax.set_ylabel("autocorrelação")
        ax.set_title("(c) Autocorrelação da CPU", loc="left")
        ax = axes[1, 1]
        lead = dep[dep["kind"] == "nivel_exogena_vs_cpu_futura"]
        for (var, part), c in zip(lead.groupby("variable"), ["#6250d6", "#1baf7a", "#eb6834", "#2a78d6"]):
            ax.plot(part["lag_min"], part["correlation"], marker="o", ms=3, color=c, label=var)
        ax.axhline(0, color=AXIS_COLOR, lw=0.8)
        ax.set_xlabel("antecedência k (min)")
        ax.set_ylabel("correlação no desenvolvimento")
        ax.set_title("(d) Exógena(t) × CPU(t+k)", loc="left")
        ax.legend(ncols=2)
        return fig, pd.concat([acf, lead])

    def f03(self):
        f = self.t["fleet_distribution"]
        cov = self.t["machine_coverage"]
        fig, axes = style.figure(6.6, 4.2, nrows=2, height_ratios=[2, 1])
        ax = axes[0]
        ax.fill_between(f["time_h"], f["machine_cpu_p10_pct"], f["machine_cpu_p90_pct"], color="#cde2fb", lw=0, label="P10–P90 por máquina")
        ax.fill_between(f["time_h"], f["machine_cpu_p25_pct"], f["machine_cpu_p75_pct"], color="#86b6ef", lw=0, label="P25–P75 por máquina")
        ax.plot(f["time_h"], f["machine_cpu_p50_pct"], color="#184f95", lw=0.7, label="mediana por máquina")
        ax.plot(f["time_h"], f["aggregated_cpu_weighted_pct"], color=OBSERVED, lw=0.8, label="agregada ponderada")
        ax.set_xlabel(TIME_LABEL)
        ax.set_ylabel("CPU (%)")
        ax.set_title("(a) Distribuição da CPU individual entre as máquinas presentes", loc="left")
        ax.legend(ncols=2, loc="upper left", fontsize=6.5)
        ax = axes[1]
        ax.hist(cov["coverage_fraction"], bins=40, color="#2a78d6")
        ax.set_yscale("log")
        ax.set_xlabel("fração dos 2.304 intervalos com registro")
        ax.set_ylabel("máquinas (log)")
        ax.set_title("(b) Cobertura temporal por máquina", loc="left")
        return fig, f

    def f04(self):
        d = self.t["population_difference"]
        cc = self.t["cohort_comparison"]
        cc = cc[cc["split"] == "test_final"]
        fig, axes = style.figure(6.6, 4.4, nrows=2)
        ax = axes[0]
        ax.plot(d["time_h"], d["difference_main_minus_cohort_pp"], color=OBSERVED, lw=0.7)
        last = d.loc[d["m_2739_present"], "time_h"].max()
        ax.axvline(last, color="#eb6834", lw=1, ls="--")
        ax.annotate(f"último registro de m_2739 ({last:.1f} h)", (last, d["difference_main_minus_cohort_pp"].max()), xytext=(5, -10), textcoords="offset points", fontsize=7, color=INK_2)
        t0 = self.info["test_range"][0] / 12
        ax.axvspan(t0, d["time_h"].max(), color="#fbd9ca", lw=0, zorder=0)
        ax.set_xlabel(TIME_LABEL)
        ax.set_ylabel("principal − coorte (p.p.)")
        ax.set_title("(a) CPU principal − CPU coorte (m_2739: CPU 0% em todos os registros)", loc="left", fontsize=8)
        ax = axes[1]
        cc = cc.assign(label=cc["model"].map(self.lab) + " " + cc["config"] + " · " + cc["horizon_steps"].map(HORIZON_LABEL))
        colors = [self.col(m) for m in cc["model"]]
        ax.bar(np.arange(len(cc)), cc["mae_difference_main_minus_cohort_pp"], color=colors, width=0.7)
        ax.set_xticks(np.arange(len(cc)), cc["label"], rotation=90, fontsize=6)
        ax.axhline(0, color=AXIS_COLOR, lw=0.8)
        ax.set_ylabel("diferença de MAE (p.p.)")
        ax.set_title("(b) MAE de teste da V2: principal − coorte", loc="left", fontsize=8)
        return fig, cc

    # 2. comparação de modelos ---------------------------------------------------------
    def _ranking(self, ax, frame, metric, h):
        part = frame[frame["horizon_steps"] == h].sort_values(metric, ascending=False)
        y = np.arange(len(part))
        ax.barh(y, part[metric], color=[self.col(m) for m in part["model"]], height=0.7)
        ax.set_yticks(y, [self.lab(m) for m in part["model"]], fontsize=7)
        pers = part.loc[part["model"] == "persistence", metric].iloc[0]
        ax.axvline(pers, color=MUTED, ls="--", lw=1)
        for yi, v in zip(y, part[metric]):
            ax.text(v, yi, f" {v:.2f}", va="center", fontsize=6.5, color=INK_2)
        ax.set_xlim(0, part[metric].max() * 1.18)
        ax.grid(axis="y", visible=False)

    def f05(self):
        m = self.metrics()
        fig, axes = style.figure(6.8, 5.9, nrows=2, ncols=3)
        for j, h in enumerate((1, 3, 6)):
            self._ranking(axes[0, j], m, "mae_pp", h)
            axes[0, j].set_title(f"MAE · {HORIZON_LABEL[h]}", loc="left")
            self._ranking(axes[1, j], m, "rmse_pp", h)
            axes[1, j].set_title(f"RMSE · {HORIZON_LABEL[h]}", loc="left")
            axes[1, j].set_xlabel("p.p. de CPU")
        handles = [Patch(color=c, label=f) for f, c in style.FAMILY_COLORS.items() if f != "V3 quantílico"]
        fig.legend(handles=[*handles, Line2D([], [], color=MUTED, ls="--", label="nível da persistência")], loc="outside lower center", ncols=5)
        fig.suptitle("Ranking no teste final (configuração A, população principal)")
        return fig, m[["model", "horizon_steps", "mae_pp", "rmse_pp", "family"]]

    def _heat(self, ax, matrix, cmap, fmt, center=None, vmin=None, vmax=None):
        values = matrix.to_numpy(dtype=float)
        if center is not None:
            span = np.nanmax(np.abs(values - center))
            vmin, vmax = center - span, center + span
        im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(matrix.shape[1]), matrix.columns, fontsize=7)
        ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=7)
        ax.grid(False)
        lo, hi = im.get_clim()
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if np.isfinite(v):
                    rel = (v - lo) / (hi - lo) if hi > lo else 0.5
                    dark = rel > 0.65 if center is None else abs(rel - 0.5) > 0.32
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5, color="white" if dark else OBSERVED)
        return im

    def f06(self):
        m = self.metrics()
        models = self.models_in(m)
        mae = m.pivot(index="model", columns="horizon_steps", values="mae_pp").reindex(models)
        imp = m.pivot(index="model", columns="horizon_steps", values="improvement_vs_persistence_pct").reindex(models)
        for frame in (mae, imp):
            frame.index = [self.lab(x) for x in frame.index]
            frame.columns = [HORIZON_LABEL[c] for c in frame.columns]
        fig, axes = style.figure(6.6, 3.8, ncols=2)
        im = self._heat(axes[0], mae, style.SEQUENTIAL, "{:.2f}")
        fig.colorbar(im, ax=axes[0], shrink=0.8, label="MAE (p.p.)")
        axes[0].set_title("(a) MAE de teste (A)", loc="left")
        im = self._heat(axes[1], imp, style.DIVERGING.reversed(), "{:+.1f}", center=0.0)
        fig.colorbar(im, ax=axes[1], shrink=0.8, label="melhoria contra a persistência (%)")
        axes[1].set_title("(b) Melhoria contra a persistência", loc="left")
        axes[1].set_yticklabels([])
        return fig, m[["model", "horizon_steps", "mae_pp", "improvement_vs_persistence_pct"]]

    def f07(self):
        v = self.metrics("validation_pooled").set_index(["model", "horizon_steps"])
        t = self.metrics().set_index(["model", "horizon_steps"])
        fig, axes = style.figure(6.8, 4.0, ncols=3, sharey=True)
        models = self.models_in(t.reset_index())[::-1]
        rows = []
        for ax, h in zip(axes, (1, 3, 6)):
            for i, model in enumerate(models):
                a, b = v.loc[(model, h), "mae_pp"], t.loc[(model, h), "mae_pp"]
                c = self.col(model)
                ax.plot([a, b], [i, i], color=c, lw=1.2)
                ax.plot(a, i, "o", mfc="white", mec=c, ms=5)
                ax.plot(b, i, "o", color=c, ms=5)
                rows.append({"model": model, "horizon_steps": h, "validation_pooled_mae_pp": a, "test_mae_pp": b})
            ax.set_title(HORIZON_LABEL[h], loc="left")
            ax.set_xlabel("MAE (p.p.)")
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(range(len(models)), [self.lab(m) for m in models], fontsize=7)
        handles = [Line2D([], [], marker="o", mfc="white", mec=INK_2, ls="", label="validação externa (720 origens)"), Line2D([], [], marker="o", color=INK_2, ls="", label="teste final (288 origens)")]
        fig.legend(handles=handles, loc="outside lower center", ncols=2)
        fig.suptitle("Validação externa contra teste final (V2 e V3, configuração A)")
        return fig, pd.DataFrame(rows)

    def f08(self):
        v = self.metrics("validation_pooled")
        t = self.metrics()
        folds = self.t["model_metrics_by_fold"]
        folds = folds[(folds["population"] == "main") & (folds["analysis_set"] == "main") & (folds["mode"] == "direct") & folds["config"].isin(["A", "-"])]
        fig, axes = style.figure(7.6, 5.2, nrows=2, ncols=2)
        for ax, frame, title in ((axes[0, 0], v, "(a) Validação externa agregada"), (axes[0, 1], t, "(b) Teste final")):
            for model in self.models_in(frame):
                part = frame[frame["model"] == model].sort_values("horizon_steps")
                emph = model in self.highlight
                ax.plot(part["horizon_min"], part["mae_pp"], color=self.col(model), marker=style.MARKERS.get(model, "o"), ms=4, lw=1.6 if emph else 0.8, alpha=1 if emph else 0.55, label=self.lab(model))
            ax.set_xticks([5, 15, 30])
            ax.set_xlabel("horizonte (min)")
            ax.set_ylabel("MAE (p.p.)")
            ax.set_title(title, loc="left")
        fig.legend(*axes[0, 1].get_legend_handles_labels(), loc="outside right center", fontsize=6.5)
        for ax, h in ((axes[1, 0], 1), (axes[1, 1], 6)):
            part = folds[folds["horizon_steps"] == h]
            for model in self.models_in(part):
                p = part[part["model"] == model].sort_values("split")
                x = p["split"].str.replace("fold_", "").astype(int)
                emph = model in self.highlight
                ax.plot(x, p["mae_pp"], color=self.col(model) if emph else "#c3c2b7", marker=style.MARKERS.get(model, "o") if emph else None, ms=3.5, lw=1.4 if emph else 0.7, label=self.lab(model) if emph else None, zorder=3 if emph else 1)
            ax.set_xticks(range(5), [f"fold {k}" for k in range(5)])
            ax.set_ylabel("MAE (p.p.)")
            ax.set_title(f"({'c' if h == 1 else 'd'}) MAE por fold · {HORIZON_LABEL[h]}", loc="left")
        fig.suptitle("Erro por horizonte e por fold (em (c) e (d), cinza = modelos fora do destaque)")
        return fig, pd.concat([v.assign(split="validation_pooled"), t])[["model", "split", "horizon_min", "mae_pp"]]

    def f09(self):
        e = self.test_errors()
        m = self.metrics()
        fig, axes = style.figure(6.8, 5.6, nrows=2, ncols=2)
        models = self.models_in(e)
        for ax, h in zip([axes[0, 0], axes[0, 1], axes[1, 0]], (1, 3, 6)):
            data = [e[(e["model"] == mo) & (e["horizon_steps"] == h)]["abs_error_pp"].to_numpy() for mo in models]
            bp = ax.boxplot(data, orientation="horizontal", whis=(5, 95), showfliers=True, flierprops={"marker": ".", "ms": 2, "mec": MUTED}, patch_artist=True, medianprops={"color": OBSERVED}, widths=0.6)
            for patch, mo in zip(bp["boxes"], models):
                patch.set_facecolor(self.col(mo))
                patch.set_alpha(0.55)
                patch.set_edgecolor(self.col(mo))
            ax.set_yticks(range(1, len(models) + 1), [self.lab(mo) for mo in models], fontsize=6.5)
            ax.set_xlabel("erro absoluto (p.p.)")
            ax.set_title(f"Erro absoluto · {HORIZON_LABEL[h]}", loc="left")
            ax.grid(axis="y", visible=False)
        ax = axes[1, 1]
        for _, r in m.iterrows():
            ax.plot(r["mae_pp"], r["rmse_pp"], marker=HORIZON_MARKERS[r["horizon_steps"]], color=self.col(r["model"]), ms=5, ls="")
        x = np.linspace(m["mae_pp"].min() * 0.95, m["mae_pp"].max() * 1.05, 10)
        ax.plot(x, x * np.sqrt(np.pi / 2), color=MUTED, ls=":", lw=1, label="RMSE/MAE = √(π/2) (erro gaussiano)")
        ax.set_xlabel("MAE (p.p.)")
        ax.set_ylabel("RMSE (p.p.)")
        ax.set_title("MAE × RMSE", loc="left")
        handles = [Line2D([], [], marker=HORIZON_MARKERS[h], color=INK_2, ls="", label=HORIZON_LABEL[h]) for h in (1, 3, 6)]
        ax.legend(handles=[*handles, Line2D([], [], color=MUTED, ls=":", label="√(π/2) · MAE")], fontsize=6.5)
        fig.suptitle("Distribuição do erro no teste (caixa P25–P75, haste P5–P95, pontos além)")
        return fig, e[["model", "horizon_steps", "origin_index", "abs_error_pp"]]

    # 3. diagnóstico temporal ------------------------------------------------------------
    def _series(self, ax, h, models, start=None, end=None):
        e = self.test_errors()
        part = e[e["horizon_steps"] == h]
        if start is not None:
            part = part[(part["target_index"] >= start) & (part["target_index"] <= end)]
        obs = part[part["model"] == "persistence"].sort_values("target_index")
        ax.plot(obs["target_time_h"], obs["y_obs_cpu_pct"], color=OBSERVED, lw=1.6, label="observado")
        for model in models:
            p = part[part["model"] == model].sort_values("target_index")
            ax.plot(p["target_time_h"], p["y_pred_cpu_pct"], color=self.col(model), lw=1.0, label=self.lab(model), ls="--" if model == "persistence" else "-")
        return part[part["model"].isin(models)]

    def f11(self):
        fig, axes = style.figure(6.8, 6.4, nrows=3, sharex=True)
        data = []
        for ax, h in zip(axes, (1, 3, 6)):
            self.shade_windows(ax)
            data.append(self._series(ax, h, self.highlight))
            ax.set_ylabel(CPU_LABEL)
            ax.set_title(f"Horizonte {HORIZON_LABEL[h]} (eixo no instante previsto)", loc="left")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=5)
        axes[-1].set_xlabel(TIME_LABEL)
        fig.suptitle("CPU observada e prevista no teste final; faixas cinza = janelas objetivas")
        return fig, pd.concat(data)

    def f12(self):
        windows = self.t["diagnostic_windows"]
        fig, axes = style.figure(6.8, 4.6, nrows=2, ncols=len(windows), sharey="row")
        data = []
        for j, (_, w) in enumerate(windows.iterrows()):
            for i, h in enumerate((1, 6)):
                ax = axes[i, j]
                data.append(self._series(ax, h, self.highlight, w["start_index"], w["end_index"]).assign(window=w["window"]))
                if i == 0:
                    ax.set_title(f"{w['window']}", loc="left")
                if j == 0:
                    ax.set_ylabel(f"{HORIZON_LABEL[h]}\n{CPU_LABEL}")
                ax.tick_params(axis="x", labelsize=6.5)
        axes[0, 0].legend(fontsize=6, loc="upper left")
        fig.supxlabel(TIME_LABEL, fontsize=8)
        fig.suptitle("Janelas de 3 h escolhidas só pela série observada (sem olhar o erro dos modelos)")
        return fig, pd.concat(data)

    def f13(self):
        e = self.test_errors()
        acf = self.t["residual_autocorrelation"]
        acf = acf[acf["config"].isin(["A", "-"]) & acf["model"].isin(self.highlight)]
        fig, axes = style.figure(6.8, 5.0, nrows=2, ncols=2, height_ratios=[1.2, 1])
        for ax, h in zip(axes[0], (1, 6)):
            part = e[e["horizon_steps"] == h]
            for model in self.highlight:
                p = part[part["model"] == model].sort_values("target_index")
                ax.plot(p["target_time_h"], p["error_obs_minus_pred_pp"], color=self.col(model), lw=0.8, label=self.lab(model))
            ax.axhline(0, color=AXIS_COLOR, lw=0.8)
            ax.set_xlabel(TIME_LABEL, fontsize=7)
            ax.set_ylabel("observado − previsto (p.p.)")
            ax.set_title(f"Resíduo ao longo do teste · {HORIZON_LABEL[h]}", loc="left")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=4)
        for ax, h in zip(axes[1], (1, 6)):
            part = acf[acf["horizon_steps"] == h]
            for model in self.highlight:
                p = part[part["model"] == model]
                ax.plot(p["lag_steps"], p["acf"], color=self.col(model), marker=style.MARKERS.get(model, "o"), ms=2.5, lw=1)
            band = part["band_95"].iloc[0]
            ax.axhspan(-band, band, color="#f0efec", zorder=0)
            ax.axvline(h - 0.5, color=MUTED, ls=":", lw=1)
            ax.set_xlabel("defasagem (passos de 5 min)")
            ax.set_ylabel("autocorrelação")
            ax.set_title(f"ACF do resíduo · {HORIZON_LABEL[h]}", loc="left")
        return fig, acf

    def f14(self):
        r = self.t["errors_by_regime"]
        r = r[(r["grouping"] == "cpu_band") & r["config"].isin(["A", "-"])]
        models = self.models_in(r)
        fig, axes = style.figure(6.8, 5.6, nrows=2, ncols=3)
        for j, h in enumerate((1, 3, 6)):
            part = r[r["horizon_steps"] == h]
            for i, (value, cmap, fmt, center, title) in enumerate((("mae_pp", style.SEQUENTIAL, "{:.1f}", None, "MAE"), ("bias_obs_minus_pred_pp", style.DIVERGING, "{:+.1f}", 0.0, "viés"))):
                mat = part.pivot(index="model", columns="group", values=value).reindex(models)
                mat.index = [self.lab(m) for m in mat.index]
                mat.columns = [c.split(" ")[0] for c in mat.columns]
                im = self._heat(axes[i, j], mat, cmap, fmt, center=center)
                axes[i, j].set_title(f"{title} · {HORIZON_LABEL[h]}", loc="left")
                if j:
                    axes[i, j].set_yticklabels([])
            fig.colorbar(im, ax=axes[1, j], shrink=0.7, orientation="horizontal", label="viés (p.p.; + = subestima)")
            n = part[part["model"] == "persistence"].set_index("group")["n"]
            axes[1, j].set_xticks(range(len(n)), [f"{g.split(' ')[0]}\nn={int(v)}" for g, v in n.sort_index().items()], fontsize=6.5)
        fig.suptitle("Erro por faixa de CPU observada no alvo (Q1–Q5 = quintis da CPU no desenvolvimento)")
        return fig, r

    def f15(self):
        r = self.t["errors_by_regime"]
        r = r[r["config"].isin(["A", "-"])]
        models = ["persistence", "xgboost_v2", "xgboost_delta", "statespace", "hybrid", self.best_neural]
        fig, axes = style.figure(6.8, 4.6, ncols=3, width_ratios=[1.1, 1.1, 1])
        bins = self.info.get("diagnostics", {}).get("change_bin_labels", [])
        for ax, grouping, title in ((axes[0], "realized_change_bin", "(a) variação realizada até o alvo"), (axes[1], "recent_change_1_step_bin", "(b) variação recente (na origem)")):
            part = r[(r["grouping"] == grouping) & (r["horizon_steps"] == 6)]
            for model in models:
                p = part[part["model"] == model].set_index("group").reindex(bins)
                ax.plot(range(len(bins)), p["mae_pp"], color=self.col(model), marker=style.MARKERS.get(model, "o"), ms=3.5, lw=1.2, label=self.lab(model), ls="--" if model in ("xgboost_delta", "statespace") else "-")
            ax.set_xticks(range(len(bins)), bins, rotation=45, fontsize=6.5)
            ax.set_xlabel("faixa de variação (p.p.)")
            ax.set_ylabel("MAE · 30 min (p.p.)")
            ax.set_title(title, loc="left", fontsize=8)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=6, fontsize=6.5)
        ax = axes[2]
        part = r[(r["grouping"] == "regime") & (r["horizon_steps"] == 6)]
        regimes = ["queda", "estável", "subida"]
        width = 0.8 / len(models)
        for k, model in enumerate(models):
            p = part[part["model"] == model].set_index("group").reindex(regimes)
            ax.bar(np.arange(3) + k * width - 0.4 + width / 2, p["mae_pp"], width=width, color=self.col(model), edgecolor="white", lw=0.5, hatch="//" if model in ("xgboost_delta", "statespace") else None)
        n = part[part["model"] == "persistence"].set_index("group").reindex(regimes)["n"]
        ax.set_xticks(range(3), [f"{g}\n(n={int(v)})" for g, v in zip(regimes, n)])
        ax.set_ylabel("MAE · 30 min (p.p.)")
        ax.set_title("(c) regime (±σ do desenv.)", loc="left", fontsize=8)
        fig.suptitle("Erro em 30 min e variação da CPU (tracejado e hachura = previsão residual)")
        return fig, r[r["model"].isin(models)]

    # 4. picos ---------------------------------------------------------------------------
    def f16(self):
        events = self.t["peak_events"]
        events = events[events["threshold_kind"] == "desenvolvimento_p90"].sort_values("peak_cpu_pct", ascending=False).head(3).sort_values("start_index")
        thr = self.info["dev_p90"]
        fig, axes = style.figure(6.8, 4.6, nrows=2, ncols=len(events), sharey=True)
        data = []
        for j, (_, ev) in enumerate(events.iterrows()):
            start, end = int(ev["start_index"]) - 12, int(ev["end_index"]) + 12
            for i, h in enumerate((1, 6)):
                ax = axes[i, j]
                data.append(self._series(ax, h, self.highlight, start, end).assign(event_id=ev["event_id"]))
                ax.axhline(thr, color="#e34948", ls=":", lw=1)
                ax.tick_params(axis="x", labelsize=6.5)
                if i == 0:
                    ax.set_title(f"evento com pico {ev['peak_cpu_pct']:.1f}%", loc="left", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{HORIZON_LABEL[h]}\n{CPU_LABEL}")
        axes[0, -1].legend(handles=[*axes[0, 0].get_legend_handles_labels()[0], Line2D([], [], color="#e34948", ls=":", label=f"P90 do desenvolvimento ({thr:.1f}%)")], fontsize=6, loc="upper right")
        fig.supxlabel(TIME_LABEL, fontsize=8)
        fig.suptitle("Três maiores picos do teste (±1 h)")
        return fig, pd.concat(data)

    def f17(self):
        p = self.t["peak_analysis"]
        p = p[p["config"].isin(["A", "-"]) & (p["threshold_kind"] == "desenvolvimento_p90")]
        e = self.test_errors()
        thr = self.info["dev_p90"]
        models = self.models_in(p)
        fig, axes = style.figure(6.8, 4.4, ncols=3)
        width = 0.27
        for k, h in enumerate((1, 3, 6)):
            part = p[p["horizon_steps"] == h].set_index("model").reindex(models)
            for ax, col in ((axes[0], "peak_bias_obs_minus_pred_pp"), (axes[1], "peak_underestimation_frequency")):
                ax.barh(np.arange(len(models)) + (k - 1) * width, part[col], height=width, color=["#cde2fb", "#6da7ec", "#184f95"][k], label=HORIZON_LABEL[h])
        for ax, xlabel, title in ((axes[0], "viés (p.p.; + = subestima)", "(a) Viés nos picos"), (axes[1], "fração subestimada", "(b) Frequência")):
            ax.set_yticks(range(len(models)), [self.lab(m) for m in models], fontsize=6.5)
            ax.set_xlabel(xlabel)
            ax.set_title(title, loc="left", fontsize=8)
            ax.grid(axis="y", visible=False)
        axes[1].set_yticklabels([])
        axes[1].axvline(0.5, color=MUTED, ls=":", lw=1)
        axes[0].legend(fontsize=6.5, loc="lower right")
        ax = axes[2]
        part = e[(e["horizon_steps"] == 6) & (e["y_obs_cpu_pct"] >= thr)]
        data = [part[part["model"] == m]["error_obs_minus_pred_pp"].to_numpy() for m in models]
        bp = ax.boxplot(data, orientation="horizontal", whis=(5, 95), patch_artist=True, medianprops={"color": OBSERVED}, flierprops={"marker": ".", "ms": 2}, widths=0.6)
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(self.col(m))
            patch.set_alpha(0.55)
        ax.axvline(0, color=AXIS_COLOR, lw=0.8)
        ax.set_yticks(range(1, len(models) + 1), [""] * len(models))
        ax.set_xlabel("observado − previsto (p.p.)")
        ax.set_title(f"(c) Erro · 30 min (n={len(data[0])})", loc="left", fontsize=8)
        fig.suptitle(f"Picos = CPU observada ≥ P90 do desenvolvimento ({thr:.1f}%)")
        return fig, p

    def f18(self):
        a = self.t["peak_alerts"]
        a = a[a["threshold_kind"] == "desenvolvimento_p90"].copy()
        a["label"] = [x.replace("xgboost_quantile ", "XGBoost ") if k == "quantil" else self.lab(x) for x, k in zip(a["signal"], a["signal_kind"])]
        order = [self.lab(m) for m in self.order if m in set(a["signal"])] + [x for x in a.loc[a["signal_kind"] == "quantil", "label"].unique()]
        fig, axes = style.figure(6.8, 4.6, ncols=3, sharey=True)
        y = np.arange(len(order))
        for ax, h in zip(axes, (1, 3, 6)):
            part = a[a["horizon_steps"] == h].set_index("label").reindex(order)
            ax.barh(y - 0.2, part["n_anticipated"], height=0.38, color="#2a78d6", label="eventos antecipados")
            ax.barh(y + 0.2, part["n_false_alarm_episodes"], height=0.38, color="#eb6834", label="episódios de falso alarme")
            ax.set_title(f"{HORIZON_LABEL[h]} · {int(part['n_events_evaluable'].iloc[0])} eventos", loc="left")
            ax.set_xlabel("contagem")
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(y, order, fontsize=6.5)
        axes[0].invert_yaxis()
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
        fig.suptitle("Antecipação de picos acima do P90 do desenvolvimento (tolerância de 1 passo)")
        return fig, a

    # 5. estatística ---------------------------------------------------------------------
    def f19(self):
        b = self.t["bootstrap_intervals"]
        b = b[(b["analysis_type"] == "pre_registrada") & (b["loss"] == "erro_absoluto") & b["comparison"].isin(["new_vs_v2_xgboost_A", "new_vs_persistence_A", "hybrid_vs_statespace", "d_vs_a"])]
        keys = ["comparison", "model_a", "config_a", "model_b", "config_b"]
        rows = b[keys].drop_duplicates().reset_index(drop=True)
        names = {"new_vs_v2_xgboost_A": "× XGBoost V2", "new_vs_persistence_A": "× persistência", "hybrid_vs_statespace": "× Kalman", "d_vs_a": "D × A"}
        labels = [f"{self.lab(r.model_a)} {names[r.comparison]}" for r in rows.itertuples()]
        fig, axes = style.figure(6.8, 5.6, ncols=3, sharey=True)
        offsets = {6: -0.22, 12: 0.0, 24: 0.22}
        shades = {6: "#9ec5f4", 12: "#184f95", 24: "#6da7ec"}
        for ax, h in zip(axes, (1, 3, 6)):
            part = b[b["horizon_steps"] == h]
            for i, r in rows.iterrows():
                q = part.merge(r.to_frame().T, on=keys)
                for _, x in q.iterrows():
                    y = i + offsets[int(x["block_steps"])]
                    sig = x["significant"]
                    ax.plot([x["ci_low"], x["ci_high"]], [y, y], color=shades[int(x["block_steps"])], lw=2.2 if x["block_steps"] == 12 else 1.2)
                    if x["block_steps"] == 12:
                        ax.plot(x["mean_diff"], y, "o", ms=4, color="#184f95" if sig else "white", mec="#184f95")
            ax.axvline(0, color="#e34948", lw=0.9)
            ax.set_title(HORIZON_LABEL[h], loc="left")
            ax.set_xlabel("Δ MAE (p.p.)")
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(range(len(rows)), labels, fontsize=6.5)
        axes[0].invert_yaxis()
        handles = [Line2D([], [], color=shades[k], lw=2, label=f"bloco {k * 5} min") for k in (6, 12, 24)] + [Line2D([], [], marker="o", color="#184f95", ls="", label="média (cheio = IC sem zero)")]
        fig.legend(handles=handles, loc="outside lower center", ncols=4, fontsize=6.5)
        fig.suptitle("Comparações pré-registradas: IC 95% (block bootstrap, 5.000 reamostragens); Δ < 0 favorece o primeiro modelo", fontsize=8.5)
        return fig, b

    def f20(self):
        s = self.t["statistical_comparisons"]
        s = s[s["analysis_type"] == "exploratoria"]
        models = [m for m in self.order if m in set(s["model_a"]) | set(s["model_b"])]
        span = float(s["mean_diff"].abs().max())
        fig, axes = style.figure(7.2, 3.9, ncols=3)
        for ax, h in zip(axes, (1, 3, 6)):
            part = s[s["horizon_steps"] == h]
            mat = pd.DataFrame(np.nan, index=models, columns=models)
            mark = pd.DataFrame("", index=models, columns=models)
            for _, r in part.iterrows():
                mat.loc[r["model_a"], r["model_b"]] = r["mean_diff"]
                mat.loc[r["model_b"], r["model_a"]] = -r["mean_diff"]
                sym = "●" if r["holm_significant"] else ("∘" if r["significant"] else "")
                mark.loc[r["model_a"], r["model_b"]] = mark.loc[r["model_b"], r["model_a"]] = sym
            im = ax.imshow(mat.to_numpy(dtype=float), cmap=style.DIVERGING, vmin=-span, vmax=span)
            for i in range(len(models)):
                for j in range(len(models)):
                    if mark.iat[i, j]:
                        ax.text(j, i, mark.iat[i, j], ha="center", va="center", fontsize=5.5, color=OBSERVED)
            ax.set_xticks(range(len(models)), [self.lab(m) for m in models], rotation=90, fontsize=5.8)
            ax.set_yticks(range(len(models)), [self.lab(m) for m in models] if h == 1 else [""] * len(models), fontsize=5.8)
            ax.grid(False)
            ax.set_title(HORIZON_LABEL[h], loc="left")
        fig.colorbar(im, ax=axes, shrink=0.75, label="MAE linha − MAE coluna (p.p.)")
        fig.suptitle("Matriz exploratória (bloco 1 h): ∘ IC 95% sem zero; ● significativo após Holm no horizonte", fontsize=8.5)
        return fig, s

    # 6. redes neurais ------------------------------------------------------------------
    def f22(self):
        h = self.t["neural_training_history"]
        h = h[(h["phase"] == "selection") & h["is_selected_grid"] & (h["config"] == "A") & (h["mode"] == "direct")]
        archs = [m for m in self.cfg["models"]["neural"] if m in set(h["model"])]
        fig, axes = style.figure(6.8, 5.8, nrows=2, ncols=len(archs), sharey="row")
        for j, arch in enumerate(archs):
            for i, hor in enumerate(("1", "6")):
                ax = axes[i, j]
                part = h[(h["model"] == arch) & (h["horizon"] == hor)]
                for _, f in part.groupby("fold"):
                    ax.plot(f["epoch"], f["train_loss_mse_standardized"], color="#2a78d6", lw=0.8, alpha=0.8)
                    ax.plot(f["epoch"], f["val_loss_mse_standardized"], color="#eb6834", lw=0.8, alpha=0.8)
                    best = f.loc[f["val_loss_mse_standardized"].idxmin()]
                    ax.plot(best["epoch"], best["val_loss_mse_standardized"], "o", color="#eb6834", ms=3)
                ax.set_yscale("log")
                if i == 0:
                    ax.set_title(self.lab(arch), loc="left")
                if j == 0:
                    ax.set_ylabel(f"{HORIZON_LABEL[int(hor)]}\nMSE padronizado (log)")
                ax.set_xlabel("época", fontsize=7)
        fig.legend(handles=[Line2D([], [], color="#2a78d6", label="treino"), Line2D([], [], color="#eb6834", label="validação interna"), Line2D([], [], marker="o", color="#eb6834", ls="", label="melhor época")], loc="outside lower center", ncols=3)
        fig.suptitle("Loss por época na seleção (ponto escolhido da grade; uma linha por fold; A)")
        return fig, h

    def f23(self):
        g = self.t["generalization"]
        w = self.t.get("neural_window_selection")
        fig, axes = style.figure(7.0, 4.2, ncols=3, width_ratios=[1.0, 1.3, 0.8])
        ax = axes[0]
        for _, r in g.iterrows():
            ax.plot(r["selection_mae_pp"], r["test_mae_pp"], marker=HORIZON_MARKERS[r["horizon_steps"]], color=self.col(r["model"]), ms=5, ls="")
        lim = [g[["selection_mae_pp", "test_mae_pp"]].min().min() * 0.95, g[["selection_mae_pp", "test_mae_pp"]].max().max() * 1.03]
        ax.plot(lim, lim, color=MUTED, ls=":", lw=1, label="teste = seleção")
        ax.set_xlabel("MAE de seleção (p.p.)")
        ax.set_ylabel("MAE de teste (p.p.)")
        ax.set_title("(a) Seleção × teste", loc="left", fontsize=8)
        ax.legend(handles=[Line2D([], [], marker=HORIZON_MARKERS[h], color=INK_2, ls="", label=HORIZON_LABEL[h]) for h in (1, 3, 6)], fontsize=6)
        ax = axes[1]
        models = self.models_in(g)
        width = 0.27
        for k, h in enumerate((1, 3, 6)):
            part = g[g["horizon_steps"] == h].set_index("model").reindex(models)
            ax.barh(np.arange(len(models)) + (k - 1) * width, part["normalized_generalization_ratio"] - 1, left=1, height=width, color=["#cde2fb", "#6da7ec", "#184f95"][k], label=HORIZON_LABEL[h])
        ax.axvline(1, color=OBSERVED, lw=0.8)
        ax.set_yticks(range(len(models)), [self.lab(m) for m in models], fontsize=6.5)
        ax.set_xlabel("razão normalizada pela persistência")
        ax.set_title("(b) (teste/seleção) ÷ persistência", loc="left", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(axis="y", visible=False)
        ax = axes[2]
        if w is not None:
            for h, part in w.groupby("horizon_steps"):
                ax.plot(part["window_hours"], part["inner_mae_pp"], marker=HORIZON_MARKERS[h], color=["#9ec5f4", "#3987e5", "#104281"][[1, 3, 6].index(h)], label=HORIZON_LABEL[h])
            ax.set_xticks([3, 6, 12])
            ax.set_xlabel("janela de entrada (h)")
            ax.set_ylabel("MAE interno do DLinear (p.p.)")
            ax.set_title("(c) Janela", loc="left", fontsize=8)
            ax.legend(fontsize=6)
        return fig, g

    def f24(self):
        c = self.t["cost_tradeoff"]
        fig, axes = style.figure(6.8, 5.0, nrows=2, ncols=3)
        for i, h in enumerate((1, 6)):
            part = c[c["horizon_steps"] == h]
            for j, (col, xlabel, log) in enumerate((("size_measure", "tamanho (parâmetros, coeficientes ou folhas)", True), ("train_seconds_test_refit", "tempo do reajuste de teste (s)", True), ("peak_cuda_memory_mib", "pico de memória CUDA (MiB)", False))):
                ax = axes[i, j]
                sub = part if col != "peak_cuda_memory_mib" else part[part["device"] == "cuda"]
                floor = 0.5 if col == "size_measure" else 1e-3
                for _, r in sub.iterrows():
                    x = max(r[col], floor) if log else r[col]
                    ax.plot(x, r["test_mae_pp"], marker=style.MARKERS.get(r["model"], "o"), color=self.col(r["model"]), ms=5, ls="", label=self.lab(r["model"]))
                if col == "train_seconds_test_refit":
                    front = sub[sub["pareto_time_mae"]].sort_values(col)
                    ax.step(np.maximum(front[col], floor), front["test_mae_pp"], where="post", color=MUTED, ls="--", lw=1)
                if log:
                    ax.set_xscale("log")
                ax.set_xlabel(xlabel, fontsize=7)
                if j == 0:
                    ax.set_ylabel(f"{HORIZON_LABEL[h]}\nMAE de teste (p.p.)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=6, fontsize=6.5)
        fig.suptitle("Custo × erro de teste (A); persistência no piso do eixo log; tracejado = fronteira de Pareto", fontsize=8.5)
        return fig, c

    # 7. ablação e importância ---------------------------------------------------------
    def f25(self):
        g = self.t["ablation_grid"]
        ab = self.t["ablation_results"]
        models = [m for m in self.order if m in set(ab["model"])]
        cols = [(c, h) for c in ("B", "C", "D") for h in (1, 3, 6)]
        mat = pd.DataFrame(np.nan, index=models, columns=pd.MultiIndex.from_tuples(cols))
        for _, r in g[g["executed"]].iterrows():
            if r["model"] in models:
                mat.loc[r["model"], (r["config"], r["horizon_steps"])] = r["delta_mae_pct"]
        fig, ax = style.figure(6.8, 3.6)
        values = mat.to_numpy(dtype=float)
        span = np.nanmax(np.abs(values))
        im = ax.imshow(values, cmap=style.DIVERGING, vmin=-span, vmax=span, aspect="auto")
        sig = ab[ab["population"] == "main"].set_index(["model", "config", "horizon_steps"])
        for i, m in enumerate(models):
            for j, (c, h) in enumerate(cols):
                v = values[i, j]
                if np.isfinite(v):
                    star = "*" if (m, c, h) in sig.index and sig.loc[(m, c, h), "significant_block_1h"] is True else ""
                    text = "0" if abs(v) < 0.5 else f"{v:+.0f}"
                    ax.text(j, i, f"{text}{star}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > span * 0.6 else OBSERVED)
                else:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", edgecolor="#c3c2b7", lw=0))
        ax.set_xticks(range(len(cols)), [f"{c}\n{HORIZON_LABEL[h]}" for c, h in cols], fontsize=6.5)
        ax.set_yticks(range(len(models)), [self.lab(m) for m in models], fontsize=7)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Δ MAE contra A (%); + = piora")
        ax.set_title("Ablação no teste (hachura = não executado; * = IC 95% pré-registrado sem zero)", loc="left", fontsize=8)
        return fig, g

    def f26(self):
        f = self.t["feature_importance"]
        share = f.groupby(["model", "config", "horizon_steps", "group"])["share"].sum().unstack(fill_value=0.0)
        groups = ["CPU na origem", "CPU defasada", "CPU janela móvel", "calendário", "memória", "rede", "disco"]
        groups = [g for g in groups if g in share.columns]
        colors = ["#184f95", "#3987e5", "#9ec5f4", "#898781", "#eb6834", "#1baf7a", "#6250d6"]
        share = share[groups]
        labels = [f"{self.lab(m)} {c} · {HORIZON_LABEL[h]}" for m, c, h in share.index]
        fig, ax = style.figure(6.8, 5.2)
        left = np.zeros(len(share))
        for g, c in zip(groups, colors):
            ax.barh(range(len(share)), share[g], left=left, color=c, label=g, height=0.75, edgecolor="white", lw=0.6)
            left += share[g].to_numpy()
        ax.set_yticks(range(len(share)), labels, fontsize=6.3)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("fração da soma das médias de |SHAP| no teste")
        ax.grid(axis="y", visible=False)
        ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=6.5)
        ax.set_title("Importância TreeSHAP por grupo (associação preditiva, não causalidade)", loc="left")
        return fig, share.reset_index()

    # 8. quantis -----------------------------------------------------------------------
    def f27(self):
        q = self.t["quantile_calibration"]
        q = q[q["config"] == "A"]
        fig, axes = style.figure(6.8, 5.2, nrows=2, ncols=2)
        ax = axes[0, 0]
        for (split, variant), part in q.groupby(["split", "variant"]):
            mk = {"validation_pooled": "o", "test_final": "s"}[split]
            c = "#6250d6" if variant == "ajustado" else ("#2a78d6" if split == "validation_pooled" else "#eb6834")
            ax.plot(part["nominal_coverage"], part["empirical_coverage"], marker=mk, color=c, ls="", ms=5, mfc="white" if variant == "bruto" and split == "test_final" else c, label=f"{'validação' if split == 'validation_pooled' else 'teste'} · {variant}")
        ax.plot([0.45, 1], [0.45, 1], color=MUTED, ls=":", lw=1)
        ax.set_xlabel("cobertura nominal P(y ≤ q)")
        ax.set_ylabel("cobertura empírica")
        ax.set_title("(a) Calibração (pontos = horizontes)", loc="left", fontsize=8)
        ax.legend(fontsize=6)
        test = q[q["split"] == "test_final"]
        for ax, col, title in ((axes[0, 1], "coverage_gap", "(b) Empírica − nominal no teste"), (axes[1, 1], "peak_coverage", "(d) Cobertura nos picos (≥ P90 do desenv.)")):
            for (name, variant), part in test.groupby(["quantile", "variant"]):
                ax.plot(part["horizon_min"], part[col], marker="o" if variant == "bruto" else "s", ls="-" if variant == "bruto" else "--", color={"Q50": "#898781", "Q90": "#2a78d6", "Q95": "#eb6834"}[name], label=f"{name} {variant}", ms=4)
            ax.set_xticks([5, 15, 30])
            ax.set_xlabel("horizonte (min)")
            ax.set_title(title, loc="left", fontsize=8)
        axes[0, 1].axhline(0, color=OBSERVED, lw=0.8)
        axes[0, 1].legend(fontsize=6, ncols=2)
        ax = axes[1, 0]
        for variant, part in test[test["quantile"] == "Q95"].groupby("variant"):
            ax.plot(part["horizon_min"], part["mean_upper_margin_above_q50_pp"], marker="o", label=f"Q95 {variant}", color="#eb6834" if variant == "bruto" else "#6250d6")
        ax.set_xticks([5, 15, 30])
        ax.set_xlabel("horizonte (min)")
        ax.set_ylabel("q95 − q50 médio (p.p.)")
        ax.set_title("(c) Margem unilateral acima da mediana", loc="left", fontsize=8)
        ax.legend(fontsize=6)
        fig.suptitle("XGBoost quantílico (A): cobertura empírica, sem garantia teórica")
        return fig, q

    def f28(self):
        q = self.t["quantile_predictions"]
        q = q[(q["config"] == "A") & (q["split"] == "test_final")]
        fig, axes = style.figure(6.8, 5.0, nrows=2, sharex=True)
        for ax, h in zip(axes, (1, 6)):
            p = q[q["horizon_steps"] == h].sort_values("target_index")
            ax.fill_between(p["target_time_h"], p["q50_cpu_pct"], p["q95_adjusted_cpu_pct"], color="#d9d4f6", lw=0, label="Q50 até Q95 ajustado")
            ax.plot(p["target_time_h"], p["q50_cpu_pct"], color="#6250d6", lw=0.9, label="Q50")
            ax.plot(p["target_time_h"], p["y_obs_cpu_pct"], color=OBSERVED, lw=1.1, label="observado")
            over = p[p["y_obs_cpu_pct"] > p["q95_adjusted_cpu_pct"]]
            ax.plot(over["target_time_h"], over["y_obs_cpu_pct"], "v", color="#e34948", ms=4, label="acima do Q95 ajustado")
            ax.set_ylabel(CPU_LABEL)
            ax.set_title(f"Horizonte {HORIZON_LABEL[h]} · {len(over)} de {len(p)} instantes acima do Q95 ajustado", loc="left")
        axes[-1].set_xlabel(TIME_LABEL)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=4)
        fig.suptitle("Quantis no teste: faixa unilateral entre a mediana e o teto Q95 ajustado")
        return fig, q

    # 9. energia -----------------------------------------------------------------------
    def f29(self):
        fan = self.info["fan"]
        series = self.t["dataset_series"]
        test = series[series["bin_index"] >= self.info["test_range"][0]]
        u = np.linspace(0, 1, 101)
        fig, axes = style.figure(6.8, 4.8, nrows=2, ncols=2)
        ax = axes[0, 0]
        rows = []
        for r, c in zip((1.0, 1.4, 1.8), ("#9ec5f4", "#3987e5", "#104281")):
            p = fan["p_idle_w"] + (fan["p_busy_w"] - fan["p_idle_w"]) * (2 * u - u**r)
            ax.plot(u * 100, p, color=c, label=f"r = {r:g}")
            rows.append(pd.DataFrame({"cpu_pct": u * 100, "r": r, "power_w": p}))
        lo, hi = test["aggregated_cpu_weighted_pct"].quantile([0.0, 1.0])
        ax.axvspan(lo, hi, color="#f0efec", zorder=0, label="faixa da CPU no teste")
        ax.set_xlabel("CPU (%)")
        ax.set_ylabel("potência modelada (W)")
        ax.set_title(f"(a) Modelo de Fan, P_idle={fan['p_idle_w']:g} W, P_busy={fan['p_busy_w']:g} W", loc="left", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")
        ax = axes[0, 1]
        ax.hist(test["aggregated_cpu_weighted_pct"], bins=30, color="#2a78d6")
        ax.set_xlabel("CPU agregada no teste (%)")
        ax.set_ylabel("intervalos")
        ax.set_title("(b) Distribuição da CPU observada no teste", loc="left", fontsize=8)
        et = self.t["energy_timeseries"]
        ref = et[(et["model"] == "persistence") & (et["step"] == 1)].sort_values("target_index")
        ax = axes[1, 0]
        ax.plot(ref["target_time_h"], ref["cpu_obs_pct"], color=OBSERVED, lw=0.9)
        ax.set_ylabel(CPU_LABEL)
        ax.set_xlabel(TIME_LABEL, fontsize=7)
        ax.set_title("(c) CPU observada", loc="left", fontsize=8)
        ax = axes[1, 1]
        for r, c in zip((1.0, 1.4, 1.8), ("#9ec5f4", "#3987e5", "#104281")):
            p = ref[ref["r"] == r]
            ax.plot(p["target_time_h"], p["power_reference_w"], color=c, lw=0.9, label=f"r = {r:g}")
        ax.set_ylabel("potência de referência (W)")
        ax.set_xlabel(TIME_LABEL, fontsize=7)
        ax.set_title("(d) Potência modelada do servidor substituto", loc="left", fontsize=8)
        ax.legend(fontsize=6.5)
        fig.suptitle("Potência de TI modelada (servidor substituto SPECpower_ssj2008; não é medição)")
        return fig, pd.concat(rows)

    def f30(self):
        r = float(self.cfg["energy"]["reference_exponent"])
        et = self.t["energy_timeseries"]
        cum = self.t["energy_cumulative"]
        models = ["xgboost_v2", "hybrid", self.best_neural]
        fig, axes = style.figure(6.8, 6.2, nrows=3)
        ax = axes[0]
        ref = et[(et["model"] == "persistence") & (et["r"] == r) & (et["step"] == 6)].sort_values("target_index")
        ax.plot(ref["target_time_h"], ref["power_reference_w"], color=OBSERVED, lw=1.3, label="referência (CPU observada)")
        for m in models:
            p = et[(et["model"] == m) & (et["r"] == r) & (et["step"] == 6)].sort_values("target_index")
            ax.plot(p["target_time_h"], p["power_projected_w"], color=self.col(m), lw=0.9, label=self.lab(m))
        ax.set_ylabel("potência (W)")
        ax.set_title(f"(a) Potência no passo de 30 min, r = {r:g}", loc="left", fontsize=8)
        ax.legend(ncols=4, fontsize=6.5, loc="upper left")
        for ax, window, title in ((axes[1], 5.0, "(b) Diferença acumulada, 288 janelas de 5 min"), (axes[2], 30.0, "(c) Diferença acumulada, 48 janelas de 30 min sem sobreposição")):
            part = cum[(cum["r"] == r) & (cum["window_min"] == window)]
            for m in self.models_in(part):
                p = part[part["model"] == m]
                emph = m in self.highlight or m == "statespace"
                ax.plot(p["window_end_time_h"], p["cumulative_projected_minus_reference_wh"], color=self.col(m), lw=1.4 if emph else 0.7, alpha=1 if emph else 0.6, marker=style.MARKERS.get(m) if window == 30 else None, ms=2.5, label=self.lab(m))
            ax.axhline(0, color=OBSERVED, lw=0.8)
            ax.set_ylabel("projetada − referência (Wh)")
            ax.set_title(title, loc="left", fontsize=8)
        fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncols=5, fontsize=6.5)
        axes[2].set_xlabel(TIME_LABEL)
        ref_total = cum[(cum["r"] == r) & (cum["window_min"] == 5.0)].groupby("model")["cumulative_reference_wh"].last().iloc[0]
        fig.suptitle(f"Energia de TI modelada no teste: referência de 24 h = {ref_total:,.0f} Wh por servidor substituto".replace(",", "."))
        return fig, cum[cum["r"] == r]

    def f31(self):
        e = self.t["energy_metrics"]
        s = self.t["energy_sensitivity"]
        r_ref = float(self.cfg["energy"]["reference_exponent"])
        models = self.models_in(e)
        fig, axes = style.figure(6.8, 5.6, nrows=2, ncols=2)
        ax = axes[0, 0]
        width = 0.27
        part = e[e["r"] == r_ref]
        for k, h in enumerate((1, 3, 6)):
            p = part[part["horizon_steps"] == h].set_index("model").reindex(models)
            ax.barh(np.arange(len(models)) + (k - 1) * width, 100 * p["energy_mean_abs_relative_error"], height=width, color=["#cde2fb", "#6da7ec", "#184f95"][k], label=HORIZON_LABEL[h])
        ax.set_yticks(range(len(models)), [self.lab(m) for m in models], fontsize=6.5)
        ax.set_xlabel("erro relativo absoluto médio (%)")
        ax.set_title(f"(a) Erro de energia por origem, r = {r_ref:g}", loc="left", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(axis="y", visible=False)
        ax = axes[0, 1]
        for k, h in enumerate((1, 3, 6)):
            p = part[part["horizon_steps"] == h].set_index("model").reindex(models)
            ax.barh(np.arange(len(models)) + (k - 1) * width, p["energy_bias_projected_minus_reference_wh"], height=width, color=["#cde2fb", "#6da7ec", "#184f95"][k])
        ax.axvline(0, color=OBSERVED, lw=0.8)
        ax.set_yticks(range(len(models)), [""] * len(models))
        ax.set_xlabel("viés projetada − referência (Wh por origem)")
        ax.set_title("(b) Viés de energia (< 0 = subestima)", loc="left", fontsize=8)
        ax.grid(axis="y", visible=False)
        ax = axes[1, 0]
        sr = s[s["parameter"] == "r"]
        for m in self.models_in(sr):
            p = sr[sr["model"] == m]
            emph = m in self.highlight
            ax.plot(p["r"], p["energy_mae_wh"], marker=style.MARKERS.get(m, "o"), color=self.col(m), lw=1.4 if emph else 0.7, alpha=1 if emph else 0.6, ms=3.5, label=self.lab(m) if emph else None)
        ax.set_xticks([1.0, 1.4, 1.8])
        ax.set_xlabel("expoente r")
        ax.set_ylabel("MAE de energia · 30 min (Wh)")
        ax.set_title("(c) Sensibilidade a r", loc="left", fontsize=8)
        ax.legend(fontsize=6)
        ax = axes[1, 1]
        for param, ls in (("p_idle", "-"), ("p_busy", "--")):
            sp = s[(s["parameter"] == param) & s["model"].isin(["xgboost_v2", "hybrid", "persistence"])]
            for m, p in sp.groupby("model"):
                ax.plot(p["factor_or_value"], p["energy_mae_wh"], ls=ls, marker=style.MARKERS.get(m, "o"), ms=3, color=self.col(m), label=f"{self.lab(m)} · {'P_idle' if param == 'p_idle' else 'P_busy'}")
        ax.set_xlabel("fator multiplicativo do parâmetro")
        ax.set_ylabel("MAE de energia · 30 min (Wh)")
        ax.set_title(f"(d) P_idle e P_busy (r = {r_ref:g})", loc="left", fontsize=8)
        ax.legend(fontsize=5.5, ncols=2)
        fig.suptitle("Erro de energia de TI modelada (não é erro contra medidor)")
        return fig, pd.concat([e.assign(table="energy_metrics"), s.assign(table="energy_sensitivity")])

    def f32(self):
        o = self.t["energy_origin_errors"]
        et = self.t["energy_timeseries"]
        r = float(self.cfg["energy"]["reference_exponent"])
        fig, axes = style.figure(6.8, 3.4, ncols=2)
        ax = axes[0]
        for m in ("xgboost_v2", self.best_neural):
            p = et[(et["model"] == m) & (et["r"] == r) & (et["step"] == 6)]
            ax.plot(p["cpu_pred_minus_obs_pp"], p["power_projected_minus_reference_w"], ".", color=self.col(m), ms=3, alpha=0.7, label=self.lab(m))
        slope = (self.info["fan"]["p_busy_w"] - self.info["fan"]["p_idle_w"]) / 100
        x = np.linspace(et["cpu_pred_minus_obs_pp"].quantile(0.005), et["cpu_pred_minus_obs_pp"].quantile(0.995), 10)
        ax.plot(x, slope * x, color=MUTED, ls=":", lw=1, label=f"linear ({slope:.2f} W/p.p.)")
        ax.set_xlabel("previsto − observado no passo 6 (p.p.)")
        ax.set_ylabel("diferença de potência (W)")
        ax.set_title(f"(a) Erro de CPU × erro de potência, r = {r:g}", loc="left", fontsize=8)
        ax.legend(fontsize=6)
        ax = axes[1]
        for m in ("xgboost_v2", self.best_neural):
            p = o[(o["model"] == m) & (o["r"] == r) & (o["horizon_steps"] == 6)]
            ax.plot(p["mean_cpu_pred_minus_obs_pp"], p["energy_projected_minus_reference_wh"], ".", color=self.col(m), ms=3, alpha=0.7, label=self.lab(m))
        ax.set_xlabel("previsto − observado, média dos 6 passos (p.p.)", fontsize=7.5)
        ax.set_ylabel("diferença de energia em 30 min (Wh)")
        ax.set_title("(b) Erro médio de CPU × erro de energia", loc="left", fontsize=8)
        ax.legend(fontsize=6)
        return fig, o[(o["r"] == r) & (o["horizon_steps"] == 6)]

    def f33(self):
        j = self.t["jensen_analysis"]
        s = self.t["jensen_summary"]
        fig, axes = style.figure(6.8, 4.8, nrows=2, ncols=2)
        colors = {1.0: "#9ec5f4", 1.4: "#3987e5", 1.8: "#104281"}
        ax = axes[0, 0]
        for r in (1.4, 1.8):
            p = j[j["r"] == r]
            ax.plot(p["time_h"], p["jensen_gap_w_per_server"], color=colors[r], lw=0.7, label=f"r = {r:g}")
        ax.set_ylabel("média P(u_i) − P(ū) (W/servidor)")
        ax.set_xlabel(TIME_LABEL, fontsize=7)
        ax.set_title("(a) Gap de Jensen por servidor substituto", loc="left", fontsize=8)
        ax.legend(fontsize=6.5)
        ax = axes[0, 1]
        for r in (1.4, 1.8):
            p = j[j["r"] == r]
            ax.plot(p["u_variance"], p["jensen_gap_w_per_server"], ".", color=colors[r], ms=2, alpha=0.6, label=f"r = {r:g}")
        ax.set_xlabel("variância da utilização entre máquinas")
        ax.set_ylabel("gap (W/servidor)")
        ax.set_title("(b) Gap × dispersão entre máquinas", loc="left", fontsize=8)
        ax = axes[1, 0]
        for r in (1.4, 1.8):
            p = j[j["r"] == r]
            ax.plot(p["time_h"], p["jensen_gap_fleet_w"] / 1000, color=colors[r], lw=0.7, label=f"r = {r:g}")
        ax.set_ylabel("Σ P(u_i) − n·P(ū) (kW, frota)")
        ax.set_xlabel(TIME_LABEL, fontsize=7)
        ax.set_title("(c) Gap para a frota de servidores substitutos", loc="left", fontsize=8)
        ax = axes[1, 1]
        for r in (1.4, 1.8):
            p = j[j["r"] == r]
            ax.plot(p["jensen_gap_w_per_server"], p["second_order_mean_of_power_w_per_server"] - p["power_of_mean_u_w_per_server"], ".", color=colors[r], ms=2, alpha=0.6, label=f"r = {r:g}")
        lim = [j["jensen_gap_w_per_server"].min(), 0]
        ax.plot(lim, lim, color=MUTED, ls=":", lw=1, label="igualdade")
        ax.set_xlabel("gap exato (W/servidor)")
        ax.set_ylabel("gap de 2ª ordem (W/servidor)")
        ax.set_title("(d) 2ª ordem × cálculo por máquina", loc="left", fontsize=8)
        ax.legend(fontsize=6)
        fig.suptitle("Efeito de agregar a CPU antes da curva de potência (r = 1 dá gap nulo)")
        return fig, s


AXIS_COLOR = style.AXIS


def catalog() -> list[FigureSpec]:
    F = Figures
    return [
        FigureSpec("01_serie_agregada_particoes", "dataset", "Série agregada e partições", "Como a CPU agregada evolui e onde ficam validação e teste?", ("dataset_series",), "Série agregada de CPU, memória e máquinas presentes da população principal do Alibaba Cluster Trace v2018, com os cinco folds de validação externa e o teste final de 24 h.", F.f01),
        FigureSpec("02_distribuicao_dependencia", "dataset", "Distribuição e dependência", "O teste tem o mesmo nível e a mesma dinâmica do desenvolvimento?", ("dataset_series", "dataset_dependence"), "Distribuição da CPU agregada no desenvolvimento e no teste, variações em 5, 15 e 30 min, autocorrelação e correlação defasada entre exógenas e CPU futura.", F.f02),
        FigureSpec("03_frota_por_maquina", "dataset", "CPU por máquina", "Quanto a CPU individual se dispersa em torno da média agregada?", ("fleet_distribution", "machine_coverage"), "Percentis da CPU individual das máquinas presentes em cada intervalo e cobertura temporal por máquina.", F.f03),
        FigureSpec("04_populacoes_m2739", "dataset", "Principal × coorte", "A máquina m_2739 altera a série e as métricas?", ("population_difference", "cohort_comparison"), "Diferença entre a série da população principal e a da coorte homogênea, causada por m_2739, e efeito no MAE de teste da V2.", F.f04),
        FigureSpec("05_ranking_mae_rmse", "comparacao", "Ranking por MAE e RMSE", "Qual a ordem dos modelos no teste em cada horizonte?", ("model_metrics",), "MAE e RMSE no teste final, configuração A, por horizonte; cor por família e linha tracejada na persistência.", F.f05),
        FigureSpec("06_heatmaps_mae_melhoria", "comparacao", "MAE e melhoria", "Quanto cada modelo melhora sobre a persistência?", ("model_metrics",), "MAE de teste e melhoria percentual de MAE contra a persistência, por modelo e horizonte (configuração A).", F.f06),
        FigureSpec("07_validacao_vs_teste", "comparacao", "Validação × teste", "A ordem dos modelos se mantém da validação para o teste?", ("model_metrics",), "MAE na validação externa agregada (círculo vazio) e no teste final (círculo cheio) para modelos da V2 e da V3.", F.f07),
        FigureSpec("08_perfil_horizonte_folds", "comparacao", "Horizonte e folds", "Como o erro cresce com o horizonte e varia entre folds?", ("model_metrics", "model_metrics_by_fold"), "MAE por horizonte na validação e no teste, e MAE por fold em 5 e 30 min.", F.f08),
        FigureSpec("09_distribuicao_erros", "comparacao", "Distribuição dos erros", "Os modelos diferem só na média ou também na cauda do erro?", ("errors_by_origin", "model_metrics"), "Distribuição do erro absoluto no teste por modelo e horizonte, e relação entre MAE e RMSE.", F.f09),
        FigureSpec("10_series_teste", "temporal", "Série de teste", "Como as previsões acompanham a CPU observada?", ("errors_by_origin", "diagnostic_windows"), "CPU observada e prevista no teste final para persistência, XGBoost da V2, Kalman + XGBoost e a rede com menor MAE interno, no instante previsto.", F.f11),
        FigureSpec("11_janelas_objetivas", "temporal", "Janelas objetivas", "Como os modelos reagem a volatilidade, subida, queda e estabilidade?", ("errors_by_origin", "diagnostic_windows"), "Janelas de 3 h escolhidas só pela série observada (maior volatilidade, maior subida, maior queda e maior estabilidade).", F.f12),
        FigureSpec("12_residuos", "temporal", "Resíduos", "Os erros são independentes no tempo?", ("errors_by_origin", "residual_autocorrelation"), "Resíduo ao longo do teste e autocorrelação dos resíduos em 5 e 30 min.", F.f13),
        FigureSpec("13_erro_faixa_cpu", "temporal", "Erro por faixa de CPU", "O erro e o viés dependem do nível de CPU?", ("errors_by_regime",), "MAE e viés por faixa de CPU observada (quintis do desenvolvimento), por modelo e horizonte.", F.f14),
        FigureSpec("14_erro_variacao_regime", "temporal", "Erro e variação", "Como o erro cresce com a variação da CPU e onde a previsão residual ajuda?", ("errors_by_regime",), "MAE em 30 min por variação realizada, por variação recente e por regime, comparando previsão direta e residual.", F.f15),
        FigureSpec("15_picos_eventos", "picos", "Maiores picos", "Os modelos acompanham os maiores picos?", ("errors_by_origin", "peak_events"), "Três maiores picos do teste com previsões em 5 e 30 min e o limiar P90 do desenvolvimento.", F.f16),
        FigureSpec("16_subestimacao_picos", "picos", "Subestimação nos picos", "Quanto e com que frequência os modelos subestimam os picos?", ("peak_analysis", "errors_by_origin"), "Viés, frequência de subestimação e distribuição do erro nos instantes acima do P90 do desenvolvimento.", F.f17),
        FigureSpec("17_alertas_picos", "picos", "Antecipação de picos", "Algum sinal antecipa picos sem excesso de falsos alarmes?", ("peak_alerts",), "Eventos de pico antecipados contra episódios de falso alarme para previsões pontuais e quantis.", F.f18),
        FigureSpec("18_forest_preregistrado", "estatistica", "Forest plot pré-registrado", "Quais diferenças pré-registradas têm intervalo sem zero e são estáveis ao bloco?", ("bootstrap_intervals",), "Diferenças de MAE das comparações pré-registradas com IC 95% por block bootstrap em blocos de 30 min, 1 h e 2 h.", F.f19),
        FigureSpec("19_matriz_pares", "estatistica", "Matriz exploratória", "Como os modelos se comparam dois a dois?", ("statistical_comparisons",), "Matriz exploratória de diferenças de MAE entre todos os pares de modelos (A), com significância antes e depois de Holm.", F.f20),
        FigureSpec("20_curvas_loss", "neural", "Loss das redes", "As redes ajustaram e pararam cedo ou tarde?", ("neural_training_history",), "Loss de treino e de validação interna por época no ponto escolhido da grade, por fold, em 5 e 30 min.", F.f22),
        FigureSpec("21_generalizacao_janela", "neural", "Generalização e janela", "As redes perderam mais que os outros modelos ao passar para o teste?", ("generalization", "neural_window_selection"), "MAE de seleção contra teste, razão de generalização normalizada pela persistência e efeito da janela de entrada.", F.f23),
        FigureSpec("22_custo_pareto", "neural", "Custo × erro", "Mais parâmetros ou mais tempo compraram menos erro?", ("cost_tradeoff",), "Tamanho, tempo de reajuste e memória CUDA contra MAE de teste, com fronteira de Pareto.", F.f24),
        FigureSpec("23_ablacao", "ablacao", "Ablação", "Memória, rede, disco e distribuição da frota reduzem o erro?", ("ablation_grid", "ablation_results"), "Variação percentual de MAE das configurações B, C e D contra A, só nas combinações executadas.", F.f25),
        FigureSpec("24_importancia_shap", "ablacao", "Importância TreeSHAP", "Quais grupos de atributos sustentam as previsões do XGBoost?", ("feature_importance",), "Fração da importância TreeSHAP por grupo de atributos nos reajustes verificados do XGBoost.", F.f26),
        FigureSpec("25_calibracao_quantis", "probabilistico", "Calibração dos quantis", "Os quantis cobrem o que prometem, inclusive nos picos?", ("quantile_calibration",), "Cobertura empírica contra nominal, diferença de cobertura, margem unilateral e cobertura nos picos dos quantis do XGBoost.", F.f27),
        FigureSpec("26_faixas_quantis", "probabilistico", "Faixas quantílicas", "Onde a CPU ultrapassa o teto previsto?", ("quantile_predictions",), "CPU observada com a mediana e o teto Q95 ajustado no teste, com ultrapassagens marcadas.", F.f28),
        FigureSpec("27_curvas_potencia", "energia", "Modelo de potência", "Que potência o modelo de Fan atribui à CPU observada?", ("dataset_series", "energy_timeseries"), "Curvas de potência do servidor substituto para r = 1,0, 1,4 e 1,8, distribuição da CPU de teste e potência de referência.", F.f29),
        FigureSpec("28_potencia_energia_acumulada", "energia", "Potência e energia acumulada", "A energia projetada acompanha a de referência ao longo do teste?", ("energy_timeseries", "energy_cumulative"), "Potência de referência e projetada no passo de 30 min e diferença acumulada de energia com janelas sem sobreposição.", F.f30),
        FigureSpec("29_erro_energia_sensibilidade", "energia", "Erro de energia e sensibilidade", "Quanto erra a energia projetada e quanto isso depende dos parâmetros?", ("energy_metrics", "energy_sensitivity"), "Erro relativo e viés de energia por modelo e horizonte e sensibilidade a r, P_idle e P_busy.", F.f31),
        FigureSpec("30_erro_cpu_energia", "energia", "CPU × potência × energia", "Como o erro de CPU se converte em erro de potência e de energia?", ("energy_timeseries", "energy_origin_errors"), "Relação entre erro de CPU e erro de potência no passo de 30 min e entre erro médio de CPU e erro de energia em 30 min.", F.f32),
        FigureSpec("31_jensen", "energia", "Efeito de Jensen", "Quanto se erra ao aplicar a curva à CPU média em vez de máquina a máquina?", ("jensen_analysis", "jensen_summary"), "Diferença entre a média das potências por máquina e a potência da utilização média, por servidor e para a frota de servidores substitutos, e aproximação de segunda ordem.", F.f33),
    ]


def render_all(tables, info, cfg, out_dir: Path, formats: list[str], dpi: int) -> list[dict]:
    """Gera cada figura nos formatos pedidos e grava o CSV dos dados desenhados."""
    import matplotlib.pyplot as plt

    figs = Figures(tables, info, cfg)
    records = []
    for spec in catalog():
        missing = [s for s in spec.sources if s not in tables]
        if missing:
            records.append({"name": spec.name, "generated": False, "reason": f"tabelas ausentes: {missing}"})
            continue
        fig, data = spec.build(figs)
        style.localize(fig)
        files = {}
        for fmt in formats:
            path = out_dir / fmt / f"{spec.name}.{fmt}"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi)
            files[fmt] = path
        plt.close(fig)
        data_path = out_dir / "data" / f"{spec.name}.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(data_path, index=False, float_format="%.12g", na_rep="")
        records.append({"name": spec.name, "generated": True, "section": spec.section, "title": spec.title, "question": spec.question, "sources": list(spec.sources), "caption": spec.caption, "files": {k: str(v) for k, v in files.items()}, "data": str(data_path)})
    return records
