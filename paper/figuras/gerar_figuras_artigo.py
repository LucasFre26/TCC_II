"""Figuras do artigo, na largura do texto (cerca de 16 cm).

Só lê os resultados gravados em `Tcc-sourcecode/results/` (blocos diários em `v3-part2/v3p2-full/`
e teste final de 24 h na análise expandida de `v3/`) e grava PDFs nesta pasta.
Uso, a partir de `Tcc-sourcecode/`:

    python paper/figuras/gerar_figuras_artigo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.analysis import style

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RESULTS = PROJECT / "results"
RUN = RESULTS / "v3-part2" / "v3p2-full"
TEST_ANALYSIS = RESULTS / "v3" / "20261104T232620Z-5b659dad" / "expanded-analysis"
CONFIG = PROJECT / "config" / "v3-analysis.yaml"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def save(fig, name: str) -> None:
    style.localize(fig)
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    import matplotlib.pyplot as plt

    plt.close(fig)


def serie(day: int = 6, steps: int = 6) -> None:
    sources = [("baselines", "xgboost_v2", "A", "XGBoost", BLUE),
               ("hierarchical", "hierarchical", "candidate", "hierárquico", ORANGE),
               ("global_model", "global_xgb", "main", "XGBoost global", AQUA)]
    fig, ax = style.figure(6.3, 2.6)
    first = True
    for family, model, variant, label, color in sources:
        p = pd.read_csv(RUN / family / "predictions.csv", low_memory=False)
        p = p[(p["model"] == model) & (p["variant"] == variant) & (p["day"] == day) & (p["horizon_steps"] == steps) & (p["phase"] != "inner")]
        hour = (p["origin_index"] + steps) * 300 / 3600
        if first:
            ax.plot(hour, p["y_true"], color=style.OBSERVED, lw=1.6, label="CPU observada")
            first = False
        ax.plot(hour, p["y_pred"], color=color, lw=0.9, label=label)
    ax.set_xlabel("horas desde o início do trace (instante previsto)")
    ax.set_ylabel("CPU agregada (%)")
    for line in ax.legend(ncol=2, loc="upper right").get_lines():
        line.set_linewidth(1.8)
    save(fig, "serie_dia6_30min")


def vies_nivel(steps: int = 6) -> None:
    s = pd.read_csv(RUN / "robustness" / "shift_bias.csv")
    s = s[s["horizon_steps"] == steps]
    series = [("xgboost_v2", "A", "XGBoost", style.INK, "-", "D"),
              ("dlinear", "v3", "DLinear sem normalização", AQUA, "-", "o"),
              ("dlinear", "window", "DLinear normalizado", AQUA, "--", "o"),
              ("tcn", "v3", "TCN sem normalização", PURPLE, "-", "s"),
              ("tcn", "window", "TCN normalizada", PURPLE, "--", "s")]
    fig, ax = style.figure(6.3, 2.7)
    ax.axhline(0, color=style.AXIS, lw=1)
    for model, variant, label, color, ls, mk in series:
        g = s[(s["model"] == model) & (s["variant"] == variant)].sort_values("level_shift_vs_train")
        r = g["corr_shift_bias_all_blocks"].iloc[0]
        ax.plot(g["level_shift_vs_train"], g["bias_obs_minus_pred"], color=color, ls=ls, marker=mk, ms=4, lw=1.1,
                mfc=color if ls == "-" else "white", label=f"{label} (r = {r:.2f})")
    ax.set_xlabel("nível do bloco − nível do treino (p.p.)")
    ax.set_ylabel("viés em 30 min, obs − prev (p.p.)")
    ax.legend(loc="upper left", fontsize=6.5)
    save(fig, "vies_deslocamento")


def energia() -> None:
    info = json.loads((RUN / "energy" / "energy_info.json").read_text(encoding="utf-8"))
    agg = pd.read_csv(RUN / "energy" / "aggregation_summary.csv")
    summ = pd.read_csv(RUN / "energy" / "summary.csv")
    primary = info["primary_substitute"]
    fig, (a, b) = style.figure(6.5, 2.7, ncols=2)
    part = agg[(agg["role"] == "development") & (agg["scenario"] == primary) & (agg["horizon_steps"] == 6)]
    routes = [("mean", "rota da média", style.INK, "o", "-"), ("group_mean_k4", "grupos, K = 4", ORANGE, "v", ":"),
              ("group_mean_k16", "grupos, K = 16", ORANGE, "^", "-"), ("group_candidate", "grupos do candidato", ORANGE, "D", "-.")]
    for route, label, color, mk, ls in routes:
        g = part[part["route"] == route].sort_values("r")
        a.plot(g["r"], 100 * g["mean_relative_aggregation"], color=color, marker=mk, ls=ls, lw=1.0, ms=4, label=label)
    a.set_xticks([1.0, 1.4, 1.8])
    a.set_xlabel("expoente r")
    a.set_ylabel("agregação (% da referência)")
    a.set_title("(a) termo de agregação, 30 min", loc="left")
    a.legend(loc="upper left", fontsize=6.3)
    scen = {primary: "2288H V5", "power_ssj2008-20170807-00775": "NF5280 M4",
            "power_ssj2008-20190730-00995": "TX2550 M5", "heterogeneo_sintetico": "sintético"}
    keys = [("persistence", "mean", "persistência, média", style.MUTED, "o"), ("xgboost_v2", "mean", "XGBoost, média", style.INK, "D"),
            ("hierarchical", "group_candidate", "hierárquico, grupos", ORANGE, "D"), ("global", "machine", "global, máquina", AQUA, "s")]
    pb = summ[(summ["role"] == "development") & (summ["r"] == info["primary_r"]) & (summ["horizon_steps"] == 6)]
    for k, (model, route, label, color, mk) in enumerate(keys):
        g = pb[(pb["model"] == model) & (pb["route"] == route)].set_index("scenario").reindex(list(scen))
        b.plot(np.arange(len(scen)) + (k - 1.5) * 0.15, 100 * g["mean_relative_abs_total"], color=color, marker=mk, ms=4.5, ls="none", label=label)
    b.set_xticks(range(len(scen)), list(scen.values()), fontsize=6.5)
    b.set_ylabel("|projetada − referência| (%)")
    b.set_title("(b) erro total por janela, 30 min, r = 1,4", loc="left")
    b.set_ylim(top=b.get_ylim()[1] + 0.9)
    b.legend(loc="upper right", ncol=2, fontsize=6.3)
    save(fig, "energia_rotas")


def picos() -> None:
    """Refaz a figura de subestimação nos picos do teste final com a rotina da análise expandida."""
    import yaml

    from energy_prediction.analysis.figures import Figures

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["models"]["labels"] = {m: label.replace(" (V2)", "") for m, label in cfg["models"]["labels"].items()}
    csv = TEST_ANALYSIS / "csv"
    peaks = pd.read_csv(csv / "peak_analysis.csv")
    errors = pd.read_csv(csv / "errors_by_origin.csv", low_memory=False)
    p90 = peaks.loc[peaks["threshold_kind"] == "desenvolvimento_p90", "threshold_cpu_pct"].iloc[0]
    fig, _ = Figures({"peak_analysis": peaks, "errors_by_origin": errors}, {"dev_p90": float(p90)}, cfg).f17()
    save(fig, "subestimacao_picos")


if __name__ == "__main__":
    style.apply()
    serie()
    vies_nivel()
    energia()
    picos()
    print("figuras gravadas em", HERE)
