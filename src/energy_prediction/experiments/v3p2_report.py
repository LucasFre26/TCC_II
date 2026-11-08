"""Relatório da V3 Parte 2, gerado só a partir dos arquivos gravados (sem treino).

Junta as comparações primárias de todas as famílias, aplica Holm sobre o conjunto primário
inteiro, grava figuras (PNG, SVG, PDF e os dados de cada uma), relatório Markdown, HTML (se o
plotly estiver instalado), tabelas LaTeX sugeridas e dicionário de dados. Uso:

    python -m energy_prediction.experiments.v3p2_report --run results/v3-part2/<run_id>
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from energy_prediction.evaluation.day_stats import holm

H = {1: "5 min", 3: "15 min", 6: "30 min"}
HYPOTHESES = {
    "H1": "previsão hierárquica (grupos de máquinas) contra o XGBoost direto",
    "H2": "modelo global por máquina contra o XGBoost direto",
    "H3": "combinação condicionada a regime contra o XGBoost direto",
    "H4": "peso nas subidas: erro nos picos contra o XGBoost direto",
    "H5": "calibração adaptativa (ACI) contra o conformal estático",
    "H6": "energia pela rota por grupo contra a rota da média",
    "H8a": "DLinear com normalização por janela contra a formulação da V3",
    "H8b": "XGBoost com peso de recência contra o XGBoost da V2",
    "H8c": "XGBoost sem calendário contra o XGBoost da V2",
}
FAMILIES = ("robustness", "hierarchical", "global_model", "regimes", "peaks", "calibration", "energy")
THREATS = [
    "Os dias 4 a 6 serviram de validação na V2 (seleção de hiperparâmetros) e na V3; só o dia 3 nunca tinha sido avaliado. Todos os resultados são exploratórios.",
    "Quatro dias de desenvolvimento: a variação entre dias é estimada com n = 4; o bootstrap trata a dependência dentro do dia.",
    "O dia 7 é o teste herdado, na quarta inspeção; ele aparece só como continuidade com a V3.",
    "A covariância da MinT e a escolha do candidato hierárquico usam a mesma validação interna.",
    "O tamanho efetivo do modelo global vem do efeito de desenho e não corrige a autocorrelação temporal.",
    "Coberturas dos limites são empíricas; nenhum método tem garantia teórica sob dependência temporal.",
    "Energia de TI modelada em servidores substitutos do SPECpower_ssj2008: diferença entre duas estimativas, não erro contra medidor; PUE não foi calculado.",
    "Referências metodológicas a conferir e incluir na bibliografia antes da redação: MinT (Wickramasuriya, Athanasopoulos e Hyndman, 2019), ACI (Gibbs e Candès, 2021), SPCI (Xu e Xie, 2023), RevIN (Kim et al., 2022), NLinear (Zeng et al., 2023).",
]


def _read(run: Path, family: str, name: str) -> pd.DataFrame:
    path = run / family / f"{name}.csv"
    if not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def primary_comparisons(run: Path, alpha: float = 0.05) -> pd.DataFrame:
    frames = [_read(run, f, "comparisons") for f in FAMILIES]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    allc = pd.concat(frames, ignore_index=True)
    prim = allc[(allc["scope"] == "primary") & allc["primary_block"].astype(bool)].copy()
    prim = prim.sort_values(["hypothesis", "horizon_steps"]).reset_index(drop=True)
    prim["p_holm_global"] = holm(prim["p_value"].to_numpy()) if len(prim) else []
    prim["holm_global_reject"] = prim["p_holm_global"] < alpha
    prim["practically_relevant"] = prim["relative_diff"].abs() >= 0.05
    prim["computed"] = True
    horizons = sorted(int(h) for h in allc["horizon_steps"].unique())
    missing = [(h, s) for h in HYPOTHESES for s in horizons if not ((prim["hypothesis"] == h) & (prim["horizon_steps"] == s)).any()]
    if missing:
        filler = pd.DataFrame([{"hypothesis": h, "horizon_steps": s, "scope": "primary", "decision": "não calculável", "computed": False} for h, s in missing])
        prim = pd.concat([prim, filler], ignore_index=True).sort_values(["hypothesis", "horizon_steps"]).reset_index(drop=True)
    return prim


def _fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer() and abs(v) < 1e4 and d >= 3:
            return str(int(v))
        return f"{v:.{d}f}".replace(".", ",")
    return str(v)


def md_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    if frame.empty:
        return "_sem linhas_\n"
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = "".join("| " + " | ".join(_fmt(r[c], digits) for c in columns) + " |\n" for _, r in frame.iterrows())
    return head + body


def _save(fig, data: pd.DataFrame, figdir: Path, name: str, records: list, caption: str) -> None:
    from energy_prediction.analysis import style

    style.localize(fig)
    for ext in ("png", "svg", "pdf"):
        (figdir / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(figdir / ext / f"{name}.{ext}", dpi=150, bbox_inches="tight", metadata={"Date": None} if ext == "pdf" else ({} if ext == "png" else {"Date": None}))
    (figdir / "data").mkdir(parents=True, exist_ok=True)
    data.to_csv(figdir / "data" / f"{name}.csv", index=False)
    records.append({"name": name, "caption": caption, "data": f"figures/data/{name}.csv"})
    import matplotlib.pyplot as plt

    plt.close(fig)


def figures(run: Path, out: Path, prim: pd.DataFrame) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    from energy_prediction.analysis import style

    style.apply()
    figdir = out / "figures"
    records: list[dict] = []
    prim = prim[prim["computed"].astype(bool)] if not prim.empty else prim
    if not prim.empty:
        fig, ax = style.figure(7.0, 0.32 * len(prim) + 1.2)
        labels = [f"{r.hypothesis} · {H.get(int(r.horizon_steps), r.horizon_steps)}" for r in prim.itertuples()]
        y = np.arange(len(prim))[::-1]
        rel = prim["relative_diff"].to_numpy() * 100
        lo = (prim["ci_low"] / prim["loss_b"]).to_numpy() * 100
        hi = (prim["ci_high"] / prim["loss_b"]).to_numpy() * 100
        ax.errorbar(rel, y, xerr=[rel - lo, hi - rel], fmt="o", color=style.INK, ecolor=style.MUTED, capsize=2, ms=4)
        ax.axvline(0, color=style.AXIS, lw=1)
        ax.set_yticks(y, labels)
        ax.set_xlabel("diferença relativa da perda (candidato − referência), % da referência")
        _save(fig, prim[["hypothesis", "horizon_steps", "relative_diff", "ci_low", "ci_high", "loss_b", "decision"]], figdir, "01_comparacoes_primarias", records, "Comparações primárias nos dias de desenvolvimento (IC de 95%, bootstrap estratificado por dia).")
    shift = _read(run, "robustness", "shift_bias")
    if not shift.empty:
        part = shift[shift["horizon_steps"] == shift["horizon_steps"].max()]
        fig, ax = style.figure(6.6, 4.0)
        for (model, variant), g in part.groupby(["model", "variant"]):
            ax.plot(g["level_shift_vs_train"], g["bias_obs_minus_pred"], marker="o", lw=0.8, label=f"{model}/{variant}")
        ax.axhline(0, color=style.AXIS, lw=1)
        ax.set_xlabel("nível do bloco − nível do treino (p.p.)")
        ax.set_ylabel("viés no bloco, obs − prev (p.p.)")
        ax.legend(fontsize=7, ncol=2)
        _save(fig, part, figdir, "02_deslocamento_vies", records, "Viés por bloco contra o deslocamento de nível em relação ao treino (30 min).")
    sc = _read(run, "global_model", "scale")
    if not sc.empty:
        fig, ax = style.figure(6.0, 3.6)
        for h, g in sc.groupby("horizon_steps"):
            ax.plot(g["n_machines"], g["mae"], marker=style.HORIZON_MARKERS.get(int(h), "o"), label=H.get(int(h), h))
        ax.set_xscale("log")
        ax.set_xlabel("máquinas amostradas (M)")
        ax.set_ylabel("MAE agregado no bloco (p.p.)")
        ax.legend()
        _save(fig, sc, figdir, "03_escala_global", records, "Diagnóstico de escala do XGBoost global num único bloco (descritivo).")
    cal = _read(run, "calibration", "summary")
    if not cal.empty:
        pooled = cal[cal["block"] == "development_pooled"]
        fig, ax = style.figure(6.6, 3.8)
        for method, g in pooled.groupby("method"):
            ax.plot(g["horizon_steps"], g["coverage_upper"], marker="o", label=method)
        ax.axhline(0.95, color=style.AXIS, ls="--", lw=1)
        ax.set_xticks([1, 3, 6], [H[1], H[3], H[6]])
        ax.set_ylabel("cobertura empírica do limite Q95")
        ax.legend(fontsize=7)
        _save(fig, pooled, figdir, "04_cobertura_calibracao", records, "Cobertura empírica do limite superior por método nos dias de desenvolvimento.")
    en = _read(run, "energy", "summary")
    if not en.empty:
        info = json.loads((run / "energy" / "energy_info.json").read_text(encoding="utf-8"))
        part = en[(en["role"] == "development") & (en["scenario"] == info["primary_substitute"]) & (en["r"] == info["primary_r"]) & (en["horizon_steps"] == en["horizon_steps"].max())]
        fig, ax = style.figure(6.6, 3.8)
        x = np.arange(len(part))
        ax.bar(x - 0.2, part["mean_aggregation_wh"], 0.4, label="aproximação por agregação", color=style.MUTED)
        ax.bar(x + 0.2, part["mean_forecast_wh"], 0.4, label="erro de previsão (média)", color=style.INK)
        ax.set_xticks(x, [f"{m}\n{r}" for m, r in zip(part["model"], part["route"])], fontsize=7)
        ax.set_ylabel("Wh por origem (30 min)")
        ax.legend(fontsize=7)
        _save(fig, part, figdir, "05_decomposicao_energia", records, "Decomposição da diferença de energia de TI modelada (servidor substituto, r = 1,4).")
    tr = _read(run, "peaks", "tradeoff")
    if not tr.empty:
        dev = tr[tr["role"] == "development"].groupby(["horizon_steps", "lambda"], as_index=False)[["mae", "peak_bias_obs_minus_pred"]].mean()
        fig, ax = style.figure(6.0, 3.6)
        for h, g in dev.groupby("horizon_steps"):
            ax.plot(g["mae"], g["peak_bias_obs_minus_pred"], marker=style.HORIZON_MARKERS.get(int(h), "o"), label=H.get(int(h), h))
        ax.set_xlabel("MAE no bloco (p.p.)")
        ax.set_ylabel("viés nos picos, obs − prev (p.p.)")
        ax.legend()
        _save(fig, dev, figdir, "06_compromisso_picos", records, "Compromisso entre erro médio e subestimação nos picos por λ (dias de desenvolvimento).")
    return records


def descriptive_sections(run: Path) -> list[str]:
    """Números descritivos lidos dos arquivos: viés × nível, cobertura, escala, energia e exploratórios com Holm."""
    out = []
    sb = _read(run, "robustness", "shift_bias")
    if not sb.empty:
        h = sb["horizon_steps"].max()
        corr = sb[sb["horizon_steps"] == h].groupby(["model", "variant"])["corr_shift_bias_all_blocks"].first().reset_index()
        out += ["## Viés e deslocamento de nível (30 min, todos os blocos)", "", "Correlação, ao longo dos blocos, entre o nível do bloco menos o do treino e o viés (obs − prev). Valor próximo de 1 indica previsão puxada para o nível do treino.", "", md_table(corr, ["model", "variant", "corr_shift_bias_all_blocks"], 2)]
    cal = _read(run, "calibration", "summary")
    if not cal.empty:
        pooled = cal[cal["block"] == "development_pooled"]
        out += ["## Cobertura dos limites (dias de desenvolvimento)", "", "Coberturas empíricas, sem garantia teórica sob dependência temporal; nominal 0,95 no limite superior e 0,90 no bilateral. `n_infinite` conta origens com limite infinito (nível efetivo do ACI ≤ 0).", "", md_table(pooled, ["method", "horizon_steps", "coverage_upper", "coverage_bilateral", "mean_width", "mean_interval_score", "coverage_upper_peaks", "n_infinite"], 3)]
    agg = _read(run, "energy", "aggregation_summary")
    if not agg.empty and (run / "energy" / "energy_info.json").is_file():
        info = json.loads((run / "energy" / "energy_info.json").read_text(encoding="utf-8"))
        part = agg[(agg["role"] == "development") & (agg["scenario"] == info["primary_substitute"]) & (agg["horizon_steps"] == agg["horizon_steps"].max())]
        out += ["## Aproximação por agregação na energia (30 min, servidor substituto da V3)", "", "Rota observada − referência por máquina, em fração da referência. Estimativa paramétrica, não medição.", "", md_table(part, ["r", "route", "mean_reference_wh", "mean_aggregation_wh", "mean_relative_aggregation"], 4)]
    sc = _read(run, "global_model", "scale")
    ss = _read(run, "global_model", "sample_summary")
    if not sc.empty:
        out += ["## Escala do modelo global (bloco único, descritivo)", "", md_table(sc, ["n_machines", "horizon_steps", "mae", "fit_seconds_total"], 3)]
    if not ss.empty:
        out += ["Tamanho efetivo estimado (efeito de desenho × origens distintas, sem correção da autocorrelação):", "", md_table(ss, ["block", "n_examples_train_h1", "n_distinct_origins_train", "design_mean_correlation", "effective_sample_estimate"], 3)]
    frames = [_read(run, f, "comparisons") for f in FAMILIES]
    frames = [f for f in frames if not f.empty and "holm_reject" in f]
    if frames:
        allc = pd.concat(frames, ignore_index=True)
        exp = allc[(allc["scope"] == "exploratory") & allc["primary_block"].astype(bool) & allc["holm_reject"].astype(bool)]
        out += ["## Exploratórios que sobrevivem a Holm por horizonte", "", "Não substituem os candidatos primários; listados para orientar trabalho futuro.", "", md_table(exp, ["family", "hypothesis", "model_a", "variant_a", "model_b", "variant_b", "horizon_steps", "mean_diff", "decision", "p_holm"], 3) if not exp.empty else "_nenhum_\n"]
    return out


def data_dictionary(run: Path) -> str:
    known = {
        "y_true": "CPU agregada observada no alvo (p.p.)", "y_pred": "CPU prevista (p.p.)", "origin_index": "índice da origem na grade de 300 s",
        "block": "bloco por dia (day_d)", "role": "development (exploratório) ou inherited_test (dia 7, quarta inspeção)", "phase": "inner (validação interna) ou block (bloco avaliado)",
        "mae": "erro absoluto médio (p.p.)", "bias_obs_minus_pred": "viés, observado − previsto (positivo = subestimação)", "mean_diff": "média da perda do candidato menos a da referência",
        "ci_low": "limite inferior do IC de 95% (bootstrap estratificado por dia)", "ci_high": "limite superior do IC de 95%", "decision": "melhor | pior | não distinguível (regra pré-registrada)",
        "reference_wh": "energia de TI modelada pela rota por máquina com CPU observada (Wh)", "total_wh": "rota prevista − referência (Wh)", "aggregation_wh": "rota observada − referência (Wh)",
        "forecast_wh": "rota prevista − rota observada (Wh)", "coverage_upper": "fração de alvos abaixo do limite superior calibrado", "interval_score": "interval score (Winkler) do limite bilateral",
    }
    lines = ["# Dicionário de dados da V3 Parte 2", "", "Unidade de CPU: pontos percentuais (p.p.). Energia: Wh de TI modelada num servidor substituto (estimativa paramétrica).", ""]
    for path in sorted(run.glob("*/*.csv")):
        if path.parent.name == "report":
            continue
        try:
            cols = list(pd.read_csv(path, nrows=0).columns)
        except pd.errors.EmptyDataError:
            cols = []
        lines.append(f"## {path.parent.name}/{path.name}")
        lines += [f"- `{c}`: {known.get(c, 'ver descrição da família')}" for c in cols]
        lines.append("")
    return "\n".join(lines)


def latex_primary(prim: pd.DataFrame) -> str:
    from energy_prediction.analysis.latex import esc, num, table_env

    prim = prim[prim["computed"].astype(bool)]
    rows = [[esc(r.hypothesis), H.get(int(r.horizon_steps), str(r.horizon_steps)), num(r.loss_a, 3), num(r.loss_b, 3), num(r.mean_diff, 3, sign=True), f"[{num(r.ci_low, 3)}; {num(r.ci_high, 3)}]", f"{int(r.days_favorable)}/{int(r.n_days)}", esc(r.decision)] for r in prim.itertuples()]
    return table_env("tab:v3p2-primarias", "Comparações primárias da V3 Parte 2 nos dias de desenvolvimento (exploratórias)", ["Hipótese", "Horizonte", "Candidato", "Referência", "Diferença", "IC 95\\%", "Dias favoráveis", "Decisão"], rows, note="Bootstrap por blocos de 1 h estratificado por dia. Dias 4 a 6 já usados na V2 e na V3.")


def markdown(run: Path, prim: pd.DataFrame, records: list[dict], manifest: dict) -> str:
    parts = ["# V3 Parte 2: relatório da corrida", ""]
    parts.append(f"Corrida `{manifest.get('run_id')}`, modo `{manifest.get('mode')}`, semente {manifest.get('seed')}, dispositivo {manifest.get('device', {}).get('effective')}.")
    if manifest.get("not_result"):
        parts.append(f"\n**{manifest['not_result']}**")
    pre = manifest.get("preregistration") or {}
    parts.append(f"Pré-registro: `{pre.get('path', 'não exigido neste modo')}` (SHA-256 `{pre.get('sha256', '—')}`).")
    parts += ["", "Potência e energia são estimativas paramétricas em servidores substitutos; não há medição elétrica e PUE não foi calculado.", "", "## Hipóteses", ""]
    parts += [f"- **{k}**: {v}." for k, v in HYPOTHESES.items()]
    parts += ["- **H7** (modelo pré-treinado): não executada; o portão de proveniência falhou (corpus com o Alibaba Cluster Trace 2018 ou não público).", ""]
    parts += ["## Comparações primárias (dias de desenvolvimento, exploratórias)", ""]
    if not prim.empty:
        view = prim.assign(horizon=prim["horizon_steps"].map(lambda h: H.get(int(h), h)))
        parts.append(md_table(view, ["hypothesis", "horizon", "loss_a", "loss_b", "mean_diff", "ci_low", "ci_high", "days_favorable", "decision", "p_holm_global"]))
        parts.append("Diferença = candidato − referência; negativo favorece o candidato. A decisão segue a regra pré-registrada (IC e pelo menos 3 de 4 dias); `p_holm_global` corrige sobre o conjunto primário inteiro.")
        parts.append("")
        for hyp, part in prim.groupby("hypothesis"):
            decisions = "; ".join(
                f"{H.get(int(r.horizon_steps), r.horizon_steps)}: {r.decision}"
                + (f" (diferença {_fmt(r.mean_diff)}, IC [{_fmt(r.ci_low)}; {_fmt(r.ci_high)}], {int(r.days_favorable)}/{int(r.n_days)} dias favoráveis, Holm global p = {_fmt(r.p_holm_global)})" if r.computed else "")
                for r in part.itertuples()
            )
            parts.append(f"- **{hyp}** ({HYPOTHESES.get(hyp, '')}): {decisions}.")
        not_computed = prim[~prim["computed"].astype(bool)]
        if not not_computed.empty:
            parts.append("")
            parts.append("Comparações primárias não calculáveis: " + ", ".join(f"{r.hypothesis} em {H.get(int(r.horizon_steps), r.horizon_steps)}" for r in not_computed.itertuples()) + ". O motivo está na seção de portões e itens não executados; Holm global foi aplicado às " + str(int(prim['computed'].sum())) + " comparações calculáveis.")
        parts.append("")
        parts += descriptive_sections(run)
    parts += ["## Teste herdado (dia 7)", "", "O dia 7 foi inspecionado na V2, na V3 e na análise expandida da V3; esta é a quarta inspeção. As diferenças abaixo são só continuidade com a V3, sem inferência e sem valor de confirmação.", ""]
    inh = [_read(run, f, "inherited_test") for f in FAMILIES]
    inh = pd.concat([f for f in inh if not f.empty], ignore_index=True) if any(not f.empty for f in inh) else pd.DataFrame()
    if not inh.empty:
        parts.append(md_table(inh, ["model_a", "variant_a", "model_b", "variant_b", "horizon_steps", "mae_a", "mae_b", "mae_diff"]))
    gates = manifest.get("gates", [])
    excluded = manifest.get("excluded", [])
    parts += ["", "## Portões e itens não executados", ""]
    parts += [f"- portão `{g.get('gate')}` ({g.get('family')}): {'cumprido' if g.get('passed') else 'não cumprido'}." for g in gates]
    parts += [f"- {e.get('family')}: {e.get('item')} — {e.get('reason')}" for e in excluded] or ["- nenhum"]
    parts += ["", "## Ameaças à validade", ""] + [f"- {t}" for t in THREATS]
    parts += ["", "## Figuras", ""] + [f"- `figures/png/{r['name']}.png`: {r['caption']} Dados: `{r['data']}`." for r in records]
    parts += ["", "## Conclusões sustentadas pelos dados", ""]
    if not prim.empty:
        done = prim[prim["computed"].astype(bool)]
        better = done[done["decision"] == "melhor"]
        worse = done[done["decision"] == "pior"]
        parts.append(f"- {len(better)} de {len(done)} comparações primárias calculáveis favoreceram o candidato pela regra pré-registrada ({', '.join(f'{r.hypothesis}/{H.get(int(r.horizon_steps))}' for r in better.itertuples()) or 'nenhuma'}); {len(worse)} favoreceram a referência ({', '.join(f'{r.hypothesis}/{H.get(int(r.horizon_steps))}' for r in worse.itertuples()) or 'nenhuma'}).")
        survivors = done[done["holm_global_reject"].astype(bool)]
        parts.append(f"- Sobreviveram a Holm sobre o conjunto primário: {', '.join(f'{r.hypothesis}/{H.get(int(r.horizon_steps))} ({r.decision})' for r in survivors.itertuples()) or 'nenhuma'}.")
        parts.append("- Todas as conclusões são exploratórias (dias de desenvolvimento já usados nas versões anteriores) e se limitam a este trace de 8 dias.")
    return "\n".join(parts) + "\n"


def html(markdown_text: str, prim: pd.DataFrame, figdir: Path) -> str | None:
    if importlib.util.find_spec("plotly") is None:
        return None
    from energy_prediction.analysis.interactive import CSS, table

    imgs = "".join(f'<figure><img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="max-width:100%"><figcaption>{p.stem}</figcaption></figure>' for p in sorted((figdir / "png").glob("*.png")))
    body = "".join(f"<p>{line}</p>" for line in markdown_text.splitlines() if line and not line.startswith("|"))
    return f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>V3 Parte 2</title><style>{CSS}</style></head><body><h1>V3 Parte 2</h1>{table(prim) if not prim.empty else ''}{body}{imgs}</body></html>"


def generate(run: Path, out: Path | None = None) -> dict:
    run = Path(run)
    out = Path(out) if out is not None else run / "report"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")) if (run / "manifest.json").is_file() else {}
    prim = primary_comparisons(run, alpha=float((manifest.get("config") or {}).get("statistics", {}).get("alpha", 0.05)))
    prim.to_csv(out / "primary_comparisons.csv", index=False)
    records = figures(run, out, prim)
    text = markdown(run, prim, records, manifest)
    (out / "relatorio.md").write_text(text, encoding="utf-8")
    (out / "data_dictionary.md").write_text(data_dictionary(run), encoding="utf-8")
    (out / "figures.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "latex").mkdir(exist_ok=True)
    if not prim.empty:
        (out / "latex" / "tabela_primarias.tex").write_text(latex_primary(prim), encoding="utf-8")
    page = html(text, prim, out / "figures")
    if page is not None:
        (out / "relatorio.html").write_text(page, encoding="utf-8")
    return {"n_primary": int(len(prim)), "html": page is not None, "figures": [r["name"] for r in records]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Relatório da V3 Parte 2 a partir dos arquivos gravados.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(generate(args.run, args.out))


if __name__ == "__main__":
    main()
