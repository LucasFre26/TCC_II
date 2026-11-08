"""Tabelas `.tex`, ambientes de figura com legendas sugeridas e proposta da seção de resultados.

Segue o padrão do artigo: legenda acima, `\\label`, `\\centering`, `\\footnotesize`, linhas `\\hline`,
vírgula decimal como `{,}` e "Fonte: Autoria própria." abaixo. Usa só `tabular` (nenhum pacote novo).
Não edita principal.tex.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from energy_prediction.analysis.facts import prereg, thousands

H = {1: "5 min", 3: "15 min", 6: "30 min"}


def esc(text: str) -> str:
    return re.sub(r"([%&_#$])", r"\\\1", str(text))


def num(value, digits: int = 2, sign: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace("-", "$-$").replace("+", "$+$").replace(".", "{,}")


def table_env(label: str, caption: str, header: list[str], rows: list[list[str]], note: str = "", wide: bool = False) -> str:
    spec = "|l|" + "c|" * (len(header) - 1)
    lines = [
        "\\begin{table}[htb]", f"\\caption{{{caption}}}", f"\\label{{tab:{label}}}", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}", "\\renewcommand{\\arraystretch}{1.15}", *(["\\resizebox{\\textwidth}{!}{%"] if wide else []), f"\\begin{{tabular}}{{{spec}}}", "\\hline",
        " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\hline",
    ]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\hline", "\\end{tabular}}\\\\" if wide else "\\end{tabular}\\\\"]
    if note:
        lines.append(f"\\footnotesize {note}\\\\")
    lines += ["\\textbf{\\footnotesize Fonte: Autoria própria.}", "\\end{table}", ""]
    return "\n".join(lines)


def tables(t: dict, f: dict, cfg: dict) -> dict[str, str]:
    L = {k: esc(v) for k, v in cfg["models"]["labels"].items()}
    order = [m for m in cfg["models"]["order"] if m in f["test_mae"].index]
    out = {}
    m = t["model_metrics"]
    test = m[(m["split"] == "test_final") & (m["population"] == "main") & (m["analysis_set"] == "main") & (m["mode"] == "direct") & m["config"].isin(["A", "-"])].set_index(["model", "horizon_steps"])
    rows = [[L[x]] + [num(test.loc[(x, h), "mae_pp"]) for h in (1, 3, 6)] + [num(test.loc[(x, h), "rmse_pp"]) for h in (1, 3, 6)] for x in order]
    out["v3_metricas_teste"] = table_env("v3-metricas-teste", "MAE e RMSE no teste final (configuração A, 288 origens), em pontos percentuais de CPU", ["Modelo", "MAE 5 min", "MAE 15 min", "MAE 30 min", "RMSE 5 min", "RMSE 15 min", "RMSE 30 min"], rows)
    rows = [[L[x]] + [num(f["val_mae"].loc[x, h]) for h in (1, 3, 6)] + [num(f["test_mae"].loc[x, h]) for h in (1, 3, 6)] for x in order]
    out["v3_validacao_teste"] = table_env("v3-validacao-teste", "MAE na validação externa agregada (720 origens) e no teste final (288 origens), configuração A", ["Modelo", "Val. 5 min", "Val. 15 min", "Val. 30 min", "Teste 5 min", "Teste 15 min", "Teste 30 min"], rows)
    rows = []
    for comp, ref in (("new_vs_v2_xgboost_A", "XGBoost (V2)"), ("new_vs_persistence_A", "Persistência")):
        for x in ("xgboost_delta", "statespace", "hybrid", "dlinear", "tcn", "nbeatsx"):
            cells = []
            for h in (1, 3, 6):
                r = prereg(t, comp, x, h)
                cells.append("--" if r is None else f"{num(r['mean_diff'], 2, True)} [{num(r['ci_low'])}; {num(r['ci_high'])}]{'*' if r['significant'] else ''}")
            rows.append([f"{L[x]} $\\times$ {ref}"] + cells)
    out["v3_comparacoes"] = table_env("v3-comparacoes", "Diferença de MAE no teste (modelo $-$ referência) e IC 95\\% por \\textit{block bootstrap} (blocos de 1 h, 5.000 reamostragens)", ["Comparação", "5 min", "15 min", "30 min"], rows, "* IC 95\\% sem zero. Valores negativos favorecem o primeiro modelo.", wide=True)
    pk = f["peak"]
    al = f["alerts"]
    rows = []
    for x in ("persistence", "xgboost_v2", "hybrid", f["best_neural"]):
        rows.append([L[x]] + [num(pk.loc[(x, h), "peak_bias_obs_minus_pred_pp"], 2, True) for h in (1, 6)] + [num(pk.loc[(x, h), "peak_underestimation_frequency"]) for h in (1, 6)] + [f"{int(al.loc[(x, h), 'n_anticipated'])}/{int(al.loc[(x, h), 'n_false_alarm_episodes'])}" for h in (1, 6)])
    for label in ("Q95 bruto", "Q95 ajustado"):
        key = f"xgboost_quantile {label}"
        rows.append([f"XGBoost quantílico {label}", "--", "--", "--", "--"] + [f"{int(al.loc[(key, h), 'n_anticipated'])}/{int(al.loc[(key, h), 'n_false_alarm_episodes'])}" for h in (1, 6)])
    out["v3_picos"] = table_env("v3-picos", f"Erro nos picos (CPU $\\geq$ P90 do desenvolvimento, {num(f['dev_p90'], 1)}\\%) e antecipação dos {f['n_events']} eventos do teste", ["Sinal", "Viés 5 min", "Viés 30 min", "Subest. 5 min", "Subest. 30 min", "Antec./falsos 5 min", "Antec./falsos 30 min"], rows, "Viés = observado $-$ previsto (p.p.); Subest. = fração dos picos subestimada; Antec./falsos = eventos antecipados/episódios de falso alarme.", wide=True)
    q = f["quant"]
    rows = []
    for name, variant in (("Q50", "bruto"), ("Q90", "bruto"), ("Q95", "bruto"), ("Q90", "ajustado"), ("Q95", "ajustado")):
        rows.append([f"{name} {variant}"] + [num(q.loc[("test_final", h, name, variant), "empirical_coverage"]) for h in (1, 3, 6)] + [num(q.loc[("test_final", h, name, variant), "peak_coverage"]) for h in (1, 6)])
    out["v3_quantis"] = table_env("v3-quantis", "Cobertura empírica $P(y \\leq q)$ dos quantis do XGBoost no teste (configuração A)", ["Quantil", "5 min", "15 min", "30 min", "Picos 5 min", "Picos 30 min"], rows, "Cobertura empírica, sem garantia teórica sob dependência temporal; o ajuste conformal foi calibrado na validação externa.")
    e = f["energy"]
    r = float(cfg["energy"]["reference_exponent"])
    rows = []
    for x in [m for m in cfg["models"]["order"] if (m, r, 6) in e.index]:
        rows.append([L[x]] + [num(100 * e.loc[(x, r, h), "energy_mean_abs_relative_error"]) for h in (1, 6)] + [num(e.loc[(x, r, h), "energy_bias_projected_minus_reference_wh"], 2, True) for h in (1, 6)] + [num(f["cum30"].loc[x, "cumulative_projected_minus_reference_wh"], 1, True)])
    out["v3_energia"] = table_env("v3-energia", f"Erro da energia de TI modelada (servidor substituto, $r={num(r, 1)}$) em relação à energia de referência", ["Modelo", "Erro rel. 5 min (\\%)", "Erro rel. 30 min (\\%)", "Viés 5 min (Wh)", "Viés 30 min (Wh)", "Acum. 24 h (Wh)"], rows, f"Estimativa paramétrica, não medição. Referência em 24 h: {thousands(f['energy_ref_24h'])} Wh. Acum.: 48 janelas de 30 min sem sobreposição.", wide=True)
    jz = f["jensen"]
    rows = [[num(rr, 1), num(jz.loc[rr, "mean_gap_w_per_server"]), num(jz.loc[rr, "min_gap_w_per_server"]), num(100 * jz.loc[rr, "mean_relative_gap"]), num(jz.loc[rr, "mean_gap_fleet_w"] / 1000, 1), num(jz.loc[rr, "mean_abs_second_order_error_w"])] for rr in jz.index]
    out["v3_jensen"] = table_env("v3-jensen", "Efeito de agregar a CPU antes da curva de potência: média de $P(u_i)$ $-$ $P(\\bar{u})$", ["$r$", "Gap médio (W/serv.)", "Gap mínimo (W/serv.)", "Gap relativo (\\%)", "Gap da frota (kW)", "Erro 2ª ordem (W)"], rows, "Frota: soma sobre as máquinas presentes, cada uma tratada como um servidor substituto (grandeza hipotética).")
    g = f["gen"]
    comp = t["model_complexity"]
    perf = t["computational_performance"]
    rows = []
    for x in ("dlinear", "tcn", "nbeatsx", "nhits", "patchtst", "xgboost_v2", "hybrid"):
        c = comp[(comp["model"] == x) & (comp["config"] == "A") & (comp["horizon"] == "6")]
        p = perf[(perf["model"] == x) & (perf["config"] == "A") & (perf["horizon"] == "6")]
        size = "--" if c.empty else str(int(c["n_trainable_parameters"].iloc[0])) if np.isfinite(c["n_trainable_parameters"].iloc[0]) else f"{int(c['n_leaves'].iloc[0])} folhas"
        rows.append([L[x], size, num(g.loc[(x, 6), "selection_mae_pp"]), num(g.loc[(x, 6), "test_mae_pp"]), num(g.loc[(x, 6), "normalized_generalization_ratio"]), "--" if p.empty else num(p["train_seconds_test_refit"].iloc[0], 3), "--" if p.empty or p["device"].iloc[0] != "cuda" else num(p["peak_cuda_memory_mib"].iloc[0], 0)])
    out["v3_redes"] = table_env("v3-redes", "Redes neurais e referências em 30 min: tamanho, MAE de seleção e de teste, razão de generalização e custo", ["Modelo", "Tamanho", "MAE seleção", "MAE teste", "Razão norm.", "Reajuste (s)", "CUDA (MiB)"], rows, "Razão norm. = (teste/seleção) dividido pela mesma razão da persistência. Redes: CUDA, corrida original; XGBoost e híbrido: CPU, medido na análise.")
    ab = f["ablation"]
    rows = []
    for x in ("xgboost_v2", "xgboost_delta", "hybrid", "dlinear", "tcn", "nbeatsx"):
        cells = []
        for c in ("B", "C", "D"):
            for h in (1, 6):
                cells.append(num(ab.loc[(x, c, h), "delta_mae_pct"], 1, True) if (x, c, h) in ab.index else "--")
        rows.append([L[x]] + cells)
    out["v3_ablacao"] = table_env("v3-ablacao", "Variação percentual do MAE de teste contra a configuração A (positivo = piora)", ["Modelo", "B 5 min", "B 30 min", "C 5 min", "C 30 min", "D 5 min", "D 30 min"], rows, "B: + memória; C: + rede e disco; D: + distribuição da CPU entre máquinas, contra A nas mesmas origens. -- = não executado.")
    return out


SELECTED_FIGURES = [
    "02_distribuicao_dependencia", "05_ranking_mae_rmse", "07_validacao_vs_teste", "10_series_teste", "14_erro_variacao_regime",
    "16_subestimacao_picos", "17_alertas_picos", "18_forest_preregistrado", "21_generalizacao_janela", "23_ablacao",
    "25_calibracao_quantis", "28_potencia_energia_acumulada", "29_erro_energia_sensibilidade", "31_jensen",
]


def figure_env(record: dict) -> str:
    name = record["name"]
    label = "fig:v3-" + name.split("_", 1)[1].replace("_", "-")
    return "\n".join(
        [
            "\\begin{figure}[htb]", "\\centering", f"\\caption[{esc(record['title'])}]{{{esc(record['caption'])}}}", f"\\label{{{label}}}",
            f"\\includegraphics[width=\\textwidth]{{figuras/v3/{name}.pdf}}\\\\", "\\textbf{\\footnotesize Fonte: Autoria própria.}", "\\end{figure}", "",
        ]
    )


PROPOSAL = """# Proposta de reorganização da seção de resultados (não aplicada a principal.tex)

Esta proposta organiza os resultados finais a partir da análise expandida. O artigo só deve ser alterado depois da revisão científica destes números. Tempo verbal: passado para o que foi executado (V1, V2, V3 e análise expandida); nenhuma etapa pendente é descrita como feita.

## Estrutura sugerida

1. **Dados e protocolo (curta).** Série agregada, partições e mudança de nível do teste (Figura `02_distribuicao_dependencia`). Declarar que o teste de 24 h foi reutilizado da V2 e não é independente do desenvolvimento.
2. **Desempenho pontual.** Tabela `tab:v3-metricas-teste`, Figura `05_ranking_mae_rmse` e Tabela `tab:v3-validacao-teste` (validação × teste, Figura `07_validacao_vs_teste`).
3. **Inferência.** Tabela `tab:v3-comparacoes` e Figura `18_forest_preregistrado`. Texto: o híbrido não se distinguiu do XGBoost da V2; as redes DLinear, TCN e N-BEATSx foram piores com IC sem zero. Mencionar Holm como sensibilidade.
4. **Por que as redes não superaram o XGBoost.** Figura `21_generalizacao_janela`, Figura `14_erro_variacao_regime` e Tabela `tab:v3-redes`. Separar o observado (viés positivo, razão de generalização maior que 1) das hipóteses (mudança de nível, poucas origens, ancoragem em cpu(t)).
5. **Picos.** Figura `16_subestimacao_picos`, Figura `17_alertas_picos` e Tabela `tab:v3-picos`. Não chamar o teto quantílico de detector de picos.
6. **Previsão probabilística.** Tabela `tab:v3-quantis` e Figura `25_calibracao_quantis`; cobertura empírica sem garantia teórica.
7. **Energia de TI modelada.** Tabela `tab:v3-energia`, Figura `28_potencia_energia_acumulada`, Figura `29_erro_energia_sensibilidade`, Tabela `tab:v3-jensen` e Figura `31_jensen`. Sempre "energia de TI modelada num servidor substituto"; PUE fora.
8. **Ablação e importância.** Tabela `tab:v3-ablacao` e Figura `23_ablacao`; importância preditiva não é causalidade.
9. **Limitações.** Itens da Seção 11 de `scientific-analysis.md`.

## Como incluir

- Copiar `latex/figuras/*.pdf` para `Tcc LaTeX/figuras/v3/`.
- Incluir as tabelas com `\\input{tabelas/v3_metricas_teste.tex}` (ou colar o conteúdo) e as figuras a partir de `latex/figuras.tex`.
- As tabelas usam só `tabular` e `\\hline`; nenhum pacote novo é necessário.
"""


def write(t: dict, f: dict, cfg: dict, figure_records: list[dict], out_dir: Path) -> list[Path]:
    written = []
    (out_dir / "tabelas").mkdir(parents=True, exist_ok=True)
    for name, content in tables(t, f, cfg).items():
        path = out_dir / "tabelas" / f"{name}.tex"
        path.write_text(content, encoding="utf-8")
        written.append(path)
    (out_dir / "figuras").mkdir(parents=True, exist_ok=True)
    envs, captions = [], ["# Legendas sugeridas", "", "| Figura | Rótulo | Legenda curta | Legenda | Pergunta |", "|---|---|---|---|---|"]
    for record in figure_records:
        if not record.get("generated"):
            continue
        label = "fig:v3-" + record["name"].split("_", 1)[1].replace("_", "-")
        captions.append(f"| `{record['name']}` | `{label}` | {record['title']} | {record['caption']} | {record['question']} |")
        if record["name"] in SELECTED_FIGURES:
            source = Path(record["files"]["pdf"])
            target = out_dir / "figuras" / source.name
            shutil.copyfile(source, target)
            written.append(target)
            envs.append(figure_env(record))
    (out_dir / "figuras.tex").write_text("% Ambientes de figura sugeridos (figuras em figuras/v3/)\n\n" + "\n".join(envs), encoding="utf-8")
    (out_dir / "legendas-sugeridas.md").write_text("\n".join(captions) + "\n", encoding="utf-8")
    (out_dir / "proposta-secao-resultados.md").write_text(PROPOSAL, encoding="utf-8")
    written += [out_dir / "figuras.tex", out_dir / "legendas-sugeridas.md", out_dir / "proposta-secao-resultados.md"]
    return written
