"""Relatório HTML navegável e autocontido (Plotly embutido uma vez; sem recurso externo).

Links relativos apontam para ../csv/ e ../figures/ dentro de expanded-analysis/.
"""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from energy_prediction.analysis import style
from energy_prediction.analysis.facts import br, thousands

H = {1: "5 min", 3: "15 min", 6: "30 min"}
SECTIONS = [
    ("visao-geral", "1. Visão geral e principais resultados"),
    ("dataset", "2. Caracterização do dataset"),
    ("comparacao", "3. Comparação de modelos"),
    ("temporal", "4. Diagnóstico temporal"),
    ("picos", "5. Picos de utilização"),
    ("estatistica", "6. Validação estatística"),
    ("neural", "7. Redes neurais"),
    ("ablacao", "8. Ablação"),
    ("probabilistico", "9. Previsão probabilística"),
    ("energia", "10. Estimativa energética"),
    ("limitacoes", "11. Limitações científicas"),
    ("conclusoes", "12. Conclusões"),
]
FIG_SECTION = {"dataset": "dataset", "comparacao": "comparacao", "temporal": "temporal", "picos": "picos", "estatistica": "estatistica", "neural": "neural", "ablacao": "ablacao", "probabilistico": "probabilistico", "energia": "energia"}

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--surface:#fcfcfb;--page:#f9f9f7;--accent:#2a78d6}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--page);line-height:1.5}
header{padding:24px 16px 8px;max-width:1180px;margin:0 auto}h1{font-size:1.5rem;margin:0 0 4px}header p{color:var(--ink2);margin:4px 0}
nav{position:sticky;top:0;background:var(--page);border-bottom:1px solid var(--grid);z-index:5}nav div{max-width:1180px;margin:0 auto;padding:8px 16px;display:flex;flex-wrap:wrap;gap:6px 14px}
nav a{color:var(--ink2);text-decoration:none;font-size:.85rem}nav a:hover{color:var(--accent)}
main{max-width:1180px;margin:0 auto;padding:0 16px 48px}section{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:16px 20px;margin:20px 0}
h2{font-size:1.2rem;margin:0 0 8px}h3{font-size:1rem;margin:18px 0 6px}p,li{max-width:80ch}.note{color:var(--ink2);font-size:.9rem}
table{border-collapse:collapse;font-size:.82rem;margin:8px 0;font-variant-numeric:tabular-nums;display:block;overflow-x:auto;max-width:100%}
th,td{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:right;white-space:nowrap}th{cursor:pointer;color:var(--ink2);background:#f0efec;position:sticky;top:0}td:first-child,th:first-child{text-align:left}
.figs{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}.figs figure{margin:0;border:1px solid var(--grid);border-radius:6px;padding:8px;background:white}
.figs img{width:100%;height:auto}.figs figcaption{font-size:.8rem;color:var(--ink2)}.files a{margin-right:10px;font-size:.85rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}.kpi{border:1px solid var(--grid);border-radius:6px;padding:10px;background:white}.kpi b{display:block;font-size:1.25rem}.kpi span{color:var(--ink2);font-size:.8rem}
.plot{width:100%;min-height:360px}
"""

SORT_JS = """
document.querySelectorAll('table.sortable th').forEach(function(th){th.addEventListener('click',function(){
var table=th.closest('table');var idx=Array.prototype.indexOf.call(th.parentNode.children,th);var body=table.tBodies[0];
var rows=Array.prototype.slice.call(body.rows);var asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
function val(r){var t=r.cells[idx].textContent.replace(/\\./g,'').replace(',','.').replace('%','').replace('\\u2212','-');var n=parseFloat(t);return isNaN(n)?r.cells[idx].textContent:n;}
rows.sort(function(a,b){var x=val(a),y=val(b);if(typeof x==='number'&&typeof y==='number'){return asc?x-y:y-x;}return asc?String(x).localeCompare(String(y)):String(y).localeCompare(String(x));});
rows.forEach(function(r){body.appendChild(r);});});});
"""


def _num(v, digits=3):
    if isinstance(v, (bool, np.bool_)):
        return "sim" if v else "não"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else br(float(v), digits)
    return html.escape(str(v))


def table(frame: pd.DataFrame, digits: int = 3, max_rows: int = 400) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = "".join("<tr>" + "".join(f"<td>{_num(v, digits)}</td>" for v in r) + "</tr>" for r in frame.head(max_rows).itertuples(index=False))
    more = f'<p class="note">Mostrando {max_rows} de {len(frame)} linhas; ver o CSV completo.</p>' if len(frame) > max_rows else ""
    return f'<table class="sortable"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>{more}'


def md_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def md_section(markdown: str, heading: str) -> str:
    """Converte uma seção '## heading' do relatório Markdown em HTML simples (parágrafos e listas)."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", markdown, flags=re.S | re.M)
    if not match:
        return ""
    out, items = [], []
    for block in match.group(1).strip().split("\n"):
        line = block.strip()
        if re.match(r"^(- |\d+\. )", line):
            item = md_inline(re.sub(r"^(- |\d+\. )", "", line))
            items.append(f"<li>{item}</li>")
            continue
        if items:
            out.append("<ul>" + "".join(items) + "</ul>")
            items = []
        if line.startswith("### "):
            out.append(f"<h3>{md_inline(line[4:])}</h3>")
        elif line:
            out.append(f"<p>{md_inline(line)}</p>")
    if items:
        out.append("<ul>" + "".join(items) + "</ul>")
    return "\n".join(out)


class PlotlyBlocks:
    def __init__(self):
        self.first = True

    def __call__(self, fig) -> str:
        fig.update_layout(template="plotly_white", font={"family": "system-ui, sans-serif", "size": 12}, margin={"l": 60, "r": 20, "t": 50, "b": 50}, separators=",.")
        out = fig.to_html(full_html=False, include_plotlyjs="inline" if self.first else False, config={"displaylogo": False, "responsive": True})
        self.first = False
        return f'<div class="plot">{out}</div>'


def build(t: dict, f: dict, info: dict, cfg: dict, figure_records: list[dict], markdown: str, csv_names: list[str]) -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        go = None
    L = cfg["models"]["labels"]
    fam = cfg["models"]["families"]
    order = cfg["models"]["order"]
    plot = PlotlyBlocks()
    col = lambda m: style.color(m, fam)  # noqa: E731
    blocks = {key: [] for key, _ in SECTIONS}

    def figs_for(section):
        cards = []
        for r in figure_records:
            if r.get("generated") and r["section"] == section:
                cards.append(f'<figure><a href="../figures/png/{r["name"]}.png"><img loading="lazy" src="../figures/png/{r["name"]}.png" alt="{html.escape(r["title"])}"></a><figcaption><b>{html.escape(r["title"])}.</b> {html.escape(r["question"])} <a href="../figures/svg/{r["name"]}.svg">SVG</a> · <a href="../figures/pdf/{r["name"]}.pdf">PDF</a> · <a href="../figures/data/{r["name"]}.csv">dados</a></figcaption></figure>')
        return '<h3>Figuras estáticas</h3><div class="figs">' + "".join(cards) + "</div>" if cards else ""

    def csv_links(names):
        return '<p class="files">CSV: ' + "".join(f'<a href="../csv/{n}.csv">{n}.csv</a>' for n in names if n in csv_names) + "</p>"

    test = f["test_mae"]
    # 1. visão geral
    kpis = [
        (br(test.loc[f["best_test"][1], 1]) + " p.p.", f"menor MAE em 5 min ({L[f['best_test'][1]]})"),
        (br(test.loc[f["best_test"][6], 6]) + " p.p.", f"menor MAE em 30 min ({L[f['best_test'][6]]})"),
        (f"{f['neural_vs_xgb_sig_count']}/9", "comparações redes × XGBoost V2 com IC sem zero (redes piores)"),
        (f"{f['residual_vs_xgb_sig_count']}/9", "comparações residual/híbrido × XGBoost V2 com IC sem zero"),
        (br(f.get("cpu_mean_test", np.nan), 1) + "% × " + br(f.get("cpu_mean_dev", np.nan), 1) + "%", "CPU média no teste × desenvolvimento"),
        (f"{f['n_events']} eventos", f"picos ≥ P90 do desenvolvimento ({f['n_single_step_events']} de 1 passo)"),
        (thousands(f["energy_ref_24h"]) + " Wh", "energia de referência modelada em 24 h (r = 1,4, servidor substituto)"),
        (br(f["jensen"].loc[1.4, "mean_gap_w_per_server"]) + " W", "gap de Jensen médio por servidor (r = 1,4)"),
    ]
    blocks["visao-geral"].append('<div class="kpis">' + "".join(f'<div class="kpi"><b>{v}</b><span>{html.escape(k)}</span></div>' for v, k in kpis) + "</div>")
    if go:
        fig = go.Figure()
        for m in [x for x in order if x in test.index]:
            fig.add_bar(name=L[m], x=[H[h] for h in (1, 3, 6)], y=[test.loc[m, h] for h in (1, 3, 6)], marker_color=col(m))
        fig.update_layout(title="MAE de teste por horizonte (A); clique na legenda para ocultar modelos", yaxis_title="MAE (p.p.)", barmode="group", height=420)
        blocks["visao-geral"].append(plot(fig))
    blocks["visao-geral"].append(md_section(markdown, "12. Conclusões sustentadas pelos dados"))

    # 2. dataset
    if "dataset_series" in t and go:
        s = t["dataset_series"]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
        fig.add_scatter(x=s["time_h"], y=s["aggregated_cpu_weighted_pct"], name="CPU agregada", line={"color": "#0b0b0b", "width": 1}, row=1, col=1)
        fig.add_scatter(x=s["time_h"], y=s["aggregated_mem_pct"], name="memória", line={"color": "#52514e", "width": 1}, row=2, col=1)
        test_start = info["test_range"][0] / 12
        fig.add_vrect(x0=test_start, x1=s["time_h"].max(), fillcolor="#fbd9ca", opacity=0.5, line_width=0, annotation_text="teste", row="all", col=1)
        fig.update_layout(title="Série agregada (tempo desde o início do trace, h; não é horário civil)", height=460)
        fig.update_yaxes(title_text="CPU (%)", row=1, col=1)
        fig.update_yaxes(title_text="memória (%)", row=2, col=1)
        blocks["dataset"].append(plot(fig))
    if "dataset_profile" in t:
        p = t["dataset_profile"]
        blocks["dataset"].append(table(p[p["variable"].isin(["cpu", "mem"])][["period", "variable", "mean", "std", "p05", "p50", "p95", "max", "n_missing"]], 2))
    blocks["dataset"].append(md_section(markdown, "9. Dataset e populações"))
    blocks["dataset"].append(csv_links(["dataset_series", "dataset_profile", "dataset_changes", "dataset_dependence", "fleet_distribution", "machine_coverage", "population_difference", "machine_m_2739", "cohort_comparison"]))

    # 3. comparação
    m = t["model_metrics"]
    show = m[(m["population"] == "main") & (m["analysis_set"] == "main") & (m["mode"] == "direct")]
    cols = ["split", "model", "config", "horizon_min", "n", "mae_pp", "rmse_pp", "bias_obs_minus_pred_pp", "p95_abs_error_pp", "max_abs_error_pp", "mase_vs_persistence", "skill_score_rmse", "improvement_vs_persistence_pct", "improvement_vs_xgboost_v2_pct", "inner_selection_mae_pp"]
    if go:
        mat = test.reindex([x for x in order if x in test.index])
        fig = go.Figure(go.Heatmap(z=mat.values, x=[H[h] for h in mat.columns], y=[L[x] for x in mat.index], colorscale=[[0, "#cde2fb"], [1, "#0d366b"]], text=np.vectorize(lambda v: br(v))(mat.values), texttemplate="%{text}", colorbar={"title": "MAE"}))
        fig.update_layout(title="MAE de teste (A)", height=460, yaxis={"autorange": "reversed"})
        blocks["comparacao"].append(plot(fig))
    blocks["comparacao"].append("<h3>Métricas (validação agregada e teste; clique no cabeçalho para ordenar)</h3>" + table(show[cols]))
    blocks["comparacao"].append(md_section(markdown, "4. Resultados quantitativos").split("<h3>4.3")[0])
    blocks["comparacao"].append(csv_links(["model_metrics", "model_metrics_by_fold", "model_rankings", "fold_wins", "predictions", "error_distribution"]))

    # 4. temporal
    if go:
        e = t["errors_by_origin"]
        e = e[e["config"].isin(["A", "-"])]
        highlight = set(cfg["models"]["highlight"]) | {f["best_neural"]}
        for h in (1, 6):
            part = e[e["horizon_steps"] == h]
            fig = go.Figure()
            obs = part[part["model"] == "persistence"].sort_values("target_index")
            fig.add_scatter(x=obs["target_time_h"], y=obs["y_obs_cpu_pct"], name="observado", line={"color": "#0b0b0b", "width": 2})
            for mo in [x for x in order if x in set(part["model"])]:
                p = part[part["model"] == mo].sort_values("target_index")
                fig.add_scatter(x=p["target_time_h"], y=p["y_pred_cpu_pct"], name=L[mo], line={"color": col(mo), "width": 1.2, "dash": "dash" if mo == "persistence" else "solid"}, visible=True if mo in highlight else "legendonly")
            fig.update_layout(title=f"CPU observada e prevista no teste, {H[h]} (eixo no instante previsto; demais modelos na legenda)", xaxis_title="tempo desde o início do trace (h)", yaxis_title="CPU (%)", height=430)
            blocks["temporal"].append(plot(fig))
    blocks["temporal"].append(table(t["diagnostic_windows"], 2))
    blocks["temporal"].append(md_section(markdown, "4. Resultados quantitativos").split("<h3>4.3")[-1] if "<h3>4.3" in md_section(markdown, "4. Resultados quantitativos") else "")
    blocks["temporal"].append(csv_links(["errors_by_origin", "errors_by_regime", "residual_autocorrelation", "diagnostic_windows"]))

    # 5. picos
    pk = t["peak_analysis"]
    pk = pk[pk["config"].isin(["A", "-"]) & (pk["threshold_kind"] == "desenvolvimento_p90")]
    blocks["picos"].append(table(pk[["model", "horizon_min", "threshold_cpu_pct", "n_peak_points", "peak_mae_pp", "peak_bias_obs_minus_pred_pp", "peak_underestimation_frequency", "peak_mean_underestimation_pp", "peak_max_underestimation_pp", "nonpeak_mae_pp"]], 2))
    al = t["peak_alerts"]
    blocks["picos"].append("<h3>Antecipação de eventos</h3>" + table(al[al["threshold_kind"] == "desenvolvimento_p90"][["signal", "horizon_min", "n_events_evaluable", "n_anticipated", "n_alert_episodes", "n_false_alarm_episodes", "mean_lead_steps"]], 2))
    blocks["picos"].append(md_section(markdown, "6. Análise dos picos"))
    blocks["picos"].append(csv_links(["peak_thresholds", "peak_analysis", "peak_events", "peak_alerts"]))

    # 6. estatística
    s = t["statistical_comparisons"]
    pre = s[(s["analysis_type"] == "pre_registrada") & (s["loss"] == "erro_absoluto")]
    if go:
        lab = [f"{L.get(a, a)} {ca} × {L.get(b, b)} {cb} · {H[h]}" for a, ca, b, cb, h in zip(pre["model_a"], pre["config_a"], pre["model_b"], pre["config_b"], pre["horizon_steps"])]
        fig = go.Figure(go.Scatter(x=pre["mean_diff"], y=lab, mode="markers", marker={"color": np.where(pre["significant"], "#184f95", "#9ec5f4"), "size": 8}, error_x={"type": "data", "symmetric": False, "array": pre["ci_high"] - pre["mean_diff"], "arrayminus": pre["mean_diff"] - pre["ci_low"], "color": "#52514e"}))
        fig.add_vline(x=0, line_color="#e34948")
        fig.update_layout(title="Comparações pré-registradas: ΔMAE e IC 95% (bloco 1 h); escuro = IC sem zero", xaxis_title="ΔMAE (p.p.)", height=max(500, 16 * len(lab)), yaxis={"autorange": "reversed", "tickfont": {"size": 10}})
        blocks["estatistica"].append(plot(fig))
    blocks["estatistica"].append(table(s[["analysis_type", "comparison", "loss", "quantile", "horizon_steps", "model_a", "config_a", "model_b", "config_b", "mean_diff", "relative_diff", "ci_low", "ci_high", "significant", "p_value_bootstrap", "p_holm", "holm_significant", "significant_across_blocks", "practical_relevance"]], 4, 300))
    blocks["estatistica"].append(csv_links(["statistical_comparisons", "bootstrap_intervals"]))

    # 7. neural
    g = t["generalization"]
    if go:
        fig = go.Figure()
        for h, c in zip((1, 3, 6), ("#9ec5f4", "#3987e5", "#104281")):
            p = g[g["horizon_steps"] == h]
            fig.add_bar(name=H[h], y=[L.get(x, x) for x in p["model"]], x=p["normalized_generalization_ratio"], orientation="h", marker_color=c)
        fig.add_vline(x=1, line_color="#0b0b0b")
        fig.update_layout(title="(MAE teste / MAE seleção) ÷ mesma razão da persistência", barmode="group", height=480)
        blocks["neural"].append(plot(fig))
    blocks["neural"].append(table(g, 3))
    if "neural_fit_summary" in t:
        blocks["neural"].append("<h3>Indicadores de ajuste (média entre folds)</h3>" + table(t["neural_fit_summary"].drop(columns=["note"]), 2))
    blocks["neural"].append(table(t["cost_tradeoff"], 3))
    blocks["neural"].append(md_section(markdown, "5. Interpretação dos resultados das redes neurais"))
    blocks["neural"].append(csv_links(["neural_training_history", "neural_fit_diagnostics", "neural_fit_summary", "generalization", "neural_window_selection", "model_complexity", "computational_performance", "cost_tradeoff"]))

    # 8. ablação
    blocks["ablacao"].append(table(t["ablation_results"], 3))
    if "feature_importance" in t:
        fi = t["feature_importance"].groupby(["model", "config", "horizon_steps", "group"])["share"].sum().unstack(fill_value=0).reset_index()
        blocks["ablacao"].append("<h3>Fração da importância TreeSHAP por grupo</h3>" + table(fi, 3))
    blocks["ablacao"].append(md_section(markdown, "10. Ablação e importância de atributos"))
    blocks["ablacao"].append(csv_links(["ablation_results", "ablation_grid", "feature_importance"]))

    # 9. probabilístico
    if go and "quantile_predictions" in t:
        q = t["quantile_predictions"]
        q = q[(q["config"] == "A") & (q["split"] == "test_final") & (q["horizon_steps"] == 6)].sort_values("target_index")
        fig = go.Figure()
        fig.add_scatter(x=q["target_time_h"], y=q["q95_adjusted_cpu_pct"], name="Q95 ajustado", line={"color": "#6250d6", "width": 0.8})
        fig.add_scatter(x=q["target_time_h"], y=q["q50_cpu_pct"], name="Q50", line={"color": "#6250d6", "width": 1.2}, fill="tonexty", fillcolor="rgba(98,80,214,0.15)")
        fig.add_scatter(x=q["target_time_h"], y=q["y_obs_cpu_pct"], name="observado", line={"color": "#0b0b0b", "width": 1.5})
        fig.update_layout(title="Quantis no teste, 30 min (faixa unilateral entre Q50 e Q95 ajustado)", xaxis_title="tempo desde o início do trace (h)", yaxis_title="CPU (%)", height=420)
        blocks["probabilistico"].append(plot(fig))
    blocks["probabilistico"].append(table(t["quantile_calibration"], 3))
    blocks["probabilistico"].append(md_section(markdown, "7. Previsão probabilística"))
    blocks["probabilistico"].append(csv_links(["quantile_predictions", "quantile_calibration"]))

    # 10. energia
    if go:
        cum = t["energy_cumulative"]
        cum = cum[(cum["r"] == float(cfg["energy"]["reference_exponent"])) & (cum["window_min"] == 30.0)]
        fig = go.Figure()
        for mo in [x for x in order if x in set(cum["model"])]:
            p = cum[cum["model"] == mo]
            fig.add_scatter(x=p["window_end_time_h"], y=p["cumulative_projected_minus_reference_wh"], name=L[mo], line={"color": col(mo)})
        fig.add_hline(y=0, line_color="#0b0b0b")
        fig.update_layout(title="Diferença acumulada de energia modelada, 48 janelas de 30 min (r = 1,4)", xaxis_title="tempo desde o início do trace (h)", yaxis_title="projetada − referência (Wh)", height=420)
        blocks["energia"].append(plot(fig))
        j = t["jensen_analysis"]
        fig = go.Figure()
        for r, c in ((1.4, "#3987e5"), (1.8, "#104281")):
            p = j[j["r"] == r]
            fig.add_scatter(x=p["time_h"], y=p["jensen_gap_w_per_server"], name=f"r = {br(r, 1)}", line={"color": c, "width": 1})
        fig.update_layout(title="Gap de Jensen por servidor substituto: média de P(u_i) − P(média de u)", xaxis_title="tempo desde o início do trace (h)", yaxis_title="W por servidor", height=380)
        blocks["energia"].append(plot(fig))
    em = t["energy_metrics"]
    blocks["energia"].append(table(em.drop(columns=["quantity"]), 3))
    blocks["energia"].append(md_section(markdown, "8. Análise energética"))
    blocks["energia"].append(csv_links(["energy_timeseries", "energy_origin_errors", "energy_metrics", "energy_cumulative", "energy_sensitivity", "jensen_analysis", "jensen_summary"]))

    blocks["limitacoes"].append(md_section(markdown, "11. Ameaças à validade"))
    blocks["conclusoes"].append(md_section(markdown, "12. Conclusões sustentadas pelos dados"))
    blocks["conclusoes"].append('<p class="note">Relatório completo: <a href="../reports/scientific-analysis.md">scientific-analysis.md</a>. Dicionário de dados: <a href="../metadata/data_dictionary.md">data_dictionary.md</a>.</p>')

    for key in FIG_SECTION:
        blocks[key].append(figs_for(FIG_SECTION[key]))

    nav = "".join(f'<a href="#{k}">{html.escape(title)}</a>' for k, title in SECTIONS)
    body = "".join(f'<section id="{k}"><h2>{html.escape(title)}</h2>{"".join(blocks[k])}</section>' for k, title in SECTIONS)
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Análise expandida da V3</title><style>{CSS}</style></head>
<body><header><h1>Análise expandida da V3: previsão de CPU e energia de TI modelada</h1>
<p>Corrida <code>{html.escape(info['run_id'])}</code> · Alibaba Cluster Trace v2018 · população principal de 4.023 máquinas · teste de 24 h (288 origens, já inspecionado na V2).</p>
<p class="note">Energia e potência são estimativas paramétricas de um servidor substituto do SPECpower_ssj2008, não medições; PUE não foi calculado. Tempo em horas desde o início do trace, não horário civil.</p></header>
<nav><div>{nav}</div></nav><main>{body}</main><script>{SORT_JS}</script></body></html>
"""
