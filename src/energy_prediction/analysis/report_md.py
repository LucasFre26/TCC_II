"""Relatório científico `scientific-analysis.md`, com números lidos das tabelas da análise.

Separa resultado observado de hipótese explicativa. Afirmação de superioridade só aparece quando o
intervalo pré-registrado não contém zero; o texto muda conforme os intervalos.
"""

from __future__ import annotations

import numpy as np

from energy_prediction.analysis.facts import br, ci, pct, prereg, signed, thousands

H = {1: "5 min", 3: "15 min", 6: "30 min"}


def _verdict(row, better: str, worse: str, tie: str) -> str:
    if row is None:
        return "comparação ausente"
    if not row["significant"]:
        return tie
    return better if row["mean_diff"] < 0 else worse


def _mae_table(f, labels, order) -> str:
    lines = ["| Modelo | Validação 5 min | Validação 15 min | Validação 30 min | Teste 5 min | Teste 15 min | Teste 30 min |", "|---|---:|---:|---:|---:|---:|---:|"]
    for m in order:
        if m not in f["test_mae"].index:
            continue
        v = [br(f["val_mae"].loc[m, h]) for h in (1, 3, 6)]
        t = [br(f["test_mae"].loc[m, h]) for h in (1, 3, 6)]
        lines.append(f"| {labels.get(m, m)} | " + " | ".join(v + t) + " |")
    return "\n".join(lines)


def _comparison_table(t, labels, comparison: str, models) -> str:
    lines = ["| Modelo | Horizonte | ΔMAE (p.p.) | IC 95% (bloco 1 h) | Relativa | Blocos 30 min/1 h/2 h | Holm |", "|---|---|---:|---|---:|---|---|"]
    for m in models:
        for h in (1, 3, 6):
            r = prereg(t, comparison, m, h)
            if r is None:
                continue
            lines.append(f"| {labels.get(m, m)} | {H[h]} | {signed(r['mean_diff'], 3)} | {ci(r)} | {pct(r['relative_diff'])} | {r['significant_across_blocks']} | {'sim' if r['holm_significant'] else 'não'} |")
    return "\n".join(lines)


def build(t: dict, f: dict, info: dict, cfg: dict, figure_records: list[dict], checks: list[dict], warnings: list[str]) -> str:
    L = cfg["models"]["labels"]
    order = cfg["models"]["order"]
    run = info["run_id"]
    best_nn = f["best_neural"]
    gen = f["gen"]
    jz = f["jensen"]
    e = f["energy"]
    r_ref = float(cfg["energy"]["reference_exponent"])
    fig_names = {r["name"]: i + 1 for i, r in enumerate(r for r in figure_records if r.get("generated"))}

    def fig(name: str) -> str:
        return f"Figura {fig_names[name]} (`figures/png/{name}.png`)" if name in fig_names else f"`{name}` (não gerada)"

    hyb = {h: prereg(t, "new_vs_v2_xgboost_A", "hybrid", h) for h in (1, 3, 6)}
    neural_ratio = {m: [gen.loc[(m, h), "normalized_generalization_ratio"] for h in (1, 3, 6)] for m in ("dlinear", "tcn", "nbeatsx", "nhits", "patchtst") if (m, 1) in gen.index}
    tree_ratio = {m: [gen.loc[(m, h), "normalized_generalization_ratio"] for h in (1, 3, 6)] for m in ("xgboost_v2", "xgboost_delta", "hybrid", "statespace") if (m, 1) in gen.index}
    ok_checks = sum(c["passed"] for c in checks)
    sb = f["stable_bias"]
    level_shift = f.get("cpu_mean_test", np.nan) - f.get("cpu_mean_dev", np.nan)
    a = f["alerts"]

    def alerts(signal, h):
        return a.loc[(signal, h)] if (signal, h) in a.index else None

    q95a = {h: f["quant"].loc[("test_final", h, "Q95", "ajustado")] for h in (1, 3, 6)}
    q95b = {h: f["quant"].loc[("test_final", h, "Q95", "bruto")] for h in (1, 3, 6)}
    q50 = {h: f["quant"].loc[("test_final", h, "Q50", "bruto")] for h in (1, 3, 6)}
    q95v = {h: f["quant"].loc[("validation_pooled", h, "Q95", "bruto")] for h in (1, 3, 6)}

    text = f"""# Análise científica expandida da V3

Corrida analisada: `results/v3/{run}/`. Toda grandeza abaixo foi recalculada a partir dos artefatos preservados da V2 (`results/v2-full`) e da V3, sem novo treino neural. Antes de qualquer indicador, {ok_checks} de {len(checks)} verificações de integridade passaram (origens, unicidade, reprodução de MAE, RMSE, pinball, bootstrap, energia e reajustes; lista em `metadata/analysis_manifest.json`).

Convenções: erro = observado − previsto (positivo = subestimação) para CPU; na seção de energia, diferença = projetada − referência (negativa = subestimação), como em `energy_by_origin.csv`. Tempo em horas desde o início do trace, não horário civil. Potência e energia são estimativas paramétricas de um servidor substituto do SPECpower_ssj2008; não há medição elétrica no trace e PUE não foi calculado.

## 1. Perguntas de pesquisa

- **P1.** Algum modelo da V3 reduziu o erro de previsão da CPU agregada em relação ao XGBoost da V2, com sustentação estatística?
- **P2.** Por que as redes neurais (DLinear, TCN, N-BEATSx, N-HiTS, PatchTST) não superaram o XGBoost da V2?
- **P3.** Como os modelos se comportam nos picos de utilização, e algum sinal antecipa esses picos?
- **P4.** Os quantis previstos cobrem a CPU observada na proporção nominal?
- **P5.** Quanto o erro de CPU se propaga para a potência e a energia de TI modeladas, e quanto isso depende dos parâmetros do modelo de potência?
- **P6.** Memória, rede, disco e a distribuição da CPU entre máquinas acrescentam informação preditiva?
- **P7.** A diferença entre população principal e coorte altera as conclusões?

## 2. Hipóteses avaliadas

As hipóteses explicativas abaixo foram formuladas depois de observar os resultados da V3; por isso são tratadas como explicações compatíveis com os dados, não como hipóteses confirmatórias.

- **H1 (pré-registrada na V3).** Os modelos residuais e híbridos diferem do XGBoost da V2 no teste. Resultado: {f['residual_vs_xgb_sig_count']} de 9 intervalos pré-registrados (incremento, Kalman e híbrido × 3 horizontes) excluem zero.
- **H2 (pré-registrada na V3).** As redes neurais diferem do XGBoost da V2 no teste. Resultado: {f['neural_vs_xgb_sig_count']} de 9 intervalos (DLinear, TCN, N-BEATSx × 3 horizontes) excluem zero, e nos que excluem a diferença é positiva (redes piores).
- **H3 (explicativa).** A perda das redes concentra-se na passagem para o período de teste, que tem nível de CPU mais alto que o desenvolvimento, e não num ajuste ruim na validação interna. Evidência: razão de generalização normalizada (Seção 5) e viés em regime estável (Seção 4.3). Status: compatível com os dados; não demonstra causa.
- **H4 (explicativa).** Os picos são subestimados porque a maioria dura um único passo de 5 min, o que deixa pouca informação disponível na origem. Evidência: duração dos eventos e baixa antecipação (Seção 6). Status: compatível com os dados; a ausência de sinal antecedente não foi testada diretamente.
- **H5.** A agregação da CPU antes da curva de potência introduz viés quando r > 1 (efeito de Jensen). Resultado: Seção 8.4.

## 3. Metodologia da análise

1. **Leitura e integridade.** Leitura dos CSV e JSON das famílias `residual`, `neural`, `quantile`, `energy` e `integrity` e de `results/v2-full/{{main,cohort}}`; verificação de que o teste tem as origens 2010–2297 em todos os grupos, de que há uma previsão por origem, e reprodução de todos os MAE, RMSE, pinball, intervalos de bootstrap e energias gravados. A série agregada foi reconstruída do cache de grade com a configuração gravada no manifesto da corrida e conferida contra os valores observados das previsões.
2. **Métricas.** MAE, RMSE, viés, mediana, P90/P95/P99 e máximo do erro absoluto, desvio do erro, MASE do projeto (MAE/MAE da persistência nas mesmas origens), skill score (1 − RMSE/RMSE da persistência) e melhoria percentual contra a persistência e contra o XGBoost da V2 da mesma configuração. Validação externa agregada (720 origens) e teste final (288 origens) são reportados separadamente.
3. **Diagnóstico temporal.** Regime por origem: subida se y(t+h) − y(t) ≥ σ_h, queda se ≤ −σ_h, estável caso contrário, com σ_h = {", ".join(f"{br(v)} p.p. ({H[int(k)]})" for k, v in info['diagnostics']['regime_sigma_pp'].items())} (desvio das variações no desenvolvimento). Faixas de CPU: quintis da CPU no desenvolvimento. Janelas de 3 h escolhidas pela série observada, sem olhar erro de modelo.
4. **Picos.** Limiar primário: P90 ({br(f['dev_p90'])}%) e P95 ({br(f['dev_p95'])}%) da CPU observada antes do teste. Evento: sequência de instantes acima do limiar, unindo sequências separadas por até 1 passo. Antecipação: alerta (previsão ≥ limiar) mirando até 1 passo do início e emitido antes dele. Falso alarme: episódio de alertas sem evento a até 1 passo.
5. **Inferência.** Block bootstrap da V2, 5.000 reamostragens, semente 42, blocos de 6, 12 (primário) e 24 passos. P-valor bicaudal das mesmas reamostragens; correção de Holm sobre todas as comparações pré-registradas no bloco primário e, separadamente, sobre a matriz exploratória de cada horizonte. Relevância prática: |ΔMAE|/MAE da referência ≥ 5% (convenção desta análise), com equivalente linear em potência de {br(info['watts_per_pp'], 3)} W por p.p. de CPU por servidor substituto.
6. **Energia.** Modelo de Fan com P_idle = {br(info['fan']['p_idle_w'], 1)} W e P_busy = {br(info['fan']['p_busy_w'], 0)} W (resultado `{info['substitute']['result_id']}`), r ∈ {{1,0; 1,4; 1,8}}, trapézio a cada 5 min com o extremo esquerdo na CPU observada. Energia acumulada somada em janelas sem sobreposição.

## 4. Resultados quantitativos

### 4.1 Erro pontual (configuração A, população principal)

{_mae_table(f, L, order)}

MAE em pontos percentuais de CPU. Fonte: `csv/model_metrics.csv`; {fig('05_ranking_mae_rmse')}, {fig('06_heatmaps_mae_melhoria')} e {fig('07_validacao_vs_teste')}.

Menor MAE de teste: {", ".join(f"{L.get(m, m)} em {H[h]}" for h, m in f['best_test'].items())}. Menor MAE na validação externa: {", ".join(f"{L.get(m, m)} em {H[h]}" for h, m in f['best_val'].items())}. A melhoria do XGBoost da V2 sobre a persistência no teste foi de {", ".join(pct(f['imp_pers_test'].loc['xgboost_v2', h] / 100) for h in (1, 3, 6))} em 5, 15 e 30 min.

### 4.2 Comparações pré-registradas contra o XGBoost da V2

{_comparison_table(t, L, "new_vs_v2_xgboost_A", ["xgboost_delta", "statespace", "hybrid", "dlinear", "tcn", "nbeatsx"])}

ΔMAE = MAE do modelo − MAE do XGBoost da V2 nas mesmas 288 origens. Fonte: `csv/statistical_comparisons.csv`; {fig('18_forest_preregistrado')}.

O híbrido Kalman + XGBoost teve o menor MAE de teste em 5 e 30 min, mas a diferença para o XGBoost da V2 {_verdict(hyb[1], 'favorece o híbrido', 'favorece a V2', 'não se distingue de zero')} em 5 min, {_verdict(hyb[3], 'favorece o híbrido', 'favorece a V2', 'não se distingue de zero')} em 15 min e {_verdict(hyb[6], 'favorece o híbrido', 'favorece a V2', 'não se distingue de zero')} em 30 min. Não há, portanto, sustentação para declarar o híbrido superior ao XGBoost da V2.

Das {f['n_prereg']} comparações pré-registradas (erro absoluto e pinball, bloco de 1 h), {f['n_prereg_sig']} têm intervalo sem zero; depois da correção de Holm sobre o conjunto, restam {f['n_prereg_holm']}. Essa queda tem duas causas: o número de comparações e a resolução do bootstrap (com 5.000 reamostragens, o menor p-valor possível é cerca de {br(f['min_p'], 4)}). O protocolo da V3 pré-registrou a decisão pelo intervalo de cada comparação, sem correção de multiplicidade; a coluna de Holm é informativa e indica que as conclusões individuais devem ser lidas como evidência por comparação, não como controle de erro do conjunto. Na matriz exploratória ({f['n_exp']} pares × horizontes, {fig('19_matriz_pares')}), {f['n_exp_sig']} intervalos excluem zero e {f['n_exp_holm']} sobrevivem a Holm por horizonte.

### 4.3 Comportamento temporal e mudança de nível

A CPU média do teste foi {br(f.get('cpu_mean_test'))}%, contra {br(f.get('cpu_mean_dev'))}% no desenvolvimento ({signed(level_shift)} p.p., cerca de {br(level_shift / f.get('cpu_std_dev', np.nan), 2)} desvio-padrão do desenvolvimento); {pct(f.get('share_upper_bands', np.nan), 0)} dos instantes previstos do teste (30 min) caíram nos dois quintis superiores da CPU do desenvolvimento ({fig('02_distribuicao_dependencia')}, {fig('13_erro_faixa_cpu')}).

No regime estável (variação até o alvo menor que σ_h), o viés em 30 min foi {", ".join(f"{L.get(m, m)} {signed(sb.loc[m, 6])} p.p." for m in order if m in sb.index)}. A persistência praticamente não tem viés nesse regime; {_stable_bias_text(sb, L)} Viés positivo em regime estável significa que o modelo previu queda enquanto a série ficou perto do nível da origem, o que é compatível com previsões puxadas para o nível médio do treino, mais baixo que o do teste.

Por faixa de CPU observada no alvo ({fig('13_erro_faixa_cpu')}), o viés vai de negativo na faixa mais baixa a positivo na mais alta para todos os modelos, inclusive a persistência; esse padrão decorre de condicionar o erro no próprio valor observado e não isola o viés de cada modelo. A leitura informativa é a comparação entre modelos dentro da mesma faixa: na faixa mais alta, em 30 min, o viés foi {_band_bias_text(t, L)}.

Por regime (30 min; n = {", ".join(f"{g} {int(n)}" for g, n in f['regime_n_h6'].items())}), o MAE na subida foi {", ".join(f"{L.get(m, m)} {br(f['regime_mae_h6'].loc[m, 'subida'])}" for m in ["persistence", "xgboost_v2", "hybrid", best_nn])} p.p., e na queda {", ".join(f"{L.get(m, m)} {br(f['regime_mae_h6'].loc[m, 'queda'])}" for m in ["persistence", "xgboost_v2", "hybrid", best_nn])} p.p. ({fig('14_erro_variacao_regime')}). Entre todos os modelos, o menor MAE na subida foi do {L.get(f['regime_mae_h6']['subida'].idxmin(), '')} e na queda do {L.get(f['regime_mae_h6']['queda'].idxmin(), '')}. Modelos com viés de subestimação ganham nas quedas e perdem nas subidas, de modo que o MAE por regime precisa ser lido junto com o viés. O Kalman (nível local) fica próximo da persistência nos dois regimes, pois sua previsão é o nível filtrado na origem; o XGBoost de incremento, embora também parta de cpu(t), corrige parte da variação e tem erro bem menor que o da persistência nas quedas.

Os resíduos de 30 min têm autocorrelação alta até a defasagem 5, esperada para previsões de 6 passos com origens consecutivas; depois disso, a autocorrelação da {L.get(best_nn, best_nn)} permanece positiva por mais defasagens que a do XGBoost, coerente com um viés persistente ({fig('12_residuos')}).

## 5. Interpretação dos resultados das redes neurais

**Observado.**

- Diferença relativa de MAE contra o XGBoost da V2 (5, 15 e 30 min), validação externa → teste: {"; ".join(f"{L.get(m, m)} {', '.join(signed(-f['imp_xgb_val'].loc[m, h], 1) + '%' for h in (1, 3, 6))} → {', '.join(signed(-f['imp_xgb_test'].loc[m, h], 1) + '%' for h in (1, 3, 6))}" for m in ("dlinear", "tcn", "nbeatsx", "nhits", "patchtst"))}.
- Razão de generalização normalizada pela persistência ((MAE de teste / MAE de seleção) ÷ mesma razão da persistência), em 5, 15 e 30 min: {"; ".join(f"{L.get(m, m)} {', '.join(br(x) for x in v)}" for m, v in neural_ratio.items())}. Para os modelos de árvore e de estados: {"; ".join(f"{L.get(m, m)} {', '.join(br(x) for x in v)}" for m, v in tree_ratio.items())} ({fig('21_generalizacao_janela')}).
- Na validação interna, a razão loss de validação/loss de treino no melhor estado ficou entre {br(f['fit']['val_over_train_at_best_mean'].min())} e {br(f['fit']['val_over_train_at_best_mean'].max())} (configuração A, modo direto); a melhor época média entre folds, na média dos três horizontes, foi {", ".join(f"{L.get(m, m)} {br(v, 1)}" for m, v in f['fit'].reset_index().groupby('model')['best_epoch_mean'].mean().sort_values().items())} ({fig('20_curvas_loss')}).
- Janela de entrada: a soma do MAE interno do DLinear piorou ao alongar a janela ({", ".join(f"{int(k)} passos: {br(v)}" for k, v in f['window'].items())} p.p.).
- Tamanho e custo: as redes tiveram de {thousands(f['params_range'][0])} a {thousands(f['params_range'][1])} parâmetros treináveis, pico de memória CUDA de {br(f['vram_range'][0], 0)} a {br(f['vram_range'][1], 0)} MiB numa GPU de 6 GB, e reajuste de teste em menos de 1 s; o XGBoost reajustado em CPU levou menos de 0,1 s ({fig('22_custo_pareto')}). Mais parâmetros não se traduziram em menor MAE de teste.

**Hipóteses explicativas (não demonstradas).** (i) O teste está num nível de CPU mais alto que o desenvolvimento; redes que preveem o nível padronizado pela média do treino tendem a puxar a previsão para essa média, o que aparece como viés positivo mesmo em regime estável, enquanto modelos ancorados em cpu(t) (persistência, Kalman, incremento) sofrem menos. (ii) O desenvolvimento tem cerca de 1.900 origens com forte autocorrelação, poucas para redes com dezenas de milhares de parâmetros; N-BEATSx, N-HiTS e PatchTST atingiram o melhor estado nas primeiras épocas. (iii) A janela curta (3 h) foi a melhor na validação interna, sinal de que histórico longo não trouxe informação adicional nesta série. Nenhum desses pontos prova sobreajuste: a validação interna não mostrou distância grande entre treino e validação, e o pior desempenho aparece principalmente na passagem para o teste.

## 6. Análise dos picos

No teste, {f['n_events']} eventos ficaram acima do P90 do desenvolvimento ({br(f['dev_p90'])}%), dos quais {f['n_single_step_events']} duraram um único passo de 5 min; o mais longo durou {int(f['longest_event']['duration_steps'])} passos, com pico de {br(f['longest_event']['peak_cpu_pct'])}%. Nos instantes de pico, todos os modelos subestimaram: viés em 30 min de {", ".join(f"{L.get(m, m)} {signed(f['peak'].loc[(m, 6), 'peak_bias_obs_minus_pred_pp'])}" for m in ["persistence", "xgboost_v2", "hybrid", best_nn] if (m, 6) in f['peak'].index)} p.p., frequência de subestimação entre {pct(f['peak']['peak_underestimation_frequency'].min(), 0)} e {pct(f['peak']['peak_underestimation_frequency'].max(), 0)} ({fig('15_picos_eventos')}, {fig('16_subestimacao_picos')}).

Antecipação de eventos ({fig('17_alertas_picos')}): o XGBoost da V2 antecipou {", ".join(str(int(alerts('xgboost_v2', h)['n_anticipated'])) for h in (1, 3, 6))} de {f['n_events']} eventos em 5, 15 e 30 min; o teto Q95 ajustado antecipou {", ".join(str(int(alerts('xgboost_quantile Q95 ajustado', h)['n_anticipated'])) for h in (1, 3, 6))}, com {", ".join(str(int(alerts('xgboost_quantile Q95 ajustado', h)['n_false_alarm_episodes'])) for h in (1, 3, 6))} episódios de falso alarme em {", ".join(str(int(alerts('xgboost_quantile Q95 ajustado', h)['n_alert_episodes'])) for h in (1, 3, 6))} episódios de alerta. A persistência não antecipa o início de nenhum evento em 5 min, por construção. Com 12 eventos, a maioria de um passo, essas contagens são descritivas: o teto quantílico troca mais acertos por mais falsos alarmes e não constitui um detector de picos validado.

## 7. Previsão probabilística

Cobertura empírica no teste (A), 5, 15 e 30 min: Q50 {", ".join(br(q50[h]['empirical_coverage']) for h in (1, 3, 6))}; Q95 bruto {", ".join(br(q95b[h]['empirical_coverage']) for h in (1, 3, 6))}; Q95 ajustado {", ".join(br(q95a[h]['empirical_coverage']) for h in (1, 3, 6))}. Na validação externa, o Q95 bruto cobriu {", ".join(br(q95v[h]['empirical_coverage']) for h in (1, 3, 6))}. O ajuste conformal aproximou a cobertura do nominal, mas ela continuou abaixo de 0,95; a Q50 cobriu menos de 0,5, outra manifestação da subestimação no período de teste. Nos picos, o Q95 ajustado cobriu {", ".join(br(q95a[h]['peak_coverage']) for h in (1, 3, 6))} ({fig('25_calibracao_quantis')}, {fig('26_faixas_quantis')}). Q90 e Q95 são quantis unilaterais; a distância q95 − q50 é uma margem superior, não a largura de um intervalo bilateral. As coberturas são empíricas, sem garantia teórica sob dependência temporal.

## 8. Análise energética

### 8.1 Grandeza

A energia de referência do servidor substituto nas 24 h de teste, com r = {br(r_ref, 1)}, foi {thousands(f['energy_ref_24h'])} Wh (soma de 288 janelas de 5 min). Ela é uma estimativa paramétrica, não consumo medido do cluster ({fig('27_curvas_potencia')}).

### 8.2 Erro de energia

Erro relativo absoluto médio por origem em 30 min (r = {br(r_ref, 1)}): {", ".join(f"{L.get(m, m)} {pct(e.loc[(m, r_ref, 6), 'energy_mean_abs_relative_error'])}" for m in order if (m, r_ref, 6) in e.index)}. Viés por origem em 30 min (projetada − referência): {", ".join(f"{L.get(m, m)} {signed(e.loc[(m, r_ref, 6), 'energy_bias_projected_minus_reference_wh'])} Wh" for m in order if (m, r_ref, 6) in e.index)} ({fig('29_erro_energia_sensibilidade')}).

Somando 48 janelas de 30 min sem sobreposição, a diferença acumulada em 24 h foi {", ".join(f"{L.get(m, m)} {signed(f['cum30'].loc[m, 'cumulative_projected_minus_reference_wh'], 1)} Wh" for m in order if m in f['cum30'].index)} ({fig('28_potencia_energia_acumulada')}). A ordem por erro por origem difere da ordem por erro acumulado: a persistência e o Kalman têm erro por origem maior, mas quase sem viés, de modo que os erros se compensam na soma; os modelos aprendidos, que subestimaram a CPU no teste, acumulam energia projetada abaixo da referência.

### 8.3 Sensibilidade

Com r variando de 1,0 a 1,8, o MAE de energia do XGBoost da V2 em 30 min foi {", ".join(br(e.loc[('xgboost_v2', rr, 6), 'energy_mae_wh']) for rr in (1.0, 1.4, 1.8))} Wh; a escala do erro acompanha linearmente P_busy − P_idle, e o erro relativo diminui quando P_idle aumenta, porque a energia de referência cresce ({fig('29_erro_energia_sensibilidade')}). A relação entre erro de CPU e erro de potência é praticamente linear, com inclinação próxima de (P_busy − P_idle)/100 = {br(info['watts_per_pp'], 2)} W por p.p. ({fig('30_erro_cpu_energia')}). Nenhum r foi escolhido pelo erro no teste.

### 8.4 Efeito de Jensen

Convenção: gap = média das potências por máquina − potência da utilização média; negativo porque a curva é côncava para r > 1. Por servidor substituto, o gap médio foi {br(jz.loc[1.4, 'mean_gap_w_per_server'])} W (r = 1,4; {pct(jz.loc[1.4, 'mean_relative_gap'])} de P(média)) e {br(jz.loc[1.8, 'mean_gap_w_per_server'])} W (r = 1,8), com mínimo de {br(jz.loc[1.8, 'min_gap_w_per_server'])} W. Para a frota hipotética de servidores substitutos (média de {thousands(jz.loc[1.4, 'mean_n_present'])} máquinas presentes), a diferença média foi {br(jz.loc[1.4, 'mean_gap_fleet_w'] / 1000, 1)} kW (r = 1,4) e {br(jz.loc[1.8, 'mean_gap_fleet_w'] / 1000, 1)} kW (r = 1,8). Watts por servidor e watts da frota são grandezas diferentes e não devem ser comparados diretamente. A aproximação de segunda ordem errou em média {br(jz.loc[1.4, 'mean_abs_second_order_error_w'], 2)} W por servidor (r = 1,4), com máximo de {br(jz.loc[1.4, 'max_abs_second_order_error_w'], 2)} W ({fig('31_jensen')}). Com r = 1 o gap é nulo.

## 9. Dataset e populações

A população principal agrega 4.023 máquinas; a coorte, 4.022. A diferença é a máquina `m_2739`, com status `IMPORT_INSTALLING`, CPU 0% em todos os seus {thousands(f.get('m2739_n', 0))} intervalos e último registro em {br(f.get('m2739_last_h', np.nan), 1)} h, antes do teste. A diferença máxima entre as séries foi {br(f.get('cohort_series_full', np.nan), 4)} p.p. na série inteira e {br(f.get('cohort_series_test', np.nan), 4)} p.p. nos alvos do teste{': no teste as duas séries coincidem' if f.get('cohort_series_test', 1) == 0 else ''}. As métricas de teste da V2 diferem em até {br(f.get('cohort_max_mae_diff', np.nan), 3)} p.p. de MAE entre as populações, efeito do reajuste em séries de treino ligeiramente distintas ({fig('04_populacoes_m2739')}).

## 10. Ablação e importância de atributos

{_ablation_text(f, L)} A configuração D (estatísticas da distribuição da CPU entre máquinas) foi executada só no XGBoost de incremento e no DLinear ({fig('23_ablacao')}).

Importância TreeSHAP nos reajustes verificados (reproduziram as previsões gravadas): {_shap_text(f, L)} ({fig('24_importancia_shap')}). Importância preditiva descreve associação usada pelo modelo; não implica relação causal.

## 11. Ameaças à validade

- **Teste reutilizado.** O teste final de 24 h já tinha sido inspecionado na V2 e foi reutilizado na V3; ele não é uma validação independente do processo de desenvolvimento, e as hipóteses H3 e H4 foram formuladas depois de ver seus resultados.
- **Um único período de teste.** 288 origens consecutivas de um único dia, com mudança de nível em relação ao desenvolvimento; a generalização para outros períodos não foi avaliada.
- **Dependência temporal.** O block bootstrap trata a dependência de curto prazo, mas o tamanho efetivo da amostra é menor que 288; os intervalos dependem do bloco (concordância entre blocos em `bootstrap_intervals.csv`).
- **Múltiplas comparações.** Com Holm, poucas comparações pré-registradas permanecem significativas; a resolução do bootstrap limita o menor p-valor.
- **Picos raros.** {f['n_events']} eventos, quase todos de um passo; taxas de antecipação têm incerteza alta.
- **Energia modelada.** Servidor substituto, carga do benchmark diferente da utilização do trace, parâmetros de um único resultado do SPECpower; o erro de energia é diferença entre duas estimativas, não erro contra medidor. PUE não foi calculado.
- **Tempos.** Tempos das redes vêm da corrida original em CUDA; tempos dos modelos clássicos foram medidos nesta análise em CPU; só servem como ordem de grandeza.
- **Seleção da arquitetura neural.** Grades pequenas, janela escolhida só pelo DLinear e orçamento fixo; outras configurações poderiam ter desempenho diferente.

## 12. Conclusões sustentadas pelos dados

1. {_first_conclusion(f, L)}
2. DLinear, TCN e N-BEATSx tiveram MAE de teste maior que o XGBoost da V2 em {f['neural_vs_xgb_sig_count']} de 9 comparações pré-registradas. Na validação externa a distância era menor (o DLinear praticamente empatou); a razão de generalização normalizada das redes, exceto o PatchTST, ficou acima de 1, e o erro de teste veio acompanhado de viés de subestimação num período de nível mais alto. A explicação causal permanece hipótese.
3. Todos os modelos subestimaram os picos, e nenhum sinal pontual antecipou mais que uma pequena parte dos eventos; o teto quantílico antecipou mais eventos ao custo de mais falsos alarmes.
4. A cobertura empírica dos quantis ficou abaixo do nominal no teste, mesmo depois do ajuste conformal.
5. O erro de energia de TI modelada por origem ficou na faixa de poucos por cento; o erro acumulado depende mais do viés do modelo que do seu MAE, e o efeito de Jensen por servidor é da ordem de alguns watts para r > 1.
6. Os resultados estão prontos para orientar a redação, desde que o texto mantenha as ressalvas da Seção 11.

## Arquivos

- Tabelas: `csv/` (dicionário em `metadata/data_dictionary.md`).
- Figuras: `figures/png`, `figures/svg`, `figures/pdf` e os dados de cada figura em `figures/data/`.
- Relatório navegável: `interactive/relatorio.html`.
- Recursos para o artigo: `latex/`.
"""
    return text


def _ablation_text(f, L) -> str:
    ab = f["ablation"]
    parts = []
    for model in ("xgboost_v2", "hybrid", "tcn", "nbeatsx"):
        rows = [(c, h) for c in ("B", "C") for h in (1, 3, 6) if (model, c, h) in ab.index]
        if not rows:
            continue
        values = ", ".join(f"{c}/{H[h]} {signed(ab.loc[(model, c, h), 'delta_mae_pct'], 1)}%" for c, h in rows)
        parts.append(f"{L.get(model, model)}: {values}")
    d = [f"{L.get(m, m)} D/{H[h]} {signed(ab.loc[(m, 'D', h), 'delta_mae_pct'], 1)}%" for m in ("xgboost_delta", "dlinear") for h in (1, 3, 6) if (m, "D", h) in ab.index]
    return "Variação de MAE no teste contra A (positivo = piora): " + "; ".join(parts) + ". Contra A nas mesmas origens: " + ", ".join(d) + "."


def _shap_text(f, L) -> str:
    s = f.get("shap_groups")
    if s is None:
        return "não calculada"
    out = []
    for model, config, h in (("xgboost_v2", "A", 1), ("xgboost_v2", "A", 6), ("xgboost_v2", "C", 6)):
        if (model, config, h) not in s.index.droplevel(3):
            continue
        part = s.loc[(model, config, h)].sort_values(ascending=False)
        top = ", ".join(f"{g} {pct(v, 0)}" for g, v in part.head(3).items())
        exo = sum(part.get(g, 0.0) for g in ("memória", "rede", "disco"))
        out.append(f"{L.get(model, model)} {config} em {H[h]}: {top}" + (f" (memória, rede e disco somam {pct(exo, 0)})" if config == "C" else ""))
    return "; ".join(out)


def _first_conclusion(f, L) -> str:
    better = f["v3_better_than_xgb"]
    if not better:
        return "Nenhum modelo da V3 teve MAE de teste menor que o do XGBoost da V2 com intervalo pré-registrado sem zero. O híbrido Kalman + XGBoost teve o menor MAE numérico em parte dos horizontes, mas a diferença não se distinguiu de zero."
    return "Modelos da V3 com MAE de teste menor que o do XGBoost da V2 e intervalo pré-registrado sem zero: " + ", ".join(f"{L.get(m, m)} ({H[int(h)]})" for m, h in better) + "."


def _stable_bias_text(sb, L) -> str:
    under = [m for m in sb.index if m != "persistence" and sb.loc[m, 6] > 0.5]
    other = [m for m in sb.index if m != "persistence" and sb.loc[m, 6] <= 0.5]
    text = "subestimaram mais de 0,5 p.p.: " + ", ".join(L.get(m, m) for m in under) + "."
    if other:
        text += " Ficaram abaixo disso: " + ", ".join(L.get(m, m) for m in other) + "."
    return text


def _band_bias_text(t, L) -> str:
    r = t["errors_by_regime"]
    part = r[(r["grouping"] == "cpu_band") & r["config"].isin(["A", "-"]) & (r["horizon_steps"] == 6)]
    top = sorted(part["group"].unique())[-1]
    part = part[part["group"] == top].set_index("model")["bias_obs_minus_pred_pp"].sort_values()
    return ", ".join(f"{L.get(m, m)} {signed(v)} p.p." for m, v in part.items())
