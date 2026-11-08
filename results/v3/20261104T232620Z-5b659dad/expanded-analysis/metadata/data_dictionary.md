# Dicionário de dados da análise expandida

Separador vírgula, ponto decimal, 12 algarismos significativos, valor ausente = célula vazia.
Natureza: identificador, observado (trace), previsto (modelo), erro, métrica (calculada nesta análise),
inferência (bootstrap), modelado (potência e energia paramétricas, não medidas), registro (copiado da corrida) ou parâmetro.
Tempo: segundos ou horas desde o início do trace, não horário civil.

## predictions.csv

Previsões pontuais da V2 e da V3, uma linha por modelo, configuração, modo, população, split, horizonte e origem.
Grão: previsão.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `version` | Versão do experimento que gerou a previsão (V2 ou V3) | - | identificador |
| `family` | Família do modelo (referência, V2, V3 residual, V3 neural) | - | identificador |
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `population` | População agregada: main (4.023 máquinas) ou cohort (4.022 máquinas ativas com 96 CPUs) | - | identificador |
| `analysis_set` | Conjunto de origens: main (tabela A/B/C) ou d_contrast (origens do contraste D) | - | identificador |
| `split` | Partição: fold_k (validação externa), validation_pooled (cinco folds juntos), test_final (teste de 24 h já inspecionado na V2) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `origin_index` | Índice do intervalo da origem (0 = início do trace) | intervalo de 300 s | identificador |
| `origin_time_s` | Tempo da origem desde o início do trace (não é horário civil) | s | identificador |
| `target_index` | Índice do instante previsto (origem + horizonte) | intervalo de 300 s | identificador |
| `target_time_s` | Tempo do instante previsto desde o início do trace (não é horário civil) | s | identificador |
| `target_time_h` | Tempo do instante previsto desde o início do trace (não é horário civil) | h | identificador |
| `y_obs_cpu_pct` | CPU agregada observada no instante previsto | % de CPU | observado |
| `y_pred_cpu_pct` | CPU agregada prevista para o instante previsto | % de CPU | previsto |
| `error_obs_minus_pred_pp` | Erro assinado = observado − previsto; positivo indica subestimação | pontos percentuais de CPU (p.p.) | erro |
| `abs_error_pp` | Erro absoluto |observado − previsto| | pontos percentuais de CPU (p.p.) | erro |

## model_metrics.csv

Métricas da validação externa agregada (validation_pooled, 720 origens) e do teste final (288 origens), com melhoria relativa e MAE interno de seleção.
Grão: modelo × configuração × modo × população × split × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `version` | Versão do experimento que gerou a previsão (V2 ou V3) | - | identificador |
| `family` | Família do modelo (referência, V2, V3 residual, V3 neural) | - | identificador |
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `population` | População agregada: main (4.023 máquinas) ou cohort (4.022 máquinas ativas com 96 CPUs) | - | identificador |
| `analysis_set` | Conjunto de origens: main (tabela A/B/C) ou d_contrast (origens do contraste D) | - | identificador |
| `split` | Partição: fold_k (validação externa), validation_pooled (cinco folds juntos), test_final (teste de 24 h já inspecionado na V2) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `n` | Número de origens ou observações usadas no cálculo | contagem | métrica |
| `mae_pp` | Erro absoluto médio | pontos percentuais de CPU (p.p.) | métrica |
| `rmse_pp` | Raiz do erro quadrático médio | pontos percentuais de CPU (p.p.) | métrica |
| `bias_obs_minus_pred_pp` | Viés = média de (observado − previsto); positivo indica subestimação média | pontos percentuais de CPU (p.p.) | métrica |
| `error_std_pp` | Desvio-padrão amostral do erro assinado | pontos percentuais de CPU (p.p.) | métrica |
| `median_abs_error_pp` | Mediana do erro absoluto | pontos percentuais de CPU (p.p.) | métrica |
| `p90_abs_error_pp` | Percentil 90 do erro absoluto | pontos percentuais de CPU (p.p.) | métrica |
| `p95_abs_error_pp` | Percentil 95 do erro absoluto | pontos percentuais de CPU (p.p.) | métrica |
| `p99_abs_error_pp` | Percentil 99 do erro absoluto | pontos percentuais de CPU (p.p.) | métrica |
| `max_abs_error_pp` | Maior erro absoluto | pontos percentuais de CPU (p.p.) | métrica |
| `underestimation_fraction` | Fração de origens com observado > previsto | fração | métrica |
| `persistence_mae_same_origins_pp` | MAE da persistência nas mesmas origens | pontos percentuais de CPU (p.p.) | métrica |
| `mase_vs_persistence` | MASE do projeto = MAE do modelo / MAE da persistência nas mesmas origens (não é a MASE escalada pelo treino) | adimensional | métrica |
| `skill_score_rmse` | Skill score = 1 − RMSE/RMSE da persistência nas mesmas origens | adimensional | métrica |
| `improvement_vs_persistence_pct` | Melhoria de MAE contra a persistência = 100(1 − MAE/MAE_pers) | % | métrica |
| `xgboost_v2_mae_same_origins_pp` | MAE do XGBoost da V2 da mesma configuração nas mesmas origens | pontos percentuais de CPU (p.p.) | métrica |
| `improvement_vs_xgboost_v2_pct` | Melhoria de MAE contra o XGBoost da V2 da mesma configuração = 100(1 − MAE/MAE_xgb) | % | métrica |
| `mae_fold_mean_pp` | Média do MAE entre os cinco folds | pontos percentuais de CPU (p.p.) | métrica |
| `mae_fold_std_pp` | Desvio-padrão do MAE entre os cinco folds | pontos percentuais de CPU (p.p.) | métrica |
| `rmse_fold_mean_pp` | Média do RMSE entre os cinco folds | pontos percentuais de CPU (p.p.) | métrica |
| `rmse_fold_std_pp` | Desvio-padrão do RMSE entre os cinco folds | pontos percentuais de CPU (p.p.) | métrica |
| `n_folds` | Número de folds | contagem | métrica |
| `inner_selection_mae_pp` | MAE da validação interna usado na seleção da V3 (copiado de metrics.csv, não recalculado) | pontos percentuais de CPU (p.p.) | registro |

## model_metrics_by_fold.csv

Métricas em cada fold de validação externa.
Grão: modelo × configuração × modo × população × fold × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `version` | Versão do experimento que gerou a previsão (V2 ou V3) | - | identificador |
| `family` | Família do modelo (referência, V2, V3 residual, V3 neural) | - | identificador |
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `population` | População agregada: main (4.023 máquinas) ou cohort (4.022 máquinas ativas com 96 CPUs) | - | identificador |
| `analysis_set` | Conjunto de origens: main (tabela A/B/C) ou d_contrast (origens do contraste D) | - | identificador |
| `split` | Partição: fold_k (validação externa), validation_pooled (cinco folds juntos), test_final (teste de 24 h já inspecionado na V2) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `n` | Número de origens ou observações usadas no cálculo | contagem | métrica |
| `mae_pp` | Erro absoluto médio | pontos percentuais de CPU (p.p.) | métrica |
| `rmse_pp` | Raiz do erro quadrático médio | pontos percentuais de CPU (p.p.) | métrica |
| `bias_obs_minus_pred_pp` | Viés = média de (observado − previsto); positivo indica subestimação média | pontos percentuais de CPU (p.p.) | métrica |
| `error_std_pp` | Desvio-padrão amostral do erro assinado | pontos percentuais de CPU (p.p.) | métrica |
| `median_abs_error_pp` | Mediana do erro absoluto | pontos percentuais de CPU (p.p.) | métrica |
| `p90_abs_error_pp` | Percentil 90 do erro absoluto | pontos percentuais de CPU (p.p.) | métrica |
| `p95_abs_error_pp` | Percentil 95 do erro absoluto | pontos percentuais de CPU (p.p.) | métrica |
| `p99_abs_error_pp` | Percentil 99 do erro absoluto | pontos percentuais de CPU (p.p.) | métrica |
| `max_abs_error_pp` | Maior erro absoluto | pontos percentuais de CPU (p.p.) | métrica |
| `underestimation_fraction` | Fração de origens com observado > previsto | fração | métrica |
| `persistence_mae_same_origins_pp` | MAE da persistência nas mesmas origens | pontos percentuais de CPU (p.p.) | métrica |
| `mase_vs_persistence` | MASE do projeto = MAE do modelo / MAE da persistência nas mesmas origens (não é a MASE escalada pelo treino) | adimensional | métrica |
| `skill_score_rmse` | Skill score = 1 − RMSE/RMSE da persistência nas mesmas origens | adimensional | métrica |
| `improvement_vs_persistence_pct` | Melhoria de MAE contra a persistência = 100(1 − MAE/MAE_pers) | % | métrica |
| `xgboost_v2_mae_same_origins_pp` | MAE do XGBoost da V2 da mesma configuração nas mesmas origens | pontos percentuais de CPU (p.p.) | métrica |
| `improvement_vs_xgboost_v2_pct` | Melhoria de MAE contra o XGBoost da V2 da mesma configuração = 100(1 − MAE/MAE_xgb) | % | métrica |

## model_rankings.csv

Posição por MAE e por RMSE dentro de cada configuração e horizonte, com a persistência como referência.
Grão: split × configuração × métrica × horizonte × posição.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `split` | Partição: fold_k (validação externa), validation_pooled (cinco folds juntos), test_final (teste de 24 h já inspecionado na V2) | - | identificador |
| `config_scope` | Configuração cujos modelos são ordenados (a persistência entra em todas) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `metric` | Métrica usada na ordenação | - | identificador |
| `rank` | Posição (1 = menor valor) | posição | métrica |
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `value_pp` | Valor da métrica ou da estatística | pontos percentuais de CPU (p.p.) | métrica |
| `n_models` | Número de modelos no escopo | contagem | métrica |

## fold_wins.csv

Número de folds em que cada modelo teve MAE menor que o XGBoost da V2.
Grão: modelo × configuração × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `reference_model` | Modelo de referência | - | identificador |
| `reference_config` | Configuração de referência | - | identificador |
| `n_folds` | Número de folds | contagem | métrica |
| `folds_better_than_reference` | Folds em que o MAE do modelo foi menor que o da referência | contagem | métrica |

## errors_by_origin.csv

Erro por origem no teste com CPU na origem, variações, regime e faixa de CPU.
Grão: previsão de teste.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `version` | Versão do experimento que gerou a previsão (V2 ou V3) | - | identificador |
| `family` | Família do modelo (referência, V2, V3 residual, V3 neural) | - | identificador |
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `population` | População agregada: main (4.023 máquinas) ou cohort (4.022 máquinas ativas com 96 CPUs) | - | identificador |
| `analysis_set` | Conjunto de origens: main (tabela A/B/C) ou d_contrast (origens do contraste D) | - | identificador |
| `split` | Partição: fold_k (validação externa), validation_pooled (cinco folds juntos), test_final (teste de 24 h já inspecionado na V2) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `origin_index` | Índice do intervalo da origem (0 = início do trace) | intervalo de 300 s | identificador |
| `origin_time_s` | Tempo da origem desde o início do trace (não é horário civil) | s | identificador |
| `target_index` | Índice do instante previsto (origem + horizonte) | intervalo de 300 s | identificador |
| `target_time_s` | Tempo do instante previsto desde o início do trace (não é horário civil) | s | identificador |
| `target_time_h` | Tempo do instante previsto desde o início do trace (não é horário civil) | h | identificador |
| `y_obs_cpu_pct` | CPU agregada observada no instante previsto | % de CPU | observado |
| `y_pred_cpu_pct` | CPU agregada prevista para o instante previsto | % de CPU | previsto |
| `error_obs_minus_pred_pp` | Erro assinado = observado − previsto; positivo indica subestimação | pontos percentuais de CPU (p.p.) | erro |
| `abs_error_pp` | Erro absoluto |observado − previsto| | pontos percentuais de CPU (p.p.) | erro |
| `cpu_at_origin_pct` | CPU agregada observada na origem | % de CPU | observado |
| `recent_change_1_step_pp` | Variação conhecida na origem: cpu(t) − cpu(t−1) | pontos percentuais de CPU (p.p.) | observado |
| `recent_change_6_steps_pp` | Variação conhecida na origem: cpu(t) − cpu(t−6) | pontos percentuais de CPU (p.p.) | observado |
| `realized_change_pp` | Variação realizada até o alvo: cpu(t+h) − cpu(t) | pontos percentuais de CPU (p.p.) | observado |
| `regime_sigma_pp` | Desvio-padrão das variações em h passos no desenvolvimento, usado como limiar de regime | pontos percentuais de CPU (p.p.) | parâmetro |
| `regime` | Regime da origem: subida (variação ≥ σ_h), queda (≤ −σ_h) ou estável | - | identificador |
| `cpu_band` | Faixa de CPU observada no alvo, pelos quintis do desenvolvimento | - | identificador |
| `recent_change_1_step_bin` | Faixa da variação de 1 passo conhecida na origem | - | identificador |
| `realized_change_bin` | Faixa da variação realizada até o alvo | - | identificador |

## errors_by_regime.csv

MAE, RMSE e viés por regime, faixa de CPU e faixa de variação.
Grão: modelo × horizonte × agrupamento × grupo.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `grouping` | Critério de agrupamento | - | identificador |
| `group` | Grupo dentro do critério (ou grupo de atributos na importância) | - | identificador |
| `n` | Número de origens ou observações usadas no cálculo | contagem | métrica |
| `mae_pp` | Erro absoluto médio | pontos percentuais de CPU (p.p.) | métrica |
| `bias_obs_minus_pred_pp` | Viés = média de (observado − previsto); positivo indica subestimação média | pontos percentuais de CPU (p.p.) | métrica |
| `rmse_pp` | Raiz do erro quadrático médio | pontos percentuais de CPU (p.p.) | métrica |

## error_distribution.csv

Quantis do erro assinado e absoluto (teste).
Grão: modelo × horizonte × tipo de erro × estatística.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `population` | População agregada: main (4.023 máquinas) ou cohort (4.022 máquinas ativas com 96 CPUs) | - | identificador |
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `split` | Partição: fold_k (validação externa), validation_pooled (cinco folds juntos), test_final (teste de 24 h já inspecionado na V2) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `error_kind` | Tipo de erro: assinado (observado − previsto) ou absoluto | - | identificador |
| `statistic` | Estatística (pNN = percentil NN) | - | identificador |
| `value_pp` | Valor da métrica ou da estatística | pontos percentuais de CPU (p.p.) | métrica |

## residual_autocorrelation.csv

Autocorrelação dos resíduos de teste e Ljung–Box.
Grão: modelo × horizonte × defasagem.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `lag_steps` | Defasagem | passos de 300 s | identificador |
| `lag_min` | Defasagem | min | identificador |
| `acf` | Autocorrelação amostral dos resíduos de teste | adimensional | métrica |
| `band_95` | Banda aproximada de 95% para ruído branco (1,96/√n) | adimensional | métrica |
| `n` | Número de origens ou observações usadas no cálculo | contagem | métrica |
| `ljung_box_q_6` | Estatística Q de Ljung–Box até 6 defasagens | adimensional | inferência |
| `ljung_box_p_6` | p-valor de Ljung–Box até 6 defasagens (resíduos de h passos são autocorrelacionados até h−1 por construção) | adimensional | inferência |
| `ljung_box_q_12` | Estatística Q de Ljung–Box até 12 defasagens | adimensional | inferência |
| `ljung_box_p_12` | p-valor de Ljung–Box até 12 defasagens | adimensional | inferência |
| `ljung_box_q_24` | Estatística Q de Ljung–Box até 24 defasagens | adimensional | inferência |
| `ljung_box_p_24` | p-valor de Ljung–Box até 24 defasagens | adimensional | inferência |

## diagnostic_windows.csv

Janelas de diagnóstico escolhidas por critério objetivo sobre a série observada.
Grão: janela.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `window` | Tipo de janela de diagnóstico | - | identificador |
| `rule` | Regra de escolha registrada | - | parâmetro |
| `criterion_value_pp` | Valor do critério na janela escolhida | pontos percentuais de CPU (p.p.) | observado |
| `start_index` | Primeiro intervalo | intervalo de 300 s | identificador |
| `end_index` | Último intervalo | intervalo de 300 s | identificador |
| `start_time_h` | Início desde o começo do trace (não é horário civil) | h | identificador |
| `end_time_h` | Fim desde o começo do trace (não é horário civil) | h | identificador |
| `length_steps` | Duração da janela | passos de 300 s | parâmetro |
| `selection_order` | Ordem de escolha (as janelas seguintes não se sobrepõem às anteriores) | posição | identificador |

## peak_thresholds.csv

Limiares de pico (desenvolvimento e amostra de teste).
Grão: limiar.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `threshold_kind` | Limiar: quantil do desenvolvimento (primário) ou da amostra de teste | - | identificador |
| `source` | Origem do limiar | - | identificador |
| `quantile` | Nível do quantil | fração | parâmetro |
| `threshold_cpu_pct` | Limiar de pico de CPU | % de CPU | parâmetro |
| `n_reference` | Número de valores usados para obter o limiar | contagem | métrica |

## peak_analysis.csv

Erro e subestimação condicionados aos picos.
Grão: modelo × horizonte × limiar.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `threshold_kind` | Limiar: quantil do desenvolvimento (primário) ou da amostra de teste | - | identificador |
| `threshold_cpu_pct` | Limiar de pico de CPU | % de CPU | parâmetro |
| `n_total` | Número de origens de teste do grupo | contagem | métrica |
| `n_peak_points` | Instantes previstos com CPU observada ≥ limiar | contagem | métrica |
| `peak_mae_pp` | MAE nos instantes de pico | pontos percentuais de CPU (p.p.) | métrica |
| `peak_bias_obs_minus_pred_pp` | Viés (observado − previsto) nos picos; positivo = subestimação | pontos percentuais de CPU (p.p.) | métrica |
| `peak_underestimation_frequency` | Fração dos picos em que o modelo subestimou | fração | métrica |
| `peak_mean_underestimation_pp` | Magnitude média da subestimação nos picos subestimados | pontos percentuais de CPU (p.p.) | métrica |
| `peak_max_underestimation_pp` | Maior subestimação nos picos | pontos percentuais de CPU (p.p.) | métrica |
| `nonpeak_mae_pp` | MAE fora dos picos | pontos percentuais de CPU (p.p.) | métrica |
| `overall_max_underestimation_pp` | Maior subestimação em todo o teste | pontos percentuais de CPU (p.p.) | métrica |

## peak_alerts.csv

Antecipação de eventos e falsos alarmes por sinal.
Grão: sinal × horizonte × limiar.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `signal` | Sinal de alerta: previsão pontual de um modelo ou quantil do XGBoost quantílico | - | identificador |
| `signal_kind` | Tipo de sinal | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `threshold_kind` | Limiar: quantil do desenvolvimento (primário) ou da amostra de teste | - | identificador |
| `threshold_cpu_pct` | Limiar de pico de CPU | % de CPU | parâmetro |
| `low_index` | Primeiro instante da faixa avaliada (comum a todos os horizontes) | intervalo de 300 s | parâmetro |
| `high_index` | Último instante da faixa avaliada | intervalo de 300 s | parâmetro |
| `n_events` | Eventos de pico na faixa | contagem | métrica |
| `n_events_evaluable` | Eventos com início avaliável | contagem | métrica |
| `n_anticipated` | Eventos antecipados (alerta a até 1 passo do início, emitido antes do início) | contagem | métrica |
| `anticipation_rate` | Eventos antecipados / eventos avaliáveis | fração | métrica |
| `n_alert_points` | Instantes com alerta | contagem | métrica |
| `n_alert_episodes` | Episódios de alertas consecutivos | contagem | métrica |
| `n_false_alarm_episodes` | Episódios sem evento a até 1 passo | contagem | métrica |
| `false_alarm_episode_fraction` | Episódios de falso alarme / episódios de alerta | fração | métrica |
| `mean_lead_steps` | Antecedência média dos alertas que anteciparam eventos | passos de 300 s | métrica |

## peak_events.csv

Eventos de pico no teste.
Grão: evento × limiar.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `event_id` | Identificador do evento de pico | - | identificador |
| `start_index` | Primeiro intervalo | intervalo de 300 s | identificador |
| `end_index` | Último intervalo | intervalo de 300 s | identificador |
| `duration_steps` | Duração do evento | passos de 300 s | observado |
| `peak_cpu_pct` | Maior CPU observada no evento | % de CPU | observado |
| `cpu_before_start_pct` | CPU observada no instante anterior ao início | % de CPU | observado |
| `onset_observed` | Verdadeiro se o instante anterior estava abaixo do limiar (início avaliável) | booleano | observado |
| `threshold_cpu_pct` | Limiar de pico de CPU | % de CPU | parâmetro |
| `threshold_kind` | Limiar: quantil do desenvolvimento (primário) ou da amostra de teste | - | identificador |

## bootstrap_intervals.csv

Block bootstrap de todas as comparações (pré-registradas e exploratórias) nos três blocos.
Grão: comparação × bloco.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `analysis_type` | pre_registrada (comparações da V3) ou exploratoria (todos os pares em A) | - | identificador |
| `family` | Família do modelo (referência, V2, V3 residual, V3 neural) | - | identificador |
| `comparison` | Nome da comparação | - | identificador |
| `loss` | Perda comparada: erro absoluto ou pinball | - | identificador |
| `quantile` | Nível do quantil | fração | parâmetro |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `model_a` | Modelo A (diferença = perda A − perda B) | - | identificador |
| `config_a` | Configuração do modelo A | - | identificador |
| `model_b` | Modelo B (referência da diferença) | - | identificador |
| `config_b` | Configuração do modelo B | - | identificador |
| `block_steps` | Tamanho do bloco do bootstrap (6 = 30 min, 12 = 1 h, 24 = 2 h) | passos de 300 s | parâmetro |
| `primary_block` | Verdadeiro para o bloco primário de 1 h | booleano | parâmetro |
| `n_boot` | Reamostragens do bootstrap | contagem | parâmetro |
| `n` | Número de origens ou observações usadas no cálculo | contagem | métrica |
| `mean_diff` | Média da diferença de perda A − B; negativa favorece A | p.p. de CPU (erro absoluto) ou unidade do pinball | inferência |
| `ci_low` | Limite inferior do intervalo de 95% (percentil 2,5) | unidade de mean_diff | inferência |
| `ci_high` | Limite superior do intervalo de 95% (percentil 97,5) | unidade de mean_diff | inferência |
| `significant` | Intervalo de 95% não contém zero | booleano | inferência |
| `p_value_bootstrap` | p-valor bicaudal das mesmas reamostragens; mínimo possível ≈ 2/(B+1) ≈ 4×10⁻⁴ | adimensional | inferência |
| `loss_a_mean` | Perda média do modelo A | unidade da perda | métrica |
| `loss_b_mean` | Perda média do modelo B | unidade da perda | métrica |
| `stored_mean_diff` | mean_diff gravado em bootstrap_ci.csv da V3 | unidade de mean_diff | registro |
| `stored_ci_low` | ci_low gravado em bootstrap_ci.csv da V3 | unidade de mean_diff | registro |
| `stored_ci_high` | ci_high gravado em bootstrap_ci.csv da V3 | unidade de mean_diff | registro |
| `stored_significant` | Decisão gravada em bootstrap_ci.csv da V3 | booleano | registro |
| `reproduction_max_abs_diff` | Maior diferença entre o recálculo e o valor gravado | unidade de mean_diff | métrica |
| `relative_diff` | mean_diff / perda média de B | fração | inferência |
| `practical_relevance` | relevante se |relative_diff| ≥ 5% (convenção desta análise), senão pequena | - | inferência |
| `watts_per_substitute_server_equivalent` | mean_diff × (P_busy − P_idle)/100: equivalente linear em potência de um servidor substituto | W (potência modelada) | modelado |
| `holm_family` | Família da correção de Holm | - | identificador |
| `p_holm` | p-valor ajustado por Holm na família (só bloco primário) | adimensional | inferência |
| `holm_significant` | p_holm < 0,05 | booleano | inferência |
| `significant_across_blocks` | Concordância da significância nos blocos de 30 min, 1 h e 2 h: todos, parcial ou nenhum | - | inferência |

## statistical_comparisons.csv

Uma linha por comparação no bloco primário de 1 h, com p-valor, Holm, relevância prática e concordância entre blocos.
Grão: comparação.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `analysis_type` | pre_registrada (comparações da V3) ou exploratoria (todos os pares em A) | - | identificador |
| `family` | Família do modelo (referência, V2, V3 residual, V3 neural) | - | identificador |
| `comparison` | Nome da comparação | - | identificador |
| `loss` | Perda comparada: erro absoluto ou pinball | - | identificador |
| `quantile` | Nível do quantil | fração | parâmetro |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `model_a` | Modelo A (diferença = perda A − perda B) | - | identificador |
| `config_a` | Configuração do modelo A | - | identificador |
| `model_b` | Modelo B (referência da diferença) | - | identificador |
| `config_b` | Configuração do modelo B | - | identificador |
| `n` | Número de origens ou observações usadas no cálculo | contagem | métrica |
| `loss_a_mean` | Perda média do modelo A | unidade da perda | métrica |
| `loss_b_mean` | Perda média do modelo B | unidade da perda | métrica |
| `mean_diff` | Média da diferença de perda A − B; negativa favorece A | p.p. de CPU (erro absoluto) ou unidade do pinball | inferência |
| `relative_diff` | mean_diff / perda média de B | fração | inferência |
| `ci_low` | Limite inferior do intervalo de 95% (percentil 2,5) | unidade de mean_diff | inferência |
| `ci_high` | Limite superior do intervalo de 95% (percentil 97,5) | unidade de mean_diff | inferência |
| `significant` | Intervalo de 95% não contém zero | booleano | inferência |
| `p_value_bootstrap` | p-valor bicaudal das mesmas reamostragens; mínimo possível ≈ 2/(B+1) ≈ 4×10⁻⁴ | adimensional | inferência |
| `p_holm` | p-valor ajustado por Holm na família (só bloco primário) | adimensional | inferência |
| `holm_significant` | p_holm < 0,05 | booleano | inferência |
| `significant_across_blocks` | Concordância da significância nos blocos de 30 min, 1 h e 2 h: todos, parcial ou nenhum | - | inferência |
| `practical_relevance` | relevante se |relative_diff| ≥ 5% (convenção desta análise), senão pequena | - | inferência |
| `watts_per_substitute_server_equivalent` | mean_diff × (P_busy − P_idle)/100: equivalente linear em potência de um servidor substituto | W (potência modelada) | modelado |

## neural_training_history.csv

Loss por época das redes (seleção, reajustes de fold e de teste).
Grão: rede × configuração × modo × horizonte × fase × grade × fold × época.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `horizon` | Horizonte em passos, 'path' para a trajetória de seis passos ou 'todos' | - | identificador |
| `phase` | Fase do treino: selection (grade com validação interna), outer_refit (fold externo) ou test_refit (teste) | - | identificador |
| `grid_index` | Índice do ponto da grade de hiperparâmetros | - | identificador |
| `fold` | Fold (vazio no reajuste de teste) | - | identificador |
| `epoch` | Época de treino | época | registro |
| `train_loss_mse_standardized` | Loss de treino: MSE do alvo padronizado pelo treino | adimensional | registro |
| `val_loss_mse_standardized` | Loss de validação interna (vazia nos reajustes, que não têm validação) | adimensional | registro |
| `selected_grid_index` | Ponto da grade escolhido na seleção | - | registro |
| `is_selected_grid` | Verdadeiro se a linha pertence ao ponto escolhido | booleano | identificador |
| `has_validation_loss` | Verdadeiro se há loss de validação na linha | booleano | identificador |

## neural_fit_diagnostics.csv

Indicadores de ajuste por fold no ponto selecionado da grade.
Grão: rede × configuração × modo × horizonte × fold.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `horizon` | Horizonte em passos, 'path' para a trajetória de seis passos ou 'todos' | - | identificador |
| `fold` | Fold (vazio no reajuste de teste) | - | identificador |
| `epochs_run` | Épocas executadas até a parada antecipada | época | registro |
| `best_epoch` | Época de menor loss de validação | época | métrica |
| `train_loss_at_best` | Loss de treino na melhor época | adimensional | métrica |
| `val_loss_at_best` | Loss de validação na melhor época | adimensional | métrica |
| `val_over_train_at_best` | Razão validação/treino na melhor época | adimensional | métrica |
| `train_loss_last` | Loss de treino na última época | adimensional | métrica |
| `val_loss_last` | Loss de validação na última época | adimensional | métrica |
| `val_increase_after_best` | Aumento relativo da loss de validação entre a melhor e a última época | fração | métrica |
| `train_loss_first` | Loss de treino na primeira época | adimensional | métrica |
| `val_loss_first` | Loss de validação na primeira época | adimensional | métrica |
| `train_reduction_fraction` | Redução relativa da loss de treino da primeira à melhor época | fração | métrica |

## neural_fit_summary.csv

Média entre folds dos indicadores de ajuste.
Grão: rede × configuração × modo × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `horizon` | Horizonte em passos, 'path' para a trajetória de seis passos ou 'todos' | - | identificador |
| `n_folds` | Número de folds | contagem | métrica |
| `best_epoch_mean` | Média da melhor época entre folds | época | métrica |
| `best_epoch_min` | Menor melhor época entre folds | época | métrica |
| `epochs_run_mean` | Média das épocas executadas | época | métrica |
| `val_over_train_at_best_mean` | Média da razão validação/treino na melhor época | adimensional | métrica |
| `val_increase_after_best_mean` | Média do aumento da loss de validação depois da melhor época | fração | métrica |
| `train_loss_at_best_mean` | Média da loss de treino na melhor época | adimensional | métrica |
| `val_loss_at_best_mean` | Média da loss de validação na melhor época | adimensional | métrica |
| `early_best_signal` | Sinal descritivo: melhor época média ≤ 5 | booleano | métrica |
| `generalization_gap_signal` | Sinal descritivo: razão validação/treino média ≥ 1,5 | booleano | métrica |
| `note` | Observação de interpretação | - | identificador |

## generalization.csv

MAE de seleção, validação externa e teste, e razão de generalização normalizada pela persistência.
Grão: modelo × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `selection_basis` | Base de seleção: validação interna (V3) ou média dos folds externos (V2) | - | identificador |
| `selection_mae_pp` | MAE na base de seleção | pontos percentuais de CPU (p.p.) | métrica |
| `validation_pooled_mae_pp` | MAE da validação externa agregada | pontos percentuais de CPU (p.p.) | métrica |
| `validation_fold_std_pp` | Desvio do MAE entre folds externos | pontos percentuais de CPU (p.p.) | métrica |
| `test_mae_pp` | MAE no teste final | pontos percentuais de CPU (p.p.) | métrica |
| `test_over_selection` | MAE de teste / MAE de seleção | adimensional | métrica |
| `persistence_test_over_selection` | Mesma razão para a persistência, na mesma base de seleção | adimensional | métrica |
| `normalized_generalization_ratio` | test_over_selection / persistence_test_over_selection; > 1 indica perda maior que a da persistência ao passar para o teste | adimensional | métrica |

## neural_window_selection.csv

MAE interno do DLinear para cada janela candidata.
Grão: janela × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `window_steps` | Janela de entrada da rede | passos de 300 s | parâmetro |
| `window_hours` | Janela de entrada da rede | h | parâmetro |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `inner_mae_pp` | MAE interno do DLinear (A) com essa janela | pontos percentuais de CPU (p.p.) | registro |
| `inner_mae_sum_pp` | Soma do MAE interno nos três horizontes (regra de escolha) | pontos percentuais de CPU (p.p.) | registro |
| `chosen` | Janela escolhida | booleano | registro |
| `rule` | Regra de escolha registrada | - | parâmetro |

## feature_importance.csv

Média do |SHAP| no teste (TreeSHAP do XGBoost) por atributo.
Grão: modelo × configuração × horizonte × atributo.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `feature` | Atributo de entrada | - | identificador |
| `mean_abs_shap_pp` | Média do |SHAP| no teste | p.p. da grandeza explicada | métrica |
| `mean_shap_pp` | Média do SHAP assinado no teste | p.p. da grandeza explicada | métrica |
| `group` | Grupo dentro do critério (ou grupo de atributos na importância) | - | identificador |
| `share` | Fração da soma das médias de |SHAP| | fração | métrica |
| `rank` | Posição (1 = menor valor) | posição | métrica |
| `explained_quantity` | Grandeza que o XGBoost prevê e o SHAP decompõe | - | identificador |
| `bias_term_pp` | Termo constante do TreeSHAP (média no teste) | p.p. da grandeza explicada | métrica |
| `additivity_max_abs_diff` | Maior diferença entre a soma das contribuições e a margem do modelo | p.p. da grandeza explicada | métrica |

## model_complexity.csv

Tamanho de cada modelo.
Grão: modelo × configuração × modo × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `horizon` | Horizonte em passos, 'path' para a trajetória de seis passos ou 'todos' | - | identificador |
| `complexity_kind` | Medida de tamanho usada | - | identificador |
| `n_trainable_parameters` | Parâmetros treináveis (redes), coeficientes (Ridge) ou parâmetros do modelo de estados | contagem | registro |
| `n_trees` | Árvores do XGBoost | contagem | métrica |
| `n_leaves` | Folhas do XGBoost | contagem | métrica |
| `n_coefficients` | Coeficientes do Ridge com intercepto | contagem | métrica |
| `final_epochs` | Épocas do reajuste de teste | época | registro |

## computational_performance.csv

Tempo de treino, de inferência e memória CUDA, com a origem da medida.
Grão: modelo × configuração × modo × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `mode` | Modo de previsão: direct (um modelo por horizonte) ou path (trajetória de seis passos, secundária) | - | identificador |
| `horizon` | Horizonte em passos, 'path' para a trajetória de seis passos ou 'todos' | - | identificador |
| `measurement_source` | Origem da medida: corrida original ou reajuste medido nesta análise | - | identificador |
| `device` | Dispositivo do treino | - | identificador |
| `device_name` | Nome do processador ou da GPU | - | identificador |
| `precision` | Precisão numérica | - | identificador |
| `n_fits` | Ajustes executados para a unidade (seleção e reajustes) | contagem | registro |
| `train_seconds_total_with_selection` | Tempo total de treino incluindo a seleção (só redes, corrida original) | s | registro |
| `train_seconds_test_refit` | Tempo do reajuste de teste | s | registro |
| `inference_seconds_test` | Tempo de inferência no teste | s | registro |
| `peak_cuda_memory_mib` | Pico de memória CUDA alocada por tensores | MiB | registro |

## cost_tradeoff.csv

Custo do reajuste de teste contra MAE de teste (A) e fronteira de Pareto.
Grão: modelo × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `test_mae_pp` | MAE no teste final | pontos percentuais de CPU (p.p.) | métrica |
| `size_measure` | Tamanho do modelo (parâmetros, coeficientes ou folhas) | contagem | métrica |
| `size_kind` | Tipo de medida de tamanho | - | identificador |
| `train_seconds_test_refit` | Tempo do reajuste de teste | s | registro |
| `device` | Dispositivo do treino | - | identificador |
| `peak_cuda_memory_mib` | Pico de memória CUDA alocada por tensores | MiB | registro |
| `measurement_source` | Origem da medida: corrida original ou reajuste medido nesta análise | - | identificador |
| `pareto_time_mae` | Na fronteira de Pareto tempo do reajuste × MAE de teste | booleano | métrica |

## ablation_results.csv

Diferença de MAE de B, C e D contra a referência A nas mesmas origens.
Grão: população × modelo × configuração × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `population` | População agregada: main (4.023 máquinas) ou cohort (4.022 máquinas ativas com 96 CPUs) | - | identificador |
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `reference_config` | Configuração de referência | - | identificador |
| `mae_pp` | Erro absoluto médio | pontos percentuais de CPU (p.p.) | métrica |
| `reference_mae_pp` | MAE da configuração de referência | pontos percentuais de CPU (p.p.) | métrica |
| `delta_mae_pp` | MAE da configuração − MAE da referência; positivo = piora | pontos percentuais de CPU (p.p.) | métrica |
| `delta_mae_pct` | Diferença relativa de MAE contra a referência | % | métrica |
| `rmse_pp` | Raiz do erro quadrático médio | pontos percentuais de CPU (p.p.) | métrica |
| `reference_rmse_pp` | RMSE da configuração de referência | pontos percentuais de CPU (p.p.) | métrica |
| `n` | Número de origens ou observações usadas no cálculo | contagem | métrica |
| `executed` | Verdadeiro se a combinação foi executada na corrida | booleano | identificador |
| `ci95_low_pp` | Limite inferior do IC 95% pré-registrado (bloco de 1 h) | pontos percentuais de CPU (p.p.) | inferência |
| `ci95_high_pp` | Limite superior do IC 95% pré-registrado (bloco de 1 h) | pontos percentuais de CPU (p.p.) | inferência |
| `significant_block_1h` | IC 95% pré-registrado sem zero | booleano | inferência |
| `p_holm` | p-valor ajustado por Holm na família (só bloco primário) | adimensional | inferência |

## ablation_grid.csv

Grade completa modelo × configuração × horizonte, com combinações não executadas marcadas.
Grão: modelo × configuração × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `delta_mae_pp` | MAE da configuração − MAE da referência; positivo = piora | pontos percentuais de CPU (p.p.) | métrica |
| `delta_mae_pct` | Diferença relativa de MAE contra a referência | % | métrica |
| `executed` | Verdadeiro se a combinação foi executada na corrida | booleano | identificador |

## quantile_predictions.csv

Quantis Q50/Q90/Q95 e tetos ajustados por origem.
Grão: configuração × horizonte × split × origem.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `population` | População agregada: main (4.023 máquinas) ou cohort (4.022 máquinas ativas com 96 CPUs) | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `split` | Partição: fold_k (validação externa), validation_pooled (cinco folds juntos), test_final (teste de 24 h já inspecionado na V2) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `origin_index` | Índice do intervalo da origem (0 = início do trace) | intervalo de 300 s | identificador |
| `origin_time_s` | Tempo da origem desde o início do trace (não é horário civil) | s | identificador |
| `target_index` | Índice do instante previsto (origem + horizonte) | intervalo de 300 s | identificador |
| `target_time_s` | Tempo do instante previsto desde o início do trace (não é horário civil) | s | identificador |
| `target_time_h` | Tempo do instante previsto desde o início do trace (não é horário civil) | h | identificador |
| `y_obs_cpu_pct` | CPU agregada observada no instante previsto | % de CPU | observado |
| `q50_cpu_pct` | Quantil 0,5 previsto | % de CPU | previsto |
| `q90_cpu_pct` | Quantil 0,9 previsto | % de CPU | previsto |
| `q95_cpu_pct` | Quantil 0,95 previsto | % de CPU | previsto |
| `q90_adjusted_cpu_pct` | Teto Q90 com ajuste conformal (só teste) | % de CPU | previsto |
| `q95_adjusted_cpu_pct` | Teto Q95 com ajuste conformal (só teste) | % de CPU | previsto |
| `upper_margin_q95_minus_q50_pp` | Margem unilateral q95 − q50 (não é largura de intervalo bilateral) | pontos percentuais de CPU (p.p.) | previsto |

## quantile_calibration.csv

Cobertura, ultrapassagem, pinball e cobertura nos picos dos quantis.
Grão: split × configuração × horizonte × quantil × variante.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `split` | Partição: fold_k (validação externa), validation_pooled (cinco folds juntos), test_final (teste de 24 h já inspecionado na V2) | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `quantile` | Nível do quantil | fração | parâmetro |
| `variant` | bruto ou ajustado (conformal) | - | identificador |
| `peak_threshold_cpu_pct` | Limiar de pico usado (P90 do desenvolvimento) | % de CPU | parâmetro |
| `n` | Número de origens ou observações usadas no cálculo | contagem | métrica |
| `nominal_coverage` | Cobertura nominal P(y ≤ q) | fração | parâmetro |
| `empirical_coverage` | Cobertura empírica P(y ≤ q) (sem garantia teórica sob dependência temporal) | fração | métrica |
| `coverage_gap` | Cobertura empírica − nominal | fração | métrica |
| `exceedance_rate` | Fração de y acima do quantil | fração | métrica |
| `pinball_pp` | Perda pinball média | pontos percentuais de CPU (p.p.) | métrica |
| `mean_exceedance_pp` | Média de y − q quando y > q | pontos percentuais de CPU (p.p.) | métrica |
| `max_exceedance_pp` | Maior y − q (negativo se nunca ultrapassou) | pontos percentuais de CPU (p.p.) | métrica |
| `mean_upper_margin_above_q50_pp` | Média de q − q50 (margem unilateral) | pontos percentuais de CPU (p.p.) | métrica |
| `n_peak` | Instantes de pico | contagem | métrica |
| `peak_coverage` | Cobertura empírica nos picos | fração | métrica |
| `peak_mean_exceedance_pp` | Média da ultrapassagem nos picos ultrapassados | pontos percentuais de CPU (p.p.) | métrica |

## energy_timeseries.csv

Potência modelada de referência e projetada em cada passo da trajetória de teste.
Grão: modelo × r × origem × passo.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `r` | Expoente do modelo de Fan | adimensional | parâmetro |
| `origin_index` | Índice do intervalo da origem (0 = início do trace) | intervalo de 300 s | identificador |
| `step` | Passo da trajetória (1 a 6) | passos de 300 s | identificador |
| `target_index` | Índice do instante previsto (origem + horizonte) | intervalo de 300 s | identificador |
| `target_time_h` | Tempo do instante previsto desde o início do trace (não é horário civil) | h | identificador |
| `cpu_obs_pct` | CPU observada no passo | % de CPU | observado |
| `cpu_pred_pct` | CPU prevista no passo | % de CPU | previsto |
| `cpu_pred_minus_obs_pp` | Previsto − observado (convenção da seção de energia) | pontos percentuais de CPU (p.p.) | erro |
| `power_reference_w` | Potência modelada com a CPU observada (servidor substituto) | W (potência modelada) | modelado |
| `power_projected_w` | Potência modelada com a CPU prevista (servidor substituto) | W (potência modelada) | modelado |
| `power_projected_minus_reference_w` | Projetada − referência; negativa = potência subestimada | W (potência modelada) | modelado |

## energy_origin_errors.csv

Energia modelada de referência e projetada por origem e horizonte.
Grão: modelo × r × horizonte × origem.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `r` | Expoente do modelo de Fan | adimensional | parâmetro |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `origin_index` | Índice do intervalo da origem (0 = início do trace) | intervalo de 300 s | identificador |
| `reference_energy_wh` | Energia modelada pela trajetória observada (trapézio, extremo esquerdo na CPU observada) | Wh (energia modelada) | modelado |
| `projected_energy_wh` | Energia modelada pela trajetória prevista | Wh (energia modelada) | modelado |
| `energy_projected_minus_reference_wh` | Projetada − referência; negativa = energia subestimada | Wh (energia modelada) | modelado |
| `energy_abs_error_wh` | |projetada − referência| | Wh (energia modelada) | modelado |
| `energy_relative_error` | (projetada − referência)/referência | fração | modelado |
| `mean_cpu_pred_minus_obs_pp` | Média de previsto − observado nos passos 1..h | pontos percentuais de CPU (p.p.) | erro |
| `terminal_cpu_pred_minus_obs_pp` | Previsto − observado no passo h | pontos percentuais de CPU (p.p.) | erro |
| `trajectory_reaches_peak` | Verdadeiro se a trajetória observada atinge o P90 do desenvolvimento | booleano | observado |
| `complete_trajectory` | Verdadeiro se todos os passos 1..h existem (sem preenchimento pelo valor terminal) | booleano | identificador |

## energy_metrics.csv

Métricas de energia e potência modeladas por modelo, r e horizonte.
Grão: modelo × r × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `quantity` | Grandeza: energia de TI modelada num servidor substituto, não medição | - | identificador |
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `r` | Expoente do modelo de Fan | adimensional | parâmetro |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `n_origins` | Origens com trajetória completa | contagem | métrica |
| `total_reference_wh` | Soma da energia de referência nas origens (janelas se sobrepõem quando h > 1) | Wh (energia modelada) | modelado |
| `total_projected_wh` | Soma da energia projetada nas origens (janelas se sobrepõem quando h > 1) | Wh (energia modelada) | modelado |
| `energy_mae_wh` | Média de |projetada − referência| | Wh (energia modelada) | modelado |
| `energy_bias_projected_minus_reference_wh` | Média de projetada − referência | Wh (energia modelada) | modelado |
| `energy_p95_abs_error_wh` | Percentil 95 de |projetada − referência| | Wh (energia modelada) | modelado |
| `energy_mean_abs_relative_error` | Média de |projetada − referência|/referência | fração | modelado |
| `energy_underestimation_fraction` | Fração de origens com energia projetada < referência | fração | modelado |
| `n_peak_origins` | Origens cuja trajetória observada atinge o P90 do desenvolvimento | contagem | métrica |
| `energy_mae_peak_origins_wh` | Erro absoluto médio de energia nas origens com pico | Wh (energia modelada) | modelado |
| `energy_bias_peak_origins_wh` | Viés de energia nas origens com pico | Wh (energia modelada) | modelado |
| `energy_mae_nonpeak_origins_wh` | Erro absoluto médio de energia nas demais origens | Wh (energia modelada) | modelado |
| `power_mae_at_horizon_w` | Erro absoluto médio de potência no passo h | W (potência modelada) | modelado |
| `power_bias_at_horizon_projected_minus_reference_w` | Viés de potência no passo h (projetada − referência) | W (potência modelada) | modelado |

## energy_cumulative.csv

Energia modelada acumulada no teste com janelas sem sobreposição.
Grão: modelo × r × janela × origem.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `r` | Expoente do modelo de Fan | adimensional | parâmetro |
| `window_min` | Duração de cada janela somada (5 ou 30 min, sem sobreposição) | min | parâmetro |
| `origin_index` | Índice do intervalo da origem (0 = início do trace) | intervalo de 300 s | identificador |
| `window_end_time_h` | Fim da janela desde o início do trace (não é horário civil) | h | identificador |
| `reference_energy_wh` | Energia modelada pela trajetória observada (trapézio, extremo esquerdo na CPU observada) | Wh (energia modelada) | modelado |
| `projected_energy_wh` | Energia modelada pela trajetória prevista | Wh (energia modelada) | modelado |
| `cumulative_reference_wh` | Energia de referência acumulada | Wh (energia modelada) | modelado |
| `cumulative_projected_wh` | Energia projetada acumulada | Wh (energia modelada) | modelado |
| `cumulative_projected_minus_reference_wh` | Diferença acumulada projetada − referência | Wh (energia modelada) | modelado |

## energy_sensitivity.csv

Sensibilidade do erro de energia a r, P_idle e P_busy.
Grão: parâmetro × valor × modelo.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `parameter` | Parâmetro variado na sensibilidade | - | identificador |
| `factor_or_value` | Valor de r ou fator multiplicativo de P_idle/P_busy | adimensional | parâmetro |
| `p_idle_w` | Potência ociosa usada | W | parâmetro |
| `p_busy_w` | Potência a 100% usada | W | parâmetro |
| `r` | Expoente do modelo de Fan | adimensional | parâmetro |
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `energy_mae_wh` | Média de |projetada − referência| | Wh (energia modelada) | modelado |
| `energy_bias_projected_minus_reference_wh` | Média de projetada − referência | Wh (energia modelada) | modelado |
| `energy_mean_abs_relative_error` | Média de |projetada − referência|/referência | fração | modelado |
| `total_reference_wh` | Soma da energia de referência nas origens (janelas se sobrepõem quando h > 1) | Wh (energia modelada) | modelado |
| `total_projected_wh` | Soma da energia projetada nas origens (janelas se sobrepõem quando h > 1) | Wh (energia modelada) | modelado |

## jensen_analysis.csv

P(média) contra média de P por intervalo, por servidor e para a frota de servidores substitutos.
Grão: intervalo × r.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `bin_index` | Índice do intervalo de 300 s | intervalo de 300 s | identificador |
| `time_h` | Tempo desde o início do trace (não é horário civil) | h | identificador |
| `r` | Expoente do modelo de Fan | adimensional | parâmetro |
| `n_present_machines` | Máquinas com CPU válida no intervalo | contagem | observado |
| `coverage` | Cobertura ponderada por cpu_num no intervalo | fração | observado |
| `u_mean` | Utilização média ponderada (CPU/100) | fração | observado |
| `u_variance` | Variância ponderada da utilização | fração² | observado |
| `power_of_mean_u_w_per_server` | P(média de u): potência de um servidor substituto à utilização média | W (potência modelada) | modelado |
| `mean_of_power_w_per_server` | Média de P(u_i): potência média por servidor substituto calculada por máquina | W (potência modelada) | modelado |
| `jensen_gap_w_per_server` | Média de P(u_i) − P(média de u), por servidor; negativo com curva côncava (r > 1) | W (potência modelada) | modelado |
| `fleet_sum_of_power_w` | Soma de P(u_i) nas máquinas presentes (frota hipotética de servidores substitutos) | W (potência modelada) | modelado |
| `fleet_n_times_power_of_mean_w` | n · P(média de u) | W (potência modelada) | modelado |
| `jensen_gap_fleet_w` | Soma de P(u_i) − n · P(média de u), frota hipotética | W (potência modelada) | modelado |
| `second_order_mean_of_power_w_per_server` | Aproximação de segunda ordem P(ū) + ½P''(ū)Var(u) (diagnóstico) | W (potência modelada) | modelado |
| `second_order_minus_exact_w_per_server` | Aproximação de segunda ordem − média de P(u_i) | W (potência modelada) | modelado |
| `unstable_approximation` | Intervalo em que a aproximação é instável (utilização média < 0,05 com r ≠ 1) | booleano | modelado |

## jensen_summary.csv

Resumo do efeito de Jensen por r.
Grão: r.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `r` | Expoente do modelo de Fan | adimensional | parâmetro |
| `n_bins` | Número de intervalos | contagem | métrica |
| `mean_gap_w_per_server` | Média do gap de Jensen por servidor | W (potência modelada) | modelado |
| `min_gap_w_per_server` | Menor gap por servidor | W (potência modelada) | modelado |
| `max_gap_w_per_server` | Maior gap por servidor | W (potência modelada) | modelado |
| `mean_power_of_mean_w` | Média de P(média de u) | W (potência modelada) | modelado |
| `mean_relative_gap` | Média de gap/P(média de u) | fração | modelado |
| `mean_gap_fleet_w` | Média do gap da frota hipotética | W (potência modelada) | modelado |
| `mean_fleet_sum_w` | Média da soma de P(u_i) | W (potência modelada) | modelado |
| `mean_n_present` | Média de máquinas presentes | contagem | observado |
| `mean_abs_second_order_error_w` | Média do |erro| da aproximação de segunda ordem | W (potência modelada) | modelado |
| `max_abs_second_order_error_w` | Maior |erro| da aproximação de segunda ordem | W (potência modelada) | modelado |
| `unstable_bins` | Intervalos com aproximação instável | contagem | modelado |

## dataset_series.csv

Série agregada da população principal por intervalo (CPU, memória, rede, disco, cobertura).
Grão: intervalo.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `bin_index` | Índice do intervalo de 300 s | intervalo de 300 s | identificador |
| `time_h` | Tempo desde o início do trace (não é horário civil) | h | identificador |
| `aggregated_cpu_weighted_pct` | CPU agregada ponderada por cpu_num (série de trabalho) | % de CPU | observado |
| `aggregated_mem_pct` | Memória agregada ponderada por cpu_num | % | observado |
| `aggregated_net_in` | net_in agregado (unidade normalizada do trace) | unidade normalizada do trace | observado |
| `aggregated_net_out` | net_out agregado (unidade normalizada do trace) | unidade normalizada do trace | observado |
| `aggregated_disk_pct` | disk_io_percent agregado | % | observado |
| `weighted_coverage` | Cobertura ponderada por cpu_num | fração | observado |
| `n_machines_present` | Máquinas da população com registro no intervalo | contagem | observado |

## dataset_profile.csv

Estatísticas descritivas da série agregada por período.
Grão: período × variável.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `period` | Período da série (treino inicial, folds, desenvolvimento, teste, completa) | - | identificador |
| `start_index` | Primeiro intervalo | intervalo de 300 s | identificador |
| `end_index` | Último intervalo | intervalo de 300 s | identificador |
| `start_time_h` | Início desde o começo do trace (não é horário civil) | h | identificador |
| `end_time_h` | Fim desde o começo do trace (não é horário civil) | h | identificador |
| `variable` | Variável agregada (cpu, mem, net_in, net_out, disk, coverage, n_machines) | - | identificador |
| `n_bins` | Número de intervalos | contagem | métrica |
| `n_missing` | Intervalos sem valor | contagem | observado |
| `mean` | Média no período | unidade da variável (% para cpu/mem/disk; unidade normalizada do trace para rede) | observado |
| `std` | Desvio-padrão no período | unidade da variável | observado |
| `min` | Mínimo no período | unidade da variável | observado |
| `p05` | Percentil 5 | unidade da variável | observado |
| `p25` | Percentil 25 | unidade da variável | observado |
| `p50` | Mediana | unidade da variável | observado |
| `p75` | Percentil 75 | unidade da variável | observado |
| `p95` | Percentil 95 | unidade da variável | observado |
| `max` | Máximo no período | unidade da variável | observado |

## dataset_changes.csv

Distribuição da variação de CPU em 1, 3 e 6 passos.
Grão: período × horizonte.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `period` | Período da série (treino inicial, folds, desenvolvimento, teste, completa) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `horizon_min` | Horizonte de previsão | min | identificador |
| `n` | Número de origens ou observações usadas no cálculo | contagem | métrica |
| `mean_change_pp` | Média de cpu(t+h) − cpu(t) | pontos percentuais de CPU (p.p.) | observado |
| `std_change_pp` | Desvio de cpu(t+h) − cpu(t) | pontos percentuais de CPU (p.p.) | observado |
| `mean_abs_change_pp` | Média de |cpu(t+h) − cpu(t)| (igual ao MAE da persistência no período) | pontos percentuais de CPU (p.p.) | observado |
| `p01_change_pp` | Percentil 1 da variação | pontos percentuais de CPU (p.p.) | observado |
| `p05_change_pp` | Percentil 5 da variação | pontos percentuais de CPU (p.p.) | observado |
| `p25_change_pp` | Percentil 25 da variação | pontos percentuais de CPU (p.p.) | observado |
| `p50_change_pp` | Mediana da variação | pontos percentuais de CPU (p.p.) | observado |
| `p75_change_pp` | Percentil 75 da variação | pontos percentuais de CPU (p.p.) | observado |
| `p95_change_pp` | Percentil 95 da variação | pontos percentuais de CPU (p.p.) | observado |
| `p99_change_pp` | Percentil 99 da variação | pontos percentuais de CPU (p.p.) | observado |

## dataset_dependence.csv

Autocorrelação da CPU e correlações defasadas com as exógenas (desenvolvimento).
Grão: tipo × variável × defasagem.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `kind` | Tipo de dependência: autocorrelação, nível da exógena × CPU futura, variação da exógena × variação futura da CPU | - | identificador |
| `variable` | Variável agregada (cpu, mem, net_in, net_out, disk, coverage, n_machines) | - | identificador |
| `lag_steps` | Defasagem | passos de 300 s | identificador |
| `lag_min` | Defasagem | min | identificador |
| `correlation` | Correlação de Pearson no desenvolvimento (associação, não antecipação causal) | adimensional | métrica |
| `n` | Número de origens ou observações usadas no cálculo | contagem | métrica |

## fleet_distribution.csv

Percentis da CPU individual e máquinas presentes por intervalo.
Grão: intervalo.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `bin_index` | Índice do intervalo de 300 s | intervalo de 300 s | identificador |
| `time_h` | Tempo desde o início do trace (não é horário civil) | h | identificador |
| `machine_cpu_p10_pct` | Percentil 10 da CPU individual | % de CPU | observado |
| `machine_cpu_p25_pct` | Percentil 25 da CPU individual | % de CPU | observado |
| `machine_cpu_p50_pct` | Mediana da CPU individual | % de CPU | observado |
| `machine_cpu_p75_pct` | Percentil 75 da CPU individual | % de CPU | observado |
| `machine_cpu_p90_pct` | Percentil 90 da CPU individual | % de CPU | observado |
| `machine_cpu_mean_pct` | Média simples da CPU individual | % de CPU | observado |
| `machine_cpu_std_pct` | Desvio-padrão da CPU individual | % de CPU | observado |
| `n_present` | Máquinas com CPU válida | contagem | observado |
| `aggregated_cpu_weighted_pct` | CPU agregada ponderada por cpu_num (série de trabalho) | % de CPU | observado |
| `weighted_coverage` | Cobertura ponderada por cpu_num | fração | observado |

## machine_coverage.csv

Cobertura temporal e CPU média por máquina.
Grão: máquina.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `machine_id` | Identificador da máquina no trace | - | identificador |
| `n_bins_present` | Intervalos com CPU válida | contagem | observado |
| `first_bin` | Primeiro intervalo com registro | intervalo de 300 s | observado |
| `last_bin` | Último intervalo com registro | intervalo de 300 s | observado |
| `coverage_fraction` | Intervalos presentes / 2.304 | fração | observado |
| `cpu_num` | Número de CPUs no metadado | contagem | observado |
| `active` | Status ativo no metadado | booleano | observado |
| `machine_mean_cpu_pct` | CPU média da máquina | % de CPU | observado |
| `machine_max_cpu_pct` | CPU máxima da máquina | % de CPU | observado |

## population_difference.csv

CPU agregada da principal e da coorte por intervalo e presença de m_2739.
Grão: intervalo.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `bin_index` | Índice do intervalo de 300 s | intervalo de 300 s | identificador |
| `time_h` | Tempo desde o início do trace (não é horário civil) | h | identificador |
| `main_cpu_pct` | CPU agregada da população principal | % de CPU | observado |
| `cohort_cpu_pct` | CPU agregada da coorte | % de CPU | observado |
| `main_n_machines` | Máquinas presentes na principal | contagem | observado |
| `cohort_n_machines` | Máquinas presentes na coorte | contagem | observado |
| `difference_main_minus_cohort_pp` | CPU principal − CPU coorte | pontos percentuais de CPU (p.p.) | observado |
| `m_2739_present` | m_2739 tem registro no intervalo | booleano | observado |
| `m_2739_cpu_pct` | CPU de m_2739 no intervalo | % de CPU | observado |

## machine_m_2739.csv

Registros de m_2739 na grade de 300 s.
Grão: intervalo com registro.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `machine_id` | Identificador da máquina no trace | - | identificador |
| `bin_index` | Índice do intervalo de 300 s | intervalo de 300 s | identificador |
| `time_h` | Tempo desde o início do trace (não é horário civil) | h | identificador |
| `cpu_pct` | CPU da máquina no intervalo | % de CPU | observado |
| `mem_pct` | Memória da máquina no intervalo | % | observado |
| `mem_gps` | mem_gps do trace (ausente na maior parte dos registros) | unidade do trace | observado |
| `net_in` | net_in do trace (normalizado pelo provedor) | unidade normalizada do trace | observado |
| `net_out` | net_out do trace (normalizado pelo provedor) | unidade normalizada do trace | observado |
| `disk_pct` | disk_io_percent da máquina | % | observado |

## cohort_comparison.csv

MAE e RMSE da V2 na principal e na coorte, e diferença máxima entre as séries.
Grão: modelo × configuração × horizonte × split.

| coluna | descrição | unidade | natureza |
|---|---|---|---|
| `model` | Modelo; sufixo _v2 indica previsões preservadas da V2 | - | identificador |
| `config` | Configuração de atributos: A (CPU), B (+memória), C (+rede e disco), D (+distribuição da frota), A_on_D (A nas origens do contraste D), - (persistência) | - | identificador |
| `horizon_steps` | Horizonte de previsão | passos de 300 s | identificador |
| `split` | Partição: fold_k (validação externa), validation_pooled (cinco folds juntos), test_final (teste de 24 h já inspecionado na V2) | - | identificador |
| `mae_pp_main` | MAE na população principal | pontos percentuais de CPU (p.p.) | métrica |
| `rmse_pp_main` | RMSE na população principal | pontos percentuais de CPU (p.p.) | métrica |
| `mae_pp_cohort` | MAE na coorte | pontos percentuais de CPU (p.p.) | métrica |
| `rmse_pp_cohort` | RMSE na coorte | pontos percentuais de CPU (p.p.) | métrica |
| `mae_difference_main_minus_cohort_pp` | MAE principal − MAE coorte | pontos percentuais de CPU (p.p.) | métrica |
| `rmse_difference_main_minus_cohort_pp` | RMSE principal − RMSE coorte | pontos percentuais de CPU (p.p.) | métrica |
| `series_max_abs_difference_test_period_pp` | Maior |CPU principal − CPU coorte| nos alvos do teste | pontos percentuais de CPU (p.p.) | observado |
| `series_max_abs_difference_full_pp` | Maior |CPU principal − CPU coorte| na série inteira | pontos percentuais de CPU (p.p.) | observado |
