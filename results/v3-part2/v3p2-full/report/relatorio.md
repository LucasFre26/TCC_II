# V3 Parte 2: relatório da corrida

Corrida `v3p2-full`, modo `full`, semente 42, dispositivo cuda.
Pré-registro: `/home/moscoso/Documents/Tcc/Tcc-sourcecode/docs/v3p2-preregistro.md` (SHA-256 `814a78fc8d805f8c7bd1dec816f520f048d903a11441853abdb9e2cbe62a0f24`).

Potência e energia são estimativas paramétricas em servidores substitutos; não há medição elétrica e PUE não foi calculado.

## Hipóteses

- **H1**: previsão hierárquica (grupos de máquinas) contra o XGBoost direto.
- **H2**: modelo global por máquina contra o XGBoost direto.
- **H3**: combinação condicionada a regime contra o XGBoost direto.
- **H4**: peso nas subidas: erro nos picos contra o XGBoost direto.
- **H5**: calibração adaptativa (ACI) contra o conformal estático.
- **H6**: energia pela rota por grupo contra a rota da média.
- **H8a**: DLinear com normalização por janela contra a formulação da V3.
- **H8b**: XGBoost com peso de recência contra o XGBoost da V2.
- **H8c**: XGBoost sem calendário contra o XGBoost da V2.
- **H7** (modelo pré-treinado): não executada; o portão de proveniência falhou (corpus com o Alibaba Cluster Trace 2018 ou não público).

## Comparações primárias (dias de desenvolvimento, exploratórias)

| hypothesis | horizon | loss_a | loss_b | mean_diff | ci_low | ci_high | days_favorable | decision | p_holm_global |
|---|---|---|---|---|---|---|---|---|---|
| H1 | 5 min | 2,910 | 2,938 | -0,029 | -0,078 | 0,015 | 2 | não distinguível | 1 |
| H1 | 15 min | 3,648 | 3,646 | 0,002 | -0,065 | 0,083 | 3 | não distinguível | 1 |
| H1 | 30 min | 3,888 | 3,958 | -0,070 | -0,155 | 0,016 | 3 | não distinguível | 1 |
| H2 | 5 min | 3,000 | 2,938 | 0,061 | -0,066 | 0,178 | 1 | não distinguível | 1 |
| H2 | 15 min | 3,642 | 3,646 | -0,005 | -0,133 | 0,126 | 1 | não distinguível | 1 |
| H2 | 30 min | 3,957 | 3,958 | -0,002 | -0,133 | 0,151 | 2 | não distinguível | 1 |
| H3 | 5 min | 2,920 | 2,938 | -0,019 | -0,082 | 0,040 | 3 | não distinguível | 1 |
| H3 | 15 min | 3,520 | 3,646 | -0,127 | -0,241 | -0,016 | 3 | melhor | 0,403 |
| H3 | 30 min | 3,918 | 3,958 | -0,040 | -0,176 | 0,113 | 2 | não distinguível | 1 |
| H4 | 5 min | 4,023 | 4,040 | -0,018 | -0,041 | 0,070 | 1 | não distinguível | 1 |
| H4 | 15 min | 4,498 | 5,009 | -0,511 | -0,821 | -0,218 | 2 | não distinguível | 0,010 |
| H4 | 30 min | 5,374 | 5,903 | -0,529 | -0,851 | -0,075 | 2 | não distinguível | 0,365 |
| H5 | 5 min | 17,319 | 17,940 | -0,620 | -1,405 | -0,007 | 3 | melhor | 0,761 |
| H5 | 15 min | 20,945 | 21,431 | -0,487 | -1,861 | 0,723 | 2 | não distinguível | 1 |
| H5 | 30 min | — | — | — | — | — | — | não calculável | — |
| H6 | 5 min | 2056,946 | 2254,581 | -197,635 | -265,782 | -131,301 | 4 | melhor | 0,010 |
| H6 | 15 min | 9491,404 | 9850,549 | -359,145 | -559,720 | -127,703 | 4 | melhor | 0,050 |
| H6 | 30 min | 20518,993 | 21166,210 | -647,217 | -1118,078 | -108,200 | 3 | melhor | 0,403 |
| H8a | 5 min | 2,889 | 2,924 | -0,035 | -0,086 | 0,014 | 3 | não distinguível | 1 |
| H8a | 15 min | 3,666 | 3,744 | -0,078 | -0,180 | 0,037 | 3 | não distinguível | 1 |
| H8a | 30 min | 4,121 | 4,537 | -0,416 | -0,657 | -0,175 | 3 | melhor | 0,028 |
| H8b | 5 min | 2,892 | 2,938 | -0,047 | -0,123 | 0,006 | 3 | não distinguível | 1 |
| H8b | 15 min | 3,789 | 3,646 | 0,142 | 0,025 | 0,245 | 1 | pior | 0,288 |
| H8b | 30 min | 4,099 | 3,958 | 0,141 | -0,025 | 0,292 | 1 | não distinguível | 1 |
| H8c | 5 min | 2,991 | 2,938 | 0,053 | -0,062 | 0,118 | 1 | não distinguível | 1 |
| H8c | 15 min | 4,006 | 3,646 | 0,359 | 0,119 | 0,509 | 0 | pior | 0,028 |
| H8c | 30 min | 4,505 | 3,958 | 0,546 | 0,227 | 0,764 | 0 | pior | 0,010 |

Diferença = candidato − referência; negativo favorece o candidato. A decisão segue a regra pré-registrada (IC e pelo menos 3 de 4 dias); `p_holm_global` corrige sobre o conjunto primário inteiro.

- **H1** (previsão hierárquica (grupos de máquinas) contra o XGBoost direto): 5 min: não distinguível (diferença -0,029, IC [-0,078; 0,015], 2/4 dias favoráveis, Holm global p = 1); 15 min: não distinguível (diferença 0,002, IC [-0,065; 0,083], 3/4 dias favoráveis, Holm global p = 1); 30 min: não distinguível (diferença -0,070, IC [-0,155; 0,016], 3/4 dias favoráveis, Holm global p = 1).
- **H2** (modelo global por máquina contra o XGBoost direto): 5 min: não distinguível (diferença 0,061, IC [-0,066; 0,178], 1/4 dias favoráveis, Holm global p = 1); 15 min: não distinguível (diferença -0,005, IC [-0,133; 0,126], 1/4 dias favoráveis, Holm global p = 1); 30 min: não distinguível (diferença -0,002, IC [-0,133; 0,151], 2/4 dias favoráveis, Holm global p = 1).
- **H3** (combinação condicionada a regime contra o XGBoost direto): 5 min: não distinguível (diferença -0,019, IC [-0,082; 0,040], 3/4 dias favoráveis, Holm global p = 1); 15 min: melhor (diferença -0,127, IC [-0,241; -0,016], 3/4 dias favoráveis, Holm global p = 0,403); 30 min: não distinguível (diferença -0,040, IC [-0,176; 0,113], 2/4 dias favoráveis, Holm global p = 1).
- **H4** (peso nas subidas: erro nos picos contra o XGBoost direto): 5 min: não distinguível (diferença -0,018, IC [-0,041; 0,070], 1/4 dias favoráveis, Holm global p = 1); 15 min: não distinguível (diferença -0,511, IC [-0,821; -0,218], 2/4 dias favoráveis, Holm global p = 0,010); 30 min: não distinguível (diferença -0,529, IC [-0,851; -0,075], 2/4 dias favoráveis, Holm global p = 0,365).
- **H5** (calibração adaptativa (ACI) contra o conformal estático): 5 min: melhor (diferença -0,620, IC [-1,405; -0,007], 3/4 dias favoráveis, Holm global p = 0,761); 15 min: não distinguível (diferença -0,487, IC [-1,861; 0,723], 2/4 dias favoráveis, Holm global p = 1); 30 min: não calculável.
- **H6** (energia pela rota por grupo contra a rota da média): 5 min: melhor (diferença -197,635, IC [-265,782; -131,301], 4/4 dias favoráveis, Holm global p = 0,010); 15 min: melhor (diferença -359,145, IC [-559,720; -127,703], 4/4 dias favoráveis, Holm global p = 0,050); 30 min: melhor (diferença -647,217, IC [-1118,078; -108,200], 3/4 dias favoráveis, Holm global p = 0,403).
- **H8a** (DLinear com normalização por janela contra a formulação da V3): 5 min: não distinguível (diferença -0,035, IC [-0,086; 0,014], 3/4 dias favoráveis, Holm global p = 1); 15 min: não distinguível (diferença -0,078, IC [-0,180; 0,037], 3/4 dias favoráveis, Holm global p = 1); 30 min: melhor (diferença -0,416, IC [-0,657; -0,175], 3/4 dias favoráveis, Holm global p = 0,028).
- **H8b** (XGBoost com peso de recência contra o XGBoost da V2): 5 min: não distinguível (diferença -0,047, IC [-0,123; 0,006], 3/4 dias favoráveis, Holm global p = 1); 15 min: pior (diferença 0,142, IC [0,025; 0,245], 1/4 dias favoráveis, Holm global p = 0,288); 30 min: não distinguível (diferença 0,141, IC [-0,025; 0,292], 1/4 dias favoráveis, Holm global p = 1).
- **H8c** (XGBoost sem calendário contra o XGBoost da V2): 5 min: não distinguível (diferença 0,053, IC [-0,062; 0,118], 1/4 dias favoráveis, Holm global p = 1); 15 min: pior (diferença 0,359, IC [0,119; 0,509], 0/4 dias favoráveis, Holm global p = 0,028); 30 min: pior (diferença 0,546, IC [0,227; 0,764], 0/4 dias favoráveis, Holm global p = 0,010).

Comparações primárias não calculáveis: H5 em 30 min. O motivo está na seção de portões e itens não executados; Holm global foi aplicado às 26 comparações calculáveis.

## Viés e deslocamento de nível (30 min, todos os blocos)

Correlação, ao longo dos blocos, entre o nível do bloco menos o do treino e o viés (obs − prev). Valor próximo de 1 indica previsão puxada para o nível do treino.

| model | variant | corr_shift_bias_all_blocks |
|---|---|---|
| dlinear | v3 | 0,99 |
| dlinear | window | 0,56 |
| patchtst | v3 | 0,87 |
| patchtst | window | 0,05 |
| tcn | v3 | 0,99 |
| tcn | window | 0,23 |
| xgboost_v2 | A | 0,99 |
| xgboost_v2 | no_calendar | 0,95 |
| xgboost_v2 | recency | 0,98 |

## Cobertura dos limites (dias de desenvolvimento)

Coberturas empíricas, sem garantia teórica sob dependência temporal; nominal 0,95 no limite superior e 0,90 no bilateral. `n_infinite` conta origens com limite infinito (nível efetivo do ACI ≤ 0).

| method | horizon_steps | coverage_upper | coverage_bilateral | mean_width | mean_interval_score | coverage_upper_peaks | n_infinite |
|---|---|---|---|---|---|---|---|
| aci_g0.005 | 1 | 0,957 | 0,905 | 13,716 | 17,319 | 0,836 | 0 |
| aci_g0.005 | 3 | 0,947 | 0,894 | 16,618 | 20,945 | 0,828 | 0 |
| aci_g0.005 | 6 | 0,947 | 0,888 | 17,905 | 22,927 | 0,883 | 63 |
| aci_g0.02 | 1 | 0,960 | 0,909 | 13,883 | 17,420 | 0,883 | 42 |
| aci_g0.02 | 3 | 0,957 | 0,904 | 15,948 | 20,918 | 0,875 | 173 |
| aci_g0.02 | 6 | 0,962 | 0,905 | 16,041 | 20,916 | 0,945 | 388 |
| rolling_n144 | 1 | 0,949 | 0,890 | 13,412 | 17,548 | 0,812 | 0 |
| rolling_n144 | 3 | 0,930 | 0,866 | 15,523 | 20,821 | 0,766 | 0 |
| rolling_n144 | 6 | 0,931 | 0,857 | 16,028 | 22,935 | 0,812 | 0 |
| rolling_n288 | 1 | 0,939 | 0,891 | 13,212 | 17,509 | 0,742 | 0 |
| rolling_n288 | 3 | 0,929 | 0,879 | 15,439 | 20,709 | 0,758 | 0 |
| rolling_n288 | 6 | 0,928 | 0,872 | 16,014 | 22,500 | 0,781 | 0 |
| spci | 1 | 0,886 | 0,802 | 10,680 | 19,602 | 0,594 | 0 |
| spci | 3 | 0,863 | 0,735 | 10,953 | 25,037 | 0,633 | 0 |
| spci | 6 | 0,842 | 0,693 | 10,838 | 29,046 | 0,570 | 0 |
| static | 1 | 0,907 | 0,859 | 12,001 | 17,940 | 0,633 | 0 |
| static | 3 | 0,888 | 0,845 | 13,726 | 21,431 | 0,633 | 0 |
| static | 6 | 0,854 | 0,803 | 13,369 | 24,668 | 0,492 | 0 |

## Aproximação por agregação na energia (30 min, servidor substituto da V3)

Rota observada − referência por máquina, em fração da referência. Estimativa paramétrica, não medição.

| r | route | mean_reference_wh | mean_aggregation_wh | mean_relative_aggregation |
|---|---|---|---|---|
| 1 | group_candidate | 411496,1717 | 0,0000 | 0,0000 |
| 1 | group_mean_k16 | 411496,1717 | 0,0000 | 0,0000 |
| 1 | group_mean_k4 | 411496,1717 | 0,0000 | 0,0000 |
| 1 | group_mean_k8 | 411496,1717 | 0,0000 | 0,0000 |
| 1 | mean | 411496,1717 | 0,0000 | 0,0000 |
| 1,4000 | group_candidate | 502169,1667 | 2552,4577 | 0,0051 |
| 1,4000 | group_mean_k16 | 502169,1667 | 1313,1282 | 0,0026 |
| 1,4000 | group_mean_k4 | 502169,1667 | 3528,3026 | 0,0070 |
| 1,4000 | group_mean_k8 | 502169,1667 | 2718,8662 | 0,0054 |
| 1,4000 | mean | 502169,1667 | 6059,9008 | 0,0121 |
| 1,8000 | group_candidate | 563999,5252 | 3731,9070 | 0,0066 |
| 1,8000 | group_mean_k16 | 563999,5252 | 2019,0826 | 0,0036 |
| 1,8000 | group_mean_k4 | 563999,5252 | 5191,1922 | 0,0092 |
| 1,8000 | group_mean_k8 | 563999,5252 | 3905,9708 | 0,0069 |
| 1,8000 | mean | 563999,5252 | 9482,6907 | 0,0168 |

## Escala do modelo global (bloco único, descritivo)

| n_machines | horizon_steps | mae | fit_seconds_total |
|---|---|---|---|
| 256 | 1 | 3,097 | 1,667 |
| 256 | 3 | 3,837 | 1,675 |
| 256 | 6 | 4,082 | 2,101 |
| 1024 | 1 | 3,098 | 4,142 |
| 1024 | 3 | 3,853 | 3,579 |
| 1024 | 6 | 4,050 | 4,109 |
| 4015 | 1 | 3,121 | 14,007 |
| 4015 | 3 | 3,929 | 12,616 |
| 4015 | 6 | 4,034 | 13,583 |

Tamanho efetivo estimado (efeito de desenho × origens distintas, sem correção da autocorrelação):

| block | n_examples_train_h1 | n_distinct_origins_train | design_mean_correlation | effective_sample_estimate |
|---|---|---|---|---|
| day_3 | 245664 | 375 | 0,581 | 645,000 |
| day_4 | 393211 | 519 | 0,589 | 880,783 |
| day_5 | 539835 | 663 | 0,632 | 1048,006 |
| day_6 | 686987 | 807 | 0,660 | 1222,523 |
| day_7 | 834099 | 951 | 0,663 | 1434,054 |

## Exploratórios que sobrevivem a Holm por horizonte

Não substituem os candidatos primários; listados para orientar trabalho futuro.

| family | hypothesis | model_a | variant_a | model_b | variant_b | horizon_steps | mean_diff | decision | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| robustness | H8a | patchtst | window | patchtst | v3 | 1 | -0,482 | melhor | 0,002 |
| robustness | H8a | patchtst | window | patchtst | v3 | 3 | -0,269 | melhor | 0,040 |
| hierarchical | H1 | hierarchical | failure_domain_knatural_ols | xgboost_v2 | A | 1 | -0,062 | melhor | 0,035 |
| hierarchical | H1 | hierarchical | mean_k8_ols | xgboost_v2 | A | 1 | -0,065 | melhor | 0,045 |
| hierarchical | H1 | hierarchical | profile_k16_ols | xgboost_v2 | A | 1 | -0,085 | melhor | 0,012 |
| peaks | H4-mae | xgboost_rise | candidate | xgboost_v2 | A | 3 | 0,164 | não distinguível | 0,000 |
| calibration | H5 | xgboost_quantile | spci | xgboost_quantile | static | 1 | 1,663 | pior | 0,006 |
| calibration | H5 | xgboost_quantile | spci | xgboost_quantile | static | 3 | 3,606 | pior | 0,001 |
| calibration | H5 | xgboost_quantile | rolling_n288 | xgboost_quantile | static | 6 | -2,168 | melhor | 0,009 |
| calibration | H5 | xgboost_quantile | spci | xgboost_quantile | static | 6 | 4,377 | pior | 0,001 |
| energy | H6 | persistence | mean_route | xgboost_v2 | mean_route | 1 | 166,919 | pior | 0,018 |
| energy | H6 | hierarchical | mean_route | xgboost_v2 | mean_route | 1 | -60,681 | melhor | 0,018 |
| energy | H6 | global | machine_route | xgboost_v2 | mean_route | 1 | -147,213 | melhor | 0,011 |
| energy | H6 | persistence | mean_route | xgboost_v2 | mean_route | 3 | 1714,229 | pior | 0,002 |
| energy | H6 | hybrid | mean_route | xgboost_v2 | mean_route | 3 | -355,871 | melhor | 0,038 |
| energy | H6 | hierarchical | mean_route | xgboost_v2 | mean_route | 3 | -152,809 | melhor | 0,038 |
| energy | H6 | global | machine_route | xgboost_v2 | mean_route | 3 | -612,710 | melhor | 0,006 |
| energy | H6 | persistence | mean_route | xgboost_v2 | mean_route | 6 | 6281,489 | pior | 0,002 |
| energy | H6 | global | machine_route | xgboost_v2 | mean_route | 6 | -1363,258 | melhor | 0,042 |

## Teste herdado (dia 7)

O dia 7 foi inspecionado na V2, na V3 e na análise expandida da V3; esta é a quarta inspeção. As diferenças abaixo são só continuidade com a V3, sem inferência e sem valor de confirmação.

| model_a | variant_a | model_b | variant_b | horizon_steps | mae_a | mae_b | mae_diff |
|---|---|---|---|---|---|---|---|
| xgboost_v2 | recency | xgboost_v2 | A | 1 | 3,136 | 3,040 | 0,096 |
| xgboost_v2 | recency | xgboost_v2 | A | 3 | 3,830 | 3,856 | -0,026 |
| xgboost_v2 | recency | xgboost_v2 | A | 6 | 4,488 | 4,347 | 0,141 |
| xgboost_v2 | no_calendar | xgboost_v2 | A | 1 | 3,172 | 3,040 | 0,133 |
| xgboost_v2 | no_calendar | xgboost_v2 | A | 3 | 4,072 | 3,856 | 0,216 |
| xgboost_v2 | no_calendar | xgboost_v2 | A | 6 | 4,549 | 4,347 | 0,201 |
| dlinear | window | dlinear | v3 | 1 | 3,220 | 3,202 | 0,018 |
| dlinear | window | dlinear | v3 | 3 | 4,323 | 4,404 | -0,080 |
| dlinear | window | dlinear | v3 | 6 | 4,855 | 5,239 | -0,384 |
| tcn | window | tcn | v3 | 1 | 3,187 | 3,304 | -0,117 |
| tcn | window | tcn | v3 | 3 | 4,151 | 4,662 | -0,511 |
| tcn | window | tcn | v3 | 6 | 4,735 | 5,354 | -0,619 |
| patchtst | window | patchtst | v3 | 1 | 3,228 | 3,291 | -0,063 |
| patchtst | window | patchtst | v3 | 3 | 4,211 | 4,027 | 0,183 |
| patchtst | window | patchtst | v3 | 6 | 4,928 | 4,418 | 0,510 |
| hierarchical | candidate | xgboost_v2 | A | 1 | 3,006 | 3,040 | -0,034 |
| hierarchical | candidate | xgboost_v2 | A | 3 | 3,987 | 3,856 | 0,131 |
| hierarchical | candidate | xgboost_v2 | A | 6 | 4,280 | 4,347 | -0,067 |
| global_xgb | main | xgboost_v2 | A | 1 | 3,335 | 3,040 | 0,295 |
| global_xgb | main | xgboost_v2 | A | 3 | 3,941 | 3,856 | 0,085 |
| global_xgb | main | xgboost_v2 | A | 6 | 4,669 | 4,347 | 0,322 |
| global_tcn | window | xgboost_v2 | A | 1 | 3,145 | 3,040 | 0,106 |
| global_tcn | window | xgboost_v2 | A | 3 | 4,126 | 3,856 | 0,270 |
| global_tcn | window | xgboost_v2 | A | 6 | 4,533 | 4,347 | 0,186 |
| combination | candidate | xgboost_v2 | A | 1 | 3,010 | 3,040 | -0,030 |
| combination | candidate | xgboost_v2 | A | 3 | 3,959 | 3,856 | 0,103 |
| combination | candidate | xgboost_v2 | A | 6 | 4,340 | 4,347 | -0,008 |
| xgboost_rise | candidate | xgboost_v2 | A | 1 | 3,040 | 3,040 | 0 |
| xgboost_rise | candidate | xgboost_v2 | A | 3 | 3,802 | 3,856 | -0,054 |
| xgboost_rise | candidate | xgboost_v2 | A | 6 | 4,183 | 4,347 | -0,164 |


## Portões e itens não executados

- portão `tcn_global` (global_model): cumprido.
- portão `spci_budget` (calibration): cumprido.
- pretrained: modelo pré-treinado (H7) — Chronos-2 lista o Alibaba Cluster Trace 2018 (5 min) no corpus de treino (Tabela 6 do relatório técnico, arXiv 2510.15821); o LOTSA, corpus do Moirai, contém alibaba_cluster_trace_2018; o corpus do Chronos-Bolt não é público. Nenhum dos três satisfaz o portão de proveniência.
- calibration: comparação aci_g0.02 × static h1 — limite infinito em parte das origens (nível efetivo α_t ≤ 0); interval score indefinido
- calibration: comparação aci_g0.02 × static h3 — limite infinito em parte das origens (nível efetivo α_t ≤ 0); interval score indefinido
- calibration: comparação aci_g0.005 × static h6 — limite infinito em parte das origens (nível efetivo α_t ≤ 0); interval score indefinido
- calibration: comparação aci_g0.02 × static h6 — limite infinito em parte das origens (nível efetivo α_t ≤ 0); interval score indefinido

## Ameaças à validade

- Os dias 4 a 6 serviram de validação na V2 (seleção de hiperparâmetros) e na V3; só o dia 3 nunca tinha sido avaliado. Todos os resultados são exploratórios.
- Quatro dias de desenvolvimento: a variação entre dias é estimada com n = 4; o bootstrap trata a dependência dentro do dia.
- O dia 7 é o teste herdado, na quarta inspeção; ele aparece só como continuidade com a V3.
- A covariância da MinT e a escolha do candidato hierárquico usam a mesma validação interna.
- O tamanho efetivo do modelo global vem do efeito de desenho e não corrige a autocorrelação temporal.
- Coberturas dos limites são empíricas; nenhum método tem garantia teórica sob dependência temporal.
- Energia de TI modelada em servidores substitutos do SPECpower_ssj2008: diferença entre duas estimativas, não erro contra medidor; PUE não foi calculado.
- Referências metodológicas a conferir e incluir na bibliografia antes da redação: MinT (Wickramasuriya, Athanasopoulos e Hyndman, 2019), ACI (Gibbs e Candès, 2021), SPCI (Xu e Xie, 2023), RevIN (Kim et al., 2022), NLinear (Zeng et al., 2023).

## Figuras

- `figures/png/01_comparacoes_primarias.png`: Comparações primárias nos dias de desenvolvimento (IC de 95%, bootstrap estratificado por dia). Dados: `figures/data/01_comparacoes_primarias.csv`.
- `figures/png/02_deslocamento_vies.png`: Viés por bloco contra o deslocamento de nível em relação ao treino (30 min). Dados: `figures/data/02_deslocamento_vies.csv`.
- `figures/png/03_escala_global.png`: Diagnóstico de escala do XGBoost global num único bloco (descritivo). Dados: `figures/data/03_escala_global.csv`.
- `figures/png/04_cobertura_calibracao.png`: Cobertura empírica do limite superior por método nos dias de desenvolvimento. Dados: `figures/data/04_cobertura_calibracao.csv`.
- `figures/png/05_decomposicao_energia.png`: Decomposição da diferença de energia de TI modelada (servidor substituto, r = 1,4). Dados: `figures/data/05_decomposicao_energia.csv`.
- `figures/png/06_compromisso_picos.png`: Compromisso entre erro médio e subestimação nos picos por λ (dias de desenvolvimento). Dados: `figures/data/06_compromisso_picos.csv`.

## Conclusões sustentadas pelos dados

- 6 de 26 comparações primárias calculáveis favoreceram o candidato pela regra pré-registrada (H3/15 min, H5/5 min, H6/5 min, H6/15 min, H6/30 min, H8a/30 min); 3 favoreceram a referência (H8b/15 min, H8c/15 min, H8c/30 min).
- Sobreviveram a Holm sobre o conjunto primário: H4/15 min (não distinguível), H6/5 min (melhor), H8a/30 min (melhor), H8c/15 min (pior), H8c/30 min (pior).
- Todas as conclusões são exploratórias (dias de desenvolvimento já usados nas versões anteriores) e se limitam a este trace de 8 dias.
