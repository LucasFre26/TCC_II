# Relatório de execução da V3 — 20261104T232620Z-5b659dad

Previsão de CPU no Alibaba Cluster Trace v2018 e energia de TI modelada pelo modelo de Fan num servidor substituto. O trace não tem medição elétrica; PUE não é calculado.

## Corrida

- semente: 42
- hash da configuração: `5b659dad552855bc4c73048e04067179131b54b4fe718801780ec756156341f2`
- dados: `{"source": "alibaba", "file_sha256": {"machine_usage": "3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a", "machine_meta": "b5b1b786b22cd413a3674b8f2ebfb2f02fac991c95df537f363ef2797c8f6d55"}, "cache": {"hit": true, "key": "3f8b24db7c0af36c3bb9"}, "synthetic": null}`
- commit: `1f92ab0cc850ded07d4a549be10778313d716915` (alterações locais: True)
- dispositivo pedido/efetivo: auto/cuda (NVIDIA GeForce RTX 4050 Laptop GPU)
- versões: `{"python": "3.11.14", "numpy": "2.4.6", "pandas": "3.0.6", "scikit-learn": "1.9.1", "xgboost": "3.2.0", "matplotlib": "3.11.2", "pyyaml": "6.0.3", "statsmodels": "0.15.0", "torch": "2.14.1"}`

## Famílias concluídas

- integrity: 6.1 s (2026-11-04T23:26:26+00:00)
- residual: 203.4 s (2026-11-04T23:29:50+00:00)
- neural: 834.0 s (2026-11-04T23:43:44+00:00)
- quantile: 196.9 s (2026-11-04T23:47:01+00:00)
- energy: 3.4 s (2026-11-04T23:47:04+00:00)

## O que não rodou (orçamento, portão ou ausência de dado)

- nada

## Auditoria da população

- principal: 4023 máquinas, peso 386208; coorte: 4022 máquinas, peso 386112.
- só na principal: m_2739 (cpu_num 96, inativa (status fora dos ativos), status {'IMPORT_INSTALLING': 5}).
- diferença máxima de CPU agregada entre as séries: 0.0258 p.p.; séries idênticas: False.

## MAE no teste final, configuração A (modo direto)

Ordem de leitura, não de seleção: o menor MAE do teste não promove modelo.

| model | MAE h1 | MAE h3 | MAE h6 |
| --- | --- | --- | --- |
| persistence | 3.3513 | 4.7018 | 5.3302 |
| xgboost_v2 | 3.0641 | 3.8562 | 4.3336 |
| xgboost_delta | 3.0880 | 3.9481 | 4.5039 |
| statespace | 3.2594 | 4.4104 | 4.9973 |
| hybrid | 3.0254 | 3.9233 | 4.3187 |
| dlinear | 3.2687 | 4.4628 | 5.1441 |
| tcn | 3.4270 | 4.7126 | 5.3777 |
| nbeatsx | 3.8065 | 5.1990 | 5.2546 |
| nhits | 4.2058 | 5.2566 | 5.7252 |
| patchtst | 3.2723 | 4.3041 | 4.3208 |


## Ablação B e C (teste final)

| model | config | MAE h1 | MAE h3 | MAE h6 |
| --- | --- | --- | --- | --- |
| dlinear | B | 3.2631 | 4.1567 | 5.0525 |
| dlinear | C | 3.7368 | 4.3653 | 5.7647 |
| hybrid | B | 3.0451 | 4.1360 | 4.4143 |
| hybrid | C | 3.1820 | 4.0191 | 4.3468 |
| nbeatsx | B | 4.0862 | 6.0722 | 6.3926 |
| nbeatsx | C | 4.3211 | 4.6962 | 4.5962 |
| tcn | B | 3.6499 | 5.5399 | 6.1215 |
| tcn | C | 4.0830 | 5.0353 | 6.2530 |
| xgboost_delta | B | 3.1825 | 4.0621 | 4.5962 |
| xgboost_delta | C | 3.0993 | 4.1167 | 4.4973 |
| xgboost_v2 | B | 3.0796 | 3.8782 | 4.4006 |
| xgboost_v2 | C | 3.0958 | 3.8460 | 4.2113 |


## Trajetória de seis passos (secundária; não substitui o modo direto)

| model | config | MAE h1 | MAE h3 | MAE h6 |
| --- | --- | --- | --- | --- |
| dlinear | A | 3.2335 | 4.4487 | 5.2340 |
| dlinear | B | 3.2782 | 4.1353 | 4.8421 |
| dlinear | C | 3.6134 | 4.3170 | 4.6894 |
| nbeatsx | A | 4.8338 | 5.2121 | 5.5590 |
| nbeatsx | B | 5.9546 | 5.9539 | 6.0409 |
| nbeatsx | C | 4.2799 | 4.6938 | 4.8796 |
| tcn | A | 3.7562 | 4.4606 | 5.2359 |
| tcn | B | 4.1459 | 5.2476 | 5.5356 |
| tcn | C | 4.0259 | 5.2559 | 5.5938 |


## Comparações pré-registradas (bootstrap por blocos de 1 h, 95%)

Diferença = perda de A menos perda de B; só é distinguível de zero quando o intervalo não contém zero. Blocos de 30 min e 2 h estão em bootstrap_ci.csv como sensibilidade.

| family | comparison | quantile | horizon_steps | model_a | config_a | model_b | config_b | mean_diff | ci_low | ci_high | significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual | new_vs_persistence_A |  | 1 | xgboost_delta | A | persistence | - | -0.2633 | -0.3916 | -0.1030 | True |
| residual | new_vs_persistence_A |  | 1 | statespace | A | persistence | - | -0.0919 | -0.2302 | 0.0549 | False |
| residual | new_vs_persistence_A |  | 1 | hybrid | A | persistence | - | -0.3259 | -0.4681 | -0.1579 | True |
| residual | new_vs_v2_xgboost_A |  | 1 | xgboost_delta | A | xgboost_v2 | A | 0.0239 | -0.0799 | 0.1129 | False |
| residual | new_vs_v2_xgboost_A |  | 1 | statespace | A | xgboost_v2 | A | 0.1954 | -0.0488 | 0.3951 | False |
| residual | new_vs_v2_xgboost_A |  | 1 | hybrid | A | xgboost_v2 | A | -0.0387 | -0.1544 | 0.0505 | False |
| residual | ablation_vs_A |  | 1 | xgboost_delta | B | xgboost_delta | A | 0.0945 | -0.0280 | 0.2257 | False |
| residual | ablation_vs_A |  | 1 | xgboost_delta | C | xgboost_delta | A | 0.0113 | -0.0431 | 0.0497 | False |
| residual | ablation_vs_A |  | 1 | hybrid | B | hybrid | A | 0.0197 | -0.0270 | 0.0684 | False |
| residual | ablation_vs_A |  | 1 | hybrid | C | hybrid | A | 0.1566 | -0.0072 | 0.2928 | False |
| residual | hybrid_vs_statespace |  | 1 | hybrid | A | statespace | A | -0.2340 | -0.4445 | -0.0367 | True |
| residual | d_vs_a |  | 1 | xgboost_delta | D | xgboost_delta | A | 0.1755 | 0.0416 | 0.3149 | True |
| residual | new_vs_persistence_A |  | 3 | xgboost_delta | A | persistence | - | -0.7537 | -1.3034 | -0.1277 | True |
| residual | new_vs_persistence_A |  | 3 | statespace | A | persistence | - | -0.2914 | -0.3966 | -0.1849 | True |
| residual | new_vs_persistence_A |  | 3 | hybrid | A | persistence | - | -0.7785 | -1.2201 | -0.2480 | True |
| residual | new_vs_v2_xgboost_A |  | 3 | xgboost_delta | A | xgboost_v2 | A | 0.0919 | -0.1841 | 0.3073 | False |
| residual | new_vs_v2_xgboost_A |  | 3 | statespace | A | xgboost_v2 | A | 0.5542 | -0.2755 | 1.2575 | False |
| residual | new_vs_v2_xgboost_A |  | 3 | hybrid | A | xgboost_v2 | A | 0.0670 | -0.2786 | 0.3704 | False |
| residual | ablation_vs_A |  | 3 | xgboost_delta | B | xgboost_delta | A | 0.1140 | 0.0212 | 0.2569 | True |
| residual | ablation_vs_A |  | 3 | xgboost_delta | C | xgboost_delta | A | 0.1686 | -0.0511 | 0.3617 | False |
| residual | ablation_vs_A |  | 3 | hybrid | B | hybrid | A | 0.2128 | 0.0065 | 0.4868 | True |
| residual | ablation_vs_A |  | 3 | hybrid | C | hybrid | A | 0.0958 | 0.0085 | 0.1649 | True |
| residual | hybrid_vs_statespace |  | 3 | hybrid | A | statespace | A | -0.4871 | -0.9378 | 0.0306 | False |
| residual | d_vs_a |  | 3 | xgboost_delta | D | xgboost_delta | A | 0.3758 | 0.1099 | 0.6504 | True |
| residual | new_vs_persistence_A |  | 6 | xgboost_delta | A | persistence | - | -0.8263 | -1.5271 | 0.0741 | False |
| residual | new_vs_persistence_A |  | 6 | statespace | A | persistence | - | -0.3329 | -0.4762 | -0.2018 | True |
| residual | new_vs_persistence_A |  | 6 | hybrid | A | persistence | - | -1.0115 | -1.6850 | -0.1393 | True |
| residual | new_vs_v2_xgboost_A |  | 6 | xgboost_delta | A | xgboost_v2 | A | 0.1703 | -0.2382 | 0.5411 | False |
| residual | new_vs_v2_xgboost_A |  | 6 | statespace | A | xgboost_v2 | A | 0.6637 | -0.4673 | 1.5065 | False |
| residual | new_vs_v2_xgboost_A |  | 6 | hybrid | A | xgboost_v2 | A | -0.0149 | -0.4335 | 0.3532 | False |
| residual | ablation_vs_A |  | 6 | xgboost_delta | B | xgboost_delta | A | 0.0923 | -0.0193 | 0.2097 | False |
| residual | ablation_vs_A |  | 6 | xgboost_delta | C | xgboost_delta | A | -0.0066 | -0.1997 | 0.1743 | False |
| residual | ablation_vs_A |  | 6 | hybrid | B | hybrid | A | 0.0956 | -0.0285 | 0.1882 | False |
| residual | ablation_vs_A |  | 6 | hybrid | C | hybrid | A | 0.0282 | -0.2125 | 0.2327 | False |
| residual | hybrid_vs_statespace |  | 6 | hybrid | A | statespace | A | -0.6786 | -1.3018 | 0.1392 | False |
| residual | d_vs_a |  | 6 | xgboost_delta | D | xgboost_delta | A | 0.7263 | 0.2601 | 1.1687 | True |
| neural | new_vs_persistence_A |  | 1 | dlinear | A | persistence | - | -0.0825 | -0.2499 | 0.1296 | False |
| neural | new_vs_v2_xgboost_A |  | 1 | dlinear | A | xgboost_v2 | A | 0.2047 | 0.0093 | 0.3760 | True |
| neural | ablation_vs_A |  | 1 | dlinear | B | dlinear | A | -0.0056 | -0.0587 | 0.0587 | False |
| neural | ablation_vs_A |  | 1 | dlinear | C | dlinear | A | 0.4681 | 0.1738 | 0.7706 | True |
| neural | new_vs_persistence_A |  | 1 | tcn | A | persistence | - | 0.0757 | -0.1714 | 0.3985 | False |
| neural | new_vs_v2_xgboost_A |  | 1 | tcn | A | xgboost_v2 | A | 0.3629 | 0.1158 | 0.6001 | True |
| neural | ablation_vs_A |  | 1 | tcn | B | tcn | A | 0.2229 | 0.0158 | 0.4648 | True |
| neural | ablation_vs_A |  | 1 | tcn | C | tcn | A | 0.6559 | 0.2055 | 1.1580 | True |
| neural | new_vs_persistence_A |  | 1 | nbeatsx | A | persistence | - | 0.4552 | -0.0526 | 1.0551 | False |
| neural | new_vs_v2_xgboost_A |  | 1 | nbeatsx | A | xgboost_v2 | A | 0.7424 | 0.2269 | 1.2689 | True |
| neural | ablation_vs_A |  | 1 | nbeatsx | B | nbeatsx | A | 0.2797 | -0.1279 | 0.6870 | False |
| neural | ablation_vs_A |  | 1 | nbeatsx | C | nbeatsx | A | 0.5147 | 0.2043 | 0.7979 | True |
| neural | d_vs_a |  | 1 | dlinear | D | dlinear | A | -0.0198 | -0.2559 | 0.1923 | False |
| neural | new_vs_persistence_A |  | 3 | dlinear | A | persistence | - | -0.2389 | -0.7045 | 0.4179 | False |
| neural | new_vs_v2_xgboost_A |  | 3 | dlinear | A | xgboost_v2 | A | 0.6066 | 0.1695 | 1.0733 | True |
| neural | ablation_vs_A |  | 3 | dlinear | B | dlinear | A | -0.3061 | -0.7455 | 0.0532 | False |
| neural | ablation_vs_A |  | 3 | dlinear | C | dlinear | A | -0.0975 | -0.4656 | 0.2334 | False |
| neural | new_vs_persistence_A |  | 3 | tcn | A | persistence | - | 0.0108 | -0.7127 | 0.9360 | False |
| neural | new_vs_v2_xgboost_A |  | 3 | tcn | A | xgboost_v2 | A | 0.8564 | 0.3354 | 1.3475 | True |
| neural | ablation_vs_A |  | 3 | tcn | B | tcn | A | 0.8273 | 0.2998 | 1.3613 | True |
| neural | ablation_vs_A |  | 3 | tcn | C | tcn | A | 0.3227 | -0.0183 | 0.6842 | False |
| neural | new_vs_persistence_A |  | 3 | nbeatsx | A | persistence | - | 0.4972 | -0.7025 | 1.7945 | False |
| neural | new_vs_v2_xgboost_A |  | 3 | nbeatsx | A | xgboost_v2 | A | 1.3428 | 0.4723 | 2.1316 | True |
| neural | ablation_vs_A |  | 3 | nbeatsx | B | nbeatsx | A | 0.8732 | 0.3845 | 1.5748 | True |
| neural | ablation_vs_A |  | 3 | nbeatsx | C | nbeatsx | A | -0.5029 | -1.1193 | 0.2535 | False |
| neural | d_vs_a |  | 3 | dlinear | D | dlinear | A | 0.7230 | -0.1267 | 1.4443 | False |
| neural | new_vs_persistence_A |  | 6 | dlinear | A | persistence | - | -0.1862 | -0.9526 | 0.8016 | False |
| neural | new_vs_v2_xgboost_A |  | 6 | dlinear | A | xgboost_v2 | A | 0.8105 | 0.3340 | 1.2672 | True |
| neural | ablation_vs_A |  | 6 | dlinear | B | dlinear | A | -0.0915 | -0.3680 | 0.1685 | False |
| neural | ablation_vs_A |  | 6 | dlinear | C | dlinear | A | 0.6207 | -0.0665 | 1.3323 | False |
| neural | new_vs_persistence_A |  | 6 | tcn | A | persistence | - | 0.0475 | -1.0169 | 1.2969 | False |
| neural | new_vs_v2_xgboost_A |  | 6 | tcn | A | xgboost_v2 | A | 1.0441 | 0.4642 | 1.5855 | True |
| neural | ablation_vs_A |  | 6 | tcn | B | tcn | A | 0.7438 | 0.0407 | 1.5999 | True |
| neural | ablation_vs_A |  | 6 | tcn | C | tcn | A | 0.8753 | 0.2297 | 1.6958 | True |
| neural | new_vs_persistence_A |  | 6 | nbeatsx | A | persistence | - | -0.0757 | -1.3479 | 1.2830 | False |
| neural | new_vs_v2_xgboost_A |  | 6 | nbeatsx | A | xgboost_v2 | A | 0.9210 | 0.2414 | 1.4908 | True |
| neural | ablation_vs_A |  | 6 | nbeatsx | B | nbeatsx | A | 1.1380 | 0.3655 | 2.1793 | True |
| neural | ablation_vs_A |  | 6 | nbeatsx | C | nbeatsx | A | -0.6584 | -1.8324 | 0.4467 | False |
| neural | d_vs_a |  | 6 | dlinear | D | dlinear | A | 1.5632 | -0.0411 | 2.9846 | False |
| quantile | ablation_vs_A | 0.5000 | 1 | xgboost_quantile | B | xgboost_quantile | A | 0.0824 | -0.0274 | 0.1665 | False |
| quantile | ablation_vs_A | 0.5000 | 1 | xgboost_quantile | C | xgboost_quantile | A | -0.0074 | -0.0757 | 0.0465 | False |
| quantile | ablation_vs_A | 0.9000 | 1 | xgboost_quantile | B | xgboost_quantile | A | -0.0508 | -0.0884 | -0.0083 | True |
| quantile | ablation_vs_A | 0.9000 | 1 | xgboost_quantile | C | xgboost_quantile | A | -0.0305 | -0.0674 | 0.0209 | False |
| quantile | ablation_vs_A | 0.9500 | 1 | xgboost_quantile | B | xgboost_quantile | A | 0.0241 | -0.0277 | 0.0718 | False |
| quantile | ablation_vs_A | 0.9500 | 1 | xgboost_quantile | C | xgboost_quantile | A | -0.0015 | -0.0381 | 0.0384 | False |
| quantile | ablation_vs_A | 0.5000 | 3 | xgboost_quantile | B | xgboost_quantile | A | -0.0403 | -0.1006 | 0.0185 | False |
| quantile | ablation_vs_A | 0.5000 | 3 | xgboost_quantile | C | xgboost_quantile | A | 0.1023 | -0.0032 | 0.2215 | False |
| quantile | ablation_vs_A | 0.9000 | 3 | xgboost_quantile | B | xgboost_quantile | A | 0.0325 | -0.0206 | 0.0954 | False |
| quantile | ablation_vs_A | 0.9000 | 3 | xgboost_quantile | C | xgboost_quantile | A | 0.0016 | -0.0512 | 0.0418 | False |
| quantile | ablation_vs_A | 0.9500 | 3 | xgboost_quantile | B | xgboost_quantile | A | -0.0060 | -0.0519 | 0.0409 | False |
| quantile | ablation_vs_A | 0.9500 | 3 | xgboost_quantile | C | xgboost_quantile | A | -0.0196 | -0.0848 | 0.0668 | False |
| quantile | ablation_vs_A | 0.5000 | 6 | xgboost_quantile | B | xgboost_quantile | A | -0.0167 | -0.1678 | 0.1073 | False |
| quantile | ablation_vs_A | 0.5000 | 6 | xgboost_quantile | C | xgboost_quantile | A | 0.0618 | -0.1022 | 0.2327 | False |
| quantile | ablation_vs_A | 0.9000 | 6 | xgboost_quantile | B | xgboost_quantile | A | -0.0713 | -0.2384 | 0.0357 | False |
| quantile | ablation_vs_A | 0.9000 | 6 | xgboost_quantile | C | xgboost_quantile | A | -0.0591 | -0.2645 | 0.0834 | False |
| quantile | ablation_vs_A | 0.9500 | 6 | xgboost_quantile | B | xgboost_quantile | A | -0.0285 | -0.0681 | 0.0024 | False |
| quantile | ablation_vs_A | 0.9500 | 6 | xgboost_quantile | C | xgboost_quantile | A | 0.1147 | 0.0192 | 0.2389 | True |


## Contraste D contra A (origens com D completa, A recalculado nelas)

| family | model | config | MAE h1 | MAE h3 | MAE h6 |
| --- | --- | --- | --- | --- | --- |
| neural | dlinear | A | 3.2687 | 4.4628 | 5.1441 |
| neural | dlinear | D | 3.2489 | 5.1858 | 6.7073 |
| residual | xgboost_delta | A | 3.0880 | 3.9481 | 4.5039 |
| residual | xgboost_delta | D | 3.2635 | 4.3239 | 5.2302 |


## Redes

- janela escolhida: 36 passos; núcleo concluído: True; tempo da família: 13.9 min.
- portão N-HiTS/PatchTST: true; ajustadas: ['nhits', 'patchtst']; 

| arch | params | train_s | peak_cuda_mb |
| --- | --- | --- | --- |
| dlinear | 1776 | 581.15 | 68.50 |
| nbeatsx | 59008 | 52.93 | 67.52 |
| nhits | 31983 | 15.96 | 65.74 |
| patchtst | 18145 | 19.55 | 73.19 |
| tcn | 10305 | 126.34 | 68.61 |


## Intervalos quantílicos no teste (cobertura empírica)

| config | horizon_steps | upper | conformal_applied | mean_width | interval_coverage | upper_coverage |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1 | Q0.95 cru | False | 6.7101 | 0.4688 | 0.8993 |
| A | 1 | Q0.95 ajustado | True | 7.4237 | 0.4792 | 0.9097 |
| A | 1 | Q0.9 ajustado | True | 5.2013 | 0.4201 | 0.8507 |
| B | 1 | Q0.95 cru | False | 6.4215 | 0.4167 | 0.8958 |
| B | 1 | Q0.95 ajustado | True | 7.1310 | 0.4236 | 0.9028 |
| B | 1 | Q0.9 ajustado | True | 4.8480 | 0.3924 | 0.8715 |
| C | 1 | Q0.95 cru | False | 7.0059 | 0.4861 | 0.9028 |
| C | 1 | Q0.95 ajustado | True | 8.2153 | 0.5035 | 0.9201 |
| C | 1 | Q0.9 ajustado | True | 5.0855 | 0.4410 | 0.8576 |
| A | 3 | Q0.95 cru | False | 7.9143 | 0.5069 | 0.8889 |
| A | 3 | Q0.95 ajustado | True | 9.3498 | 0.5417 | 0.9236 |
| A | 3 | Q0.9 ajustado | True | 6.6649 | 0.4688 | 0.8507 |
| B | 3 | Q0.95 cru | False | 7.7297 | 0.4757 | 0.8854 |
| B | 3 | Q0.95 ajustado | True | 9.2016 | 0.5278 | 0.9375 |
| B | 3 | Q0.9 ajustado | True | 6.2697 | 0.4340 | 0.8438 |
| C | 3 | Q0.95 cru | False | 8.1216 | 0.5347 | 0.8750 |
| C | 3 | Q0.95 ajustado | True | 9.0947 | 0.5764 | 0.9167 |
| C | 3 | Q0.9 ajustado | True | 7.2558 | 0.5000 | 0.8403 |
| A | 6 | Q0.95 cru | False | 8.6200 | 0.5139 | 0.8924 |
| A | 6 | Q0.95 ajustado | True | 9.3070 | 0.5486 | 0.9271 |
| A | 6 | Q0.9 ajustado | True | 7.5990 | 0.4688 | 0.8472 |
| B | 6 | Q0.95 cru | False | 8.8459 | 0.5208 | 0.9167 |
| B | 6 | Q0.95 ajustado | True | 9.7284 | 0.5486 | 0.9444 |
| B | 6 | Q0.9 ajustado | True | 7.2303 | 0.4618 | 0.8576 |
| C | 6 | Q0.95 cru | False | 7.5257 | 0.5035 | 0.8368 |
| C | 6 | Q0.95 ajustado | True | 9.0824 | 0.5833 | 0.9167 |
| C | 6 | Q0.9 ajustado | True | 7.4202 | 0.4861 | 0.8194 |


## Energia de TI modelada (servidor substituto)

- servidor: Huawei Technologies Co., Ltd Fusion Server 2288H V5 (power_ssj2008-20170621-00759, publicado em 2017-07-11); P_idle = 49.6 W, P_busy = 461 W.
- O trace não fornece potência nem energia medidas. A potência obtida da CPU agregada é a de um servidor substituto, não a potência total da frota; a soma das máquinas presentes está em aggregate_effect.csv como coluna separada.
- Diferença entre energia de TI modelada com a trajetória prevista e com a trajetória observada, no mesmo servidor substituto. Não é erro contra medidor.

| model | r | horizon_steps | n_complete | total_reference_wh | total_projected_wh | mean_absolute_error_wh | mean_absolute_relative_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 1.000 | 1 | 288 | 5342.669 | 5340.105 | 0.574 | 0.031 |
| persistence | 1.000 | 3 | 288 | 16037.038 | 16020.316 | 3.004 | 0.054 |
| persistence | 1.000 | 6 | 288 | 32102.048 | 32040.632 | 7.034 | 0.063 |
| statespace | 1.000 | 1 | 288 | 5342.669 | 5338.413 | 0.559 | 0.030 |
| statespace | 1.000 | 3 | 288 | 16037.038 | 16011.852 | 2.787 | 0.050 |
| statespace | 1.000 | 6 | 288 | 32102.048 | 32022.011 | 6.480 | 0.058 |
| ridge_v2 | 1.000 | 1 | 288 | 5342.669 | 5301.969 | 0.563 | 0.030 |
| ridge_v2 | 1.000 | 3 | 288 | 16037.038 | 15745.351 | 2.819 | 0.050 |
| ridge_v2 | 1.000 | 6 | 288 | 32102.048 | 31206.218 | 6.588 | 0.058 |
| xgboost_v2 | 1.000 | 1 | 288 | 5342.669 | 5308.138 | 0.525 | 0.028 |
| xgboost_v2 | 1.000 | 3 | 288 | 16037.038 | 15823.529 | 2.421 | 0.043 |
| xgboost_v2 | 1.000 | 6 | 288 | 32102.048 | 31517.584 | 5.138 | 0.046 |
| xgboost_delta | 1.000 | 1 | 288 | 5342.669 | 5315.185 | 0.529 | 0.029 |
| xgboost_delta | 1.000 | 3 | 288 | 16037.038 | 15817.813 | 2.506 | 0.045 |
| xgboost_delta | 1.000 | 6 | 288 | 32102.048 | 31474.072 | 5.548 | 0.049 |
| hybrid | 1.000 | 1 | 288 | 5342.669 | 5315.450 | 0.519 | 0.028 |
| hybrid | 1.000 | 3 | 288 | 16037.038 | 15842.276 | 2.450 | 0.044 |
| hybrid | 1.000 | 6 | 288 | 32102.048 | 31532.118 | 5.477 | 0.049 |
| dlinear | 1.000 | 1 | 288 | 5342.669 | 5314.494 | 0.554 | 0.030 |
| dlinear | 1.000 | 3 | 288 | 16037.038 | 15791.265 | 2.761 | 0.049 |
| dlinear | 1.000 | 6 | 288 | 32102.048 | 31321.783 | 6.577 | 0.058 |
| tcn | 1.000 | 1 | 288 | 5342.669 | 5279.979 | 0.644 | 0.034 |
| tcn | 1.000 | 3 | 288 | 16037.038 | 15710.756 | 3.080 | 0.054 |
| tcn | 1.000 | 6 | 288 | 32102.048 | 31281.607 | 6.910 | 0.061 |
| nbeatsx | 1.000 | 1 | 288 | 5342.669 | 5230.687 | 0.829 | 0.044 |
| nbeatsx | 1.000 | 3 | 288 | 16037.038 | 15445.122 | 3.879 | 0.068 |
| nbeatsx | 1.000 | 6 | 288 | 32102.048 | 30660.593 | 8.269 | 0.073 |
| persistence | 1.400 | 1 | 288 | 6530.201 | 6527.313 | 0.577 | 0.026 |
| persistence | 1.400 | 3 | 288 | 19600.558 | 19581.938 | 3.005 | 0.044 |
| persistence | 1.400 | 6 | 288 | 39232.069 | 39163.877 | 7.032 | 0.052 |
| statespace | 1.400 | 1 | 288 | 6530.201 | 6526.837 | 0.561 | 0.025 |
| statespace | 1.400 | 3 | 288 | 19600.558 | 19579.559 | 2.790 | 0.041 |
| statespace | 1.400 | 6 | 288 | 39232.069 | 39158.641 | 6.487 | 0.048 |
| ridge_v2 | 1.400 | 1 | 288 | 6530.201 | 6492.470 | 0.567 | 0.025 |
| ridge_v2 | 1.400 | 3 | 288 | 19600.558 | 19326.509 | 2.832 | 0.041 |
| ridge_v2 | 1.400 | 6 | 288 | 39232.069 | 38375.275 | 6.620 | 0.048 |
| xgboost_v2 | 1.400 | 1 | 288 | 6530.201 | 6498.922 | 0.528 | 0.023 |
| xgboost_v2 | 1.400 | 3 | 288 | 19600.558 | 19404.476 | 2.429 | 0.036 |
| xgboost_v2 | 1.400 | 6 | 288 | 39232.069 | 38683.209 | 5.169 | 0.038 |
| xgboost_delta | 1.400 | 1 | 288 | 6530.201 | 6504.926 | 0.533 | 0.024 |
| xgboost_delta | 1.400 | 3 | 288 | 19600.558 | 19396.495 | 2.514 | 0.037 |
| xgboost_delta | 1.400 | 6 | 288 | 39232.069 | 38639.057 | 5.563 | 0.041 |
| hybrid | 1.400 | 1 | 288 | 6530.201 | 6505.550 | 0.522 | 0.023 |
| hybrid | 1.400 | 3 | 288 | 19600.558 | 19421.797 | 2.458 | 0.036 |
| hybrid | 1.400 | 6 | 288 | 39232.069 | 38697.522 | 5.497 | 0.040 |
| dlinear | 1.400 | 1 | 288 | 6530.201 | 6504.599 | 0.558 | 0.025 |
| dlinear | 1.400 | 3 | 288 | 19600.558 | 19372.127 | 2.772 | 0.041 |
| dlinear | 1.400 | 6 | 288 | 39232.069 | 38489.672 | 6.608 | 0.048 |
| tcn | 1.400 | 1 | 288 | 6530.201 | 6469.516 | 0.649 | 0.029 |
| tcn | 1.400 | 3 | 288 | 19600.558 | 19287.164 | 3.094 | 0.045 |
| tcn | 1.400 | 6 | 288 | 39232.069 | 38439.381 | 6.945 | 0.050 |
| nbeatsx | 1.400 | 1 | 288 | 6530.201 | 6417.976 | 0.840 | 0.037 |
| nbeatsx | 1.400 | 3 | 288 | 19600.558 | 19005.664 | 3.934 | 0.057 |
| nbeatsx | 1.400 | 6 | 288 | 39232.069 | 37783.350 | 8.389 | 0.061 |
| persistence | 1.800 | 1 | 288 | 7366.475 | 7363.150 | 0.623 | 0.025 |
| persistence | 1.800 | 3 | 288 | 22110.784 | 22089.449 | 3.231 | 0.043 |
| persistence | 1.800 | 6 | 288 | 44256.967 | 44178.898 | 7.561 | 0.050 |
| statespace | 1.800 | 1 | 288 | 7366.475 | 7363.587 | 0.606 | 0.024 |
| statespace | 1.800 | 3 | 288 | 22110.784 | 22091.634 | 3.002 | 0.040 |
| statespace | 1.800 | 6 | 288 | 44256.967 | 44183.705 | 6.983 | 0.046 |
| ridge_v2 | 1.800 | 1 | 288 | 7366.475 | 7328.029 | 0.613 | 0.024 |
| ridge_v2 | 1.800 | 3 | 288 | 22110.784 | 21828.455 | 3.058 | 0.040 |
| ridge_v2 | 1.800 | 6 | 288 | 44256.967 | 43362.097 | 7.156 | 0.046 |
| xgboost_v2 | 1.800 | 1 | 288 | 7366.475 | 7335.311 | 0.571 | 0.023 |
| xgboost_v2 | 1.800 | 3 | 288 | 22110.784 | 21912.777 | 2.622 | 0.034 |
| xgboost_v2 | 1.800 | 6 | 288 | 44256.967 | 43691.960 | 5.590 | 0.036 |
| xgboost_delta | 1.800 | 1 | 288 | 7366.475 | 7340.900 | 0.577 | 0.023 |
| xgboost_delta | 1.800 | 3 | 288 | 22110.784 | 21902.233 | 2.711 | 0.035 |
| xgboost_delta | 1.800 | 6 | 288 | 44256.967 | 43643.803 | 6.004 | 0.039 |
| hybrid | 1.800 | 1 | 288 | 7366.475 | 7341.868 | 0.565 | 0.022 |
| hybrid | 1.800 | 3 | 288 | 22110.784 | 21930.267 | 2.653 | 0.035 |
| hybrid | 1.800 | 6 | 288 | 44256.967 | 43707.296 | 5.935 | 0.039 |
| dlinear | 1.800 | 1 | 288 | 7366.475 | 7340.864 | 0.603 | 0.024 |
| dlinear | 1.800 | 3 | 288 | 22110.784 | 21877.632 | 2.993 | 0.039 |
| dlinear | 1.800 | 6 | 288 | 44256.967 | 43485.009 | 7.139 | 0.046 |
| tcn | 1.800 | 1 | 288 | 7366.475 | 7302.490 | 0.702 | 0.028 |
| tcn | 1.800 | 3 | 288 | 22110.784 | 21782.218 | 3.343 | 0.043 |
| tcn | 1.800 | 6 | 288 | 44256.967 | 43422.354 | 7.510 | 0.048 |
| nbeatsx | 1.800 | 1 | 288 | 7366.475 | 7245.203 | 0.914 | 0.036 |
| nbeatsx | 1.800 | 3 | 288 | 22110.784 | 21466.812 | 4.280 | 0.055 |
| nbeatsx | 1.800 | 6 | 288 | 44256.967 | 42688.959 | 9.133 | 0.059 |


### P(média) contra média de P (efeito de agregação)

| r | mean_p_of_mean_w | mean_mean_of_p_w | mean_jensen_gap_w | max_abs_jensen_gap_w | mean_sum_of_p_w | unstable_bins |
| --- | --- | --- | --- | --- | --- | --- |
| 1.000 | 205.903 | 205.903 | 0.000 | 0.000 | 772609.325 | 0 |
| 1.400 | 254.325 | 251.164 | -3.161 | 15.236 | 941934.013 | 0 |
| 1.800 | 287.049 | 282.010 | -5.039 | 27.808 | 1057493.133 | 0 |


## Figuras

- figures/observed_vs_predicted_A.png
- figures/mae_by_horizon_A.png
- figures/residual_histograms.png
- figures/fan_power_curves.png
- figures/energy_reference_vs_projected.png
- figures/energy_error_vs_r.png
