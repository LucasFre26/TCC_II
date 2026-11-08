# Resultados V2 (cohort)

Série de utilização de CPU do Alibaba Cluster Trace v2018 (machine_usage). Não há medição elétrica; potência, energia e PUE não foram calculados nesta versão.

## Teste final (avaliado uma vez)

| model | config | horizon_seconds | split | n | mae | rmse | mase | skill_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | - | 300.0 | test_final | 288 | 3.3605 | 4.4604 | 1.0000 | 0.0000 |
| ridge | A | 300.0 | test_final | 288 | 3.2648 | 4.3132 | 0.9715 | 0.0330 |
| ridge | B | 300.0 | test_final | 288 | 3.2350 | 4.2387 | 0.9627 | 0.0497 |
| ridge | C | 300.0 | test_final | 288 | 3.3336 | 4.3625 | 0.9920 | 0.0219 |
| xgboost | A | 300.0 | test_final | 288 | 3.0448 | 4.0287 | 0.9061 | 0.0968 |
| xgboost | B | 300.0 | test_final | 288 | 3.0735 | 4.0518 | 0.9146 | 0.0916 |
| xgboost | C | 300.0 | test_final | 288 | 3.0493 | 4.0309 | 0.9074 | 0.0963 |
| persistence | - | 900.0 | test_final | 288 | 4.6880 | 6.1744 | 1.0000 | 0.0000 |
| ridge | A | 900.0 | test_final | 288 | 4.4604 | 5.7759 | 0.9515 | 0.0645 |
| ridge | B | 900.0 | test_final | 288 | 4.2418 | 5.4812 | 0.9048 | 0.1123 |
| ridge | C | 900.0 | test_final | 288 | 4.5845 | 5.8221 | 0.9779 | 0.0571 |
| xgboost | A | 900.0 | test_final | 288 | 3.9590 | 5.0687 | 0.8445 | 0.1791 |
| xgboost | B | 900.0 | test_final | 288 | 4.0090 | 5.1583 | 0.8552 | 0.1646 |
| xgboost | C | 900.0 | test_final | 288 | 3.8907 | 5.0096 | 0.8299 | 0.1887 |
| persistence | - | 1800.0 | test_final | 288 | 5.2907 | 7.1172 | 1.0000 | 0.0000 |
| ridge | A | 1800.0 | test_final | 288 | 5.1322 | 6.6513 | 0.9700 | 0.0654 |
| ridge | B | 1800.0 | test_final | 288 | 4.6787 | 6.0955 | 0.8843 | 0.1435 |
| ridge | C | 1800.0 | test_final | 288 | 5.1448 | 6.5505 | 0.9724 | 0.0796 |
| xgboost | A | 1800.0 | test_final | 288 | 4.3084 | 5.6683 | 0.8143 | 0.2036 |
| xgboost | B | 1800.0 | test_final | 288 | 4.0097 | 5.2236 | 0.7579 | 0.2661 |
| xgboost | C | 1800.0 | test_final | 288 | 4.1229 | 5.2720 | 0.7793 | 0.2592 |

## Média entre folds de desenvolvimento

| model | config | horizon_seconds | split | n | mae | rmse | mase | skill_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | - | 300.0 | aggregate_mean | 720 | 3.0054 | 3.8760 | 1.0000 | 0.0000 |
| ridge | A | 300.0 | aggregate_mean | 720 | 2.7601 | 3.5780 | 0.9178 | 0.0756 |
| ridge | B | 300.0 | aggregate_mean | 720 | 2.8261 | 3.6145 | 0.9430 | 0.0618 |
| ridge | C | 300.0 | aggregate_mean | 720 | 2.8834 | 3.6920 | 0.9649 | 0.0401 |
| xgboost | A | 300.0 | aggregate_mean | 720 | 2.7241 | 3.5289 | 0.9185 | 0.0771 |
| xgboost | B | 300.0 | aggregate_mean | 720 | 2.7363 | 3.5630 | 0.9225 | 0.0678 |
| xgboost | C | 300.0 | aggregate_mean | 720 | 2.8071 | 3.6260 | 0.9493 | 0.0498 |
| persistence | - | 900.0 | aggregate_mean | 720 | 3.9791 | 5.1495 | 1.0000 | 0.0000 |
| ridge | A | 900.0 | aggregate_mean | 720 | 3.5119 | 4.5503 | 0.8943 | 0.1095 |
| ridge | B | 900.0 | aggregate_mean | 720 | 3.6694 | 4.6713 | 0.9477 | 0.0742 |
| ridge | C | 900.0 | aggregate_mean | 720 | 3.7463 | 4.7674 | 0.9616 | 0.0571 |
| xgboost | A | 900.0 | aggregate_mean | 720 | 3.4475 | 4.3807 | 0.8971 | 0.1208 |
| xgboost | B | 900.0 | aggregate_mean | 720 | 3.5128 | 4.4411 | 0.9211 | 0.1032 |
| xgboost | C | 900.0 | aggregate_mean | 720 | 3.7164 | 4.6373 | 0.9871 | 0.0504 |
| persistence | - | 1800.0 | aggregate_mean | 720 | 4.5385 | 6.0057 | 1.0000 | 0.0000 |
| ridge | A | 1800.0 | aggregate_mean | 720 | 3.8703 | 5.0853 | 0.8646 | 0.1485 |
| ridge | B | 1800.0 | aggregate_mean | 720 | 4.0536 | 5.2075 | 0.9187 | 0.1149 |
| ridge | C | 1800.0 | aggregate_mean | 720 | 4.1867 | 5.3149 | 0.9372 | 0.1030 |
| xgboost | A | 1800.0 | aggregate_mean | 720 | 3.8056 | 4.8655 | 0.8654 | 0.1651 |
| xgboost | B | 1800.0 | aggregate_mean | 720 | 3.8370 | 4.8820 | 0.8739 | 0.1619 |
| xgboost | C | 1800.0 | aggregate_mean | 720 | 4.2214 | 5.2140 | 0.9703 | 0.0908 |
