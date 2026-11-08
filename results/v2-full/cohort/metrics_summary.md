# Resultados V2 (cohort)

Série de utilização de CPU do Alibaba Cluster Trace v2018 (machine_usage). Não há medição elétrica; potência, energia e PUE não foram calculados nesta versão.

## Teste final (avaliado uma vez)

| model | config | horizon_seconds | split | n | mae | rmse | mase | skill_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | - | 300.0 | test_final | 288 | 3.3513 | 4.4648 | 1.0000 | 0.0000 |
| ridge | A | 300.0 | test_final | 288 | 3.2858 | 4.3324 | 0.9805 | 0.0296 |
| ridge | B | 300.0 | test_final | 288 | 3.2518 | 4.2684 | 0.9703 | 0.0440 |
| ridge | C | 300.0 | test_final | 288 | 3.3542 | 4.3362 | 1.0009 | 0.0288 |
| xgboost | A | 300.0 | test_final | 288 | 3.0517 | 4.0371 | 0.9106 | 0.0958 |
| xgboost | B | 300.0 | test_final | 288 | 3.0912 | 4.0816 | 0.9224 | 0.0858 |
| xgboost | C | 300.0 | test_final | 288 | 3.0822 | 4.0583 | 0.9197 | 0.0910 |
| persistence | - | 900.0 | test_final | 288 | 4.7018 | 6.1822 | 1.0000 | 0.0000 |
| ridge | A | 900.0 | test_final | 288 | 4.5058 | 5.8266 | 0.9583 | 0.0575 |
| ridge | B | 900.0 | test_final | 288 | 4.3340 | 5.5806 | 0.9218 | 0.0973 |
| ridge | C | 900.0 | test_final | 288 | 4.6890 | 5.8110 | 0.9973 | 0.0600 |
| xgboost | A | 900.0 | test_final | 288 | 3.8532 | 5.0285 | 0.8195 | 0.1866 |
| xgboost | B | 900.0 | test_final | 288 | 3.8796 | 5.0085 | 0.8251 | 0.1898 |
| xgboost | C | 900.0 | test_final | 288 | 3.8089 | 4.9241 | 0.8101 | 0.2035 |
| persistence | - | 1800.0 | test_final | 288 | 5.3302 | 7.1495 | 1.0000 | 0.0000 |
| ridge | A | 1800.0 | test_final | 288 | 5.2190 | 6.7322 | 0.9791 | 0.0584 |
| ridge | B | 1800.0 | test_final | 288 | 4.8059 | 6.2262 | 0.9016 | 0.1291 |
| ridge | C | 1800.0 | test_final | 288 | 5.1710 | 6.5410 | 0.9701 | 0.0851 |
| xgboost | A | 1800.0 | test_final | 288 | 4.2996 | 5.5838 | 0.8066 | 0.2190 |
| xgboost | B | 1800.0 | test_final | 288 | 4.3089 | 5.5728 | 0.8084 | 0.2205 |
| xgboost | C | 1800.0 | test_final | 288 | 4.2653 | 5.5152 | 0.8002 | 0.2286 |

## Média entre folds de desenvolvimento

| model | config | horizon_seconds | split | n | mae | rmse | mase | skill_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | - | 300.0 | aggregate_mean | 720 | 3.0418 | 3.9181 | 1.0000 | 0.0000 |
| ridge | A | 300.0 | aggregate_mean | 720 | 2.7888 | 3.6150 | 0.9159 | 0.0764 |
| ridge | B | 300.0 | aggregate_mean | 720 | 2.8486 | 3.6505 | 0.9386 | 0.0631 |
| ridge | C | 300.0 | aggregate_mean | 720 | 2.8964 | 3.6950 | 0.9567 | 0.0505 |
| xgboost | A | 300.0 | aggregate_mean | 720 | 2.7626 | 3.5789 | 0.9180 | 0.0762 |
| xgboost | B | 300.0 | aggregate_mean | 720 | 2.7859 | 3.6413 | 0.9271 | 0.0616 |
| xgboost | C | 300.0 | aggregate_mean | 720 | 2.8179 | 3.6493 | 0.9410 | 0.0551 |
| persistence | - | 900.0 | aggregate_mean | 720 | 4.0406 | 5.2270 | 1.0000 | 0.0000 |
| ridge | A | 900.0 | aggregate_mean | 720 | 3.5353 | 4.5948 | 0.8861 | 0.1144 |
| ridge | B | 900.0 | aggregate_mean | 720 | 3.6957 | 4.7162 | 0.9390 | 0.0801 |
| ridge | C | 900.0 | aggregate_mean | 720 | 3.7152 | 4.7050 | 0.9478 | 0.0781 |
| xgboost | A | 900.0 | aggregate_mean | 720 | 3.4865 | 4.4634 | 0.8968 | 0.1147 |
| xgboost | B | 900.0 | aggregate_mean | 720 | 3.5165 | 4.4968 | 0.9071 | 0.1042 |
| xgboost | C | 900.0 | aggregate_mean | 720 | 3.6666 | 4.6558 | 0.9523 | 0.0702 |
| persistence | - | 1800.0 | aggregate_mean | 720 | 4.5709 | 6.0701 | 1.0000 | 0.0000 |
| ridge | A | 1800.0 | aggregate_mean | 720 | 3.8736 | 5.1070 | 0.8592 | 0.1546 |
| ridge | B | 1800.0 | aggregate_mean | 720 | 4.0769 | 5.2451 | 0.9170 | 0.1185 |
| ridge | C | 1800.0 | aggregate_mean | 720 | 4.0926 | 5.2273 | 0.9261 | 0.1195 |
| xgboost | A | 1800.0 | aggregate_mean | 720 | 3.8072 | 4.8588 | 0.8603 | 0.1760 |
| xgboost | B | 1800.0 | aggregate_mean | 720 | 3.8551 | 4.9142 | 0.8698 | 0.1670 |
| xgboost | C | 1800.0 | aggregate_mean | 720 | 4.4477 | 5.5000 | 1.0268 | 0.0403 |
