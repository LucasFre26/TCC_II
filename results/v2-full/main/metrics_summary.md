# Resultados V2 (main)

Série de utilização de CPU do Alibaba Cluster Trace v2018 (machine_usage). Não há medição elétrica; potência, energia e PUE não foram calculados nesta versão.

## Teste final (avaliado uma vez)

| model | config | horizon_seconds | split | n | mae | rmse | mase | skill_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | - | 300.0 | test_final | 288 | 3.3513 | 4.4648 | 1.0000 | 0.0000 |
| ridge | A | 300.0 | test_final | 288 | 3.2858 | 4.3325 | 0.9805 | 0.0296 |
| ridge | B | 300.0 | test_final | 288 | 3.2517 | 4.2681 | 0.9703 | 0.0441 |
| ridge | C | 300.0 | test_final | 288 | 3.3549 | 4.3369 | 1.0011 | 0.0286 |
| xgboost | A | 300.0 | test_final | 288 | 3.0641 | 4.0636 | 0.9143 | 0.0898 |
| xgboost | B | 300.0 | test_final | 288 | 3.0796 | 4.0439 | 0.9189 | 0.0943 |
| xgboost | C | 300.0 | test_final | 288 | 3.0958 | 4.0619 | 0.9238 | 0.0902 |
| persistence | - | 900.0 | test_final | 288 | 4.7018 | 6.1822 | 1.0000 | 0.0000 |
| ridge | A | 900.0 | test_final | 288 | 4.5060 | 5.8270 | 0.9584 | 0.0575 |
| ridge | B | 900.0 | test_final | 288 | 4.3334 | 5.5799 | 0.9217 | 0.0974 |
| ridge | C | 900.0 | test_final | 288 | 4.6908 | 5.8133 | 0.9977 | 0.0597 |
| xgboost | A | 900.0 | test_final | 288 | 3.8562 | 4.9875 | 0.8202 | 0.1932 |
| xgboost | B | 900.0 | test_final | 288 | 3.8782 | 4.9960 | 0.8248 | 0.1919 |
| xgboost | C | 900.0 | test_final | 288 | 3.8460 | 4.9968 | 0.8180 | 0.1917 |
| persistence | - | 1800.0 | test_final | 288 | 5.3302 | 7.1495 | 1.0000 | 0.0000 |
| ridge | A | 1800.0 | test_final | 288 | 5.2195 | 6.7329 | 0.9792 | 0.0583 |
| ridge | B | 1800.0 | test_final | 288 | 4.8051 | 6.2253 | 0.9015 | 0.1293 |
| ridge | C | 1800.0 | test_final | 288 | 5.1699 | 6.5397 | 0.9699 | 0.0853 |
| xgboost | A | 1800.0 | test_final | 288 | 4.3336 | 5.6967 | 0.8130 | 0.2032 |
| xgboost | B | 1800.0 | test_final | 288 | 4.4006 | 5.7760 | 0.8256 | 0.1921 |
| xgboost | C | 1800.0 | test_final | 288 | 4.2113 | 5.4332 | 0.7901 | 0.2401 |

## Média entre folds de desenvolvimento

| model | config | horizon_seconds | split | n | mae | rmse | mase | skill_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | - | 300.0 | aggregate_mean | 720 | 3.0414 | 3.9176 | 1.0000 | 0.0000 |
| ridge | A | 300.0 | aggregate_mean | 720 | 2.7884 | 3.6146 | 0.9159 | 0.0764 |
| ridge | B | 300.0 | aggregate_mean | 720 | 2.8478 | 3.6496 | 0.9385 | 0.0632 |
| ridge | C | 300.0 | aggregate_mean | 720 | 2.8941 | 3.6953 | 0.9559 | 0.0506 |
| xgboost | A | 300.0 | aggregate_mean | 720 | 2.7551 | 3.5696 | 0.9154 | 0.0787 |
| xgboost | B | 300.0 | aggregate_mean | 720 | 2.7745 | 3.6213 | 0.9202 | 0.0676 |
| xgboost | C | 300.0 | aggregate_mean | 720 | 2.8208 | 3.6369 | 0.9421 | 0.0587 |
| persistence | - | 900.0 | aggregate_mean | 720 | 4.0401 | 5.2262 | 1.0000 | 0.0000 |
| ridge | A | 900.0 | aggregate_mean | 720 | 3.5350 | 4.5944 | 0.8862 | 0.1143 |
| ridge | B | 900.0 | aggregate_mean | 720 | 3.6943 | 4.7147 | 0.9387 | 0.0804 |
| ridge | C | 900.0 | aggregate_mean | 720 | 3.7135 | 4.7037 | 0.9467 | 0.0789 |
| xgboost | A | 900.0 | aggregate_mean | 720 | 3.4901 | 4.4647 | 0.8966 | 0.1147 |
| xgboost | B | 900.0 | aggregate_mean | 720 | 3.5620 | 4.4903 | 0.9199 | 0.1047 |
| xgboost | C | 900.0 | aggregate_mean | 720 | 3.7345 | 4.6937 | 0.9794 | 0.0553 |
| persistence | - | 1800.0 | aggregate_mean | 720 | 4.5702 | 6.0692 | 1.0000 | 0.0000 |
| ridge | A | 1800.0 | aggregate_mean | 720 | 3.8735 | 5.1067 | 0.8593 | 0.1545 |
| ridge | B | 1800.0 | aggregate_mean | 720 | 4.0752 | 5.2432 | 0.9167 | 0.1187 |
| ridge | C | 1800.0 | aggregate_mean | 720 | 4.0954 | 5.2337 | 0.9265 | 0.1187 |
| xgboost | A | 1800.0 | aggregate_mean | 720 | 3.8006 | 4.9535 | 0.8621 | 0.1519 |
| xgboost | B | 1800.0 | aggregate_mean | 720 | 3.8697 | 4.9498 | 0.8759 | 0.1582 |
| xgboost | C | 1800.0 | aggregate_mean | 720 | 4.5428 | 5.6932 | 1.0624 | -0.0016 |
