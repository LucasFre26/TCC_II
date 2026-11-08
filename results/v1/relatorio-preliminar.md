# Resultados preliminares da V1

Estes são resultados preliminares. A série não é CPU do Alibaba, não é utilização de CPU do Alibaba, não é energia do Alibaba, não há medição elétrica, e o modelo de potência não foi aplicado. O pacote não calcula energia.

A comparação abaixo descreve esta execução. Ela não altera hipóteses nem conclusões do TCC.

## Subset

- Linhas brutas de GPU: 157417
- Linhas brutas de memória: 214890
- Contêineres de GPU: 143
- Contêineres de memória: 207
- Instantes após o inner join: 1441
- Instantes excluídos por existir em só uma série: 0
- Passo mediano observado: 57.0 s
- Fração de zeros nas linhas brutas de GPU: 0.668473
- Fração de zeros da GPU agregada: 0.073560
- Linhas de memória com valor negativo: 1
- Linhas de memória com par instante-contêiner repetido: 2
- Média do alvo no horizonte 1, treino / validação / teste: 10.323568 / 2.226880 / 1.549409
- Treino / validação / purga / teste: 884 / 221 / 27 / 282
- Duração da execução: 0.732 s
- Semente: 42

## Métricas no teste

| model | horizon | variant | mae | rmse | mase | skill_score | alpha |
| --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 1 | gpu_only | 0.528588 | 0.801084 | 1.000000 | 0.000000 |  |
| moving_average | 1 | gpu_only | 0.952057 | 1.324809 | 1.801134 | -0.653770 |  |
| ridge | 1 | gpu_only | 0.865827 | 1.091439 | 1.638001 | -0.362452 | 0.1 |
| hist_gradient_boosting | 1 | gpu_only | 1.823286 | 2.156455 | 3.449354 | -1.691920 |  |
| ridge | 1 | gpu_memory | 0.854959 | 1.105269 | 1.617440 | -0.379716 | 10.0 |
| hist_gradient_boosting | 1 | gpu_memory | 2.129754 | 2.504089 | 4.029141 | -2.125875 |  |
| persistence | 5 | gpu_only | 0.895120 | 1.359368 | 1.000000 | 0.000000 |  |
| moving_average | 5 | gpu_only | 1.272517 | 1.682212 | 1.421616 | -0.237496 |  |
| ridge | 5 | gpu_only | 1.411594 | 1.714619 | 1.576988 | -0.261336 | 10.0 |
| hist_gradient_boosting | 5 | gpu_only | 2.871503 | 3.389624 | 3.207953 | -1.493530 |  |
| ridge | 5 | gpu_memory | 1.509055 | 1.846419 | 1.685869 | -0.358292 | 10.0 |
| hist_gradient_boosting | 5 | gpu_memory | 3.366773 | 3.805703 | 3.761253 | -1.799613 |  |
| persistence | 15 | gpu_only | 1.586487 | 2.037162 | 1.000000 | 0.000000 |  |
| moving_average | 15 | gpu_only | 2.016085 | 2.252467 | 1.270785 | -0.105689 |  |
| ridge | 15 | gpu_only | 1.951572 | 2.349109 | 1.230122 | -0.153128 | 10.0 |
| hist_gradient_boosting | 15 | gpu_only | 2.824690 | 3.360078 | 1.780468 | -0.649392 |  |
| ridge | 15 | gpu_memory | 2.129877 | 2.600330 | 1.342511 | -0.276447 | 10.0 |
| hist_gradient_boosting | 15 | gpu_memory | 2.844102 | 3.386223 | 1.792704 | -0.662226 |  |


## Leitura descritiva

No horizonte 1, a menor MAE desta execução foi 0.528588 (persistence, variante gpu_only). A persistência teve MAE 0.528588 e MASE 1 por construção. No horizonte 5, a menor MAE desta execução foi 0.895120 (persistence, variante gpu_only). A persistência teve MAE 0.895120 e MASE 1 por construção. No horizonte 15, a menor MAE desta execução foi 1.586487 (persistence, variante gpu_only). A persistência teve MAE 1.586487 e MASE 1 por construção.
Os modelos não observam o duty cycle na origem; a persistência observa. A média do alvo no horizonte 1 cai entre treino e teste, então a vantagem da persistência descreve esta execução.

## Sanidade

- Maior correlação absoluta, no treino, entre feature e alvo do horizonte 1: 0.8719813077237554
- Linhas com skill score acima de 0,95: []
