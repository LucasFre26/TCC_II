# Resultados preliminares da V1

Os números em `results/v1/` são resultados preliminares de uma prova de conceito metodológica. Eles não são conclusão do TCC. A série não é utilização de CPU do Alibaba, não há medição elétrica e o modelo de potência não foi aplicado. O pacote não calcula energia.

## Dataset e motivo da escolha

As entradas são `pod_gpu_duty_cycle_anon.tar.gz` e `pod_memory_util_anon.tar.gz`, em `data/clusterdata-master/cluster-trace-v2026-GenAI/`. O Alibaba Cluster Trace v2018, série prevista no artigo para utilização de CPU, não está no disco. Esta série local é a única utilização temporal disponível e serve para ensaiar o protocolo de previsão: alvo futuro, corte temporal com purga, persistência, MAE, RMSE, MASE e skill score.

## Subset

A tabela de trabalho é a média aritmética, sem ponderar por capacidade, do duty cycle de GPU e da memória em cada `timestamp_anon`, restrita ao inner join dos dois agregados. Instantes presentes em só uma série ficam de fora e são contados. Zeros e valores fora do intervalo permanecem. O passo mediano precisa ser 57 segundos; outro passo interrompe a execução.

## Alvo

O alvo é o duty cycle agregado de GPU 1, 5 e 15 passos à frente. Cada horizonte tem modelo direto. A origem precisa ter os 15 passos completos para entrar na tabela.

## Features

Conhecidas na origem t, sem observação futura:

- defasagens de GPU em t-1, t-2, t-3, t-6 e t-12;
- média e desvio (ddof 0) de GPU nas janelas de 6 e 12 passos estritamente anteriores a t;
- memória agregada em t, t-1 e t-6;
- seno e cosseno da posição do instante na grade, não hora civil.

Os modelos não recebem o duty cycle em t. A persistência usa esse valor. Seno e cosseno entram nas duas variantes, para a ablação isolar a memória.

## Preprocessing

Ordenação por instante, rejeição de timestamp duplicado, inner join e padronização com média e desvio do treino de cada ajuste. A validação escolhe o alpha. O teste não entra nesse ajuste. A purga tem 27 origens, soma do horizonte 15 com o lookback 12, imediatamente antes do teste.

## Baselines e algoritmos

Baselines univariados, só na variante `gpu_only`: persistência do valor em t e média móvel dos 12 passos anteriores, a mesma previsão nos três horizontes.

Modelos, nas variantes `gpu_only` e `gpu_memory`: Ridge com alpha em {0,1; 1; 10}, escolhido pela MAE de validação, e `HistGradientBoostingRegressor` com profundidade 3, 100 iterações, taxa de aprendizado 0,1, `early_stopping=False` e semente 42. Nenhum outro algoritmo é ajustado.

## Métricas

MAE, RMSE, MASE e skill score no bloco de teste. MASE e skill score usam a persistência do mesmo horizonte e do mesmo teste. Persistência contra ela mesma tem MASE 1 e skill score 0.

## Limitações

A série cobre cerca de um dia, é dominada por zeros e não é utilização de CPU de servidor. Há um único corte temporal. A média não ponderada mistura contêineres ociosos e ativos. O passo de 57 segundos é o passo observado, não uma afirmação sobre o scrape original. Os arquivos de resultados estão em `results/v1/`.
