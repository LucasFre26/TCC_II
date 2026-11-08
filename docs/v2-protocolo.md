# Protocolo da V2

Documento técnico da change `v2-alibaba-cpu-forecasting`. Resultados ainda não existem para o Alibaba v2018 real: o código foi validado só com dados sintéticos. Nada aqui altera as conclusões do artigo.

## Objetivo

Repetir o ensaio de previsão da V1 no experimento que a metodologia do artigo define: prever a utilização futura de CPU de `machine_usage` (Alibaba Cluster Trace v2018) com memória, rede e disco como exógenas, comparando persistência, Ridge e XGBoost sob validação temporal robusta. A V2 não estima potência, energia nem PUE.

## Três níveis de dados

| Nível | Conteúdo | Uso |
|---|---|---|
| Protótipo / desenvolvimento local | arquivos sintéticos gerados em `tests/synthetic.py` | testes e desenvolvimento sem o v2018 |
| Dataset da V2 | `machine_usage` + `machine_meta` reais, com janela e amostra de máquinas configuráveis (`config/v2.yaml`) | primeira execução real |
| Experimento final do TCC | `machine_usage` completo, mais LSTM/TCN e modelo de potência | etapas posteriores, fora da V2 |

O trace GenAI da V1 não é dataset da V2.

## Pipeline e decisões

1. **Ingestão** (`data/alibaba.py`): leitura em streaming do `tar.gz`; CSV com ou sem cabeçalho (os oficiais não têm, vale a ordem de `schema.txt`); valores fora de [0, 100] viram ausentes (inclui `-1` e `101` de disco) e são contados por coluna; duplicatas (mesma máquina e instante) mantêm a primeira linha e são contadas; ordenação por (máquina, intervalo).
2. **Granularidade:** passo predominante das primeiras `sample_rows` linhas, com fração de conformidade. `grid_step: auto` usa esse passo e falha se não houver passo predominante; um valor numérico vale sempre, com aviso no perfil.
3. **Grade e cache:** cada máquina é reamostrada por média em intervalos da grade. O cache (`data/processed/grid_<chave>.csv.gz`) é chaveado pelo hash SHA-256 dos brutos e pelos parâmetros (passo, janela, máquinas).
4. **Agregação** (`preprocessing/cluster.py`): `Σ w·x / Σ w` sobre as máquinas com valor no intervalo, com `w = cpu_num`. O mesmo peso vale para as exógenas. A cobertura é a soma dos pesos presentes sobre o peso total considerado. Máquinas sem metadado são excluídas e contadas. Conflito de `cpu_num`: valor mais frequente, empate no menor.
5. **Coorte homogênea:** máquinas ativas (todos os registros de `status` anteriores ao fim da janela em `active_statuses`) com o mesmo `cpu_num`. Padrão: o mais comum entre as ativas.
6. **Features** (`features/cpu_features.py`): por grupo (`cpu`, `mem`, `net`, `disk`), valor em `t`, lags e média/desvio móveis terminando em `t`; seno/cosseno de `time_stamp mod 86400`. Origens com NaN em qualquer feature usada pela maior configuração (C) ou em algum alvo saem de todas as configurações, então A, B e C avaliam as mesmas origens. Não há imputação.
7. **Horizontes:** um alvo por horizonte (`y_h<passos>`), modelo independente por horizonte. Origens sem alvo completo para o maior horizonte saem de todos.
8. **Validação** (`validation/rolling_origin.py`): região de desenvolvimento, purga, teste final. Folds expansivos com validação contígua no fim do desenvolvimento e purga entre treino e validação. A purga mínima é o maior horizonte mais a maior janela, em passos. Tamanhos em passos de tempo (`origin_index`), não em posições, para a purga continuar em tempo quando há origens removidas.
9. **Modelos** (`models/registry.py`): persistência (`cpu_t`), Ridge (scaler ajustado só nos dados de `fit`) e XGBoost (`tree_method=hist`, semente e `n_jobs` da configuração, sem scaler).
10. **Protocolo** (`evaluation/protocol.py`): por horizonte, modelo e configuração, cada ponto da grade é ajustado por fold e avaliado na validação; vence o menor MAE médio (empate: primeiro da grade); refit na região de desenvolvimento; teste final uma única vez.
11. **Métricas:** `mae`, `rmse`, `mase`, `skill_score` reaproveitados da V1. Referência com erro zero produz valor indefinido (NaN), não exceção.

## Limitações conhecidas

- **Duplicatas entre blocos de leitura:** a deduplicação é por bloco (`chunk_rows`). Duplicatas separadas por mais de um bloco não são detectadas. O perfil informa a contagem detectada.
- **Métricas por fold com hiperparâmetro escolhido nos mesmos folds:** os números por fold são levemente otimistas para Ridge e XGBoost. O teste final não tem esse viés.
- **Purga conservadora:** somar horizonte e janela protege o alvo (só o horizonte é estritamente necessário). Mantida por fidelidade ao artigo.
- **Sazonalidade:** só diária (7 a 8 dias de trace).
- **Coorte:** a seleção usa informação do período inteiro (só metadados, não valores do alvo). Análise de sensibilidade, não de generalização.
- **Alvo em nível:** árvores não extrapolam nível; sob mudança de regime o XGBoost pode perder para a persistência. O alvo em delta é sensibilidade futura, não implementada.
- **Análise de erro (2026-10-21):** `error_extremes.csv` (erro máximo, P95, MAE e viés nos picos), `bootstrap_ci.csv` (IC 95% da diferença de erro absoluto por block bootstrap, blocos de 12 passos e sensibilidade 6/24) e `exogenous_lead.json` (correlação de nível e de antecedência das exógenas na série agregada). Calculados sobre `predictions.csv` do teste final; não alteram a seleção de modelos. Parâmetros no bloco `error_analysis` de `config/v2.yaml`. SHAP continua fora.

## Medidas no trace real (2026-09-23)

Leitura completa de `machine_usage.tar.gz` (hash SHA-256 conferido com o publicado): 246.934.820 linhas, 4.034 máquinas em `machine_meta`, 8 dias completos.

- Timestamps em múltiplos de 10 s, sem passo predominante (mediana 60 s, p95 270 s): `grid_step: auto` falha por desenho; adotado 300 s explícito.
- Linhas por dia: ~20 milhões nos dias 0 e 1, ~34 milhões nos dias 2 a 7. O 8º dia está completo.
- `mem_gps` e `mkpi`: valor em 20,7% das linhas; fora do grupo memória.
- Disco: 3.770 valores `-1`/`101` no subconjunto de 200 máquinas (0,03%).
- `cpu_num` = 96 em todas as máquinas; `status` = `USING` (5 registros `IMPORT_INSTALLING`).
- Arquivo ordenado por máquina, não por tempo. Leitura completa: ~200 s.

**Primeira execução (subconjunto de 200 máquinas, 8 dias, 2.304 passos de 5 min):** rodou de ponta a ponta em 366 s. Serviu de verificação de pipeline: as métricas por fold variam muito (skill do XGBoost, config C, 15 min: −0,39 a +0,24 entre folds), então o teste final sozinho não representa o desempenho.

**Execução na frota inteira (config `v2-full.yaml`, `results/v2-full`):** 246.934.820 linhas lidas, 4.023 máquinas agregadas (das 4.034; 11 sem metadado ou status ativo), série de 2.304 passos de 300 s, cobertura média 0,905. Rodou em ~888 s (semente 42). Resultado adotado no artigo: a ordenação de modelos (XGBoost < Ridge < persistência em MAE) e os valores de MAE coincidem com os das 200 máquinas a menos de alguns décimos de ponto percentual, o que mostra que o resultado não depende do tamanho da amostra. A reprodução das 200 máquinas neste ambiente bateu com `results/v2` a 10⁻¹² (ponto flutuante), confirmando a integridade do pipeline.

## Decisões

Resolvidas com dados: passo da grade (300 s), folds e teste final (5 x 12 h, 24 h), grupo memória (só `mem_util_percent`), status ativo (`USING`), cache em CSV comprimido, subconjunto vs frota inteira (adotada a frota inteira em `results/v2-full`; a amostra de 200 máquinas é apenas verificação de pipeline).

Abertas:

1. **Ponderação e coorte por `cpu_num` são degeneradas** (todas as máquinas têm 96): a coorte coincide com a população ativa e as métricas de `main` e `cohort` são idênticas. Decisão do autor: manter e declarar, coorte por `failure_domain` ou por cobertura completa.
2. Densidade de amostragem diferente nos dias 0 e 1.
3. MASE: a equação do artigo usa outro denominador (naive de 1 passo); texto e código usam a persistência do mesmo horizonte.
4. SHAP; alvo em delta. (Block bootstrap já executado no teste final da frota inteira.)
