# Prova de conceito V1

Previsão de curto prazo do duty cycle agregado de GPU no trace local GenAI. Os números produzidos são resultados preliminares. Esta série não é CPU do Alibaba, não há medição elétrica e o modelo de potência não foi aplicado. O pacote se chama `energy_prediction` por causa da estrutura prevista para o TCC, mas nesta V1 ele não calcula energia.

## Como reproduzir

No diretório `Tcc-sourcecode`, com Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m pytest
.venv/bin/python -m energy_prediction.experiments.run_v1 --config config/v1.yaml
```

A semente é 42, em `config/v1.yaml`. Os caminhos são resolvidos a partir de `Tcc-sourcecode`, não do diretório corrente.

## Entradas

- `data/clusterdata-master/cluster-trace-v2026-GenAI/pod_gpu_duty_cycle_anon.tar.gz`
- `data/clusterdata-master/cluster-trace-v2026-GenAI/pod_memory_util_anon.tar.gz`

A leitura é feita dentro dos arquivos compactados. Os brutos não são extraídos nem alterados.

## Saídas

- `results/v1/metrics.csv`
- `results/v1/summary.json`
- `results/v1/figures/gpu_series.png`
- `results/v1/figures/horizon1_observed_vs_predicted.png`
- `results/v1/tables/metrics.md`
- `results/v1/relatorio-preliminar.md`

## O que a V1 não faz

- não baixa o Alibaba Cluster Trace v2018;
- não prevê utilização de CPU;
- não aplica o modelo de potência de Fan, nem estima kWh ou PUE;
- não ajusta LSTM, TCN, XGBoost, Random Forest, Isolation Forest, K-Means ou rede neural;
- não modela cada contêiner;
- não usa `lora_request_trace.csv`, QPS, fila, jobs spot, pods do v2023 ou DLRM;
- não reescreve o artigo LaTeX nem a bibliografia.

---

# V2: previsão de CPU no Alibaba Cluster Trace v2018

A V2 evolui o protocolo da V1 para o experimento descrito na metodologia do artigo: prever `cpu_util_percent` futuro de `machine_usage`, com memória, rede e disco como exógenas. **Ela não calcula potência, energia nem PUE e não ajusta LSTM/TCN.** Os resultados são de protocolo; enquanto o v2018 não for executado, nada em `results/v2/` deve ser lido como resultado do TCC. Detalhes, decisões pendentes e limitações: [`docs/v2-protocolo.md`](docs/v2-protocolo.md).

## Como reproduzir

```bash
.venv/bin/python -m pytest                       # testes sintéticos; não precisam do v2018
.venv/bin/python -m energy_prediction.experiments.run_v2 --config config/v2.yaml
```

A execução real exige `machine_usage.tar.gz` (~1,7 GB) e `machine_meta.tar.gz` em `data/alibaba-v2018/` (baixar com `data/clusterdata-master/cluster-trace-v2018/fetchData.sh`). Esses arquivos e o cache `data/processed/` não entram no git. Sem eles, a V2 termina com erro explícito e não grava resultado.

## Diferenças em relação à V1

| Eixo | V1 | V2 |
|---|---|---|
| Alvo | duty cycle de GPU (GenAI) | `cpu_util_percent` (Alibaba v2018) |
| Agregação | média simples por instante | média ponderada por `cpu_num` + coorte homogênea |
| Informação na origem | persistência via `u(t)`, modelos sem `u(t)` | ambos com `cpu_t` (verificado por teste) |
| Validação | um corte, sem purga entre treino e validação | rolling-origin com purga em todo limite e teste final isolado |
| Modelos | persistência, média móvel, Ridge, HGB | persistência, Ridge, XGBoost |
| Ablação | GPU com/sem memória | A: CPU; B: CPU+memória; C: CPU+memória+rede+disco |
| Saída | `results/v1/` | `results/v2/{main,cohort}/` |

A V1 continua executável com `config/v1.yaml`. Um teste de regressão confere que ela reproduz `results/v1/metrics.csv`.

## Pipeline

```text
machine_usage.tar.gz --[streaming, uma passagem]--> grade (máquina, intervalo)  -> cache data/processed/
machine_meta.tar.gz  --> cpu_num, status
        -> agregação (ponderada por cpu_num | coorte homogênea) -> série de trabalho + cobertura
        -> features causais (cpu_t, lags, janelas, exógenas, seno/cosseno de 24 h do trace)
        -> folds expansivos + purga + teste final -> persistência | Ridge | XGBoost x {A, B, C}
        -> métricas por fold, horizonte, modelo e configuração -> results/v2/
```

- **Granularidade:** detectada nos dados (passo predominante por máquina). Nada assume 5 minutos. Horizontes, lags e janelas são declarados em tempo (`15min`) e convertidos em passos; múltiplo inexato é erro.
- **Calendário:** seno/cosseno de `time_stamp` módulo 86 400 s. É a posição no ciclo de 24 h do trace, não hora civil.
- **Persistência:** `u_hat(t+h) = u(t)`. A coluna `cpu_t` faz parte de todas as configurações de ablação.
- **Purga:** pelo menos o maior horizonte mais a maior janela, em todo limite (treino/validação e desenvolvimento/teste).
- **Seleção:** alpha do Ridge e grade do XGBoost só pelo MAE de validação dos folds. O teste final é avaliado uma vez, com refit na região de desenvolvimento.
- **Métricas:** MAE, RMSE, MASE (MAE do modelo sobre o MAE da persistência no mesmo horizonte e nas mesmas origens) e skill score (1 − RMSE/RMSE da persistência), por fold, média e desvio entre folds e teste final.

## Saídas (`results/v2/`)

`config_effective.yaml`, `dataset_profile.json`, `run_summary.json` e, por análise (`main`, `cohort`): `metrics.csv`, `metrics_summary.md`, `folds.json`, `predictions.csv`, `selections.json`, `series_profile.json` e `figures/` (real vs previsto, comparação por horizonte, comparação da ablação). A pasta é regravada por inteiro a cada execução bem-sucedida e nunca toca `results/v1/`.

## O que a V2 não faz

- não calcula potência, energia, PUE nem usa SPECpower;
- não ajusta LSTM, TCN, Random Forest ou outros modelos;
- não implementa block bootstrap nem SHAP (os erros por origem ficam em `predictions.csv` para isso);
- não usa `container_usage`.

# V3: experimentos finais e energia de TI modelada

Mesmas origens, horizontes (5, 15 e 30 min), folds e purga da análise principal da V2 (`results/v2-full/main`), com seleção de hiperparâmetros por validação interna dentro do treino de cada fold. O teste final de 24 h é o mesmo já inspecionado na V2.

## Como reproduzir

```bash
.venv/bin/python -m pip install -e ".[test,v3]"   # torch e statsmodels com versão fixada
.venv/bin/python -m pytest
.venv/bin/python -m energy_prediction.experiments.run_v3 --config config/v3-smoke.yaml --family integrity
.venv/bin/python -m energy_prediction.experiments.run_v3 --config config/v3.yaml                 # todas as famílias
.venv/bin/python -m energy_prediction.experiments.run_v3 --config config/v3.yaml --family residual --run-id <id>
.venv/bin/python -m energy_prediction.experiments.v3_report --run results/v3/<id>
```

`--device auto|cuda|cpu` escolhe onde as redes treinam; `auto` usa CUDA só se `torch.cuda.is_available()`. Modelos de árvore, filtro de Kalman, bootstrap e energia ficam em CPU.

## Famílias (`results/v3/<run_id>/<família>/`)

- `integrity`: auditoria da população (principal contra coorte), divisão interna e alinhamento das origens com a V2.
- `residual`: XGBoost no incremento de CPU, nível e tendência por filtro de Kalman (estado filtrado) e híbrido Kalman + XGBoost; contraste da configuração D.
- `neural`: DLinear, TCN causal e N-BEATSx em PyTorch, modo direto (tabela principal) e trajetória de seis passos (secundária); N-HiTS e PatchTST só se o portão pré-registrado for cumprido.
- `quantile`: XGBoost quantílico (Q50, Q90, Q95), pinball, picos e intervalo com teto ajustado na validação externa.
- `energy`: modelo de Fan num servidor substituto do SPECpower_ssj2008 (`config/specpower_substitute.yaml`), r em {1,0; 1,4; 1,8}, energia de referência e projetada, efeito de agregação e propagação do erro.
- `report/`: relatório de execução, limitações e figuras, gerados só a partir dos arquivos.

## O que a V3 não faz

- não mede potência nem energia: a energia é estimativa paramétrica num servidor substituto;
- não calcula PUE;
- não altera `results/v1`, `results/v2` nem `results/v2-full` (a saída nesses caminhos é recusada);
- não instala nem altera driver de GPU.

# Análise expandida da V3

Lê os artefatos preservados da V2 (`results/v2-full`) e de uma corrida da V3 e grava, sem alterar nenhum deles, uma análise completa em `results/v3/<run_id>/expanded-analysis/`. Nenhuma rede é treinada de novo; só o XGBoost, o Ridge, o Kalman e o híbrido são reajustados em CPU, com os hiperparâmetros gravados, para TreeSHAP e medição de tempo, e só depois de reproduzirem as previsões gravadas.

```bash
.venv/bin/python -m pip install -e ".[test,v3,analysis]"   # plotly para o HTML interativo
.venv/bin/python -m energy_prediction.analysis.run --config config/v3-analysis.yaml
```

A corrida e os limiares ficam em `config/v3-analysis.yaml`. Com `dataset.enabled: false`, ou sem o cache de grade em `data/processed/`, as seções de dataset, coorte reconstruída, TreeSHAP e tempo dos modelos clássicos são puladas e registradas no manifesto.

## Saídas (`expanded-analysis/`)

- `csv/`: métricas por split e fold, rankings, erro por origem e por regime, picos e eventos, bootstrap e comparações (com Holm), histórico das redes, complexidade e custo, ablação, TreeSHAP, quantis, potência e energia modeladas, sensibilidade, Jensen, perfil do dataset e diferença entre populações.
- `metadata/`: dicionário de dados com descrição, unidade e natureza de cada coluna (`data_dictionary.md`), catálogo de figuras e `analysis_manifest.json` (hashes de entrada e saída, versões, verificações de integridade e avisos).
- `figures/{png,svg,pdf}/` e `figures/data/`: figuras e os dados de cada uma.
- `interactive/relatorio.html`: relatório navegável e autocontido.
- `reports/scientific-analysis.md`: perguntas, hipóteses, resultados, ameaças à validade e conclusões.
- `latex/`: tabelas `.tex`, figuras em PDF, legendas sugeridas e proposta de reorganização da seção de resultados.

## Convenções

- **Erro de CPU:** observado − previsto; positivo indica subestimação.
- **Diferença de energia:** projetada − referência.
- **Tempo:** horas desde o início do trace, não horário civil.
- **Integridade:** a análise para se algum número gravado não for reproduzido, e confere o SHA-256 dos resultados preservados antes e depois.

# V3 Parte 2: protocolo por dia, previsão hierárquica, modelo global e energia da frota

Reavalia a previsão de CPU do Alibaba v2018 num protocolo cronológico por dia e testa hipóteses que a V3 deixou abertas: grupos de máquinas (H1), modelo global por máquina (H2), combinação por regime (H3), peso nas subidas (H4), calibração adaptativa (H5), energia por grupo e por máquina (H6) e robustez à mudança de nível (H8). O modelo pré-treinado (H7) não roda: o portão de proveniência falhou (detalhes em `docs/v3p2-preregistro.md`). Protocolo, decisões e limitações: [`docs/v3p2-protocolo.md`](docs/v3p2-protocolo.md).

## Como reproduzir

```bash
.venv/bin/python -m pip install -e ".[test,v3,analysis]"
.venv/bin/python -m pytest
.venv/bin/python -m energy_prediction.experiments.run_v3p2 --config config/v3p2-smoke.yaml        # sintético, sem GPU
.venv/bin/python -m energy_prediction.experiments.run_v3p2 --config config/v3p2-smoke-real.yaml   # frota real, grades reduzidas
.venv/bin/python -m energy_prediction.experiments.run_v3p2 --config config/v3p2.yaml              # corrida completa (exige o pré-registro)
.venv/bin/python -m energy_prediction.experiments.run_v3p2 --config config/v3p2.yaml --family hierarchical --run-id <id>
.venv/bin/python -m energy_prediction.experiments.v3p2_report --run results/v3-part2/<id>
```

`--device auto|cuda|cpu` escolhe onde as redes treinam (e o XGBoost global, quando o build tem CUDA); `--no-amp` desliga a precisão mista das redes para conferência em FP32. O painel por máquina é construído do cache de grade já usado pela V3 e gravado em `data/processed/panel_<chave>/` (fora do git).

## Protocolo

- Blocos de um dia: dias 3, 4, 5 e 6 são de desenvolvimento (exploratórios, porque os dias 4 a 6 já serviram de validação na V2 e na V3); o dia 7 (origens 2010–2297) é o teste herdado da V3, na quarta inspeção, reportado só como continuidade.
- Em cada bloco: treino até o início do bloco menos 18 passos, validação interna nas últimas 12 h do treino, reajuste e previsão única do bloco.
- Comparações pareadas com bootstrap por blocos estratificado por dia; decisão pré-registrada (IC e pelo menos 3 de 4 dias) e Holm sobre as 27 comparações primárias.
- Integridade: o painel reproduz a série agregada e o XGBoost da V2, reajustado no dia 7, reproduz as previsões preservadas.

## Famílias (`results/v3-part2/<run_id>/<família>/`)

- `integrity`: blocos, conferência do painel, população e reprodução da V2.
- `baselines`: persistência, XGBoost da V2, XGBoost de incremento, Kalman e híbrido no novo protocolo, com trajetórias de 1 a 6 passos.
- `robustness`: DLinear, TCN e PatchTST na formulação da V3 e com normalização por janela (3 sementes); XGBoost com peso de recência e sem calendário; tabela de deslocamento de nível × viés.
- `hierarchical`: grupos por média, volatilidade, perfil horário e domínio de falha (só com o treino de cada bloco), XGBoost por grupo, reconstrução e reconciliação (de baixo para cima, OLS, MinT com encolhimento).
- `global_model`: XGBoost global sobre máquinas amostradas, agregação ponderada, diagnóstico de escala, tamanho efetivo, previsões por máquina e TCN global sob portão.
- `regimes`: detector causal de regime e combinação de especialistas.
- `peaks`: XGBoost com peso nas subidas, compromisso entre erro médio e picos, antecipação e falso alarme.
- `calibration`: quantis Q05–Q95 e limites conformais estático, em janela móvel, ACI com retorno atrasado e SPCI (sob orçamento).
- `energy`: rotas de potência média, por grupo e por máquina, decomposição em agregação e previsão, sensibilidade a r e aos substitutos de `config/specpower_substitutes.yaml`, cenário heterogêneo sintético e incerteza da energia acumulada.
- `report`: comparações primárias com Holm global, relatório Markdown e HTML, figuras com dados, tabela LaTeX e dicionário de dados, gerados só a partir dos arquivos.

## O que a V3 Parte 2 não faz

- não mede potência nem energia e não calcula PUE: são estimativas paramétricas em servidores substitutos;
- não apresenta o dia 7 como confirmação independente;
- não altera `results/v1`, `results/v2`, `results/v2-full` nem `results/v3` (a saída nesses caminhos é recusada);
- não usa modelo pré-treinado nem `container_usage`.

## Artigo principal

O artigo LaTeX e seus arquivos necessários para compilação estão em [`paper/`](paper/).
O pacote inclui o PDF atualizado e as figuras citadas no texto. Para regenerar as
figuras a partir dos resultados, consulte [`paper/README.md`](paper/README.md).
