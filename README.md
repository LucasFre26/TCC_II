# TCC — predição de carga e estimativa de energia em datacenters

Implementação experimental do TCC **Predição Energética em Datacenters utilizando
Inteligência Artificial: Uma Abordagem Baseada em Modelagem de Consumo**. O projeto
avalia previsões de curto prazo da utilização de CPU e, em etapas posteriores, usa
essas previsões para estimar a energia de TI por meio de modelos paramétricos de
potência.

> **Escopo e interpretação:** o Alibaba Cluster Trace utilizado não contém medições
> elétricas. Portanto, a energia apresentada é modelada, não medida; o PUE não é
> calculado. Os resultados da V3 Parte 2 são exploratórios: os dias de desenvolvimento
> já foram usados em análises anteriores e o dia 7 é um teste herdado, não uma
> confirmação independente. O projeto não controla equipamentos nem demonstra
> redução de consumo em um datacenter real.

## Experimentos

| Etapa | Dados e objetivo | O que produz / limites |
|---|---|---|
| **V1** | Trace local GenAI; previsão do *duty cycle* agregado de GPU. | Prova de conceito preliminar. Não prevê CPU nem estima potência, energia ou PUE. |
| **V2** | Alibaba Cluster Trace v2018; previsão de `cpu_util_percent` com validação temporal. | Linha de base com persistência, Ridge e XGBoost. Não estima energia. |
| **V3** | Extensão da previsão de CPU e avaliação de modelos de complexidade crescente. | Inclui uma estimativa paramétrica de energia de TI baseada em servidores substitutos do SPECpower; não mede energia nem calcula PUE. |
| **V3 Parte 2** | Protocolo por blocos diários; modelos hierárquicos e globais, regimes, picos, calibração e energia por grupo ou máquina. | Análises exploratórias e continuidade no dia 7; não é validação independente nem medição elétrica. |

A análise expandida da V3 organiza diagnósticos, comparações, métricas e figuras a
partir dos artefatos preservados, sem retreinar as redes. Os protocolos e limitações
estão descritos em [`docs/`](docs/).

## Estrutura

```text
Tcc-sourcecode/
├── config/       # Configurações versionadas dos experimentos
├── data/         # Dados locais e caches (não incluir dados brutos no Git)
├── docs/         # Protocolos, pré-registro e decisões experimentais
├── paper/        # Artigo LaTeX, bibliografia, figuras e PDF
├── results/      # Resultados preservados, relatórios e figuras
├── src/          # Pacote Python energy_prediction
└── tests/        # Testes unitários, sintéticos e de integridade
```

## Requisitos e instalação

- Python **3.11 ou superior**.
- Dependências opcionais variam por etapa: `v3` instala PyTorch e Statsmodels;
  `analysis` instala Plotly para o relatório interativo.
- GPU NVIDIA é opcional. Com `--device auto`, as redes usam CUDA quando disponível;
  os testes sintéticos não exigem GPU.

Na raiz de `Tcc-sourcecode`, crie e ative um ambiente virtual.

**Windows / PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências para executar todos os experimentos e testes:

```bash
python -m pip install -e ".[test,v3,analysis]"
python -m pytest
```

Para executar somente a V1 e os testes básicos, `python -m pip install -e ".[test]"`
é suficiente. O comando de testes usa dados sintéticos, exceto nos casos marcados
como dependentes de CUDA.

## Executar

Execute os comandos a partir da raiz desta pasta. Os caminhos das configurações e
dos artefatos são definidos em relação ao projeto; não é necessário mudar para
`src/`.

### Verificação rápida

O modo *smoke* da V3 Parte 2 usa dados sintéticos e uma família pequena, sem exigir o
Alibaba v2018:

```bash
python -m energy_prediction.experiments.run_v3p2 --config config/v3p2-smoke.yaml
```

### Experimentos

```bash
# V1: exige os arquivos GenAI listados em config/v1.yaml
python -m energy_prediction.experiments.run_v1 --config config/v1.yaml

# V2: previsão de CPU no Alibaba v2018
python -m energy_prediction.experiments.run_v2 --config config/v2.yaml

# V3: smoke sintético ou experimento completo
python -m energy_prediction.experiments.run_v3 --config config/v3-smoke.yaml --family integrity
python -m energy_prediction.experiments.run_v3 --config config/v3.yaml

# V3 Parte 2: smoke, frota real reduzida ou corrida completa
python -m energy_prediction.experiments.run_v3p2 --config config/v3p2-smoke.yaml
python -m energy_prediction.experiments.run_v3p2 --config config/v3p2-smoke-real.yaml
python -m energy_prediction.experiments.run_v3p2 --config config/v3p2.yaml
```

As corridas completas podem exigir bastante tempo, memória e espaço em disco. A V3 e
a V3 Parte 2 aceitam `--device auto|cuda|cpu`; a V3 Parte 2 também aceita `--no-amp`
para executar as redes em FP32. Confira cada arquivo em [`config/`](config/) antes
de iniciar uma corrida.

### Análises e relatórios

```bash
# Análise expandida da V3
python -m energy_prediction.analysis.run --config config/v3-analysis.yaml

# Relatório de uma execução já existente
python -m energy_prediction.experiments.v3_report --run results/v3/<run_id>
python -m energy_prediction.experiments.v3p2_report --run results/v3-part2/<run_id>
```

O relatório da análise expandida é gravado em
`results/v3/<run_id>/expanded-analysis/`. Relatórios de cada corrida ficam dentro
do respectivo diretório em `results/`.

## Dados

Os dados brutos não são distribuídos neste repositório. Obtenha-os das fontes
originais e coloque-os nos caminhos definidos nas configurações:

- **V1:** `pod_gpu_duty_cycle_anon.tar.gz` e `pod_memory_util_anon.tar.gz`, em
  `data/clusterdata-master/cluster-trace-v2026-GenAI/`. A leitura é feita diretamente
  nos arquivos compactados.
- **V2, V3 e V3 Parte 2:** `machine_usage.tar.gz` e `machine_meta.tar.gz`, em
  `data/alibaba-v2018/`. `machine_usage.tar.gz` tem aproximadamente 1,7 GB. A
  execução cria caches locais em `data/processed/`.

Consulte [`config/v1.yaml`](config/v1.yaml),
[`config/v2.yaml`](config/v2.yaml) e os protocolos em [`docs/`](docs/) para
parâmetros, requisitos de entrada e decisões de cada experimento. Sem os arquivos
reais, use os testes e as configurações *smoke* sintéticas.

## Reprodutibilidade e resultados

- A semente padrão dos experimentos é 42; as configurações registram parâmetros,
  divisões temporais e modelos.
- Resultados existentes estão organizados em `results/v1/`, `results/v2/`,
  `results/v2-full/`, `results/v3/` e `results/v3-part2/`.
- As corridas gravam configurações efetivas, manifestos, métricas e previsões junto
  aos relatórios. Não interprete resultados sintéticos como validação em dados reais.
- A V3 Parte 2 confere o pré-registro e a integridade de artefatos anteriores. Para
  detalhes sobre os dias de desenvolvimento e o teste herdado, leia
  [`docs/v3p2-protocolo.md`](docs/v3p2-protocolo.md) e
  [`docs/v3p2-preregistro.md`](docs/v3p2-preregistro.md).

## Artigo

O artigo principal e os recursos necessários estão em [`paper/`](paper/). Para
compilá-lo, entre nessa pasta e execute `latexmk -pdf principal.tex`. As instruções
para regenerar as figuras experimentais estão em [`paper/README.md`](paper/README.md).
