# Previsão de Carga Computacional de Curto Prazo para Estimativa da Demanda Energética de TI em Datacenters: Um Estudo com o Alibaba Cluster Trace

Aplicacao Python para experimentos de previsao de utilizacao de GPU (V1) e CPU (V2). O repositorio contem o codigo, configuracoes e testes. Os datasets e resultados nao sao versionados.

## Requisitos

- Python 3.11 ou superior
- Git
- Acesso a internet para instalar as dependencias e, quando necessario, baixar os datasets

## Instalar

Abra um terminal nesta pasta e crie um ambiente virtual.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Se o comando `py -3.11` nao existir, instale Python 3.11+ e confirme que o launcher `py` esta habilitado.

### Linux ou macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Testes

Os testes usam dados sinteticos e nao baixam datasets. O teste de regressao da V1 e ignorado quando os arquivos reais de entrada e a saida historica nao existem.

```powershell
python -m pytest
```

O mesmo comando funciona em Linux e macOS. Para executar um teste especifico:

```powershell
python -m pytest tests/test_v2_run.py
```

## Executar a V1: GPU / GenAI

A V1 precisa de dois arquivos que nao estao neste repositorio:

- `data/clusterdata-master/cluster-trace-v2026-GenAI/pod_gpu_duty_cycle_anon.tar.gz`
- `data/clusterdata-master/cluster-trace-v2026-GenAI/pod_memory_util_anon.tar.gz`

Obtenha esses arquivos a partir da distribuicao do dataset GenTD26 e coloque-os exatamente nesses caminhos. O programa le os CSVs dentro dos arquivos compactados, sem extraí-los.

Com os arquivos presentes, execute na raiz do repositorio:

```powershell
python -m energy_prediction.experiments.run_v1 --config config/v1.yaml
```

Os resultados sao gravados em `results/v1/`.

## Executar a V2: CPU / Alibaba v2018

A V2 precisa de `machine_usage.tar.gz` e `machine_meta.tar.gz` do Alibaba Cluster Trace v2018. O arquivo de uso tem cerca de 1,7 GB. Baixe somente os dois arquivos necessarios; nao e preciso baixar `container_usage`, tarefas ou instancias.

### Windows (PowerShell)

```powershell
New-Item -ItemType Directory -Force data\alibaba-v2018 | Out-Null
Invoke-WebRequest -Uri https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz -OutFile data\alibaba-v2018\machine_usage.tar.gz
Invoke-WebRequest -Uri https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_meta.tar.gz -OutFile data\alibaba-v2018\machine_meta.tar.gz
python -m energy_prediction.experiments.run_v2 --config config/v2.yaml
```

### Linux ou macOS

```bash
mkdir -p data/alibaba-v2018
curl -fL https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz -o data/alibaba-v2018/machine_usage.tar.gz
curl -fL https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_meta.tar.gz -o data/alibaba-v2018/machine_meta.tar.gz
python -m energy_prediction.experiments.run_v2 --config config/v2.yaml
```

Os caminhos podem ser alterados em `config/v2.yaml`. A configuracao inicial usa uma amostra de 200 maquinas e uma grade de 5 minutos para desenvolvimento. Os dados processados ficam em `data/processed/` e os resultados em `results/v2/`; ambos sao locais e ignorados pelo Git.

## Estrutura

```text
config/       Configuracoes dos experimentos V1 e V2
src/          Pacote energy_prediction
 tests/       Testes unitarios e de integracao sinteticos
pyproject.toml Dependencias, instalacao do pacote e configuracao do pytest
README.md     Este guia
```

## Escopo e interpretacao

- A V1 e uma prova de conceito de previsao de duty cycle agregado de GPU. Nao calcula energia, potencia ou PUE.
- A V2 preve utilizacao futura de CPU do Alibaba v2018; tambem nao calcula energia, potencia ou PUE.
- Executar os testes sinteticos valida partes do pipeline, mas nao substitui executar os experimentos com os datasets reais.
