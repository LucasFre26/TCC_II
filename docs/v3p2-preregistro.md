# Pré-registro da V3 Parte 2

Documento fixado antes da corrida completa (`config/v3p2.yaml`). O SHA-256 deste arquivo é gravado no manifesto da corrida; alterá-lo durante a corrida a interrompe. Nenhum resultado da V3 Parte 2 foi observado antes desta versão; os números da V3 citados abaixo vêm de `results/v3/20261104T232620Z-5b659dad/` e de `results/v2-full/`.

## 1. Protocolo

- **Série:** CPU agregada da frota do Alibaba Cluster Trace v2018 (4.023 máquinas, média ponderada por `cpu_num` das máquinas presentes), passo de 300 s, 2.304 intervalos (8 dias). Atributos idênticos aos da V3; origens incompletas na configuração C são descartadas, como na V2.
- **Blocos:** bloco do dia d = origens 288·d − 6 a 288·d + 281 (todos os alvos no dia d). Desenvolvimento: dias 3, 4, 5 e 6. Teste herdado: dia 7 (origens 2010–2297).
- **Treino e seleção:** treino do bloco = origens até o início do bloco menos 18 passos; validação interna = últimas 144 origens do treino, com purga de 18 e treino interno mínimo de 576. Hiperparâmetros só pela validação interna; reajuste no treino do bloco; previsão única do bloco.
- **Rótulo:** todo resultado é exploratório. Os dias 4 a 6 serviram de validação na V2 (seleção de hiperparâmetros) e na V3; o dia 3 nunca foi avaliado. O dia 7 já foi inspecionado na V2, na V3 e na análise expandida: esta é a quarta inspeção, reportada só como continuidade, nunca como confirmação independente.
- **Integridade:** a média ponderada do painel por máquina precisa reproduzir a série agregada (tolerância de 1e-4 p.p.) e o XGBoost da V2, reajustado no dia 7 com treino 12–1991 e os hiperparâmetros gravados, precisa reproduzir as previsões preservadas (tolerância de 1e-6 p.p.). Falha em qualquer um interrompe a corrida.

## 2. Estatística e regra de decisão

- Comparações pareadas nas mesmas origens dos dias de desenvolvimento. Perda por origem (definida por hipótese abaixo); diferença = candidato − referência.
- Block bootstrap estratificado por dia: blocos móveis de 12 passos (primário), 6 e 24 (sensibilidade), reamostrados dentro de cada dia; 5.000 reamostragens; semente 42. P-valor bicaudal com correção (k + 1)/(B + 1).
- **Melhor:** IC de 95% inteiro abaixo de zero e diferença favorável em pelo menos 3 dos 4 dias. **Pior:** IC inteiro acima de zero e diferença desfavorável em pelo menos 3 dos 4 dias. **Não distinguível:** qualquer outro caso.
- Holm sobre o conjunto primário inteiro (9 comparações × 3 horizontes = 27 testes, bloco primário), além da regra acima. Matrizes exploratórias: Holm por horizonte, dentro de cada família.
- Relevância prática: |diferença relativa| ≥ 5%.

## 3. Hipóteses, candidatos e referências

| Hipótese | Candidato único | Referência | Perda por origem |
|---|---|---|---|
| H1 hierárquica | critério × K × reconciliação escolhido por horizonte no MAE agregado da validação interna | XGBoost V2 direto (A) | erro absoluto |
| H2 global | XGBoost global, M = 1.024 máquinas amostradas por quintil de média de treino, passo 2 entre origens | XGBoost V2 direto | erro absoluto |
| H3 regimes | combinador condicionado (seleção ou inverso do MAE) escolhido por metades cronológicas da validação interna | XGBoost V2 direto | erro absoluto |
| H4 picos | XGBoost com peso 1 + λ nas subidas (λ escolhido pela regra abaixo) | XGBoost V2 direto | erro (obs − prev) nos alvos acima do P90 da CPU do treino; negativo = menos subestimação |
| H5 calibração | ACI com retorno atrasado, γ = 0,005 | conformal estático (regra da V3) | interval score de [Q05, Q95] calibrado (α = 0,1) |
| H6 energia | rota por grupo do candidato hierárquico | rota da média com o XGBoost V2 | \|diferença total de energia\| (Wh), r = 1,4, servidor substituto da V3 |
| H8a robustez | DLinear com normalização por janela (média de 3 sementes) | DLinear na formulação da V3 (média de 3 sementes) | erro absoluto |
| H8b robustez | XGBoost V2 com peso de recência (meia-vida ∈ {144, 288, 576} passos, escolhida na validação interna) | XGBoost V2 | erro absoluto |
| H8c robustez | XGBoost V2 sem seno e cosseno de 24 h | XGBoost V2 | erro absoluto |

Exploratórios: TCN e PatchTST normalizados por janela contra a formulação da V3; redes normalizadas contra o XGBoost; cada variante da matriz hierárquica; TCN global (se o portão passar); combinadores individuais; valores de λ; janela móvel (n = 288, 144), ACI com γ = 0,02 e SPCI; demais modelos e rotas de energia, r e substitutos.

### Expectativas registradas (não são resultados)

- H1 e H2: ganho pequeno ou nulo é o esperado, porque a primeira componente principal explica cerca de 72% da variância dos níveis por máquina e o salto de nível do dia 7 atinge todos os quintis. Um nulo é informativo.
- H8a: a normalização por janela deve reduzir o viés que acompanha o deslocamento de nível; não se espera, por isso, superar o XGBoost.
- H4: reduzir a subestimação nos picos deve custar erro médio; a regra limita esse custo.
- H5: o ACI deve aproximar a cobertura do nominal em dias com mudança de nível, com cobertura empírica e sem garantia teórica.
- H6: a rota da média superestima a energia por máquina quando r > 1 (curva côncava); a rota por grupo deve reduzir essa parcela.

## 4. Regras específicas

- **H4:** λ ∈ {0, 1, 3}; σ_h = desvio de y(s+h) − y(s) no treino do bloco; escolhido o λ de menor subestimação média nos picos da validação interna entre os que não aumentam o MAE interno em mais de 5% em relação a λ = 0; sem candidato, λ = 0.
- **H2, portão da TCN global:** roda só se o MAE agregado interno médio do XGBoost global ficar até 5% acima do MAE interno do XGBoost direto em pelo menos um horizonte.
- **H5, SPCI:** roda só se o tempo acumulado ficar dentro de 30 min.
- **H6:** conjunto fixo por origem (máquinas presentes na origem com CPU observada nos passos 1 a 6); r ∈ {1,0; 1,4; 1,8}; substitutos de `config/specpower_substitutes.yaml` (servidor da V3; NF5280 M4 com 88 threads como fallback; PRIMERGY TX2550 M5 com 96 threads como extensão fora do período do critério); cenário heterogêneo sintético por hash do identificador. Nenhum r e nenhum substituto é escolhido pelo erro.
- **H7, modelo pré-treinado:** não executado. O Chronos-2 lista o Alibaba Cluster Trace 2018 (5 min) no corpus de treino; o LOTSA (Moirai) contém `alibaba_cluster_trace_2018`; o corpus do Chronos-Bolt não é público.

## 5. Ameaças à validade conhecidas antes da corrida

- Dias de desenvolvimento já usados na V2 e na V3; n = 4 dias para a variação entre dias.
- A covariância da MinT e a escolha do candidato hierárquico usam a mesma validação interna.
- O tamanho efetivo do modelo global é uma estimativa pelo efeito de desenho, sem correção da autocorrelação temporal.
- Energia em servidores substitutos; diferença entre duas estimativas modeladas, não erro contra medidor.
