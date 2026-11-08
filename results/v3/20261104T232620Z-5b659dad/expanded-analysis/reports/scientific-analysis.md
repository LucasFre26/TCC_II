# Análise científica expandida da V3

Corrida analisada: `results/v3/20261104T232620Z-5b659dad/`. Toda grandeza abaixo foi recalculada a partir dos artefatos preservados da V2 (`results/v2-full`) e da V3, sem novo treino neural. Antes de qualquer indicador, 30 de 30 verificações de integridade passaram (origens, unicidade, reprodução de MAE, RMSE, pinball, bootstrap, energia e reajustes; lista em `metadata/analysis_manifest.json`).

Convenções: erro = observado − previsto (positivo = subestimação) para CPU; na seção de energia, diferença = projetada − referência (negativa = subestimação), como em `energy_by_origin.csv`. Tempo em horas desde o início do trace, não horário civil. Potência e energia são estimativas paramétricas de um servidor substituto do SPECpower_ssj2008; não há medição elétrica no trace e PUE não foi calculado.

## 1. Perguntas de pesquisa

- **P1.** Algum modelo da V3 reduziu o erro de previsão da CPU agregada em relação ao XGBoost da V2, com sustentação estatística?
- **P2.** Por que as redes neurais (DLinear, TCN, N-BEATSx, N-HiTS, PatchTST) não superaram o XGBoost da V2?
- **P3.** Como os modelos se comportam nos picos de utilização, e algum sinal antecipa esses picos?
- **P4.** Os quantis previstos cobrem a CPU observada na proporção nominal?
- **P5.** Quanto o erro de CPU se propaga para a potência e a energia de TI modeladas, e quanto isso depende dos parâmetros do modelo de potência?
- **P6.** Memória, rede, disco e a distribuição da CPU entre máquinas acrescentam informação preditiva?
- **P7.** A diferença entre população principal e coorte altera as conclusões?

## 2. Hipóteses avaliadas

As hipóteses explicativas abaixo foram formuladas depois de observar os resultados da V3; por isso são tratadas como explicações compatíveis com os dados, não como hipóteses confirmatórias.

- **H1 (pré-registrada na V3).** Os modelos residuais e híbridos diferem do XGBoost da V2 no teste. Resultado: 0 de 9 intervalos pré-registrados (incremento, Kalman e híbrido × 3 horizontes) excluem zero.
- **H2 (pré-registrada na V3).** As redes neurais diferem do XGBoost da V2 no teste. Resultado: 9 de 9 intervalos (DLinear, TCN, N-BEATSx × 3 horizontes) excluem zero, e nos que excluem a diferença é positiva (redes piores).
- **H3 (explicativa).** A perda das redes concentra-se na passagem para o período de teste, que tem nível de CPU mais alto que o desenvolvimento, e não num ajuste ruim na validação interna. Evidência: razão de generalização normalizada (Seção 5) e viés em regime estável (Seção 4.3). Status: compatível com os dados; não demonstra causa.
- **H4 (explicativa).** Os picos são subestimados porque a maioria dura um único passo de 5 min, o que deixa pouca informação disponível na origem. Evidência: duração dos eventos e baixa antecipação (Seção 6). Status: compatível com os dados; a ausência de sinal antecedente não foi testada diretamente.
- **H5.** A agregação da CPU antes da curva de potência introduz viés quando r > 1 (efeito de Jensen). Resultado: Seção 8.4.

## 3. Metodologia da análise

1. **Leitura e integridade.** Leitura dos CSV e JSON das famílias `residual`, `neural`, `quantile`, `energy` e `integrity` e de `results/v2-full/{main,cohort}`; verificação de que o teste tem as origens 2010–2297 em todos os grupos, de que há uma previsão por origem, e reprodução de todos os MAE, RMSE, pinball, intervalos de bootstrap e energias gravados. A série agregada foi reconstruída do cache de grade com a configuração gravada no manifesto da corrida e conferida contra os valores observados das previsões.
2. **Métricas.** MAE, RMSE, viés, mediana, P90/P95/P99 e máximo do erro absoluto, desvio do erro, MASE do projeto (MAE/MAE da persistência nas mesmas origens), skill score (1 − RMSE/RMSE da persistência) e melhoria percentual contra a persistência e contra o XGBoost da V2 da mesma configuração. Validação externa agregada (720 origens) e teste final (288 origens) são reportados separadamente.
3. **Diagnóstico temporal.** Regime por origem: subida se y(t+h) − y(t) ≥ σ_h, queda se ≤ −σ_h, estável caso contrário, com σ_h = 4,27 p.p. (5 min), 5,90 p.p. (15 min), 6,70 p.p. (30 min) (desvio das variações no desenvolvimento). Faixas de CPU: quintis da CPU no desenvolvimento. Janelas de 3 h escolhidas pela série observada, sem olhar erro de modelo.
4. **Picos.** Limiar primário: P90 (50,40%) e P95 (54,38%) da CPU observada antes do teste. Evento: sequência de instantes acima do limiar, unindo sequências separadas por até 1 passo. Antecipação: alerta (previsão ≥ limiar) mirando até 1 passo do início e emitido antes dele. Falso alarme: episódio de alertas sem evento a até 1 passo.
5. **Inferência.** Block bootstrap da V2, 5.000 reamostragens, semente 42, blocos de 6, 12 (primário) e 24 passos. P-valor bicaudal das mesmas reamostragens; correção de Holm sobre todas as comparações pré-registradas no bloco primário e, separadamente, sobre a matriz exploratória de cada horizonte. Relevância prática: |ΔMAE|/MAE da referência ≥ 5% (convenção desta análise), com equivalente linear em potência de 4,114 W por p.p. de CPU por servidor substituto.
6. **Energia.** Modelo de Fan com P_idle = 49,6 W e P_busy = 461 W (resultado `power_ssj2008-20170621-00759`), r ∈ {1,0; 1,4; 1,8}, trapézio a cada 5 min com o extremo esquerdo na CPU observada. Energia acumulada somada em janelas sem sobreposição.

## 4. Resultados quantitativos

### 4.1 Erro pontual (configuração A, população principal)

| Modelo | Validação 5 min | Validação 15 min | Validação 30 min | Teste 5 min | Teste 15 min | Teste 30 min |
|---|---:|---:|---:|---:|---:|---:|
| Persistência | 3,04 | 4,04 | 4,57 | 3,35 | 4,70 | 5,33 |
| Ridge (V2) | 2,79 | 3,53 | 3,87 | 3,29 | 4,51 | 5,22 |
| XGBoost (V2) | 2,76 | 3,49 | 3,80 | 3,06 | 3,86 | 4,33 |
| XGBoost incremento | 2,78 | 3,54 | 3,80 | 3,09 | 3,95 | 4,50 |
| Kalman (nível local) | 2,89 | 3,86 | 4,40 | 3,26 | 4,41 | 5,00 |
| Kalman + XGBoost | 2,73 | 3,42 | 3,77 | 3,03 | 3,92 | 4,32 |
| DLinear | 2,73 | 3,50 | 3,86 | 3,27 | 4,46 | 5,14 |
| TCN | 3,03 | 3,64 | 3,94 | 3,43 | 4,71 | 5,38 |
| N-BEATSx | 3,28 | 3,73 | 4,00 | 3,81 | 5,20 | 5,25 |
| N-HiTS | 3,10 | 3,67 | 4,12 | 4,21 | 5,26 | 5,73 |
| PatchTST | 3,31 | 3,87 | 4,27 | 3,27 | 4,30 | 4,32 |

MAE em pontos percentuais de CPU. Fonte: `csv/model_metrics.csv`; Figura 5 (`figures/png/05_ranking_mae_rmse.png`), Figura 6 (`figures/png/06_heatmaps_mae_melhoria.png`) e Figura 7 (`figures/png/07_validacao_vs_teste.png`).

Menor MAE de teste: Kalman + XGBoost em 5 min, XGBoost (V2) em 15 min, Kalman + XGBoost em 30 min. Menor MAE na validação externa: DLinear em 5 min, Kalman + XGBoost em 15 min, Kalman + XGBoost em 30 min. A melhoria do XGBoost da V2 sobre a persistência no teste foi de 8,6%, 18,0%, 18,7% em 5, 15 e 30 min.

### 4.2 Comparações pré-registradas contra o XGBoost da V2

| Modelo | Horizonte | ΔMAE (p.p.) | IC 95% (bloco 1 h) | Relativa | Blocos 30 min/1 h/2 h | Holm |
|---|---|---:|---|---:|---|---|
| XGBoost incremento | 5 min | +0,024 | [-0,08; 0,11] | 0,8% | nenhum | não |
| XGBoost incremento | 15 min | +0,092 | [-0,18; 0,31] | 2,4% | nenhum | não |
| XGBoost incremento | 30 min | +0,170 | [-0,24; 0,54] | 3,9% | nenhum | não |
| Kalman (nível local) | 5 min | +0,195 | [-0,05; 0,40] | 6,4% | nenhum | não |
| Kalman (nível local) | 15 min | +0,554 | [-0,28; 1,26] | 14,4% | nenhum | não |
| Kalman (nível local) | 30 min | +0,664 | [-0,47; 1,51] | 15,3% | nenhum | não |
| Kalman + XGBoost | 5 min | -0,039 | [-0,15; 0,05] | -1,3% | nenhum | não |
| Kalman + XGBoost | 15 min | +0,067 | [-0,28; 0,37] | 1,7% | nenhum | não |
| Kalman + XGBoost | 30 min | -0,015 | [-0,43; 0,35] | -0,3% | nenhum | não |
| DLinear | 5 min | +0,205 | [0,01; 0,38] | 6,7% | parcial | não |
| DLinear | 15 min | +0,607 | [0,17; 1,07] | 15,7% | todos | não |
| DLinear | 30 min | +0,810 | [0,33; 1,27] | 18,7% | todos | não |
| TCN | 5 min | +0,363 | [0,12; 0,60] | 11,8% | todos | não |
| TCN | 15 min | +0,856 | [0,34; 1,35] | 22,2% | todos | não |
| TCN | 30 min | +1,044 | [0,46; 1,59] | 24,1% | todos | sim |
| N-BEATSx | 5 min | +0,742 | [0,23; 1,27] | 24,2% | todos | não |
| N-BEATSx | 15 min | +1,343 | [0,47; 2,13] | 34,8% | todos | não |
| N-BEATSx | 30 min | +0,921 | [0,24; 1,49] | 21,3% | todos | não |

ΔMAE = MAE do modelo − MAE do XGBoost da V2 nas mesmas 288 origens. Fonte: `csv/statistical_comparisons.csv`; Figura 18 (`figures/png/18_forest_preregistrado.png`).

O híbrido Kalman + XGBoost teve o menor MAE de teste em 5 e 30 min, mas a diferença para o XGBoost da V2 não se distingue de zero em 5 min, não se distingue de zero em 15 min e não se distingue de zero em 30 min. Não há, portanto, sustentação para declarar o híbrido superior ao XGBoost da V2.

Das 93 comparações pré-registradas (erro absoluto e pinball, bloco de 1 h), 34 têm intervalo sem zero; depois da correção de Holm sobre o conjunto, restam 4. Essa queda tem duas causas: o número de comparações e a resolução do bootstrap (com 5.000 reamostragens, o menor p-valor possível é cerca de 0,0004). O protocolo da V3 pré-registrou a decisão pelo intervalo de cada comparação, sem correção de multiplicidade; a coluna de Holm é informativa e indica que as conclusões individuais devem ser lidas como evidência por comparação, não como controle de erro do conjunto. Na matriz exploratória (165 pares × horizontes, Figura 19 (`figures/png/19_matriz_pares.png`)), 83 intervalos excluem zero e 26 sobrevivem a Holm por horizonte.

### 4.3 Comportamento temporal e mudança de nível

A CPU média do teste foi 42,03%, contra 37,40% no desenvolvimento (+4,63 p.p., cerca de 0,49 desvio-padrão do desenvolvimento); 72% dos instantes previstos do teste (30 min) caíram nos dois quintis superiores da CPU do desenvolvimento (Figura 2 (`figures/png/02_distribuicao_dependencia.png`), Figura 13 (`figures/png/13_erro_faixa_cpu.png`)).

No regime estável (variação até o alvo menor que σ_h), o viés em 30 min foi Persistência -0,10 p.p., Ridge (V2) +2,14 p.p., XGBoost (V2) +1,17 p.p., XGBoost incremento +1,71 p.p., Kalman (nível local) -0,20 p.p., Kalman + XGBoost +1,42 p.p., DLinear +1,59 p.p., TCN +3,15 p.p., N-BEATSx +2,73 p.p., N-HiTS +3,57 p.p., PatchTST -0,30 p.p.. A persistência praticamente não tem viés nesse regime; subestimaram mais de 0,5 p.p.: DLinear, Kalman + XGBoost, N-BEATSx, N-HiTS, Ridge (V2), TCN, XGBoost incremento, XGBoost (V2). Ficaram abaixo disso: PatchTST, Kalman (nível local). Viés positivo em regime estável significa que o modelo previu queda enquanto a série ficou perto do nível da origem, o que é compatível com previsões puxadas para o nível médio do treino, mais baixo que o do teste.

Por faixa de CPU observada no alvo (Figura 13 (`figures/png/13_erro_faixa_cpu.png`)), o viés vai de negativo na faixa mais baixa a positivo na mais alta para todos os modelos, inclusive a persistência; esse padrão decorre de condicionar o erro no próprio valor observado e não isola o viés de cada modelo. A leitura informativa é a comparação entre modelos dentro da mesma faixa: na faixa mais alta, em 30 min, o viés foi Persistência +3,50 p.p., Kalman (nível local) +3,56 p.p., PatchTST +3,97 p.p., XGBoost (V2) +5,03 p.p., Kalman + XGBoost +5,19 p.p., XGBoost incremento +5,54 p.p., DLinear +6,56 p.p., N-BEATSx +6,87 p.p., Ridge (V2) +7,31 p.p., N-HiTS +7,47 p.p., TCN +8,18 p.p..

Por regime (30 min; n = estável 204, subida 45, queda 39), o MAE na subida foi Persistência 11,13, XGBoost (V2) 7,44, Kalman + XGBoost 8,09, DLinear 9,00 p.p., e na queda Persistência 10,73, XGBoost (V2) 4,55, Kalman + XGBoost 5,43, DLinear 5,60 p.p. (Figura 14 (`figures/png/14_erro_variacao_regime.png`)). Entre todos os modelos, o menor MAE na subida foi do PatchTST e na queda do N-BEATSx. Modelos com viés de subestimação ganham nas quedas e perdem nas subidas, de modo que o MAE por regime precisa ser lido junto com o viés. O Kalman (nível local) fica próximo da persistência nos dois regimes, pois sua previsão é o nível filtrado na origem; o XGBoost de incremento, embora também parta de cpu(t), corrige parte da variação e tem erro bem menor que o da persistência nas quedas.

Os resíduos de 30 min têm autocorrelação alta até a defasagem 5, esperada para previsões de 6 passos com origens consecutivas; depois disso, a autocorrelação da DLinear permanece positiva por mais defasagens que a do XGBoost, coerente com um viés persistente (Figura 12 (`figures/png/12_residuos.png`)).

## 5. Interpretação dos resultados das redes neurais

**Observado.**

- Diferença relativa de MAE contra o XGBoost da V2 (5, 15 e 30 min), validação externa → teste: DLinear -0,9%, +0,4%, +1,5% → +6,7%, +15,7%, +18,7%; TCN +9,8%, +4,2%, +3,7% → +11,8%, +22,2%, +24,1%; N-BEATSx +18,9%, +6,8%, +5,3% → +24,2%, +34,8%, +21,3%; N-HiTS +12,5%, +5,2%, +8,5% → +37,3%, +36,3%, +32,1%; PatchTST +20,0%, +10,8%, +12,4% → +6,8%, +11,6%, -0,3%.
- Razão de generalização normalizada pela persistência ((MAE de teste / MAE de seleção) ÷ mesma razão da persistência), em 5, 15 e 30 min: DLinear 1,09, 1,11, 1,16; TCN 1,11, 1,16, 1,24; N-BEATSx 1,12, 1,22, 1,18; N-HiTS 1,24, 1,27, 1,29; PatchTST 0,97, 1,04, 0,95. Para os modelos de árvore e de estados: XGBoost (V2) 1,01, 0,95, 0,98; XGBoost incremento 0,98, 0,98, 1,00; Kalman + XGBoost 1,01, 1,00, 0,98; Kalman (nível local) 1,02, 0,98, 0,98 (Figura 21 (`figures/png/21_generalizacao_janela.png`)).
- Na validação interna, a razão loss de validação/loss de treino no melhor estado ficou entre 0,84 e 1,16 (configuração A, modo direto); a melhor época média entre folds, na média dos três horizontes, foi PatchTST 4,9, N-HiTS 5,9, N-BEATSx 7,1, TCN 9,1, DLinear 26,2 (Figura 20 (`figures/png/20_curvas_loss.png`)).
- Janela de entrada: a soma do MAE interno do DLinear piorou ao alongar a janela (36 passos: 10,24, 72 passos: 10,89, 144 passos: 11,34 p.p.).
- Tamanho e custo: as redes tiveram de 78 a 59.008 parâmetros treináveis, pico de memória CUDA de 65 a 73 MiB numa GPU de 6 GB, e reajuste de teste em menos de 1 s; o XGBoost reajustado em CPU levou menos de 0,1 s (Figura 22 (`figures/png/22_custo_pareto.png`)). Mais parâmetros não se traduziram em menor MAE de teste.

**Hipóteses explicativas (não demonstradas).** (i) O teste está num nível de CPU mais alto que o desenvolvimento; redes que preveem o nível padronizado pela média do treino tendem a puxar a previsão para essa média, o que aparece como viés positivo mesmo em regime estável, enquanto modelos ancorados em cpu(t) (persistência, Kalman, incremento) sofrem menos. (ii) O desenvolvimento tem cerca de 1.900 origens com forte autocorrelação, poucas para redes com dezenas de milhares de parâmetros; N-BEATSx, N-HiTS e PatchTST atingiram o melhor estado nas primeiras épocas. (iii) A janela curta (3 h) foi a melhor na validação interna, sinal de que histórico longo não trouxe informação adicional nesta série. Nenhum desses pontos prova sobreajuste: a validação interna não mostrou distância grande entre treino e validação, e o pior desempenho aparece principalmente na passagem para o teste.

## 6. Análise dos picos

No teste, 12 eventos ficaram acima do P90 do desenvolvimento (50,40%), dos quais 11 duraram um único passo de 5 min; o mais longo durou 24 passos, com pico de 78,46%. Nos instantes de pico, todos os modelos subestimaram: viés em 30 min de Persistência +4,27, XGBoost (V2) +4,41, Kalman + XGBoost +4,79, DLinear +7,47 p.p., frequência de subestimação entre 60% e 94% (Figura 15 (`figures/png/15_picos_eventos.png`), Figura 16 (`figures/png/16_subestimacao_picos.png`)).

Antecipação de eventos (Figura 17 (`figures/png/17_alertas_picos.png`)): o XGBoost da V2 antecipou 1, 2, 2 de 12 eventos em 5, 15 e 30 min; o teto Q95 ajustado antecipou 9, 10, 10, com 5, 6, 9 episódios de falso alarme em 12, 9, 16 episódios de alerta. A persistência não antecipa o início de nenhum evento em 5 min, por construção. Com 12 eventos, a maioria de um passo, essas contagens são descritivas: o teto quantílico troca mais acertos por mais falsos alarmes e não constitui um detector de picos validado.

## 7. Previsão probabilística

Cobertura empírica no teste (A), 5, 15 e 30 min: Q50 0,43, 0,38, 0,38; Q95 bruto 0,90, 0,89, 0,89; Q95 ajustado 0,91, 0,92, 0,93. Na validação externa, o Q95 bruto cobriu 0,93, 0,92, 0,94. O ajuste conformal aproximou a cobertura do nominal, mas ela continuou abaixo de 0,95; a Q50 cobriu menos de 0,5, outra manifestação da subestimação no período de teste. Nos picos, o Q95 ajustado cobriu 0,66, 0,71, 0,71 (Figura 25 (`figures/png/25_calibracao_quantis.png`), Figura 26 (`figures/png/26_faixas_quantis.png`)). Q90 e Q95 são quantis unilaterais; a distância q95 − q50 é uma margem superior, não a largura de um intervalo bilateral. As coberturas são empíricas, sem garantia teórica sob dependência temporal.

## 8. Análise energética

### 8.1 Grandeza

A energia de referência do servidor substituto nas 24 h de teste, com r = 1,4, foi 6.530 Wh (soma de 288 janelas de 5 min). Ela é uma estimativa paramétrica, não consumo medido do cluster (Figura 27 (`figures/png/27_curvas_potencia.png`)).

### 8.2 Erro de energia

Erro relativo absoluto médio por origem em 30 min (r = 1,4): Persistência 5,2%, Ridge (V2) 4,8%, XGBoost (V2) 3,8%, XGBoost incremento 4,1%, Kalman (nível local) 4,8%, Kalman + XGBoost 4,0%, DLinear 4,8%, TCN 5,0%, N-BEATSx 6,1%. Viés por origem em 30 min (projetada − referência): Persistência -0,24 Wh, Ridge (V2) -2,97 Wh, XGBoost (V2) -1,91 Wh, XGBoost incremento -2,06 Wh, Kalman (nível local) -0,25 Wh, Kalman + XGBoost -1,86 Wh, DLinear -2,58 Wh, TCN -2,75 Wh, N-BEATSx -5,03 Wh (Figura 29 (`figures/png/29_erro_energia_sensibilidade.png`)).

Somando 48 janelas de 30 min sem sobreposição, a diferença acumulada em 24 h foi Persistência +49,8 Wh, Ridge (V2) -123,7 Wh, XGBoost (V2) -80,7 Wh, XGBoost incremento -61,5 Wh, Kalman (nível local) +5,8 Wh, Kalman + XGBoost -69,7 Wh, DLinear -115,2 Wh, TCN -115,4 Wh, N-BEATSx -238,5 Wh (Figura 28 (`figures/png/28_potencia_energia_acumulada.png`)). A ordem por erro por origem difere da ordem por erro acumulado: a persistência e o Kalman têm erro por origem maior, mas quase sem viés, de modo que os erros se compensam na soma; os modelos aprendidos, que subestimaram a CPU no teste, acumulam energia projetada abaixo da referência.

### 8.3 Sensibilidade

Com r variando de 1,0 a 1,8, o MAE de energia do XGBoost da V2 em 30 min foi 5,14, 5,17, 5,59 Wh; a escala do erro acompanha linearmente P_busy − P_idle, e o erro relativo diminui quando P_idle aumenta, porque a energia de referência cresce (Figura 29 (`figures/png/29_erro_energia_sensibilidade.png`)). A relação entre erro de CPU e erro de potência é praticamente linear, com inclinação próxima de (P_busy − P_idle)/100 = 4,11 W por p.p. (Figura 30 (`figures/png/30_erro_cpu_energia.png`)). Nenhum r foi escolhido pelo erro no teste.

### 8.4 Efeito de Jensen

Convenção: gap = média das potências por máquina − potência da utilização média; negativo porque a curva é côncava para r > 1. Por servidor substituto, o gap médio foi -3,16 W (r = 1,4; -1,2% de P(média)) e -5,04 W (r = 1,8), com mínimo de -27,81 W. Para a frota hipotética de servidores substitutos (média de 3.740 máquinas presentes), a diferença média foi -11,9 kW (r = 1,4) e -19,0 kW (r = 1,8). Watts por servidor e watts da frota são grandezas diferentes e não devem ser comparados diretamente. A aproximação de segunda ordem errou em média 0,45 W por servidor (r = 1,4), com máximo de 1,85 W (Figura 31 (`figures/png/31_jensen.png`)). Com r = 1 o gap é nulo.

## 9. Dataset e populações

A população principal agrega 4.023 máquinas; a coorte, 4.022. A diferença é a máquina `m_2739`, com status `IMPORT_INSTALLING`, CPU 0% em todos os seus 1.533 intervalos e último registro em 145,4 h, antes do teste. A diferença máxima entre as séries foi 0,0258 p.p. na série inteira e 0,0000 p.p. nos alvos do teste: no teste as duas séries coincidem. As métricas de teste da V2 diferem em até 0,092 p.p. de MAE entre as populações, efeito do reajuste em séries de treino ligeiramente distintas (Figura 4 (`figures/png/04_populacoes_m2739.png`)).

## 10. Ablação e importância de atributos

Variação de MAE no teste contra A (positivo = piora): XGBoost (V2): B/5 min +0,5%, B/15 min +0,6%, B/30 min +1,5%, C/5 min +1,0%, C/15 min -0,3%, C/30 min -2,8%; Kalman + XGBoost: B/5 min +0,7%, B/15 min +5,4%, B/30 min +2,2%, C/5 min +5,2%, C/15 min +2,4%, C/30 min +0,7%; TCN: B/5 min +6,5%, B/15 min +17,6%, B/30 min +13,8%, C/5 min +19,1%, C/15 min +6,8%, C/30 min +16,3%; N-BEATSx: B/5 min +7,3%, B/15 min +16,8%, B/30 min +21,7%, C/5 min +13,5%, C/15 min -9,7%, C/30 min -12,5%. Contra A nas mesmas origens: XGBoost incremento D/5 min +5,7%, XGBoost incremento D/15 min +9,5%, XGBoost incremento D/30 min +16,1%, DLinear D/5 min -0,6%, DLinear D/15 min +16,2%, DLinear D/30 min +30,4%. A configuração D (estatísticas da distribuição da CPU entre máquinas) foi executada só no XGBoost de incremento e no DLinear (Figura 23 (`figures/png/23_ablacao.png`)).

Importância TreeSHAP nos reajustes verificados (reproduziram as previsões gravadas): XGBoost (V2) A em 5 min: CPU na origem 54%, CPU janela móvel 24%, calendário 13%; XGBoost (V2) A em 30 min: CPU janela móvel 53%, calendário 30%, CPU defasada 10%; XGBoost (V2) C em 30 min: CPU janela móvel 40%, calendário 27%, rede 11% (memória, rede e disco somam 25%) (Figura 24 (`figures/png/24_importancia_shap.png`)). Importância preditiva descreve associação usada pelo modelo; não implica relação causal.

## 11. Ameaças à validade

- **Teste reutilizado.** O teste final de 24 h já tinha sido inspecionado na V2 e foi reutilizado na V3; ele não é uma validação independente do processo de desenvolvimento, e as hipóteses H3 e H4 foram formuladas depois de ver seus resultados.
- **Um único período de teste.** 288 origens consecutivas de um único dia, com mudança de nível em relação ao desenvolvimento; a generalização para outros períodos não foi avaliada.
- **Dependência temporal.** O block bootstrap trata a dependência de curto prazo, mas o tamanho efetivo da amostra é menor que 288; os intervalos dependem do bloco (concordância entre blocos em `bootstrap_intervals.csv`).
- **Múltiplas comparações.** Com Holm, poucas comparações pré-registradas permanecem significativas; a resolução do bootstrap limita o menor p-valor.
- **Picos raros.** 12 eventos, quase todos de um passo; taxas de antecipação têm incerteza alta.
- **Energia modelada.** Servidor substituto, carga do benchmark diferente da utilização do trace, parâmetros de um único resultado do SPECpower; o erro de energia é diferença entre duas estimativas, não erro contra medidor. PUE não foi calculado.
- **Tempos.** Tempos das redes vêm da corrida original em CUDA; tempos dos modelos clássicos foram medidos nesta análise em CPU; só servem como ordem de grandeza.
- **Seleção da arquitetura neural.** Grades pequenas, janela escolhida só pelo DLinear e orçamento fixo; outras configurações poderiam ter desempenho diferente.

## 12. Conclusões sustentadas pelos dados

1. Nenhum modelo da V3 teve MAE de teste menor que o do XGBoost da V2 com intervalo pré-registrado sem zero. O híbrido Kalman + XGBoost teve o menor MAE numérico em parte dos horizontes, mas a diferença não se distinguiu de zero.
2. DLinear, TCN e N-BEATSx tiveram MAE de teste maior que o XGBoost da V2 em 9 de 9 comparações pré-registradas. Na validação externa a distância era menor (o DLinear praticamente empatou); a razão de generalização normalizada das redes, exceto o PatchTST, ficou acima de 1, e o erro de teste veio acompanhado de viés de subestimação num período de nível mais alto. A explicação causal permanece hipótese.
3. Todos os modelos subestimaram os picos, e nenhum sinal pontual antecipou mais que uma pequena parte dos eventos; o teto quantílico antecipou mais eventos ao custo de mais falsos alarmes.
4. A cobertura empírica dos quantis ficou abaixo do nominal no teste, mesmo depois do ajuste conformal.
5. O erro de energia de TI modelada por origem ficou na faixa de poucos por cento; o erro acumulado depende mais do viés do modelo que do seu MAE, e o efeito de Jensen por servidor é da ordem de alguns watts para r > 1.
6. Os resultados estão prontos para orientar a redação, desde que o texto mantenha as ressalvas da Seção 11.

## Arquivos

- Tabelas: `csv/` (dicionário em `metadata/data_dictionary.md`).
- Figuras: `figures/png`, `figures/svg`, `figures/pdf` e os dados de cada figura em `figures/data/`.
- Relatório navegável: `interactive/relatorio.html`.
- Recursos para o artigo: `latex/`.
