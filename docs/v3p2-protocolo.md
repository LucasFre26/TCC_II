# Protocolo da V3 Parte 2

## Por que um protocolo por dia

A auditoria da V3 mostrou três fatos que orientam este protocolo:

1. O dia de teste da V3 (dia 7) tem CPU média de 42,2%, contra 37,6% no treino, e um perfil intradiário diferente: a correlação do perfil com os demais dias foi de 0,43 a 0,63, enquanto os outros dias se correlacionam entre si de 0,65 a 0,81.
2. O viés dos modelos aprendidos é próximo de zero na validação e positivo no teste, e seu sinal acompanha o deslocamento de nível de cada corte (folds e teste), como uma reversão à média do treino.
3. O mesmo dia de teste já foi inspecionado na V2, na V3 e na análise expandida da V3.

Com um único dia de teste, a variação entre dias é zero por construção, e o bootstrap por blocos de 1 h mede só a variação dentro daquele dia. A Parte 2 troca o critério primário por quatro blocos diários (dias 3 a 6) e mantém o dia 7 só como continuidade.

## Blocos, treino e seleção

- Bloco do dia d: origens 288·d − 6 a 288·d + 281, para que todos os alvos (até 30 min) caiam no dia d.
- Treino do bloco: origens anteriores ao início do bloco menos 18 passos (maior horizonte mais maior janela).
- Validação interna: as últimas 144 origens do treino (12 h), com purga de 18 passos; treino interno mínimo de 576 origens (48 h).
- Grupos, limiares de regime, σ das subidas, normalização, pesos de recência e calibração inicial usam só o treino do bloco.
- O modelo é reajustado no treino do bloco com o hiperparâmetro escolhido e prevê o bloco uma vez.

## Estatística

- Diferença por origem, candidato − referência, nas mesmas origens.
- Block bootstrap estratificado por dia (blocos de 1 h primários, 30 min e 2 h como sensibilidade), 5.000 reamostragens, semente 42.
- Decisão pré-registrada: melhor quando o IC de 95% fica abaixo de zero e a diferença favorece o candidato em pelo menos 3 dos 4 dias; pior no caso simétrico; não distinguível nos demais.
- Holm sobre as 27 comparações primárias (9 hipóteses × 3 horizontes); Holm por horizonte nas matrizes exploratórias.

## Integridade

- O painel denso intervalo × máquina (float32) reproduz a série agregada com diferença máxima de cerca de 1e-7 p.p. na frota real (tolerância de 1e-4 p.p.).
- O XGBoost da V2, reajustado no dia 7 com treino 12–1991 e os hiperparâmetros gravados, reproduz as previsões preservadas (diferença máxima de cerca de 7e-15 p.p.).
- Os arquivos de `results/v2-full` e `results/v3` têm o SHA-256 conferido no início e no fim da corrida.

## Decisões de desenho registradas durante a implementação

- O combinador condicionado da H3 é escolhido por metades cronológicas da validação interna (pesos na primeira metade, MAE na segunda), para não escolher pelo ajuste dentro da mesma amostra.
- Na rota por máquina da energia, uma máquina do conjunto fixo sem previsão do modelo global usa a própria CPU na origem; a fração é registrada.
- Nenhum resultado SPECpower_ssj2008 de 2 chips com 96 threads foi publicado em 2017–2018. O conjunto de substitutos traz o servidor da V3, o mais próximo em threads no período (NF5280 M4, 88 threads) e, como extensão rotulada, o mais antigo com 96 threads (PRIMERGY TX2550 M5, 2019).
- O ACI com γ = 0,02 pode produzir limites infinitos (nível efetivo ≤ 0); essas comparações são registradas como não calculáveis.

## Limitações

- Resultados exploratórios: dias de desenvolvimento já usados nas versões anteriores; n = 4 dias.
- O trace tem 8 dias; nenhum período independente novo existe.
- A frota oscila em conjunto (primeira componente principal com cerca de 72% da variância dos níveis por máquina), o que limita o ganho esperado de grupos e de modelos globais.
- Energia de TI modelada em servidores substitutos; diferenças entre estimativas modeladas, não erro contra medidor; sem PUE.
