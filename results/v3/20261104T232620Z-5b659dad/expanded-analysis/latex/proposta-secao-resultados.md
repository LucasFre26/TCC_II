# Proposta de reorganização da seção de resultados (não aplicada a principal.tex)

Esta proposta organiza os resultados finais a partir da análise expandida. O artigo só deve ser alterado depois da revisão científica destes números. Tempo verbal: passado para o que foi executado (V1, V2, V3 e análise expandida); nenhuma etapa pendente é descrita como feita.

## Estrutura sugerida

1. **Dados e protocolo (curta).** Série agregada, partições e mudança de nível do teste (Figura `02_distribuicao_dependencia`). Declarar que o teste de 24 h foi reutilizado da V2 e não é independente do desenvolvimento.
2. **Desempenho pontual.** Tabela `tab:v3-metricas-teste`, Figura `05_ranking_mae_rmse` e Tabela `tab:v3-validacao-teste` (validação × teste, Figura `07_validacao_vs_teste`).
3. **Inferência.** Tabela `tab:v3-comparacoes` e Figura `18_forest_preregistrado`. Texto: o híbrido não se distinguiu do XGBoost da V2; as redes DLinear, TCN e N-BEATSx foram piores com IC sem zero. Mencionar Holm como sensibilidade.
4. **Por que as redes não superaram o XGBoost.** Figura `21_generalizacao_janela`, Figura `14_erro_variacao_regime` e Tabela `tab:v3-redes`. Separar o observado (viés positivo, razão de generalização maior que 1) das hipóteses (mudança de nível, poucas origens, ancoragem em cpu(t)).
5. **Picos.** Figura `16_subestimacao_picos`, Figura `17_alertas_picos` e Tabela `tab:v3-picos`. Não chamar o teto quantílico de detector de picos.
6. **Previsão probabilística.** Tabela `tab:v3-quantis` e Figura `25_calibracao_quantis`; cobertura empírica sem garantia teórica.
7. **Energia de TI modelada.** Tabela `tab:v3-energia`, Figura `28_potencia_energia_acumulada`, Figura `29_erro_energia_sensibilidade`, Tabela `tab:v3-jensen` e Figura `31_jensen`. Sempre "energia de TI modelada num servidor substituto"; PUE fora.
8. **Ablação e importância.** Tabela `tab:v3-ablacao` e Figura `23_ablacao`; importância preditiva não é causalidade.
9. **Limitações.** Itens da Seção 11 de `scientific-analysis.md`.

## Como incluir

- Copiar `latex/figuras/*.pdf` para `Tcc LaTeX/figuras/v3/`.
- Incluir as tabelas com `\input{tabelas/v3_metricas_teste.tex}` (ou colar o conteúdo) e as figuras a partir de `latex/figuras.tex`.
- As tabelas usam só `tabular` e `\hline`; nenhum pacote novo é necessário.
