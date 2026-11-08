# Limitações da V3

- O bloco de teste final de 24 h é o mesmo já inspecionado na V2; reutilizá-lo não constitui um teste independente novo, e nenhum segundo teste foi recortado dentro dele.
- A análise principal agrega todas as máquinas da grade com metadado, sem filtro de status; a coorte usa só máquinas ativas com o mesmo cpu_num. As duas populações diferem (ver population_audit.json) e suas métricas são reportadas em separado.
- Potência e energia são estimativas paramétricas do modelo de Fan num servidor substituto do SPECpower_ssj2008: a carga do benchmark não é a utilização de CPU do trace, e o hardware não é o do cluster. O trace não fornece potência nem energia medidas.
- A potência calculada a partir da CPU agregada é a de um servidor substituto, não a potência total da frota; a soma das máquinas presentes aparece em coluna separada.
- Os passos 2, 4 e 5 das trajetórias de Ridge, XGBoost da V2, incremento e híbrido herdam o hiperparâmetro dos passos 3 e 6 (marca inherited_params); não houve busca nova para eles.
- A cobertura dos intervalos quantílicos é empírica; sob dependência temporal não há garantia teórica.
- As variâncias do modelo de espaço de estados são estimadas por máxima verossimilhança no segmento de treino; o estado na origem é sempre filtrado.
- A diferença entre energia projetada e de referência é entre duas grandezas modeladas, não erro contra medidor, e não é intervalo de confiança da energia elétrica do cluster.
- PUE não foi calculado.
