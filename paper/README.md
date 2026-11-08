# Artigo principal

Esta pasta contém uma cópia autocontida das fontes do artigo principal, os arquivos
locais do template ABakos/abnTeX2, a bibliografia, as figuras usadas no texto e o PDF
compilado atualizado.

## Compilação

Com uma distribuição LaTeX que inclua `latexmk`, execute dentro desta pasta:

```bash
latexmk -pdf principal.tex
```

## Figuras experimentais

As figuras experimentais podem ser regeneradas a partir dos resultados preservados
em `../results/`, executando da raiz `Tcc-sourcecode` após instalar o projeto:

```bash
python -m pip install -e ".[test,v3,analysis]"
python paper/figuras/gerar_figuras_artigo.py
```

O script atualiza os PDFs experimentais nesta pasta. A imagem institucional
`figuras/pucmg.png` é um recurso estático usado pelo artigo e não é gerada pelo
pipeline experimental.
