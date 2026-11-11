# Ranking ENEM por Escola (Brasil + Rede Adventista)

```bash
cd enem_ranking
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1) baixar microdados do INEP (ENEM + Censo Escolar para os nomes das escolas)
.venv/bin/python pipeline.py download --anos 2021 2022 2023 2024 2025 --censo 2024
#    (se der erro de certificado SSL do INEP, acrescente --inseguro)

# 2) agregar por escola -> data/processed/escolas_ano.parquet
.venv/bin/python pipeline.py process --anos 2021 2022 2023 2024 2025 --censo 2024

# 3) abrir o painel
.venv/bin/streamlit run app.py
```

Para testar sem baixar nada: `.venv/bin/python gerar_amostra.py` (gera dados **fictícios** e **sobrescreve** `data/processed/escolas_ano.parquet` — rode `pipeline.py process` de novo para voltar aos dados reais).

## Arquivos
- `pipeline.py` – download e agregação (lê os CSVs direto do zip, em blocos, sem descompactar).
- `analise.py` – ranking, média de vários anos e diagnóstico "Top 5" (gap por área, metas, tendência).
- `app.py` – painel Streamlit: Ranking geral · Rede Adventista · Como chegar ao Top 5 · Metodologia.
- `data/adventistas_extra.csv` (opcional) – coluna `CO_ESCOLA` para marcar escolas adventistas que o nome não identifica.

## Limitação importante
O INEP retirou o código da escola (`CO_ESCOLA`) dos microdados de 2020–2023 (LGPD). Esses anos são detectados e
ignorados no ranking por escola. A partir de 2024 o pacote vem dividido em PARTICIPANTES/RESULTADOS e o pipeline
procura o `CO_ESCOLA` no arquivo de RESULTADOS (confirmado em 2024 e 2025).

Outros detalhes:
- O `RESULTADOS_2025.csv` vem compactado em Deflate64, que o Python não lê; o pipeline usa o `unzip` do sistema nesse caso.
- O código da escola é autodeclarado pelo participante: cerca de 7 mil pares escola-ano (34 mil alunos) têm códigos que não
  existem no Censo 2024 nem no 2023. Eles aparecem como "NÃO LOCALIZADA NO CENSO" e ficam ocultos por padrão no painel.
- Com só 2 anos de dados por escola, a tendência e a projeção de "anos até o Top 5" são apenas indicativas.

## Relatório em PDF por escola
- Pré-requisito (uma vez): `.venv/bin/python coerencia.py --anos 2024 2025` — calcula acertos de cada aluno a partir
  das respostas/gabaritos e a perda de pontos por incoerência na TRI (~1 min).
- No painel: aba **Como chegar ao Top 5** → escolha a escola → **Gerar relatório PDF** → **Baixar PDF**.
- Linha de comando: `.venv/bin/python relatorio.py <CO_ESCOLA> [ano]` → `relatorios/`.
- Referência de coerência: *percentil 75* (nota de alunos coerentes com o mesmo nº de acertos; padrão) ou *mediana nacional*.
- O texto "Por que é tão importante investir nisto?" é editável no painel (ou deixe vazio para omitir).

## Análise por habilidade (descritores da Matriz de Referência)
- `matriz.py` extrai as 120 habilidades (H1–H30 × 4 áreas) do PDF oficial do INEP (`data/ref/matriz.pdf`) → `data/ref/habilidades.csv`.
- `habilidades.py --anos 2024 2025` cruza `ITENS_PROVA` (caderno + posição → habilidade e dificuldade TRI) com as respostas
  de cada aluno → % de acerto por habilidade por escola, Brasil e itens fáceis (~1,5 min).
- Painel: aba **🧩 Habilidades**. PDF: conclusão nº 4 + tabela de habilidades prioritárias e questões fáceis.
- Cada habilidade aparece em 1–3 questões por prova: somamos os anos e marcamos ⚠️ quando a escola tem < 30 respostas.
- Em 2025, 5 cadernos (≈2% dos alunos, provavelmente a reaplicação) não constam no ITENS_PROVA e ficam fora desta análise.
