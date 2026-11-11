"""Gera microdados FICTÍCIOS no formato do INEP para testar o pipeline e o painel sem baixar ~GBs.

Cria zips em data_amostra/raw (ENEM 2021-2023 sem CO_ESCOLA, 2024-2025 com RESULTADOS + CO_ESCOLA,
e um Censo Escolar) e roda pipeline.processar -> data/processed/escolas_ano.parquet.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data_amostra" / "raw"
rng = np.random.default_rng(42)

MUNIS = [("SP", 3550308, "São Paulo"), ("SP", 3509502, "Campinas"), ("SP", 3543402, "Ribeirão Preto"),
         ("RJ", 3304557, "Rio de Janeiro"), ("MG", 3106200, "Belo Horizonte"), ("BA", 2927408, "Salvador"),
         ("PR", 4106902, "Curitiba"), ("AM", 1302603, "Manaus"), ("PA", 1501402, "Belém"), ("CE", 2304400, "Fortaleza")]

escolas = []
co = 11000000
for uf, cm, mun in MUNIS:
    for i in range(rng.integers(25, 45)):
        co += 1
        dep = rng.choice([2, 4, 3, 1], p=[0.55, 0.35, 0.05, 0.05])
        adv = dep == 4 and rng.random() < 0.12
        nome = (f"COLEGIO ADVENTISTA DE {mun.upper()} {i}" if adv else
                f"{'COLEGIO' if dep == 4 else 'EE' if dep == 2 else 'IF' if dep == 1 else 'EM'} {mun.upper()} {i}")
        base = {1: 640, 2: 500, 3: 480, 4: 600}[dep] + (25 if adv else 0) + rng.normal(0, 40)
        escolas.append(dict(CO_ENTIDADE=co, NO_ENTIDADE=nome, SG_UF=uf, NO_MUNICIPIO=mun, CO_MUNICIPIO=cm,
                            TP_DEPENDENCIA=dep, TP_CATEGORIA_ESCOLA_PRIVADA=3 if adv else (1 if dep == 4 else np.nan),
                            base=base, alunos=int(rng.integers(5, 120))))
esc = pd.DataFrame(escolas)


def alunos(ano: int) -> pd.DataFrame:
    rows = []
    for _, e in esc.iterrows():
        n = max(1, int(e["alunos"] * rng.uniform(0.8, 1.2)))
        nivel = e["base"] + (ano - 2021) * rng.normal(3, 4) + rng.normal(0, 30, n)
        d = pd.DataFrame({
            "NU_SEQUENCIAL": range(n), "NU_ANO": ano, "CO_ESCOLA": e["CO_ENTIDADE"],
            "CO_MUNICIPIO_ESC": e["CO_MUNICIPIO"], "NO_MUNICIPIO_ESC": e["NO_MUNICIPIO"], "SG_UF_ESC": e["SG_UF"],
            "TP_DEPENDENCIA_ADM_ESC": e["TP_DEPENDENCIA"],
            "TP_PRESENCA_CN": 1, "TP_PRESENCA_CH": 1, "TP_PRESENCA_LC": 1, "TP_PRESENCA_MT": 1,
            "NU_NOTA_CN": (nivel - 20 + rng.normal(0, 50, n)).clip(300, 870),
            "NU_NOTA_CH": (nivel + 5 + rng.normal(0, 45, n)).clip(300, 840),
            "NU_NOTA_LC": (nivel - 5 + rng.normal(0, 40, n)).clip(300, 800),
            "NU_NOTA_MT": (nivel + 10 + rng.normal(0, 80, n)).clip(320, 980),
            "NU_NOTA_REDACAO": (np.round((nivel + 80 + rng.normal(0, 120, n)) / 20) * 20).clip(0, 1000),
        })
        ausentes = rng.random(n) < 0.08
        d.loc[ausentes, ["TP_PRESENCA_MT", "TP_PRESENCA_CN"]] = 0
        d.loc[ausentes, ["NU_NOTA_MT", "NU_NOTA_CN"]] = np.nan
        rows.append(d)
    return pd.concat(rows)


def zipar(caminho: Path, nome_csv: str, df: pd.DataFrame) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    df.to_csv(buf, sep=";", index=False)
    with zipfile.ZipFile(caminho, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"DADOS/{nome_csv}", buf.getvalue().encode("latin-1"))


for ano in [2021, 2022, 2023, 2024, 2025]:
    d = alunos(ano)
    if ano < 2024:  # modelo LGPD: sem escola
        zipar(RAW / f"microdados_enem_{ano}.zip", f"MICRODADOS_ENEM_{ano}.csv",
              d.drop(columns=["CO_ESCOLA", "CO_MUNICIPIO_ESC", "NO_MUNICIPIO_ESC", "SG_UF_ESC", "TP_DEPENDENCIA_ADM_ESC"]))
    else:
        zipar(RAW / f"microdados_enem_{ano}.zip", f"RESULTADOS_{ano}.csv", d)
zipar(RAW / "microdados_censo_escolar_2024.zip", "microdados_ed_basica_2024.csv", esc.drop(columns=["base", "alunos"]))

pipeline.RAW = RAW
pipeline.processar([2021, 2022, 2023, 2024, 2025], [2024])
