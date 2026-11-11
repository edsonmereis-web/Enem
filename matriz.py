"""Extrai competências e habilidades (H1–H30) da Matriz de Referência do ENEM (PDF oficial do INEP)
-> data/ref/habilidades.csv (area, competencia, desc_competencia, hab, desc_hab)."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
AREAS = {"Linguagens": "LC", "Matemática": "MT", "Ciências da Natureza": "CN", "Ciências Humanas": "CH"}


def extrair(txt: str) -> pd.DataFrame:
    linhas = [l.strip() for l in txt.replace("\r", "\n").split("\n")]
    area = comp = None
    comp_txt, rows, atual = {}, [], None
    for l in linhas:
        m = re.match(r"Matriz de Referência de (.+?) e suas Tecnologias", l)
        if m:
            area = next((v for k, v in AREAS.items() if m.group(1).startswith(k)), None)
            atual = None
            continue
        if area and re.match(r"(Objetos de conhecimento|Anexo|EIXOS COGNITIVOS)", l, re.I):
            area, atual = None, None  # fim da parte de habilidades
            continue
        if not area or not l:
            continue
        m = re.match(r"Competência de área (\d+)\s*[-–]\s*(.*)", l)
        if m:
            comp = int(m.group(1)); comp_txt[(area, comp)] = m.group(2); atual = ("C", area, comp)
            continue
        m = re.match(r"H(\d+)\s*[-–]\s*(.*)", l)
        if m:
            rows.append({"area": area, "competencia": comp, "hab": int(m.group(1)), "desc_hab": m.group(2)})
            atual = ("H", len(rows) - 1)
            continue
        if atual and atual[0] == "H":
            rows[atual[1]]["desc_hab"] += " " + l
        elif atual and atual[0] == "C":
            comp_txt[(atual[1], atual[2])] += " " + l
    df = pd.DataFrame(rows).drop_duplicates(["area", "hab"])
    df["desc_competencia"] = [comp_txt[(a, c)] for a, c in zip(df["area"], df["competencia"])]
    for c in ("desc_hab", "desc_competencia"):
        df[c] = df[c].str.replace(r"\s+", " ", regex=True).str.replace(r"(?<=\w)-\s+(?=[a-zà-ú])", "", regex=True).str.strip()
    return df[["area", "competencia", "desc_competencia", "hab", "desc_hab"]]


if __name__ == "__main__":
    t = (ROOT / "data" / "ref" / "matriz.txt").read_text()
    df = extrair(t)
    df.to_csv(ROOT / "data" / "ref" / "habilidades.csv", index=False)
    print(df.groupby("area").agg(habs=("hab", "nunique"), comps=("competencia", "nunique"), maxh=("hab", "max")))
