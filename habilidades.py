"""
Acerto por habilidade (descritor H1–H30 da Matriz de Referência) por escola.

Cruza ITENS_PROVA (caderno + posição -> item, habilidade, dificuldade TRI) com as respostas de cada aluno.

Uso:  python habilidades.py --anos 2024 2025
Saídas (data/processed/):
  habilidades_escola.parquet   ano, co_escola, area, hab, acertos, respostas
  habilidades_nacional.parquet ano, area, hab, acertos, respostas
  faceis_escola.parquet        ano, co_escola, area, acertos, respostas  (só itens fáceis)
  faceis_nacional.parquet      ano, area, acertos, respostas
"""
from __future__ import annotations

import argparse
import zipfile

import numpy as np
import pandas as pd

from coerencia import _matriz
from pipeline import PROC, RAW, _abrir, _achar_csv

AREAS = ["CN", "CH", "LC", "MT"]
OFFSET = {"LC": 1, "CH": 46, "CN": 91, "MT": 136}
NHAB = 30
CHUNK = 400_000
X = ord("X")


def ler_itens(zf: zipfile.ZipFile, ano: int) -> pd.DataFrame:
    membro = _achar_csv(zf, [rf"ITENS_PROVA_{ano}"])
    with _abrir(zf, membro) as fh:
        it = pd.read_csv(fh, sep=";", encoding="latin-1")
    it["TP_LINGUA"] = it["TP_LINGUA"].fillna(-1).astype(int)
    it = it.drop_duplicates(["CO_PROVA", "CO_POSICAO", "TP_LINGUA"])
    it["idx"] = it["CO_POSICAO"] - it["SG_AREA"].map(OFFSET)
    # item fácil = 25% de menor dificuldade (parâmetro b da TRI) dentro da área no ano
    corte = it.groupby("SG_AREA")["NU_PARAM_B"].transform(lambda s: s.quantile(0.25))
    it["facil"] = it["NU_PARAM_B"] <= corte
    return it


def mapas(it: pd.DataFrame, area: str) -> dict:
    """(CO_PROVA, lingua) -> (hab[45] 0-based ou -1, facil[45] bool)."""
    out = {}
    a = it[it["SG_AREA"] == area]
    for prova, g in a.groupby("CO_PROVA"):
        linguas = [0, 1] if area == "LC" else [-1]
        for lg in linguas:
            gg = g[(g["TP_LINGUA"] == lg) | (g["TP_LINGUA"] == -1)] if area == "LC" else g
            gg = gg[(gg["idx"] >= 0) & (gg["idx"] < 45)]
            hab = np.full(45, -1, dtype=np.int16)
            fac = np.zeros(45, dtype=bool)
            hab[gg["idx"].to_numpy()] = gg["CO_HABILIDADE"].to_numpy() - 1
            fac[gg["idx"].to_numpy()] = gg["facil"].to_numpy()
            out[(int(prova), lg)] = (hab, fac)
    return out


def processar_ano(ano: int):
    zf = zipfile.ZipFile(RAW / f"microdados_enem_{ano}.zip")
    it = ler_itens(zf, ano)
    maps = {a: mapas(it, a) for a in AREAS}
    membro = _achar_csv(zf, [rf"RESULTADOS_{ano}"])
    usar = ["CO_ESCOLA", "TP_LINGUA"] + [f"{p}_{a}" for a in AREAS for p in
                                          ("TP_PRESENCA", "CO_PROVA", "TX_RESPOSTAS", "TX_GABARITO")]
    esc_parts, nac = [], {a: np.zeros((2, NHAB), dtype=np.int64) for a in AREAS}
    fac_parts, fac_nac = [], {a: np.zeros(2, dtype=np.int64) for a in AREAS}
    sem_mapa = 0
    with _abrir(zf, membro) as fh:
        for i, df in enumerate(pd.read_csv(fh, sep=";", encoding="latin-1", usecols=usar, chunksize=CHUNK,
                                           dtype={c: str for c in usar if c.startswith("TX_")})):
            for a in AREAS:
                d = df[df[f"TP_PRESENCA_{a}"] == 1].dropna(subset=[f"CO_PROVA_{a}", f"TX_RESPOSTAS_{a}"])
                if d.empty:
                    continue
                r = _matriz(d[f"TX_RESPOSTAS_{a}"], 45)
                lg = d["TP_LINGUA"].fillna(0).astype(int).to_numpy()
                if a == "LC":
                    g = _matriz(d[f"TX_GABARITO_{a}"], 50)
                    g = np.concatenate([np.where((lg == 1)[:, None], g[:, 5:10], g[:, 0:5]), g[:, 10:]], axis=1)
                else:
                    g = _matriz(d[f"TX_GABARITO_{a}"], 45)
                    lg = np.full(len(d), -1)
                certo = (r == g) & (g != X)
                valido = g != X
                prova = d[f"CO_PROVA_{a}"].astype(int).to_numpy()
                esc = d["CO_ESCOLA"].fillna(-1).astype("int64").to_numpy()
                ac_h = np.zeros((len(d), NHAB), dtype=np.int16)
                rs_h = np.zeros((len(d), NHAB), dtype=np.int16)
                ac_f = np.zeros(len(d), dtype=np.int16)
                rs_f = np.zeros(len(d), dtype=np.int16)
                chave = pd.Series(list(zip(prova, lg)))
                for k, idx in chave.groupby(chave).groups.items():
                    if k not in maps[a]:
                        sem_mapa += len(idx)
                        continue
                    hab, fac = maps[a][k]
                    ok = hab >= 0
                    onehot = np.zeros((45, NHAB), dtype=np.int16)
                    onehot[np.where(ok)[0], hab[ok]] = 1
                    idx = np.asarray(idx)
                    ac_h[idx] = certo[idx].astype(np.int16) @ onehot
                    rs_h[idx] = valido[idx].astype(np.int16) @ onehot
                    ac_f[idx] = (certo[idx] & fac).sum(axis=1)
                    rs_f[idx] = (valido[idx] & fac).sum(axis=1)
                nac[a][0] += ac_h.sum(axis=0)
                nac[a][1] += rs_h.sum(axis=0)
                fac_nac[a] += [ac_f.sum(), rs_f.sum()]
                m = esc > 0
                if m.any():
                    cols = [f"a{h}" for h in range(NHAB)] + [f"r{h}" for h in range(NHAB)]
                    t = pd.DataFrame(np.hstack([ac_h[m], rs_h[m]]), columns=cols)
                    t["fa"], t["fr"], t["co_escola"] = ac_f[m], rs_f[m], esc[m]
                    s = t.groupby("co_escola").sum()
                    s["area"] = a
                    esc_parts.append(s.reset_index())
            print(f"\r  {ano}: chunk {i+1}", end="", flush=True)
    print()
    if sem_mapa:
        print(f"  [aviso] {sem_mapa:,} respostas sem caderno correspondente em ITENS_PROVA")
    e = pd.concat(esc_parts).groupby(["co_escola", "area"]).sum().reset_index()
    longo = []
    for h in range(NHAB):
        longo.append(pd.DataFrame({"co_escola": e["co_escola"], "area": e["area"], "hab": h + 1,
                                   "acertos": e[f"a{h}"], "respostas": e[f"r{h}"]}))
    hab_esc = pd.concat(longo)
    hab_esc = hab_esc[hab_esc["respostas"] > 0]
    hab_esc.insert(0, "ano", ano)
    hab_nac = pd.concat([pd.DataFrame({"ano": ano, "area": a, "hab": np.arange(1, NHAB + 1),
                                       "acertos": nac[a][0], "respostas": nac[a][1]}) for a in AREAS])
    hab_nac = hab_nac[hab_nac["respostas"] > 0]
    fac_esc = e[["co_escola", "area", "fa", "fr"]].rename(columns={"fa": "acertos", "fr": "respostas"})
    fac_esc.insert(0, "ano", ano)
    fac_n = pd.DataFrame([{"ano": ano, "area": a, "acertos": fac_nac[a][0], "respostas": fac_nac[a][1]} for a in AREAS])
    return hab_esc, hab_nac, fac_esc, fac_n


def processar(anos: list[int]) -> None:
    res = [processar_ano(a) for a in anos]
    nomes = ["habilidades_escola", "habilidades_nacional", "faceis_escola", "faceis_nacional"]
    for k, nome in enumerate(nomes):
        df = pd.concat([r[k] for r in res], ignore_index=True)
        if nome == "habilidades_escola":  # arquivo grande: tipos compactos (limite de 25 MB do upload web)
            df = df.sort_values(["co_escola", "area", "hab", "ano"])
            df = df.astype({"ano": "int16", "hab": "int8", "acertos": "int16", "respostas": "int16", "area": "category"})
        else:
            df = df.astype({c: "int64" for c in ("acertos", "respostas") if c in df})
        df.to_parquet(PROC / f"{nome}.parquet", index=False, compression="zstd")
        print(f"[ok] {nome}: {len(df):,} linhas")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anos", nargs="+", type=int, default=[2024, 2025])
    processar(p.parse_args().anos)
