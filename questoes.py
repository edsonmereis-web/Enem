"""
Taxa de acerto por QUESTÃO por escola.

As questões são numeradas como no caderno AZUL da aplicação principal (1–45 LC, 46–90 CH, 91–135 CN, 136–180 MT).
Cada caderno (cor/aplicação) embaralha as questões; o CO_ITEM de ITENS_PROVA permite levar a resposta de qualquer
caderno para o número da questão no caderno azul. Itens que não estão no caderno azul principal (reaplicação, PPL)
ficam de fora.

Uso:  python questoes.py --anos 2024 2025
Saídas (data/processed/):
  questoes_escola_<ano>.parquet  co_escola, questao, lingua, acertos, respostas   (um arquivo por ano: limite de 25 MB)
  questoes_itens.parquet         ano, area, questao, lingua, co_item, hab, gabarito, dificuldade, acertos, respostas
                                 (acertos/respostas = Brasil)
lingua: -1 = questão comum, 0 = Inglês, 1 = Espanhol (questões 1–5 de LC).
"""
from __future__ import annotations

import argparse
import zipfile

import numpy as np
import pandas as pd

from coerencia import _matriz
from habilidades import AREAS, CHUNK, X, ler_itens
from pipeline import PROC, RAW, _abrir, _achar_csv


def mapas_itens(it: pd.DataFrame, area: str) -> tuple[np.ndarray, dict]:
    """Lista de CO_ITEM da área e, para cada (CO_PROVA, lingua), vetor posição(0–44) -> coluna do item (-1 = sem item)."""
    a = it[it["SG_AREA"] == area]
    itens = np.sort(a["CO_ITEM"].unique())
    col = {c: i for i, c in enumerate(itens)}
    out = {}
    for prova, g in a.groupby("CO_PROVA"):
        for lg in ([0, 1] if area == "LC" else [-1]):
            gg = g[(g["TP_LINGUA"] == lg) | (g["TP_LINGUA"] == -1)] if area == "LC" else g
            gg = gg[(gg["idx"] >= 0) & (gg["idx"] < 45)]
            v = np.full(45, -1, dtype=np.int32)
            v[gg["idx"].to_numpy()] = gg["CO_ITEM"].map(col).to_numpy()
            out[(int(prova), lg)] = v
    return itens, out


def processar_ano(ano: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    zf = zipfile.ZipFile(RAW / f"microdados_enem_{ano}.zip")
    it = ler_itens(zf, ano)
    info = {a: mapas_itens(it, a) for a in AREAS}
    membro = _achar_csv(zf, [rf"RESULTADOS_{ano}"])
    usar = ["CO_ESCOLA", "TP_LINGUA"] + [f"{p}_{a}" for a in AREAS for p in
                                          ("TP_PRESENCA", "CO_PROVA", "TX_RESPOSTAS", "TX_GABARITO")]
    partes = {a: [] for a in AREAS}
    nac = {a: np.zeros((2, len(info[a][0])), dtype=np.int64) for a in AREAS}
    with _abrir(zf, membro) as fh:
        for i, df in enumerate(pd.read_csv(fh, sep=";", encoding="latin-1", usecols=usar, chunksize=CHUNK,
                                           dtype={c: str for c in usar if c.startswith("TX_")})):
            for a in AREAS:
                itens, maps = info[a]
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
                certo = ((r == g) & (g != X)).astype(np.int8)
                valido = (g != X).astype(np.int8)
                prova = d[f"CO_PROVA_{a}"].astype(int).to_numpy()
                esc = d["CO_ESCOLA"].fillna(-1).astype("int64").to_numpy()
                n_it = len(itens)
                ac = np.zeros((len(d), n_it), dtype=np.int8)
                rs = np.zeros((len(d), n_it), dtype=np.int8)
                chave = pd.Series(list(zip(prova, lg)))
                for k, idx in chave.groupby(chave).groups.items():
                    if k not in maps:
                        continue
                    v = maps[k]
                    ok = np.where(v >= 0)[0]
                    idx = np.asarray(idx)
                    ac[np.ix_(idx, v[ok])] = certo[np.ix_(idx, ok)]
                    rs[np.ix_(idx, v[ok])] = valido[np.ix_(idx, ok)]
                nac[a][0] += ac.sum(axis=0)
                nac[a][1] += rs.sum(axis=0)
                m = esc > 0
                if m.any():
                    t = pd.DataFrame(np.hstack([ac[m], rs[m]]).astype(np.int32))
                    t["co_escola"] = esc[m]
                    partes[a].append(t.groupby("co_escola").sum())
            print(f"\r  {ano}: chunk {i+1}", end="", flush=True)
    print()

    # referência: caderno AZUL com mais respostas (aplicação principal); questão = CO_POSICAO nele
    ref_rows, esc_rows = [], []
    for a in AREAS:
        itens, maps = info[a]
        n_it = len(itens)
        resp_item = pd.Series(nac[a][1], index=itens)
        azul = it[(it["SG_AREA"] == a) & (it["TX_COR"].str.upper() == "AZUL")]
        prova_ref = azul.groupby("CO_PROVA")["CO_ITEM"].apply(lambda s: resp_item.reindex(s.unique()).sum()).idxmax()
        ref = azul[azul["CO_PROVA"] == prova_ref].copy()
        ref["lingua"] = ref["TP_LINGUA"] if a == "LC" else -1
        ref = ref.drop_duplicates(["CO_ITEM"])
        ref["col"] = ref["CO_ITEM"].map({c: i for i, c in enumerate(itens)})
        ref["acertos"] = nac[a][0][ref["col"]]
        ref["respostas"] = nac[a][1][ref["col"]]
        ref_rows.append(pd.DataFrame({
            "ano": ano, "area": a, "questao": ref["CO_POSICAO"].astype(int), "lingua": ref["lingua"].astype(int),
            "co_item": ref["CO_ITEM"], "hab": ref["CO_HABILIDADE"], "gabarito": ref["TX_GABARITO"],
            "dificuldade": ref["NU_PARAM_B"], "anulada": ref["IN_ITEM_ABAN"].fillna(0).astype(int),
            "acertos": ref["acertos"].astype("int64"), "respostas": ref["respostas"].astype("int64")}))
        e = pd.concat(partes[a]).groupby(level=0).sum()
        for _, rr in ref.iterrows():
            c = int(rr["col"])
            s = pd.DataFrame({"co_escola": e.index.to_numpy(), "questao": int(rr["CO_POSICAO"]),
                              "lingua": int(rr["lingua"]), "acertos": e[c].to_numpy(), "respostas": e[n_it + c].to_numpy()})
            esc_rows.append(s[s["respostas"] > 0])
        print(f"  {a}: caderno de referência {prova_ref} ({len(ref)} questões)")
    itens_ref = pd.concat(ref_rows, ignore_index=True)
    q = pd.concat(esc_rows, ignore_index=True).sort_values(["co_escola", "questao", "lingua"])
    q = q.astype({"co_escola": "int32", "questao": "int16", "lingua": "int8", "acertos": "int16", "respostas": "int16"})
    return q, itens_ref


def processar(anos: list[int]) -> None:
    refs = []
    for ano in anos:
        q, ref = processar_ano(ano)
        q.to_parquet(PROC / f"questoes_escola_{ano}.parquet", index=False, compression="zstd")
        print(f"[ok] questoes_escola_{ano}: {len(q):,} linhas")
        refs.append(ref)
    pd.concat(refs, ignore_index=True).to_parquet(PROC / "questoes_itens.parquet", index=False)
    print("[ok] questoes_itens")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anos", nargs="+", type=int, default=[2024, 2025])
    processar(p.parse_args().anos)
