"""
Análise de coerência pedagógica (TRI) a partir das respostas dos microdados do ENEM.

Na TRI, dois alunos com o MESMO número de acertos podem ter notas diferentes: quem erra questões
fáceis e acerta difíceis ("chute") recebe nota menor. Comparamos a nota de cada aluno com a
MEDIANA NACIONAL de quem teve o mesmo número de acertos naquela área. A diferença é a
"perda por incoerência" (positiva = o aluno ficou abaixo do padrão nacional).

Uso:  python coerencia.py --anos 2024 2025
Saídas: data/processed/curva_tri.parquet       (ano, area, acertos -> mediana nacional)
        data/processed/coerencia_escola.parquet (ano, co_escola -> perdas médias, melhor aluno)
"""
from __future__ import annotations

import argparse
import zipfile

import numpy as np
import pandas as pd

from pipeline import PROC, RAW, _abrir, _achar_csv, _ler_header

AREAS = ["CN", "CH", "LC", "MT"]
CHUNK = 400_000


def _matriz(s: pd.Series, largura: int) -> np.ndarray:
    """Série de strings de largura fixa -> matriz uint8 (n, largura)."""
    s = s.fillna("").str.pad(largura, fillchar=".").str[:largura]
    return np.frombuffer("".join(s.tolist()).encode("latin-1"), dtype=np.uint8).reshape(len(s), largura)


def acertos(resp: pd.Series, gab: pd.Series, lingua: pd.Series | None = None) -> np.ndarray:
    r = _matriz(resp, 45)
    if lingua is not None:  # LC: gabarito tem 50 itens = 5 inglês + 5 espanhol + 40 comuns
        g = _matriz(gab, 50)
        esp = (lingua.to_numpy() == 1)[:, None]
        g = np.concatenate([np.where(esp, g[:, 5:10], g[:, 0:5]), g[:, 10:]], axis=1)
    else:
        g = _matriz(gab, 45)
    ok = (r == g) & (g != ord("X"))  # X = item anulado
    return ok.sum(axis=1).astype(np.int16)


def ler_alunos(ano: int) -> pd.DataFrame:
    zf = zipfile.ZipFile(RAW / f"microdados_enem_{ano}.zip")
    membro = _achar_csv(zf, [rf"RESULTADOS_{ano}"])
    header = _ler_header(zf, membro)
    usar = (["CO_ESCOLA", "TP_LINGUA", "NU_NOTA_REDACAO"] + [f"TP_PRESENCA_{a}" for a in AREAS]
            + [f"NU_NOTA_{a}" for a in AREAS] + [f"TX_RESPOSTAS_{a}" for a in AREAS] + [f"TX_GABARITO_{a}" for a in AREAS])
    usar = [c for c in usar if c in header]
    partes = []
    with _abrir(zf, membro) as fh:
        for i, df in enumerate(pd.read_csv(fh, sep=";", encoding="latin-1", usecols=usar, chunksize=CHUNK,
                                           dtype={f"TX_{t}_{a}": str for t in ("RESPOSTAS", "GABARITO") for a in AREAS})):
            df = df[(df[[f"TP_PRESENCA_{a}" for a in AREAS]] == 1).all(axis=1)]
            df = df.dropna(subset=[f"NU_NOTA_{a}" for a in AREAS])
            df = df[(df[[f"NU_NOTA_{a}" for a in AREAS]] > 0).all(axis=1)]
            out = pd.DataFrame({"co_escola": df["CO_ESCOLA"].fillna(-1).astype("int64").to_numpy()})
            for a in AREAS:
                out[f"ac_{a.lower()}"] = acertos(df[f"TX_RESPOSTAS_{a}"], df[f"TX_GABARITO_{a}"],
                                                 df["TP_LINGUA"] if a == "LC" else None)
                out[f"nota_{a.lower()}"] = df[f"NU_NOTA_{a}"].to_numpy(np.float32)
            out["redacao"] = df["NU_NOTA_REDACAO"].to_numpy(np.float32)
            partes.append(out)
            print(f"\r  {ano}: chunk {i+1}", end="", flush=True)
    print()
    return pd.concat(partes, ignore_index=True)


def processar(anos: list[int]) -> None:
    curvas, escolas = [], []
    for ano in anos:
        al = ler_alunos(ano)
        print(f"[ok] {ano}: {len(al):,} alunos presentes nas 4 provas objetivas")
        for a in [x.lower() for x in AREAS]:
            gq = al.groupby(f"ac_{a}")[f"nota_{a}"]
            curva = pd.concat([gq.median().rename("mediana"), gq.size().rename("alunos"),
                               gq.quantile(0.75).rename("p75"), gq.quantile(0.9).rename("p90")], axis=1)
            curva = curva.rename_axis("acertos").reset_index()
            curva["ano"], curva["area"] = ano, a
            curvas.append(curva)
            esperado = al[f"ac_{a}"].map(curva.set_index("acertos")["mediana"])
            al[f"esp_{a}"] = esperado.astype(np.float32)
            al[f"perda_{a}"] = (al[f"esp_{a}"] - al[f"nota_{a}"]).astype(np.float32)
            # referência "alta coerência": percentil 75 de quem teve o mesmo nº de acertos
            al[f"esp75_{a}"] = al[f"ac_{a}"].map(curva.set_index("acertos")["p75"]).astype(np.float32)
            al[f"perda75_{a}"] = (al[f"esp75_{a}"] - al[f"nota_{a}"]).astype(np.float32)
        al["ac_total"] = al[[f"ac_{a.lower()}" for a in AREAS]].sum(axis=1)
        al["nota_obj"] = al[[f"nota_{a.lower()}" for a in AREAS]].mean(axis=1)
        al["esp_obj"] = al[[f"esp_{a.lower()}" for a in AREAS]].mean(axis=1)
        al["perda_obj"] = al["esp_obj"] - al["nota_obj"]
        al["esp75_obj"] = al[[f"esp75_{a.lower()}" for a in AREAS]].mean(axis=1)
        al["perda75_obj"] = al["esp75_obj"] - al["nota_obj"]

        e = al[al["co_escola"] > 0]
        g = e.groupby("co_escola")
        agg = g[[f"perda_{a.lower()}" for a in AREAS] + [f"perda75_{a.lower()}" for a in AREAS]
                + ["perda_obj", "perda75_obj", "ac_total"]].mean()
        agg["pct_alunos_com_perda10"] = g["perda_obj"].apply(lambda s: (s > 10).mean())
        agg["pct75_alunos_com_perda10"] = g["perda75_obj"].apply(lambda s: (s > 10).mean())
        # melhor aluno da escola (maior nº de acertos; desempate pela nota)
        top = e.sort_values(["ac_total", "nota_obj"], ascending=False).drop_duplicates("co_escola").set_index("co_escola")
        agg = agg.join(top[["ac_total", "nota_obj", "esp_obj", "esp75_obj"]].rename(columns={
            "ac_total": "melhor_acertos", "nota_obj": "melhor_nota_obj", "esp_obj": "melhor_esperada_obj",
            "esp75_obj": "melhor_esperada75_obj"}))
        agg = agg.rename(columns={"ac_total": "media_acertos"})
        agg["ano"] = ano
        escolas.append(agg.reset_index())
        nac = al["perda_obj"].mean()
        print(f"     perda média nacional por prova: {nac:.1f} pts (≈0 por construção)")
    pd.concat(curvas).to_parquet(PROC / "curva_tri.parquet", index=False)
    pd.concat(escolas).to_parquet(PROC / "coerencia_escola.parquet", index=False)
    print(f"[ok] -> {PROC/'coerencia_escola.parquet'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anos", nargs="+", type=int, default=[2024, 2025])
    processar(p.parse_args().anos)
