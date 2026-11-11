"""
Pipeline dos microdados do ENEM -> ranking de escolas.

Uso:
    python pipeline.py download --anos 2021 2022 2023 2024 2025 --censo 2024
    python pipeline.py process  --anos 2021 2022 2023 2024 2025 --censo 2024

Saída: data/processed/escolas_ano.parquet (uma linha por escola x ano)

Notas metodológicas
- Fonte: microdados do ENEM (INEP). A partir de 2024 o INEP divide o pacote em
  PARTICIPANTES_AAAA.csv e RESULTADOS_AAAA.csv; o código da escola (CO_ESCOLA)
  fica no arquivo de RESULTADOS. Nos pacotes 2020-2023 (modelo simplificado
  por causa da LGPD) o CO_ESCOLA não foi publicado: esses anos são detectados
  automaticamente e ficam fora do ranking por escola (aviso no log).
- Nomes das escolas e identificação da rede adventista vêm do Censo Escolar
  (CO_ENTIDADE = CO_ESCOLA).
- Só entram participantes presentes nos 4 cadernos e com as 5 notas; a média
  geral é a média simples de CN, CH, LC, MT e Redação.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

ENEM_URL = "https://download.inep.gov.br/microdados/microdados_enem_{ano}.zip"
CENSO_URL = "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_{ano}.zip"

AREAS = ["CN", "CH", "LC", "MT", "REDACAO"]
NOTA_COLS = [f"NU_NOTA_{a}" for a in AREAS]
PRESENCA_COLS = ["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]
ESC_COLS = ["CO_ESCOLA", "CO_MUNICIPIO_ESC", "NO_MUNICIPIO_ESC", "SG_UF_ESC", "TP_DEPENDENCIA_ADM_ESC"]

DEPENDENCIA = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}

# Padrões usados para reconhecer escolas da rede adventista pelo nome no Censo.
# Ajuste/complete se notar alguma escola faltando (ou use data/adventistas_extra.csv
# com uma coluna CO_ESCOLA para forçar a inclusão).
ADVENTISTA_RE = re.compile(r"ADVENTISTA|\bIASP\b|\bUNASP\b|\bIAE\b|\bIAP\b|\bIABC\b|EDUCANDARIO NORDESTINO", re.I)

CHUNK = 500_000


# --------------------------------------------------------------------------- download
def baixar(url: str, destino: Path, inseguro: bool = False) -> None:
    import requests

    if destino.exists() and destino.stat().st_size > 0:
        print(f"[ok] já existe: {destino.name}")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".part")
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=120, verify=not inseguro) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        feito = 0
        with open(tmp, "wb") as f:
            for bloco in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(bloco)
                feito += len(bloco)
                if total:
                    print(f"\r  {feito/1e6:,.0f} / {total/1e6:,.0f} MB", end="", flush=True)
    print()
    tmp.rename(destino)


# --------------------------------------------------------------------------- ENEM
def _achar_csv(zf: zipfile.ZipFile, padroes: list[str]) -> str | None:
    nomes = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    for p in padroes:
        for n in nomes:
            if re.search(p, Path(n).name, re.I):
                return n
    return None


@contextlib.contextmanager
def _abrir(zf: zipfile.ZipFile, membro: str):
    """Abre um membro do zip. Alguns pacotes do INEP usam Deflate64 (tipo 9), que o zipfile
    do Python não suporta – nesse caso lê via `unzip -p` do sistema."""
    if zf.getinfo(membro).compress_type == 9:
        proc = subprocess.Popen(["unzip", "-p", zf.filename, membro], stdout=subprocess.PIPE)
        try:
            yield proc.stdout
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()
    else:
        with zf.open(membro) as fh:
            yield fh


def _ler_header(zf: zipfile.ZipFile, membro: str) -> list[str]:
    with _abrir(zf, membro) as fh:
        linha = io.TextIOWrapper(fh, encoding="latin-1").readline()
    return [c.strip().strip('"') for c in linha.split(";")]


def agregar_enem(ano: int) -> pd.DataFrame | None:
    caminho = RAW / f"microdados_enem_{ano}.zip"
    if not caminho.exists():
        print(f"[aviso] {caminho.name} não encontrado – rode o download antes.")
        return None
    zf = zipfile.ZipFile(caminho)
    membro = _achar_csv(zf, [rf"RESULTADOS_{ano}", rf"MICRODADOS_ENEM_{ano}"])
    if membro is None:
        print(f"[aviso] {ano}: nenhum CSV de resultados no zip.")
        return None
    header = _ler_header(zf, membro)
    if "CO_ESCOLA" not in header:
        print(f"[aviso] {ano}: '{Path(membro).name}' não tem CO_ESCOLA (modelo LGPD) – ano ignorado no ranking por escola.")
        return None

    usar = [c for c in ESC_COLS + NOTA_COLS + PRESENCA_COLS if c in header]
    print(f"[process] ENEM {ano}: {Path(membro).name}")
    partes = []
    with _abrir(zf, membro) as fh:
        leitor = pd.read_csv(fh, sep=";", encoding="latin-1", usecols=usar, chunksize=CHUNK, low_memory=False)
        for i, df in enumerate(leitor):
            df = df[df["CO_ESCOLA"].notna()]
            for c in PRESENCA_COLS:
                if c in df:
                    df = df[df[c] == 1]
            df = df.dropna(subset=NOTA_COLS)
            df = df[(df[["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT"]] > 0).all(axis=1)]
            if df.empty:
                continue
            df = df.assign(GERAL=df[NOTA_COLS].mean(axis=1))
            for c in NOTA_COLS + ["GERAL"]:
                df[c + "_SQ"] = df[c] ** 2
            df["ACIMA_700"] = (df["GERAL"] >= 700).astype(int)
            df["RED_900"] = (df["NU_NOTA_REDACAO"] >= 900).astype(int)
            df["CO_ESCOLA"] = df["CO_ESCOLA"].astype("int64")
            soma_cols = NOTA_COLS + ["GERAL"] + [c + "_SQ" for c in NOTA_COLS + ["GERAL"]] + ["ACIMA_700", "RED_900"]
            g = df.groupby("CO_ESCOLA")
            agg = g[soma_cols].sum()
            agg["N"] = g.size()
            meta_cols = [c for c in ESC_COLS[1:] if c in df]
            agg = agg.join(g[meta_cols].first()) if meta_cols else agg
            partes.append(agg)
            print(f"\r  chunk {i+1}", end="", flush=True)
    print()
    if not partes:
        return None

    tudo = pd.concat(partes)
    meta_cols = [c for c in ESC_COLS[1:] if c in tudo]
    somas = tudo.drop(columns=meta_cols).groupby(level=0).sum()
    meta = tudo[meta_cols].groupby(level=0).first()
    out = pd.DataFrame(index=somas.index)
    out["n"] = somas["N"]
    for c, nome in zip(NOTA_COLS + ["GERAL"], ["cn", "ch", "lc", "mt", "redacao", "geral"]):
        media = somas[c] / somas["N"]
        out[f"media_{nome}"] = media
        var = (somas[c + "_SQ"] / somas["N"] - media**2).clip(lower=0)
        out[f"dp_{nome}"] = var**0.5
    out["pct_acima_700"] = somas["ACIMA_700"] / somas["N"]
    out["pct_redacao_900"] = somas["RED_900"] / somas["N"]
    out = out.join(meta)
    out["ano"] = ano
    return out.reset_index().rename(columns={
        "CO_ESCOLA": "co_escola", "CO_MUNICIPIO_ESC": "co_municipio", "NO_MUNICIPIO_ESC": "municipio_enem",
        "SG_UF_ESC": "uf_enem", "TP_DEPENDENCIA_ADM_ESC": "dep_enem",
    })


# --------------------------------------------------------------------------- Censo
def ler_censo(ano: int) -> pd.DataFrame | None:
    caminho = RAW / f"microdados_censo_escolar_{ano}.zip"
    if not caminho.exists():
        print(f"[aviso] {caminho.name} não encontrado – sem nomes de escola.")
        return None
    zf = zipfile.ZipFile(caminho)
    membro = _achar_csv(zf, [rf"microdados_ed_basica_{ano}", r"ed_basica", r"escolas"])
    if membro is None:
        print("[aviso] CSV de escolas não encontrado no zip do Censo.")
        return None
    header = _ler_header(zf, membro)
    usar = [c for c in ["CO_ENTIDADE", "NO_ENTIDADE", "SG_UF", "NO_MUNICIPIO", "CO_MUNICIPIO",
                        "TP_DEPENDENCIA", "TP_CATEGORIA_ESCOLA_PRIVADA"] if c in header]
    print(f"[process] Censo {ano}: {Path(membro).name}")
    with _abrir(zf, membro) as fh:
        df = pd.read_csv(fh, sep=";", encoding="latin-1", usecols=usar, low_memory=False)
    return df.rename(columns={
        "CO_ENTIDADE": "co_escola", "NO_ENTIDADE": "nome", "SG_UF": "uf", "NO_MUNICIPIO": "municipio",
        "CO_MUNICIPIO": "co_municipio_censo", "TP_DEPENDENCIA": "dep_censo",
        "TP_CATEGORIA_ESCOLA_PRIVADA": "cat_privada",
    })


# --------------------------------------------------------------------------- montagem
def processar(anos: list[int], anos_censo: list[int]) -> Path:
    PROC.mkdir(parents=True, exist_ok=True)
    frames = [f for f in (agregar_enem(a) for a in anos) if f is not None]
    if not frames:
        sys.exit("Nenhum ano com CO_ESCOLA disponível – nada para ranquear.")
    enem = pd.concat(frames, ignore_index=True)

    # Censo: o ano mais recente tem prioridade; anos anteriores completam escolas fechadas.
    censos = [c for c in (ler_censo(a) for a in sorted(anos_censo, reverse=True)) if c is not None]
    if censos:
        censo = pd.concat(censos).drop_duplicates("co_escola", keep="first")
        enem = enem.merge(censo, on="co_escola", how="left")
    for col in ["nome", "uf", "municipio", "dep_censo", "cat_privada"]:
        if col not in enem:
            enem[col] = pd.NA

    # CO_ESCOLA é autodeclarado pelo participante; códigos ausentes do Censo ficam marcados
    enem["no_censo"] = enem["nome"].notna()
    enem["nome"] = enem["nome"].fillna("NÃO LOCALIZADA NO CENSO (cód. " + enem["co_escola"].astype(str) + ")")
    enem["uf"] = enem["uf"].fillna(enem.get("uf_enem"))
    enem["municipio"] = enem["municipio"].fillna(enem.get("municipio_enem"))
    dep = enem["dep_censo"].fillna(enem.get("dep_enem"))
    enem["dependencia"] = dep.map(DEPENDENCIA).fillna("—")

    extra = ROOT / "data" / "adventistas_extra.csv"
    forcar = set(pd.read_csv(extra)["CO_ESCOLA"].astype(int)) if extra.exists() else set()
    enem["adventista"] = enem["nome"].str.contains(ADVENTISTA_RE, na=False) | enem["co_escola"].isin(forcar)

    cols = ["ano", "co_escola", "nome", "uf", "municipio", "dependencia", "adventista", "no_censo", "n",
            "media_geral", "media_cn", "media_ch", "media_lc", "media_mt", "media_redacao",
            "dp_geral", "dp_cn", "dp_ch", "dp_lc", "dp_mt", "dp_redacao",
            "pct_acima_700", "pct_redacao_900"]
    enem = enem[cols].sort_values(["ano", "media_geral"], ascending=[True, False])
    destino = PROC / "escolas_ano.parquet"
    enem.to_parquet(destino, index=False)
    print(f"[ok] {len(enem):,} linhas escola-ano -> {destino}")
    print(enem.groupby("ano").agg(escolas=("co_escola", "nunique"), alunos=("n", "sum"),
                                  adventistas=("adventista", "sum")).to_string())
    return destino


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("etapa", choices=["download", "process", "all"])
    p.add_argument("--anos", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    p.add_argument("--censo", nargs="+", type=int, default=[2024],
                   help="ano(s) do Censo Escolar usados para nomes das escolas")
    p.add_argument("--inseguro", action="store_true",
                   help="não validar o certificado SSL do INEP (o servidor às vezes tem cadeia incompleta)")
    a = p.parse_args()

    if a.etapa in ("download", "all"):
        for ano in a.anos:
            try:
                baixar(ENEM_URL.format(ano=ano), RAW / f"microdados_enem_{ano}.zip", a.inseguro)
            except Exception as e:  # noqa: BLE001
                print(f"[erro] ENEM {ano}: {e}")
        for ano in a.censo:
            try:
                baixar(CENSO_URL.format(ano=ano), RAW / f"microdados_censo_escolar_{ano}.zip", a.inseguro)
            except Exception as e:  # noqa: BLE001
                print(f"[erro] Censo {ano}: {e}")
    if a.etapa in ("process", "all"):
        processar(a.anos, a.censo)


if __name__ == "__main__":
    main()
