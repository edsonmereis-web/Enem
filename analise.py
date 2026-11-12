"""Funções de ranking e da análise "como chegar ao Top 5"."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

AREAS = {"media_cn": "Ciências da Natureza", "media_ch": "Ciências Humanas",
         "media_lc": "Linguagens", "media_mt": "Matemática", "media_redacao": "Redação"}


def carregar(caminho) -> pd.DataFrame:
    return pd.read_parquet(caminho)


def ranquear(df: pd.DataFrame, min_alunos: int = 10, por: list[str] | None = None) -> pd.DataFrame:
    """Adiciona coluna 'posicao' (1 = melhor) por ano dentro do recorte `por` (ex.: ['uf'])."""
    d = df[df["n"] >= min_alunos].copy()
    grupos = ["ano"] + (por or [])
    d["posicao"] = d.groupby(grupos)["media_geral"].rank(ascending=False, method="min").astype(int)
    return d.sort_values(grupos + ["posicao"])


def media_periodo(df: pd.DataFrame, min_alunos: int = 10, min_anos: int = 1) -> pd.DataFrame:
    """Consolida vários anos numa média ponderada pelo nº de participantes."""
    d = df[df["n"] >= min_alunos].copy()
    cols = ["media_geral"] + list(AREAS) + ["pct_acima_700", "pct_redacao_900"]
    for c in cols:
        d[c] = d[c] * d["n"]
    g = d.groupby(["co_escola", "nome", "uf", "municipio", "dependencia", "adventista"], dropna=False)
    out = g[cols].sum().div(g["n"].sum(), axis=0)
    out["n"] = g["n"].sum()
    out["anos"] = g["ano"].nunique()
    out = out[out["anos"] >= min_anos].reset_index()
    return out.sort_values("media_geral", ascending=False)


def tendencia(serie_anos: pd.Series, serie_valores: pd.Series) -> float:
    """Inclinação (pontos/ano) por regressão linear simples; 0 se < 2 pontos."""
    x, y = np.asarray(serie_anos, float), np.asarray(serie_valores, float)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return 0.0
    return float(np.polyfit(x[ok], y[ok], 1)[0])


@dataclass
class DiagnosticoTop5:
    escopo: str
    ano: int
    posicao: int
    total: int
    nota: float
    corte_top5: float
    gap: float
    tabela_areas: pd.DataFrame
    top5: pd.DataFrame
    historico: pd.DataFrame
    tend_escola: float
    tend_corte: float
    anos_estimados: float | None
    recomendacoes: list[str] = field(default_factory=list)


def diagnostico_top5(df: pd.DataFrame, co_escola: int, escopo: str, ano: int | None = None,
                     min_alunos: int = 10) -> DiagnosticoTop5 | None:
    """escopo: 'municipio' ou 'uf'."""
    esc_all = df[df["co_escola"] == co_escola]
    if esc_all.empty:
        return None
    ano = ano or int(esc_all["ano"].max())
    linha = esc_all[esc_all["ano"] == ano]
    if linha.empty:
        return None
    linha = linha.iloc[0]
    chave = ["uf", "municipio"] if escopo == "municipio" else ["uf"]
    mask = np.logical_and.reduce([df[c] == linha[c] for c in chave])
    universo = df[mask]
    # a escola sempre entra no próprio ranking, mesmo abaixo do mínimo de alunos
    universo = universo[(universo["n"] >= min_alunos) | (universo["co_escola"] == co_escola)]
    rank = ranquear(universo, 0)
    r_ano = rank[rank["ano"] == ano].sort_values("posicao")
    eu = r_ano[r_ano["co_escola"] == co_escola].iloc[0]
    outros = r_ano[r_ano["co_escola"] != co_escola]
    top5 = r_ano.head(5)
    # corte = nota que garante 5º lugar = 5ª melhor nota entre as OUTRAS escolas
    corte = float(outros["media_geral"].iloc[4]) if len(outros) >= 5 else float(outros["media_geral"].min() if len(outros) else eu["media_geral"])
    gap = max(0.0, corte - float(eu["media_geral"]) + 0.01) if int(eu["posicao"]) > 5 else 0.0

    # Gap por área contra a média das 5 melhores (excluindo a própria escola)
    ref = outros.head(5)
    linhas = []
    for col, nome in AREAS.items():
        linhas.append({"area": nome, "escola": eu[col], "media_top5": ref[col].mean(),
                       "gap_top5": ref[col].mean() - eu[col]})
    t = pd.DataFrame(linhas)
    pos = t["gap_top5"].clip(lower=0)
    # Precisamos somar 5*gap pontos nas áreas (média geral = média das 5 áreas).
    # Distribui proporcionalmente onde a escola está mais atrás das líderes.
    t["ganho_necessario"] = (pos / pos.sum() * 5 * gap) if pos.sum() > 0 else 5 * gap / 5
    t["impacto_na_geral_por_10pts"] = 2.0  # +10 pts numa área = +2 pts na média geral
    t = t.sort_values("gap_top5", ascending=False)

    # Histórico: nota da escola x corte do top 5 em cada ano
    hist = []
    for a, g in rank.groupby("ano"):
        g_out = g[g["co_escola"] != co_escola].sort_values("media_geral", ascending=False)
        me = g[g["co_escola"] == co_escola]
        hist.append({"ano": a,
                     "escola": float(me["media_geral"].iloc[0]) if not me.empty else np.nan,
                     "posicao": int(me["posicao"].iloc[0]) if not me.empty else np.nan,
                     "corte_top5": float(g_out["media_geral"].iloc[4]) if len(g_out) >= 5 else np.nan,
                     "lider": float(g["media_geral"].max())})
    hist = pd.DataFrame(hist)
    te = tendencia(hist["ano"], hist["escola"])
    tc = tendencia(hist["ano"], hist["corte_top5"])
    anos_est = None
    if gap > 0 and te - tc > 0.1:
        anos_est = gap / (te - tc)

    rec = _recomendacoes(eu, t, gap, te, tc, anos_est, len(r_ano))
    if gap > 0 and hist["escola"].notna().sum() < 3:
        rec[-1] += " _(tendência calculada com poucos anos — use como indicativo.)_"
    return DiagnosticoTop5(escopo, ano, int(eu["posicao"]), len(r_ano), float(eu["media_geral"]), corte, gap,
                           t, top5, hist, te, tc, anos_est, rec)


def _recomendacoes(eu, t, gap, te, tc, anos_est, total) -> list[str]:
    rec = []
    if gap <= 0:
        rec.append("A escola já está no Top 5 deste recorte. Foco: manter a vantagem sobre o 6º colocado e reduzir variação entre turmas.")
    else:
        rec.append(f"Faltam **{gap:.1f} pontos** na média geral — o equivalente a **{5*gap:.0f} pontos** somados nas 5 áreas "
                   f"(cada +10 pts em uma área = +2 pts na média geral).")
        prio = t[t["gap_top5"] > 0].head(2)
        for _, r in prio.iterrows():
            rec.append(f"Prioridade: **{r['area']}** — {r['gap_top5']:.0f} pts abaixo da média do Top 5; "
                       f"meta sugerida de **+{r['ganho_necessario']:.0f} pts** nesta área.")
    red = t.set_index("area").loc["Redação"]
    if red["gap_top5"] > 20:
        rec.append("Redação costuma ser a alavanca mais rápida: produção semanal com correção por competência (C1–C5) "
                   "e meta de % de alunos ≥ 900 rende muitos pontos em pouco tempo.")
    if eu.get("dp_geral", 0) and eu["dp_geral"] > 70:
        rec.append(f"Alta dispersão entre alunos (desvio-padrão {eu['dp_geral']:.0f}): reforço focado no terço inferior "
                   "da turma eleva a média mais do que treinar quem já está no topo.")
    if eu["n"] < 20:
        rec.append(f"Apenas {int(eu['n'])} participantes: a média oscila muito ano a ano; incentive a presença de todos os "
                   "concluintes nos dois dias e acompanhe a tendência de vários anos, não só um.")
    if gap > 0:
        if anos_est is not None:
            rec.append(f"No ritmo atual (escola {te:+.1f} pts/ano vs. corte do Top 5 {tc:+.1f} pts/ano) o Top 5 seria "
                       f"alcançado em ~{anos_est:.1f} ano(s).")
        else:
            rec.append(f"No ritmo atual (escola {te:+.1f} pts/ano vs. corte {tc:+.1f} pts/ano) a distância não está "
                       "diminuindo — é preciso acelerar a evolução anual.")
    return rec


# ============================================================================ habilidades
NOME_AREA = {"CN": "Ciências da Natureza", "CH": "Ciências Humanas", "LC": "Linguagens", "MT": "Matemática"}


def _pct(d: pd.DataFrame, chaves: list[str], nome: str) -> pd.DataFrame:
    d = d.astype({"acertos": "int64", "respostas": "int64", "area": "object"})  # evita overflow de int16
    g = d.groupby(chaves)[["acertos", "respostas"]].sum()
    g[nome] = g["acertos"] / g["respostas"] * 100
    return g[[nome]] if nome != "pct_escola" else g.rename(columns={"respostas": "respostas_escola"})[[nome, "respostas_escola"]]


def top5_cidade(df: pd.DataFrame, co_escola: int, ano: int, min_alunos: int = 10, escopo: str = "municipio") -> list[int]:
    """Códigos das 5 melhores escolas (média geral) da cidade ou do estado, excluindo a própria escola."""
    eu = df[(df["co_escola"] == co_escola) & (df["ano"] == ano)].iloc[0]
    m = (df["ano"] == ano) & (df["uf"] == eu["uf"]) & (df["n"] >= min_alunos) & (df["co_escola"] != co_escola)
    if escopo == "municipio":
        m &= df["municipio"] == eu["municipio"]
    c = df[m]
    return c.nlargest(5, "media_geral")["co_escola"].tolist()


def perfil_habilidades(hab_esc: pd.DataFrame, hab_nac: pd.DataFrame, df: pd.DataFrame, matriz: pd.DataFrame,
                       co_escola: int, anos: list[int], ano_ref: int, min_alunos: int = 10,
                       escopo: str = "municipio") -> pd.DataFrame:
    """% de acerto por habilidade: escola × Brasil × UF × Top 5 da cidade (anos agregados)."""
    he = hab_esc[hab_esc["ano"].isin(anos)]
    eu = df[(df["co_escola"] == co_escola) & (df["ano"] == ano_ref)].iloc[0]
    k = ["area", "hab"]
    esc = _pct(he[he["co_escola"] == co_escola], k, "pct_escola")
    br = _pct(hab_nac[hab_nac["ano"].isin(anos)], k, "pct_brasil")
    esc_uf = set(df.loc[df["uf"] == eu["uf"], "co_escola"])
    uf = _pct(he[he["co_escola"].isin(esc_uf)], k, "pct_uf")
    t5 = top5_cidade(df, co_escola, ano_ref, min_alunos, escopo)
    top = _pct(he[he["co_escola"].isin(t5)], k, "pct_top5")
    out = esc.join([br, uf, top]).reset_index()
    out = out.merge(matriz, on=["area", "hab"], how="left")
    out["gap_top5"] = out["pct_top5"] - out["pct_escola"]
    out["gap_brasil"] = out["pct_brasil"] - out["pct_escola"]
    out["area_nome"] = out["area"].map(NOME_AREA)
    out["rotulo"] = out["area"] + " · H" + out["hab"].astype(str)
    out["amostra_pequena"] = out["respostas_escola"] < 30
    # ≈ questões dessa habilidade por aluno no período e acertos a mais por aluno se igualasse o Top 5
    alunos = df[(df["co_escola"] == co_escola) & df["ano"].isin(anos)]["n"].sum()
    out["questoes_por_aluno"] = out["respostas_escola"] / alunos if alunos else np.nan
    out["acertos_a_mais"] = (out["gap_top5"].clip(lower=0) / 100 * out["questoes_por_aluno"])
    return out


def perfil_competencias(perfil: pd.DataFrame, hab_esc: pd.DataFrame, co_escola: int, anos: list[int]) -> pd.DataFrame:
    """Agrega o perfil por competência (ponderando pelo nº de respostas da escola)."""
    p = perfil.copy()
    w = p["respostas_escola"]
    cols = ["pct_escola", "pct_brasil", "pct_uf", "pct_top5"]
    for c in cols:
        p[c] = p[c] * w
    g = p.groupby(["area", "area_nome", "competencia", "desc_competencia"], dropna=False)
    out = g[cols].sum().div(g["respostas_escola"].sum(), axis=0)
    out["respostas_escola"] = g["respostas_escola"].sum()
    out = out.reset_index()
    out["gap_top5"] = out["pct_top5"] - out["pct_escola"]
    return out


def itens_faceis(fac_esc: pd.DataFrame, fac_nac: pd.DataFrame, co_escola: int, anos: list[int]) -> pd.DataFrame:
    e = fac_esc[(fac_esc["co_escola"] == co_escola) & fac_esc["ano"].isin(anos)].groupby("area")[["acertos", "respostas"]].sum()
    n = fac_nac[fac_nac["ano"].isin(anos)].groupby("area")[["acertos", "respostas"]].sum()
    out = pd.DataFrame({"pct_escola": e["acertos"] / e["respostas"] * 100, "pct_brasil": n["acertos"] / n["respostas"] * 100})
    out["erros_escola"] = 100 - out["pct_escola"]
    out["area_nome"] = out.index.map(NOME_AREA)
    return out.reset_index()


LINGUA = {-1: "", 0: " (Inglês)", 1: " (Espanhol)"}


def acerto_questoes(q_ano: pd.DataFrame, itens: pd.DataFrame, co_escola: int, outras: list[int],
                    area: str, ano: int) -> pd.DataFrame:
    """% de acerto por questão: escola × média simples das `outras` escolas (cada escola pesa igual) × Brasil.

    q_ano: questoes_escola_<ano> (co_escola, questao, lingua, acertos, respostas); itens: questoes_itens.
    """
    it = itens[(itens["ano"] == ano) & (itens["area"] == area)].copy()
    chave = ["questao", "lingua"]
    q = q_ano[q_ano["questao"].isin(it["questao"])]
    esc = q[q["co_escola"] == co_escola].copy()
    esc["pct_escola"] = esc["acertos"].astype("int64") / esc["respostas"] * 100
    out = q[q["co_escola"].isin(set(outras)) & (q["respostas"] > 0)].copy()
    out["pct"] = out["acertos"].astype("int64") / out["respostas"] * 100
    med = out.groupby(chave).agg(pct_outras=("pct", "mean"), n_outras=("co_escola", "nunique")).reset_index()
    it["pct_brasil"] = it["acertos"] / it["respostas"].where(it["respostas"] > 0) * 100
    r = (it.merge(esc[chave + ["pct_escola", "respostas"]].rename(columns={"respostas": "respostas_escola"}),
                  on=chave, how="left").merge(med, on=chave, how="left"))
    r = r[r["respostas_escola"].fillna(0) > 0].sort_values(chave)
    r["rotulo"] = "Q" + r["questao"].astype(str) + r["lingua"].map(LINGUA)
    r["diferenca"] = r["pct_escola"] - r["pct_outras"]
    return r.reset_index(drop=True)
