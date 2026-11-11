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
