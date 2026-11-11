"""Gera o relatório em PDF de diagnóstico ENEM para uma escola.

Uso no código:  pdf_bytes = gerar_pdf(df, coer, co_escola, ano, ...)
Uso na linha de comando:  python relatorio.py <CO_ESCOLA> [ano]  -> relatorios/<escola>.pdf
"""
from __future__ import annotations

import datetime as dt
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

from analise import AREAS, diagnostico_top5, itens_faceis, perfil_habilidades, ranquear  # noqa: E402

AZUL = colors.HexColor("#1f4e8c")
CINZA = colors.HexColor("#6b7280")
CLARO = colors.HexColor("#eef2f8")
AZUL_MPL, CINZA_MPL, OURO_MPL = "#1f4e8c", "#b8c0cc", "#d29922"

TEXTO_PADRAO_INVESTIR = (
    "O SESI de Duque de Caxias, em seis anos, saiu da última posição para o 1º lugar no ENEM da cidade. "
    "Isso resultou em ganho de 174 alunos em 2024, com um aumento de receita de R$ 1,9 milhão. "
    "Um valor que supera em muitas vezes o investimento necessário para implementar uma solução como o dudow."
)


def _fmt(x: float, casas: int = 1) -> str:
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _estilos():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=ss["Normal"], fontName="Helvetica", fontSize=10.5, leading=15)
    return {
        "titulo": ParagraphStyle("t", parent=base, fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=AZUL),
        "sub": ParagraphStyle("s", parent=base, textColor=CINZA, fontSize=10),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=13.5, leading=18,
                             textColor=AZUL, spaceBefore=12, spaceAfter=4),
        "item": ParagraphStyle("it", parent=base, fontName="Helvetica-Bold", fontSize=11.5, leading=16,
                               spaceBefore=8, spaceAfter=2),
        "p": ParagraphStyle("p", parent=base, spaceAfter=5),
        "nota": ParagraphStyle("n", parent=base, fontSize=8, leading=11, textColor=CINZA),
        "kpi_v": ParagraphStyle("kv", parent=base, fontName="Helvetica-Bold", fontSize=17, leading=20,
                                alignment=TA_CENTER, textColor=AZUL),
        "kpi_l": ParagraphStyle("kl", parent=base, fontSize=8, leading=10, alignment=TA_CENTER, textColor=CINZA),
    }


def _fig_png(fig, largura_cm: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=largura_cm * cm, height=largura_cm * cm * h / w)


def _estilo_eixos(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#999")
    ax.spines["bottom"].set_color("#999")
    ax.tick_params(colors="#444", labelsize=8)


def _abreviar(nome: str, n: int = 38) -> str:
    nome = re.sub(r"\s+", " ", str(nome)).strip()
    return nome if len(nome) <= n else nome[: n - 1] + "…"


def _grafico_ranking(rk: pd.DataFrame, co: int, titulo: str) -> Image:
    pos_eu = int(rk.loc[rk["co_escola"] == co, "posicao"].iloc[0])
    vis = rk[(rk["posicao"] <= 10) | (rk["co_escola"] == co)].copy()
    if pos_eu > 10:  # mostra também os vizinhos imediatos
        vis = pd.concat([rk[rk["posicao"] <= 7], rk[(rk["posicao"] >= pos_eu - 2) & (rk["posicao"] <= pos_eu + 1)]])
        vis = vis.drop_duplicates("co_escola")
    vis = vis.sort_values("posicao", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(vis) + 0.8))
    cores = [AZUL_MPL if c == co else (OURO_MPL if p <= 5 else CINZA_MPL) for c, p in zip(vis["co_escola"], vis["posicao"])]
    rotulos = [f"{p}º  {_abreviar(n)}" for p, n in zip(vis["posicao"], vis["nome"])]
    ax.barh(rotulos, vis["media_geral"], color=cores, height=0.7)
    for y, v in enumerate(vis["media_geral"]):
        ax.text(v + 2, y, _fmt(v), va="center", fontsize=7.5, color="#333")
    lo = vis["media_geral"].min()
    ax.set_xlim(max(0, lo - 60), vis["media_geral"].max() + 25)
    ax.set_title(titulo, fontsize=10, loc="left", color="#222")
    _estilo_eixos(ax)
    ax.tick_params(axis="y", labelsize=7.5)
    return _fig_png(fig, 16.5)


def _grafico_areas(tab: pd.DataFrame) -> Image:
    t = tab.set_index("area").loc[list(AREAS.values())]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    x = range(len(t))
    ax.bar([i - 0.2 for i in x], t["escola"], width=0.4, color=AZUL_MPL, label="Escola")
    ax.bar([i + 0.2 for i in x], t["media_top5"], width=0.4, color=CINZA_MPL, label="Média do Top 5 da cidade")
    for i, (a, b) in enumerate(zip(t["escola"], t["media_top5"])):
        ax.text(i - 0.2, a + 5, f"{a:.0f}", ha="center", fontsize=7, color="#222")
        ax.text(i + 0.2, b + 5, f"{b:.0f}", ha="center", fontsize=7, color="#555")
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace("Ciências da ", "C. ").replace("Ciências ", "C. ") for n in t.index], fontsize=8)
    ax.set_ylim(min(t["escola"].min(), t["media_top5"].min()) - 120, max(t["escola"].max(), t["media_top5"].max()) + 40)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    _estilo_eixos(ax)
    return _fig_png(fig, 16)


def _grafico_coerencia(c: pd.Series, pre: str, ref_txt: str) -> Image:
    nomes = ["C. Natureza", "C. Humanas", "Linguagens", "Matemática"]
    vals = [float(c[f"{pre}_{a}"]) for a in ("cn", "ch", "lc", "mt")]
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    cores = ["#c0392b" if v > 0 else "#2e8b57" for v in vals]
    ax.bar(nomes, vals, color=cores, width=0.55)
    amp = max(abs(min(vals + [0])), abs(max(vals + [0])), 5)
    ax.set_ylim(min(vals + [0]) - amp * 0.25, max(vals + [0]) + amp * 0.25)
    for i, v in enumerate(vals):
        ax.text(i, v + (amp * 0.06 if v >= 0 else -amp * 0.06), f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, color="#222")
    ax.axhline(0, color="#777", lw=0.8)
    ax.set_ylabel("pontos por aluno", fontsize=8)
    ax.set_title(f"Pontos perdidos por incoerência vs. {ref_txt}\n(vermelho = perda; verde = acima da referência)",
                 fontsize=9, loc="left", pad=8)
    _estilo_eixos(ax)
    return _fig_png(fig, 16)


def _kpis(est, itens):
    cel = [[Paragraph(v, est["kpi_v"]) for v, _ in itens], [Paragraph(l, est["kpi_l"]) for _, l in itens]]
    t = Table(cel, colWidths=[17.0 / len(itens) * cm] * len(itens))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CLARO), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
                           ("LINEAFTER", (0, 0), (-2, -1), 0.5, colors.white)]))
    return t


def _nivel_redacao(n: float) -> str:
    if n >= 800:
        return "excelente"
    if n >= 700:
        return "bom"
    if n >= 620:
        return "razoável para bom"
    if n >= 540:
        return "razoável"
    return "abaixo do esperado"


def gerar_pdf(df: pd.DataFrame, coer: pd.DataFrame | None, co_escola: int, ano: int, min_alunos: int = 10,
              remetente: str = "", texto_investir: str = TEXTO_PADRAO_INVESTIR, destino: Path | None = None,
              referencia: str = "p75", hab: dict | None = None) -> bytes:
    """referencia: 'p75' (alunos coerentes: percentil 75 de quem teve os mesmos acertos) ou 'mediana'."""
    est = _estilos()
    pre = "perda75" if referencia == "p75" else "perda"
    ref_txt = ("o padrão de alta coerência (percentil 75 nacional)" if referencia == "p75"
               else "a mediana nacional")
    eu = df[(df["co_escola"] == co_escola) & (df["ano"] == ano)].iloc[0]
    mun, uf = eu["municipio"], eu["uf"]
    dm = diagnostico_top5(df, co_escola, "municipio", ano, min_alunos)
    du = diagnostico_top5(df, co_escola, "uf", ano, min_alunos)

    uni = df[(df["uf"] == uf) & (df["municipio"] == mun) & (df["ano"] == ano)]
    uni = uni[(uni["n"] >= min_alunos) | (uni["co_escola"] == co_escola)]
    rk = ranquear(uni, 0)

    c = None
    if coer is not None:
        cc = coer[(coer["co_escola"] == co_escola) & (coer["ano"] == ano)]
        c = cc.iloc[0] if len(cc) else None

    nota = float(eu["media_geral"])
    pct_mun = dm.gap / nota * 100 if dm.gap > 0 else 0
    pct_uf = du.gap / nota * 100 if du.gap > 0 else 0
    nome = str(eu["nome"]).title()
    mesma_uf = du.total == dm.total  # ex.: Brasília/DF

    story = []
    story += [Paragraph(f"Diagnóstico ENEM {ano}", est["titulo"]),
              Paragraph(f"<b>{nome}</b> · {mun}/{uf} · {eu['dependencia']} · {int(eu['n'])} participantes", est["sub"]),
              Paragraph(f"{'Preparado por ' + remetente + ' · ' if remetente else ''}"
                        f"{dt.date.today().strftime('%d/%m/%Y')}", est["sub"]), Spacer(1, 10)]
    story.append(_kpis(est, [
        (f"{dm.posicao}º de {dm.total}", f"Posição em {mun}"),
        (f"{du.posicao}º de {du.total}", f"Posição em {uf}") if not mesma_uf else
        (f"{int(eu['n'])}", "Participantes"),
        (_fmt(nota), "Média geral"),
        ("Top 5 ✓" if dm.gap <= 0 else f"+{_fmt(pct_mun)}%", "Para o Top 5 da cidade"),
    ]))

    # ---------------------------------------------------------------- conclusões
    story.append(Paragraph("Principais conclusões", est["h2"]))
    n = 1

    # 1. Potencial no ranking
    acima = rk[(rk["posicao"] < dm.posicao) & (rk["posicao"] >= 5)]
    if dm.gap > 0:
        alvo = nota + dm.gap
        txt = (f"Se o colégio elevar em cerca de <b>{_fmt(pct_mun, 0)}%</b> a nota média dos alunos no ENEM "
               f"(de {_fmt(nota)} para {_fmt(alvo)} pontos), tem potencial real para figurar entre os "
               f"<b>cinco melhores colégios de {mun}</b>.")
        if len(acima):
            nomes = ", ".join(_abreviar(x, 45).title() for x in acima.sort_values("posicao")["nome"].head(6))
            txt += f" Isso o colocaria à frente de concorrentes como {nomes}."
        if du.gap > 0 and not mesma_uf:
            txt += (f" No estado, a distância para o Top 5 de {uf} é de {_fmt(du.gap)} pontos "
                    f"(+{_fmt(pct_uf, 0)}%).")
        txt += (" Subir no ranking do ENEM da cidade é um dos fatores de maior visibilidade para as famílias "
                "na escolha da escola e tende a se refletir na captação de matrículas.")
        titulo = f"{n}. A escola está a {_fmt(dm.gap)} pontos do Top 5 de {mun}."
    else:
        lider = rk.iloc[0]
        dist1 = float(lider["media_geral"]) - nota
        titulo = f"{n}. A escola já está no Top 5 de {mun} ({dm.posicao}º lugar)."
        txt = ("O desafio agora é consolidar a posição e disputar a liderança"
               + (f": a distância para o 1º colocado ({_abreviar(lider['nome'], 45).title()}) é de "
                  f"{_fmt(dist1)} pontos (+{_fmt(dist1 / nota * 100, 0)}%)." if dist1 > 0 else ".")
               + ("" if mesma_uf else
                  f" No estado, faltam {_fmt(du.gap)} pontos (+{_fmt(pct_uf, 0)}%) para o Top 5 de {uf}."
                  if du.gap > 0 else f" A escola também está no Top 5 de {uf}."))
    story += [Paragraph(titulo, est["item"]), Paragraph(txt, est["p"])]
    n += 1

    # 2. Coerência
    if c is not None:
        perda = float(c[f"{pre}_obj"])
        pct10 = float(c["pct75_alunos_com_perda10" if referencia == "p75" else "pct_alunos_com_perda10"]) * 100
        perda4 = perda * 4
        if perda > 1:
            titulo = f"{n}. Identificamos uma perda média de {_fmt(perda, 0)} pontos por prova por falta de coerência."
            txt = (f"Na Teoria de Resposta ao Item (TRI), alunos que acertam questões difíceis mas erram fáceis são "
                   f"penalizados. Comparando cada aluno com {ref_txt} de quem teve o mesmo número de acertos, "
                   f"os alunos da escola ficaram em média <b>{_fmt(perda)} pontos abaixo</b> do padrão em cada prova objetiva "
                   f"(≈{_fmt(perda4, 0)} pontos somando as quatro). {_fmt(pct10, 0)}% dos alunos "
                   f"perderam mais de 10 pontos por prova. Esse é um ponto grave, mas é possível corrigir.")
        else:
            titulo = f"{n}. A coerência pedagógica dos alunos está no padrão de referência ou acima."
            txt = (f"Comparando cada aluno com {ref_txt} de quem teve o mesmo número de acertos, a escola ficou "
                   f"{_fmt(abs(perda))} pontos {'acima' if perda < 0 else 'abaixo'} do padrão por prova objetiva. "
                   f"Ainda assim, {_fmt(pct10, 0)}% dos alunos perderam mais de 10 pontos por prova.")
        story += [Paragraph(titulo, est["item"]), Paragraph(txt, est["p"])]
        ac, real = int(c["melhor_acertos"]), float(c["melhor_nota_obj"])
        esp = float(c["melhor_esperada75_obj" if referencia == "p75" else "melhor_esperada_obj"])
        if esp - real > 3:
            story.append(Paragraph(
                f"Para ilustrar: o aluno com maior número de acertos da escola acertou <b>{ac} de 180 questões</b>. "
                f"Tomando como referência {ref_txt}, isso corresponderia a uma média de <b>{_fmt(esp, 0)} pontos</b> nas "
                f"provas objetivas. Pela falta de coerência, a nota dele foi <b>{_fmt(real, 0)}</b> — "
                f"{_fmt(esp - real, 0)} pontos a menos. Estamos literalmente “deixando pontos e vagas na mesa”.", est["p"]))
        else:
            story.append(Paragraph(
                f"O aluno com maior número de acertos da escola acertou {ac} de 180 questões e obteve média de "
                f"{_fmt(real, 0)} pontos nas provas objetivas (referência para esse nº de acertos: {_fmt(esp, 0)}).",
                est["p"]))
        n += 1

    # 3. Redação x objetivas
    t = dm.tabela_areas.set_index("area")
    red = float(eu["media_redacao"])
    obj = t.drop(index="Redação")
    pior = obj["gap_top5"].idxmax()
    gap_red = float(t.loc["Redação", "gap_top5"])
    gap_obj = float(obj["gap_top5"].mean())
    if gap_obj >= gap_red:
        titulo = f"{n}. O que limita a nota é a parte objetiva da prova."
        txt = (f"A redação da escola está em um patamar <b>{_nivel_redacao(red)}</b> (média {_fmt(red, 0)}). "
               f"O ponto que mais limita a nota — e o ranking — são as questões objetivas: em média a escola está "
               f"{_fmt(gap_obj, 0)} pontos abaixo das cinco melhores da cidade nessas provas, com maior distância em "
               f"<b>{pior}</b> ({_fmt(obj.loc[pior, 'gap_top5'], 0)} pts). Isso pode ser corrigido de forma consistente "
               f"com um trabalho individualizado de preparação.")
    else:
        titulo = f"{n}. A redação é a maior oportunidade de ganho."
        txt = (f"A média de redação da escola ({_fmt(red, 0)}) está {_fmt(gap_red, 0)} pontos abaixo da média das cinco "
               f"melhores da cidade — distância maior que nas provas objetivas ({_fmt(gap_obj, 0)} pts em média). "
               f"Produção semanal com correção por competência costuma gerar ganhos rápidos.")
    if dm.gap > 0:
        metas = t[t["ganho_necessario"] > 0.5].sort_values("ganho_necessario", ascending=False)
        txt += " Metas sugeridas para alcançar o Top 5: " + "; ".join(
            f"{a} +{_fmt(v, 0)}" for a, v in metas["ganho_necessario"].items()) + " pontos."
    story += [Paragraph(titulo, est["item"]), Paragraph(txt, est["p"])]

    perf = fac = None
    if hab is not None:
        anos_h = sorted(df.loc[df["co_escola"] == co_escola, "ano"].unique())
        perf = perfil_habilidades(hab["habilidades_escola"], hab["habilidades_nacional"], df, hab["matriz"],
                                  co_escola, anos_h, ano, min_alunos)
        fac = itens_faceis(hab["faceis_escola"], hab["faceis_nacional"], co_escola, anos_h)
        prio = perf.dropna(subset=["gap_top5"]).sort_values("gap_top5", ascending=False)
        prio = prio[~prio["amostra_pequena"]].head(3) if (~prio["amostra_pequena"]).sum() >= 3 else prio.head(3)
        n += 1
        pior_f = fac.sort_values("pct_escola").iloc[0]
        txt = ("Cruzando cada questão com a habilidade avaliada na Matriz de Referência do ENEM, as maiores distâncias "
               "para as cinco melhores escolas da cidade estão em: "
               + "; ".join(f"<b>{r['area_nome']} H{r['hab']}</b> ({r['desc_hab'].rstrip('.')}: "
                           f"{_fmt(r['pct_escola'], 0)}% de acerto contra {_fmt(r['pct_top5'], 0)}%)"
                           for _, r in prio.iterrows()) + ". "
               f"Nas questões fáceis, a escola acerta {_fmt(pior_f['pct_escola'], 0)}% em {pior_f['area_nome']} "
               f"(Brasil: {_fmt(pior_f['pct_brasil'], 0)}%).")
        story += [Paragraph(f"{n}. As habilidades que mais limitam a escola.", est["item"]), Paragraph(txt, est["p"])]

    if texto_investir and texto_investir.strip():
        bloco = [Paragraph("Por que é tão importante investir nisto?", est["h2"])]
        bloco += [Paragraph(par.strip(), est["p"]) for par in texto_investir.split("\n") if par.strip()]
        story.append(KeepTogether(bloco))

    # ---------------------------------------------------------------- gráficos
    story.append(KeepTogether([
        Paragraph(f"Ranking ENEM {ano} — {mun}/{uf}", est["h2"]),
        _grafico_ranking(rk, co_escola, f"Média geral (escolas com ≥ {min_alunos} participantes)"),
        Paragraph("Azul: a escola · Dourado: Top 5 · Cinza: demais escolas.", est["nota"])]))
    story.append(KeepTogether([Paragraph("Escola × cinco melhores da cidade, por área", est["h2"]),
                               _grafico_areas(dm.tabela_areas)]))
    if c is not None:
        story.append(KeepTogether([Paragraph("Coerência pedagógica (TRI)", est["h2"]), _grafico_coerencia(c, pre, ref_txt)]))

    tab = dm.tabela_areas.set_index("area").loc[list(AREAS.values())].reset_index()
    linhas = [["Área", "Escola", "Média Top 5", "Distância", "Meta de ganho"]] + [
        [r["area"], _fmt(r["escola"]), _fmt(r["media_top5"]), _fmt(r["gap_top5"]),
         ("+" + _fmt(r["ganho_necessario"], 0)) if dm.gap > 0 and r["ganho_necessario"] > 0.5 else "—"]
        for _, r in tab.iterrows()]
    tb = Table(linhas, colWidths=[5 * cm, 2.8 * cm, 3 * cm, 2.8 * cm, 3 * cm])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), AZUL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLARO]),
                            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(KeepTogether([Paragraph("Metas por área para alcançar o Top 5 da cidade", est["h2"]), tb]))

    if perf is not None:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Habilidades prioritárias (Matriz de Referência do ENEM)", est["h2"]))
        story.append(Paragraph(
            "Percentual de acerto da escola em cada habilidade, comparado com as cinco melhores escolas da cidade e com o "
            "Brasil. Ordenado pela maior distância para o Top 5. Anos somados: "
            + ", ".join(str(a) for a in sorted(df.loc[df['co_escola'] == co_escola, 'ano'].unique())) + ".", est["p"]))
        cel = ParagraphStyle("cel", parent=est["p"], fontSize=8, leading=10, spaceAfter=0)
        top = perf.dropna(subset=["gap_top5"]).sort_values("gap_top5", ascending=False).head(12)
        linhas = [["Hab.", "Descrição", "Escola", "Top 5", "Brasil", "Dist."]] + [
            [Paragraph(f"<b>{r['area']} H{r['hab']}</b>" + (" *" if r["amostra_pequena"] else ""), cel),
             Paragraph(r["desc_hab"], cel), f"{r['pct_escola']:.0f}%", f"{r['pct_top5']:.0f}%",
             f"{r['pct_brasil']:.0f}%", f"{r['gap_top5']:+.0f}"] for _, r in top.iterrows()]
        tb = Table(linhas, colWidths=[1.7 * cm, 9.6 * cm, 1.5 * cm, 1.4 * cm, 1.5 * cm, 1.3 * cm], repeatRows=1)
        tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), AZUL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("ALIGN", (2, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLARO]),
                                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(tb)
        story.append(Paragraph("* menos de 30 respostas da escola nessa habilidade — resultado instável. "
                               "Dist. = pontos percentuais abaixo da média das cinco melhores escolas da cidade.", est["nota"]))
        f = fac.copy()
        linhas = [["Área", "Acerto nas questões fáceis", "Brasil", "Diferença"]] + [
            [r["area_nome"], f"{r['pct_escola']:.0f}%", f"{r['pct_brasil']:.0f}%", f"{r['pct_escola'] - r['pct_brasil']:+.0f} p.p."]
            for _, r in f.iterrows()]
        tf = Table(linhas, colWidths=[5 * cm, 5 * cm, 3 * cm, 3 * cm])
        tf.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), AZUL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLARO])]))
        story.append(KeepTogether([Paragraph("Questões fáceis (25% de menor dificuldade na TRI)", est["h2"]), tf]))

    story += [Spacer(1, 8), Paragraph(
        "<b>Metodologia.</b> Fonte: microdados do ENEM (INEP) e Censo Escolar. Média geral = média simples das notas de "
        "Ciências da Natureza, Ciências Humanas, Linguagens, Matemática e Redação dos participantes presentes nos dois dias "
        "que informaram a escola. Rankings consideram escolas com pelo menos "
        f"{min_alunos} participantes. Coerência: para cada aluno e prova, comparamos a nota TRI com "
        f"{ref_txt} dos participantes com o mesmo número de acertos; a diferença positiva indica pontos perdidos. As metas por área "
        "distribuem a distância para o 5º colocado proporcionalmente às áreas em que a escola está mais distante das "
        "cinco melhores. O ENEM não é uma avaliação oficial de escolas; resultados dependem do perfil dos participantes.",
        est["nota"])]

    buf = io.BytesIO()

    def rodape(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(CINZA)
        canvas.drawString(2 * cm, 1.2 * cm, f"Diagnóstico ENEM {ano} · {_abreviar(nome, 60)}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title=f"Diagnóstico ENEM {ano} - {nome}", author=remetente or "")
    doc.build(story, onFirstPage=rodape, onLaterPages=rodape)
    pdf = buf.getvalue()
    if destino:
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        Path(destino).write_bytes(pdf)
    return pdf


def nome_arquivo(nome: str, ano: int) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", str(nome)).strip("_")[:60]
    return f"Diagnostico_ENEM_{ano}_{base}.pdf"


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parent / "data" / "processed"
    df = pd.read_parquet(root / "escolas_ano.parquet")
    df = df[df["no_censo"]] if "no_censo" in df else df
    coer_p = root / "coerencia_escola.parquet"
    coer = pd.read_parquet(coer_p) if coer_p.exists() else None
    co = int(sys.argv[1])
    ano = int(sys.argv[2]) if len(sys.argv) > 2 else int(df.loc[df["co_escola"] == co, "ano"].max())
    nome = df.loc[df["co_escola"] == co, "nome"].iloc[0]
    out = Path(__file__).resolve().parent / "relatorios" / nome_arquivo(nome, ano)
    hab = None
    if (root / "habilidades_escola.parquet").exists():
        hab = {k: pd.read_parquet(root / f"{k}.parquet") for k in
               ("habilidades_escola", "habilidades_nacional", "faceis_escola", "faceis_nacional")}
        hab["matriz"] = pd.read_csv(root.parent / "ref" / "habilidades.csv")
    gerar_pdf(df, coer, co, ano, destino=out, hab=hab)
    print(out)
