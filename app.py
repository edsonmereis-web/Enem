"""Painel: Ranking ENEM por escola. Rode com:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analise import (AREAS, NOME_AREA, carregar, diagnostico_top5, itens_faceis, media_periodo, perfil_competencias,
                     perfil_habilidades, ranquear)
from relatorio import TEXTO_PADRAO_INVESTIR, gerar_pdf, nome_arquivo

ROOT = Path(__file__).resolve().parent


def _achar(nome: str) -> Path:
    """Procura o arquivo em data/processed/ ou na raiz do projeto (upload pelo site do GitHub)."""
    for p in (ROOT / "data" / "processed" / nome, ROOT / nome):
        if p.exists():
            return p
    return ROOT / "data" / "processed" / nome


DADOS = _achar("escolas_ano.parquet")
COER = _achar("coerencia_escola.parquet")
HAB = {k: _achar(f"{k}.parquet") for k in ("habilidades_escola", "habilidades_nacional", "faceis_escola", "faceis_nacional")}
MATRIZ = next((p for p in (ROOT / "data" / "ref" / "habilidades.csv", ROOT / "habilidades.csv") if p.exists()), None)

st.set_page_config(page_title="Ranking ENEM por Escola", page_icon="🏫", layout="wide")


@st.cache_data
def dados(mtime: float) -> pd.DataFrame:  # mtime invalida o cache quando o parquet é regerado
    return carregar(DADOS)


@st.cache_data
def dados_habilidades(mtime: float) -> dict:
    out = {k: pd.read_parquet(p) for k, p in HAB.items()}
    out["matriz"] = pd.read_csv(MATRIZ)
    return out


def tem_habilidades() -> bool:
    return MATRIZ is not None and all(p.exists() for p in HAB.values())


@st.cache_data
def dados_coerencia(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(COER)


if not DADOS.exists():
    st.error("Arquivo de dados não encontrado. Rode `python pipeline.py all` (ou `python gerar_amostra.py` para testar).")
    st.stop()

df = dados(DADOS.stat().st_mtime)
anos = sorted(df["ano"].unique())

# ------------------------------------------------------------------ filtros
st.sidebar.header("Filtros")
periodo = st.sidebar.selectbox("Período", ["Média de todos os anos"] + [str(a) for a in reversed(anos)], index=1)
ufs = sorted(df["uf"].dropna().unique())
uf = st.sidebar.selectbox("UF", ["Brasil"] + ufs)
# Município sempre disponível: com UF = Brasil lista todos os municípios do país ("Cidade/UF").
pares = df[["uf", "municipio"]].dropna().drop_duplicates()
if uf != "Brasil":
    pares = pares[pares["uf"] == uf]
pares = sorted(map(tuple, pares[["municipio", "uf"]].values))
escolha_mun = st.sidebar.selectbox(
    "Município", [None] + pares, format_func=lambda p: "Todos" if p is None else (p[0] if uf != "Brasil" else f"{p[0]}/{p[1]}"),
    help="Digite para pesquisar. Com UF = Brasil, aparecem os municípios de todo o país.")
municipio = "Todos"
if escolha_mun is not None:
    municipio, uf = escolha_mun
deps = st.sidebar.multiselect("Dependência", sorted(df["dependencia"].unique()), default=sorted(df["dependencia"].unique()))
min_alunos = st.sidebar.slider("Mínimo de participantes por escola", 1, 100, 10,
                               help="Escolas com poucos alunos têm médias muito instáveis.")
CRITERIOS = {"media_geral": "Média geral", "media_redacao": "Redação", "media_mt": "Matemática",
             "media_cn": "Ciências da Natureza", "media_ch": "Ciências Humanas", "media_lc": "Linguagens"}
CRIT = st.sidebar.selectbox("Ranking por", list(CRITERIOS), format_func=CRITERIOS.get,
                            help="Critério de ordenação dos rankings (abas Ranking geral e Rede Adventista).")
CRIT_NOME = CRITERIOS[CRIT]
busca = st.sidebar.text_input("Buscar escola pelo nome")
so_censo = st.sidebar.checkbox("Ocultar códigos não localizados no Censo", value=True,
                               help="O código da escola no ENEM é autodeclarado; alguns não existem no Censo Escolar.")
if so_censo and "no_censo" in df:
    df = df[df["no_censo"]]
if periodo == "Média de todos os anos":
    min_anos = st.sidebar.slider("Mínimo de anos com resultado", 1, len(anos), min(2, len(anos)))

st.sidebar.caption(f"Anos com código de escola nos microdados: {', '.join(map(str, anos))}")


def recorte(d: pd.DataFrame) -> pd.DataFrame:
    if uf != "Brasil":
        d = d[d["uf"] == uf]
    if municipio != "Todos":
        d = d[d["municipio"] == municipio]
    if so_censo and "no_censo" in d:
        d = d[d["no_censo"]]
    return d[d["dependencia"].isin(deps)]


def tabela_ranking(d: pd.DataFrame) -> pd.DataFrame:
    if periodo == "Média de todos os anos":
        t = media_periodo(d, min_alunos, min_anos)
    else:
        t = d[(d["ano"] == int(periodo)) & (d["n"] >= min_alunos)].sort_values(CRIT, ascending=False)
    t = t.sort_values(CRIT, ascending=False).reset_index(drop=True)
    t.insert(0, "posição", t[CRIT].rank(ascending=False, method="min").astype(int))
    return t


COLS = {"posição": "Pos.", "nome": "Escola", "municipio": "Município", "uf": "UF", "dependencia": "Rede",
        "n": "Particip.", "media_geral": "Média geral", "media_cn": "CN", "media_ch": "CH", "media_lc": "LC",
        "media_mt": "MT", "media_redacao": "Redação", "pct_acima_700": "% ≥700",
        "pct_redacao_900": "% red. ≥900", "anos": "Anos"}


def em_pct(t: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas de proporção (0–1) para 0–100 para exibição."""
    return t.assign(**{c: t[c] * 100 for c in ("pct_acima_700", "pct_redacao_900") if c in t})


def mostrar(t: pd.DataFrame, altura=600):
    t = em_pct(t)
    if busca:
        t = t[t["nome"].str.contains(busca, case=False, na=False)]
    cols = [c for c in COLS if c in t]
    st.dataframe(
        t[cols].rename(columns=COLS), height=altura, hide_index=True, use_container_width=True,
        column_config={
            "Média geral": st.column_config.ProgressColumn(format="%.1f", min_value=300, max_value=850),
            **{k: st.column_config.NumberColumn(format="%.1f") for k in ["CN", "CH", "LC", "MT", "Redação"]},
            "% ≥700": st.column_config.NumberColumn(format="%.0f%%"),
            "% red. ≥900": st.column_config.NumberColumn(format="%.0f%%"),
        })
    st.download_button("⬇️ Baixar CSV", t[cols].rename(columns=COLS).to_csv(index=False, sep=";", decimal=","),
                       file_name="ranking_enem.csv", mime="text/csv")


local = "Brasil" if uf == "Brasil" else (uf if municipio == "Todos" else f"{municipio}/{uf}")
st.title("🏫 Ranking ENEM por Escola")
st.caption(f"Ranking por **{CRIT_NOME}** · Recorte: **{local}** · Período: **{periodo}** · mínimo de {min_alunos} participantes")

aba1, aba2, aba3, aba5, aba4 = st.tabs(["📊 Ranking geral", "📘 Rede Adventista", "🎯 Como chegar ao Top 5",
                                        "🧩 Habilidades", "ℹ️ Metodologia"])

# ------------------------------------------------------------------ ranking geral
with aba1:
    rk = tabela_ranking(recorte(df))
    pct = lambda s: f"{100*s:.0f}%"  # noqa: E731
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Escolas no ranking", f"{len(rk):,}".replace(",", "."))
    c2.metric("Média das escolas", f"{rk[CRIT].mean():.1f}" if len(rk) else "—")
    c3.metric("Nota de corte do Top 5", f"{rk[CRIT].iloc[4]:.1f}" if len(rk) >= 5 else "—")
    c4.metric("Adventistas no recorte", int(rk["adventista"].sum()) if len(rk) else 0)
    mostrar(rk)

    if len(rk):
        g1, g2 = st.columns(2)
        with g1:
            fig = px.box(rk, x="dependencia", y=CRIT, points=False, title="Distribuição por rede",
                         labels={"dependencia": "", CRIT: CRIT_NOME})
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            top = rk.head(15).iloc[::-1]
            fig = px.bar(top, x=CRIT, y="nome", orientation="h", title="Top 15",
                         color="adventista", color_discrete_map={True: "#1f6feb", False: "#9aa4b2"},
                         labels={CRIT: CRIT_NOME, "nome": "", "adventista": "Adventista"})
            fig.update_xaxes(range=[top[CRIT].min() - 20, top[CRIT].max() + 5])
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ adventistas
with aba2:
    adv = df[df["adventista"]]
    if adv.empty:
        st.info("Nenhuma escola adventista identificada nos dados.")
    else:
        rk_adv = tabela_ranking(recorte(adv))
        st.subheader(f"Ranking da rede adventista — {local}")
        st.caption("Posição dentro da própria rede. As colunas à direita mostram onde cada escola fica no ranking "
                   "geral do seu município e do seu estado no mesmo período.")
        # posição de cada escola adventista nos rankings gerais (município e UF)
        if periodo == "Média de todos os anos":
            base = media_periodo(df[df["dependencia"].isin(deps)], min_alunos, min_anos)
        else:
            base = df[(df["ano"] == int(periodo)) & (df["n"] >= min_alunos) & df["dependencia"].isin(deps)]
        base = base.assign(
            pos_municipio=base.groupby(["uf", "municipio"])[CRIT].rank(ascending=False, method="min"),
            pos_uf=base.groupby("uf")[CRIT].rank(ascending=False, method="min"),
            pos_brasil=base[CRIT].rank(ascending=False, method="min"))
        corte_mun = base.groupby(["uf", "municipio"])[CRIT].apply(
            lambda s: s.nlargest(5).min() if len(s) >= 5 else float("nan")).rename("corte_top5_mun")
        corte_uf = base.groupby("uf")[CRIT].apply(lambda s: s.nlargest(5).min()).rename("corte_top5_uf")
        rk_adv = (rk_adv.merge(base[["co_escola", "pos_municipio", "pos_uf", "pos_brasil"]], on="co_escola", how="left")
                  .merge(corte_mun, on=["uf", "municipio"], how="left").merge(corte_uf, on="uf", how="left"))
        rk_adv["falta_top5_mun"] = (rk_adv["corte_top5_mun"] - rk_adv[CRIT]).clip(lower=0)
        rk_adv["falta_top5_uf"] = (rk_adv["corte_top5_uf"] - rk_adv[CRIT]).clip(lower=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Escolas adventistas", len(rk_adv))
        c2.metric("Média da rede", f"{rk_adv[CRIT].mean():.1f}" if len(rk_adv) else "—")
        c3.metric("No Top 5 do município", int((rk_adv["pos_municipio"] <= 5).sum()))
        c4.metric("No Top 5 do estado", int((rk_adv["pos_uf"] <= 5).sum()))

        t = em_pct(rk_adv)
        if busca:
            t = t[t["nome"].str.contains(busca, case=False, na=False)]
        st.dataframe(
            t[["posição", "nome", "municipio", "uf", "n", "media_geral", "media_mt", "media_redacao", "pct_redacao_900",
               "pos_municipio", "falta_top5_mun", "pos_uf", "falta_top5_uf", "pos_brasil"]].rename(columns={
                **COLS, "pos_municipio": "Pos. município", "falta_top5_mun": "Falta p/ Top 5 mun.",
                "pos_uf": "Pos. UF", "falta_top5_uf": "Falta p/ Top 5 UF", "pos_brasil": "Pos. Brasil"}),
            hide_index=True, use_container_width=True, height=550,
            column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                           ["Média geral", "MT", "Redação", "Falta p/ Top 5 mun.", "Falta p/ Top 5 UF"]}
            | {c: st.column_config.NumberColumn(format="%d") for c in ["Pos. município", "Pos. UF", "Pos. Brasil"]}
           | {"% red. ≥900": st.column_config.NumberColumn(format="%.0f%%")})
        st.download_button("⬇️ Baixar CSV (adventistas)", t.to_csv(index=False, sep=";", decimal=","),
                           file_name="ranking_adventistas.csv", mime="text/csv")

        # evolução da rede x demais privadas
        evo = (df[df["n"] >= min_alunos].assign(grupo=lambda d: d["adventista"].map({True: "Adventistas", False: None})
                                                .fillna(d["dependencia"]))
               .groupby(["ano", "grupo"]).apply(lambda g: (g[CRIT] * g["n"]).sum() / g["n"].sum())
               .rename("media").reset_index())
        if uf != "Brasil":
            st.caption("Gráfico de evolução considera o Brasil inteiro.")
        fig = px.line(evo, x="ano", y="media", color="grupo", markers=True,
                      title=f"Evolução — {CRIT_NOME} (média ponderada por participantes)",
                      labels={"media": CRIT_NOME, "ano": "Ano", "grupo": ""})
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ top 5
with aba3:
    st.subheader("Diagnóstico: o que falta para entrar no Top 5")
    so_adv = st.toggle("Listar apenas escolas adventistas", value=True)
    cand = recorte(df[df["adventista"]] if so_adv else df)
    cand = cand.sort_values("ano").drop_duplicates("co_escola", keep="last").sort_values(["uf", "municipio", "nome"])
    if busca:
        cand = cand[cand["nome"].str.contains(busca, case=False, na=False)]
    if cand.empty:
        st.info("Nenhuma escola no recorte. Ajuste os filtros na barra lateral.")
    else:
        opcoes = dict(zip(cand["co_escola"], cand["nome"] + " — " + cand["municipio"] + "/" + cand["uf"]))
        escolha = st.selectbox("Escola", list(opcoes), format_func=opcoes.get)
        anos_esc = sorted(df.loc[df["co_escola"] == escolha, "ano"].unique(), reverse=True)
        pref = None if periodo == "Média de todos os anos" else int(periodo)
        ano_d = st.selectbox("Ano de referência", anos_esc, index=anos_esc.index(pref) if pref in anos_esc else 0)
        cols = st.columns(2)
        for escopo, col in zip(["municipio", "uf"], cols):
            with col:
                d = diagnostico_top5(df, escolha, escopo, ano_d, min_alunos)
                titulo = "No município" if escopo == "municipio" else "No estado"
                st.markdown(f"### {titulo}")
                if d is None:
                    st.warning("Escola sem resultado no ano escolhido.")
                    continue
                m1, m2, m3 = st.columns(3)
                m1.metric("Posição", f"{d.posicao}º de {d.total}")
                m2.metric("Média geral", f"{d.nota:.1f}")
                m3.metric("Falta p/ Top 5", "✅ no Top 5" if d.gap <= 0 else f"{d.gap:.1f} pts")

                for r in d.recomendacoes:
                    st.markdown(f"- {r}")

                fig = go.Figure()
                fig.add_bar(name="Escola", x=d.tabela_areas["area"], y=d.tabela_areas["escola"], marker_color="#1f6feb")
                fig.add_bar(name="Média do Top 5", x=d.tabela_areas["area"], y=d.tabela_areas["media_top5"],
                            marker_color="#9aa4b2")
                fig.update_layout(barmode="group", title="Escola × Top 5 por área", height=330,
                                  legend=dict(orientation="h", y=-0.2), margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, key=f"areas_{escopo}")

                if d.gap > 0:
                    st.markdown("**Meta por área** (distribuída onde a distância para as líderes é maior)")
                    st.dataframe(d.tabela_areas[["area", "escola", "media_top5", "gap_top5", "ganho_necessario"]]
                                 .rename(columns={"area": "Área", "escola": "Escola", "media_top5": "Top 5",
                                                  "gap_top5": "Distância", "ganho_necessario": "Meta de ganho"}),
                                 hide_index=True, use_container_width=True,
                                 column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                                ["Escola", "Top 5", "Distância", "Meta de ganho"]})

                h = d.historico
                fig = go.Figure()
                fig.add_scatter(x=h["ano"], y=h["escola"], name="Escola", mode="lines+markers", line_color="#1f6feb")
                fig.add_scatter(x=h["ano"], y=h["corte_top5"], name="Corte Top 5", mode="lines+markers",
                                line=dict(color="#d29922", dash="dash"))
                fig.add_scatter(x=h["ano"], y=h["lider"], name="1º lugar", mode="lines+markers", line_color="#9aa4b2")
                fig.update_layout(title="Evolução: escola × corte do Top 5", height=330, xaxis_dtick=1,
                                  legend=dict(orientation="h", y=-0.2), margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, key=f"hist_{escopo}")

                with st.expander("Ver Top 5 atual"):
                    st.dataframe(d.top5[["posicao", "nome", "dependencia", "n", "media_geral", "media_mt", "media_redacao"]]
                                 .rename(columns={"posicao": "Pos.", **COLS}), hide_index=True, use_container_width=True)

        # -------------------------------------------------------------- descritores com maior déficit
        st.divider()
        st.subheader("🧩 Descritores com maior déficit")
        if not tem_habilidades():
            st.info("Rode `python habilidades.py` para gerar a análise por descritor.")
        else:
            Hd = dados_habilidades(HAB["habilidades_escola"].stat().st_mtime)
            anos_desc = sorted(df.loc[df["co_escola"] == escolha, "ano"].unique())
            st.caption(f"Habilidades (descritores H1–H30 da Matriz de Referência) em que a escola mais fica atrás das "
                       f"5 melhores escolas (média geral de {ano_d}). Acertos somados de "
                       f"{', '.join(map(str, anos_desc))}. ⚠️ = menos de 30 respostas da escola (resultado instável). "
                       "“Acertos a mais” ≈ questões a mais por aluno, por ano, se a escola igualasse o Top 5.")
            dc1, dc2 = st.columns([1, 1])
            esc_desc = dc1.radio("Comparar com o Top 5", ["municipio", "uf"], horizontal=True, key="desc_esc",
                                 format_func={"municipio": "da cidade", "uf": "do estado"}.get)
            area_desc = dc2.multiselect("Áreas", list(NOME_AREA), default=list(NOME_AREA), format_func=NOME_AREA.get,
                                        key="desc_area")
            pdesc = perfil_habilidades(Hd["habilidades_escola"], Hd["habilidades_nacional"], df, Hd["matriz"],
                                       escolha, anos_desc, ano_d, min_alunos, esc_desc)
            pdesc = pdesc[pdesc["area"].isin(area_desc)].dropna(subset=["gap_top5"])
            n_top = st.slider("Quantidade de descritores", 5, 30, 10, key="desc_n")
            top_d = pdesc.sort_values("gap_top5", ascending=False).head(n_top).copy()
            if top_d.empty or top_d["gap_top5"].max() <= 0:
                st.success("A escola não tem déficit em relação ao Top 5 nas áreas selecionadas.")
            else:
                top_d["Descritor"] = top_d["rotulo"] + top_d["amostra_pequena"].map({True: " ⚠️", False: ""})
                g1, g2 = st.columns([3, 2])
                with g1:
                    st.dataframe(
                        top_d[["Descritor", "desc_hab", "pct_escola", "pct_top5", "pct_brasil", "gap_top5",
                               "acertos_a_mais"]].rename(columns={
                            "desc_hab": "Descrição", "pct_escola": "Escola %", "pct_top5": "Top 5 %",
                            "pct_brasil": "Brasil %", "gap_top5": "Déficit (p.p.)", "acertos_a_mais": "Acertos a mais"}),
                        hide_index=True, use_container_width=True, height=min(38 * (len(top_d) + 1), 600),
                        column_config={c: st.column_config.NumberColumn(format="%.0f") for c in
                                       ["Escola %", "Top 5 %", "Brasil %", "Déficit (p.p.)"]}
                        | {"Acertos a mais": st.column_config.NumberColumn(format="%.2f"),
                           "Descrição": st.column_config.TextColumn(width="medium")})
                with g2:
                    gd = top_d.iloc[::-1]
                    fig = go.Figure(go.Bar(x=gd["gap_top5"], y=gd["Descritor"], orientation="h",
                                           marker_color=["#c0392b" if v > 0 else "#2e8b57" for v in gd["gap_top5"]],
                                           customdata=gd[["desc_hab"]],
                                           hovertemplate="%{y}: %{x:.0f} p.p.<br>%{customdata[0]}<extra></extra>"))
                    fig.update_layout(title="Déficit vs Top 5 (pontos percentuais)", height=max(300, 28 * len(gd) + 80),
                                      margin=dict(t=40, l=10, r=10, b=10), xaxis_title="p.p.")
                    st.plotly_chart(fig, use_container_width=True, key="desc_fig")
                por_area = (pdesc[pdesc["gap_top5"] > 0].groupby("area")
                            .agg(descritores=("hab", "size"), acertos=("acertos_a_mais", "sum")).reset_index())
                if len(por_area):
                    st.markdown("**Resumo por área** — " + " · ".join(
                        f"{NOME_AREA[r['area']]}: {int(r['descritores'])} descritores abaixo do Top 5 "
                        f"(≈{r['acertos']:.1f} acertos a mais por aluno)" for _, r in por_area.iterrows()))

        # -------------------------------------------------------------- relatório PDF
        st.divider()
        st.subheader("📄 Relatório em PDF para a escola")
        coer = dados_coerencia(COER.stat().st_mtime) if COER.exists() else None
        if coer is None:
            st.caption("Rode `python coerencia.py` para incluir a análise de coerência (TRI) no relatório.")
        r1, r2 = st.columns(2)
        remetente = r1.text_input("Preparado por (aparece no cabeçalho)", value="")
        ref = r2.radio("Referência de coerência", ["p75", "mediana"], horizontal=True,
                       format_func={"p75": "Alta coerência (percentil 75)", "mediana": "Mediana nacional"}.get,
                       help="Nota que alunos com o MESMO nº de acertos obtêm: percentil 75 (mais exigente) ou mediana.")
        texto = st.text_area("Seção “Por que é tão importante investir nisto?” (deixe vazio para omitir)",
                             value=TEXTO_PADRAO_INVESTIR, height=110)
        if st.button("Gerar relatório PDF", type="primary"):
            with st.spinner("Gerando PDF…"):
                st.session_state["pdf"] = (escolha, ano_d, gerar_pdf(df, coer, escolha, ano_d, min_alunos,
                                                                     remetente, texto, referencia=ref,
                                                                     hab=dados_habilidades(HAB["habilidades_escola"].stat().st_mtime)
                                                                     if tem_habilidades() else None))
        if st.session_state.get("pdf") and st.session_state["pdf"][:2] == (escolha, ano_d):
            st.download_button("⬇️ Baixar PDF", st.session_state["pdf"][2], mime="application/pdf",
                               file_name=nome_arquivo(df.loc[df["co_escola"] == escolha, "nome"].iloc[0], ano_d))

# ------------------------------------------------------------------ habilidades
with aba5:
    st.subheader("Acerto por habilidade (Matriz de Referência do ENEM)")
    if not tem_habilidades():
        st.info("Rode `python habilidades.py` para gerar a análise por habilidade.")
    else:
        H = dados_habilidades(HAB["habilidades_escola"].stat().st_mtime)
        so_adv_h = st.toggle("Listar apenas escolas adventistas", value=True, key="adv_hab")
        cand_h = recorte(df[df["adventista"]] if so_adv_h else df)
        cand_h = cand_h.sort_values("ano").drop_duplicates("co_escola", keep="last").sort_values(["uf", "municipio", "nome"])
        if busca:
            cand_h = cand_h[cand_h["nome"].str.contains(busca, case=False, na=False)]
        if cand_h.empty:
            st.info("Nenhuma escola no recorte. Ajuste os filtros na barra lateral.")
        else:
            op_h = dict(zip(cand_h["co_escola"], cand_h["nome"] + " — " + cand_h["municipio"] + "/" + cand_h["uf"]))
            esc_h = st.selectbox("Escola", list(op_h), format_func=op_h.get, key="esc_hab")
            anos_e = sorted(df.loc[df["co_escola"] == esc_h, "ano"].unique())
            anos_sel = st.multiselect("Anos considerados (somados)", anos_e, default=anos_e,
                                      help="Somar anos aumenta o nº de questões por habilidade e reduz a oscilação.")
            anos_sel = anos_sel or anos_e
            ano_ref = max(anos_sel)
            perf = perfil_habilidades(H["habilidades_escola"], H["habilidades_nacional"], df, H["matriz"],
                                      esc_h, anos_sel, ano_ref, min_alunos)
            fac = itens_faceis(H["faceis_escola"], H["faceis_nacional"], esc_h, anos_sel)

            st.markdown("**Questões fáceis** (25% de menor dificuldade na TRI): erro aqui é o que mais derruba a nota.")
            cols_f = st.columns(len(fac))
            for c, (_, r) in zip(cols_f, fac.iterrows()):
                c.metric(f"{r['area_nome']} · acerto", f"{r['pct_escola']:.0f}%",
                         f"{r['pct_escola'] - r['pct_brasil']:+.0f} p.p. vs Brasil")

            st.markdown("#### Habilidades prioritárias")
            st.caption("Maior distância para as 5 melhores escolas da cidade. ⚠️ = menos de 30 respostas da escola "
                       "(resultado instável).")
            prio = perf.dropna(subset=["gap_top5"]).sort_values("gap_top5", ascending=False).head(12).copy()
            prio["Habilidade"] = prio["rotulo"] + prio["amostra_pequena"].map({True: " ⚠️", False: ""})
            st.dataframe(prio[["Habilidade", "desc_hab", "pct_escola", "pct_top5", "pct_brasil", "gap_top5",
                               "respostas_escola"]].rename(columns={
                "desc_hab": "Descrição", "pct_escola": "Escola %", "pct_top5": "Top 5 cidade %",
                "pct_brasil": "Brasil %", "gap_top5": "Distância (p.p.)", "respostas_escola": "Respostas"}),
                hide_index=True, use_container_width=True,
                column_config={c: st.column_config.NumberColumn(format="%.0f") for c in
                               ["Escola %", "Top 5 cidade %", "Brasil %", "Distância (p.p.)"]}
                | {"Descrição": st.column_config.TextColumn(width="large")})

            area_sel = st.radio("Área", list(NOME_AREA), format_func=NOME_AREA.get, horizontal=True, key="area_hab")
            pa = perf[perf["area"] == area_sel].sort_values("hab")
            fig = go.Figure()
            fig.add_bar(x="H" + pa["hab"].astype(str), y=pa["pct_escola"], name="Escola", marker_color="#1f6feb",
                        customdata=pa[["desc_hab"]], hovertemplate="%{x}: %{y:.0f}%<br>%{customdata[0]}<extra></extra>")
            fig.add_scatter(x="H" + pa["hab"].astype(str), y=pa["pct_top5"], name="Top 5 cidade", mode="markers",
                            marker=dict(color="#d29922", size=9, symbol="diamond"))
            fig.add_scatter(x="H" + pa["hab"].astype(str), y=pa["pct_brasil"], name="Brasil", mode="markers",
                            marker=dict(color="#6b7280", size=7))
            fig.update_layout(title=f"{NOME_AREA[area_sel]} — % de acerto por habilidade", height=380,
                              yaxis_title="% de acerto", legend=dict(orientation="h", y=-0.2), margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

            comp = perfil_competencias(pa, None, esc_h, anos_sel)
            st.markdown("**Por competência**")
            st.dataframe(comp[["competencia", "desc_competencia", "pct_escola", "pct_top5", "pct_brasil", "gap_top5"]]
                         .rename(columns={"competencia": "C", "desc_competencia": "Competência", "pct_escola": "Escola %",
                                          "pct_top5": "Top 5 %", "pct_brasil": "Brasil %", "gap_top5": "Distância"}),
                         hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.0f") for c in
                                        ["Escola %", "Top 5 %", "Brasil %", "Distância"]})
            st.download_button("⬇️ Baixar CSV (todas as habilidades)", perf.to_csv(index=False, sep=";", decimal=","),
                               file_name="habilidades_escola.csv", mime="text/csv")

# ------------------------------------------------------------------ metodologia
with aba4:
    st.markdown("""
**Fonte:** microdados do ENEM (INEP) + Censo Escolar (nomes, rede e identificação das escolas).

**Média geral** = média simples das 5 notas (CN, CH, LC, MT e Redação) dos participantes presentes nos dois dias.
A média da escola é a média dos seus participantes que informaram aquela escola.

**Código da escola:** o INEP retirou o `CO_ESCOLA` dos microdados publicados de 2020 a 2023 (adequação à LGPD).
O pipeline detecta isso automaticamente; anos sem o código não entram no ranking por escola.

**Escolas adventistas:** identificadas pelo nome no Censo Escolar (ex.: "Colégio Adventista…", "Escola Adventista…",
IASP, UNASP…). Para incluir uma escola manualmente, adicione o código INEP em `data/adventistas_extra.csv`.

**Cuidados:** ranking do ENEM não é avaliação oficial de qualidade; escolas com poucos participantes oscilam muito,
e o perfil socioeconômico dos alunos influencia fortemente as notas. Use o filtro de mínimo de participantes
e a visão de vários anos para comparações mais justas.

**Meta por área (aba Top 5):** como a média geral é a média das 5 áreas, cada +10 pts em uma área vale +2 pts na geral.
A diferença para o 5º colocado é convertida em pontos totais e distribuída proporcionalmente às áreas em que a escola
está mais distante da média das 5 líderes.
""")
