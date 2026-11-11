"""Painel: Ranking ENEM por escola. Rode com:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analise import AREAS, carregar, diagnostico_top5, media_periodo, ranquear
from relatorio import TEXTO_PADRAO_INVESTIR, gerar_pdf, nome_arquivo

ROOT = Path(__file__).resolve().parent
DADOS = ROOT / "data" / "processed" / "escolas_ano.parquet"
COER = ROOT / "data" / "processed" / "coerencia_escola.parquet"

st.set_page_config(page_title="Ranking ENEM por Escola", page_icon="🏫", layout="wide")


@st.cache_data
def dados(mtime: float) -> pd.DataFrame:  # mtime invalida o cache quando o parquet é regerado
    return carregar(DADOS)


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
        t = d[(d["ano"] == int(periodo)) & (d["n"] >= min_alunos)].sort_values("media_geral", ascending=False)
    t = t.reset_index(drop=True)
    t.insert(0, "posição", t["media_geral"].rank(ascending=False, method="min").astype(int))
    return t


COLS = {"posição": "Pos.", "nome": "Escola", "municipio": "Município", "uf": "UF", "dependencia": "Rede",
        "n": "Particip.", "media_geral": "Média geral", "media_cn": "CN", "media_ch": "CH", "media_lc": "LC",
        "media_mt": "MT", "media_redacao": "Redação", "pct_acima_700": "% ≥700", "anos": "Anos"}


def mostrar(t: pd.DataFrame, altura=600):
    if busca:
        t = t[t["nome"].str.contains(busca, case=False, na=False)]
    cols = [c for c in COLS if c in t]
    st.dataframe(
        t[cols].rename(columns=COLS), height=altura, hide_index=True, use_container_width=True,
        column_config={
            "Média geral": st.column_config.ProgressColumn(format="%.1f", min_value=300, max_value=850),
            **{k: st.column_config.NumberColumn(format="%.1f") for k in ["CN", "CH", "LC", "MT", "Redação"]},
            "% ≥700": st.column_config.NumberColumn(format="%.0f%%"),
        })
    st.download_button("⬇️ Baixar CSV", t[cols].rename(columns=COLS).to_csv(index=False, sep=";", decimal=","),
                       file_name="ranking_enem.csv", mime="text/csv")


local = "Brasil" if uf == "Brasil" else (uf if municipio == "Todos" else f"{municipio}/{uf}")
st.title("🏫 Ranking ENEM por Escola")
st.caption(f"Recorte: **{local}** · Período: **{periodo}** · mínimo de {min_alunos} participantes")

aba1, aba2, aba3, aba4 = st.tabs(["📊 Ranking geral", "📘 Rede Adventista", "🎯 Como chegar ao Top 5", "ℹ️ Metodologia"])

# ------------------------------------------------------------------ ranking geral
with aba1:
    rk = tabela_ranking(recorte(df))
    pct = lambda s: f"{100*s:.0f}%"  # noqa: E731
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Escolas no ranking", f"{len(rk):,}".replace(",", "."))
    c2.metric("Média das escolas", f"{rk['media_geral'].mean():.1f}" if len(rk) else "—")
    c3.metric("Nota de corte do Top 5", f"{rk['media_geral'].iloc[4]:.1f}" if len(rk) >= 5 else "—")
    c4.metric("Adventistas no recorte", int(rk["adventista"].sum()) if len(rk) else 0)
    mostrar(rk)

    if len(rk):
        g1, g2 = st.columns(2)
        with g1:
            fig = px.box(rk, x="dependencia", y="media_geral", points=False, title="Distribuição por rede",
                         labels={"dependencia": "", "media_geral": "Média geral"})
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            top = rk.head(15).iloc[::-1]
            fig = px.bar(top, x="media_geral", y="nome", orientation="h", title="Top 15",
                         color="adventista", color_discrete_map={True: "#1f6feb", False: "#9aa4b2"},
                         labels={"media_geral": "Média geral", "nome": "", "adventista": "Adventista"})
            fig.update_xaxes(range=[top["media_geral"].min() - 20, top["media_geral"].max() + 5])
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
            pos_municipio=base.groupby(["uf", "municipio"])["media_geral"].rank(ascending=False, method="min"),
            pos_uf=base.groupby("uf")["media_geral"].rank(ascending=False, method="min"),
            pos_brasil=base["media_geral"].rank(ascending=False, method="min"))
        corte_mun = base.groupby(["uf", "municipio"])["media_geral"].apply(
            lambda s: s.nlargest(5).min() if len(s) >= 5 else float("nan")).rename("corte_top5_mun")
        corte_uf = base.groupby("uf")["media_geral"].apply(lambda s: s.nlargest(5).min()).rename("corte_top5_uf")
        rk_adv = (rk_adv.merge(base[["co_escola", "pos_municipio", "pos_uf", "pos_brasil"]], on="co_escola", how="left")
                  .merge(corte_mun, on=["uf", "municipio"], how="left").merge(corte_uf, on="uf", how="left"))
        rk_adv["falta_top5_mun"] = (rk_adv["corte_top5_mun"] - rk_adv["media_geral"]).clip(lower=0)
        rk_adv["falta_top5_uf"] = (rk_adv["corte_top5_uf"] - rk_adv["media_geral"]).clip(lower=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Escolas adventistas", len(rk_adv))
        c2.metric("Média da rede", f"{rk_adv['media_geral'].mean():.1f}" if len(rk_adv) else "—")
        c3.metric("No Top 5 do município", int((rk_adv["pos_municipio"] <= 5).sum()))
        c4.metric("No Top 5 do estado", int((rk_adv["pos_uf"] <= 5).sum()))

        t = rk_adv.copy()
        if busca:
            t = t[t["nome"].str.contains(busca, case=False, na=False)]
        st.dataframe(
            t[["posição", "nome", "municipio", "uf", "n", "media_geral", "media_mt", "media_redacao",
               "pos_municipio", "falta_top5_mun", "pos_uf", "falta_top5_uf", "pos_brasil"]].rename(columns={
                **COLS, "pos_municipio": "Pos. município", "falta_top5_mun": "Falta p/ Top 5 mun.",
                "pos_uf": "Pos. UF", "falta_top5_uf": "Falta p/ Top 5 UF", "pos_brasil": "Pos. Brasil"}),
            hide_index=True, use_container_width=True, height=550,
            column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                           ["Média geral", "MT", "Redação", "Falta p/ Top 5 mun.", "Falta p/ Top 5 UF"]}
            | {c: st.column_config.NumberColumn(format="%d") for c in ["Pos. município", "Pos. UF", "Pos. Brasil"]})
        st.download_button("⬇️ Baixar CSV (adventistas)", t.to_csv(index=False, sep=";", decimal=","),
                           file_name="ranking_adventistas.csv", mime="text/csv")

        # evolução da rede x demais privadas
        evo = (df[df["n"] >= min_alunos].assign(grupo=lambda d: d["adventista"].map({True: "Adventistas", False: None})
                                                .fillna(d["dependencia"]))
               .groupby(["ano", "grupo"]).apply(lambda g: (g["media_geral"] * g["n"]).sum() / g["n"].sum())
               .rename("media").reset_index())
        if uf != "Brasil":
            st.caption("Gráfico de evolução considera o Brasil inteiro.")
        fig = px.line(evo, x="ano", y="media", color="grupo", markers=True,
                      title="Evolução da média (ponderada por participantes)",
                      labels={"media": "Média geral", "ano": "Ano", "grupo": ""})
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
                                                                     remetente, texto, referencia=ref))
        if st.session_state.get("pdf") and st.session_state["pdf"][:2] == (escolha, ano_d):
            st.download_button("⬇️ Baixar PDF", st.session_state["pdf"][2], mime="application/pdf",
                               file_name=nome_arquivo(df.loc[df["co_escola"] == escolha, "nome"].iloc[0], ano_d))

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
