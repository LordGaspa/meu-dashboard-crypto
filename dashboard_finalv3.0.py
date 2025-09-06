# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# CÓDIGO ÔMEGA - VERSÃO ABSOLUTA (v8 com API Keys)
#
# OBJETIVO:
# - Adicionar autenticação via API Key para funcionar no Streamlit Cloud.
# - Unificar painéis de informação para evitar sobreposição.
# - Corrigir a opacidade das camadas para garantir a visibilidade do Buy & Hold.
# ----------------------------------------------------------------------------

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from binance.client import Client

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    layout="wide",
    page_title="Código Ômega - Performance",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>[data-testid="stMetricValue"] {font-size: 2.2em;}</style>""",
    unsafe_allow_html=True,
)

# --- 2. PARÂMETROS ---
PERIODO_CANDLE = Client.KLINE_INTERVAL_12HOUR
MEDIA_RAPIDA_PER, MEDIA_LENTA_PER, MEDIA_FILTRO_TENDENCIA_PER = 9, 60, 200
ATR_PERIODO, ATR_MULTIPLICADOR = 14, 3.0
CAPITAL_INICIAL, TAXA_CORRETAGEM, ANOS_DE_DADOS_BACKTEST = 1000.0, 0.001, 8


# --- 3. FUNÇÕES ---
def calcular_atr(df, period=14):
    ranges = pd.concat(
        [
            df["maxima"] - df["minima"],
            (df["maxima"] - df["fechamento"].shift()).abs(),
            (df["minima"] - df["fechamento"].shift()).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()


def executar_backtest_completo(df):
    capital, posicionado, quantidade_ativo, stop_loss_price = (
        CAPITAL_INICIAL,
        False,
        0.0,
        0.0,
    )
    trades, historico_capital = [], []
    for i in range(len(df)):
        row = df.iloc[i]
        data_atual = df.index[i]
        valor_portfolio = (
            capital if not posicionado else quantidade_ativo * row["fechamento"]
        )
        historico_capital.append(valor_portfolio)
        if i == 0:
            continue
        prev_row = df.iloc[i - 1]
        sinal_compra = (
            prev_row["media_rapida"] > prev_row["media_lenta"]
            and prev_row["fechamento"] > prev_row["media_filtro"]
        )
        sinal_venda_cruz = prev_row["media_rapida"] < prev_row["media_lenta"]
        if not posicionado and sinal_compra:
            preco_de_compra = row["abertura"]
            if preco_de_compra <= 0 or np.isnan(preco_de_compra):
                continue
            quantidade_compra = (capital / preco_de_compra) * (1 - TAXA_CORRETAGEM)
            capital, quantidade_ativo, posicionado = 0.0, quantidade_compra, True
            atr_da_compra = row["atr"] if not pd.isna(row["atr"]) else 0.0
            stop_loss_price = preco_de_compra - (atr_da_compra * ATR_MULTIPLICADOR)
            trades.append(
                {"data": data_atual, "tipo": "COMPRA", "preco": preco_de_compra}
            )
        elif posicionado:
            stop_ativado = row["minima"] < stop_loss_price
            if stop_ativado or sinal_venda_cruz:
                preco_saida = stop_loss_price if stop_ativado else row["abertura"]
                if preco_saida <= 0 or np.isnan(preco_saida):
                    continue
                capital = (quantidade_ativo * preco_saida) * (1 - TAXA_CORRETAGEM)
                quantidade_ativo, posicionado = 0.0, False
                tipo_venda = "VENDA_STOP_ATR" if stop_ativado else "VENDA_CRUZ"
                trades.append(
                    {"data": data_atual, "tipo": tipo_venda, "preco": preco_saida}
                )
    capital_final = (
        capital
        if not posicionado
        else (quantidade_ativo * df.iloc[-1]["fechamento"]) * (1 - TAXA_CORRETAGEM)
    )
    return float(capital_final), trades, historico_capital


def analisar_resultados_backtest(capital_final, trades, historico_capital, df):
    df_trades = (
        pd.DataFrame(trades)
        if trades
        else pd.DataFrame(columns=["data", "tipo", "preco"])
    )
    compras = df_trades[df_trades["tipo"] == "COMPRA"].reset_index(drop=True)
    vendas = df_trades[df_trades["tipo"].str.contains("VENDA")].reset_index(drop=True)
    
    trades_completos = []
    capital_acumulado = CAPITAL_INICIAL
    
    num_ciclos = min(len(compras), len(vendas))
    
    for i in range(num_ciclos):
        compra = compras.iloc[i]
        venda = vendas.iloc[i]
        
        resultado_bruto = venda["preco"] - compra["preco"]
        resultado_pct = (resultado_bruto / compra["preco"]) * 100
        
        capital_antes_trade = capital_acumulado
        quantidade = (capital_antes_trade / compra["preco"])
        capital_depois_trade = quantidade * venda["preco"] * (1 - TAXA_CORRETAGEM)**2
        ganho_trade = capital_depois_trade - capital_antes_trade
        capital_acumulado = capital_depois_trade

        trades_completos.append({
            "Data Compra": compra["data"],
            "Preço Compra": compra["preco"],
            "Data Venda": venda["data"],
            "Preço Venda": venda["preco"],
            "Resultado R$": ganho_trade,
            "Resultado %": resultado_pct,
            "Capital Acumulado": capital_acumulado,
        })

    df_trades_completos = pd.DataFrame(trades_completos)

    resultados = {"capital_final_estrategia": float(capital_final)}
    resultados["retorno_estrategia_pct"] = (
        (capital_final - CAPITAL_INICIAL) / CAPITAL_INICIAL
    ) * 100.0
    preco_inicial_bh = df["abertura"].iloc[0]
    preco_final_bh = df["fechamento"].iloc[-1]
    resultados["retorno_bh_pct"] = (
        ((preco_final_bh - preco_inicial_bh) / preco_inicial_bh) * 100.0
        if preco_inicial_bh > 0
        else 0.0
    )
    
    resultados["num_ciclos"] = num_ciclos
    if num_ciclos > 0:
        vitorias = (df_trades_completos["Resultado R$"] > 0).sum()
        resultados["win_rate_pct"] = (vitorias / num_ciclos) * 100.0
    else:
        resultados["win_rate_pct"] = 0.0
        
    df_capital = pd.DataFrame(historico_capital, columns=["capital"])
    if not df_capital.empty:
        pico_anterior = df_capital["capital"].cummax()
        drawdown = (pico_anterior - df_capital["capital"]) / pico_anterior.replace(
            0, np.nan
        )
        resultados["max_drawdown_pct"] = (
            float(drawdown.max() * 100.0) if drawdown.notna().any() else 0.0
        )
    else:
        resultados["max_drawdown_pct"] = 0.0
        
    return resultados, df_trades_completos


@st.cache_data(ttl=60 * 60 * 6)
def carregar_e_processar_dados(symbol):
    # --- ALTERAÇÃO AQUI ---
    # Carrega as chaves de API dos segredos do Streamlit de forma segura
    api_key = st.secrets.get("API_KEY")
    api_secret = st.secrets.get("API_SECRET")
    
    # Inicializa o cliente com as chaves de API para autenticação
    client = Client(api_key, api_secret)
    # --- FIM DA ALTERAÇÃO ---
 
    start_date = (
        datetime.now() - timedelta(days=ANOS_DE_DADOS_BACKTEST * 365)
    ).strftime("%d %b, %Y")
    klines = client.get_historical_klines(symbol, PERIODO_CANDLE, start_date)
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame(
        klines,
        columns=[
            "tempo_abertura", "abertura", "maxima", "minima", "fechamento",
            "volume", "tempo_fechamento", "volume_quote", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    for col in ["abertura", "maxima", "minima", "fechamento", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df["tempo_fechamento"] = pd.to_datetime(df["tempo_fechamento"], unit="ms")
    df.set_index("tempo_fechamento", inplace=True)
    df["media_rapida"] = df["fechamento"].rolling(window=MEDIA_RAPIDA_PER).mean()
    df["media_lenta"] = df["fechamento"].rolling(window=MEDIA_LENTA_PER).mean()
    df["media_filtro"] = (
        df["fechamento"].rolling(window=MEDIA_FILTRO_TENDENCIA_PER).mean()
    )
    df["atr"] = calcular_atr(df, ATR_PERIODO)
    df.dropna(inplace=True)
    return df


def encontrar_sinal_vigente(df):
    for i in range(len(df) - 1, 0, -1):
        candle_atual, candle_anterior = df.iloc[i], df.iloc[i - 1]
        if (
            candle_anterior["media_rapida"] <= candle_anterior["media_lenta"]
            and candle_atual["media_rapida"] > candle_atual["media_lenta"]
            and candle_atual["fechamento"] > candle_atual["media_filtro"]
        ):
            return (
                "TENDÊNCIA DE ALTA 🟢",
                candle_atual.name,
                candle_atual["fechamento"],
                candle_atual["atr"],
            )
        elif (
            candle_anterior["media_rapida"] >= candle_anterior["media_lenta"]
            and candle_atual["media_rapida"] < candle_atual["media_lenta"]
        ):
            return (
                "TENDÊNCIA DE BAIXA 🔴",
                candle_atual.name,
                candle_atual["fechamento"],
                None,
            )
    return "NEUTRO ⚪", None, None, None


def gerar_grafico_de_impacto(df, historico_capital, symbol, resultados):
    capital_bh = (CAPITAL_INICIAL / df["abertura"].iloc[0]) * df["fechamento"]
    capital_estrategia = pd.Series(
        historico_capital, index=df.index[: len(historico_capital)]
    )
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            name="Buy & Hold",
            x=capital_bh.index,
            y=capital_bh,
            line=dict(color="#299958", width=2),
            fill="tozeroy",
            fillcolor="rgba(91, 203, 138, 0.5)",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Código Ômega",
            x=capital_estrategia.index,
            y=capital_estrategia,
            line=dict(color="#F39C12", width=3),
            fill="tozeroy",
            fillcolor="rgba(243, 156, 18, 0.4)",
        )
    )

    pico_valor = capital_estrategia.max()
    texto_metricas = (
        f"<b>Retorno Ômega:</b> <span style='color: #F39C12;'>{resultados['retorno_estrategia_pct']:,.2f}%</span><br>"
        f"<b>Retorno B&H:</b> {resultados['retorno_bh_pct']:,.2f}%<br>"
        f"────────────────────<br>"
        f"<b>Pico de Capital:</b> ${pico_valor:,.2f}<br>"
        f"<b>Taxa de Acerto:</b> {resultados['win_rate_pct']:.2f}%<br>"
        f"<b>Nº de Trades:</b> {resultados['num_ciclos']}"
    )
    fig.add_annotation(
        text=texto_metricas,
        xref="paper", yref="paper", x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        font=dict(size=16, color="black"),
        showarrow=False, bgcolor="rgba(255, 255, 255, 0.75)",
        bordercolor="rgba(169, 169, 169, 0.5)", borderwidth=1,
        align="left",
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Performance: Código Ômega vs. Mercado ({symbol})</b>",
            y=0.95, x=0.5, font=dict(size=24, color="black"),
        ),
        template="plotly_white", height=600,
        xaxis=dict(title="", tickfont=dict(size=14, color="black"), gridcolor="#D3D3D3"),
        yaxis=dict(
            title="Capital (USDT)", tickfont=dict(size=14, color="black"),
            gridcolor="#D3D3D3",
        ),
        legend=dict(
            x=0.02, y=0.65, xanchor="left", yanchor="top",
            font=dict(size=14, color="black"),
            bgcolor="rgba(255,255,255,0.5)",
        ),
        hovermode="x unified", paper_bgcolor="#F0F0F0", plot_bgcolor="#F0F0F0",
    )
    return fig


def gerar_grafico_de_trades(df, trades, symbol):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Preço", x=df.index, y=df["fechamento"],
            line=dict(color="blue", width=1.5),
        )
    )
    if trades:
        df_trades = pd.DataFrame(trades)
        compras = df_trades[df_trades["tipo"] == "COMPRA"]
        vendas = df_trades[df_trades["tipo"].str.contains("VENDA")]
        fig.add_trace(
            go.Scatter(
                name="Compra", x=compras["data"], y=compras["preco"], mode="markers",
                marker=dict(
                    color="green", symbol="triangle-up", size=10,
                    line=dict(width=1, color="DarkSlateGrey"),
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                name="Venda", x=vendas["data"], y=vendas["preco"], mode="markers",
                marker=dict(
                    color="red", symbol="triangle-down", size=10,
                    line=dict(width=1, color="DarkSlateGrey"),
                ),
            )
        )
    fig.update_layout(
        title=f"Preço e Pontos de Trade para {symbol}", template="plotly_white",
        height=600, xaxis_title="", yaxis_title="Preço (USDT)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#F0F0F0", plot_bgcolor="#F0F0F0",
    )
    return fig


# --- 4. INTERFACE ---
st.title("🚀 Código Ômega - Performance")
st.sidebar.title("Controles")
ativo_selecionado = st.sidebar.selectbox("Selecione o Ativo:", ("BTCUSDT", "ETHUSDT"))
dados_completos = carregar_e_processar_dados(ativo_selecionado)

if dados_completos.empty:
    st.error(f"Não foi possível carregar os dados para {ativo_selecionado}. Verifique o símbolo ou a conexão com a API.")
else:
    with st.spinner(f"Executando backtest para {ativo_selecionado}..."):
        capital_final, trades, historico_capital = executar_backtest_completo(
            dados_completos.copy()
        )
        resultados, df_lista_trades = analisar_resultados_backtest(
            capital_final, trades, historico_capital, dados_completos
        )
    
    sinal_vigente, data_sinal, preco_no_sinal, atr_no_sinal = encontrar_sinal_vigente(
        dados_completos
    )
    preco_atual = dados_completos.iloc[-1]["fechamento"]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header(f"Painel: {ativo_selecionado}")
        st.metric("Preço Atual", f"${preco_atual:,.2f}")
        st.markdown("---")
        st.subheader("Sinal Vigente:")
        if "ALTA" in sinal_vigente:
            st.markdown(f"<h2 style='text-align: center; color: green;'>{sinal_vigente.split('🟢')[0].strip()} 🟢</h2>", unsafe_allow_html=True)
        elif "BAIXA" in sinal_vigente:
            st.markdown(f"<h2 style='text-align: center; color: red;'>{sinal_vigente.split('🔴')[0].strip()} 🔴</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='text-align: center; color: black;'>{sinal_vigente.split('⚪')[0].strip()} ⚪</h2>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Contexto do Último Sinal")
        if data_sinal:
            variacao = ((preco_atual - preco_no_sinal) / preco_no_sinal) * 100
            cor_variacao = "green" if variacao >= 0 else "red"
            st.markdown(
                f"**Início em:** {data_sinal.strftime('%d/%m/%y às %H:%M')}<br>"
                f"**Preço no Sinal:** ${preco_no_sinal:,.2f}<br>"
                f"**Variação:** <span style='color: {cor_variacao};'>{variacao:+.2f}%</span>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Aguardando novo sinal de entrada.")
        
        st.markdown("---")
        st.subheader("Análise de Risco (Nova Entrada)")
        if "ALTA" in sinal_vigente and atr_no_sinal:
            stop_calculado = preco_no_sinal - (atr_no_sinal * ATR_MULTIPLICADOR)
            risco_calculado = ((preco_no_sinal - stop_calculado) / preco_no_sinal) * 100
            st.markdown(
                f"**Stop Sugerido:** <span style='color: orange;'>${stop_calculado:,.2f}</span><br>"
                f"**Risco:** <span style='color: orange;'>{risco_calculado:.2f}%</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<span style='color: red;'><b>ENTRADA NÃO RECOMENDADA</b></span>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader("📊 Métricas de Performance")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Retorno Código Ômega", f"{resultados['retorno_estrategia_pct']:,.2f}%")
            st.metric("Nº de Trades", f"{resultados['num_ciclos']}")
        with m_col2:
            st.metric("Retorno Buy & Hold", f"{resultados['retorno_bh_pct']:,.2f}%")
            st.metric("Taxa de Acerto", f"{resultados['win_rate_pct']:.2f}%")
        st.metric(
            "Rebaixamento Máximo",
            f"-{resultados['max_drawdown_pct']:.2f}%",
            help="A maior queda do capital da estratégia, de um pico a um vale.",
        )

    with col2:
        tab_performance, tab_trades, tab_lista = st.tabs(
            ["Análise de Performance", "Gráfico de Trades", "Lista de Trades"]
        )
        with tab_performance:
            figura_de_impacto = gerar_grafico_de_impacto(
                dados_completos, historico_capital, ativo_selecionado, resultados
            )
            st.plotly_chart(figura_de_impacto, use_container_width=True)
            
        with tab_trades:
            figura_de_trades = gerar_grafico_de_trades(
                dados_completos, trades, ativo_selecionado
            )
            st.plotly_chart(figura_de_trades, use_container_width=True)

        with tab_lista:
            st.subheader("Histórico Detalhado de Operações")
            if not df_lista_trades.empty:
                # Formatação para melhor visualização
                df_formatado = df_lista_trades.copy()
                df_formatado["Data Compra"] = df_formatado["Data Compra"].dt.strftime('%d/%m/%Y')
                df_formatado["Data Venda"] = df_formatado["Data Venda"].dt.strftime('%d/%m/%Y')
                df_formatado["Preço Compra"] = df_formatado["Preço Compra"].apply(lambda x: f"${x:,.2f}")
                df_formatado["Preço Venda"] = df_formatado["Preço Venda"].apply(lambda x: f"${x:,.2f}")
                df_formatado["Resultado R$"] = df_formatado["Resultado R$"].apply(lambda x: f"${x:,.2f}")
                df_formatado["Resultado %"] = df_formatado["Resultado %"].apply(lambda x: f"{x:,.2f}%")
                df_formatado["Capital Acumulado"] = df_formatado["Capital Acumulado"].apply(lambda x: f"${x:,.2f}")
                st.dataframe(df_formatado, use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum trade completo foi realizado no período analisado.")

