import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator

# ==========================
# Coleta de dados com yfinance
# ==========================

@st.cache_data(show_spinner=True)
def fetch_yf_data(ticker='BTC-USD', interval='15m', start_date=None, end_date=None):
    df = yf.download(ticker, start=start_date, end=end_date + datetime.timedelta(days=1), interval=interval, progress=False)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df.reset_index()
    df = df[['Datetime', 'open', 'high', 'low', 'close', 'volume']]
    df = df.rename(columns={'Datetime': 'timestamp'})
    return df

# ==========================
# Indicadores e Estratégia
# ==========================

def apply_indicators(df, atr_period=9, ema_short=8, ema_long=20):
    if df.empty or 'close' not in df.columns or df['close'].isnull().all():
        raise ValueError("Dados inválidos para aplicar indicadores.")
    
    df = df.copy()
    df['ema_short'] = EMAIndicator(df['close'].fillna(method="ffill"), window=ema_short).ema_indicator()
    df['ema_long'] = EMAIndicator(df['close'].fillna(method="ffill"), window=ema_long).ema_indicator()
    
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_period)
    df['atr'] = atr.average_true_range()
    return df

def check_microtrap(df, idx):
    if idx < 3:
        return None
    c1, c2, c3, c4 = df.iloc[idx - 3], df.iloc[idx - 2], df.iloc[idx - 1], df.iloc[idx]
    if c1['low'] < c2['low'] and c3['low'] < c1['low'] and c4['close'] > c3['high']:
        return 'buy'
    if c1['high'] > c2['high'] and c3['high'] > c1['high'] and c4['close'] < c3['low']:
        return 'sell'
    return None

def backtest(df, risk=0.01, rr=0.8, capital=10000):
    results = []
    position = None

    for i in range(20, len(df) - 1):
        signal = check_microtrap(df, i)
        trend_up = df['ema_short'][i] > df['ema_long'][i] and df['ema_short'][i] > df['close'][i]
        trend_down = df['ema_short'][i] < df['ema_long'][i] and df['ema_short'][i] < df['close'][i]

        if signal == 'buy' and trend_up:
            entry = df['close'][i]
            stop = df['low'][i] - df['atr'][i]
            target = entry + (entry - stop) * rr
            position = {'type': 'buy', 'entry': entry, 'stop': stop, 'target': target, 'timestamp': df['timestamp'][i]}

        elif signal == 'sell' and trend_down:
            entry = df['close'][i]
            stop = df['high'][i] + df['atr'][i]
            target = entry - (stop - entry) * rr
            position = {'type': 'sell', 'entry': entry, 'stop': stop, 'target': target, 'timestamp': df['timestamp'][i]}

        if position:
            next_candle = df.iloc[i + 1]
            if position['type'] == 'buy':
                if next_candle['low'] <= position['stop']:
                    pnl = -risk * capital
                    results.append({**position, 'exit_price': position['stop'], 'pnl': pnl})
                    position = None
                elif next_candle['high'] >= position['target']:
                    pnl = risk * capital * rr
                    results.append({**position, 'exit_price': position['target'], 'pnl': pnl})
                    position = None
            elif position['type'] == 'sell':
                if next_candle['high'] >= position['stop']:
                    pnl = -risk * capital
                    results.append({**position, 'exit_price': position['stop'], 'pnl': pnl})
                    position = None
                elif next_candle['low'] <= position['target']:
                    pnl = risk * capital * rr
                    results.append({**position, 'exit_price': position['target'], 'pnl': pnl})
                    position = None

    return pd.DataFrame(results)

def plot_equity_curve(df):
    capital = 10000
    df['capital'] = capital + df['pnl'].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['capital'], mode='lines+markers', name='Capital'))
    fig.update_layout(title='Evolução do Capital', xaxis_title='Data', yaxis_title='Capital (USD)')
    return fig

# ==========================
# Streamlit App
# ==========================

st.set_page_config(page_title="Backtest Microtrap - yFinance", layout="wide")
st.title("📊 Backtest: Estratégia Microtrap com yFinance")
st.markdown("💡 Estratégia baseada em price action (microtrap), médias móveis e ATR. Usando dados do Yahoo Finance.")

# Sidebar de configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    ticker = st.selectbox("Ativo", ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD'])
    start_date = st.date_input("Data Inicial", datetime.date.today() - datetime.timedelta(days=60))
    end_date = st.date_input("Data Final", datetime.date.today())

    atr_period = st.slider("Período ATR", 5, 20, 9)
    ema_short = st.slider("EMA Curta", 5, 15, 8)
    ema_long = st.slider("EMA Longa", 15, 40, 20)
    risk = st.slider("Risco por Trade (%)", 0.5, 5.0, 1.0) / 100
    rr = st.slider("Razão Risco/Retorno", 0.5, 3.0, 0.8)

# Executar backtest
if st.button("🚀 Rodar Backtest"):
    with st.spinner("Carregando dados e executando..."):
        df = fetch_yf_data(ticker=ticker, start_date=start_date, end_date=end_date)

        # Validações seguras
        try:
            if df.empty:
                st.error("❌ Nenhum dado retornado para o período/ativo escolhido.")
            elif not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                st.error("❌ Colunas obrigatórias ausentes nos dados.")
            elif df['close'].isnull().all():
                st.error("❌ Todos os valores da coluna 'close' são nulos. Tente outro período ou ativo.")
            else:
                df = apply_indicators(df, atr_period=atr_period, ema_short=ema_short, ema_long=ema_long)
                results = backtest(df, risk=risk, rr=rr)

                if not results.empty:
                    total_trades = len(results)
                    net_profit = round(results['pnl'].sum(), 2)
                    wins = results[results['pnl'] > 0]
                    losses = results[results['pnl'] < 0]
                    winrate = round((len(wins) / total_trades) * 100, 2) if total_trades > 0 else 0
                    avg_win = wins['pnl'].mean() if not wins.empty else 0
                    avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
                    payoff = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0

                    st.success(f"""
✅ Total de operações: {total_trades}  
💰 Lucro líquido: {net_profit} USD  
🏆 Winrate: {winrate}%  
📊 Payoff: {payoff}
""")

                    st.plotly_chart(plot_equity_curve(results), use_container_width=True)
                    st.subheader("📈 Operações Executadas")
                    st.dataframe(results)

                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Baixar CSV", data=csv, file_name="backtest_microtrap.csv", mime='text/csv')
                else:
                    st.warning("Nenhuma operação encontrada com os parâmetros definidos.")
        except Exception as e:
            st.error(f"❌ Erro inesperado: {e}")
else:
    st.info("Configure os parâmetros e clique em **Rodar Backtest**.")
