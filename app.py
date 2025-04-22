import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator

@st.cache_data(show_spinner=True)
def fetch_binance_data(symbol='BTC/USDT', timeframe='1m', start_date=None, end_date=None):
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date.strftime('%Y-%m-%dT%H:%M:%S'))
    end_timestamp = exchange.parse8601(end_date.strftime('%Y-%m-%dT%H:%M:%S'))
    all_candles = []
    limit = 1000

    while since < end_timestamp:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break
        since = candles[-1][0] + 60 * 1000
        all_candles += candles
        if len(candles) < limit:
            break

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[(df['timestamp'] >= pd.to_datetime(start_date)) & (df['timestamp'] <= pd.to_datetime(end_date))]
    return df

def apply_indicators(df, atr_period=9, ema_short=8, ema_long=20):
    df['ema_short'] = EMAIndicator(df['close'], window=ema_short).ema_indicator()
    df['ema_long'] = EMAIndicator(df['close'], window=ema_long).ema_indicator()
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
    fig.update_layout(title='Evolução do Capital', xaxis_title='Data', yaxis_title='Capital (USDT)')
    return fig

# ==========================
# Streamlit App
# ==========================

st.set_page_config(page_title="Backtest Microtrap", layout="wide")
st.title("📊 Backtest: Estratégia Microtrap")
st.markdown("💡 Estratégia baseada em price action (microtrap), médias móveis e ATR.")

with st.sidebar:
    st.header("⚙️ Configurações")
    symbol = st.selectbox("Par de Moeda", ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT'])
    start_date = st.date_input("Data Inicial", datetime.date.today() - datetime.timedelta(days=90))
    end_date = st.date_input("Data Final", datetime.date.today())

    atr_period = st.slider("Período ATR", 5, 20, 9)
    ema_short = st.slider("EMA Curta", 5, 15, 8)
    ema_long = st.slider("EMA Longa", 15, 40, 20)
    risk = st.slider("Risco por Trade (%)", 0.5, 5.0, 1.0) / 100
    rr = st.slider("Razão Risco/Retorno", 0.5, 3.0, 0.8)

if st.button("🚀 Rodar Backtest"):
    with st.spinner("Buscando dados e executando backtest..."):
        df = fetch_binance_data(symbol=symbol, start_date=start_date, end_date=end_date)
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
💰 Lucro líquido: {net_profit} USDT  
🏆 Winrate: {winrate}%  
📊 Payoff: {payoff}
""")

            st.plotly_chart(plot_equity_curve(results), use_container_width=True)
            st.subheader("📈 Operações Executadas")
            st.dataframe(results)

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar CSV", data=csv, file_name="backtest_microtrap.csv", mime='text/csv')
        else:
            st.warning("Nenhuma operação encontrada com os parâmetros atuais.")
else:
    st.info("Configure os parâmetros no menu lateral e clique em **Rodar Backtest**.")

