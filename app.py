import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# ========== Coleta de dados ==========
@st.cache_data(show_spinner=True)
def fetch_data(ticker, start_date, end_date, interval):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date + datetime.timedelta(days=1),
        interval=interval,
        progress=False
    )
    if df.empty:
        return df
    df.reset_index(inplace=True)
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
    elif 'Date' in df.columns:
        df.rename(columns={'Date': 'timestamp'}, inplace=True)

    df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    return df

# ========== Aplicar indicadores ==========
def apply_indicators(df, rsi_period, ema_short, ema_long):
    df['rsi'] = RSIIndicator(df['close'], window=rsi_period).rsi()
    df['ema_short'] = EMAIndicator(df['close'], window=ema_short).ema_indicator()
    df['ema_long'] = EMAIndicator(df['close'], window=ema_long).ema_indicator()
    return df

# ========== Backtest com reinvestimento + alavancagem ==========
def backtest(df, rsi_overbought, rsi_oversold, initial_capital=10000, leverage=1):
    trades = []
    position = None
    capital = initial_capital
    df = df.copy().reset_index(drop=True)

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        cross_up = prev['ema_short'] < prev['ema_long'] and row['ema_short'] >= row['ema_long']
        cross_down = prev['ema_short'] > prev['ema_long'] and row['ema_short'] <= row['ema_long']

        if position:
            if position['type'] == 'buy' and cross_down:
                exit_price = row['close']
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                pnl = pnl_pct * capital * leverage
                capital += pnl
                trades.append({
                    **position,
                    'exit_price': exit_price,
                    'exit_time': row['timestamp'],
                    'exit_reason': 'ema_cross',
                    'pnl': pnl,
                    'capital': capital
                })
                position = None
                continue

            elif position['type'] == 'sell' and cross_up:
                exit_price = row['close']
                pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
                pnl = pnl_pct * capital * leverage
                capital += pnl
                trades.append({
                    **position,
                    'exit_price': exit_price,
                    'exit_time': row['timestamp'],
                    'exit_reason': 'ema_cross',
                    'pnl': pnl,
                    'capital': capital
                })
                position = None
                continue

        if position:
            if position['type'] == 'buy' and row['rsi'] > rsi_overbought:
                exit_price = row['close']
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                pnl = pnl_pct * capital * leverage
                capital += pnl
                trades.append({
                    **position,
                    'exit_price': exit_price,
                    'exit_time': row['timestamp'],
                    'exit_reason': 'rsi>overbought',
                    'pnl': pnl,
                    'capital': capital
                })
                position = None
                continue

            elif position['type'] == 'sell' and row['rsi'] < rsi_oversold:
                exit_price = row['close']
                pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
                pnl = pnl_pct * capital * leverage
                capital += pnl
                trades.append({
                    **position,
                    'exit_price': exit_price,
                    'exit_time': row['timestamp'],
                    'exit_reason': 'rsi<oversold',
                    'pnl': pnl,
                    'capital': capital
                })
                position = None
                continue

        if not position:
            if row['rsi'] < rsi_oversold and row['ema_short'] > row['ema_long']:
                position = {
                    'type': 'buy',
                    'entry_price': row['close'],
                    'entry_time': row['timestamp']
                }

            elif row['rsi'] > rsi_overbought and row['ema_short'] < row['ema_long']:
                position = {
                    'type': 'sell',
                    'entry_price': row['close'],
                    'entry_time': row['timestamp']
                }

    if position:
        row = df.iloc[-1]
        exit_price = row['close']
        pnl_pct = (
            (exit_price - position['entry_price']) / position['entry_price']
            if position['type'] == 'buy'
            else (position['entry_price'] - exit_price) / position['entry_price']
        )
        pnl = pnl_pct * capital * leverage
        capital += pnl
        trades.append({
            **position,
            'exit_price': exit_price,
            'exit_time': row['timestamp'],
            'exit_reason': 'final',
            'pnl': pnl,
            'capital': capital
        })

    return pd.DataFrame(trades)

# ========== Curva de Capital ==========
def plot_equity(trades):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trades['exit_time'], y=trades['capital'], mode='lines+markers', name='Capital'))
    fig.update_layout(title='📈 Curva de Capital', xaxis_title='Data', yaxis_title='Capital (USD)')
    return fig

# ========== Streamlit App ==========
st.set_page_config(page_title="RSI + EMA Backtest (Alavancado)", layout="wide")
st.title("📊 Backtest: RSI 3 + EMAs com Reinvestimento e Alavancagem")

# ========== Sidebar ==========
with st.sidebar:
    st.header("⚙️ Parâmetros")
    ticker = st.selectbox("Par", ['BTC-USD', 'BNB-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD'])
    interval = st.selectbox("Tempo Gráfico", ['5m', '15m', '1h', '4h', '1d'], index=3)
    start_date = st.date_input("Data Inicial", datetime.date.today() - datetime.timedelta(days=30))
    end_date = st.date_input("Data Final", datetime.date.today())
    initial_capital = st.number_input("Capital Inicial (USD)", value=10000, min_value=100)
    leverage = st.slider("Alavancagem (x)", 1, 15, 1)
    rsi_period = st.slider("RSI Período", 2, 14, 3)
    rsi_oversold = st.slider("RSI Sobrevendido", 10, 40, 30)
    rsi_overbought = st.slider("RSI Sobrecomprado", 60, 90, 70)
    ema_short = st.slider("EMA Curta", 3, 20, 6)
    ema_long = st.slider("EMA Longa", 10, 50, 14)

    if interval in ['1m', '5m', '15m']:
        st.caption("⚠️ Histórico limitado para intervalos curtos.")

# ========== Execução ==========
if st.button("🚀 Rodar Backtest"):
    with st.spinner("Buscando dados e executando..."):
        df = fetch_data(ticker, start_date, end_date, interval)
        if df.empty:
            st.error("❌ Nenhum dado retornado.")
        else:
            df = apply_indicators(df, rsi_period, ema_short, ema_long)
            trades = backtest(df, rsi_overbought, rsi_oversold, initial_capital, leverage)

            if not trades.empty:
                final_capital = trades['capital'].iloc[-1]
                lucro_total = final_capital - initial_capital
                lucro_percentual = (lucro_total / initial_capital) * 100
                total_trades = len(trades)
                wins = trades[trades['pnl'] > 0]
                losses = trades[trades['pnl'] < 0]
                winrate = round((len(wins) / total_trades) * 100, 2)
                payoff = round(wins['pnl'].mean() / abs(losses['pnl'].mean()), 2) if not losses.empty else 0

                st.success(f"""
✅ Total de operações: {total_trades}  
💰 Lucro líquido: {lucro_total:.2f} USD  
📈 Lucro percentual: {lucro_percentual:.2f}%  
🏆 Winrate: {winrate}%  
📊 Payoff: {payoff}
""")
                st.plotly_chart(plot_equity(trades), use_container_width=True)
                st.subheader("📜 Operações Executadas")
                st.dataframe(trades)

                csv = trades.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Baixar CSV", data=csv, file_name="backtest_rsi_ema.csv", mime="text/csv")
            else:
                st.warning("⚠️ Nenhuma operação encontrada com os parâmetros definidos.")
else:
    st.info("Configure os parâmetros e clique em **Rodar Backtest**.")
