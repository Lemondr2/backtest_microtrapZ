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

# ========== Backtest ==========
def backtest(df, rsi_period, rsi_overbought, rsi_oversold):
    position = None
    trades = []
    capital = 10000
    df = df.copy().reset_index(drop=True)

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        cross_up = df['ema_short'][i - 1] < df['ema_long'][i - 1] and df['ema_short'][i] >= df['ema_long'][i]
        cross_down = df['ema_short'][i - 1] > df['ema_long'][i - 1] and df['ema_short'][i] <= df['ema_long'][i]

        if position:
            if position['type'] == 'buy' and cross_down:
                position['exit_price'] = row['close']
                position['exit_time'] = row['timestamp']
                position['exit_reason'] = 'ema_cross'
                trades.append(position)
                position = None
                continue
            elif position['type'] == 'sell' and cross_up:
                position['exit_price'] = row['close']
                position['exit_time'] = row['timestamp']
                position['exit_reason'] = 'ema_cross'
                trades.append(position)
                position = None
                continue

        if position:
            if position['type'] == 'buy' and row['rsi'] > rsi_overbought:
                position['exit_price'] = row['close']
                position['exit_time'] = row['timestamp']
                position['exit_reason'] = 'rsi>70'
                trades.append(position)
                position = None
                continue
            elif position['type'] == 'sell' and row['rsi'] < rsi_oversold:
                position['exit_price'] = row['close']
                position['exit_time'] = row['timestamp']
                position['exit_reason'] = 'rsi<30'
                trades.append(position)
                position = None
                continue

        if not position:
            if (
                row['rsi'] < rsi_oversold
                and row['close'] > row['ema_short'] > row['ema_long']
            ):
                position = {
                    'type': 'buy',
                    'entry_price': row['close'],
                    'entry_time': row['timestamp']
                }

            elif (
                row['rsi'] > rsi_overbought
                and row['close'] < row['ema_short'] < row['ema_long']
            ):
                position = {
                    'type': 'sell',
                    'entry_price': row['close'],
                    'entry_time': row['timestamp']
                }

    if position:
        position['exit_price'] = df.iloc[-1]['close']
        position['exit_time'] = df.iloc[-1]['timestamp']
        position['exit_reason'] = 'final'
        trades.append(position)

    trades = pd.DataFrame(trades)
    if not trades.empty:
        trades['pnl'] = np.where(
            trades['type'] == 'buy',
            trades['exit_price'] - trades['entry_price'],
            trades['entry_price'] - trades['exit_price']
        )
        trades['capital'] = capital + trades['pnl'].cumsum()

    return trades

# ========== Curva de Capital ==========
def plot_equity(trades):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trades['exit_time'], y=trades['capital'], mode='lines+markers', name='Capital'))
    fig.update_layout(title='📈 Curva de Capital', xaxis_title='Data', yaxis_title='Capital (USD)')
    return fig

# ========== Streamlit App ==========
st.set_page_config(page_title="RSI + EMA Backtest", layout="wide")
st.title("📊 Backtest: Estratégia RSI 3 + EMAs")

# ========== Sidebar ==========
with st.sidebar:
    st.header("⚙️ Parâmetros")
    ticker = st.selectbox("Par", ['BTC-USD', 'BNB-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD'])
    interval = st.selectbox("Tempo Gráfico", ['5m', '15m', '1h', '4h', '1d'], index=3)
    start_date = st.date_input("Data Inicial", datetime.date.today() - datetime.timedelta(days=30))
    end_date = st.date_input("Data Final", datetime.date.today())
    rsi_period = st.slider("RSI Período", 2, 14, 3)
    rsi_oversold = st.slider("RSI Sobrevendido", 10, 40, 30)
    rsi_overbought = st.slider("RSI Sobrecomprado", 60, 90, 70)
    ema_short = st.slider("EMA Curta", 3, 20, 6)
    ema_long = st.slider("EMA Longa", 10, 50, 14)

    if interval in ['1m', '5m', '15m']:
        st.caption("⚠️ Intervalos intradiários podem ter no máximo 30 dias de dados históricos.")

# ========== Execução ==========
if st.button("🚀 Rodar Backtest"):
    with st.spinner("Buscando dados e executando backtest..."):
        df = fetch_data(ticker, start_date, end_date, interval)
        if df.empty:
            st.error("❌ Nenhum dado retornado para o período selecionado.")
        else:
            df = apply_indicators(df, rsi_period, ema_short, ema_long)
            trades = backtest(df, rsi_period, rsi_overbought, rsi_oversold)

            if not trades.empty:
                total_trades = len(trades)
                lucro_total = round(trades['pnl'].sum(), 2)
                wins = trades[trades['pnl'] > 0]
                losses = trades[trades['pnl'] < 0]
                winrate = round((len(wins) / total_trades) * 100, 2)
                payoff = round(wins['pnl'].mean() / abs(losses['pnl'].mean()), 2) if not losses.empty else 0

                st.success(f"""
✅ Total de operações: {total_trades}  
💰 Lucro líquido: {lucro_total:.2f} USD  
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
    st.info("Configure os parâmetros no menu lateral e clique em **Rodar Backtest**.")
