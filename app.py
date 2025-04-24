import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# ==============================
# Função para buscar os dados
# ==============================

@st.cache_data(show_spinner=True)
def fetch_data(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date + datetime.timedelta(days=1),
        interval='4h',
        progress=False
    )
    if df.empty:
        return df
    df.reset_index(inplace=True)
    df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
    df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    return df

# ==============================
# Estratégia RSI + EMAs
# ==============================

def apply_indicators(df):
    df['rsi'] = RSIIndicator(df['close'], window=3).rsi()
    df['ema_6'] = EMAIndicator(df['close'], window=6).ema_indicator()
    df['ema_14'] = EMAIndicator(df['close'], window=14).ema_indicator()
    return df

def backtest_rsi_strategy(df, capital=10000):
    df = df.copy().reset_index(drop=True)
    position = None
    results = []
    stop_price = None
    capital_series = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        ema_cross_up = df['ema_6'][i - 1] < df['ema_14'][i - 1] and df['ema_6'][i] >= df['ema_14'][i]
        ema_cross_down = df['ema_6'][i - 1] > df['ema_14'][i - 1] and df['ema_6'][i] <= df['ema_14'][i]

        # Stop por cruzamento de médias
        if position:
            if position['type'] == 'buy' and ema_cross_down:
                stop_price = row['low']
                if row['close'] <= stop_price:
                    results.append({**position, 'exit_reason': 'stop_cross', 'exit_price': stop_price, 'exit_time': row['timestamp']})
                    position = None
                    continue
            elif position['type'] == 'sell' and ema_cross_up:
                stop_price = row['high']
                if row['close'] >= stop_price:
                    results.append({**position, 'exit_reason': 'stop_cross', 'exit_price': stop_price, 'exit_time': row['timestamp']})
                    position = None
                    continue

        # Saída por RSI
        if position:
            if position['type'] == 'buy' and prev_row['rsi'] > 70:
                results.append({**position, 'exit_reason': 'rsi>70', 'exit_price': row['open'], 'exit_time': row['timestamp']})
                position = None
                continue
            elif position['type'] == 'sell' and prev_row['rsi'] < 30:
                results.append({**position, 'exit_reason': 'rsi<30', 'exit_price': row['open'], 'exit_time': row['timestamp']})
                position = None
                continue

        # Entrada comprada
        if not position:
            if prev_row['rsi'] < 30 and prev_row['close'] > prev_row['ema_6'] > prev_row['ema_14']:
                position = {
                    'type': 'buy',
                    'entry_price': row['open'],
                    'entry_time': row['timestamp']
                }
                continue

            # Entrada vendida
            if prev_row['rsi'] > 70 and prev_row['close'] < prev_row['ema_6'] < prev_row['ema_14']:
                position = {
                    'type': 'sell',
                    'entry_price': row['open'],
                    'entry_time': row['timestamp']
                }
                continue

    # Fechar operação aberta no final do dataset
    if position:
        position['exit_reason'] = 'final'
        position['exit_price'] = df.iloc[-1]['close']
        position['exit_time'] = df.iloc[-1]['timestamp']
        results.append(position)

    # Resultados em DataFrame
    trades = pd.DataFrame(results)
    if not trades.empty:
        trades['pnl'] = np.where(
            trades['type'] == 'buy',
            trades['exit_price'] - trades['entry_price'],
            trades['entry_price'] - trades['exit_price']
        )
        trades['pnl_usd'] = trades['pnl']
        trades['capital'] = capital + trades['pnl_usd'].cumsum()

    return trades

# ==============================
# Plot da Curva de Capital
# ==============================

def plot_equity(trades):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trades['exit_time'], y=trades['capital'], mode='lines+markers', name='Capital'))
    fig.update_layout(title='📈 Curva de Capital', xaxis_title='Data', yaxis_title='Capital (USD)')
    return fig

# ==============================
# Streamlit App
# ==============================

st.set_page_config(page_title="Backtest RSI 3p com EMAs", layout="wide")
st.title("📊 Backtest: Estratégia RSI 3 + EMAs (4H)")
st.markdown("📈 Estratégia baseada em sobrecompra/sobrevenda do RSI 3 períodos com contexto de EMAs e stop por cruzamento.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    ticker = st.selectbox("Par de Moeda", ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD'])
    start_date = st.date_input("Data Inicial", datetime.date.today() - datetime.timedelta(days=90))
    end_date = st.date_input("Data Final", datetime.date.today())
    capital = st.number_input("Capital Inicial (USD)", value=10000)

# Execução
if st.button("🚀 Rodar Backtest"):
    with st.spinner("Buscando dados e executando estratégia..."):
        df = fetch_data(ticker, start_date, end_date)

        if df.empty:
            st.error("❌ Nenhum dado encontrado para o período selecionado.")
        else:
            df = apply_indicators(df)
            trades = backtest_rsi_strategy(df, capital)

            if not trades.empty:
                lucro_total = round(trades['pnl_usd'].sum(), 2)
                win_trades = trades[trades['pnl_usd'] > 0]
                loss_trades = trades[trades['pnl_usd'] < 0]
                winrate = round(len(win_trades) / len(trades) * 100, 2)
                payoff = round(win_trades['pnl_usd'].mean() / abs(loss_trades['pnl_usd'].mean()), 2) if not loss_trades.empty else 0

                st.success(f"""
✅ Total de operações: {len(trades)}  
💰 Lucro líquido: {lucro_total} USD  
🏆 Winrate: {winrate}%  
📊 Payoff: {payoff}
""")

                st.plotly_chart(plot_equity(trades), use_container_width=True)
                st.subheader("📜 Operações Executadas")
                st.dataframe(trades)

                csv = trades.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Baixar Resultados (.CSV)", data=csv, file_name="backtest_rsi3_emas.csv", mime="text/csv")
            else:
                st.warning("⚠️ Nenhuma operação foi encontrada com os parâmetros atuais.")
else:
    st.info("Configure os parâmetros no menu lateral e clique em **Rodar Backtest**.")
