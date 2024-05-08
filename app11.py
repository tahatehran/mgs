import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import sys
import time
import numpy as np 

try:
    import ta
except ImportError:
    subprocess.check_call([f"{sys.executable}", "-m", "pip", "install", "ta"])
    import ta

from ta.momentum import rsi
from ta.trend import macd, macd_signal, macd_diff

def get_tehran_time():
    response = requests.get('http://worldtimeapi.org/api/timezone/Asia/Tehran')
    time_data = response.json()
    return time_data['datetime']

@st.cache_data
def get_current_price(currency='BTC-USD'):
    url = f'https://api.coindesk.com/v1/bpi/currentprice/{currency}.json'
    response = requests.get(url)
    data = response.json()
    price = data['bpi'][currency.split('-')[1]]['rate_float']
    return price

def generate_signals(df, rsi_period=14, macd_short_period=12, macd_long_period=26, macd_signal_period=9, target_pct=0.05, stop_loss_pct=0.02):
    signals = pd.DataFrame(index=df.index)
    signals['rsi'] = rsi(df['close'], window=rsi_period)
    
    try:
        macd, signal, hist = macd(df['close'], window_slow=macd_long_period, window_fast=macd_short_period, window_signal=macd_signal_period)
        signals['macd'] = macd
        signals['signal'] = signal
    except:
        signals['macd'] = 0
        signals['signal'] = 0
        st.write("Error calculating MACD. Using default values.")
    
    # Calculate buy and sell signals
    signals['buy'] = (signals['rsi'] < 30) & (signals['macd'] > signals['signal'])
    signals['sell'] = (signals['rsi'] > 70) & (signals['macd'] < signals['signal'])
    
    # Calculate target and stop-loss prices
    signals['target'] = df['close'] * (1 + target_pct)
    signals['stop_loss'] = df['close'] * (1 - stop_loss_pct)
    
    # Calculate confidence levels
    signals['buy_confidence'] = 1 - (signals['rsi'] - 30) / 30
    signals['sell_confidence'] = 1 - (70 - signals['rsi']) / 30
    
    # Calculate error rates
    signals['buy_error'] = 1 - signals['buy_confidence']
    signals['sell_error'] = 1 - signals['sell_confidence']
    
    return signals

st.set_page_config(page_title='Cryptocurrency Price Tracker', page_icon=':moneybag:', layout='wide')
st.header('Cryptocurrency Price Tracker :chart_with_upwards_trend:')

tehran_time = get_tehran_time()
st.sidebar.write('Current time in Tehran:', tehran_time)

currency = st.sidebar.selectbox('Select Cryptocurrency', ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD'])
price = get_current_price(currency)
st.write(f'Current {currency.split("-")[0]} Price in {currency.split("-")[1]}: **{price}**')

if st.button('Refresh'):
    st.experimental_rerun()

st.subheader('Trading Signals :signal_strength:')

data_loading_text = st.empty()
data_loading_text.write('Loading data...')

try:
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    url = f'https://api.coindesk.com/v1/bpi/historical/close.json?start={start_date}&end={datetime.now().strftime("%Y-%m-%d")}&currency={currency.split("-")[1]}'
    response = requests.get(url)
    data = response.json()
    prices = pd.Series(data['bpi']).to_frame('close')
    prices.index = pd.to_datetime(prices.index)
except Exception as e:
    st.error(f'Error loading data: {e}')
else:
    data_loading_text.empty()

    signals_1h = generate_signals(prices.resample('1H').last())
    signals_4h = generate_signals(prices.resample('4H').last())
    signals_1d = generate_signals(prices)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f'Buy signals in {currency.split("-")[0]}: :arrow_up:', (signals_1d['buy'] == True).sum())
    with col2:
        st.write(f'Sell signals in {currency.split("-")[0]}: :arrow_down:', (signals_1d['sell'] == True).sum())

    st.subheader('Trading Targets and Stop-Loss Levels')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f'1 Hour Target: {signals_1h["target"].iloc[-1]:.2f}')
        st.write(f'1 Hour Stop-Loss: {signals_1h["stop_loss"].iloc[-1]:.2f}')
    with col2:
        st.write(f'4 Hour Target: {signals_4h["target"].iloc[-1]:.2f}')
        st.write(f'4 Hour Stop-Loss: {signals_4h["stop_loss"].iloc[-1]:.2f}')
    with col3:
        st.write(f'1 Day Target: {signals_1d["target"].iloc[-1]:.2f}')
        st.write(f'1 Day Stop-Loss: {signals_1d["stop_loss"].iloc[-1]:.2f}')

    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    csv = convert_df(pd.concat([signals_1h, signals_4h, signals_1d], keys=['1h', '4h', '1d']).dropna())
    st.download_button(
        label="Download Signals CSV", data=csv, file_name='signals.csv', mime='text/csv')

    st.subheader('Historical Prices and Trading Signals')
    if start_date == prices.index[0]:
        start_index = max(0, -30) + 0
    else:
        start_index = 0
    signals_df = pd.concat([prices, signals_1d['rsi'], signals_1d['macd'] - signals_1d['signal'],
                           signals_1d.reindex(prices.index).ffill().buy.astype(int),
                           signals_1d.reindex(prices.index).ffill().sell.astype(int)], axis=1)
    st.line_chart(data=signals_df.iloc[start_index:])

    st.subheader('Signals Adjustment Parameters')
    col1_adjust, col2_adjust = st.columns(2)

    with col1_adjust:
        rsi_period = st.number_input('RSI Period', min_value=5, max_value=100, value=14, step=1)
    with col2_adjust:
        short_period = st.number_input('MACD Short Period', min_value=2, max_value=50, value=12, step=1)
    col1_long, col2_long = st.columns(2)
    with col1_long:
        l1 = st.number_input('Long Period', min_value=short_period + 1, max_value=100, value=26)
    with col2_long:
        l0 = st.number_input('Short Period', min_value=2, max_value=50, value=12)
    col1_target, col2_target = st.columns(2)
    with col1_target:
        target_pct = st.number_input('Target Percentage', min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    with col2_target:
        stop_loss_pct = st.number_input('Stop-Loss Percentage', min_value=0.01, max_value=0.20, value=0.02, step=0.01)

    if (signals_1d['buy'].sum() + signals_1d['sell'].sum()) > 0:
        st.warning('Warning: Parameter adjustments may impact previous signals.')
        if st.button('Regenerate Trading Signals with Adjusted Parameters'):
            try:
                signals_1h = generate_signals(prices.resample('1H').last(), rsi_period=rsi_period, macd_short_period=l0,
                                              macd_long_period=l1, macd_signal_period=9, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
                signals_4h = generate_signals(prices.resample('4H').last(), rsi_period=rsi_period, macd_short_period=l0,
                                              macd_long_period=l1, macd_signal_period=9, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
                signals_1d = generate_signals(prices, rsi_period=rsi_period, macd_short_period=l0,
                                              macd_long_period=l1, macd_signal_period=9, target_pct=target_pct, stop_loss_pct=stop_loss_pct)

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f'Buy signals in {currency.split("-")[0]}: :arrow_up:', (signals_1d['buy'] == True).sum())
                with col2:
                    st.write(f'Sell signals in {currency.split("-")[0]}: :arrow_down:', (signals_1d['sell'] == True).sum())

                st.subheader('Trading Targets and Stop-Loss Levels')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f'1 Hour Target: {signals_1h["target"].iloc[-1]:.2f}')
                    st.write(f'1 Hour Stop-Loss: {signals_1h["stop_loss"].iloc[-1]:.2f}')
                with col2:
                    st.write(f'4 Hour Target: {signals_4h["target"].iloc[-1]:.2f}')
                    st.write(f'4 Hour Stop-Loss: {signals_4h["stop_loss"].iloc[-1]:.2f}')
                with col3:
                    st.write(f'1 Day Target: {signals_1d["target"].iloc[-1]:.2f}')
                    st.write(f'1 Day Stop-Loss: {signals_1d["stop_loss"].iloc[-1]:.2f}')

                if start_date == prices.index[0]:
                    start_index = max(0, -30) + 0
                else:
                    start_index = 0
                signals_df = pd.concat([prices, signals_1d['rsi'], signals_1d['macd'] - signals_1d['signal'],
                                       signals_1d.reindex(prices.index).ffill().buy.astype(int),
                                       signals_1d.reindex(prices.index).ffill().sell.astype(int)], axis=1)
                st.line_chart(data=signals_df.iloc[start_index:])
            except Exception as e:
                st.error(f'Error generating signals with adjusted parameters: {e}')

    # Automatic update
    while True:
        try:
            new_price = get_current_price(currency)
            if new_price != price:
                st.write(f'Current {currency.split("-")[0]} Price in {currency.split("-")[1]}: **{new_price}**')
                price = new_price

                signals_1h = generate_signals(prices.resample('1H').last(), rsi_period=rsi_period, macd_short_period=l0,
                                              macd_long_period=l1, macd_signal_period=9, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
                signals_4h = generate_signals(prices.resample('4H').last(), rsi_period=rsi_period, macd_short_period=l0,
                                              macd_long_period=l1, macd_signal_period=9, target_pct=target_pct, stop_loss_pct=stop_loss_pct)
                signals_1d = generate_signals(prices, rsi_period=rsi_period, macd_short_period=l0,
                                              macd_long_period=l1, macd_signal_period=9, target_pct=target_pct, stop_loss_pct=stop_loss_pct)

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f'Buy signals in {currency.split("-")[0]}: :arrow_up:', (signals_1d['buy'] == True).sum())
                with col2:
                    st.write(f'Sell signals in {currency.split("-")[0]}: :arrow_down:', (signals_1d['sell'] == True).sum())

                st.subheader('Trading Targets and Stop-Loss Levels')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f'1 Hour Target: {signals_1h["target"].iloc[-1]:.2f}')
                    st.write(f'1 Hour Stop-Loss: {signals_1h["stop_loss"].iloc[-1]:.2f}')
                with col2:
                    st.write(f'4 Hour Target: {signals_4h["target"].iloc[-1]:.2f}')
                    st.write(f'4 Hour Stop-Loss: {signals_4h["stop_loss"].iloc[-1]:.2f}')
                with col3:
                    st.write(f'1 Day Target: {signals_1d["target"].iloc[-1]:.2f}')
                    st.write(f'1 Day Stop-Loss: {signals_1d["stop_loss"].iloc[-1]:.2f}')

                if start_date == prices.index[0]:
                    start_index = max(0, -30) + 0
                else:
                    start_index = 0
                signals_df = pd.concat([prices, signals_1d['rsi'], signals_1d['macd'] - signals_1d['signal'],
                                       signals_1d.reindex(prices.index).ffill().buy.astype(int),
                                       signals_1d.reindex(prices.index).ffill().sell.astype(int)], axis=1)
                st.line_chart(data=signals_df.iloc[start_index:])

        except Exception as e:
            st.error(f'Error updating data: {e}')

        time.sleep(60) 