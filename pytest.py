import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Function to get current Bitcoin price
def get_current_bitcoin_price(currency='USD'):
    url = f'https://api.coindesk.com/v1/bpi/currentprice/{currency}.json'
    response = requests.get(url)
    data = response.json()
    return data['bpi'][currency]['rate_float']

# Function to get historical Bitcoin price data
def get_historical_bitcoin_prices(currency='USD', days=30):
    url = f'https://api.coindesk.com/v1/bpi/historical/close.json?currency={currency}&start={datetime.now().strftime("%Y-%m-%d")}&end={datetime.now().strftime("%Y-%m-%d")}'
    response = requests.get(url)
    data = response.json()
    if 'bpi' in data:
        historical_prices = pd.Series(data['bpi']).sort_index()
        df = pd.DataFrame({'price': historical_prices.values, 'date': historical_prices.index})
        return df.tail(days)
    else:
        return pd.DataFrame()

# Custom Bitcoin Trading Environment
class BitcoinTradingEnv:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.current_index = 0
        self.portfolio_value = 1000
        self.num_shares = 0

    def step(self, action):
        current_price = self.historical_data.iloc[self.current_index]['price']
        reward = 0
        done = False

        if action == 0:  # Hold
            pass
        elif action == 1:  # Buy
            shares_to_buy = self.portfolio_value // current_price
            self.portfolio_value -= shares_to_buy * current_price
            self.num_shares += shares_to_buy
        else:  # Sell
            self.portfolio_value += self.num_shares * current_price
            self.num_shares = 0

        self.current_index += 1
        if self.current_index >= len(self.historical_data):
            done = True

        observation = [self.portfolio_value, self.num_shares, current_price]
        return np.array(observation, dtype=np.float32), reward, done, {}

    def reset(self):
        self.current_index = 0
        self.portfolio_value = 1000
        self.num_shares = 0
        return np.array([self.portfolio_value, self.num_shares, self.historical_data.iloc[0]['price']], dtype=np.float32)


# Tabular Regression
def tabular_regression(historical_data):
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    X = (historical_data['date'] - historical_data['date'].min()).dt.total_seconds().to_numpy().reshape(-1, 1)
    y = historical_data['price'].values
    model = LinearRegression()
    model.fit(X, y)
    future_dates = pd.date_range(start=historical_data['date'].max(), periods=7, freq='D')
    future_seconds = (future_dates - historical_data['date'].min()).total_seconds().to_numpy().reshape(-1, 1)
    future_prices = model.predict(future_seconds)
    return [{'date': str(date), 'price': price} for date, price in zip(future_dates, future_prices)]
    
# Page header
st.header('Bitcoin Price Tracker :chart_with_upwards_trend:')

# Display current time in Tehran
url = 'http://worldtimeapi.org/api/timezone/Asia/Tehran'
response = requests.get(url)
tehran_time = response.json()['datetime']
time_format = '%Y-%m-%dT%H:%M:%S.%f%z'
tehran_time_formatted = datetime.strptime(tehran_time, time_format).strftime('%Y-%m-%d %H:%M:%S %Z')
time_options = ['Tehran', 'UTC', 'Local']
selected_time = st.sidebar.selectbox('Select Time Zone', time_options)
if selected_time == 'Tehran':
    st.sidebar.write(f'Current time in Tehran: {tehran_time_formatted}')
elif selected_time == 'UTC':
    st.sidebar.write(f'Current time in UTC: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}')
else:
    st.sidebar.write(f'Current local time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")}')

# Display current Bitcoin price
currency = st.sidebar.selectbox('Select Currency', ['USD', 'EUR', 'GBP'])
current_price = get_current_bitcoin_price(currency)
st.write(f'Current Bitcoin Price in {currency}: **{current_price}**')

# Refresh button
if st.button('Refresh'):
    st.experimental_rerun()

# Tabular Regression section
st.subheader('Tabular Regression')
historical_df = get_historical_bitcoin_prices(currency, days=30)
future_prices = tabular_regression(historical_df)
st.write('Predicted Future Bitcoin Prices:')
for data in future_prices:
    st.write(f"Date: {data['date']}, Price: {data['price']}")

# Custom Bitcoin Trading Environment section
st.subheader('Custom Bitcoin Trading Environment')
env = BitcoinTradingEnv(historical_df)
obs = env.reset()
done = False
while not done:
    action = np.random.randint(0, 3)
    obs, reward, done, _ = env.step(action)
    st.write(f"Portfolio Value: {obs[0]}, Shares: {obs[1]}, Price: {obs[2]}")

# Copyright notice
st.write('Copyright © 2023 Taha Tehrani Nasab. All rights reserved.')
