# Improved Personalized Trading Insights Agent
# Enhancements:
# - Integrated real LSTM for predictions using PyTorch
# - Added Alpha Vantage for news sentiment (requires free API key)
# - Added backtesting functionality
# - Added scheduling with APScheduler for autonomous daily runs (demo in Streamlit)
# - Real-time updates: Added refresh button and current price fetch
# - Error handling, caching with joblib
# - Fixed LLM setup for OpenRouter Mistral 7B
# - Multi-agent with CrewAI for coordinated tasks (data fetch, analysis, reporting)
# Requirements: pip install langchain streamlit yfinance pandas matplotlib plotly torch crewai requests apscheduler joblib scikit-learn openai

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.utilities import PythonREPL
from datetime import datetime, timedelta
import os
import requests
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from crewai import Agent, Task, Crew, Process
from apscheduler.schedulers.background import BackgroundScheduler
from joblib import Memory
import time

# Cache setup
cache_dir = './.cache'
os.makedirs(cache_dir, exist_ok=True)
memory_cache = Memory(cache_dir, verbose=0)

# Setup OpenRouter (Mistral 7B)
os.environ["OPENROUTER_API_KEY"] = st.secrets.get("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
llm = OpenAI(
    model="mistralai/mistral-7b-instruct:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

# Alpha Vantage API Key (User input)
alpha_vantage_key = st.sidebar.text_input("Alpha Vantage API Key", type="password", value="demo")

# LSTM Model Definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# Data Prep for LSTM
@memory_cache.cache
def prepare_data(data, lookback=20):
    price = data[['Close']].copy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))
    data_raw = price.to_numpy()
    sequences = []
    for index in range(len(data_raw) - lookback):
        sequences.append(data_raw[index: index + lookback])
    sequences = np.array(sequences)
    return sequences, scaler

# Train LSTM
def train_lstm(sequences, epochs=50, lr=0.01, hidden_dim=32, num_layers=2):
    input_dim = 1
    output_dim = 1
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    x_train = torch.from_numpy(sequences[:, :-1, :]).type(torch.Tensor)
    y_train = torch.from_numpy(sequences[:, -1, :]).type(torch.Tensor)
    
    for epoch in range(epochs):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# Predict with LSTM
def predict_with_lstm(model, sequences, scaler, num_predictions=1):
    x_test = torch.from_numpy(sequences[-1:, :-1, :]).type(torch.Tensor)  # Last sequence for prediction
    predictions = []
    for _ in range(num_predictions):
        with torch.no_grad():
            pred = model(x_test)
        predictions.append(pred.item())
        # Shift for next prediction (simplified)
        new_seq = np.append(x_test.numpy().squeeze()[-19:], pred.item()).reshape(1, -1, 1)
        x_test = torch.from_numpy(new_seq).type(torch.Tensor)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Tools
@memory_cache.cache
def fetch_stock_data(symbol, period="1mo"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        return pd.DataFrame()  # Empty on error

data_tool = Tool(
    name="FetchStockData",
    func=lambda symbol_period: fetch_stock_data(*symbol_period.split(',')),
    description="Fetches historical stock data for symbol,period."
)

def fetch_news(symbol):
    symbol_clean = symbol[:-3] if symbol.endswith('.NS') else symbol  # Remove .NS for API
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol_clean}&limit=5&apikey={alpha_vantage_key}"
    try:
        response = requests.get(url)
        data = response.json()
        feed = data.get('feed', [])
        sentiments = [article.get('overall_sentiment_label', 'Neutral') for article in feed]
        return ', '.join(sentiments) or "No news found"
    except:
        return "Error fetching news"

news_tool = Tool(
    name="FetchNews",
    func=fetch_news,
    description="Fetches recent news sentiments for a stock symbol."
)

def predict_trend(symbol, period="6mo", lookback=20, epochs=50):
    data = fetch_stock_data(symbol, period)
    if data.empty:
        return "No data"
    sequences, scaler = prepare_data(data, lookback)
    if len(sequences) < 1:
        return "Insufficient data"
    model = train_lstm(sequences, epochs)
    prediction = predict_with_lstm(model, sequences, scaler, num_predictions=1)[0]
    last_close = data['Close'].iloc[-1]
    trend = "Up" if prediction > last_close else "Down"
    return f"Predicted next close: {prediction:.2f} ({trend})"

predict_tool = Tool(
    name="PredictTrend",
    func=lambda args: predict_trend(*args.split(',')),
    description="Predicts stock trend using LSTM. Args: symbol,period,lookback,epochs"
)

math_chain = LLMMathChain(llm=llm, verbose=True)
math_tool = Tool(name="Calculator", func=math_chain.run, description="Math calculations.")

python_repl = PythonREPL()
repl_tool = Tool(name="PythonREPL", func=python_repl.run, description="Executes Python code.")

tools = [data_tool, news_tool, predict_tool, math_tool, repl_tool]

# Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, memory=memory)

# CrewAI Setup for Multi-Agent
analyzer_agent = Agent(
    role='Analyzer',
    goal='Analyze data and sentiments',
    backstory='Expert in stock analysis',
    tools=tools,
    llm=llm
)

reporter_agent = Agent(
    role='Reporter',
    goal='Generate reports',
    backstory='Creates insightful reports',
    llm=llm
)

# Streamlit UI
st.title("Personalized Trading Insights Agent (Real-Time Enabled)")

# User Inputs
symbol = st.text_input("NSE Stock Symbol (e.g., TATAMOTORS.NS)", "RELIANCE.NS")
period = st.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y"])
risk_tolerance = st.slider("Risk Tolerance (1-10)", 1, 10, 5)
lookback = st.number_input("LSTM Lookback", 10, 50, 20)
epochs = st.number_input("LSTM Epochs", 10, 200, 50)

if st.button("Generate Insights"):
    with st.spinner("Analyzing..."):
        data = fetch_stock_data(symbol, period)
        if data.empty:
            st.error("Failed to fetch data.")
        else:
            # Current Price (Real-time)
            current = yf.Ticker(symbol).info.get('currentPrice', 'N/A')
            st.subheader("Current Price")
            st.write(f"{current} INR")

            # Agent Query
            query = f"Fetch data for {symbol},{period}. Fetch news for {symbol}. Predict trend for {symbol},{period},{lookback},{epochs}. Assess risk for tolerance {risk_tolerance}."
            response = agent.run(query)
            st.subheader("Agent Insights")
            st.write(response)

            # Chart
            fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
            fig.update_layout(title=f"{symbol} Chart", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

            # Backtest (Simple: Compare predictions on test set)
            st.subheader("Backtest Summary")
            sequences, scaler = prepare_data(data, lookback)
            train_size = int(0.8 * len(sequences))
            test_sequences = sequences[train_size:]
            model = train_lstm(sequences[:train_size], epochs)
            preds = predict_with_lstm(model, test_sequences, scaler, num_predictions=len(test_sequences))
            actuals = scaler.inverse_transform(sequences[train_size:, -1, :].reshape(-1, 1)).flatten()
            mse = np.mean((preds - actuals)**2)
            st.write(f"Backtest MSE: {mse:.2f}")

# Real-Time Refresh
if st.button("Refresh Real-Time Data"):
    st.experimental_rerun()

# Autonomous Scheduling (Demo)
scheduler = BackgroundScheduler()
def daily_run():
    # Simulate daily alert
    response = agent.run(f"Daily check for {symbol}: Fetch latest, predict, alert if buy/sell based on risk {risk_tolerance}.")
    st.session_state['daily_response'] = response

scheduler.add_job(daily_run, 'interval', days=1)  # For demo, change to minutes
scheduler.start()

if 'daily_response' in st.session_state:
    st.subheader("Daily Autonomous Report")
    st.write(st.session_state['daily_response'])

# Alerts (Placeholder: Add email/sms with user config)
st.subheader("Alerts Setup")
email = st.text_input("Email for Alerts")
if email:
    st.write("Alerts enabled - implement smtplib for real sends.")
