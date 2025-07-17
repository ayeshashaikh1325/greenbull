import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import warnings
from streamlit_extras.let_it_rain import rain
from textblob import TextBlob
import requests


warnings.filterwarnings("ignore", message="missing ScriptRunContext!")

# Load the trained model
model = load_model('final_best_stock_model.h5')

# Streamlit UI
st.set_page_config(page_title="Stock Prediction", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stTextInput, .stDateInput, .stButton {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Stock Price Prediction")
st.sidebar.header("Enter Stock Details")

# User Inputs
stock_symbol = st.sidebar.text_input("Stock Ticker", "AAPL")  # Default AAPL
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2012-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2022-01-01"))

# Fetch stock data
if st.sidebar.button("Predict"):
    st.subheader(f"Stock Price Prediction for {stock_symbol}")
    df = yf.download(stock_symbol, start=start_date, end=end_date)

    # Moving Averages
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # Preprocess Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])

    x_test, y_test = [], []
    for i in range(100, len(scaled_data)):
        x_test.append(scaled_data[i - 100:i])
        y_test.append(scaled_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot Actual vs Predicted Prices
    st.subheader("📊 Actual vs Predicted Prices")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=y_test.flatten(), mode='lines', name='Actual Prices'))
    fig1.add_trace(go.Scatter(y=predictions.flatten(), mode='lines', name='Predicted Prices', line=dict(dash='dot')))
    fig1.update_layout(title=f"{stock_symbol} Stock Price Prediction", xaxis_title="Time",
                       yaxis_title="Stock Price ($)", template="plotly_dark")
    st.plotly_chart(fig1)
    #data table
    st.subheader("📄 Stock Data Overview")
    st.dataframe(df.tail(10))  # Show the last 10 rows

    # Plot 100-Day Moving Average
    st.subheader("📈 100-Day Moving Average")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=df['MA100'], mode='lines', name='100-Day MA', line=dict(color='yellow')))
    fig2.update_layout(title=f"{stock_symbol} 100-Day Moving Average", xaxis_title="Time",
                       yaxis_title="Stock Price ($)", template="plotly_dark")
    st.plotly_chart(fig2)

    # Plot 200-Day Moving Average
    st.subheader("📉 200-Day Moving Average")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=df['MA200'], mode='lines', name='200-Day MA', line=dict(color='red')))
    fig3.update_layout(title=f"{stock_symbol} 200-Day Moving Average", xaxis_title="Time",
                       yaxis_title="Stock Price ($)", template="plotly_dark")
    st.plotly_chart(fig3)

    # Sentiment Analysis
    st.subheader("📰 News & Sentiment Analysis")
    news_url = f"https://finance.yahoo.com/quote/{stock_symbol}/news?p={stock_symbol}"
    st.write(f"[Check latest news here]({news_url})")


    def analyze_sentiment(text):
        sentiment = TextBlob(text).sentiment.polarity
        return "🔴 Bearish" if sentiment < 0 else "🟢 Bullish"


    headlines = ["Stock hits new high!", "Market faces crash fears",
                 "Tech stocks rally"]  # Replace with actual news fetch logic
    for headline in headlines:
        st.write(f"🗞 {headline} - {analyze_sentiment(headline)}")



    # Error Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import math

    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    accuracy = 100 - mape

    st.subheader("📉 Model Performance Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    st.write(f"📊 Model Accuracy: {accuracy:.2f}%")

    # Add animated rain effect
    rain(emoji="💰", font_size=20, falling_speed=5, animation_length=5)

    # Model Summary
    import io
    from contextlib import redirect_stdout

    st.subheader("📑 Model Summary")

    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        model.summary()
    summary_string = summary_buffer.getvalue()
    st.text(summary_string)