import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("📈 Stock Price Prediction App")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker ", "INFY.NS")

# Fetch stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01", auto_adjust=True)
    data = data.reset_index()
    return data

data = load_data(ticker)

# Data preprocessing
df = data.copy()
df.ffill(inplace=True)
df.dropna(inplace=True)
df["Day"] = (df["Date"] - df["Date"].min()).dt.days

# Split data
X = df[["Day"]]
y = df["Close"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"**Model Accuracy (R² Score):** {r2 * 100:.2f}%")
st.write(f"**Mean Absolute Error:** {mae}")
st.write(f"**Root Mean Squared Error:** {rmse}")

# Plot results
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X_test, y_test, color="blue", label="Actual Prices")
ax.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted Prices")
ax.set_xlabel("Days")
ax.set_ylabel("Stock Price")
ax.set_title(f"{ticker} Stock Price Prediction")
ax.legend()
st.pyplot(fig)
