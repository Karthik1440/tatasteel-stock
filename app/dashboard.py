# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

# ----------------- APP TITLE -----------------
st.title("📈 Tata Steel Stock Analysis Dashboard")

# ----------------- PATH SETUP -----------------
# Get the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjust paths relative to this script
data_path = os.path.join(BASE_DIR, "..", "data", "TATASTEEL.csv")
model_path = os.path.join(BASE_DIR, "..", "model", "model.pkl")

st.write("📂 Looking for data at:", data_path)
st.write("📦 Looking for model at:", model_path)

# ----------------- LOAD DATA -----------------
try:
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    st.success("✅ Data loaded successfully!")
except Exception as e:
    st.error(f"❌ Data loading error: {e}")
    st.stop()

# ----------------- LOAD MODEL -----------------
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.stop()

# ----------------- VISUALIZATION -----------------
st.subheader("Closing Price (2000–2021)")
fig1 = plt.figure(figsize=(10,5))
plt.plot(data['Close'], color='blue')
plt.xlabel("Year")
plt.ylabel("Price (₹)")
plt.grid(True)
st.pyplot(fig1)

# Moving averages
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

st.subheader("20-Day & 50-Day Moving Averages")
fig2 = plt.figure(figsize=(10,5))
plt.plot(data['Close'], label="Close", color='blue')
plt.plot(data['MA20'], label="MA20", color='orange')
plt.plot(data['MA50'], label="MA50", color='green')
plt.xlabel("Year")
plt.ylabel("Price (₹)")
plt.legend()
plt.grid(True)
st.pyplot(fig2)

# ----------------- NEXT DAY PREDICTION -----------------
st.subheader("Next Day Price Prediction")

# Take last 60 days for prediction
latest = data[['Close', 'MA20', 'MA50']].tail(60)
latest = latest.dropna()

# Debug: show latest row
st.write("Latest row used for prediction:")
st.write(latest.tail(1))

if not latest.empty:
    last_row = latest[['Close', 'MA20', 'MA50']].tail(1)
    try:
        pred = model.predict(last_row)[0]
        st.success(f"Predicted Next Closing Price: ₹{round(pred, 2)}")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
else:
    st.warning("Not enough data for prediction.")
