import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Project Ravi | Financial Intelligence", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🦅 Project Ravi: Market Analysis Engine")
st.subheader("Automated Technical Intelligence & Signal Generation")
st.write("---")

# --- Sidebar Inputs ---
st.sidebar.header("Control Panel")
ticker = st.sidebar.text_input("Enter Ticker (e.g., HINDCOPPER.NS, BEL.NS, BTC-USD)", "HINDCOPPER.NS")
timeframe = st.sidebar.selectbox("Select Timeframe", ("1y", "2y", "5y", "max"))

# --- Data Engine ---
@st.cache_data
def load_data(symbol, period):
    data = yf.download(symbol, period=period)
    # Ensure column names are flat (yfinance sometimes returns multi-index)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    return data

try:
    # 1. Fetch Data
    df = load_data(ticker, timeframe)
    stock_info = yf.Ticker(ticker).info
    
    if not df.empty:
        # 2. Key Metrics Row
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        price_diff = current_price - prev_close
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"₹{current_price:.2f}", f"{price_diff:.2f}")
        col2.metric("Sector", stock_info.get('sector', 'N/A'))
        col3.metric("52 Week High", f"₹{stock_info.get('fiftyTwoWeekHigh', 0):.2f}")
        col4.metric("Market Cap", f"{stock_info.get('marketCap', 0):,}")

        # 3. Technical Logic (SMA Crossover)
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        last_sma50 = df['SMA50'].iloc[-1]
        last_sma200 = df['SMA200'].iloc[-1]

        # 4. Signal Logic
        st.write("### AI Recommendation Signal")
        if last_sma50 > last_sma200:
            st.success(f"**SIGNAL: BUY (BULLISH)** - The 50-day SMA (₹{last_sma50:.2f}) is above the 200-day SMA (₹{last_sma200:.2f}). This indicates a strong upward trend.")
        elif last_sma50 < last_sma200:
            st.error(f"**SIGNAL: SELL (BEARISH)** - The 50-day SMA (₹{last_sma50:.2f}) has dropped below the 200-day SMA (₹{last_sma200:.2f}). This indicates a downward trend.")
        else:
            st.warning("**SIGNAL: NEUTRAL** - No clear trend detected. High volatility expected.")

        # 5. Interactive Technical Chart
        st.write("---")
        st.write("### Technical Analysis Chart")
        
        fig = go.Figure()
        # Candlestick chart
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'], name='Price'))
        
        # Moving Averages
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='orange', width=1.5), name='50 Day SMA'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='blue', width=1.5), name='200 Day SMA'))
        
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

        # 6. Raw Data Insights
        with st.expander("View Raw Intelligence Data"):
            st.dataframe(df.tail(10))

    else:
        st.warning("No data found. Please ensure the ticker suffix is correct (e.g., .NS for National Stock Exchange).")

except Exception as e:
    st.error(f"Engine Error: {e}")