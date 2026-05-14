import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- Page Config ---
st.set_page_config(page_title="Project Ravi | Quant Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("🦅 Project Ravi v3.0: Complete Market Intelligence")
st.write("---")

# --- 1. Sidebar Controls ---
st.sidebar.header("Market Controls")
st.sidebar.write("Enter ANY Yahoo Finance Ticker.")
st.sidebar.caption("NSE: Add '.NS' | BSE: Add '.BO' | Indexes: ^NSEI, ^NSEBANK")

ticker = st.sidebar.text_input("Search Ticker:", "HINDCOPPER.NS").upper()

# --- 2. Data Engine ---
@st.cache_data(ttl=300) # Cache clears every 5 mins for fresh intraday data
def get_daily_data(symbol):
    df = yf.download(symbol, period="1y", progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

@st.cache_data(ttl=300)
def get_intraday_data(symbol):
    # Fetching 5-minute intervals for the last 5 days
    df = yf.download(symbol, period="5d", interval="5m", progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

@st.cache_data
def get_fundamentals(symbol):
    return yf.Ticker(symbol).info

try:
    # Load primary data
    df_daily = get_daily_data(ticker)
    info = get_fundamentals(ticker)
    
    if not df_daily.empty:
        # Top Level Header
        company_name = info.get('longName', ticker)
        current_price = df_daily['Close'].iloc[-1]
        st.header(f"{company_name} (₹{current_price:.2f})")
        
        # --- UI TABS ---
        tab1, tab2, tab3 = st.tabs(["📈 Technicals & AI", "🏢 Fundamentals", "⚡ Intraday Predictor (Indexes)"])

        # ==========================================
        # TAB 1: TECHNICALS & DAILY AI
        # ==========================================
        with tab1:
            st.subheader("Daily Technical Trend")
            df_daily['SMA50'] = df_daily['Close'].rolling(window=50).mean()
            df_daily['SMA200'] = df_daily['Close'].rolling(window=200).mean()
            
            fig1 = go.Figure()
            fig1.add_trace(go.Candlestick(x=df_daily.index, open=df_daily['Open'], high=df_daily['High'], low=df_daily['Low'], close=df_daily['Close'], name='Price'))
            fig1.add_trace(go.Scatter(x=df_daily.index, y=df_daily['SMA50'], line=dict(color='orange', width=1.5), name='50 SMA'))
            fig1.add_trace(go.Scatter(x=df_daily.index, y=df_daily['SMA200'], line=dict(color='blue', width=1.5), name='200 SMA'))
            fig1.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig1, use_container_width=True)

        # ==========================================
        # TAB 2: FUNDAMENTALS & OVERVIEW
        # ==========================================
        with tab2:
            st.subheader("Company Overview")
            st.write(info.get('longBusinessSummary', 'No description available.'))
            
            st.write("---")
            st.subheader("Key Financial Ratios")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Market Cap", f"₹{info.get('marketCap', 0) / 10000000:{',.2f'}} Cr")
            col2.metric("Trailing P/E", f"{info.get('trailingPE', 'N/A')}")
            col3.metric("52 Week High", f"₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
            col4.metric("Dividend Yield", f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A")
            
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Price to Book", f"{info.get('priceToBook', 'N/A')}")
            col6.metric("Profit Margin", f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else "N/A")
            col7.metric("Return on Equity", f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else "N/A")
            col8.metric("Debt to Equity", f"{info.get('debtToEquity', 'N/A')}")

        # ==========================================
        # TAB 3: INTRADAY PREDICTOR (5-Min Candles)
        # ==========================================
        with tab3:
            st.subheader(f"Intraday AI Prediction (5-Minute Intervals) - {ticker}")
            st.info("Best used for high-liquidity assets like ^NSEI (Nifty 50) and ^NSEBANK (Bank Nifty).")
            
            df_intra = get_intraday_data(ticker)
            
            if not df_intra.empty:
                # Prepare ML Data on Intraday timeframe
                df_intra['Time_Index'] = np.arange(len(df_intra))
                # Use the last 2 hours (approx 24 candles of 5 mins) to gauge immediate momentum
                lookback = min(24, len(df_intra)) 
                X_intra = df_intra[['Time_Index']].values[-lookback:]
                y_intra = df_intra['Close'].values[-lookback:]
                
                model_intra = LinearRegression()
                model_intra.fit(X_intra, y_intra)
                
                # Predict next 5-min candle
                next_candle_idx = np.array([[len(df_intra)]])
                pred_price = model_intra.predict(next_candle_idx)[0]
                current_intra_price = df_intra['Close'].iloc[-1]
                
                diff = pred_price - current_intra_price
                
                st.write(f"**Current Price:** ₹{current_intra_price:.2f}")
                if diff > 0:
                    st.success(f"**Next 5-Min Forecast:** UP to ₹{pred_price:.2f} (+₹{diff:.2f})")
                else:
                    st.error(f"**Next 5-Min Forecast:** DOWN to ₹{pred_price:.2f} (₹{diff:.2f})")
                
                # Plot Intraday Chart
                fig_intra = go.Figure()
                fig_intra.add_trace(go.Scatter(x=df_intra.index, y=df_intra['Close'], mode='lines', name='Intraday Price', line=dict(color='cyan')))
                
                # Add Trendline
                trend_intra = model_intra.predict(X_intra)
                fig_intra.add_trace(go.Scatter(x=df_intra.index[-lookback:], y=trend_intra, mode='lines', name='AI Short-Term Momentum', line=dict(dash='dot', color='red')))
                
                fig_intra.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_intra, use_container_width=True)
                
            else:
                st.warning("Intraday data not available for this ticker.")

    else:
        st.error("Invalid Ticker or No Data Found. Did you add '.NS' for NSE stocks?")

except Exception as e:
    st.error(f"System Error: {e}")