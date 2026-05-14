import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- Page Config ---
st.set_page_config(page_title="Project Ravi | AI Prediction", layout="wide")

st.title("🦅 Project Ravi: AI Market Predictor")
st.write("---")

# --- 1. Stock Selection List ---
st.sidebar.header("Market Controls")
stock_dict = {
    "Hindustan Copper": "HINDCOPPER.NS",
    "Bharat Electronics (BEL)": "BEL.NS",
    "MTAR Technologies": "MTARTECH.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "NIFTY 50 Index": "^NSEI"
}
selected_stock_name = st.sidebar.selectbox("Select a Stock", list(stock_dict.keys()))
ticker = stock_dict[selected_stock_name]

# Allow custom input if they want something else
custom_ticker = st.sidebar.text_input("Or type custom ticker (e.g., TCS.NS)", "")
if custom_ticker:
    ticker = custom_ticker

# --- 2. Data Engine ---
@st.cache_data
def get_data(symbol):
    data = yf.download(symbol, period="1y")
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    return data

try:
    df = get_data(ticker)
    
    if not df.empty:
        # Display Basic Metrics
        current_price = df['Close'].iloc[-1]
        st.metric(label=f"Current Price: {selected_stock_name}", value=f"₹{current_price:.2f}")

        # --- 3. THE PREDICTION ENGINE (Data Science Part) ---
        st.subheader("🤖 AI Price Prediction (Next 24 Hours)")
        
        # Prepare data for Linear Regression
        df['Day_Num'] = np.arange(len(df))
        X = df[['Day_Num']].values[-30:] # Use last 30 days to find trend
        y = df['Close'].values[-30:]
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict Tomorrow
        tomorrow_day_num = np.array([[len(df)]])
        predicted_price = model.predict(tomorrow_day_num)[0]
        
        # Display Prediction
        change = predicted_price - current_price
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Predicted Closing Price for Tomorrow:**")
            st.title(f"₹{predicted_price:.2f}")
        with col2:
            if change > 0:
                st.success(f"Forecast: UP (+₹{change:.2f})")
            else:
                st.error(f"Forecast: DOWN (₹{abs(change):.2f})")

        # --- 4. Technical Chart ---
        st.write("---")
        st.subheader("Technical Trend Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual Price'))
        
        # Adding the "Trend Line" to the chart
        full_X = df[['Day_Num']].values
        trend_line = model.predict(full_X)
        fig.add_trace(go.Scatter(x=df.index, y=trend_line, name='AI Trend Line', line=dict(dash='dot', color='red')))
        
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Invalid Ticker. Please check the symbol.")

except Exception as e:
    st.error(f"Engine Error: {e}")