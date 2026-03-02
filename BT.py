import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(layout="wide")

# ==============================
# MACD CALCULATION
# ==============================
def calculate_macd(df, fast=12, slow=26, signal=9):
    df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    return df


# ==============================
# BURST DETECTOR (3-4 candles)
# ==============================
def detect_burst(df, percent=3, min_candles=3):

    df = df.copy()
    df['pct_move'] = (df['close'] - df['open']) / df['open'] * 100

    results = []

    for i in range(len(df) - min_candles):

        window = df.iloc[i:i+min_candles]

        if all(window['pct_move'] >= percent):
            results.append(("Bullish", df.index[i]))

        if all(window['pct_move'] <= -percent):
            results.append(("Bearish", df.index[i]))

    return results


# ==============================
# HTF MACD STATUS
# ==============================
def get_macd_status(htf_df, timestamp):

    df = htf_df[htf_df.index <= timestamp]

    if len(df) < 2:
        return None

    row = df.iloc[-1]
    prev = df.iloc[-2]

    status = {
        "MACD_Zone": "Above 0" if row['MACD'] > 0 else "Below 0",
        "MACD_vs_Signal": "Above" if row['MACD'] > row['Signal'] else "Below",
        "Direction": "Uptick" if row['MACD'] > prev['MACD'] else "Downtick",
        "Crossover":
            "Bullish Cross" if prev['MACD'] < prev['Signal'] and row['MACD'] > row['Signal']
            else "Bearish Cross" if prev['MACD'] > prev['Signal'] and row['MACD'] < row['Signal']
            else "No Cross"
    }

    return status


# ==============================
# POST BURST PERFORMANCE
# ==============================
def post_performance(df, timestamp, candles_forward=5):

    if timestamp not in df.index:
        return None

    idx = df.index.get_loc(timestamp)

    if idx + candles_forward >= len(df):
        return None

    entry_price = df.iloc[idx]['close']
    exit_price = df.iloc[idx + candles_forward]['close']

    return round((exit_price - entry_price) / entry_price * 100, 2)


# ==============================
# STREAMLIT UI
# ==============================
st.title("📊 Multi-Timeframe MACD Burst Analyzer")

# Sidebar controls
percent = st.sidebar.number_input("Minimum Candle % Move", value=3.0)
min_candles = st.sidebar.selectbox("Consecutive Candles", [3,4])
htf_choice = st.sidebar.selectbox("Higher Timeframe", ["Daily","Weekly","Monthly"])
forward_candles = st.sidebar.number_input("Forward Candles for Performance", value=5)

# Folder mapping
folder_map = {
    "Daily": "stock_data_D",
    "Weekly": "stock_data_W",
    "Monthly": "stock_data_M"
}

stock_list = os.listdir("stock_data_1H")
stock = st.selectbox("Select Stock", stock_list)

if stock:

    try:
        df_1h = pd.read_parquet(f"stock_data_1H/{stock}")
        df_htf = pd.read_parquet(f"{folder_map[htf_choice]}/{stock}")

        df_1h.index = pd.to_datetime(df_1h.index)
        df_htf.index = pd.to_datetime(df_htf.index)

        df_htf = calculate_macd(df_htf)

        bursts = detect_burst(df_1h, percent, min_candles)

        results = []

        for move_type, timestamp in bursts:

            status = get_macd_status(df_htf, timestamp)
            perf = post_performance(df_1h, timestamp, forward_candles)

            if status:
                row = {
                    "Time": timestamp,
                    "Move": move_type,
                    "MACD Zone": status["MACD_Zone"],
                    "MACD vs Signal": status["MACD_vs_Signal"],
                    "Direction": status["Direction"],
                    "Crossover": status["Crossover"],
                    f"{forward_candles} Candle Return %": perf
                }
                results.append(row)

        if results:
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True)

            st.subheader("📈 Probability Summary")

            summary = result_df.groupby(["Move","MACD Zone","Direction"]) \
                [f"{forward_candles} Candle Return %"] \
                .mean().reset_index()

            st.dataframe(summary, use_container_width=True)

        else:
            st.warning("No Burst Found.")

    except Exception as e:
        st.error(f"Error: {e}")
