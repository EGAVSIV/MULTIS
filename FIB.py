import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from pathlib import Path
from time import sleep

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Master Scanner - Fibonacci Only", layout="wide", page_icon="🌀")

st.title("📐 Fibonacci Retracement Scanner")

# =====================================================
# BACKGROUND IMAGE
# =====================================================
def set_bg_image(image_path: str):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

BASE_PATH = os.path.dirname(__file__)
bg_path = os.path.join(BASE_PATH, "Assest", "BG11.png")

if os.path.exists(bg_path):
    set_bg_image(bg_path)

# =====================================================
# DATA FOLDERS
# =====================================================
DATA_D   = "stock_data_D"
DATA_W   = "stock_data_W"
DATA_M   = "stock_data_M"
DATA_15M = "stock_data_15"
DATA_1H  = "stock_data_1H"

TIMEFRAME_MAP = {
    "15 Min": DATA_15M,
    "1 Hour": DATA_1H,
    "Daily": DATA_D,
    "Weekly": DATA_W,
    "Monthly": DATA_M,
}

# =====================================================
# SETTINGS SECTION
# =====================================================
st.markdown("### ⚙️ Scan Settings")

col1, col2, col3 = st.columns(3)

# ---- Timeframe ----
with col1:
    timeframe = st.selectbox("⏱ Timeframe", list(TIMEFRAME_MAP.keys()))
    data_folder = TIMEFRAME_MAP[timeframe]

# ---- Lookback Bars ----
with col2:
    lookback = st.number_input("🔢 Lookback Bars", min_value=10, max_value=500, value=50)

# ---- Fib Zone ----
with col3:
    fib_option = st.selectbox(
        "📐 Fib Zone",
        ["61-78", "58-61", "35-50"]
    )

# =====================================================
# GET MAX DATE FROM FOLDER
# =====================================================
def get_last_date(folder):
    folder_path = Path(folder)
    max_date = None

    for f in folder_path.glob("*.parquet"):
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        d = df.index.max()
        if max_date is None or d > max_date:
            max_date = d

    return max_date

max_date_available = get_last_date(data_folder)

if max_date_available is None:
    st.error("No data available in selected timeframe.")
    st.stop()

selected_date = st.date_input(
    "📅 Select Scan Date",
    value=max_date_available.date(),
    max_value=max_date_available.date()
)

scan_date = pd.to_datetime(selected_date)

# =====================================================
# BUILD SYMBOL LIST
# =====================================================
symbols = sorted([f.stem for f in Path(data_folder).glob("*.parquet")])

# =====================================================
# FIB CALCULATION FUNCTION
# =====================================================
def calculate_fib_levels(df, lookback, fib_option):
    df = df.tail(lookback)

    high_price = df["high"].max()
    low_price = df["low"].min()

    if high_price == low_price:
        return None

    diff = high_price - low_price

    if fib_option == "61-78":
        level1 = high_price - diff * 0.61
        level2 = high_price - diff * 0.78

    elif fib_option == "58-61":
        level1 = high_price - diff * 0.58
        level2 = high_price - diff * 0.61

    elif fib_option == "35-50":
        level1 = high_price - diff * 0.35
        level2 = high_price - diff * 0.50

    else:
        return None

    return min(level1, level2), max(level1, level2)

# =====================================================
# SCAN FUNCTION
# =====================================================
@st.cache_data(show_spinner=False)
def run_scan(scan_date_str, symbols, folder, lookback, fib_option):

    scan_date = pd.to_datetime(scan_date_str)
    results = []

    for symbol in symbols:
        try:
            df = pd.read_parquet(f"{folder}/{symbol}.parquet")
            df.index = pd.to_datetime(df.index)
            df = df[df.index <= scan_date]

            if len(df) < lookback:
                continue

            levels = calculate_fib_levels(df, lookback, fib_option)
            if levels is None:
                continue

            zone_low, zone_high = levels
            current_close = df["close"].iloc[-1]

            if zone_low <= current_close <= zone_high:
                results.append({
                    "Stock": symbol,
                    "Close": round(current_close, 2),
                    "Zone Low": round(zone_low, 2),
                    "Zone High": round(zone_high, 2),
                })

        except Exception:
            continue

    return pd.DataFrame(results)

# =====================================================
# RUN BUTTON
# =====================================================
if st.button("🚀 Run Fibonacci Scan"):

    with st.spinner("Scanning stocks..."):
        df_result = run_scan(
            str(scan_date),
            symbols,
            data_folder,
            lookback,
            fib_option
        )
        sleep(0.5)

    if df_result.empty:
        st.warning("No stocks found in selected Fibonacci zone.")
    else:
        st.success(f"Found {len(df_result)} Stocks in Fib Zone ✅")
        st.dataframe(df_result, use_container_width=True)
