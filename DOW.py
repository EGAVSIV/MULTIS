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
st.set_page_config(page_title="Master Scanner - Dow Theory", layout="wide", page_icon="🟢")
st.title("🌊 Dow Theory Trend + Multi Fib Entry Scanner")

# =====================================================
# BACKGROUND
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
# DATA FOLDERS (ALL TIMEFRAMES)
# =====================================================
DATA_FOLDERS = {
    "Daily (D)": "stock_data_D",
    "Weekly (W)": "stock_data_W",
    "Monthly (M)": "stock_data_M",
    "15 Min": "stock_data_15",
    "1 Hour": "stock_data_1H"
}

# =====================================================
# TIMEFRAME SELECTION
# =====================================================
st.markdown("### ⏳ Select Timeframe Before Scanning")
selected_tf = st.selectbox("Choose Timeframe", list(DATA_FOLDERS.keys()))
DATA_PATH = DATA_FOLDERS[selected_tf]

symbols = sorted(list(set([f.stem for f in Path(DATA_PATH).glob("*.parquet")])))

# =====================================================
# DATE SELECTION
# =====================================================
def get_last_date():
    for f in Path(DATA_PATH).glob("*.parquet"):
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        return df.index.max()
    return None

last_date = get_last_date()
selected_date = st.date_input("📅 Select Scan Date", value=last_date.date())
scan_date_ts = pd.to_datetime(selected_date)

def filter_until_date(df, date):
    df.index = pd.to_datetime(df.index)
    return df[df.index <= date].copy()

# =====================================================
# DOW THEORY FUNCTIONS
# =====================================================
def detect_swings(df, order=3):
    high = df["high"].values
    low = df["low"].values
    swing_high = np.zeros(len(df), dtype=bool)
    swing_low = np.zeros(len(df), dtype=bool)

    for i in range(order, len(df) - order):
        if high[i] == max(high[i-order:i+order+1]):
            swing_high[i] = True
        if low[i] == min(low[i-order:i+order+1]):
            swing_low[i] = True

    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df

def label_structure(df):
    swings = df[(df["swing_high"]) | (df["swing_low"])].copy()
    if swings.empty:
        return swings

    swings["type"] = np.where(swings["swing_high"], "H", "L")
    swings["price"] = np.where(swings["swing_high"], swings["high"], swings["low"])
    swings["label"] = None

    last_H, last_L = None, None
    for idx in swings.index:
        row = swings.loc[idx]
        if row["type"] == "H":
            swings.at[idx, "label"] = "HH" if last_H and row["price"] > last_H else "LH"
            last_H = row["price"]
        else:
            swings.at[idx, "label"] = "HL" if last_L and row["price"] > last_L else "LL"
            last_L = row["price"]
    return swings

def classify_last_bucket(swings):
    labels = list(swings["label"].dropna())
    if len(labels) < 4:
        return "Sideways"

    last4 = labels[-4:]
    if all(l in ["HH", "HL"] for l in last4):
        return "Uptrend"
    if all(l in ["LL", "LH"] for l in last4):
        return "Downtrend"
    return "Sideways"

# =====================================================
# FIB LEVEL ENGINE
# =====================================================
FIB_RANGES = {
    "23.6%": (0.21, 0.25),
    "38.2%": (0.36, 0.40),
    "50%": (0.49, 0.52),
    "61.8%": (0.60, 0.62),
    "78.6%": (0.76, 0.78)
}

def calculate_fib_entries(swings, df, bucket):
    close = df["close"].iloc[-1]
    results = {}

    if bucket == "Uptrend":
        HL = swings[swings["label"] == "HL"].tail(1)
        HH = swings[swings["label"] == "HH"].tail(1)
        if HL.empty or HH.empty:
            return {}

        low = HL["price"].iloc[0]
        high = HH["price"].iloc[0]

        for name, (lo_per, hi_per) in FIB_RANGES.items():
            lo_level = high - (high - low) * hi_per
            hi_level = high - (high - low) * lo_per
            results[name] = lo_level <= close <= hi_level

    elif bucket == "Downtrend":
        LH = swings[swings["label"] == "LH"].tail(1)
        LL = swings[swings["label"] == "LL"].tail(1)
        if LH.empty or LL.empty:
            return {}

        high = LH["price"].iloc[0]
        low = LL["price"].iloc[0]

        for name, (lo_per, hi_per) in FIB_RANGES.items():
            lo_level = low + (high - low) * lo_per
            hi_level = low + (high - low) * hi_per
            results[name] = lo_level <= close <= hi_level

    return results

# =====================================================
# SCAN FUNCTION
# =====================================================
@st.cache_data
def run_scan(scan_date_str, symbols, DATA_PATH):
    scan_date = pd.to_datetime(scan_date_str)
    output = []

    for symbol in symbols:
        try:
            df = pd.read_parquet(f"{DATA_PATH}/{symbol}.parquet")
            df = filter_until_date(df, scan_date)
            if len(df) < 120:
                continue

            df = detect_swings(df)
            swings = label_structure(df)
            if swings.empty:
                continue

            bucket = classify_last_bucket(swings)
            fib_hits = calculate_fib_entries(swings, df, bucket)

            row = {"Stock": symbol, "Trend": bucket}
            row.update(fib_hits)
            output.append(row)

        except:
            continue

    return pd.DataFrame(output)

# =====================================================
# RUN BUTTON
# =====================================================
if st.button("🚀 Run Scanner"):
    with st.spinner("Scanning... Please wait ⏳"):
        df_result = run_scan(str(selected_date), symbols, DATA_PATH)
        st.success("Scan Completed ✅")
        st.dataframe(df_result, use_container_width=True)
