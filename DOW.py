import streamlit as st
import pandas as pd
from pathlib import Path
import base64
import os

st.set_page_config(page_title="MTF Dow Theory Scanner", layout="wide", page_icon="📊")

st.title("📈 Multi-Timeframe Dow Theory + 50/61/78 Retracement Scanner")

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
# DATA FOLDERS
# =====================================================
DATA_FOLDERS = {
    "15M": "stock_data_15",
    "1H": "stock_data_1H",
    "Daily": "stock_data_D",
    "Weekly": "stock_data_W",
    "Monthly": "stock_data_M",
}

symbols = list(set([f.stem for f in Path(DATA_FOLDERS["Daily"]).glob("*.parquet")]))

# =====================================================
# UTILITY
# =====================================================
def filter_until_date(df, date):
    df.index = pd.to_datetime(df.index)
    return df[df.index <= date].copy()

# =====================================================
# SWING DETECTION
# =====================================================
def detect_swings(df, lookback=3):
    df = df.copy()

    df["swing_high"] = df["high"][
        (df["high"] > df["high"].shift(lookback)) &
        (df["high"] > df["high"].shift(-lookback))
    ]

    df["swing_low"] = df["low"][
        (df["low"] < df["low"].shift(lookback)) &
        (df["low"] < df["low"].shift(-lookback))
    ]

    swings = []

    for i in range(len(df)):
        if not pd.isna(df["swing_high"].iloc[i]):
            swings.append(("H", df["high"].iloc[i]))
        if not pd.isna(df["swing_low"].iloc[i]):
            swings.append(("L", df["low"].iloc[i]))

    return swings[-6:]

# =====================================================
# STRUCTURE CLASSIFICATION
# =====================================================
def classify_structure(swings):
    if len(swings) < 4:
        return "No Clear Trend", None

    types = [x[0] for x in swings[-4:]]
    prices = [x[1] for x in swings[-4:]]

    if types == ["H","L","H","L"]:
        if prices[2] > prices[0] and prices[3] > prices[1]:
            return "Uptrend", "bullish"
        if prices[2] > prices[0] and prices[3] < prices[1]:
            return "Reversal to Down", "bearish"

    if types == ["L","H","L","H"]:
        if prices[2] < prices[0] and prices[3] < prices[1]:
            return "Downtrend", "bearish"
        if prices[2] < prices[0] and prices[3] > prices[1]:
            return "Reversal to Up", "bullish"

    return "Triangle / Compression", None

# =====================================================
# RETRACEMENT CHECK
# =====================================================
def check_retracement(df, swings, bias, tolerance=0.01):

    if len(swings) < 2:
        return False, False, False

    last_high = None
    last_low = None

    for s in reversed(swings):
        if s[0] == "H" and last_high is None:
            last_high = s[1]
        if s[0] == "L" and last_low is None:
            last_low = s[1]
        if last_high and last_low:
            break

    if not last_high or not last_low:
        return False, False, False

    current = df["close"].iloc[-1]

    move = last_high - last_low

    if bias == "bullish":
        fib50 = last_high - 0.50 * move
        fib61 = last_high - 0.618 * move
        fib78 = last_high - 0.786 * move

    elif bias == "bearish":
        fib50 = last_low + 0.50 * move
        fib61 = last_low + 0.618 * move
        fib78 = last_low + 0.786 * move
    else:
        return False, False, False

    hit50 = abs(current - fib50)/fib50 <= tolerance
    hit61 = abs(current - fib61)/fib61 <= tolerance
    hit78 = abs(current - fib78)/fib78 <= tolerance

    return hit50, hit61, hit78

# =====================================================
# SCAN ENGINE
# =====================================================
@st.cache_data
def run_scan(scan_date):

    results = []

    for symbol in symbols:
        try:
            row = {"Stock": symbol}

            for tf, folder in DATA_FOLDERS.items():

                df = pd.read_parquet(f"{folder}/{symbol}.parquet")
                df = filter_until_date(df, scan_date)

                if len(df) < 50:
                    row[f"{tf} Trend"] = "No Data"
                    continue

                swings = detect_swings(df)
                trend, bias = classify_structure(swings)

                row[f"{tf} Trend"] = trend

                hit50, hit61, hit78 = check_retracement(df, swings, bias)

                row[f"{tf} 50%"] = hit50
                row[f"{tf} 61%"] = hit61
                row[f"{tf} 78%"] = hit78

            results.append(row)

        except:
            continue

    return pd.DataFrame(results)

# =====================================================
# DATE SELECTOR
# =====================================================
sample_file = list(Path(DATA_FOLDERS["Daily"]).glob("*.parquet"))[0]
df_temp = pd.read_parquet(sample_file)
df_temp.index = pd.to_datetime(df_temp.index)
last_date = df_temp.index.max()

selected_date = st.date_input("📅 Select Scan Date", value=last_date.date())
selected_date = pd.to_datetime(selected_date)

# =====================================================
# RUN
# =====================================================
if st.button("🚀 Run Multi-TF Dow Scan"):

    df_result = run_scan(selected_date)

    st.success("Scan Completed ✅")

    trend_filter = st.selectbox(
        "Filter Trend (Daily)",
        ["All", "Uptrend", "Downtrend", "Reversal to Down", "Reversal to Up", "Triangle / Compression"]
    )

    if trend_filter != "All":
        df_result = df_result[df_result["Daily Trend"] == trend_filter]

    st.dataframe(df_result, use_container_width=True)
