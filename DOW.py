import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Master Scanner - Dow Theory", layout="wide", page_icon="🟢")

st.title("🌊 Dow Theory Trend + 61% Fib Entry Scanner")

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
else:
    st.warning(f"Background not found at: {bg_path}")

# =====================================================
# DATA FOLDERS (YOU CAN ADD 15m, 1H LATER)
# =====================================================
DATA_D = "stock_data_D"
DATA_W = "stock_data_W"
DATA_M = "stock_data_M"

symbols = list(set([f.stem for f in Path(DATA_D).glob("*.parquet")]))

# =====================================================
# BASIC HELPERS
# =====================================================
def get_last_daily_date():
    for f in Path(DATA_D).glob("*.parquet"):
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        return df.index.max()
    return None

last_daily_date = get_last_daily_date()

st.markdown("### 🕯 Lastest Candle")
if last_daily_date is not None:
    st.markdown(f"📅 **Daily: {last_daily_date.date()}**")
else:
    st.markdown("No daily data found.")

selected_date = st.date_input(
    "📅 Select Scan Date",
    value=last_daily_date.date() if last_daily_date is not None else pd.to_datetime("today").date()
)
selected_date = pd.to_datetime(selected_date)


def filter_until_date(df, date):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df[df.index <= date].copy()

# =====================================================
# DOW THEORY SWING + TREND
# =====================================================
def detect_swings(df, order=3):
    """
    Detect swing highs / lows with a simple window method.
    order = bars left/right to check.
    """
    high = df["high"].values
    low = df["low"].values

    swing_high = np.zeros(len(df), dtype=bool)
    swing_low = np.zeros(len(df), dtype=bool)

    for i in range(order, len(df) - order):
        if high[i] == max(high[i - order:i + order + 1]):
            swing_high[i] = True
        if low[i] == min(low[i - order:i + order + 1]):
            swing_low[i] = True

    out = df.copy()
    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    return out


def label_structure(df):
    """
    From swing points, create sequence of HH, HL, LH, LL.
    """
    swings = df[(df["swing_high"]) | (df["swing_low"])].copy()
    if swings.empty:
        return swings

    swings["type"] = np.where(swings["swing_high"], "H", "L")
    swings["price"] = np.where(swings["swing_high"], swings["high"], swings["low"])
    swings["label"] = None

    last_H = None
    last_L = None

    for idx in swings.index:
        row = swings.loc[idx]
        if row["type"] == "H":
            if last_H is None:
                swings.at[idx, "label"] = "H"
            else:
                swings.at[idx, "label"] = "HH" if row["price"] > last_H else "LH"
            last_H = row["price"]
        else:  # type L
            if last_L is None:
                swings.at[idx, "label"] = "L"
            else:
                swings.at[idx, "label"] = "HL" if row["price"] > last_L else "LL"
            last_L = row["price"]

    return swings


def classify_last_bucket(swings):
    """
    Map last 4 labels to 5 buckets:
    Uptrend, Downtrend, Reversal To Uptrend,
    Reversal To Downtrend, Triangle / Sideways.
    """
    labels = list(swings["label"].dropna())
    if len(labels) < 4:
        return "Triangle / Sideways"

    last4 = labels[-4:]

    # exact reversal patterns
    if last4 == ["HH", "HL", "LH", "LL"]:
        return "Reversal To Downtrend"
    if last4 == ["LL", "LH", "HL", "HH"]:
        return "Reversal To Uptrend"

    # strong uptrend: combination of HH + HL
    if all(l in ["HH", "HL"] for l in last4):
        return "Uptrend"

    # strong downtrend
    if all(l in ["LL", "LH"] for l in last4):
        return "Downtrend"

    return "Triangle / Sideways"

# =====================================================
# 61% FIBONACCI RETRACEMENT
# =====================================================
def fib_61_zone_up(swings, tol=0.01):
    """
    Uptrend leg: last HL -> last HH.
    level = 61.8% retracement of that leg.
    """
    sw = swings.copy()
    last_HL = sw[sw["label"] == "HL"].tail(1)
    last_HH = sw[sw["label"] == "HH"].tail(1)
    if last_HL.empty or last_HH.empty:
        return None

    if last_HL.index[-1] > last_HH.index[-1]:
        # low after high => leg not completed
        return None

    low_price = last_HL["price"].iloc[0]
    high_price = last_HH["price"].iloc[0]

    # price at 61.8% retrace from high towards low
    level = high_price - (high_price - low_price) * 0.618
    lo = level * (1 - tol)
    hi = level * (1 + tol)
    return level, lo, hi


def fib_61_zone_down(swings, tol=0.01):
    """
    Downtrend leg: last LH -> last LL.
    """
    sw = swings.copy()
    last_LH = sw[sw["label"] == "LH"].tail(1)
    last_LL = sw[sw["label"] == "LL"].tail(1)
    if last_LH.empty or last_LL.empty:
        return None

    if last_LH.index[-1] > last_LL.index[-1]:
        return None

    high_price = last_LH["price"].iloc[0]
    low_price = last_LL["price"].iloc[0]

    # price at 61.8% retrace from low towards high
    level = low_price + (high_price - low_price) * 0.618
    lo = level * (1 - tol)
    hi = level * (1 + tol)
    return level, lo, hi


def check_61_entry(df, swings, bucket, tol=0.01):
    """
    Returns (is_entry, fib_level) for current close.
    """
    close = df["close"].iloc[-1]

    if bucket == "Uptrend":
        z = fib_61_zone_up(swings, tol)
        if z is None:
            return False, None
        level, lo, hi = z
        return (lo <= close <= hi), float(level)

    if bucket == "Downtrend":
        z = fib_61_zone_down(swings, tol)
        if z is None:
            return False, None
        level, lo, hi = z
        return (lo <= close <= hi), float(level)

    return False, None

# =====================================================
# SCAN ENGINE (DAILY; EXTEND TO 15m/1H/W/M IF NEEDED)
# =====================================================
@st.cache_data
def run_scan(scan_date):
    results = []

    for symbol in symbols:
        try:
            df_d = pd.read_parquet(f"{DATA_D}/{symbol}.parquet")
            df_d = filter_until_date(df_d, scan_date)

            # need enough data for swings
            if len(df_d) < 150:
                continue

            # ---- Dow Theory on Daily ----
            df_sw = detect_swings(df_d, order=3)
            swings = label_structure(df_sw)
            if swings.empty or swings["label"].dropna().shape[0] < 4:
                continue

            bucket = classify_last_bucket(swings)
            is_61, lvl_61 = check_61_entry(df_d, swings, bucket, tol=0.01)

            results.append({
                "Stock": symbol,
                "Trend Bucket": bucket,
                "61% Bullish Entry": bool(is_61) if bucket == "Uptrend" else False,
                "61% Bearish Entry": bool(is_61) if bucket == "Downtrend" else False,
                "61% Level": lvl_61
            })

        except Exception as e:
            # you can use st.write or logging for debugging if required
            continue

    if not results:
        return pd.DataFrame(columns=[
            "Stock", "Trend Bucket",
            "61% Bullish Entry", "61% Bearish Entry", "61% Level"
        ])

    return pd.DataFrame(results)

# =====================================================
# RUN BUTTON
# =====================================================
if "scan_done" not in st.session_state:
    st.session_state.scan_done = False

if st.button("🚀 Run Dow Theory Scan For Selected Date"):
    st.session_state.df_result = run_scan(selected_date)
    st.session_state.scan_done = True

# =====================================================
# DISPLAY RESULTS + FILTERS
# =====================================================
if st.session_state.scan_done:
    df_result = st.session_state.df_result

    st.success(f"Scan Completed For {selected_date.date()} ✅")

    col1, col2 = st.columns(2)

    with col1:
        cat1 = st.selectbox(
            "Trend Scan (Dow Theory)",
            [
                "All",
                "Uptrend",
                "Downtrend",
                "Reversal To Uptrend",
                "Reversal To Downtrend",
                "Triangle / Sideways"
            ]
        )

    with col2:
        cat2 = st.selectbox(
            "Entry Scan (61% Fib)",
            [
                "All",
                "Bullish 61 Entry",
                "Bearish 61 Entry"
            ]
        )

    filtered_df = df_result.copy()

    if cat1 != "All":
        filtered_df = filtered_df[filtered_df["Trend Bucket"] == cat1]

    if cat2 == "Bullish 61 Entry":
        filtered_df = filtered_df[filtered_df["61% Bullish Entry"] == True]
    elif cat2 == "Bearish 61 Entry":
        filtered_df = filtered_df[filtered_df["61% Bearish Entry"] == True]

    st.subheader("📊 Filtered Stocks")
    st.dataframe(filtered_df, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

<div style="line-height: 1.6;">
<b>Designed by:-<br>
Gaurav Singh Yadav</b><br><br>

Built With 💖 🫶<br>
Energyx🔥 | Commodity🛢️ | Quant Intelligence 🧠<br><br>

📱 +91-8003994518 〽️<br>

💬
<a href="https://wa.me/918003994518" target="_blank">
<i class="fa fa-whatsapp" style="color:#25D366;"></i> WhatsApp
</a><br>

📧 <a href="mailto:yadav.gauravsingh@gmail.com">yadav.gauravsingh@gmail.com</a> ™️
</div>
""", unsafe_allow_html=True)
