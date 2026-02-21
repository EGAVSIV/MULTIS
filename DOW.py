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

st.title("🌊 Dow Theory Trend + Fib Entry Scanner")

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
# TIMEFRAME + DATE SELECTION
# =====================================================
st.markdown("### ⚙️ Scan Settings")

col_tf, col_date = st.columns(2)

with col_tf:
    if "timeframe" not in st.session_state:
        st.session_state.timeframe = "Daily"
    timeframe = st.selectbox(
        "⏱ Timeframe",
        options=list(TIMEFRAME_MAP.keys()),
        index=list(TIMEFRAME_MAP.keys()).index(st.session_state.timeframe)
    )
    st.session_state.timeframe = timeframe

data_folder = TIMEFRAME_MAP[timeframe]

def get_last_date_for_folder(folder):
    folder_path = Path(folder)
    if not folder_path.exists():
        return None
    for f in folder_path.glob("*.parquet"):
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        return df.index.max()
    return None

last_tf_date = get_last_date_for_folder(data_folder)

with col_date:
    st.markdown("Latest candle in selected timeframe:")
    if last_tf_date is not None:
        st.markdown(f"📅 **{timeframe}: {last_tf_date.date()}**")
        default_date = last_tf_date.date()
    else:
        st.markdown("No data found.")
        default_date = pd.to_datetime("today").date()

if "scan_date" not in st.session_state:
    st.session_state.scan_date = default_date

selected_date = st.date_input(
    "📅 Select Scan Date",
    value=st.session_state.scan_date
)
st.session_state.scan_date = selected_date
scan_date_ts = pd.to_datetime(selected_date)

# build symbol list for selected timeframe
symbols = sorted(list(set([f.stem for f in Path(data_folder).glob("*.parquet")])))

def filter_until_date(df, date):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df[df.index <= date].copy()

# =====================================================
# DOW THEORY SWING + TREND
# =====================================================
def detect_swings(df, order=3):
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
        else:
            if last_L is None:
                swings.at[idx, "label"] = "L"
            else:
                swings.at[idx, "label"] = "HL" if row["price"] > last_L else "LL"
            last_L = row["price"]

    return swings


def classify_last_bucket(swings):
    labels = list(swings["label"].dropna())
    if len(labels) < 4:
        return "Triangle / Sideways"

    last4 = labels[-4:]

    if last4 == ["HH", "HL", "LH", "LL"]:
        return "Reversal To Downtrend"
    if last4 == ["LL", "LH", "HL", "HH"]:
        return "Reversal To Uptrend"

    if all(l in ["HH", "HL"] for l in last4):
        return "Uptrend"

    if all(l in ["LL", "LH"] for l in last4):
        return "Downtrend"

    return "Triangle / Sideways"

# =====================================================
# FIBONACCI RETRACEMENTS (MULTI LEVEL)
# =====================================================
FIB_LEVELS = {
    "23%": 0.23,
    "38%": 0.38,
    "50%": 0.50,
    "61.8%": 0.618,
    "78%": 0.78,
}

FIB_RANGES = {
    "23%": (0.21, 0.25),
    "38%": (0.36, 0.40),
    "50%": (0.49, 0.52),
    "61.8%": (0.60, 0.62),
    "78%": (0.76, 0.78),
}

def fib_levels_up(swings):
    sw = swings.copy()
    last_HL = sw[sw["label"] == "HL"].tail(1)
    last_HH = sw[sw["label"] == "HH"].tail(1)
    if last_HL.empty or last_HH.empty:
        return None

    if last_HL.index[-1] > last_HH.index[-1]:
        return None

    low_price = last_HL["price"].iloc[0]
    high_price = last_HH["price"].iloc[0]
    if high_price == low_price:
        return None

    leg = high_price - low_price
    out = {}
    for name, pct in FIB_LEVELS.items():
        # uptrend retracement from high toward low
        price = high_price - leg * pct
        out[name] = price
    return out, low_price, high_price


def fib_levels_down(swings):
    sw = swings.copy()
    last_LH = sw[sw["label"] == "LH"].tail(1)
    last_LL = sw[sw["label"] == "LL"].tail(1)
    if last_LH.empty or last_LL.empty:
        return None

    if last_LH.index[-1] > last_LL.index[-1]:
        return None

    high_price = last_LH["price"].iloc[0]
    low_price = last_LL["price"].iloc[0]
    if high_price == low_price:
        return None

    leg = high_price - low_price
    out = {}
    for name, pct in FIB_LEVELS.items():
        # downtrend retracement from low toward high
        price = low_price + leg * pct
        out[name] = price
    return out, low_price, high_price


def check_fib_entries(df, swings, bucket):
    close = df["close"].iloc[-1]

    fib_hits = {
        "23% Bull": False, "38% Bull": False, "50% Bull": False,
        "61.8% Bull": False, "78% Bull": False,
        "23% Bear": False, "38% Bear": False, "50% Bear": False,
        "61.8% Bear": False, "78% Bear": False,
    }
    fib_prices = {f"{k} Up": None for k in FIB_LEVELS.keys()}
    fib_prices.update({f"{k} Down": None for k in FIB_LEVELS.keys()})

    if bucket == "Uptrend":
        res = fib_levels_up(swings)
        if res is None:
            return fib_hits, fib_prices
        prices_dict, low_price, high_price = res

        for name, price in prices_dict.items():
            low_pct, high_pct = FIB_RANGES[name]
            # convert pct window in terms of price distance from high
            leg = high_price - low_price
            price_low = high_price - leg * high_pct
            price_high = high_price - leg * low_pct
            fib_prices[f"{name} Up"] = float(price)

            if price_low <= close <= price_high:
                fib_hits[f"{name} Bull"] = True

    if bucket == "Downtrend":
        res = fib_levels_down(swings)
        if res is None:
            return fib_hits, fib_prices
        prices_dict, low_price, high_price = res

        for name, price in prices_dict.items():
            low_pct, high_pct = FIB_RANGES[name]
            leg = high_price - low_price
            price_low = low_price + leg * low_pct
            price_high = low_price + leg * high_pct
            fib_prices[f"{name} Down"] = float(price)

            if price_low <= close <= price_high:
                fib_hits[f"{name} Bear"] = True

    return fib_hits, fib_prices

# =====================================================
# HEAVY SCAN FUNCTION (CACHED)
# =====================================================
@st.cache_data(show_spinner=False)
def run_scan_cached(scan_date_str, symbol_list, tf_folder):
    scan_date = pd.to_datetime(scan_date_str)
    results = []

    for symbol in symbol_list:
        try:
            df = pd.read_parquet(f"{tf_folder}/{symbol}.parquet")
            df = filter_until_date(df, scan_date)
            if len(df) < 150:
                continue

            df_sw = detect_swings(df, order=3)
            swings = label_structure(df_sw)
            if swings.empty or swings["label"].dropna().shape[0] < 4:
                continue

            bucket = classify_last_bucket(swings)
            fib_hits, fib_prices = check_fib_entries(df, swings, bucket)

            row = {
                "Stock": symbol,
                "Timeframe": tf_folder,
                "Trend Bucket": bucket,
            }
            row.update(fib_hits)
            row.update(fib_prices)
            results.append(row)

        except Exception:
            continue

    if not results:
        cols = ["Stock", "Timeframe", "Trend Bucket"]
        cols += list(fib_hits.keys()) + list(fib_prices.keys())
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(results)

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "scan_done" not in st.session_state:
    st.session_state.scan_done = False
if "df_result" not in st.session_state:
    st.session_state.df_result = pd.DataFrame()
if "is_scanning" not in st.session_state:
    st.session_state.is_scanning = False

# =====================================================
# BUTTON CALLBACK
# =====================================================
def start_scan():
    st.session_state.is_scanning = True
    st.session_state.scan_done = False

    df = run_scan_cached(str(st.session_state.scan_date), symbols, data_folder)
    st.session_state.df_result = df
    st.session_state.scan_done = True
    st.session_state.is_scanning = False

# =====================================================
# RUN BUTTON
# =====================================================
st.button("🚀 Run Dow Theory + Fib Scan", on_click=start_scan)

if st.session_state.is_scanning:
    with st.spinner("Scanning stocks... This may take some time."):
        sleep(0.5)

# =====================================================
# DISPLAY RESULTS + FILTERS
# =====================================================
if st.session_state.scan_done and not st.session_state.df_result.empty:
    df_result = st.session_state.df_result

    st.success(f"Scan Completed For {st.session_state.scan_date} ({timeframe}) ✅")

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
            "Fib Entry Type",
            [
                "All",
                "23% Bull", "38% Bull", "50% Bull", "61.8% Bull", "78% Bull",
                "23% Bear", "38% Bear", "50% Bear", "61.8% Bear", "78% Bear",
            ]
        )

    filtered_df = df_result.copy()

    if cat1 != "All":
        filtered_df = filtered_df[filtered_df["Trend Bucket"] == cat1]

    if cat2 != "All":
        filtered_df = filtered_df[filtered_df[cat2] == True]

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
