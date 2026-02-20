import streamlit as st
import pandas as pd
import talib as ta
from pathlib import Path
import base64
import os

st.set_page_config(page_title="Master Scanner", layout="wide",page_icon="🟢")

st.title("📈 Wave Going Against The Tide + SW & MOM Scans")


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

# =====================================================
# APPLY BACKGROUND
# =====================================================

BASE_PATH = os.path.dirname(__file__)
bg_path = os.path.join(BASE_PATH, "Assest", "BG11.png")

if os.path.exists(bg_path):
    set_bg_image(bg_path)
else:
    st.warning(f"Background not found at: {bg_path}")

# ==========================================
# FOLDERS
# ==========================================
DATA_D = "stock_data_D"
DATA_W = "stock_data_W"
DATA_M = "stock_data_M"

symbols = list(set([f.stem for f in Path(DATA_D).glob("*.parquet")]))

# ==========================================
# GET LAST AVAILABLE DAILY DATE
# ==========================================
def get_last_daily_date():
    for f in Path(DATA_D).glob("*.parquet"):
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        return df.index.max()
    return None

last_daily_date = get_last_daily_date()

st.markdown("### 🕯 Lastest Candle")
st.markdown(f"📅 **Daily: {last_daily_date.date()}**")

# ==========================================
# DATE SELECTOR
# ==========================================
selected_date = st.date_input(
    "📅 Select Scan Date",
    value=last_daily_date.date()
)

selected_date = pd.to_datetime(selected_date)

# ==========================================
# FUNCTIONS
# ==========================================
def filter_until_date(df, date):
    df.index = pd.to_datetime(df.index)
    return df[df.index <= date].copy()

def get_macd_trend(series):
    macd, signal, hist = ta.MACD(series, 12, 26, 9)
    macd = macd.dropna()
    if len(macd) < 3:
        return None, None
    now = "Up Tick" if macd.iloc[-1] > macd.iloc[-2] else "Down Tick"
    prev = "Up Tick" if macd.iloc[-2] > macd.iloc[-3] else "Down Tick"
    return now, prev

def classify_trend(d, d1, w, m):

    if d == "Up Tick" and w == "Up Tick" and m == "Up Tick":
        return "Running Uptrend" if d1 == "Up Tick" else "D Aligned Up With W_M"

    if d == "Down Tick" and w == "Down Tick" and m == "Down Tick":
        return "Running Down Trend" if d1 == "Down Tick" else "D Aligned Down With W_M"

    if d == "Down Tick" and w == "Up Tick" and m == "Up Tick":
        return "D (Wave) Going Down/W_M_UP(TIDE)"

    if d == "Up Tick" and w == "Down Tick" and m == "Down Tick":
        return "D(Wave) Going Up /W_M_DN(TIDE)"

    return "No Clear Trend"


# ==========================================
# SCAN ENGINE
# ==========================================
@st.cache_data
def run_scan(scan_date):

    results = []

    for symbol in symbols:
        try:
            df_d = pd.read_parquet(f"{DATA_D}/{symbol}.parquet")
            df_w = pd.read_parquet(f"{DATA_W}/{symbol}.parquet")
            df_m = pd.read_parquet(f"{DATA_M}/{symbol}.parquet")

            df_d = filter_until_date(df_d, scan_date)
            df_w = filter_until_date(df_w, scan_date)
            df_m = filter_until_date(df_m, scan_date)

            if len(df_d) < 100:
                continue

            # ===== MTF =====
            d_now, d_prev = get_macd_trend(df_d["close"])
            w_now, _ = get_macd_trend(df_w["close"])
            m_now, _ = get_macd_trend(df_m["close"])

            if not d_now or not w_now or not m_now:
                continue

            trend_status = classify_trend(d_now, d_prev, w_now, m_now)

            # ===== DAILY STRUCTURE =====
            df_d["ema13"] = ta.EMA(df_d["close"], 13)
            df_d["ema50"] = ta.EMA(df_d["close"], 50)
            df_d["ema100"] = ta.EMA(df_d["close"], 100)
            df_d["rsi"] = ta.RSI(df_d["close"], 14)
            df_d["adx"] = ta.ADX(df_d["high"], df_d["low"], df_d["close"], 14)

            latest = df_d.iloc[-1]

            bullish_momentum = (
                latest["ema13"] > latest["ema50"] > latest["ema100"]
                and latest["adx"] > 20
                and latest["rsi"] > 55
                and latest["close"] > latest["ema13"]
            )

            bearish_momentum = (
                latest["ema13"] < latest["ema50"] < latest["ema100"]
                and latest["adx"] > 20
                and latest["rsi"] < 45
                and latest["close"] < latest["ema13"]
            )

            bullish_swing = (
                latest["ema13"] > latest["ema50"]
                and latest["rsi"] < 55
            )

            bearish_swing = (
                latest["ema13"] < latest["ema50"]
                and latest["rsi"] > 45
            )

            results.append({
                "Stock": symbol,
                "Category 1": trend_status,
                "Bullish Momentum": bullish_momentum,
                "Bearish Momentum": bearish_momentum,
                "Bullish Swing": bullish_swing,
                "Bearish Swing": bearish_swing
            })

        except:
            continue

    return pd.DataFrame(results)


# ==========================================
# RUN BUTTON
# ==========================================
if "scan_done" not in st.session_state:
    st.session_state.scan_done = False

if st.button("🚀 Run Scan For Selected Date"):
    st.session_state.df_result = run_scan(selected_date)
    st.session_state.scan_done = True

# ==========================================
# DISPLAY
# ==========================================
if st.session_state.scan_done:

    df_result = st.session_state.df_result

    st.success(f"Scan Completed For {selected_date.date()} ✅")

    col1, col2 = st.columns(2)

    with col1:
        cat1 = st.selectbox(
            "Category 1 Scan",
            [
                "All",
                "Running Uptrend",
                "Running Down Trend",
                "D Aligned Up With W_M",
                "D Aligned Down With W_M",
                "D (Wave) Going Down/W_M_UP(TIDE)",
                "D(Wave) Going Up /W_M_DN(TIDE)",
                "No Clear Trend"
            ]
        )

    with col2:
        cat2 = st.selectbox(
            "Category 2 Scan",
            [
                "All",
                "Bullish Momentum",
                "Bearish Momentum",
                "Bullish Swing",
                "Bearish Swing"
            ]
        )

    filtered_df = df_result.copy()

    if cat1 != "All":
        filtered_df = filtered_df[filtered_df["Category 1"] == cat1]

    if cat2 != "All":
        filtered_df = filtered_df[filtered_df[cat2] == True]

    st.subheader("📊 Filtered Stocks")
    st.dataframe(filtered_df, use_container_width=True)



st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

<div style="line-height: 1.6;">
<b>Designed by:-<br>
Gaurav Singh Yadav</b><br><br>

🩷💛🩵💙🩶💜🤍🤎💖 Built With Love 🫶<br>
Energy | Commodity | Quant Intelligence 📶<br><br>

📱 +91-8003994518 〽️<br>

💬 
<a href="https://wa.me/918003994518" target="_blank">
<i class="fa fa-whatsapp" style="color:#25D366;"></i> WhatsApp
</a><br>

📧 <a href="mailto:yadav.gauravsingh@gmail.com">yadav.gauravsingh@gmail.com</a> ™️
</div>
""", unsafe_allow_html=True)
