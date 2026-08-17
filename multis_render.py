import os
import sys
import base64
import numpy as np
import pandas as pd
import talib
import plotly.express as px
import streamlit as st

from login import login_page
from database import create_database, upgrade_database
from styles import load_css, show_footer
from utils import logout, greeting
from admin import admin_panel
from streamlit_autorefresh import st_autorefresh

# ======================================
# STREAMLIT PAGE CONFIG & INITIALIZATION
# ======================================
st.set_page_config(
    page_title="FNO_STOCK_SCAN",
    layout="wide",
    page_icon="🧮"
)

st_autorefresh(
    interval=600000,
    key="auto_refresh"
)

# Python 3.13 image module fallback
if sys.version_info >= (3, 13):
    import types
    imghdr = types.ModuleType("imghdr")
    imghdr.what = lambda *args, **kwargs: None
    sys.modules["imghdr"] = imghdr

try:
    EMAIL_ADDRESS = st.secrets.get("EMAIL_ADDRESS", os.getenv("EMAIL_ADDRESS", "default_email@example.com"))
    EMAIL_PASSWORD = st.secrets.get("EMAIL_PASSWORD", os.getenv("EMAIL_PASSWORD", ""))
except Exception:
    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "default_email@example.com")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")



BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Database Setup
create_database()
upgrade_database()

# Load CSS
load_css()

# ======================================
# GLOBAL CONSTANTS & HELPERS
# ======================================
SAFE_COLS = [
    "Symbol",
    "Signal",
    "Trend",
    "State",
    "Setup",
    "Divergence",
    "RSI",
    "Zone",
    "Confluence",
    "Bias",
    "Probability",
    "TV_Link",
]

BULL_KEYWORDS = ["Bullish", "BUY", "Breakout", "Uptrend", "Momentum"]
BEAR_KEYWORDS = ["Bearish", "SELL", "Breakdown", "Downtrend"]

def empty_result_df():
    return pd.DataFrame({c: [] for c in SAFE_COLS})

def set_bg_image(image_path: str):
    if not os.path.exists(image_path):
        return
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

# Set background if available
bg_path = os.path.join(BASE_PATH, "Assest", "BG11.png")
if os.path.exists(bg_path):
    set_bg_image(bg_path)

# ======================================
# AUTHENTICATION & ROUTING
# ======================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_page()
    st.stop()

if st.session_state.get("role") == "Admin":
    admin_panel()
    st.stop()

# ======================================
# USER HEADER
# ======================================
col_h1, col_h2 = st.columns([6, 1])
with col_h1:
    st.success(
        f"{greeting()}, {st.session_state.get('fullname', st.session_state.get('username', 'User'))}! 👋"
    )
    st.caption(
        f"👤 Username : {st.session_state.get('username')}  |  "
        f"💳 Role : {st.session_state.get('role')}  |  "
        f"📅 Expiry : {st.session_state.get('expiry_date')}"
    )
with col_h2:
    if st.button("🚪 Logout"):
        logout()

st.markdown(
    """
    <h1 style="color: #1E88E5; font-weight: 700; margin-bottom: 0.2rem;">
        📊 Multi-Timeframe Stock Screener
    </h1>
    """,
    unsafe_allow_html=True,
)

# ======================================
# DATA TIMEFRAMES & UTILITIES
# ======================================
TIMEFRAMES = {
    "15 Min": os.path.join(BASE_PATH, "stock_data_15"),
    "1 Hour": os.path.join(BASE_PATH, "stock_data_1H"),
    "Daily": os.path.join(BASE_PATH, "stock_data_D"),
    "Weekly": os.path.join(BASE_PATH, "stock_data_W"),
    "Monthly": os.path.join(BASE_PATH, "stock_data_M"),
}

def get_last_candle_by_tf(folder_path: str):
    last_dt = None
    if not os.path.isdir(folder_path):
        return None
    
    for f in os.listdir(folder_path):
        if not f.endswith(".parquet"):
            continue
        try:
            df = pd.read_parquet(os.path.join(folder_path, f))
            if df.empty:
                continue

            if isinstance(df.index, pd.DatetimeIndex):
                dt = df.index[-1]
            elif "datetime" in df.columns:
                dt = pd.to_datetime(df["datetime"]).iloc[-1]
            else:
                continue

            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            else:
                dt = dt.tz_convert("UTC")

            dt = dt.tz_convert("Asia/Kolkata")

            if last_dt is None or dt > last_dt:
                last_dt = dt
        except Exception:
            continue

    return last_dt

@st.cache_data(show_spinner=False)
def load_data(folder: str):
    data = {}
    if not os.path.exists(folder):
        return data

    for f in os.listdir(folder):
        if not f.endswith(".parquet"):
            continue

        sym = f.replace(".parquet", "")
        df = pd.read_parquet(os.path.join(folder, f))

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").set_index("datetime")

        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(df.columns):
            continue

        data[sym] = df

    return data

def make_tradingview_link(sym: str) -> str:
    base = "https://in.tradingview.com/chart/LqUZraZ9/"
    return f"{base}?symbol=NSE%3A{sym}"

# Sidebar Setup
tf_options = list(TIMEFRAMES.keys())
tf = st.sidebar.selectbox("Timeframe", tf_options, index=tf_options.index("Daily"))

sample_data = load_data(TIMEFRAMES[tf])
all_symbols = sorted(sample_data.keys()) if sample_data else []
st.sidebar.markdown("### 🔍 Single Stock Scan")
selected_symbol = st.sidebar.selectbox(
    "Select Stock (for current timeframe)",
    all_symbols if all_symbols else ["NA"],
)

last_15m = get_last_candle_by_tf(TIMEFRAMES["15 Min"])
last_1h = get_last_candle_by_tf(TIMEFRAMES["1 Hour"])
last_d = get_last_candle_by_tf(TIMEFRAMES["Daily"])
last_w = get_last_candle_by_tf(TIMEFRAMES["Weekly"])
last_m = get_last_candle_by_tf(TIMEFRAMES["Monthly"])

col_ref, col_info = st.columns([1, 6])
with col_ref:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.session_state["needs_refresh_msg"] = True
        st.rerun()

if st.session_state.get("needs_refresh_msg", False):
    st.toast("🍏 Fresh data loaded successfully!", icon="✅")
    st.session_state["needs_refresh_msg"] = False

with col_info:
    st.markdown(
        f"""
🕯 **Last Candle (IST)**  
⏱ **15 Min**: {last_15m.strftime('%d %b %Y %H:%M') if last_15m else 'NA'}  |  
⏰ **1 Hour**: {last_1h.strftime('%d %b %Y %H:%M') if last_1h else 'NA'}  |  
📅 **Daily**: {last_d.date() if last_d else 'NA'}  |  
📆 **Weekly**: {last_w.date() if last_w else 'NA'}  |  
🗓 **Monthly**: {last_m.date() if last_m else 'NA'}
 """,
        unsafe_allow_html=False,
    )

st.markdown("---")

# Backtest Controls
st.sidebar.markdown("### 📅 Backtest Date")
analysis_date = st.sidebar.date_input(
    "Select Analysis Date",
    value=last_d.date() if last_d else pd.Timestamp.today().date(),
)
st.sidebar.info(f"Backtest Mode Active\nData cutoff: {analysis_date}")

def trim_df_to_date(df: pd.DataFrame, anchor_date):
    if df is None or df.empty:
        return None
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index.date <= anchor_date]
    elif "datetime" in df.columns:
        df = df[df["datetime"].dt.date <= anchor_date]
    
    if len(df) < 120:
        return None
    return df

# ======================================
# SCANNER IMPLEMENTATIONS
# ======================================
def rsi_market_pulse(df):
    if len(df) < 14:
        return None
    rsi = talib.RSI(df["close"], 14).iloc[-1]
    if rsi > 60:
        zone = "RSI > 60"
    elif rsi < 40:
        zone = "RSI < 40"
    else:
        zone = "RSI 40–60"
    return round(rsi, 2), zone

def volume_shocker(df):
    if len(df) < 20:
        return False
    vol_sma = df["volume"].rolling(10).mean()
    last, prev = df.iloc[-1], df.iloc[-2]
    return (
        last["volume"] > 2 * vol_sma.iloc[-1]
        and prev["close"] * 0.95 <= last["close"] <= prev["close"] * 1.05
    )

def nrb_7(df):
    if len(df) < 20:
        return None
    base = df.iloc[-7]
    inside = df.iloc[-6:-1]
    last = df.iloc[-1]
    base_high, base_low = base["high"], base["low"]
    
    cond_high_low = inside["high"].max() <= base_high and inside["low"].min() >= base_low
    cond_open_close = (
        inside["open"].max() <= base_high
        and inside["open"].min() >= base_low
        and inside["close"].max() <= base_high
        and inside["close"].min() >= base_low
    )
    if not (cond_high_low and cond_open_close):
        return None
    
    avg_vol = df["volume"].rolling(10).mean().iloc[-2]
    if last["volume"] < 1.5 * avg_vol:
        return None
    
    if last["close"] > base_high:
        return "NRB-7 Bullish Breakout + Volume"
    if last["close"] < base_low:
        return "NRB-7 Bearish Breakdown + Volume"
    return None

def counter_attack(df):
    if len(df) < 2:
        return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    mid = (prev["open"] + prev["close"]) / 2
    if prev["close"] < prev["open"] and curr["close"] > curr["open"]:
        if curr["open"] < prev["close"] and curr["close"] >= mid:
            return "Bullish"
    if prev["close"] > prev["open"] and curr["close"] < curr["open"]:
        if curr["open"] > prev["close"] and curr["close"] <= mid:
            return "Bearish"
    return None

def breakaway_gap(df):
    if len(df) < 50:
        return None
    df = df.copy()
    df["EMA20"] = talib.EMA(df["close"], 20)
    df["EMA50"] = talib.EMA(df["close"], 50)
    prev, curr = df.iloc[-2], df.iloc[-1]
    
    if curr["open"] > prev["high"] * 1.005 and curr["low"] > prev["high"]:
        if curr["EMA20"] < curr["EMA50"]:
            return "Bullish Breakaway Gap"
    if curr["open"] < prev["low"] * 0.995 and curr["high"] < prev["low"]:
        if curr["EMA20"] > curr["EMA50"]:
            return "Bearish Breakaway Gap"
    return None

def rsi_adx(df):
    if len(df) < 20:
        return None
    rsi = talib.RSI(df["close"], 14).iloc[-1]
    adx = talib.ADX(df["high"], df["low"], df["close"], 14).iloc[-1]
    if adx > 50 and rsi < 20:
        return "Bullish Reversal"
    if adx > 50 and rsi > 80:
        return "Probabale Bearish Reversal"
    return None

def rsi_wm(df_tf, df_w, df_m):
    r_tf = talib.RSI(df_tf["close"], 14).iloc[-1]
    r_w = talib.RSI(df_w["close"], 14).iloc[-1]
    r_m = talib.RSI(df_m["close"], 14).iloc[-1]
    if r_w > 60 and r_m > 60 and r_tf < 40:
        return "Bullish WM Reversal"
    if r_w < 40 and r_m < 40 and r_tf > 60:
        return "Bearish WM Reversal"
    return None

def macd_market_pulse(df):
    if len(df) < 30:
        return None
    macd, signal, _ = talib.MACD(df["close"], 12, 26, 9)
    m, s, pm = macd.iloc[-1], signal.iloc[-1], macd.iloc[-2]
    
    if m > 0 and m > s and m > pm:
        return "Strong Bullish"
    if m > 0 and m > s and m < pm:
        return "Bullish Cooling"
    if m > 0 and m < s and m > pm:
        return "Bullish Reversal Watch"
    if m > 0 and m < s and m < pm:
        return "Weak Bullish"
    if m < 0 and m > s and m > pm:
        return "Bearish Reversal Watch"
    if m < 0 and m > s and m < pm:
        return "Weak Bearish"
    if m < 0 and m < s and m > pm:
        return "Bearish Recovery Attempt"
    if m < 0 and m < s and m < pm:
        return "Strong Bearish"
    return None

def macd_normal_divergence(df, lookback=30):
    if len(df) < lookback:
        return None
    macd, _, _ = talib.MACD(df["close"], 12, 26, 9)
    price_low1, price_low2 = df["low"].iloc[-lookback:-15].min(), df["low"].iloc[-15:].min()
    macd_low1, macd_low2 = macd.iloc[-lookback:-15].min(), macd.iloc[-15:].min()
    if price_low2 < price_low1 and macd_low2 > macd_low1:
        return "Bullish ND"
    price_high1, price_high2 = df["high"].iloc[-lookback:-15].max(), df["high"].iloc[-15:].max()
    macd_high1, macd_high2 = macd.iloc[-lookback:-15].max(), macd.iloc[-15:].max()
    if price_high2 > price_high1 and macd_high2 < macd_high1:
        return "Bearish ND"
    return None

def macd_rd(df, df_htf):
    if len(df) < 60 or len(df_htf) < 30:
        return None
    macd, signal, _ = talib.MACD(df["close"], 12, 26, 9)
    latest, prev, sig = macd.iloc[-1], macd.iloc[-2], signal.iloc[-1]
    max60 = macd.rolling(60).max().iloc[-1]
    
    macd_htf, _, _ = talib.MACD(df_htf["close"], 12, 26, 9)
    macd_htf_val = macd_htf.iloc[-1]
    macd_htf_uptick = macd_htf.iloc[-1] > macd_htf.iloc[-2]
    
    ema50_ltf = talib.EMA(df["close"], 50).iloc[-1]
    ema50_htf = talib.EMA(df_htf["close"], 50).iloc[-1]
    ema_condition = df["close"].iloc[-1] > ema50_ltf and df_htf["close"].iloc[-1] > ema50_htf
    
    if (
        latest > prev and latest > 0 and sig < latest
        and macd_htf_val > 0 and max60 > 0 and (latest / max60) < 0.25
        and ema_condition and macd_htf_uptick
    ):
        return "MACD RD (Compression + Trend Aligned)"
    return None

def third_wave_finder(df, lookback_cross=50, tolerance=0.02):
    if len(df) < lookback_cross + 5:
        return False
    close = df["close"]
    ema20, ema50 = talib.EMA(close, 20), talib.EMA(close, 50)
    
    cross_idx = None
    for i in range(len(close) - 1, 0, -1):
        if ema20.iloc[i] > ema50.iloc[i] and ema20.iloc[i - 1] <= ema50.iloc[i - 1]:
            cross_idx = i
            break
    if cross_idx is None:
        return False
    
    start_idx = max(0, cross_idx - lookback_cross)
    pre_ema20, pre_ema50 = ema20.iloc[start_idx:cross_idx], ema50.iloc[start_idx:cross_idx]
    if pre_ema20.empty or (pre_ema20 < pre_ema50).mean() < 0.7:
        return False
    
    ema50_now, price_now = ema50.iloc[-1], close.iloc[-1]
    if ema50_now == 0 or np.isnan(ema50_now):
        return False
    
    if abs(price_now - ema50_now) / ema50_now > tolerance or ema20.iloc[-1] <= ema50.iloc[-1]:
        return False
    return True

def c_wave_finder(df, lookback_cross=50, tolerance=0.02):
    if len(df) < lookback_cross + 5:
        return False
    close = df["close"]
    ema20, ema50 = talib.EMA(close, 20), talib.EMA(close, 50)
    
    cross_idx = None
    for i in range(len(close) - 1, 0, -1):
        if ema20.iloc[i] < ema50.iloc[i] and ema20.iloc[i - 1] >= ema50.iloc[i - 1]:
            cross_idx = i
            break
    if cross_idx is None:
        return False
    
    start_idx = max(0, cross_idx - lookback_cross)
    pre_ema20, pre_ema50 = ema20.iloc[start_idx:cross_idx], ema50.iloc[start_idx:cross_idx]
    if pre_ema20.empty or (pre_ema20 > pre_ema50).mean() < 0.7:
        return False
    
    ema50_now, price_now = ema50.iloc[-1], close.iloc[-1]
    if ema50_now == 0 or np.isnan(ema50_now):
        return False
    
    if abs(price_now - ema50_now) / ema50_now > tolerance or ema20.iloc[-1] >= ema50.iloc[-1]:
        return False
    return True

def macd_peak_bearish_divergence(df):
    if len(df) < 80:
        return None
    macd, _, _ = talib.MACD(df["close"], 12, 26, 9)
    old_slice, new_slice = slice(-60, -30), slice(-30, None)
    price_high1, price_high2 = df["high"].iloc[old_slice].max(), df["high"].iloc[new_slice].max()
    idx1, idx2 = df["high"].iloc[old_slice].idxmax(), df["high"].iloc[new_slice].idxmax()
    
    if price_high2 > price_high1 and macd.loc[idx2] < macd.loc[idx1]:
        return "Bearish MACD Peak Divergence"
    return None

def macd_base_bullish_divergence(df):
    if len(df) < 80:
        return None
    macd, _, _ = talib.MACD(df["close"], 12, 26, 9)
    price_low1, price_low2 = df["low"].iloc[-60:-30].min(), df["low"].iloc[-30:].min()
    macd_low1, macd_low2 = macd.iloc[-60:-30].min(), macd.iloc[-30:].min()
    if price_low2 < price_low1 and macd_low2 > macd_low1:
        return "Bullish MACD Base Divergence"
    return None

def trend_alignment(df):
    if len(df) < 100:
        return None
    ema20, ema50, ema100 = talib.EMA(df["close"], 20), talib.EMA(df["close"], 50), talib.EMA(df["close"], 100)
    if ema20.iloc[-1] > ema50.iloc[-1] > ema100.iloc[-1]:
        return "Strong Uptrend"
    if ema20.iloc[-1] < ema50.iloc[-1] < ema100.iloc[-1]:
        return "Strong Downtrend"
    return None

def pullback_to_ema(df):
    if len(df) < 60:
        return None
    ema20, ema50 = talib.EMA(df["close"], 20), talib.EMA(df["close"], 50)
    last = df.iloc[-1]
    if ema20.iloc[-1] > ema50.iloc[-1] and last["low"] <= ema20.iloc[-1] and last["close"] > ema20.iloc[-1]:
        return "Bullish EMA Pullback"
    if ema20.iloc[-1] < ema50.iloc[-1] and last["high"] >= ema20.iloc[-1] and last["close"] < ema20.iloc[-1]:
        return "Bearish EMA Pullback"
    return None

def confluence_setup(df):
    if len(df) < 60:
        return None
    rsi = talib.RSI(df["close"], 14).iloc[-1]
    macd, sig, _ = talib.MACD(df["close"], 12, 26, 9)
    ema20, ema50 = talib.EMA(df["close"], 20), talib.EMA(df["close"], 50)
    if rsi > 50 and macd.iloc[-1] > sig.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
        return "Bullish Confluence"
    if rsi < 50 and macd.iloc[-1] < sig.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]:
        return "Bearish Confluence"
    return None

def macd_hook_up(df):
    if len(df) < 35:
        return None
    macd, signal, hist = talib.MACD(df["close"], 12, 26, 9)
    if (
        macd.iloc[-1] > 0 and macd.iloc[-1] > signal.iloc[-1]
        and macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-2] < macd.iloc[-3]
        and macd.iloc[-1] > macd.iloc[-2] and hist.iloc[-1] > hist.iloc[-2]
    ):
        return "MACD Hook Up"
    return None

def macd_hook_down(df):
    if len(df) < 35:
        return None
    macd, signal, hist = talib.MACD(df["close"], 12, 26, 9)
    if (
        macd.iloc[-1] < 0 and macd.iloc[-1] < signal.iloc[-1]
        and macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-2] > macd.iloc[-3]
        and macd.iloc[-1] < macd.iloc[-2] and hist.iloc[-1] < hist.iloc[-2]
    ):
        return "MACD Hook Down"
    return None

def macd_histogram_divergence(df):
    if len(df) < 50:
        return None
    _, _, hist = talib.MACD(df["close"], 12, 26, 9)
    price_low1, price_low2 = df["low"].iloc[-40:-20].min(), df["low"].iloc[-20:].min()
    hist_low1, hist_low2 = hist.iloc[-40:-20].min(), hist.iloc[-20:].min()
    if price_low2 < price_low1 and hist_low2 > hist_low1:
        return "Bullish Histogram Divergence"
    price_high1, price_high2 = df["high"].iloc[-40:-20].max(), df["high"].iloc[-20:].max()
    hist_high1, hist_high2 = hist.iloc[-40:-20].max(), hist.iloc[-20:].max()
    if price_high2 > price_high1 and hist_high2 < hist_high1:
        return "Bearish Histogram Divergence"
    return None

def ema50_stoch_oversold(df):
    if len(df) < 50:
        return None
    ema50 = talib.EMA(df["close"], 50)
    slowk, slowd = talib.STOCH(df["high"], df["low"], df["close"], fastk_period=14, slowk_period=3, slowd_period=3)
    price, ema_val = df["close"].iloc[-1], ema50.iloc[-1]
    near_ema = abs(price - ema_val) / ema_val <= 0.005
    stoch_cross = slowk.iloc[-2] < slowd.iloc[-2] and slowk.iloc[-1] > slowd.iloc[-1] and slowk.iloc[-1] < 20
    if near_ema and stoch_cross:
        return "EMA50 + Stoch Oversold Buy"
    return None

def dark_cloud_cover(df):
    if len(df) < 15:
        return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    if prev["close"] <= prev["open"] or curr["close"] >= curr["open"] or curr["open"] <= prev["close"]:
        return None
    if talib.RSI(df["close"], 14).iloc[-2] <= 60:
        return None
    if curr["close"] >= (prev["open"] + prev["close"]) / 2:
        return None
    return "Dark Cloud Cover (Bearish | RSI>60)"

def morning_star_bottom(df):
    if len(df) < 60:
        return None
    ema50 = talib.EMA(df["close"], 50)
    if df["close"].iloc[-1] > ema50.iloc[-1]:
        return None
    pattern = talib.CDLMORNINGSTAR(df["open"], df["high"], df["low"], df["close"]).iloc[-1]
    return "Morning Star (Bottom)" if pattern > 0 else None

def evening_star_top(df):
    if len(df) < 60:
        return None
    ema50 = talib.EMA(df["close"], 50)
    if df["close"].iloc[-1] < ema50.iloc[-1]:
        return None
    pattern = talib.CDLEVENINGSTAR(df["open"], df["high"], df["low"], df["close"]).iloc[-1]
    return "Evening Star (Top)" if pattern < 0 else None

def bullish_gsas(df_tf, df_htf):
    rsi = talib.RSI(df_tf["close"], 14)
    adx = talib.ADX(df_tf["high"], df_tf["low"], df_tf["close"], 14)
    ubb, _, _ = talib.BBANDS(df_tf["close"], 20)
    macd_htf, sig_htf, _ = talib.MACD(df_htf["close"], 12, 26, 9)
    ema20_htf = talib.EMA(df_htf["close"], 20)
    if (
        rsi.iloc[-1] > 60 and ubb.iloc[-1] > ubb.iloc[-2]
        and adx.iloc[-1] > adx.iloc[-2] and adx.iloc[-2] < adx.iloc[-3]
        and macd_htf.iloc[-1] > sig_htf.iloc[-1]
        and df_htf["close"].iloc[-1] > ema20_htf.iloc[-1]
    ):
        return "Bullish GSAS"
    return None

def bearish_gsas(df_tf, df_htf):
    rsi = talib.RSI(df_tf["close"], 14)
    adx = talib.ADX(df_tf["high"], df_tf["low"], df_tf["close"], 14)
    _, _, lbb = talib.BBANDS(df_tf["close"], 20)
    macd_htf, sig_htf, _ = talib.MACD(df_htf["close"], 12, 26, 9)
    ema20_htf = talib.EMA(df_htf["close"], 20)
    if (
        rsi.iloc[-1] < 60 and lbb.iloc[-1] < lbb.iloc[-2]
        and adx.iloc[-1] > adx.iloc[-2] and adx.iloc[-2] < adx.iloc[-3]
        and macd_htf.iloc[-1] < sig_htf.iloc[-1]
        and df_htf["close"].iloc[-1] < ema20_htf.iloc[-1]
    ):
        return "Bearish GSAS"
    return None

def rsi_swing(df):
    if len(df) < 20:
        return None
    rsi = talib.RSI(df["close"], 14)
    rsi_prev, rsi_curr = rsi.iloc[-2], rsi.iloc[-1]
    if rsi_prev < 40 and rsi_curr > 40:
        return "RSI Bullish Swing"
    if rsi_prev > 60 and rsi_curr < 60:
        return "RSI Bearish Swing"
    return None

def ema50_fake_breakdown(df):
    if len(df) < 55:
        return None
    df = df.copy()
    df["EMA20"] = talib.EMA(df["close"], 20)
    df["EMA50"] = talib.EMA(df["close"], 50)
    prev, curr = df.iloc[-2], df.iloc[-1]
    if curr["close"] > curr["EMA50"] and prev["close"] < prev["EMA50"] and curr["EMA20"] > curr["EMA50"]:
        return "50 EMA Fake Breakdown"
    return None

def ema50_fake_breakout(df):
    if len(df) < 55:
        return None
    df = df.copy()
    df["EMA20"] = talib.EMA(df["close"], 20)
    df["EMA50"] = talib.EMA(df["close"], 50)
    prev, curr = df.iloc[-2], df.iloc[-1]
    if curr["close"] < curr["EMA50"] and prev["close"] > prev["EMA50"] and curr["EMA20"] < curr["EMA50"]:
        return "50 EMA Fake Breakout"
    return None

def kdj(df, period=9, signal=3):
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    rng = (high_max - low_min).replace(0, np.nan)
    rsv = (100 * (df["close"] - low_min) / rng).clip(lower=0, upper=100)
    
    def bcwsma(series, length, m=1):
        out = []
        for i, val in enumerate(series):
            if i == 0 or np.isnan(val):
                out.append(val)
            else:
                out.append((m * val + (length - m) * out[i - 1]) / length)
        return pd.Series(out, index=series.index)
    
    pK = bcwsma(rsv, signal, 1)
    pD = bcwsma(pK, signal, 1)
    pJ = 3 * pK - 2 * pD
    return pK, pD, pJ

def kdj_buy(df):
    if len(df) < 20:
        return None
    pK, pD, pJ = kdj(df)
    if pd.isna(pD.iloc[-1]) or pd.isna(pJ.iloc[-1]):
        return None
    if (pJ.iloc[-2] < pD.iloc[-2]) and (pJ.iloc[-1] > pD.iloc[-1]) and (pD.iloc[-1] < 30) and (pJ.iloc[-1] < 30):
        return "KDJ BUY (J↑D oversold)"
    return None

def kdj_sell(df):
    if len(df) < 20:
        return None
    pK, pD, pJ = kdj(df)
    if pd.isna(pD.iloc[-1]) or pd.isna(pJ.iloc[-1]):
        return None
    if (pJ.iloc[-2] > pD.iloc[-2]) and (pJ.iloc[-1] < pD.iloc[-1]) and (pD.iloc[-1] > 70) and (pJ.iloc[-1] > 70):
        return "KDJ SELL (J↓D overbought)"
    return None

def consecutive_close_momentum(df, min_count=3):
    if len(df) < min_count + 1:
        return None
    closes = df["close"].values
    direction = "Bull" if closes[-1] > closes[-2] else "Bear" if closes[-1] < closes[-2] else None
    if not direction:
        return None
    
    count = 1
    for i in range(len(closes) - 2, 0, -1):
        if (direction == "Bull" and closes[i] > closes[i - 1]) or (direction == "Bear" and closes[i] < closes[i - 1]):
            count += 1
        else:
            break
    return (direction, count) if count >= min_count else None

def camarilla_breakout(df):
    if len(df) < 2:
        return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    rng = prev["high"] - prev["low"]
    H4 = prev["close"] + (rng * 1.1 / 2)
    L4 = prev["close"] - (rng * 1.1 / 2)
    if curr["close"] > H4:
        return "Bullish Camarilla Breakout"
    if curr["close"] < L4:
        return "Bearish Camarilla Breakdown"
    return None

def cpr_breakout(df):
    if len(df) < 2:
        return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    pivot = (prev["high"] + prev["low"] + prev["close"]) / 3
    bc = (prev["high"] + prev["low"]) / 2
    tc = (pivot * 2) - bc
    top, bottom = max(tc, bc), min(tc, bc)
    if curr["close"] > top:
        return "Bullish CPR Breakout"
    if curr["close"] < bottom:
        return "Bearish CPR Breakdown"
    return None

def inside_bar_breakout(df):
    if len(df) < 4:
        return None
    mother, inside1, inside2, curr = df.iloc[-4], df.iloc[-3], df.iloc[-2], df.iloc[-1]
    inside_ok = (
        inside1["high"] < mother["high"] and inside1["low"] > mother["low"]
        and inside2["high"] < mother["high"] and inside2["low"] > mother["low"]
    )
    if not inside_ok:
        return None
    if curr["close"] > mother["high"]:
        return "Bullish Inside Bar Breakout (3-bar coil)"
    if curr["close"] < mother["low"]:
        return "Bearish Inside Bar Breakdown (3-bar coil)"
    return None

def adx_expansion(df):
    if len(df) < 30:
        return None
    adx = talib.ADX(df["high"], df["low"], df["close"], 14)
    ema20 = talib.EMA(df["close"], 20)
    if adx.iloc[-2] < 20 and adx.iloc[-1] > 25:
        if df["close"].iloc[-1] > ema20.iloc[-1]:
            return "Bullish ADX Expansion"
        if df["close"].iloc[-1] < ema20.iloc[-1]:
            return "Bearish ADX Expansion"
    return None

def range_expansion_day(df, lookback=5):
    if len(df) < lookback + 2:
        return None
    today = df.iloc[-1]
    avg_range = (df["high"] - df["low"]).iloc[-lookback - 1 : -1].mean()
    if (today["high"] - today["low"]) > 1.5 * avg_range:
        return "Bullish Range Expansion Day" if today["close"] > today["open"] else "Bearish Range Expansion Day"
    return None

def failed_breakout_breakdown(df, lookback=20):
    if len(df) < lookback + 2:
        return None
    recent_high = df["high"].iloc[-lookback:-1].max()
    recent_low = df["low"].iloc[-lookback:-1].min()
    prev, curr = df.iloc[-2], df.iloc[-1]
    if prev["high"] > recent_high and curr["close"] < recent_high:
        return "Failed Breakout (Bearish)"
    if prev["low"] < recent_low and curr["close"] > recent_low:
        return "Failed Breakdown (Bullish)"
    return None

def ema_compression_expansion(df):
    if len(df) < 60:
        return None
    ema20, ema50, ema100 = talib.EMA(df["close"], 20), talib.EMA(df["close"], 50), talib.EMA(df["close"], 100)
    compression = (
        abs(ema20.iloc[-2] - ema50.iloc[-2]) / ema50.iloc[-2] < 0.003
        and abs(ema50.iloc[-2] - ema100.iloc[-2]) / ema100.iloc[-2] < 0.003
    )
    if not compression:
        return None
    if ema20.iloc[-1] > ema50.iloc[-1] > ema100.iloc[-1]:
        return "Bullish EMA Compression Break"
    if ema20.iloc[-1] < ema50.iloc[-1] < ema100.iloc[-1]:
        return "Bearish EMA Compression Break"
    return None

def rsi_macd_cross_swing(df):
    if len(df) < 50:
        return None
    rsi = talib.RSI(df["close"], 14)
    macd, signal, _ = talib.MACD(df["close"], 12, 26, 9)
    if rsi.iloc[-2] < 40 and rsi.iloc[-1] > 40 and macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        return "Bullish RSI+MACD Cross"
    if rsi.iloc[-2] > 60 and rsi.iloc[-1] < 60 and macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
        return "Bearish RSI+MACD Cross"
    return None

def atr_percent(df, period=14):
    if len(df) < period + 1:
        return None
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)
    val, close = atr.iloc[-1], df["close"].iloc[-1]
    if pd.isna(val) or close <= 0:
        return None
    return (val / close) * 100.0

def liquidity_sweep_reversal(df, lookback=20):
    if len(df) < lookback + 2:
        return None
    prev_high = df["high"].iloc[-lookback:-1].max()
    prev_low = df["low"].iloc[-lookback:-1].min()
    last = df.iloc[-1]
    if last["low"] < prev_low and last["close"] > prev_low:
        return "Bullish Liquidity Sweep"
    if last["high"] > prev_high and last["close"] < prev_high:
        return "Bearish Liquidity Sweep"
    return None

def island_reversal(df):
    if len(df) < 5:
        return None
    a, b, c, d = df.iloc[-4], df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if b["low"] > a["high"] and d["open"] < c["low"]:
        return "Bullish Island Reversal"
    if b["high"] < a["low"] and d["open"] > c["high"]:
        return "Bearish Island Reversal"
    return None

def wyckoff_spring_upthrust(df, lookback=30):
    if len(df) < lookback + 2:
        return None
    range_high = df["high"].iloc[-lookback:-1].max()
    range_low = df["low"].iloc[-lookback:-1].min()
    last = df.iloc[-1]
    if last["low"] < range_low and last["close"] > range_low:
        return "Wyckoff Spring (Bullish)"
    if last["high"] > range_high and last["close"] < range_high:
        return "Wyckoff Upthrust (Bearish)"
    return None

def smart_money_trap(df):
    if len(df) < 3:
        return None
    prev, last = df.iloc[-2], df.iloc[-1]
    if prev["close"] > prev["high"] * 0.99 and last["close"] < prev["low"]:
        return "Bull Trap Reversal"
    if prev["close"] < prev["low"] * 1.01 and last["close"] > prev["high"]:
        return "Bear Trap Reversal"
    return None

def bump_and_run_reversal(df):
    if len(df) < 40:
        return None
    slope1 = (df["close"].iloc[-30] - df["close"].iloc[-40]) / 10
    slope2 = (df["close"].iloc[-1] - df["close"].iloc[-10]) / 10
    if slope2 > slope1 * 2 and df["close"].iloc[-1] < df["close"].iloc[-5]:
        return "BARR Top Reversal"
    if slope2 < slope1 * 2 and df["close"].iloc[-1] > df["close"].iloc[-5]:
        return "BARR Bottom Reversal"
    return None

def exhaustion_bar(df):
    if len(df) < 20:
        return None
    avg_range = (df["high"] - df["low"]).rolling(10).mean().iloc[-2]
    last = df.iloc[-1]
    if (last["high"] - last["low"]) > 2 * avg_range:
        return "Bearish Exhaustion" if last["close"] < last["open"] else "Bullish Exhaustion"
    return None

def shakeout_trap(df, lookback=20):
    if len(df) < lookback + 2:
        return None
    high = df["high"].iloc[-lookback:-1].max()
    low = df["low"].iloc[-lookback:-1].min()
    prev, last = df.iloc[-2], df.iloc[-1]
    if prev["low"] < low and last["close"] > low:
        return "Bullish Shakeout"
    if prev["high"] > high and last["close"] < high:
        return "Bearish Shakeout"
    return None

def hidden_pivot_reversal(df, lookback=25):
    if len(df) < lookback:
        return None
    highs, lows = df["high"].iloc[-lookback:], df["low"].iloc[-lookback:]
    if highs.iloc[-1] > highs.iloc[:-1].max() and df["close"].iloc[-1] < highs.iloc[:-1].max():
        return "Hidden Pivot Bearish Reversal"
    if lows.iloc[-1] < lows.iloc[:-1].min() and df["close"].iloc[-1] > lows.iloc[:-1].min():
        return "Hidden Pivot Bullish Reversal"
    return None

def springer_reversal(df, lookback=25):
    if len(df) < lookback + 5:
        return None
    support = df["low"].iloc[-lookback:-5].min()
    recent = df.iloc[-1]
    if recent["low"] < support and recent["close"] > support:
        return "Springer Reversal (Bullish)"
    return None

def calculate_confluence(row):
    score = 0
    text = " ".join([
        str(row.get("Signal", "")),
        str(row.get("Trend", "")),
        str(row.get("State", "")),
        str(row.get("Setup", "")),
        str(row.get("Divergence", "")),
    ])
    
    for k in BULL_KEYWORDS:
        if k in text:
            score += 1
    for k in BEAR_KEYWORDS:
        if k in text:
            score -= 1
            
    score = max(min(score, 5), -5)
    bias = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"
    abs_score = abs(score)
    prob = "High" if abs_score >= 4 else "Medium" if abs_score >= 3 else "Low"
    return score, bias, prob

def run_all_scanners_for_symbol(sym, df, tf, analysis_date, data_all_tfs):
    results = {
        "RSI Market Pulse": rsi_market_pulse(df) is not None,
        "Volume Shocker": volume_shocker(df),
        "NRB-7 Breakout": nrb_7(df) is not None,
        "Counter Attack": counter_attack(df) is not None,
        "Breakaway Gaps": breakaway_gap(df) is not None,
        "RSI + ADX": rsi_adx(df) is not None,
        "MACD Market Pulse": macd_market_pulse(df) is not None,
        "MACD Normal Divergence": macd_normal_divergence(df) is not None,
        "MACD Bearish Peak Divergence": macd_peak_bearish_divergence(df) is not None,
        "MACD Bullish Base Divergence": macd_base_bullish_divergence(df) is not None,
        "Trend Alignment (EMA)": trend_alignment(df) is not None,
        "Pullback to EMA": pullback_to_ema(df) is not None,
        "High Probability Confluence": confluence_setup(df) is not None,
        "MACD Hook Up": macd_hook_up(df) is not None,
        "MACD Hook Down": macd_hook_down(df) is not None,
        "MACD Histogram Divergence": macd_histogram_divergence(df) is not None,
        "EMA50 + Stoch Oversold": ema50_stoch_oversold(df) is not None,
        "Dark Cloud Cover": dark_cloud_cover(df) is not None,
        "Morning Star (Bottom)": morning_star_bottom(df) is not None,
        "Evening Star (Top)": evening_star_top(df) is not None,
        "50 EMA Fake Breakdown": ema50_fake_breakdown(df) is not None,
        "50 EMA Fake Breakout": ema50_fake_breakout(df) is not None,
        "KDJ BUY (Oversold)": kdj_buy(df) is not None,
        "KDJ SELL (Overbought)": kdj_sell(df) is not None,
        "Probable Momentum (Consecutive Close)": consecutive_close_momentum(df, min_count=3) is not None,
        "Camarilla Breakout / Breakdown": camarilla_breakout(df) is not None,
        "CPR Breakout / Breakdown": cpr_breakout(df) is not None,
        "Inside Bar Breakout": inside_bar_breakout(df) is not None,
        "ADX Expansion (Trend Ignition)": adx_expansion(df) is not None,
        "Range Expansion Day": range_expansion_day(df) is not None,
        "Failed Breakout / Breakdown": failed_breakout_breakdown(df) is not None,
        "EMA Compression → Expansion": ema_compression_expansion(df) is not None,
        "RSI Swing": rsi_swing(df) is not None,
    }

    if "Weekly" in data_all_tfs and "Monthly" in data_all_tfs:
        data_w, data_m = data_all_tfs["Weekly"], data_all_tfs["Monthly"]
        if sym in data_w and sym in data_m:
            df_w = trim_df_to_date(data_w[sym], analysis_date)
            df_m = trim_df_to_date(data_m[sym], analysis_date)
            results["RSI WM 60–40"] = rsi_wm(df, df_w, df_m) is not None if df_w is not None and df_m is not None else False
        else:
            results["RSI WM 60–40"] = False
    else:
        results["RSI WM 60–40"] = False

    htf_map = {"15 Min": "1 Hour", "1 Hour": "Daily", "Daily": "Weekly", "Weekly": "Monthly"}
    data_htf = data_all_tfs.get(htf_map.get(tf, ""))

    if data_htf and sym in data_htf:
        df_htf = trim_df_to_date(data_htf[sym], analysis_date)
        if df_htf is not None:
            results["MACD RD (4th Wave)"] = macd_rd(df, df_htf) is not None
            results["Bullish GSAS"] = bullish_gsas(df, df_htf) is not None
            results["Bearish GSAS"] = bearish_gsas(df, df_htf) is not None
        else:
            results["MACD RD (4th Wave)"] = False
            results["Bullish GSAS"] = False
            results["Bearish GSAS"] = False
    else:
        results["MACD RD (4th Wave)"] = False
        results["Bullish GSAS"] = False
        results["Bearish GSAS"] = False

    results["Probable 3rd Wave"] = third_wave_finder(df)
    results["Probable C Wave"] = c_wave_finder(df)
    results["Top 10 by ATR %"] = atr_percent(df) is not None

    return results

# ======================================
# SCANNER TILE CONFIGURATION & SELECTION
# ======================================
SCANNERS = [
    {"name": "RSI Market Pulse", "color": "#1abc9c"},
    {"name": "Volume Shocker", "color": "#1abc9c"},
    {"name": "NRB-7 Breakout", "color": "#1abc9c"},
    {"name": "Counter Attack", "color": "#1abc9c"},
    {"name": "Breakaway Gaps", "color": "#e67e22"},
    {"name": "RSI + ADX", "color": "#e67e22"},
    {"name": "RSI WM 60–40", "color": "#e67e22"},
    {"name": "MACD Market Pulse", "color": "#e67e22"},
    {"name": "MACD Normal Divergence", "color": "#f1c40f"},
    {"name": "MACD RD (4th Wave)", "color": "#f1c40f"},
    {"name": "Probable 3rd Wave", "color": "#f1c40f"},
    {"name": "Probable C Wave", "color": "#f1c40f"},
    {"name": "MACD Bearish Peak Divergence", "color": "#3498db"},
    {"name": "MACD Bullish Base Divergence", "color": "#3498db"},
    {"name": "Trend Alignment (EMA)", "color": "#3498db"},
    {"name": "Pullback to EMA", "color": "#3498db"},
    {"name": "High Probability Confluence", "color": "#e84393"},
    {"name": "MACD Hook Up", "color": "#e84393"},
    {"name": "MACD Hook Down", "color": "#e84393"},
    {"name": "MACD Histogram Divergence", "color": "#e84393"},
    {"name": "EMA50 + Stoch Oversold", "color": "#f1c40f"},
    {"name": "Dark Cloud Cover", "color": "#f1c40f"},
    {"name": "Morning Star (Bottom)", "color": "#f1c40f"},
    {"name": "Evening Star (Top)", "color": "#f1c40f"},
    {"name": "Bullish GSAS", "color": "#27ae60"},
    {"name": "Bearish GSAS", "color": "#27ae60"},
    {"name": "50 EMA Fake Breakdown", "color": "#27ae60"},
    {"name": "50 EMA Fake Breakout", "color": "#27ae60"},
    {"name": "KDJ BUY (Oversold)", "color": "#f39c12"},
    {"name": "KDJ SELL (Overbought)", "color": "#f39c12"},
    {"name": "Probable Momentum (Consecutive Close)", "color": "#f39c12"},
    {"name": "Camarilla Breakout / Breakdown", "color": "#f39c12"},
    {"name": "CPR Breakout / Breakdown", "color": "#e67e22"},
    {"name": "Inside Bar Breakout", "color": "#e67e22"},
    {"name": "ADX Expansion (Trend Ignition)", "color": "#e67e22"},
    {"name": "Range Expansion Day", "color": "#e67e22"},
    {"name": "Failed Breakout / Breakdown", "color": "#34495e"},
    {"name": "EMA Compression → Expansion", "color": "#34495e"},
    {"name": "Top 10 by ATR %", "color": "#34495e"},
    {"name": "Liquidity Sweep Reversal", "color": "#34495e"},
    {"name": "Island Reversal", "color": "#ff6b81"},
    {"name": "Wyckoff Spring / Upthrust", "color": "#ff6b81"},
    {"name": "Smart Money Trap", "color": "#ff6b81"},
    {"name": "Bump & Run Reversal", "color": "#ff6b81"},
    {"name": "Exhaustion Bar", "color": "#3498db"},
    {"name": "Shakeout / Trap", "color": "#3498db"},
    {"name": "Hidden Pivot Reversal", "color": "#3498db"},
    {"name": "Springer Reversal", "color": "#3498db"},
    {"name": "RSI + MACD Cross Swing", "color": "#9b59b6"},
    {"name": "RSI Swing", "color": "#8e44ad"},
]

if "scanner" not in st.session_state:
    st.session_state["scanner"] = SCANNERS[0]["name"]

st.markdown("### 🎯 Select Scanner")

cols_per_row = 4
clicked_scanner = None

for i in range(0, len(SCANNERS), cols_per_row):
    row = SCANNERS[i : i + cols_per_row]
    cols = st.columns(len(row))
    for col, sc in zip(cols, row):
        with col:
            is_active = st.session_state["scanner"] == sc["name"]
            bg = sc["color"]
            border = "#ffffff" if not is_active else "#000000"
            opacity = "1.0" if is_active else "0.85"

            st.markdown(
                f"""
                <div style="
                    border-radius: 12px;
                    padding: 14px 10px;
                    text-align: center;
                    background: {bg};
                    border: 3px solid {border};
                    box-shadow: 0 3px 8px rgba(0,0,0,0.35);
                    opacity: {opacity};
                    margin-bottom: 6px;
                ">
                    <span style="font-weight: 700; font-size: 14px; color: white;">
                        {sc["name"]}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Scan: {sc['name']}", key=f"btn_{sc['name']}"):
                clicked_scanner = sc["name"]

if clicked_scanner is not None:
    st.session_state["scanner"] = clicked_scanner

scanner = st.session_state["scanner"]
st.markdown(f"**Active Scanner:** `{scanner}`  |  **Timeframe:** `{tf}`")

run = clicked_scanner is not None

# ======================================
# EXECUTION ENGINE
# ======================================
if run:
    data = load_data(TIMEFRAMES[tf])
    if not data:
        st.warning("No data found.")
        st.stop()

    results = []
    atr_list = []

    htf_map = {"15 Min": "1 Hour", "1 Hour": "Daily", "Daily": "Weekly", "Weekly": "Monthly"}
    data_htf = load_data(TIMEFRAMES[htf_map[tf]]) if scanner in ["Bullish GSAS", "Bearish GSAS", "MACD RD (4th Wave)"] and tf in htf_map else None

    if scanner == "RSI WM 60–40":
        data_w = load_data(TIMEFRAMES["Weekly"])
        data_m = load_data(TIMEFRAMES["Monthly"])

    for sym, df in data.items():
        df = trim_df_to_date(df, analysis_date)
        if df is None:
            continue

        base_row = {
            "Symbol": sym,
            "Signal": "",
            "Trend": "",
            "State": "",
            "Setup": "",
            "Divergence": "",
            "RSI": "",
            "Zone": "",
            "Confluence": 0,
            "Bias": "",
            "Probability": "",
            "TV_Link": make_tradingview_link(sym),
        }

        if scanner == "Top 10 by ATR %":
            v = atr_percent(df)
            if v is not None:
                row = base_row.copy()
                row["Signal"] = "High ATR %"
                row["State"] = f"{v:.2f}%"
                atr_list.append((sym, v, row))
            continue

        if scanner == "RSI Market Pulse":
            r = rsi_market_pulse(df)
            if r:
                row = base_row.copy()
                row["RSI"], row["Zone"] = r[0], r[1]
                results.append(row)
        elif scanner == "Volume Shocker" and volume_shocker(df):
            row = base_row.copy()
            row["Signal"] = "Volume Shocker"
            results.append(row)
        elif scanner == "NRB-7 Breakout":
            sig = nrb_7(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Counter Attack":
            sig = counter_attack(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Breakaway Gaps":
            sig = breakaway_gap(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "RSI + ADX":
            sig = rsi_adx(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "RSI WM 60–40":
            if sym in data_w and sym in data_m:
                df_wt = trim_df_to_date(data_w[sym], analysis_date)
                df_mt = trim_df_to_date(data_m[sym], analysis_date)
                if df_wt is not None and df_mt is not None:
                    sig = rsi_wm(df, df_wt, df_mt)
                    if sig:
                        row = base_row.copy()
                        row["Signal"] = sig
                        results.append(row)
        elif scanner == "MACD Market Pulse":
            sig = macd_market_pulse(df)
            if sig:
                row = base_row.copy()
                row["State"] = sig
                results.append(row)
        elif scanner == "MACD Normal Divergence":
            sig = macd_normal_divergence(df)
            if sig:
                row = base_row.copy()
                row["Divergence"] = sig
                results.append(row)
        elif scanner == "MACD RD (4th Wave)":
            if data_htf and sym in data_htf:
                df_htf = trim_df_to_date(data_htf[sym], analysis_date)
                if df_htf is not None:
                    sig = macd_rd(df, df_htf)
                    if sig:
                        row = base_row.copy()
                        row["Signal"] = sig
                        results.append(row)
        elif scanner == "Probable 3rd Wave" and third_wave_finder(df):
            row = base_row.copy()
            row["Signal"] = "Probable 3rd Wave"
            results.append(row)
        elif scanner == "Probable C Wave" and c_wave_finder(df):
            row = base_row.copy()
            row["Signal"] = "Probable C Wave"
            results.append(row)
        elif scanner == "MACD Bearish Peak Divergence":
            sig = macd_peak_bearish_divergence(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = row["Divergence"] = sig
                results.append(row)
        elif scanner == "MACD Bullish Base Divergence":
            sig = macd_base_bullish_divergence(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = row["Divergence"] = sig
                results.append(row)
        elif scanner == "Trend Alignment (EMA)":
            sig = trend_alignment(df)
            if sig:
                row = base_row.copy()
                row["Trend"] = sig
                results.append(row)
        elif scanner == "Pullback to EMA":
            sig = pullback_to_ema(df)
            if sig:
                row = base_row.copy()
                row["Setup"] = sig
                results.append(row)
        elif scanner == "High Probability Confluence":
            sig = confluence_setup(df)
            if sig:
                row = base_row.copy()
                row["Setup"] = sig
                results.append(row)
        elif scanner == "MACD Hook Up":
            sig = macd_hook_up(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "MACD Hook Down":
            sig = macd_hook_down(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "MACD Histogram Divergence":
            sig = macd_histogram_divergence(df)
            if sig:
                row = base_row.copy()
                row["Divergence"] = sig
                results.append(row)
        elif scanner == "EMA50 + Stoch Oversold":
            sig = ema50_stoch_oversold(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Dark Cloud Cover":
            sig = dark_cloud_cover(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Morning Star (Bottom)":
            sig = morning_star_bottom(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Evening Star (Top)":
            sig = evening_star_top(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner in ["Bullish GSAS", "Bearish GSAS"]:
            if data_htf and sym in data_htf:
                df_htf = trim_df_to_date(data_htf[sym], analysis_date)
                if df_htf is not None:
                    sig = bullish_gsas(df, df_htf) if scanner == "Bullish GSAS" else bearish_gsas(df, df_htf)
                    if sig:
                        row = base_row.copy()
                        row["Signal"] = sig
                        results.append(row)
        elif scanner == "50 EMA Fake Breakdown":
            sig = ema50_fake_breakdown(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "50 EMA Fake Breakout":
            sig = ema50_fake_breakout(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "KDJ BUY (Oversold)":
            sig = kdj_buy(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "KDJ SELL (Overbought)":
            sig = kdj_sell(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Probable Momentum (Consecutive Close)":
            res = consecutive_close_momentum(df, min_count=3)
            if res:
                direction, days = res
                row = base_row.copy()
                row["Signal"] = f"{direction} Momentum"
                row["State"] = f"{days} Consecutive Days"
                results.append(row)
        elif scanner == "Camarilla Breakout / Breakdown":
            sig = camarilla_breakout(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "CPR Breakout / Breakdown":
            sig = cpr_breakout(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Inside Bar Breakout":
            sig = inside_bar_breakout(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "ADX Expansion (Trend Ignition)":
            sig = adx_expansion(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Range Expansion Day":
            sig = range_expansion_day(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Failed Breakout / Breakdown":
            sig = failed_breakout_breakdown(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Liquidity Sweep Reversal":
            sig = liquidity_sweep_reversal(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Island Reversal":
            sig = island_reversal(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Wyckoff Spring / Upthrust":
            sig = wyckoff_spring_upthrust(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Smart Money Trap":
            sig = smart_money_trap(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Bump & Run Reversal":
            sig = bump_and_run_reversal(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Exhaustion Bar":
            sig = exhaustion_bar(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Shakeout / Trap":
            sig = shakeout_trap(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "RSI + MACD Cross Swing":
            sig = rsi_macd_cross_swing(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Hidden Pivot Reversal":
            sig = hidden_pivot_reversal(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "Springer Reversal":
            sig = springer_reversal(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "EMA Compression → Expansion":
            sig = ema_compression_expansion(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)
        elif scanner == "RSI Swing":
            sig = rsi_swing(df)
            if sig:
                row = base_row.copy()
                row["Signal"] = sig
                results.append(row)

    if scanner == "Top 10 by ATR %":
        if not atr_list:
            st.info("No symbols with valid ATR %.")
            df_res = empty_result_df()
        else:
            atr_list.sort(key=lambda x: x[1], reverse=True)
            results = [r[2] for r in atr_list[:10]]

    if not results:
        st.info("No stocks matched.")
        df_res = empty_result_df()
    else:
        df_res = pd.DataFrame(results)

        for c in SAFE_COLS:
            if c not in df_res.columns:
                df_res[c] = "" if c != "Confluence" else 0

        for i, row in df_res.iterrows():
            score, bias, prob = calculate_confluence(row)
            df_res.at[i, "Confluence"] = score
            df_res.at[i, "Bias"] = bias
            df_res.at[i, "Probability"] = prob

        bias_rank = {"Bullish": 0, "Neutral": 1, "Bearish": 2}
        df_res["_bias_rank"] = df_res["Bias"].map(bias_rank)
        df_res = df_res.sort_values(
            by=["Confluence", "_bias_rank"], ascending=[False, True]
        ).drop(columns="_bias_rank")

        df_res = df_res[SAFE_COLS]
        df_res = df_res.replace([np.inf, -np.inf], "").fillna("")

        st.dataframe(
            df_res,
            use_container_width=True,
            hide_index=True,
            column_config={
                "TV_Link": st.column_config.LinkColumn(
                    "TradingView", display_text="Open Chart"
                )
            }
        )

        # Plotly Donut Chart Rendering
        if scanner == "RSI Market Pulse" and not df_res.empty:
            df_res['Zone'] = df_res['Zone'].astype(str).str.strip()
            df_filtered = df_res[df_res['Zone'].isin(["RSI > 60", "RSI 40–60", "RSI < 40"])]
            
            if not df_filtered.empty:
                zone_counts = df_filtered['Zone'].value_counts().reset_index()
                zone_counts.columns = ['Zone', 'Count']

                fig = px.pie(
                    zone_counts,
                    names='Zone',
                    values='Count',
                    title="🎯 RSI Market Pulse Distribution",
                    hole=0.5,
                    color="Zone",
                    color_discrete_map={
                        "RSI > 60": "#2ecc71",
                        "RSI 40–60": "#f1c40f",
                        "RSI < 40": "#e74c3c",
                    },
                )
                fig.update_traces(
                    textinfo="percent+value", 
                    textfont_size=13,
                    marker=dict(line=dict(color='#FFFFFF', width=2))
                )
                fig.update_layout(
                    showlegend=True,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### 🧾 Scanner Matrix for Selected Stock")

if selected_symbol != "NA":
    data_single_tf = load_data(TIMEFRAMES[tf])
    if selected_symbol in data_single_tf:
        df_sym = trim_df_to_date(data_single_tf[selected_symbol], analysis_date)
        if df_sym is not None:
            data_all_tfs = {
                tf: data_single_tf,
                "1 Hour": load_data(TIMEFRAMES["1 Hour"]),
                "Daily": load_data(TIMEFRAMES["Daily"]),
                "Weekly": load_data(TIMEFRAMES["Weekly"]),
                "Monthly": load_data(TIMEFRAMES["Monthly"]),
            }

            results_dict = run_all_scanners_for_symbol(
                selected_symbol,
                df_sym,
                tf,
                analysis_date,
                data_all_tfs,
            )

            mat_df = pd.DataFrame(
                {
                    "Scanner": list(results_dict.keys()),
                    "Result": ["Yes" if v else "No" for v in results_dict.values()],
                }
            )
            st.dataframe(mat_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data for this symbol at selected date.")
    else:
        st.info("Symbol data not found for this timeframe.")

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    
<div style="line-height: 1.6;">
<b>Designed by:-<br>
Gaurav Singh Yaadav</b><br><br>
    
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

show_footer()
