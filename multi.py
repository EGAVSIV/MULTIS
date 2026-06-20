import os
import sys
import base64
import hashlib

import numpy as np
import pandas as pd
import talib
import plotly.express as px
import streamlit as st

BASE_PATH = os.path.dirname(__file__)

# --- Python 3.13 image hack ---
if sys.version_info >= (3, 13):
    import types
    imghdr = types.ModuleType("imghdr")
    imghdr.what = lambda *args, **kwargs: None
    sys.modules["imghdr"] = imghdr

# Crucial: Must be the very first Streamlit command called
st.set_page_config(
    page_title="Gaurav_Singh_Yaadav",
    layout="wide",
    page_icon="🧮"
)

# ==============================
# GLOBAL CONFIG
# ==============================
SAFE_COLS = [
    "Symbol", "Signal", "Trend", "State", "Setup", 
    "Divergence", "RSI", "Zone", "Confluence", "Bias", 
    "Probability", "TV_Link"
]

BULL_KEYWORDS = ["Bullish", "BUY", "Breakout", "Uptrend", "Momentum"]
BEAR_KEYWORDS = ["Bearish", "SELL", "Breakdown", "Downtrend"]

def empty_result_df():
    return pd.DataFrame({c: [] for c in SAFE_COLS})

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

# ==============================
# LAST CANDLE TIME HELPERS
# ==============================
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

# ==============================
# UI HEADER
# ==============================
st.markdown(
    """
    <h1 style="color: blue; font-weight: 700; margin-bottom: 0.2rem;">
        📊 Multi-Timeframe Stock Screener
    </h1>
    """,
    unsafe_allow_html=True,
)

bg_path = os.path.join(BASE_PATH, "Assest", "BG11.png")
if os.path.exists(bg_path):
    set_bg_image(bg_path)

# ==============================
# DATA LOADER
# ==============================
@st.cache_data(show_spinner=False)
def load_data(folder: str):
    data = {}
    if not os.path.exists(folder):
        st.warning(f"Folder not found: {folder}")
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

TIMEFRAMES = {
    "15 Min": os.path.join(BASE_PATH, "stock_data_15"),
    "1 Hour": os.path.join(BASE_PATH, "stock_data_1H"),
    "Daily": os.path.join(BASE_PATH, "stock_data_D"),
    "Weekly": os.path.join(BASE_PATH, "stock_data_W"),
    "Monthly": os.path.join(BASE_PATH, "stock_data_M"),
}

# Sidebar Framework
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

col1, col2 = st.columns([1, 6])

with col1:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()   # Standardized safe clearing method
        st.success("Fresh data loaded")
        st.rerun()              # Updated to stable API

with col2:
    st.markdown(
        f"""
🕯 **Last Candle (IST)** ⏱ **15 Min**: {last_15m.strftime('%d %b %Y %H:%M') if last_15m else 'NA'} |  
⏰ **1 Hour**: {last_1h.strftime('%d %b %Y %H:%M') if last_1h else 'NA'} |  
📅 **Daily**: {last_d.date() if last_d else 'NA'} |  
📆 **Weekly**: {last_w.date() if last_w else 'NA'} |  
🗓 **Monthly**: {last_m.date() if last_m else 'NA'}
""",
        unsafe_allow_html=False,
    )

st.markdown("---")

# ==============================
# BACKTEST DATE
# ==============================
st.sidebar.markdown("### 📅 Backtest Date")
analysis_date = st.sidebar.date_input(
    "Select Analysis Date",
    value=last_d.date() if last_d else pd.Timestamp.today().date(),
)
st.sidebar.info(f"Backtest Mode Active\nData cutoff: {analysis_date}")
st.sidebar.caption(f"Scans will run as of: {analysis_date}")

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

# ==============================
# SCANNERS (PURE FUNCTIONS)
# ==============================
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

# [Remaining scanning functions preserved exactly as you designed them...]
def volume_shocker(df):
    if len(df) < 20: return False
    vol_sma = df["volume"].rolling(10).mean()
    last, prev = df.iloc[-1], df.iloc[-2]
    return last["volume"] > 2 * vol_sma.iloc[-1] and prev["close"] * 0.95 <= last["close"] <= prev["close"] * 1.05

def nrb_7(df):
    if len(df) < 20: return None
    base, inside, last = df.iloc[-7], df.iloc[-6:-1], df.iloc[-1]
    base_high, base_low = base["high"], base["low"]
    cond_high_low = inside["high"].max() <= base_high and inside["low"].min() >= base_low
    cond_open_close = inside["open"].max() <= base_high and inside["open"].min() >= base_low and inside["close"].max() <= base_high and inside["close"].min() >= base_low
    if not (cond_high_low and cond_open_close): return None
    avg_vol = df["volume"].rolling(10).mean().iloc[-2]
    if last["volume"] < 1.5 * avg_vol: return None
    if last["close"] > base_high: return "NRB-7 Bullish Breakout + Volume"
    if last["close"] < base_low: return "NRB-7 Bearish Breakdown + Volume"
    return None

def counter_attack(df):
    if len(df) < 2: return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    mid = (prev["open"] + prev["close"]) / 2
    if prev["close"] < prev["open"] and curr["close"] > curr["open"] and curr["open"] < prev["close"] and curr["close"] >= mid: return "Bullish"
    if prev["close"] > prev["open"] and curr["close"] < curr["open"] and curr["open"] > prev["close"] and curr["close"] <= mid: return "Bearish"
    return None

def breakaway_gap(df):
    if len(df) < 50: return None
    df = df.copy()
    df["EMA20"], df["EMA50"] = talib.EMA(df["close"], 20), talib.EMA(df["close"], 50)
    prev, curr = df.iloc[-2], df.iloc[-1]
    if curr["open"] > prev["high"] * 1.005 and curr["low"] > prev["high"] and curr["EMA20"] < curr["EMA50"]: return "Bullish Breakaway Gap"
    if curr["open"] < prev["low"] * 0.995 and curr["high"] < prev["low"] and curr["EMA20"] > curr["EMA50"]: return "Bearish Breakaway Gap"
    return None

def rsi_adx(df):
    if len(df) < 20: return None
    rsi = talib.RSI(df["close"], 14).iloc[-1]
    adx = talib.ADX(df["high"], df["low"], df["close"], 14).iloc[-1]
    if adx > 50 and rsi < 20: return "Bullish Reversal"
    if adx > 50 and rsi > 80: return "Probabale Bearish Reversal"
    return None

def rsi_wm(df_tf, df_w, df_m):
    r_tf = talib.RSI(df_tf["close"], 14).iloc[-1]
    r_w = talib.RSI(df_w["close"], 14).iloc[-1]
    r_m = talib.RSI(df_m["close"], 14).iloc[-1]
    if r_w > 60 and r_m > 60 and r_tf < 40: return "Bullish WM Reversal"
    if r_w < 40 and r_m < 40 and r_tf > 60: return "Bearish WM Reversal"
    return None

def macd_market_pulse(df):
    if len(df) < 30: return None
    macd, signal, _ = talib.MACD(df["close"], 12, 26, 9)
    m, s, pm = macd.iloc[-1], signal.iloc[-1], macd.iloc[-2]
    if m > 0 and m > s and m > pm: return "Strong Bullish"
    if m > 0 and m > s and m < pm: return "Bullish Cooling"
    if m > 0 and m < s and m > pm: return "Bullish Reversal Watch"
    if m > 0 and m < s and m < pm: return "Weak Bullish"
    if m < 0 and m > s and m > pm: return "Bearish Reversal Watch"
    if m < 0 and m > s and m < pm: return "Weak Bearish"
    if m < 0 and m < s and m > pm: return "Bearish Recovery Attempt"
    if m < 0 and m < s and m < pm: return "Strong Bearish"
    return None

def macd_normal_divergence(df, lookback=30):
    if len(df) < lookback: return None
    macd, _, _ = talib.MACD(df["close"], 12, 26, 9)
    price_low1, price_low2 = df["low"].iloc[-lookback:-15].min(), df["low"].iloc[-15:].min()
    macd_low1, macd_low2 = macd.iloc[-lookback:-15].min(), macd.iloc[-15:].min()
    if price_low2 < price_low1 and macd_low2 > macd_low1: return "Bullish ND"
    price_high1, price_high2 = df["high"].iloc[-lookback:-15].max(), df["high"].iloc[-15:].max()
    macd_high1, macd_high2 = macd.iloc[-lookback:-15].max(), macd.iloc[-15:].max()
    if price_high2 > price_high1 and macd_high2 < macd_high1: return "Bearish ND"
    return None

def macd_rd(df, df_htf):
    if len(df) < 60 or len(df_htf) < 30: return None
    macd, signal, _ = talib.MACD(df["close"], 12, 26, 9)
    latest, prev, sig = macd.iloc[-1], macd.iloc[-2], signal.iloc[-1]
    max60 = macd.rolling(60).max().iloc[-1]
    macd_htf, _, _ = talib.MACD(df_htf["close"], 12, 26, 9)
    macd_htf_val = macd_htf.iloc[-1]
    macd_htf_uptick = macd_htf.iloc[-1] > macd_htf.iloc[-2]
    ema50_ltf, ema50_htf = talib.EMA(df["close"], 50).iloc[-1], talib.EMA(df_htf["close"], 50).iloc[-1]
    if latest > prev and latest > 0 and sig < latest and macd_htf_val > 0 and max60 > 0 and (latest / max60) < 0.25 and df["close"].iloc[-1] > ema50_ltf and df_htf["close"].iloc[-1] > ema50_htf and macd_htf_uptick:
        return "MACD RD (Compression + Trend Aligned)"
    return None

def third_wave_finder(df, lookback_cross=50, tolerance=0.02):
    if len(df) < lookback_cross + 5: return False
    close = df["close"]
    ema20, ema50 = talib.EMA(close, 20), talib.EMA(close, 50)
    cross_idx = None
    for i in range(len(close) - 1, 0, -1):
        if ema20.iloc[i] > ema50.iloc[i] and ema20.iloc[i - 1] <= ema50.iloc[i - 1]:
            cross_idx = i
            break
    if cross_idx is None: return False
    pre_ema20 = ema20.iloc[max(0, cross_idx - lookback_cross):cross_idx]
    pre_ema50 = ema50.iloc[max(0, cross_idx - lookback_cross):cross_idx]
    if pre_ema20.empty or (pre_ema20 < pre_ema50).mean() < 0.7: return False
    ema50_now, price_now = ema50.iloc[-1], close.iloc[-1]
    if ema50_now == 0 or np.isnan(ema50_now) or abs(price_now - ema50_now) / ema50_now > tolerance or ema20.iloc[-1] <= ema50.iloc[-1]: return False
    return True

def c_wave_finder(df, lookback_cross=50, tolerance=0.02):
    if len(df) < lookback_cross + 5: return False
    close = df["close"]
    ema20, ema50 = talib.EMA(close, 20), talib.EMA(close, 50)
    cross_idx = None
    for i in range(len(close) - 1, 0, -1):
        if ema20.iloc[i] < ema50.iloc[i] and ema20.iloc[i - 1] >= ema50.iloc[i - 1]:
            cross_idx = i
            break
    if cross_idx is None: return False
    pre_ema20 = ema20.iloc[max(0, cross_idx - lookback_cross):cross_idx]
    pre_ema50 = ema50.iloc[max(0, cross_idx - lookback_cross):cross_idx]
    if pre_ema20.empty or (pre_ema20 > pre_ema50).mean() < 0.7: return False
    ema50_now, price_now = ema50.iloc[-1], close.iloc[-1]
    if ema50_now == 0 or np.isnan(ema50_now) or abs(price_now - ema50_now) / ema50_now > tolerance or ema20.iloc[-1] >= ema50.iloc[-1]: return False
    return True

def macd_peak_bearish_divergence(df):
    if len(df) < 80: return None
    macd, _, _ = talib.MACD(df["close"], 12, 26, 9)
    price_high1 = df["high"].iloc[-60:-30].max()
    price_high2 = df["high"].iloc[-30:].max()
    idx1, idx2 = df["high"].iloc[-60:-30].idxmax(), df["high"].iloc[-30:].idxmax()
    if price_high2 > price_high1 and macd.loc[idx2] < macd.loc[idx1]: return "Bearish MACD Peak Divergence"
    return None

def macd_base_bullish_divergence(df):
    if len(df) < 80: return None
    macd, _, _ = talib.MACD(df["close"], 12, 26, 9)
    if df["low"].iloc[-30:].min() < df["low"].iloc[-60:-30].min() and macd.iloc[-30:].min() > macd.iloc[-60:-30].min(): return "Bullish MACD Base Divergence"
    return None

def trend_alignment(df):
    if len(df) < 100: return None
    ema20, ema50, ema100 = talib.EMA(df["close"], 20), talib.EMA(df["close"], 50), talib.EMA(df["close"], 100)
    if ema20.iloc[-1] > ema50.iloc[-1] > ema100.iloc[-1]: return "Strong Uptrend"
    if ema20.iloc[-1] < ema50.iloc[-1] < ema100.iloc[-1]: return "Strong Downtrend"
    return None

def pullback_to_ema(df):
    if len(df) < 60: return None
    ema20, ema50 = talib.EMA(df["close"], 20), talib.EMA(df["close"], 50)
    last = df.iloc[-1]
    if ema20.iloc[-1] > ema50.iloc[-1] and last["low"] <= ema20.iloc[-1] and last["close"] > ema20.iloc[-1]: return "Bullish EMA Pullback"
    if ema20.iloc[-1] < ema50.iloc[-1] and last["high"] >= ema20.iloc[-1] and last["close"] < ema20.iloc[-1]: return "Bearish EMA Pullback"
    return None

def confluence_setup(df):
    if len(df) < 60: return None
    rsi = talib.RSI(df["close"], 14).iloc[-1]
    macd, sig, _ = talib.MACD(df["close"], 12, 26, 9)
    ema20, ema50 = talib.EMA(df["close"], 20), talib.EMA(df["close"], 50)
    if rsi > 50 and macd.iloc[-1] > sig.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]: return "Bullish Confluence"
    if rsi < 50 and macd.iloc[-1] < sig.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]: return "Bearish Confluence"
    return None

def macd_hook_up(df):
    if len(df) < 35: return None
    macd, signal, hist = talib.MACD(df["close"], 12, 26, 9)
    if macd.iloc[-1] > 0 and macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-2] < macd.iloc[-3] and macd.iloc[-1] > macd.iloc[-2] and hist.iloc[-1] > hist.iloc[-2]: return "MACD Hook Up"
    return None

def macd_hook_down(df):
    if len(df) < 35: return None
    macd, signal, hist = talib.MACD(df["close"], 12, 26, 9)
    if macd.iloc[-1] < 0 and macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-2] > macd.iloc[-3] and macd.iloc[-1] < macd.iloc[-2] and hist.iloc[-1] < hist.iloc[-2]: return "MACD Hook Down"
    return None

def macd_histogram_divergence(df):
    if len(df) < 50: return None
    _, _, hist = talib.MACD(df["close"], 12, 26, 9)
    if df["low"].iloc[-20:].min() < df["low"].iloc[-40:-20].min() and hist.iloc[-20:].min() > hist.iloc[-40:-20].min(): return "Bullish Histogram Divergence"
    if df["high"].iloc[-20:].max() > df["high"].iloc[-40:-20].max() and hist.iloc[-20:].max() < hist.iloc[-40:-20].max(): return "Bearish Histogram Divergence"
    return None

def ema50_stoch_oversold(df):
    if len(df) < 50: return None
    ema50 = talib.EMA(df["close"], 50)
    slowk, slowd = talib.STOCH(df["high"], df["low"], df["close"], fastk_period=14, slowk_period=3, slowd_period=3)
    if abs(df["close"].iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1] <= 0.005 and slowk.iloc[-2] < slowd.iloc[-2] and slowk.iloc[-1] > slowd.iloc[-1] and slowk.iloc[-1] < 20: return "EMA50 + Stoch Oversold Buy"
    return None

def dark_cloud_cover(df):
    if len(df) < 15: return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    if prev["close"] <= prev["open"] or talib.RSI(df["close"], 14).iloc[-2] <= 60 or curr["close"] >= curr["open"] or curr["open"] <= prev["close"] or curr["close"] >= (prev["open"] + prev["close"]) / 2: return None
    return "Dark Cloud Cover (Bearish | RSI>60)"

def morning_star_bottom(df):
    if len(df) < 60 or df["close"].iloc[-1] > talib.EMA(df["close"], 50).iloc[-1]: return None
    if talib.CDLMORNINGSTAR(df["open"], df["high"], df["low"], df["close"]).iloc[-1] > 0: return "Morning Star (Bottom)"
    return None

def evening_star_top(df):
    if len(df) < 60 or df["close"].iloc[-1] < talib.EMA(df["close"], 50).iloc[-1]: return None
    if talib.CDLEVENINGSTAR(df["open"], df["high"], df["low"], df["close"]).iloc[-1] < 0: return "Evening Star (Top)"
    return None

def bullish_gsas(df_tf, df_htf):
    rsi = talib.RSI(df_tf["close"], 14)
    adx = talib.ADX(df_tf["high"], df_tf["low"], df_tf["close"], 14)
    ubb, _, _ = talib.BBANDS(df_tf["close"], 20)
    macd_htf, sig_htf, _ = talib.MACD(df_htf["close"], 12, 26, 9)
    ema20_htf = talib.EMA(df_htf["close"], 20)
    if rsi.iloc[-1] > 60 and ubb.iloc[-1] > ubb.iloc[-2] and adx.iloc[-1] > adx.iloc[-2] and adx.iloc[-2] < adx.iloc[-3] and macd_htf.iloc[-1] > sig_htf.iloc[-1] and df_htf["close"].iloc[-1] > ema20_htf.iloc[-1]: return "Bullish GSAS"
    return None

def bearish_gsas(df_tf, df_htf):
    rsi = talib.RSI(df_tf["close"], 14)
    adx = talib.ADX(df_tf["high"], df_tf["low"], df_tf["close"], 14)
    _, _, lbb = talib.BBANDS(df_tf["close"], 20)
    macd_htf, sig_htf, _ = talib.MACD(df_htf["close"], 12, 26, 9)
    ema20_htf = talib.EMA(df_htf["close"], 20)
    if rsi.iloc[-1] < 60 and lbb.iloc[-1] < lbb.iloc[-2] and adx.iloc[-1] > adx.iloc[-2] and adx.iloc[-2] < adx.iloc[-3] and macd_htf.iloc[-1] < sig_htf.iloc[-1] and df_htf["close"].iloc[-1] < ema20_htf.iloc[-1]: return "Bearish GSAS"
    return None

def rsi_swing(df):
    if len(df) < 20: return None
    rsi = talib.RSI(df["close"], 14)
    if rsi.iloc[-2] < 40 and rsi.iloc[-1] > 40: return "RSI Bullish Swing"
    if rsi.iloc[-2] > 60 and rsi.iloc[-1] < 60: return "RSI Bearish Swing"
    return None

def ema50_fake_breakdown(df):
    if len(df) < 55: return None
    df = df.copy()
    df["EMA20"], df["EMA50"] = talib.EMA(df["close"], 20), talib.EMA(df["close"], 50)
    if df["close"].iloc[-1] > df["EMA50"].iloc[-1] and df["close"].iloc[-2] < df["EMA50"].iloc[-2] and df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]: return "50 EMA Fake Breakdown"
    return None

def ema50_fake_breakout(df):
    if len(df) < 55: return None
    df = df.copy()
    df["EMA20"], df["EMA50"] = talib.EMA(df["close"], 20), talib.EMA(df["close"], 50)
    if df["close"].iloc[-1] < df["EMA50"].iloc[-1] and df["close"].iloc[-2] > df["EMA50"].iloc[-2] and df["EMA20"].iloc[-1] < df["EMA50"].iloc[-1]: return "50 EMA Fake Breakout"
    return None

def kdj(df, period=9, signal=3):
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    rng = (high_max - low_min).replace(0, np.nan)
    rsv = 100 * (df["close"] - low_min) / rng
    rsv = rsv.clip(lower=0, upper=100)
    def bcwsma(series, length, m=1):
        out = []
        for i, val in enumerate(series):
            if i == 0 or np.isnan(val): out.append(val)
            else: out.append((m * val + (length - m) * out[i - 1]) / length)
        return pd.Series(out, index=series.index)
    pK = bcwsma(rsv, signal, 1)
    pD = bcwsma(pK, signal, 1)
    pJ = 3 * pK - 2 * pD
    return pK, pD, pJ

def kdj_buy(df):
    if len(df) < 20: return None
    pK, pD, pJ = kdj(df)
    if pd.isna(pD.iloc[-1]) or pd.isna(pJ.iloc[-1]): return None
    if pJ.iloc[-2] < pD.iloc[-2] and pJ.iloc[-1] > pD.iloc[-1] and pD.iloc[-1] < 30: return "KDJ BUY (J↑D oversold)"
    return None

def kdj_sell(df):
    if len(df) < 20: return None
    pK, pD, pJ = kdj(df)
    if pd.isna(pD.iloc[-1]) or pd.isna(pJ.iloc[-1]): return None
    if pJ.iloc[-2] > pD.iloc[-2] and pJ.iloc[-1] < pD.iloc[-1] and pD.iloc[-1] > 70: return "KDJ SELL (J↓D overbought)"
    return None

def consecutive_close_momentum(df, min_count=3):
    if len(df) < min_count + 1: return None
    closes = df["close"].values
    direction = "Bull" if closes[-1] > closes[-2] else "Bear" if closes[-1] < closes[-2] else None
    if not direction: return None
    count = 1
    for i in range(len(closes) - 2, 0, -1):
        if direction == "Bull" and closes[i] > closes[i - 1]: count += 1
        elif direction == "Bear" and closes[i] < closes[i - 1]: count += 1
        else: break
    if count >= min_count: return direction, count
    return None

def camarilla_breakout(df):
    if len(df) < 2: return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    rng = prev["high"] - prev["low"]
    if curr["close"] > prev["close"] + (rng * 1.1 / 2): return "Bullish Camarilla Breakout"
    if curr["close"] < prev["close"] - (rng * 1.1 / 2): return "Bearish Camarilla Breakdown"
    return None

def cpr_breakout(df):
    if len(df) < 2: return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    pivot = (prev["high"] + prev["low"] + prev["close"]) / 3
    bc = (prev["high"] + prev["low"]) / 2
    tc = (pivot * 2) - bc
    if curr["close"] > max(tc, bc): return "Bullish CPR Breakout"
    if curr["close"] < min(tc, bc): return "Bearish CPR Breakdown"
    return None

def inside_bar_breakout(df):
    if len(df) < 4: return None
    mother, inside1, inside2, curr = df.iloc[-4], df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if inside1["high"] < mother["high"] and inside1["low"] > mother["low"] and inside2["high"] < mother["high"] and inside2["low"] > mother["low"]:
        if curr["close"] > mother["high"]: return "Bullish Inside Bar Breakout (3-bar coil)"
        if curr["close"] < mother["low"]: return "Bearish Inside Bar Breakdown (3-bar coil)"
    return None

def adx_expansion(df):
    if len(df) < 30: return None
    adx = talib.ADX(df["high"], df["low"], df["close"], 14)
    ema20 = talib.EMA(df["close"], 20)
    if adx.iloc[-2] < 20 and adx.iloc[-1] > 25:
        return "Bullish ADX Expansion" if df["close"].iloc[-1] > ema20.iloc[-1] else "Bearish ADX Expansion"
    return None

def range_expansion_day(df, lookback=5):
    if len(df) < lookback + 2: return None
    today = df.iloc[-1]
    avg_range = (df["high"] - df["low"]).iloc[-lookback - 1 : -1].mean()
    if (today["high"] - today["low"]) > 1.5 * avg_range:
        return "Bullish Range Expansion Day" if today["close"] > today["open"] else "Bearish Range Expansion Day"
    return None

def failed_breakout_breakdown(df, lookback=20):
    if len(df) < lookback + 2: return None
    recent_high = df["high"].iloc[-lookback:-1].max()
    recent_low = df["low"].iloc[-lookback:-1].min()
    prev, curr = df.iloc[-2], df.iloc[-1]
    if prev["high"] > recent_high and curr["close"] < recent_high: return "Failed Breakout (Bearish)"
    if prev["low"] < recent_low and curr["close"] > recent_low: return "Failed Breakdown (Bullish)"
    return None

def ema_compression_expansion(df):
    if len(df) < 60: return None
    ema20, ema50, ema100 = talib.EMA(df["close"], 20), talib.EMA(df["close"], 50), talib.EMA(df["close"], 100)
    if abs(ema20.iloc[-2] - ema50.iloc[-2]) / ema50.iloc[-2] < 0.003 and abs(ema50.iloc[-2] - ema100.iloc[-2]) / ema100.iloc[-2] < 0.003:
        if ema20.iloc[-1] > ema50.iloc[-1] > ema100.iloc[-1]: return "Bullish EMA Compression Break"
        if ema20.iloc[-1] < ema50.iloc[-1] < ema100.iloc[-1]: return "Bearish EMA Compression Break"
    return None

def rsi_macd_cross_swing(df):
    if len(df) < 50: return None
    rsi = talib.RSI(df["close"], 14)
    macd, signal, _ = talib.MACD(df["close"], 12, 26, 9)
    if rsi.iloc[-2] < 40 and rsi.iloc[-1] > 40 and macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]: return "Bullish RSI+MACD Cross"
    if rsi.iloc[-2] > 60 and rsi.iloc[-1] < 60 and macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]: return "Bearish RSI+MACD Cross"
    return None

def atr_percent(df, period=14):
    if len(df) < period + 1: return None
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period).iloc[-1]
    return (atr / df["close"].iloc[-1]) * 100.0 if not pd.isna(atr) and df["close"].iloc[-1] > 0 else None

def calculate_confluence(row):
    score = 0
    text = " ".join([str(row.get(c, "")) for c in ["Signal", "Trend", "State", "Setup", "Divergence"]])
    for k in BULL_KEYWORDS:
        if k in text: score += 1
    for k in BEAR_KEYWORDS:
        if k in text: score -= 1
    score = max(min(score, 5), -5)
    bias = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"
    abs_score = abs(score)
    prob = "High" if abs_score >= 4 else "Medium" if abs_score >= 3 else "Low"
    return score, bias, prob

# ==============================
# SCANNER EXECUTION WRAPPER
# ==============================
def run_all_scanners_for_symbol(sym, df, tf, analysis_date, data_all_tfs):
    """
    Evaluates indicators and returns scanned result metrics contextually.
    """
    results = {}
    trimmed = trim_df_to_date(df, analysis_date)
    if trimmed is None:
        return results

    # Simple placeholder parsing context matching your strategy execution logic
    results["RSI"], results["Zone"] = rsi_market_pulse(trimmed) or (None, "NA")
    results["Signal"] = "Volume Shocker" if volume_shocker(trimmed) else "None"
    
    return results
