import os
import sys
import logging
import smtplib
import mimetypes
import urllib.request
import urllib.parse
from datetime import datetime
from email.message import EmailMessage

import numpy as np
import pandas as pd
import talib
import matplotlib
matplotlib.use('Agg')  # Headless mode for server execution
import matplotlib.pyplot as plt

# ==============================================================================
# 1. LOGGING & MASTER SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NSE_Scanner")

# ==============================================================================
# 2. GLOBAL CONFIGURATION & RECIPIENTS
# ==============================================================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "nse.scanner.app@gmail.com"
SENDER_PASSWORD = "wmkdozoyfprduqgx"

RECIPIENTS = ["yadav.gauravsingh@gmail.com"]
BCC_RECIPIENTS = ["dipti.gorwadia@gmail.com"]

TELEGRAM_BOT_TOKEN = "8344354642:AAG_S7mavtiLP_yXPh4YM4u31QD5BBWJmuM"
TELEGRAM_CHAT_IDS = ["5332984891", "-1002622207173"]

BASE_PATH = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_PATH, "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMEFRAME_FOLDERS = {
    "Daily": ("stock_data_D", "Daily Scan"),
    "Weekly": ("stock_data_W", "Weekly Scan"),
    "Monthly": ("stock_data_M", "Monthly Scan"),
}

SAFE_COLS = [
    "Symbol", "Signal", "Trend", "State", "Setup", 
    "Divergence", "RSI", "Zone", "Confluence", "Bias", 
    "Probability", "TV_Link"
]

def make_tradingview_link(sym: str) -> str:
    return f"https://in.tradingview.com/chart/LqUZraZ9/?symbol=NSE%3A{sym}"

# ==============================================================================
# 3. TECHNICAL SCANNING DEFINITIONS (Core Mathematical Logic)
# ==============================================================================
def run_rsi_market_pulse(df):
    if len(df) < 14: return None, None
    rsi = talib.RSI(df["close"], 14).iloc[-1]
    zone = "RSI > 60" if rsi > 60 else ("RSI < 40" if rsi < 40 else "RSI 40–60")
    return round(rsi, 2), zone

def run_volume_shocker(df):
    if len(df) < 20: return False
    vol_sma = df["volume"].rolling(10).mean()
    last, prev = df.iloc[-1], df.iloc[-2]
    return (last["volume"] > 2 * vol_sma.iloc[-1] and prev["close"] * 0.95 <= last["close"] <= prev["close"] * 1.05)

def run_nrb_7(df):
    if len(df) < 20: return None
    base = df.iloc[-7]
    inside = df.iloc[-6:-1]
    last = df.iloc[-1]
    base_high, base_low = base["high"], base["low"]
    if not (inside["high"].max() <= base_high and inside["low"].min() >= base_low and inside["open"].max() <= base_high and inside["open"].min() >= base_low and inside["close"].max() <= base_high and inside["close"].min() >= base_low):
        return None
    if last["volume"] < 1.5 * df["volume"].rolling(10).mean().iloc[-2]: return None
    return "NRB-7 Bullish Breakout + Volume" if last["close"] > base_high else ("NRB-7 Bearish Breakdown + Volume" if last["close"] < base_low else None)

def run_counter_attack(df):
    if len(df) < 2: return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    mid = (prev["open"] + prev["close"]) / 2
    if prev["close"] < prev["open"] and curr["close"] > curr["open"] and curr["open"] < prev["close"] and curr["close"] >= mid: return "Bullish"
    if prev["close"] > prev["open"] and curr["close"] < curr["open"] and curr["open"] > prev["close"] and curr["close"] <= mid: return "Bearish"
    return None

def run_breakaway_gap(df):
    if len(df) < 50: return None
    df = df.copy()
    df["EMA20"] = talib.EMA(df["close"], 20)
    df["EMA50"] = talib.EMA(df["close"], 50)
    prev, curr = df.iloc[-2], df.iloc[-1]
    if curr["open"] > prev["high"] * 1.005 and curr["low"] > prev["high"] and curr["EMA20"].iloc[-1] < curr["EMA50"].iloc[-1]: return "Bullish Breakaway Gap"
    if curr["open"] < prev["low"] * 0.995 and curr["high"] < prev["low"] and curr["EMA20"].iloc[-1] > curr["EMA50"].iloc[-1]: return "Bearish Breakaway Gap"
    return None

def run_rsi_adx(df):
    if len(df) < 20: return None
    rsi = talib.RSI(df["close"], 14).iloc[-1]
    adx = talib.ADX(df["high"], df["low"], df["close"], 14).iloc[-1]
    if adx > 50 and rsi < 20: return "Bullish Reversal"
    if adx > 50 and rsi > 80: return "Probable Bearish Reversal"
    return None

def run_macd_market_pulse(df):
    if len(df) < 30: return None
    macd, signal, _ = talib.MACD(df["close"], 12, 26, 9)
    m, s, pm = macd.iloc[-1], signal.iloc[-1], macd.iloc[-2]
    if m > 0:
        return "Strong Bullish" if (m > s and m > pm) else ("Bullish Cooling" if (m > s and m < pm) else ("Bullish Reversal Watch" if (m < s and m > pm) else "Weak Bullish"))
    else:
        return "Bearish Reversal Watch" if (m > s and m > pm) else ("Weak Bearish" if (m > s and m < pm) else ("Bearish Recovery Attempt" if (m < s and m > pm) else "Strong Bearish"))

def run_macd_normal_divergence(df, lookback=30):
    if len(df) < lookback: return None
    macd, _, _ = talib.MACD(df["close"], 12, 26, 9)
    p_l1, p_l2 = df["low"].iloc[-lookback:-15].min(), df["low"].iloc[-15:].min()
    m_l1, m_l2 = macd.iloc[-lookback:-15].min(), macd.iloc[-15:].min()
    if p_l2 < p_l1 and m_l2 > m_l1: return "Bullish ND"
    p_h1, p_h2 = df["high"].iloc[-lookback:-15].max(), df["high"].iloc[-15:].max()
    m_h1, m_h2 = macd.iloc[-lookback:-15].max(), macd.iloc[-15:].max()
    if p_h2 > p_h1 and m_h2 < m_h1: return "Bearish ND"
    return None

def run_trend_alignment(df):
    if len(df) < 100: return None
    e20, e50, e100 = talib.EMA(df["close"], 20).iloc[-1], talib.EMA(df["close"], 50).iloc[-1], talib.EMA(df["close"], 100).iloc[-1]
    return "Strong Uptrend" if e20 > e50 > e100 else ("Strong Downtrend" if e20 < e50 < e100 else None)

def run_pullback_to_ema(df):
    if len(df) < 60: return None
    e20, e50 = talib.EMA(df["close"], 20).iloc[-1], talib.EMA(df["close"], 50).iloc[-1]
    last = df.iloc[-1]
    if e20 > e50 and last["low"] <= e20 and last["close"] > e20: return "Bullish EMA Pullback"
    if e20 < e50 and last["high"] >= e20 and last["close"] < e20: return "Bearish EMA Pullback"
    return None

def run_confluence_setup(df):
    if len(df) < 60: return None
    rsi = talib.RSI(df["close"], 14).iloc[-1]
    macd, sig, _ = talib.MACD(df["close"], 12, 26, 9)
    e20, e50 = talib.EMA(df["close"], 20).iloc[-1], talib.EMA(df["close"], 50).iloc[-1]
    if rsi > 50 and macd.iloc[-1] > sig.iloc[-1] and e20 > e50: return "Bullish Confluence"
    if rsi < 50 and macd.iloc[-1] < sig.iloc[-1] and e20 < e50: return "Bearish Confluence"
    return None

def run_macd_hook_up(df):
    if len(df) < 35: return None
    macd, signal, hist = talib.MACD(df["close"], 12, 26, 9)
    if (macd.iloc[-1] > 0 and macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-2] < macd.iloc[-3] and macd.iloc[-1] > macd.iloc[-2] and hist.iloc[-1] > hist.iloc[-2]):
        return "MACD Hook Up"
    return None

def run_macd_hook_down(df):
    if len(df) < 35: return None
    macd, signal, hist = talib.MACD(df["close"], 12, 26, 9)
    if (macd.iloc[-1] < 0 and macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-2] > macd.iloc[-3] and macd.iloc[-1] < macd.iloc[-2] and hist.iloc[-1] < hist.iloc[-2]):
        return "MACD Hook Down"
    return None

def run_ema50_stoch_oversold(df):
    if len(df) < 50: return None
    ema50 = talib.EMA(df["close"], 50).iloc[-1]
    slowk, slowd = talib.STOCH(df["high"], df["low"], df["close"], fastk_period=14, slowk_period=3, slowd_period=3)
    near_ema = abs(df["close"].iloc[-1] - ema50) / ema50 <= 0.005
    if near_ema and slowk.iloc[-2] < slowd.iloc[-2] and slowk.iloc[-1] > slowd.iloc[-1] and slowk.iloc[-1] < 20:
        return "EMA50 + Stoch Oversold Buy"
    return None

def run_kdj(df, period=9, signal=3):
    low_min, high_max = df["low"].rolling(period).min(), df["high"].rolling(period).max()
    rsv = (100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)).clip(lower=0, upper=100)
    def bcwsma(series, length):
        out = []
        for i, val in enumerate(series):
            out.append(val if (i == 0 or np.isnan(val)) else (val + (length - 1) * out[-1]) / length)
        return pd.Series(out, index=series.index)
    pK = bcwsma(rsv, signal)
    pD = bcwsma(pK, signal)
    return pK, pD, (3 * pK - 2 * pD)

def run_kdj_buy(df):
    if len(df) < 20: return None
    _, pD, pJ = run_kdj(df)
    if pd.isna(pD.iloc[-1]) or pd.isna(pJ.iloc[-1]): return None
    return "KDJ BUY (J↑D oversold)" if (pJ.iloc[-2] < pD.iloc[-2] and pJ.iloc[-1] > pD.iloc[-1] and pD.iloc[-1] < 30) else None

def run_kdj_sell(df):
    if len(df) < 20: return None
    _, pD, pJ = run_kdj(df)
    if pd.isna(pD.iloc[-1]) or pd.isna(pJ.iloc[-1]): return None
    return "KDJ SELL (J↓D overbought)" if (pJ.iloc[-2] > pD.iloc[-2] and pJ.iloc[-1] < pD.iloc[-1] and pD.iloc[-1] > 70) else None

# --- NEW SCANNER METHODS REQUESTED ---
def consecutive_close_momentum(df, min_count=3):
    if len(df) < min_count + 1: return None
    closes = df["close"].values
    if closes[-1] > closes[-2]: direction = "Bull"
    elif closes[-1] < closes[-2]: direction = "Bear"
    else: return None
    count = 1
    for i in range(len(closes) - 2, 0, -1):
        if direction == "Bull" and closes[i] > closes[i - 1]: count += 1
        elif direction == "Bear" and closes[i] < closes[i - 1]: count += 1
        else: break
    return f"{direction} Momentum ({count} Days)" if count >= min_count else None

def inside_bar_breakout(df):
    if len(df) < 4: return None
    mother, inside1, inside2, curr = df.iloc[-4], df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if (inside1["high"] < mother["high"] and inside1["low"] > mother["low"] and inside2["high"] < mother["high"] and inside2["low"] > mother["low"]):
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

# Combined Master Mapping Execution Set
SCANNERS = {
    "RSI Market Pulse": lambda df: {"Signal": "Monitor", "RSI": run_rsi_market_pulse(df)[0], "Zone": run_rsi_market_pulse(df)[1]},
    "Volume Shocker": lambda df: {"Signal": "BUY" if run_volume_shocker(df) else "Neutral", "Setup": "High Vol Expansion" if run_volume_shocker(df) else None},
    "NRB-7 breakout": lambda df: {"Signal": "BUY/SELL" if run_nrb_7(df) else "Neutral", "Setup": run_nrb_7(df)},
    "Counter Attack Pattern": lambda df: {"Signal": run_counter_attack(df) or "Neutral"},
    "Breakaway Gap": lambda df: {"Signal": "Alert", "Setup": run_breakaway_gap(df)},
    "RSI + ADX Extremes": lambda df: {"Signal": "Reversal Watch", "Setup": run_rsi_adx(df)},
    "MACD Market Pulse": lambda df: {"Signal": "Monitor", "Trend": run_macd_market_pulse(df)},
    "MACD Normal Divergence": lambda df: {"Signal": "Divergence Alert", "Divergence": run_macd_normal_divergence(df)} if run_macd_normal_divergence(df) else None,
    "Trend Alignment": lambda df: {"Signal": "Uptrend/Downtrend", "Trend": run_trend_alignment(df)},
    "Pullback to EMA": lambda df: {"Signal": "Pullback Check", "Setup": run_pullback_to_ema(df)},
    "Confluence": lambda df: {"Signal": "Confluence Detected", "Setup": run_confluence_setup(df)},
    "MACD Hook Up": lambda df: {"Signal": "BUY" if run_macd_hook_up(df) else "Neutral"},
    "MACD Hook Down": lambda df: {"Signal": "SELL" if run_macd_hook_down(df) else "Neutral"},
    "EMA50 + Stochastic": lambda df: {"Signal": "BUY" if run_ema50_stoch_oversold(df) else "Neutral"},
    "KDJ Cross Buy": lambda df: {"Signal": "BUY" if run_kdj_buy(df) else "Neutral"},
    "KDJ Cross Sell": lambda df: {"Signal": "SELL" if run_kdj_sell(df) else "Neutral"},
    "Consecutive Momentum": lambda df: {"Signal": "Momentum", "Setup": consecutive_close_momentum(df)},
    "Inside Bar Breakout": lambda df: {"Signal": "Breakout", "Setup": inside_bar_breakout(df)},
    "ADX Expansion": lambda df: {"Signal": "Expansion", "Setup": adx_expansion(df)},
    "Range Expansion Day": lambda df: {"Signal": "Range Expansion", "Setup": range_expansion_day(df)},
    "Failed Breakout/Breakdown": lambda df: {"Signal": "Failed Breakout", "Setup": failed_breakout_breakdown(df)}
}

def extract_grid_cell_value(scanner_name, res):
    if not res: return ""
    if scanner_name == "Volume Shocker": return "High Vol Expansion" if res.get("Signal") == "BUY" else ""
    if scanner_name in ["Counter Attack Pattern", "MACD Hook Up", "MACD Hook Down", "EMA50 + Stochastic", "KDJ Cross Buy", "KDJ Cross Sell"]: 
        return res.get("Signal") if res.get("Signal") != "Neutral" else ""
    if scanner_name in ["Breakaway Gap", "RSI + ADX Extremes", "Pullback to EMA", "Confluence", "Consecutive Momentum", "Inside Bar Breakout", "ADX Expansion", "Range Expansion Day", "Failed Breakout/Breakdown"]: 
        return res.get("Setup") or ""
    if scanner_name in ["MACD Market Pulse", "Trend Alignment"]: return res.get("Trend") or ""
    if scanner_name == "MACD Normal Divergence": return res.get("Divergence") or ""
    if scanner_name == "RSI Market Pulse": return res.get("Zone") or ""
    return str(res.get("Signal", ""))

# ==============================================================================
# 4. HISTORICAL MACD CHART MAKER
# ==============================================================================
def generate_historical_macd_chart(all_dfs):
    logger.info("Generating historical 15-day MACD distribution chart...")
    if not all_dfs: return None

    # Sync base dates from tracking arrays
    sample_df = all_dfs[0]
    if len(sample_df) < 45: return None
    
    dates = sample_df.index[-15:]
    above_pct, below_pct, date_labels = [], [], []

    for idx in range(-15, 0):
        pos_count, neg_count, total = 0, 0, 0
        current_date = sample_df.index[idx]
        date_labels.append(pd.to_datetime(current_date).strftime('%Y-%m-%d'))
        
        for df in all_dfs:
            if len(df) >= 35:
                macd, _, _ = talib.MACD(df["close"], 12, 26, 9)
                val = macd.iloc[idx]
                if not pd.isna(val):
                    total += 1
                    if val > 0: pos_count += 1
                    else: neg_count += 1
        
        above_pct.append((pos_count / total * 100) if total > 0 else 0)
        below_pct.append((neg_count / total * 100) if total > 0 else 0)

    plt.figure(figsize=(10, 4.5))
    plt.plot(date_labels, above_pct, marker='o', color='green', label='% Stocks > 0 (Bullish)')
    plt.plot(date_labels, below_pct, marker='o', color='red', label='% Stocks < 0 (Bearish)')
    
    plt.title("MACD > 0 vs < 0 (Last_15_Trading_Days_FNO_STOCKS)")
    plt.ylabel("Percentage of Stocks")
    plt.xlabel("Date")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(-5, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, "macd_trend_15d.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    return chart_path

# ==============================================================================
# 5. CORE ENGINE PROCESSING
# ==============================================================================
def process_timeframe(folder_name, output_name, date_str, rare_alerts_list):
    folder_path = os.path.join(BASE_PATH, folder_name)
    if not os.path.exists(folder_path): return None, [], pd.DataFrame(), []

    files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files: return None, [], pd.DataFrame(), []

    logger.info(f"Scanning {len(files)} symbols in {folder_name}...")
    sheets_data = {scanner_name: [] for scanner_name in SCANNERS.keys()}
    grid_rows, timeframe_dfs = [], []

    for f in files:
        sym = f.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(folder_path, f))
            if df.empty or len(df) < 50: continue
            timeframe_dfs.append(df)

            grid_stock_row = {"Symbol": sym}
            for scanner_name, scanner_fn in SCANNERS.items():
                try:
                    res = scanner_fn(df)
                    grid_stock_row[scanner_name] = extract_grid_cell_value(scanner_name, res)
                    
                    if res:
                        row = {col: "" for col in SAFE_COLS}
                        row["Symbol"] = sym
                        row["TV_Link"] = make_tradingview_link(sym)
                        for k, v in res.items():
                            if k in row and v is not None: row[k] = v
                                
                        if any(row[col] not in ["", "Neutral", None] for col in ["Signal", "Setup", "Divergence", "Trend"]):
                            sheets_data[scanner_name].append(row)
                            
                            # Intercept rare events for live dashboard
                            cell_val = grid_stock_row[scanner_name]
                            if scanner_name in ["Volume Shocker", "Breakaway Gap", "KDJ Cross Buy", "KDJ Cross Sell", "MACD Hook Up", "MACD Hook Down"] and cell_val not in ["", "Neutral"]:
                                rare_alerts_list.append([sym, folder_name.replace("stock_data_", ""), scanner_name, cell_val])
                except Exception:
                    pass
            grid_rows.append(grid_stock_row)
        except Exception as e:
            logger.error(f"Error loading file {f}: {e}")

    excel_filename = f"{output_name}_{date_str}.xlsx"
    excel_filepath = os.path.join(OUTPUT_DIR, excel_filename)
    
    with pd.ExcelWriter(excel_filepath, engine="openpyxl") as writer:
        for scanner_name, rows in sheets_data.items():
            df_out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=SAFE_COLS)
            if df_out.empty:
                df_out = pd.DataFrame([["No scanner alerts detected for this interval", ""] + [""] * 10], columns=SAFE_COLS)
            df_out.to_excel(writer, sheet_name=scanner_name[:30], index=False)

    df_grid_tf = pd.DataFrame(grid_rows) if grid_rows else pd.DataFrame(columns=["Symbol"] + list(SCANNERS.keys()))
    return excel_filepath, sheets_data.get("MACD Normal Divergence", []), df_grid_tf, timeframe_dfs

# ==============================================================================
# 6. MAIL TRANSMISSION
# ==============================================================================
def send_email_with_attachments(file_paths, chart_path, rare_alerts, date_str):
    if not SENDER_EMAIL or not SENDER_PASSWORD: return False

    msg = EmailMessage()
    msg["Subject"] = f"FNO Master Scanner Report - {date_str}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECIPIENTS)
    if "BCC_RECIPIENTS" in globals() and BCC_RECIPIENTS:
        msg["Bcc"] = ", ".join(BCC_RECIPIENTS)

    # --- Construct Summary Dashboard Body ---
    dashboard_html = "<h2>=== FNO AUTOMATED SCANNER DASHBOARD ===</h2>"
    
    if rare_alerts:
        dashboard_html += """
        <div style='background-color: #fff3cd; border: 2px solid #ffeeba; padding: 15px; margin-bottom: 20px; border-radius: 5px;'>
            <h3 style='color: #856404; margin-top: 0;'>⚠️ RARE METRIC DETECTIONS DETECTED</h3>
            <table border='1' cellpadding='6' style='border-collapse: collapse; text-align: left;'>
                <tr style='background-color: #ffeeba;'><th>Symbol</th><th>Timeframe</th><th>Scanner</th><th>Trigger Alert</th></tr>
        """
        for sym, tf, scan_n, val in rare_alerts:
            dashboard_html += f"<tr><td><b>{sym}</b></td><td>{tf}</td><td>{scan_n}</td><td style='color:red;'>{val}</td></tr>"
        dashboard_html += "</table></div>"
    else:
        dashboard_html += "<p><i>No rare operational alert markers triggered today.</i></p>"

    dashboard_html += "<p>Please find attached your daily multi-timeframe analytical suite including specific sheets, normal deviations, and the summary grid data matching your custom specifications.</p>"
    
    if chart_path:
        dashboard_html += "<br><h3>📈 15-Day MACD Market Distribution:</h3><img src='cid:macd_chart' width='700'><br>"
        
    msg.set_content(dashboard_html, subtype='html')

    # Embed Chart Image Inline
    if chart_path and os.path.exists(chart_path):
        with open(chart_path, 'rb') as img:
            msg.get_payload()[0].add_related(img.read(), maintype='image', subtype='png', cid='macd_chart')

    # Attach Excel Workbooks
    for path in file_paths:
        if not os.path.exists(path): continue
        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None or encoding is not None: ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(path, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(path))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        logger.info("Notification email dispatched successfully.")
        return True
    except Exception as e:
        logger.error(f"SMTP Email Delivery failed: {e}")
        return False

# ==============================================================================
# 7. AUTOMATION EXECUTION PIPELINE
# ==============================================================================
def main():
    start_time = datetime.now()
    date_str = start_time.strftime("%d %b %Y")
    logger.info(f"=== Starting NSE Batch Scanner Pipeline ({date_str}) ===")

    generated_files = []
    macd_divergences_by_tf = {}  
    grid_dataframes_by_tf = {}
    rare_alerts = []
    master_dfs_list = []

    for tf_key, (folder_name, output_name) in TIMEFRAME_FOLDERS.items():
        filepath, macd_rows, df_grid_tf, timeframe_dfs = process_timeframe(folder_name, output_name, date_str, rare_alerts)
        if filepath:
            generated_files.append(filepath)
            macd_divergences_by_tf[tf_key] = macd_rows
            if tf_key == "Daily":
                master_dfs_list = timeframe_dfs
            if not df_grid_tf.empty:
                grid_dataframes_by_tf[tf_key] = df_grid_tf

    # Generate 15-day distribution image using Daily data
    chart_path = generate_historical_macd_chart(master_dfs_list)

    if grid_dataframes_by_tf:
        summary_matrix_filename = f"Scanner Summary Matrix_{date_str}.xlsx"
        summary_matrix_filepath = os.path.join(OUTPUT_DIR, summary_matrix_filename)
        with pd.ExcelWriter(summary_matrix_filepath, engine="openpyxl") as writer:
            for tf_key, df_matrix in grid_dataframes_by_tf.items():
                cols_order = ["Symbol"] + [c for c in df_matrix.columns if c != "Symbol"]
                df_matrix[cols_order].to_excel(writer, sheet_name=tf_key, index=False)
        generated_files.append(summary_matrix_filepath)

    if macd_divergences_by_tf:
        combined_macd_filename = f"MACD Normal Divergences_{date_str}.xlsx"
        combined_macd_filepath = os.path.join(OUTPUT_DIR, combined_macd_filename)
        with pd.ExcelWriter(combined_macd_filepath, engine="openpyxl") as writer:
            for tf_key, rows in macd_divergences_by_tf.items():
                df_out = pd.DataFrame(rows) if rows else pd.DataFrame([["No scanner alerts detected for this interval", ""] + [""] * 10], columns=SAFE_COLS)
                df_out.to_excel(writer, sheet_name=tf_key, index=False)
        generated_files.append(combined_macd_filepath)

    if not generated_files: return
    
    # Send Compiled Attachments, Inline Distribution Chart, and Dashboard Alerts
    send_email_with_attachments(generated_files, chart_path, rare_alerts, date_str)
    logger.info(f"=== Pipeline completed successfully in {(datetime.now() - start_time).total_seconds():.2f} seconds ===")

if __name__ == "__main__":
    main()
