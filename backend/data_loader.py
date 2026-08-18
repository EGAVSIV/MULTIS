from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_DIR / "COPIEDDATA"
TIMEFRAME_FOLDERS = {
    "15 Min":"stock_data_15","1 Hour":"stock_data_1H","Daily":"stock_data_D",
    "Weekly":"stock_data_W","Monthly":"stock_data_M"
}
_cache={}

def _df(payload):
    if isinstance(payload, dict):
        for k in ("data","records","candles","results"):
            if isinstance(payload.get(k), list): payload=payload[k]; break
    df=pd.DataFrame(payload)
    df.columns=[str(c).strip().lower() for c in df.columns]
    aliases={"timestamp":"datetime","date":"datetime","time":"datetime","o":"open","h":"high","l":"low","c":"close","v":"volume","vol":"volume"}
    df=df.rename(columns={k:v for k,v in aliases.items() if k in df.columns})
    if "datetime" not in df.columns: return None
    df["datetime"]=pd.to_datetime(df["datetime"],errors="coerce")
    for c in ("open","high","low","close","volume"):
        if c not in df.columns:
            if c=="volume": df[c]=0
            else: return None
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["datetime","open","high","low","close"]).sort_values("datetime").drop_duplicates("datetime")
    return df.set_index("datetime") if not df.empty else None

def load_data(timeframe, refresh=False):
    if timeframe not in TIMEFRAME_FOLDERS: raise ValueError(f"Unknown timeframe: {timeframe}")
    if timeframe in _cache and not refresh: return _cache[timeframe]
    folder=DATA_ROOT/TIMEFRAME_FOLDERS[timeframe]
    if not folder.exists(): raise FileNotFoundError(f"Data folder not found: {folder}")
    out={}
    for p in folder.rglob("*.json"):
        try:
            x=_df(json.loads(p.read_text(encoding="utf-8")))
            if x is not None: out[p.stem.upper()]=x
        except Exception as e: print("Skip",p,e)
    _cache[timeframe]=out
    return out

def trim_df_to_date(df, analysis_date):
    if not analysis_date: return df
    ts=pd.Timestamp(analysis_date)
    x=df.loc[df.index<=ts+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)]
    return x if len(x) else None

def cache_status():
    return {tf:len(load_data(tf)) for tf in TIMEFRAME_FOLDERS if (DATA_ROOT/TIMEFRAME_FOLDERS[tf]).exists()}
