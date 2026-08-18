from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIMEFRAME_FOLDERS = {
    "15 Min": ROOT / "stock_data_15",
    "1 Hour": ROOT / "stock_data_1H",
    "Daily": ROOT / "stock_data_D",
    "Weekly": ROOT / "stock_data_W",
    "Monthly": ROOT / "stock_data_M",
}
REQUIRED = {"open","high","low","close","volume"}

def _read_json(path: Path) -> pd.DataFrame:
    # Supports records, column-oriented, pandas split/table JSON and common GitHub JSON outputs.
    try:
        df = pd.read_json(path, orient="records")
    except Exception:
        try:
            df = pd.read_json(path)
        except Exception:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key in ("data","records","candles","result"):
                    if isinstance(raw.get(key), list):
                        raw = raw[key]; break
            df = pd.DataFrame(raw)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    # normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"timestamp":"datetime","date":"datetime","time":"datetime","vol":"volume"}
    df = df.rename(columns={k:v for k,v in aliases.items() if k in df.columns})
    if "datetime" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index":"datetime"})
        else:
            raise ValueError("JSON has no datetime/timestamp/date field")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in REQUIRED:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["datetime",*REQUIRED]).sort_values("datetime").set_index("datetime")
    return df

def load_timeframe(timeframe: str) -> dict:
    folder = TIMEFRAME_FOLDERS.get(timeframe)
    if folder is None: raise ValueError(f"Unknown timeframe: {timeframe}")
    data = {}
    if not folder.exists(): return data
    for path in folder.glob("*.json"):
        try:
            df = _read_json(path)
            if REQUIRED.issubset(df.columns) and not df.empty:
                data[path.stem] = df
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
    return data

def last_candle(timeframe: str):
    data = load_timeframe(timeframe)
    last = None
    for df in data.values():
        if not df.empty:
            dt = df.index.max()
            if last is None or dt > last: last = dt
    if last is None: return None
    if getattr(last, "tzinfo", None) is None: last = last.tz_localize("UTC")
    return last.tz_convert("Asia/Kolkata")
