from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_DIR / "COPIEDDATA"

TIMEFRAME_FOLDERS = {
    "15 Min": "stock_data_15",
    "1 Hour": "stock_data_1H",
    "Daily": "stock_data_D",
    "Weekly": "stock_data_W",
    "Monthly": "stock_data_M",
}

CACHE_TTL_SECONDS = 300
_cache: dict[str, dict[str, pd.DataFrame]] = {}
_cache_time: dict[str, float] = {}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    aliases = {
        "timestamp": "datetime", "date": "datetime", "time": "datetime",
        "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "vol": "volume",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    if "datetime" not in df.columns:
        for candidate in ("index", "unnamed: 0", "unnamed: 0.1"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "datetime"})
                break

    if "datetime" not in df.columns:
        return None

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            return None
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return None

    try:
        if getattr(df["datetime"].dt, "tz", None) is not None:
            df["datetime"] = (
                df["datetime"]
                .dt.tz_convert("Asia/Kolkata")
                .dt.tz_localize(None)
            )
    except Exception:
        pass

    return (
        df.sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .set_index("datetime")
    )


def _payload_to_df(payload):
    if isinstance(payload, list):
        return _normalise_columns(pd.DataFrame(payload))

    if isinstance(payload, dict):
        for key in ("data", "records", "candles", "results"):
            if isinstance(payload.get(key), list):
                return _normalise_columns(pd.DataFrame(payload[key]))
        try:
            return _normalise_columns(pd.DataFrame(payload))
        except Exception:
            return None

    return None


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as fh:
            return _payload_to_df(json.load(fh))
    except Exception as exc:
        print(f"Skipping {path}: {exc}")
        return None


def load_data(timeframe: str, force_refresh: bool = False):
    if timeframe not in TIMEFRAME_FOLDERS:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    now = time.time()
    if (
        not force_refresh
        and timeframe in _cache
        and now - _cache_time.get(timeframe, 0) < CACHE_TTL_SECONDS
    ):
        return _cache[timeframe]

    folder = DATA_ROOT / TIMEFRAME_FOLDERS[timeframe]
    if not folder.exists():
        raise FileNotFoundError(f"Copied data folder not found: {folder}")

    data = {}
    for path in sorted(folder.rglob("*.json")):
        if path.name == ".copy_manifest.json":
            continue
        df = _read_json(path)
        if df is not None and not df.empty:
            data[path.stem.upper()] = df

    _cache[timeframe] = data
    _cache_time[timeframe] = time.time()

    print(f"Loaded {timeframe}: {len(data)} symbols from {folder}")
    return data


def refresh_timeframe(timeframe: str):
    return load_data(timeframe, force_refresh=True)


def clear_cache():
    _cache.clear()
    _cache_time.clear()


def get_cache_status():
    now = time.time()
    return {
        timeframe: {
            "cached": timeframe in _cache,
            "age_seconds": (
                round(now - _cache_time[timeframe], 1)
                if timeframe in _cache_time else None
            ),
            "symbols": len(_cache.get(timeframe, {})),
            "folder": str(DATA_ROOT / folder),
        }
        for timeframe, folder in TIMEFRAME_FOLDERS.items()
    }
