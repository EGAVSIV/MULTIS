import time
from io import StringIO
import requests
import pandas as pd

GITHUB_OWNER = "EGAVSIV"
GITHUB_REPO = "Data-Collector"
GITHUB_BRANCH = "main"

GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}"

TIMEFRAME_FOLDERS = {
    "15 Min": "stock_data_15",
    "1 Hour": "stock_data_1H",
    "Daily": "stock_data_D",
    "Weekly": "stock_data_W",
    "Monthly": "stock_data_M",
}

# Data is checked frequently so a new scanner request can see fresh GitHub data.
CACHE_TTL_SECONDS = 60

_cache = {}
_cache_time = {}

session = requests.Session()
session.headers.update({
    "Accept": "application/vnd.github+json",
    "User-Agent": "EGAVSIV-Multis-Scanner"
})


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    aliases = {
        "timestamp": "datetime",
        "date": "datetime",
        "time": "datetime",
        "datetime": "datetime",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vol": "volume",
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
        df["volume"] = 0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return None

    # Convert timezone-aware timestamps to IST-naive timestamps.
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


def _json_to_dataframe(payload):
    if isinstance(payload, list):
        return _normalise_columns(pd.DataFrame(payload))

    if isinstance(payload, dict):
        for key in ("data", "records", "candles", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return _normalise_columns(pd.DataFrame(value))

        try:
            return _normalise_columns(pd.DataFrame(payload))
        except Exception:
            return None

    return None


def _get_file_list(folder: str):
    response = session.get(
        f"{GITHUB_API_BASE}/{folder}",
        params={"ref": GITHUB_BRANCH},
        timeout=30,
    )
    response.raise_for_status()

    items = response.json()
    if not isinstance(items, list):
        return []

    return [
        item for item in items
        if item.get("type") == "file"
        and str(item.get("name", "")).lower().endswith(".json")
    ]


def _download_json(download_url: str):
    response = session.get(download_url, timeout=30)
    response.raise_for_status()
    return _json_to_dataframe(response.json())


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

    folder = TIMEFRAME_FOLDERS[timeframe]
    files = _get_file_list(folder)

    data = {}
    for item in files:
        symbol = str(item["name"]).rsplit(".", 1)[0]
        url = item.get("download_url") or f"{RAW_BASE}/{folder}/{item['name']}"
        try:
            df = _download_json(url)
            if df is not None and not df.empty:
                data[symbol] = df
        except Exception as exc:
            print(f"Skipping {timeframe}/{symbol}: {exc}")

    _cache[timeframe] = data
    _cache_time[timeframe] = time.time()
    print(f"Loaded {timeframe}: {len(data)} symbols")
    return data


def refresh_timeframe(timeframe: str):
    return load_data(timeframe, force_refresh=True)


def clear_cache():
    _cache.clear()
    _cache_time.clear()


def get_cache_status():
    now = time.time()
    return {
        tf: {
            "cached": tf in _cache,
            "age_seconds": round(now - _cache_time[tf], 1) if tf in _cache_time else None,
            "symbols": len(_cache.get(tf, {})),
        }
        for tf in TIMEFRAME_FOLDERS
    }
