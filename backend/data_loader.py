from pathlib import Path
import json
import time
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

DATA_ROOT = PROJECT_DIR / "COPIEDDATA"


# ============================================================
# TIMEFRAME MAPPING
# ============================================================

TIMEFRAME_FOLDERS = {
    "15 Min": "stock_data_15",
    "1 Hour": "stock_data_1H",
    "Daily": "stock_data_D",
    "Weekly": "stock_data_W",
    "Monthly": "stock_data_M",
}


# ============================================================
# CACHE
# ============================================================

CACHE_TTL_SECONDS = 300

_cache = {}
_cache_time = {}


# ============================================================
# NORMALIZE DATA
# ============================================================

def _normalise_columns(df):

    if df is None or df.empty:
        return None

    df = df.copy()

    # Normalize column names
    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    aliases = {
        "timestamp": "datetime",
        "date": "datetime",
        "time": "datetime",

        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",

        "v": "volume",
        "vol": "volume",
    }

    df = df.rename(
        columns={
            old: new
            for old, new in aliases.items()
            if old in df.columns
        }
    )

    # --------------------------------------------------------
    # Find datetime column
    # --------------------------------------------------------

    if "datetime" not in df.columns:

        possible_columns = [
            "index",
            "unnamed: 0",
            "unnamed: 0.1"
        ]

        for col in possible_columns:

            if col in df.columns:

                df = df.rename(
                    columns={col: "datetime"}
                )

                break

    if "datetime" not in df.columns:
        return None

    # --------------------------------------------------------
    # Convert datetime
    # --------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["datetime"]
    )

    # --------------------------------------------------------
    # Required OHLC columns
    # --------------------------------------------------------

    required_columns = [
        "open",
        "high",
        "low",
        "close"
    ]

    for column in required_columns:

        if column not in df.columns:
            return None

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if "volume" not in df.columns:

        df["volume"] = 0

    else:

        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce"
        ).fillna(0)

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    if df.empty:
        return None

    # --------------------------------------------------------
    # Timezone handling
    # --------------------------------------------------------

    try:

        if getattr(
            df["datetime"].dt,
            "tz",
            None
        ) is not None:

            df["datetime"] = (
                df["datetime"]
                .dt
                .tz_convert("Asia/Kolkata")
                .dt
                .tz_localize(None)
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # Final dataframe
    # --------------------------------------------------------

    return (
        df
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last"
        )
        .set_index("datetime")
    )


# ============================================================
# JSON TO DATAFRAME
# ============================================================

def json_to_dataframe(payload):

    if isinstance(payload, list):

        return _normalise_columns(
            pd.DataFrame(payload)
        )

    if isinstance(payload, dict):

        # Common JSON structures

        possible_keys = [
            "data",
            "records",
            "candles",
            "results"
        ]

        for key in possible_keys:

            value = payload.get(key)

            if isinstance(value, list):

                return _normalise_columns(
                    pd.DataFrame(value)
                )

        # Try dictionary directly

        try:

            return _normalise_columns(
                pd.DataFrame(payload)
            )

        except Exception:

            return None

    return None


# ============================================================
# LOAD SINGLE JSON FILE
# ============================================================

def load_json_file(file_path):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            payload = json.load(file)

        return json_to_dataframe(payload)

    except Exception as exc:

        print(
            f"Skipping {file_path.name}: {exc}"
        )

        return None


# ============================================================
# LOAD TIMEFRAME DATA
# ============================================================

def load_data(
    timeframe,
    force_refresh=False
):

    if timeframe not in TIMEFRAME_FOLDERS:

        raise ValueError(
            f"Unknown timeframe: {timeframe}"
        )

    # --------------------------------------------------------
    # Return cache
    # --------------------------------------------------------

    now = time.time()

    if (
        not force_refresh
        and timeframe in _cache
        and
        now - _cache_time.get(
            timeframe,
            0
        ) < CACHE_TTL_SECONDS
    ):

        return _cache[timeframe]

    # --------------------------------------------------------
    # Local folder
    # --------------------------------------------------------

    folder_name = TIMEFRAME_FOLDERS[timeframe]

    folder_path = (
        DATA_ROOT /
        folder_name
    )

    if not folder_path.exists():

        raise FileNotFoundError(
            f"Data folder not found: "
            f"{folder_path}"
        )

    # --------------------------------------------------------
    # Read JSON files
    # --------------------------------------------------------

    data = {}

    json_files = sorted(
        folder_path.glob("*.json")
    )

    print(
        f"Loading {timeframe} "
        f"from {folder_path}"
    )

    print(
        f"JSON files found: "
        f"{len(json_files)}"
    )

    for file_path in json_files:

        symbol = file_path.stem.upper()

        df = load_json_file(
            file_path
        )

        if (
            df is not None
            and not df.empty
        ):

            data[symbol] = df

    # --------------------------------------------------------
    # Update cache
    # --------------------------------------------------------

    _cache[timeframe] = data

    _cache_time[timeframe] = time.time()

    print(
        f"Loaded {timeframe}: "
        f"{len(data)} symbols"
    )

    return data


# ============================================================
# REFRESH TIMEFRAME
# ============================================================

def refresh_timeframe(timeframe):

    return load_data(
        timeframe,
        force_refresh=True
    )


# ============================================================
# CLEAR CACHE
# ============================================================

def clear_cache():

    _cache.clear()

    _cache_time.clear()


# ============================================================
# CACHE STATUS
# ============================================================

def get_cache_status():

    now = time.time()

    result = {}

    for timeframe in TIMEFRAME_FOLDERS:

        result[timeframe] = {

            "cached":
                timeframe in _cache,

            "age_seconds":
                round(
                    now -
                    _cache_time[timeframe],
                    1
                )
                if timeframe in _cache_time
                else None,

            "symbols":
                len(
                    _cache.get(
                        timeframe,
                        {}
                    )
                )
        }

    return result
