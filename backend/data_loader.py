import time
import requests
import pandas as pd

# ============================================================
# GITHUB DATA REPOSITORY CONFIGURATION
# ============================================================

GITHUB_OWNER = "EGAVSIV"
GITHUB_REPO = "Data-Collector"
GITHUB_BRANCH = "main"

GITHUB_API_BASE = (
    f"https://api.github.com/repos/"
    f"{GITHUB_OWNER}/{GITHUB_REPO}/contents"
)

RAW_BASE = (
    f"https://raw.githubusercontent.com/"
    f"{GITHUB_OWNER}/{GITHUB_REPO}/"
    f"{GITHUB_BRANCH}"
)


# ============================================================
# TIMEFRAME → GITHUB FOLDER
# ============================================================

TIMEFRAME_FOLDERS = {
    "15 Min": "stock_data_15",
    "1 Hour": "stock_data_1H",
    "Daily": "stock_data_D",
    "Weekly": "stock_data_W",
    "Monthly": "stock_data_M",
}


# ============================================================
# CACHE SETTINGS
# ============================================================

CACHE_TTL_SECONDS = 30 * 60

_data_cache = {}
_cache_time = {}


# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "Accept": "application/vnd.github+json",
    "User-Agent": "NSE-Stock-Scanner"
})


# ============================================================
# NORMALIZE DATAFRAME
# ============================================================

def normalize_dataframe(df: pd.DataFrame):

    if df is None or df.empty:
        return None

    df = df.copy()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    column_mapping = {
        "timestamp": "datetime",
        "date": "datetime",
        "time": "datetime",

        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }

    df = df.rename(columns=column_mapping)

    # --------------------------------------------------------
    # Handle datetime
    # --------------------------------------------------------

    if "datetime" not in df.columns:

        # Sometimes JSON index becomes unnamed
        possible_datetime_columns = [
            "index",
            "unnamed: 0",
            "unnamed: 0.1"
        ]

        for col in possible_datetime_columns:

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

    try:

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
            utc=True
        )

        # Convert to IST then remove timezone
        if df["datetime"].notna().any():

            df["datetime"] = (
                df["datetime"]
                .dt.tz_convert("Asia/Kolkata")
                .dt.tz_localize(None)
            )

    except Exception:

        try:

            df["datetime"] = pd.to_datetime(
                df["datetime"],
                errors="coerce"
            )

        except Exception:

            return None

    # --------------------------------------------------------
    # Remove invalid datetime
    # --------------------------------------------------------

    df = df.dropna(subset=["datetime"])

    if df.empty:
        return None

    # --------------------------------------------------------
    # Required OHLC columns
    # --------------------------------------------------------

    required_columns = [
        "open",
        "high",
        "low",
        "close"
    ]

    for col in required_columns:

        if col not in df.columns:
            return None

        df[col] = pd.to_numeric(
            df[col],
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
    # Remove invalid OHLC rows
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
    # Sort and set index
    # --------------------------------------------------------

    df = (
        df
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last"
        )
        .set_index("datetime")
    )

    return df


# ============================================================
# READ JSON FILE
# ============================================================

def read_json_from_github(download_url):

    try:

        response = session.get(
            download_url,
            timeout=30
        )

        response.raise_for_status()

        json_data = response.json()

    except Exception as e:

        print(
            f"❌ JSON download error: {e}"
        )

        return None

    try:

        # ----------------------------------------------------
        # CASE 1: Normal list
        # ----------------------------------------------------

        if isinstance(json_data, list):

            df = pd.DataFrame(json_data)

        # ----------------------------------------------------
        # CASE 2: Dictionary
        # ----------------------------------------------------

        elif isinstance(json_data, dict):

            # Common wrapper keys
            for key in [
                "data",
                "records",
                "candles",
                "results"
            ]:

                if key in json_data:

                    value = json_data[key]

                    if isinstance(value, list):

                        df = pd.DataFrame(value)

                        return normalize_dataframe(df)

            # Direct dictionary structure
            df = pd.DataFrame(json_data)

        else:

            return None

        return normalize_dataframe(df)

    except Exception as e:

        print(
            f"❌ JSON parsing error: {e}"
        )

        return None


# ============================================================
# GET FILE LIST FROM GITHUB
# ============================================================

def get_github_file_list(folder):

    url = f"{GITHUB_API_BASE}/{folder}"

    params = {
        "ref": GITHUB_BRANCH
    }

    try:

        response = session.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        files = response.json()

        if not isinstance(files, list):

            print(
                f"❌ Unexpected GitHub response "
                f"for {folder}"
            )

            return []

        json_files = []

        for item in files:

            if item.get("type") != "file":
                continue

            name = item.get("name", "")

            if name.lower().endswith(".json"):

                json_files.append({
                    "name": name,
                    "download_url":
                        item.get("download_url")
                        or (
                            f"{RAW_BASE}/"
                            f"{folder}/{name}"
                        )
                })

        return json_files

    except Exception as e:

        print(
            f"❌ GitHub file list error "
            f"for {folder}: {e}"
        )

        return []


# ============================================================
# LOAD ONE TIMEFRAME
# ============================================================

def load_data(timeframe, force_refresh=False):

    if timeframe not in TIMEFRAME_FOLDERS:

        print(
            f"❌ Unknown timeframe: {timeframe}"
        )

        return {}

    # --------------------------------------------------------
    # CACHE CHECK
    # --------------------------------------------------------

    now = time.time()

    if (
        not force_refresh
        and timeframe in _data_cache
        and timeframe in _cache_time
    ):

        age = now - _cache_time[timeframe]

        if age < CACHE_TTL_SECONDS:

            print(
                f"⚡ Using cached data: "
                f"{timeframe}"
            )

            return _data_cache[timeframe]

    folder = TIMEFRAME_FOLDERS[timeframe]

    print(
        f"\n📥 Loading {timeframe} "
        f"from GitHub..."
    )

    # --------------------------------------------------------
    # GET JSON FILE LIST
    # --------------------------------------------------------

    files = get_github_file_list(folder)

    if not files:

        print(
            f"⚠️ No JSON files found "
            f"in {folder}"
        )

        return {}

    print(
        f"📂 Found {len(files)} JSON files"
    )

    data = {}

    success_count = 0
    failed_count = 0

    # --------------------------------------------------------
    # DOWNLOAD EACH SYMBOL
    # --------------------------------------------------------

    for i, item in enumerate(files, start=1):

        filename = item["name"]

        symbol = filename.rsplit(
            ".",
            1
        )[0]

        try:

            df = read_json_from_github(
                item["download_url"]
            )

            if df is not None and not df.empty:

                data[symbol] = df

                success_count += 1

            else:

                failed_count += 1

        except Exception as e:

            failed_count += 1

            print(
                f"❌ Error loading "
                f"{symbol}: {e}"
            )

        # Progress log every 25 files
        if i % 25 == 0:

            print(
                f"⏳ {timeframe}: "
                f"{i}/{len(files)} processed"
            )

    # --------------------------------------------------------
    # SAVE CACHE
    # --------------------------------------------------------

    _data_cache[timeframe] = data

    _cache_time[timeframe] = time.time()

    print(
        f"✅ {timeframe} loaded: "
        f"{success_count} symbols | "
        f"Failed: {failed_count}"
    )

    return data


# ============================================================
# FORCE REFRESH ALL DATA
# ============================================================

def refresh_all_data():

    print("\n🔄 FORCE REFRESHING ALL DATA")

    _data_cache.clear()

    _cache_time.clear()

    results = {}

    for timeframe in TIMEFRAME_FOLDERS:

        results[timeframe] = load_data(
            timeframe,
            force_refresh=True
        )

    return results


# ============================================================
# GET CACHE STATUS
# ============================================================

def get_cache_status():

    now = time.time()

    status = {}

    for timeframe in TIMEFRAME_FOLDERS:

        if timeframe in _cache_time:

            age_seconds = (
                now -
                _cache_time[timeframe]
            )

            status[timeframe] = {
                "cached": True,
                "age_seconds": round(
                    age_seconds,
                    2
                ),
                "symbols": len(
                    _data_cache.get(
                        timeframe,
                        {}
                    )
                )
            }

        else:

            status[timeframe] = {
                "cached": False,
                "age_seconds": None,
                "symbols": 0
            }

    return status


# ============================================================
# CLEAR CACHE
# ============================================================

def clear_cache():

    _data_cache.clear()

    _cache_time.clear()

    print("🗑️ Data cache cleared")
