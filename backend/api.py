from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
import math
import pandas as pd

# ============================================================
# DATA LOADER
# ============================================================

from data_loader import (
    load_data,
    TIMEFRAME_FOLDERS,
    get_cache_status,
    refresh_all_data,
    clear_cache
)

# ============================================================
# SCANNER ENGINE
# ============================================================

from scanner_engine import (
    SCANNERS,
    run_scanner,
    trim_df_to_date,
    run_all_scanners_for_symbol
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="NSE Stock Scanner API",
    description="Multi-Timeframe NSE Stock Scanner",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://allscans.raosab.in",
        "https://egavsiv.github.io",
        "http://localhost",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    """
    Convert Pandas / NumPy values into JSON-safe values.
    """

    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()

    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

    if pd.isna(value):
        return None

    return value


def dataframe_to_records(df):

    if df is None or df.empty:
        return []

    records = []

    for row in df.to_dict(orient="records"):

        clean_row = {
            key: clean_value(value)
            for key, value in row.items()
        }

        records.append(clean_row)

    return records


def get_last_candle(timeframe):

    """
    Get the latest candle timestamp
    from all symbols in a timeframe.
    """

    try:

        data = load_data(timeframe)

        if not data:
            return None

        latest_time = None

        for symbol, df in data.items():

            if df is None or df.empty:
                continue

            try:

                timestamp = df.index[-1]

                if latest_time is None:

                    latest_time = timestamp

                elif timestamp > latest_time:

                    latest_time = timestamp

            except Exception:
                continue

        return latest_time

    except Exception as e:

        print(
            f"❌ Error getting last candle "
            f"for {timeframe}: {e}"
        )

        return None


# ============================================================
# REQUEST MODELS
# ============================================================

class ScanRequest(BaseModel):

    scanner: str
    timeframe: str = "Daily"
    analysis_date: date | None = None


class MatrixRequest(BaseModel):

    symbol: str
    timeframe: str = "Daily"
    analysis_date: date | None = None


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "NSE Stock Scanner API is running",
        "status": "ok"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "service": "NSE Stock Scanner API"
    }


# ============================================================
# GET TIMEFRAMES
# ============================================================

@app.get("/api/timeframes")
def timeframes():

    return list(
        TIMEFRAME_FOLDERS.keys()
    )


# ============================================================
# GET SCANNERS
# ============================================================

@app.get("/api/scanners")
def scanners():

    return SCANNERS


# ============================================================
# GET SYMBOLS
# ============================================================

@app.get("/api/symbols")
def symbols(
    timeframe: str = "Daily"
):

    if timeframe not in TIMEFRAME_FOLDERS:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe: {timeframe}"
        )

    try:

        data = load_data(timeframe)

        return sorted(
            data.keys()
        )

    except Exception as e:

        print(
            f"❌ Symbols API error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# GET LAST CANDLE TIMES
# ============================================================

@app.get("/api/last-candles")
def last_candles():

    output = {}

    for timeframe in TIMEFRAME_FOLDERS:

        latest = get_last_candle(
            timeframe
        )

        output[timeframe] = (

            latest.isoformat()
            if latest is not None
            else None

        )

    return output


# ============================================================
# GET CACHE STATUS
# ============================================================

@app.get("/api/cache-status")
def cache_status():

    return get_cache_status()


# ============================================================
# REFRESH ALL DATA
# ============================================================

@app.post("/api/refresh-data")
def refresh_data():

    try:

        results = refresh_all_data()

        summary = {}

        for timeframe, data in results.items():

            summary[timeframe] = {
                "symbols": len(data)
            }

        return {
            "status": "success",
            "message": "All data refreshed successfully",
            "data": summary
        }

    except Exception as e:

        print(
            f"❌ Refresh error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# CLEAR CACHE
# ============================================================

@app.post("/api/clear-cache")
def clear_data_cache():

    try:

        clear_cache()

        return {
            "status": "success",
            "message": "Data cache cleared"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# RUN SCANNER
# ============================================================

@app.post("/api/scan")
def scan(
    request: ScanRequest
):

    # --------------------------------------------------------
    # VALIDATE TIMEFRAME
    # --------------------------------------------------------

    if request.timeframe not in TIMEFRAME_FOLDERS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid timeframe: "
                f"{request.timeframe}"
            )
        )

    # --------------------------------------------------------
    # VALIDATE SCANNER
    # --------------------------------------------------------

    valid_scanners = {
        item["name"]
        for item in SCANNERS
    }

    if request.scanner not in valid_scanners:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid scanner: "
                f"{request.scanner}"
            )
        )

    try:

        # ----------------------------------------------------
        # GET ANALYSIS DATE
        # ----------------------------------------------------

        if request.analysis_date:

            anchor_date = request.analysis_date

        else:

            latest = get_last_candle(
                request.timeframe
            )

            if latest is not None:

                anchor_date = latest.date()

            else:

                anchor_date = date.today()

        print(
            f"\n🔍 RUNNING SCANNER"
        )

        print(
            f"Scanner: {request.scanner}"
        )

        print(
            f"Timeframe: {request.timeframe}"
        )

        print(
            f"Analysis Date: {anchor_date}"
        )

        # ----------------------------------------------------
        # RUN SCANNER ENGINE
        # ----------------------------------------------------

        result_df = run_scanner(

            request.scanner,
            request.timeframe,
            anchor_date,
            load_data

        )

        # ----------------------------------------------------
        # RSI ZONE DISTRIBUTION
        # ----------------------------------------------------

        zones = {}

        if (

            request.scanner
            == "RSI Market Pulse"

            and result_df is not None

            and not result_df.empty

            and "Zone" in result_df.columns

        ):

            zones = (

                result_df["Zone"]
                .astype(str)
                .value_counts()
                .to_dict()

            )

        # ----------------------------------------------------
        # RETURN RESPONSE
        # ----------------------------------------------------

        return {

            "status": "success",

            "scanner":
                request.scanner,

            "timeframe":
                request.timeframe,

            "analysis_date":
                str(anchor_date),

            "total_matches":

                len(result_df)

                if result_df is not None
                else 0,

            "zones":
                zones,

            "results":

                dataframe_to_records(
                    result_df
                )

        }

    except Exception as e:

        print(
            f"❌ Scanner API error: {e}"
        )

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# SINGLE STOCK SCANNER MATRIX
# ============================================================

@app.post("/api/matrix")
def matrix(
    request: MatrixRequest
):

    # --------------------------------------------------------
    # VALIDATE TIMEFRAME
    # --------------------------------------------------------

    if request.timeframe not in TIMEFRAME_FOLDERS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid timeframe: "
                f"{request.timeframe}"
            )
        )

    try:

        # ----------------------------------------------------
        # LOAD CURRENT TIMEFRAME DATA
        # ----------------------------------------------------

        data = load_data(
            request.timeframe
        )

        if not data:

            raise HTTPException(
                status_code=404,
                detail="No data found"
            )

        # ----------------------------------------------------
        # CHECK SYMBOL
        # ----------------------------------------------------

        if request.symbol not in data:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Symbol not found: "
                    f"{request.symbol}"
                )
            )

        # ----------------------------------------------------
        # GET ANALYSIS DATE
        # ----------------------------------------------------

        if request.analysis_date:

            anchor_date = request.analysis_date

        else:

            latest = get_last_candle(
                request.timeframe
            )

            if latest is not None:

                anchor_date = latest.date()

            else:

                anchor_date = date.today()

        # ----------------------------------------------------
        # TRIM SYMBOL DATA
        # ----------------------------------------------------

        symbol_df = trim_df_to_date(

            data[request.symbol],

            anchor_date

        )

        if symbol_df is None:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Not enough data for "
                    "selected symbol/date"
                )
            )

        # ----------------------------------------------------
        # LOAD ALL TIMEFRAMES
        # ----------------------------------------------------

        print(
            f"\n📊 RUNNING MATRIX"
        )

        print(
            f"Symbol: {request.symbol}"
        )

        print(
            f"Timeframe: "
            f"{request.timeframe}"
        )

        data_all_tfs = {}

        for timeframe in TIMEFRAME_FOLDERS:

            data_all_tfs[timeframe] = (
                load_data(timeframe)
            )

        # ----------------------------------------------------
        # RUN ALL SCANNERS
        # ----------------------------------------------------

        result = run_all_scanners_for_symbol(

            request.symbol,

            symbol_df,

            request.timeframe,

            anchor_date,

            data_all_tfs

        )

        # ----------------------------------------------------
        # FORMAT RESULT
        # ----------------------------------------------------

        formatted_results = []

        for scanner_name, scanner_result in result.items():

            formatted_results.append({

                "Scanner":
                    scanner_name,

                "Result":
                    bool(scanner_result)

            })

        return {

            "status": "success",

            "symbol":
                request.symbol,

            "timeframe":
                request.timeframe,

            "analysis_date":
                str(anchor_date),

            "results":
                formatted_results

        }

    except HTTPException:

        raise

    except Exception as e:

        print(
            f"❌ Matrix API error: {e}"
        )

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "api:app",

        host="0.0.0.0",

        port=8000,

        reload=True

    )
