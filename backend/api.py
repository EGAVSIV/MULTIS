from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
import math
import pandas as pd
from data_loader import load_timeframe, last_candle, TIMEFRAME_FOLDERS
from scanner_engine import SCANNERS, run_scanner, trim_df_to_date, run_all_scanners_for_symbol

app = FastAPI(title="NSE Stock Scanner API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

def clean(v):
    if isinstance(v, (pd.Timestamp, date)): return str(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    return v

def records(df):
    return [{k: clean(v) for k,v in r.items()} for r in df.to_dict(orient="records")]

class ScanRequest(BaseModel):
    scanner: str
    timeframe: str = "Daily"
    analysis_date: date | None = None

class MatrixRequest(BaseModel):
    symbol: str
    timeframe: str = "Daily"
    analysis_date: date | None = None

@app.get("/api/health")
def health(): return {"status":"ok"}

@app.get("/api/timeframes")
def timeframes(): return list(TIMEFRAME_FOLDERS.keys())

@app.get("/api/scanners")
def scanners(): return SCANNERS

@app.get("/api/symbols")
def symbols(timeframe: str="Daily"):
    if timeframe not in TIMEFRAME_FOLDERS: raise HTTPException(400,"Invalid timeframe")
    return sorted(load_timeframe(timeframe).keys())

@app.get("/api/last-candles")
def last_candles():
    return {tf: (last_candle(tf).isoformat() if last_candle(tf) is not None else None) for tf in TIMEFRAME_FOLDERS}

@app.post("/api/scan")
def scan(req: ScanRequest):
    if req.timeframe not in TIMEFRAME_FOLDERS: raise HTTPException(400,"Invalid timeframe")
    valid = {x["name"] for x in SCANNERS}
    if req.scanner not in valid: raise HTTPException(400,"Invalid scanner")
    anchor = req.analysis_date or (last_candle(req.timeframe).date() if last_candle(req.timeframe) is not None else date.today())
    df = run_scanner(req.scanner, req.timeframe, anchor, load_timeframe)
    zones = {}
    if req.scanner == "RSI Market Pulse" and not df.empty and "Zone" in df:
        zones = df["Zone"].astype(str).value_counts().to_dict()
    return {"scanner":req.scanner,"timeframe":req.timeframe,"analysis_date":str(anchor),"total_matches":len(df),"zones":zones,"results":records(df)}

@app.post("/api/matrix")
def matrix(req: MatrixRequest):
    if req.timeframe not in TIMEFRAME_FOLDERS: raise HTTPException(400,"Invalid timeframe")
    data = load_timeframe(req.timeframe)
    if req.symbol not in data: raise HTTPException(404,"Symbol not found")
    anchor = req.analysis_date or (last_candle(req.timeframe).date() if last_candle(req.timeframe) is not None else date.today())
    df = trim_df_to_date(data[req.symbol], anchor)
    if df is None: raise HTTPException(400,"Not enough data")
    all_data = {tf: load_timeframe(tf) for tf in TIMEFRAME_FOLDERS}
    result = run_all_scanners_for_symbol(req.symbol, df, req.timeframe, anchor, all_data)
    return {"symbol":req.symbol,"timeframe":req.timeframe,"analysis_date":str(anchor),"results":[{"Scanner":k,"Result":bool(v)} for k,v in result.items()]}
