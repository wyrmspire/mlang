from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import yfinance as yf
from datetime import datetime, timedelta

from src.config import ONE_MIN_PARQUET_DIR, PATTERNS_DIR, LOCAL_TZ
from src.generator import get_generator
from src.utils.logging_utils import get_logger

logger = get_logger("api")

app = FastAPI(title="MES Pattern Generator API")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(f"Validation Error: {exc.errors()}")
    logger.error(f"Request Body: {body.decode()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode()},
    )

# Allow CORS for frontend

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data cache
_data_cache = {
    "1min": None,
    "last_loaded": None
}

def get_1min_data():
    """
    Load or return cached 1-minute data.
    """
    parquet_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    if not parquet_path.exists():
        return pd.DataFrame()
    
    # Reload if null or if file changed? For now simple cache
    if _data_cache["1min"] is None:
        logger.info("Loading 1-min parquet into memory...")
        df = pd.read_parquet(parquet_path)
        # Ensure date is string for JSON
        df['time_str'] = df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S') 
        # Convert date column to string
        df['date_str'] = df['date'].astype(str)
        _data_cache["1min"] = df
        
    return _data_cache["1min"]

# --- Models ---
class GenerateSessionRequest(BaseModel):
    day_of_week: int
    session_type: str = "RTH"
    start_price: float = 5800.0
    date: Optional[str] = None # YYYY-MM-DD
    timeframe: str = "1m"

class Candle(BaseModel):
    time: float # Unix timestamp (seconds)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0
    synthetic_day: Optional[int] = None

class CandleResponse(BaseModel):
    data: List[Candle]

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/dates")
def get_available_dates():
    df = get_1min_data()
    if df.empty:
        return []
    # Get unique dates
    dates = df['date_str'].unique().tolist()
    # Sort descending
    dates.sort(reverse=True)
    return dates

def _resample_df(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "1m":
        return df
        
    rule_map = {"1m": "1T", "5m": "5T", "15m": "15T", "30m": "30T", "60m": "60T", "1h": "60T"}
    rule = rule_map.get(timeframe, "1T")
    
    # Check if 'time' is index or column
    if 'time' in df.columns:
        df = df.set_index('time')
        
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Handle synthetic_day if present (take mode? or first?)
    # Usually synthetic day is constant for a day, but across boundary...
    # For resampling, 'first' is safe enough for day index
    if 'synthetic_day' in df.columns:
        agg_dict['synthetic_day'] = 'first'
        
    resampled = df.resample(rule).agg(agg_dict).dropna()
    return resampled.reset_index()

@app.get("/api/candles")
def get_candles(
    date: str = Query(..., description="Date YYYY-MM-DD"),
    timeframe: str = "1m",
    session_type: Optional[str] = None
):
    df = get_1min_data()
    if df.empty:
        return {"data": []}
    
    # Filter by date
    subset = df[df['date_str'] == date].copy()
    if subset.empty:
        return {"data": []}
        
    if session_type and session_type != "all":
        subset = subset[subset['session_type'] == session_type]
        
    subset = _resample_df(subset, timeframe)
        
    results = []
    for _, row in subset.iterrows():
        results.append({
            "time": int(row['time'].timestamp()),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row.get('volume', 0)
        })
        
    return {"data": results}

@app.post("/api/generate/session")
def generate_session(req: GenerateSessionRequest):
    gen = get_generator()
    
    synthetic_df = gen.generate_session(
        day_of_week=req.day_of_week,
        session_type=req.session_type,
        start_price=req.start_price,
        date=req.date
    )
    
    if synthetic_df.empty:
        raise HTTPException(status_code=400, detail="Generation failed (no patterns?)")
    
    # Resample
    synthetic_df = _resample_df(synthetic_df, req.timeframe)
        
    # Return candles
    results = []
    for _, row in synthetic_df.iterrows():
        results.append({
            "time": int(row['time'].timestamp()),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume']
        })
        
    return {"data": results}

class GenerateMultiDayRequest(BaseModel):
    num_days: int
    session_type: str = "RTH"
    initial_price: float = 5800.0
    start_date: Optional[str] = None # YYYY-MM-DD
    timeframe: str = "1m"

@app.get("/api/candles-range")
def get_candles_range(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    timeframe: str = "1m",
    session_type: Optional[str] = None
):
    df = get_1min_data()
    if df.empty: return {"data": []}
    
    # Filter by date range (inclusive strings work for ISO dates)
    subset = df[(df['date_str'] >= start_date) & (df['date_str'] <= end_date)].copy()
    
    if session_type and session_type != "all":
        subset = subset[subset['session_type'] == session_type]
        
    subset = _resample_df(subset, timeframe)

    results = []
    for _, row in subset.iterrows():
        results.append({
            "time": int(row['time'].timestamp()),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row.get('volume', 0)
        })
    return {"data": results}

@app.post("/api/generate/multi-day")
def generate_multi_day(req: GenerateMultiDayRequest):
    gen = get_generator()
    synthetic_df = gen.generate_multi_day(
        num_days=req.num_days,
        session_type=req.session_type,
        initial_price=req.initial_price,
        start_date=req.start_date
    )
    
    if synthetic_df.empty:
        raise HTTPException(status_code=400, detail="Generation failed")
    
    # Resample
    synthetic_df = _resample_df(synthetic_df, req.timeframe)
        
    results = []
    for _, row in synthetic_df.iterrows():
        results.append({
            "time": int(row['time'].timestamp()),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume'],
            "synthetic_day": row.get('synthetic_day', 0)
        })
    return {"data": results}

@app.get("/api/setup-stats")
def get_setup_stats(min_samples: int = 100, min_expansion_rate: float = 0.5):
    """
    Return setups (clusters) with basic statistics.
    """
    from src.config import PROCESSED_DIR
    
    meta_path = PROCESSED_DIR / "mes_setup_rules.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Setup metadata not found. Run setup_miner.py first.")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    setup_stats = meta.get("setup_stats", [])
    
    # Filter
    filtered = [
        s for s in setup_stats
        if s["count"] >= min_samples and s["exp_rate"] >= min_expansion_rate
    ]
    
    # Sort by expansion rate desc
    filtered.sort(key=lambda x: x["exp_rate"], reverse=True)
    
    return {
        "setups": filtered,
        "features": meta.get("feature_cols", []),
        "meta": meta
    }

@app.get("/api/setup-rules")
def get_setup_rules():
    """
    Return the decision tree rules text.
    """
    from src.config import PROCESSED_DIR
    path = PROCESSED_DIR / "mes_setup_decision_tree.json"
    if not path.exists():
         raise HTTPException(status_code=404, detail="Rules not found")
         
    with open(path, 'r') as f:
        data = json.load(f)
        
    return data

@app.get("/api/pattern-buckets")
def get_pattern_buckets():
    gen = get_generator()
    if not gen.cluster_meta:
        return []
        
    summary = []
    for m in gen.cluster_meta:
        summary.append({
            "session": m['session_type'],
            "dow": m['day_of_week'],
            "hour": m['hour_bucket'],
            "k": m['k'],
            "samples": m['total_samples']
        })
    return summary

# --- YFinance Endpoints ---

@app.get("/api/yfinance/candles/mock")
def get_mock_candles(
    bars: int = Query(500, description="Number of bars to generate"),
    timeframe: str = Query("1m", description="Timeframe interval")
):
    """
    Generate mock OHLC data for testing playback mode.
    Creates realistic price movement with trends and volatility.
    """
    import random
    
    # Starting parameters
    start_price = 5800.0
    volatility = 2.0
    trend = 0.05
    
    # Map timeframe to seconds
    interval_map = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '60m': 3600,
        '1h': 3600
    }
    interval_seconds = interval_map.get(timeframe, 60)  # Default to 1m if unknown
    
    # Generate timestamp (starting from now - bars * interval)
    end_time = int(datetime.now().timestamp())
    start_time = end_time - (bars * interval_seconds)
    
    results = []
    current_price = start_price
    
    for i in range(bars):
        timestamp = start_time + (i * interval_seconds)
        
        # Generate OHLC
        open_price = current_price
        
        # Random walk with slight trend
        change_pct = random.gauss(trend / bars, volatility / 100)
        close_price = open_price * (1 + change_pct)
        
        # High/Low with some intrabar movement
        intrabar_range = abs(close_price - open_price) * random.uniform(1.2, 2.0)
        high_price = max(open_price, close_price) + random.uniform(0, intrabar_range)
        low_price = min(open_price, close_price) - random.uniform(0, intrabar_range)
        
        volume = random.uniform(1000, 5000)
        
        results.append({
            "time": timestamp,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": round(volume, 0)
        })
        
        current_price = close_price
    
    # Update cache for playback analysis
    # Convert to DataFrame with time as column (not index) for model inference
    df_cache = pd.DataFrame(results)
    df_cache['time'] = pd.to_datetime(df_cache['time'], unit='s')
    
    _last_yf_cache['symbol'] = 'MOCK'
    _last_yf_cache['data'] = df_cache
    _last_yf_cache['interval'] = timeframe
    
    logger.info(f"Generated {len(results)} mock candles")
    return {"data": results, "symbol": "MOCK", "timeframe": timeframe}

@app.get("/api/yfinance/candles")
def get_yfinance_candles(
    symbol: str = Query("MES=F", description="Ticker symbol (e.g., MES=F, ES=F, AAPL)"),
    days: int = Query(5, description="Number of days of historical 1m data to fetch"),
    timeframe: str = Query("1m", description="Timeframe (1m, 5m, 15m, 60m, 1h, 1d)")
):
    """
    Fetch 1-minute OHLC data from yfinance for a given symbol.
    Default symbol is MES=F (Micro E-mini S&P 500).
    """
    try:
        # Calculate date range (yahoo finance 1m only goes back ~7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching {symbol} 1m data from {start_date} to {end_date}")
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        # Map timeframe to yfinance interval if needed, or just use timeframe if valid
        # Valid: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        # We use '1m' or '5m' generally.
        yf_interval = timeframe if timeframe in ['1m', '5m', '15m', '60m', '1h', '1d'] else "1m"
        
        df = ticker.history(start=start_date, end=end_date, interval=yf_interval)
        
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data returned for {symbol}. Check symbol spelling.")
            
        # Update Cache
        _last_yf_cache['symbol'] = symbol
        _last_yf_cache['data'] = df
        _last_yf_cache['interval'] = yf_interval

        
        # No manual resampling needed if we fetched the correct interval
        # But we might want to standardize columns
        
        # Convert to candle format
        results = []
        for idx, row in df.iterrows():
            results.append({
                "time": int(idx.timestamp()),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row.get('Volume', 0))
            })
        
        logger.info(f"Returning {len(results)} candles for {symbol}")
        return {"data": results, "symbol": symbol, "timeframe": timeframe}
    
    except Exception as e:
        logger.error(f"YFinance fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"YFinance error: {str(e)}")

# --- Analysis Endpoint (Re-implemented) ---
from src.model_inference import ModelInference

# Global inference cache
_inference_engine = None

def get_inference_engine(model_name: str):
    global _inference_engine
    if _inference_engine is None or _inference_engine.model_name != model_name:
        _inference_engine = ModelInference(model_name)
    return _inference_engine

@app.post("/api/yfinance/playback/analyze")
def analyze_playback_candle(
    candle_index: int = Query(..., description="Index in the client's data array"),
    model_name: str = Query(..., description="Model to use"),
    symbol: str = Query("MES=F"),
    date: str = Query(...),
    # We need the full context. Client sends index.
    # But backend is stateless regarding the client's specific array? 
    # NO. The Backend `get_yfinance_candles` fetches data but doesn't persist it for this session uniquely?
    # Actually, `ModelInference.analyze` needs the DATAFRAME.
    # We can't easily re-fetch yfinance data on every tick safely/quickly.
    # SOLUTION: Client should probably send the last N candles? Or Backend caches the LAST fetched yfinance data?
    # Let's use a simple global cache for the last fetched symbol/data in memory.
    # See `_last_yf_cache` below.
): 
    # Need access to data
    # Allow mock data to be used for any symbol
    if _last_yf_cache['symbol'] != symbol and _last_yf_cache['symbol'] != 'MOCK':
         # Warn or try to re-fetch?
         # If client just called get_yfinance_candles, it should be here.
         raise HTTPException(status_code=400, detail=f"Data not cached for {symbol}. Call fetch first. Cached: {_last_yf_cache['symbol']}")
         
    df = _last_yf_cache['data']
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data cached.")
        
    engine = get_inference_engine(model_name)
    
    # Run analysis
    # Note: candle_index must align with the cached dataframe
    result = engine.analyze(candle_index, df)
    
    return {"signal": result["signal"] if result else None}

# Simple cache system for YFinance data to support playback analysis
_last_yf_cache = {
    "symbol": None,
    "data": None, # DataFrame
    "interval": None
}

# Update fetch to populate cache

@app.get("/api/yfinance/models")
def get_yfinance_models():
    """Get list of available models for playback."""
    from src.model_inference import get_available_models
    models = get_available_models()
    return {"models": models}

