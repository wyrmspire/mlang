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

from src.config import ONE_MIN_PARQUET_DIR, PATTERNS_DIR, LOCAL_TZ
from src.generator import get_generator
from src.utils.logging_utils import get_logger
from src.yfinance_loader import get_yfinance_loader
from src.model_inference import ModelInference, get_available_models

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


# ============================================================================
# YFinance Endpoints for Historical Playback
# ============================================================================

class YFinanceFetchRequest(BaseModel):
    symbol: str = "ES=F"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days_back: int = 7


class TradeSignal(BaseModel):
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    confidence: float
    entry_time: float


@app.get("/api/yfinance/models")
def get_models_list():
    """Get list of available trained models."""
    try:
        models = get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {"models": []}


@app.post("/api/yfinance/fetch")
def fetch_yfinance_data(req: YFinanceFetchRequest):
    """
    Fetch data from YFinance.
    Returns 1-minute candles for the specified period.
    """
    try:
        loader = get_yfinance_loader(req.symbol)
        df = loader.fetch_data(
            start_date=req.start_date,
            end_date=req.end_date,
            days_back=req.days_back
        )
        
        if df.empty:
            return {
                "success": False,
                "message": "No data available for the specified period",
                "data": [],
                "dates": []
            }
        
        # Get available dates
        dates = loader.get_available_dates(df)
        
        # Convert to list of candles
        candles = []
        for _, row in df.iterrows():
            candles.append({
                "time": int(row['time'].timestamp()),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
        
        return {
            "success": True,
            "data": candles,
            "dates": dates,
            "symbol": req.symbol
        }
        
    except Exception as e:
        logger.error(f"Error fetching YFinance data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class PlaybackRequest(BaseModel):
    symbol: str = "ES=F"
    date: str
    model_name: str = "rejection_cnn_v1"
    risk_amount: float = 300.0
    days_back: int = 14  # History to load for analysis


@app.post("/api/yfinance/playback/analyze")
def analyze_playback_candle(
    candle_index: int = Query(..., description="Current candle index in the data"),
    model_name: str = Query("rejection_cnn_v1", description="Model to use for analysis"),
    symbol: str = Query("ES=F", description="Symbol"),
    date: str = Query(..., description="Date being analyzed"),
    days_back: int = Query(14, description="Days of history to load")
):
    """
    Analyze current market conditions and return trade signal if setup is found.
    This simulates real-time analysis during playback.
    """
    try:
        # Get the data from cache
        loader = get_yfinance_loader(symbol)
        
        # Fetch data for the date (will use cache if available)
        df = loader.fetch_data(days_back=days_back)  # Get enough history
        
        if df.empty:
            return {"signal": None}
        
        # Filter to requested date and earlier (can only look back)
        df = df[df['date'] <= date].copy()
        
        if len(df) <= candle_index:
            return {"signal": None}
        
        # Initialize model inference
        model_inf = ModelInference(model_name)
        
        # Check for trading setup at current position
        setup = model_inf.check_for_setup(df, candle_index)
        
        if setup is None:
            return {"signal": None}
        
        # Return signal with entry time
        return {
            "signal": {
                "direction": setup["direction"],
                "entry_price": setup["entry_price"],
                "sl_price": setup["sl_price"],
                "tp_price": setup["tp_price"],
                "confidence": setup["confidence"],
                "entry_time": int(df.iloc[candle_index]['time'].timestamp()),
                "atr": setup.get("atr", 0),
                "setup_type": setup.get("setup_type", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing playback: {e}")
        return {"signal": None}
