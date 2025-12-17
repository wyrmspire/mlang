"""
Data Loader
Load continuous contract data from JSON.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from src.config import CONTINUOUS_CONTRACT_PATH, NY_TZ, PROCESSED_DIR


def load_continuous_contract(
    path: Optional[Path] = None,
    tz: ZoneInfo = NY_TZ
) -> pd.DataFrame:
    """
    Load continuous contract data from JSON file.
    
    Args:
        path: Path to JSON file. Defaults to CONTINUOUS_CONTRACT_PATH.
        tz: Target timezone. Defaults to NY_TZ.
        
    Returns:
        DataFrame with columns: time, open, high, low, close, volume
        Index is NOT set (time is a column).
        Time column is timezone-aware in target tz.
    """
    path = path or CONTINUOUS_CONTRACT_PATH
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Could be {time: [...], open: [...], ...} format
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data)}")
    
    # Normalize column names
    df.columns = df.columns.str.lower()
    
    # Ensure required columns exist
    required = ['time', 'open', 'high', 'low', 'close']
    missing = set(required) - set(df.columns)
    if missing:
        # Try alternate names
        renames = {
            'timestamp': 'time',
            'datetime': 'time',
            'date': 'time',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
        }
        df = df.rename(columns=renames)
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # Add volume if missing
    if 'volume' not in df.columns:
        df['volume'] = 0
    
    # Parse time
    df['time'] = pd.to_datetime(df['time'], utc=True)
    
    # Convert to target timezone
    df['time'] = df['time'].dt.tz_convert(tz)
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_processed_1m(
    name: str = "continuous_1m",
    tz: ZoneInfo = NY_TZ
) -> pd.DataFrame:
    """
    Load processed 1-minute parquet file.
    Falls back to loading from JSON if parquet doesn't exist.
    """
    parquet_path = PROCESSED_DIR / f"{name}.parquet"
    
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time']).dt.tz_convert(tz)
        return df
    
    # Fall back to JSON
    return load_continuous_contract(tz=tz)


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Save processed DataFrame to parquet."""
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path)
    return path
