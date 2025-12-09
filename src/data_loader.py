import json
import pandas as pd
from pathlib import Path
from typing import List, Union
from src.config import RAW_DATA_DIR, MES_PREFIX
from src.utils.logging_utils import get_logger

logger = get_logger("data_loader")

def load_mes_json_file(path: Path) -> pd.DataFrame:
    """
    Load a single JSON or NDJSON file and return a DataFrame of MES bars.
    """
    try:
        with open(path, 'r') as f:
            # Try loading as standard JSON array first
            try:
                data = json.load(f)
                if not isinstance(data, list):
                     # If it's a single object, wrap in list
                    data = [data]
            except json.JSONDecodeError:
                # Fallback to NDJSON (line delimited)
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]

        if not data:
            logger.warning(f"File {path.name} is empty or invalid.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # Filter for MES symbols
        if 'original_symbol' in df.columns:
            df = df[df['original_symbol'].str.startswith(MES_PREFIX, na=False)]
        
        # Ensure required columns exist
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"File {path.name} missing columns: {missing}")
            return pd.DataFrame()

        # Parse time (assumes ISO 8601 with Z)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        
        # Sort by time just in case
        df.sort_values('time', inplace=True)
        
        return df[['time', 'open', 'high', 'low', 'close', 'volume', 'original_symbol']]

    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

def load_all_mes_bars(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load all matching JSON files from the raw directory and combine them.
    """
    all_dfs = []
    files = list(raw_dir.glob("*.json"))
    
    if not files:
        logger.warning(f"No JSON files found in {raw_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(files)} raw data files.")
    
    for p in files:
        df = load_mes_json_file(p)
        if not df.empty:
            all_dfs.append(df)
            
    if not all_dfs:
        return pd.DataFrame()
        
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Drop duplicates if any (by time and symbol)
    before_dedup = len(combined)
    combined.drop_duplicates(subset=['time', 'original_symbol'], keep='last', inplace=True)
    if len(combined) < before_dedup:
        logger.info(f"Dropped {before_dedup - len(combined)} duplicate records.")
        
    # Sort globally
    combined.sort_values('time', inplace=True)
    combined.reset_index(drop=True, inplace=True)
    
    logger.info(f"Loaded total {len(combined)} MES bars.")
    return combined

if __name__ == "__main__":
    # Test run
    df = load_all_mes_bars()
    print(df.head())
    print(df.info())
