"""
Data Skills
Workflows for data ingestion, processing, and sharding.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd

from src.data.loader import load_continuous_contract, save_processed
from src.data.resample import resample_all_timeframes
from src.config import CONTINUOUS_CONTRACT_PATH, PROCESSED_DIR

def ingest_raw_data(source_path: Optional[Path] = None) -> dict:
    """
    Skill: Ingest raw JSON data, process it, and save as Parquet.
    Returns a dict with paths to processed files.
    """
    source_path = source_path or CONTINUOUS_CONTRACT_PATH
    print(f"Ingesting raw data from {source_path}")
    
    # 1. Load raw
    df = load_continuous_contract(source_path)
    
    # 2. Resample all timeframes
    htf_data = resample_all_timeframes(df)
    
    # 3. Save processed files
    results = {}
    for tf, tf_df in htf_data.items():
        name = f"continuous_{tf}"
        path = save_processed(tf_df, name)
        results[tf] = path
        print(f"  Saved {tf} to {path}")
    
    return results

def get_data_summary() -> str:
    """
    Skill: Provide a human/agent readable summary of available data.
    """
    if not PROCESSED_DIR.exists():
        return "No processed data found. Run ingest_raw_data() first."
    
    files = list(PROCESSED_DIR.glob("*.parquet"))
    if not files:
        return "No processed data found in data/processed."
    
    summary = ["Available processed data:"]
    for f in files:
        # Get basic stats
        df = pd.read_parquet(f)
        start = df['time'].min()
        end = df['time'].max()
        summary.append(f"- {f.name}: {len(df)} bars ({start} to {end})")
    
    return "\n".join(summary)
