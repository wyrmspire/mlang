import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import ONE_MIN_PARQUET_DIR

def test():
    print("Start Loading...", flush=True)
    input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows.", flush=True)
    
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time').sort_index()
    print("Index set.", flush=True)
    
    print("Resampling...", flush=True)
    df_5m = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    print(f"Resampled to {len(df_5m)} rows.", flush=True)

if __name__ == "__main__":
    test()
