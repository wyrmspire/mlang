
import pandas as pd
import json
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, PROCESSED_DIR

def process_continuous():
    raw_path = DATA_DIR / "raw" / "continuous_contract.json"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found.")
        return

    print(f"Loading {raw_path}...")
    with open(raw_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} records. Converting to DataFrame...")
    df = pd.DataFrame(data)
    
    # Convert time
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    
    # Basic Clean
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df.astype(float)
    
    # Check for gaps?
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    
    out_path = PROCESSED_DIR / "continuous_1m.parquet"
    df.to_parquet(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    process_continuous()
