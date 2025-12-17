import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import PROCESSED_DIR

df = pd.read_parquet(PROCESSED_DIR / "continuous_1m.parquet")
print(df.columns)
print(df.head())
