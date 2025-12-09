import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Subdirectories for processed data
ONE_MIN_PARQUET_DIR = PROCESSED_DIR / "mes_1min_parquet"
HOUR_FEATURES_DIR = PROCESSED_DIR / "mes_hour_features_parquet"
PATTERNS_DIR = PROCESSED_DIR / "mes_patterns"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DIR, ONE_MIN_PARQUET_DIR, HOUR_FEATURES_DIR, PATTERNS_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data Constants
MES_PREFIX = "MES"
LOCAL_TZ = "America/Chicago"  # Central Time for CME session logic

# Generator Constants
MIN_HOURS_FOR_PATTERN = 5  # Minimum samples to form a cluster bucket
DEFAULT_CLUSTERS = 3        # Default k for k-means
