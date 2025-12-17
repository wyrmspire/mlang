"""
MLang2 Configuration
Central configuration for paths, constants, and defaults.
"""

from pathlib import Path
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import List

# =============================================================================
# BASE PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = BASE_DIR / "cache"
SHARDS_DIR = BASE_DIR / "shards"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DIR, CACHE_DIR, SHARDS_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TIMEZONE
# =============================================================================

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
DEFAULT_TZ = NY_TZ

# =============================================================================
# SESSION TIMES (New York)
# =============================================================================

SESSION_RTH_START = "09:30"   # Regular Trading Hours
SESSION_RTH_END = "16:00"
SESSION_GLOBEX_START = "18:00"
SESSION_GLOBEX_END = "09:30"

# =============================================================================
# INSTRUMENT CONSTANTS (MES)
# =============================================================================

TICK_SIZE = 0.25
POINT_VALUE = 5.0
COMMISSION_PER_SIDE = 1.25  # ~$2.50 round trip

# =============================================================================
# INDICATOR DEFAULTS
# =============================================================================

DEFAULT_EMA_PERIOD = 200
DEFAULT_RSI_PERIOD = 14
DEFAULT_ADR_PERIOD = 14
DEFAULT_ATR_PERIOD = 14

# =============================================================================
# FEATURE DEFAULTS
# =============================================================================

DEFAULT_LOOKBACK_MINUTES = 120  # 2 hours
DEFAULT_LOOKBACK_1M = 120       # 2 hours of 1m bars
DEFAULT_LOOKBACK_5M = 24        # 2 hours of 5m bars
DEFAULT_LOOKBACK_15M = 8        # 2 hours of 15m bars

# =============================================================================
# SIMULATION DEFAULTS
# =============================================================================

DEFAULT_MAX_BARS_IN_TRADE = 200
DEFAULT_SLIPPAGE_TICKS = 0.5

# =============================================================================
# DATA FILES
# =============================================================================

CONTINUOUS_CONTRACT_PATH = RAW_DATA_DIR / "continuous_contract.json"
