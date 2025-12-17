        test_end = test_start + test_duration
        
        # Check if we have enough data for this split
        if test_end > end_time:
            break
        
        split = Split(
            split_idx=split_idx,
            train_start=current_start,
            train_end=train_end,
            embargo_start=embargo_start,
            embargo_end=embargo_end,
            test_start=test_start,
            test_end=test_end,
        )
        splits.append(split)
        
        # Move to next window
        current_start = test_start  # Or test_end for non-overlapping
        split_idx += 1
    
    return splits


def apply_split(
    df: pd.DataFrame,
    split: Split,
    time_col: str = 'time'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a split to get train and test DataFrames.
    """
    times = pd.to_datetime(df[time_col])
    
    train_mask = (times >= split.train_start) & (times < split.train_end)
    test_mask = (times >= split.test_start) & (times < split.test_end)
    
    return df[train_mask].copy(), df[test_mask].copy()


def apply_embargo_to_records(
    train_records: list,
    test_start: pd.Timestamp,
    embargo_bars: int
) -> list:
    """
    Remove training records within embargo window of test start.
    
    Prevents information leakage from rolling features.
    """
    # Calculate cutoff time
    embargo_duration = pd.Timedelta(minutes=embargo_bars)
    cutoff = test_start - embargo_duration
    
    # Filter records
    filtered = [r for r in train_records if r.timestamp < cutoff]
    
    removed = len(train_records) - len(filtered)
    if removed > 0:
        print(f"Embargo: removed {removed} records within {embargo_bars} bars of test start")
    
    return filtered

```

### src/experiments/sweep.py

```python
"""
Parameter Sweep
Run experiments across parameter grids.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import itertools
import copy
from pathlib import Path
import json

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment, ExperimentResult
from src.config import RESULTS_DIR


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""
    base_config: ExperimentConfig
    sweep_params: Dict[str, List[Any]] = field(default_factory=dict)
    
    # e.g. {'oco_config.tp_multiple': [1.2, 1.4, 1.6],
    #       'oco_config.stop_atr': [0.8, 1.0, 1.2]}


def generate_sweep_configs(sweep: SweepConfig) -> List[ExperimentConfig]:
    """
    Generate all config combinations from sweep parameters.
    """
    if not sweep.sweep_params:
        return [sweep.base_config]
    
    # Generate all combinations
    param_names = list(sweep.sweep_params.keys())
    param_values = list(sweep.sweep_params.values())
    
    configs = []
    for values in itertools.product(*param_values):
        # Deep copy base config
        config = copy.deepcopy(sweep.base_config)
        
        # Apply parameter values
        for name, value in zip(param_names, values):
            _set_nested_attr(config, name, value)
        
        # Update name
        param_str = '_'.join(f"{n.split('.')[-1]}={v}" for n, v in zip(param_names, values))
        config.name = f"{sweep.base_config.name}_{param_str}"
        
        configs.append(config)
    
    return configs


def _set_nested_attr(obj, path: str, value):
    """Set nested attribute using dot notation."""
    parts = path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def run_sweep(sweep: SweepConfig) -> List[ExperimentResult]:
    """
    Run all experiments in a sweep.
    """
    configs = generate_sweep_configs(sweep)
    print(f"Running sweep with {len(configs)} configurations")
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}/{len(configs)}: {config.name} ---")
        
        try:
            result = run_experiment(config)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Save sweep results
    sweep_id = sweep.base_config.name
    save_sweep_results(results, sweep_id)
    
    return results


def save_sweep_results(results: List[ExperimentResult], sweep_id: str):
    """Save sweep results to JSON."""
    output_path = RESULTS_DIR / f"sweep_{sweep_id}.json"
    
    data = {
        'sweep_id': sweep_id,
        'num_experiments': len(results),
        'results': [r.to_dict() for r in results],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"Saved sweep results to {output_path}")

```

### src/features/__init__.py

```python
# Features module
"""Causal feature computation - no future leaking."""

```

### src/features/context.py

```python
"""
Context Features
Derived context vector (x_context) for MLP input.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.features.state import MarketState
from src.features.indicators import IndicatorValues
from src.features.levels import LevelValues
from src.features.time_features import TimeFeatures


@dataclass
class ContextFeatures:
    """
    Context feature vector for MLP input.
    
    These are scalar/low-dim features derived from indicators and state,
    separate from the raw OHLCV windows used by CNN.
    """
    # EMA distances (normalized by ATR)
    dist_ema_5m_200_atr: float = 0.0
    dist_ema_15m_200_atr: float = 0.0
    
    # VWAP distances
    dist_vwap_session_atr: float = 0.0
    dist_vwap_weekly_atr: float = 0.0
    
    # Level distances
    dist_nearest_1h_level_atr: float = 0.0
    dist_nearest_4h_level_atr: float = 0.0
    dist_pdh_atr: float = 0.0
    dist_pdl_atr: float = 0.0
    
    # Volatility
    adr_pct_used: float = 0.0
    
    # Momentum
    rsi_5m_14: float = 50.0
    rsi_15m_14: float = 50.0
    
    # Volume
    relative_volume: float = 1.0
    
    # Time (cyclical)
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    dow_sin: float = 0.0
    dow_cos: float = 0.0
    
    # Time (flags)
    is_rth: float = 0.0
    is_first_hour: float = 0.0
    is_last_hour: float = 0.0
    mins_into_session: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.dist_ema_5m_200_atr,
            self.dist_ema_15m_200_atr,
            self.dist_vwap_session_atr,
            self.dist_vwap_weekly_atr,
            self.dist_nearest_1h_level_atr,
            self.dist_nearest_4h_level_atr,
            self.dist_pdh_atr,
            self.dist_pdl_atr,
            self.adr_pct_used,
            self.rsi_5m_14 / 100.0,  # Normalize to [0, 1]
            self.rsi_15m_14 / 100.0,
            self.relative_volume,
            self.hour_sin,
            self.hour_cos,
            self.dow_sin,
            self.dow_cos,
            self.is_rth,
            self.is_first_hour,
            self.is_last_hour,
            self.mins_into_session / 390.0,  # Normalize by RTH length
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered feature names."""
        return [
            'dist_ema_5m_200_atr',
            'dist_ema_15m_200_atr',
            'dist_vwap_session_atr',
            'dist_vwap_weekly_atr',
            'dist_nearest_1h_level_atr',
            'dist_nearest_4h_level_atr',
            'dist_pdh_atr',
            'dist_pdl_atr',
            'adr_pct_used',
            'rsi_5m_14_norm',
            'rsi_15m_14_norm',
            'relative_volume',
            'hour_sin',
            'hour_cos',
            'dow_sin',
            'dow_cos',
            'is_rth',
            'is_first_hour',
            'is_last_hour',
            'mins_into_session_norm',
        ]
    
    @staticmethod
    def dim() -> int:
        """Get feature dimension."""
        return 20


def compute_context_features(
    current_price: float,
    indicators: IndicatorValues,
    levels: LevelValues,
    time_features: TimeFeatures,
    atr: float = 1.0
) -> ContextFeatures:
    """
    Compute context feature vector from component features.
    
    All distances are normalized by ATR for scale-independence.
    """
    if atr <= 0:
        atr = 1.0
    
    # EMA distances
    dist_ema_5m = (current_price - indicators.ema_5m_200) / atr if indicators.ema_5m_200 else 0
    dist_ema_15m = (current_price - indicators.ema_15m_200) / atr if indicators.ema_15m_200 else 0
    
    # VWAP distances
    dist_vwap_session = (current_price - indicators.vwap_session) / atr if indicators.vwap_session else 0
    dist_vwap_weekly = (current_price - indicators.vwap_weekly) / atr if indicators.vwap_weekly else 0
    
    # Level distances (use nearest, sign indicates above/below)
    dist_1h = min(abs(levels.dist_1h_high), abs(levels.dist_1h_low)) / atr
    if levels.dist_1h_high < levels.dist_1h_low:
        dist_1h = -dist_1h  # Closer to resistance (above)
    
    dist_4h = min(abs(levels.dist_4h_high), abs(levels.dist_4h_low)) / atr
    if levels.dist_4h_high < levels.dist_4h_low:
        dist_4h = -dist_4h
    
    return ContextFeatures(
        dist_ema_5m_200_atr=dist_ema_5m,
        dist_ema_15m_200_atr=dist_ema_15m,
        dist_vwap_session_atr=dist_vwap_session,
        dist_vwap_weekly_atr=dist_vwap_weekly,
        dist_nearest_1h_level_atr=dist_1h,
        dist_nearest_4h_level_atr=dist_4h,
        dist_pdh_atr=levels.dist_pdh / atr if levels.pdh else 0,
        dist_pdl_atr=levels.dist_pdl / atr if levels.pdl else 0,
        adr_pct_used=indicators.adr_pct_used,
        rsi_5m_14=indicators.rsi_5m_14,
        rsi_15m_14=indicators.rsi_15m_14,
        relative_volume=indicators.relative_volume,
        hour_sin=time_features.hour_sin,
        hour_cos=time_features.hour_cos,
        dow_sin=time_features.dow_sin,
        dow_cos=time_features.dow_cos,
        is_rth=float(time_features.is_rth),
        is_first_hour=float(time_features.is_first_hour),
        is_last_hour=float(time_features.is_last_hour),
        mins_into_session=float(time_features.mins_into_session),
    )

```

### src/features/indicators.py

```python
"""
Technical Indicators
EMA, RSI, ATR, ADR, VWAP calculations.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo

from src.config import (
    DEFAULT_EMA_PERIOD, DEFAULT_RSI_PERIOD, 
    DEFAULT_ATR_PERIOD, DEFAULT_ADR_PERIOD,
    NY_TZ, SESSION_RTH_START
)


# =============================================================================
# EMA
# =============================================================================

def calculate_ema(series: pd.Series, period: int = DEFAULT_EMA_PERIOD) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def add_ema(df: pd.DataFrame, period: int = DEFAULT_EMA_PERIOD, col: str = 'close') -> pd.Series:
    """Add EMA column to dataframe."""
    return calculate_ema(df[col], period)


# =============================================================================
# RSI
# =============================================================================

def calculate_rsi(series: pd.Series, period: int = DEFAULT_RSI_PERIOD) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Neutral when undefined


# =============================================================================
# ATR
# =============================================================================

def calculate_atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.Series:
    """
    Calculate Average True Range.
    
    Uses shifted ATR (value at T uses data up to T-1) to prevent look-ahead.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Shift by 1 to make it causal
    return atr.shift(1)


# =============================================================================
# ADR (Average Daily Range)
# =============================================================================

def calculate_adr(
    df: pd.DataFrame, 
    period: int = DEFAULT_ADR_PERIOD,
    tz: ZoneInfo = NY_TZ
) -> pd.Series:
    """
    Calculate Average Daily Range.
    
    Returns ADR value aligned to each bar.
    """
    # Ensure we have a time column
    if 'time' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['time']).dt.date
    elif df.index.name == 'time':
        df = df.copy()
        df['date'] = df.index.date
    else:
        raise ValueError("DataFrame must have 'time' column or datetime index")
    
    # Calculate daily range
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min'
    })
    daily['daily_range'] = daily['high'] - daily['low']
    
    # Rolling average
    daily['adr'] = daily['daily_range'].rolling(window=period).mean().shift(1)
    
    # Map back to each bar
    df['adr'] = df['date'].map(daily['adr'])
    
    return df['adr']


def get_adr_percent_used(
    current_price: float,
    daily_open: float,
    adr: float
) -> float:
    """
    Calculate how much of ADR has been consumed.
    
    Returns value in [0, 1+] where 1.0 = full ADR used.
    """
    if adr <= 0:
        return 0.0
    movement = abs(current_price - daily_open)
    return movement / adr


# =============================================================================
# VWAP
# =============================================================================

def calculate_vwap(
    df: pd.DataFrame,
    period: str = 'session',  # 'session', 'weekly', 'daily'
    tz: ZoneInfo = NY_TZ,
    session_start: str = SESSION_RTH_START
) -> pd.Series:
    """
    Calculate Volume-Weighted Average Price.
    
    Args:
        df: DataFrame with time, high, low, close, volume
        period: 'session', 'weekly', or 'daily'
        tz: Timezone for period boundaries
        session_start: Session start time (for session VWAP)
        
    Returns:
        VWAP series aligned to each bar.
    """
    df = df.copy()
    
    # Ensure we have time
    if 'time' not in df.columns:
        raise ValueError("DataFrame must have 'time' column")
    
    # Convert to target timezone
    df['time_tz'] = pd.to_datetime(df['time']).dt.tz_convert(tz)
    
    # Typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']
    
    # Determine period grouping
    if period == 'session':
        # Group by session (resets at session_start)
        hour, minute = map(int, session_start.split(':'))
        df['session_date'] = df['time_tz'].apply(
            lambda t: t.date() if t.hour >= hour else (t - pd.Timedelta(days=1)).date()
        )
        group_col = 'session_date'
        
    elif period == 'weekly':
        df['week'] = df['time_tz'].dt.isocalendar().week
        df['year'] = df['time_tz'].dt.year
        df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str)
        group_col = 'year_week'
        
    else:  # daily
        df['date'] = df['time_tz'].dt.date
        group_col = 'date'
    
    # Cumulative VWAP within each period
    df['cum_tp_vol'] = df.groupby(group_col)['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby(group_col)['volume'].cumsum()
    
    vwap = df['cum_tp_vol'] / df['cum_vol'].replace(0, np.nan)
    
    return vwap.fillna(method='ffill')


# =============================================================================
# Indicator Bundle
# =============================================================================

@dataclass
class IndicatorValues:
    """Bundle of indicator values at a point in time."""
    ema_5m_200: float = 0.0
    ema_15m_200: float = 0.0
    rsi_5m_14: float = 50.0
    rsi_15m_14: float = 50.0
    atr_5m_14: float = 0.0
    atr_15m_14: float = 0.0
    adr_14: float = 0.0
    adr_pct_used: float = 0.0
    vwap_session: float = 0.0
    vwap_weekly: float = 0.0
    relative_volume: float = 1.0


def compute_indicators_at_bar(
    bar_idx: int,
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    current_time: pd.Timestamp,
    current_price: float,
    daily_open: float = None
) -> IndicatorValues:
    """
    Compute all indicators at a specific bar.
    
    Note: For efficiency, indicators should be pre-computed and looked up.
    This function is for reference/testing.
    """
    # This is a simplified version - in practice, use FeatureStore
    # to cache these computations
    
    values = IndicatorValues()
    
    # Get indices for lookups
    # ... (implementation would look up pre-computed values)
    
    return values

```

### src/features/levels.py

```python
"""
Price Levels
Support/resistance level detection and distance calculation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from zoneinfo import ZoneInfo

from src.config import NY_TZ


@dataclass
class LevelValues:
    """Bundle of level-related values."""
    # 1h timeframe levels
    nearest_1h_high: float = 0.0
    nearest_1h_low: float = 0.0
    dist_1h_high: float = 0.0
    dist_1h_low: float = 0.0
    
    # 4h timeframe levels
    nearest_4h_high: float = 0.0
    nearest_4h_low: float = 0.0
    dist_4h_high: float = 0.0
    dist_4h_low: float = 0.0
    
    # Previous day levels
    pdh: float = 0.0   # Previous Day High
    pdl: float = 0.0   # Previous Day Low
    pdc: float = 0.0   # Previous Day Close
    dist_pdh: float = 0.0
    dist_pdl: float = 0.0
    
    # Current day
    current_day_high: float = 0.0
    current_day_low: float = 0.0


def get_htf_levels(
    df_htf: pd.DataFrame,
    current_time: pd.Timestamp,
    lookback_bars: int = 10
) -> List[Tuple[float, str]]:
    """
    Get high/low levels from higher timeframe bars.
    
    Returns list of (price, type) tuples where type is 'high' or 'low'.
    """
    # Filter to bars before current time
    mask = df_htf['time'] <= current_time
    recent = df_htf.loc[mask].tail(lookback_bars)
    
    levels = []
    for _, row in recent.iterrows():
        levels.append((row['high'], 'high'))
        levels.append((row['low'], 'low'))
    
    return levels


def get_nearest_level(
    price: float,
    levels: List[float]
) -> Tuple[float, float, str]:
    """
    Find nearest level to current price.
    
    Returns:
        (level_price, distance, 'above' or 'below')
    """
    if not levels:
        return (0.0, 0.0, 'none')
    
    above_levels = [l for l in levels if l >= price]
    below_levels = [l for l in levels if l < price]
    
    nearest_above = min(above_levels) if above_levels else None
    nearest_below = max(below_levels) if below_levels else None
    
    if nearest_above is None:
        return (nearest_below, price - nearest_below, 'below')
    if nearest_below is None:
        return (nearest_above, nearest_above - price, 'above')
    
    dist_above = nearest_above - price
    dist_below = price - nearest_below
    
    if dist_above <= dist_below:
        return (nearest_above, dist_above, 'above')
    else:
        return (nearest_below, dist_below, 'below')


def get_previous_day_levels(
    df: pd.DataFrame,
    current_time: pd.Timestamp,
    tz: ZoneInfo = NY_TZ
) -> dict:
    """
    Get previous day high, low, close.
    
    Uses New York timezone for day boundaries.
    """
    df = df.copy()
    
    # Convert to NY time
    if 'time' in df.columns:
        df['time_ny'] = pd.to_datetime(df['time']).dt.tz_convert(tz)
    else:
        df['time_ny'] = df.index.tz_convert(tz)
    
    df['date'] = df['time_ny'].dt.date
    current_date = current_time.astimezone(tz).date()
    
    # Get previous trading day
    unique_dates = sorted(df['date'].unique())
    if current_date not in unique_dates:
        # Find most recent date before current
        prev_dates = [d for d in unique_dates if d < current_date]
        if not prev_dates:
            return {'pdh': None, 'pdl': None, 'pdc': None}
        prev_date = max(prev_dates)
    else:
        idx = unique_dates.index(current_date)
        if idx == 0:
            return {'pdh': None, 'pdl': None, 'pdc': None}
        prev_date = unique_dates[idx - 1]
    
    # Get previous day data
    prev_day_data = df[df['date'] == prev_date]
    
    if prev_day_data.empty:
        return {'pdh': None, 'pdl': None, 'pdc': None}
    
    return {
        'pdh': prev_day_data['high'].max(),
        'pdl': prev_day_data['low'].min(),
        'pdc': prev_day_data['close'].iloc[-1],
    }


def compute_level_distances(
    current_price: float,
    levels: LevelValues,
    atr: float = 1.0
) -> dict:
    """
    Compute distances to all levels, normalized by ATR.
    
    Returns dict with distance values (positive = above, negative = below).
    """
    if atr <= 0:
        atr = 1.0
    
    return {
        'dist_1h_high_atr': (levels.nearest_1h_high - current_price) / atr if levels.nearest_1h_high else 0,
        'dist_1h_low_atr': (levels.nearest_1h_low - current_price) / atr if levels.nearest_1h_low else 0,
        'dist_4h_high_atr': (levels.nearest_4h_high - current_price) / atr if levels.nearest_4h_high else 0,
        'dist_4h_low_atr': (levels.nearest_4h_low - current_price) / atr if levels.nearest_4h_low else 0,
        'dist_pdh_atr': (levels.pdh - current_price) / atr if levels.pdh else 0,
        'dist_pdl_atr': (levels.pdl - current_price) / atr if levels.pdl else 0,
    }


def compute_levels_at_bar(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1m: pd.DataFrame,
    current_time: pd.Timestamp,
    current_price: float
) -> LevelValues:
    """
    Compute all level values at a specific point in time.
    """
    levels = LevelValues()
    
    # 1h levels
    if df_1h is not None and not df_1h.empty:
        h1_levels = get_htf_levels(df_1h, current_time, lookback_bars=10)
        highs_1h = [l[0] for l in h1_levels if l[1] == 'high']
        lows_1h = [l[0] for l in h1_levels if l[1] == 'low']
        
        if highs_1h:
            above = [h for h in highs_1h if h >= current_price]
            levels.nearest_1h_high = min(above) if above else max(highs_1h)
            levels.dist_1h_high = levels.nearest_1h_high - current_price
        
        if lows_1h:
            below = [l for l in lows_1h if l <= current_price]
            levels.nearest_1h_low = max(below) if below else min(lows_1h)
            levels.dist_1h_low = current_price - levels.nearest_1h_low
    
    # 4h levels
    if df_4h is not None and not df_4h.empty:
        h4_levels = get_htf_levels(df_4h, current_time, lookback_bars=6)
        highs_4h = [l[0] for l in h4_levels if l[1] == 'high']
        lows_4h = [l[0] for l in h4_levels if l[1] == 'low']
        
        if highs_4h:
            above = [h for h in highs_4h if h >= current_price]
            levels.nearest_4h_high = min(above) if above else max(highs_4h)
            levels.dist_4h_high = levels.nearest_4h_high - current_price
        
        if lows_4h:
            below = [l for l in lows_4h if l <= current_price]
            levels.nearest_4h_low = max(below) if below else min(lows_4h)
            levels.dist_4h_low = current_price - levels.nearest_4h_low
    
    # Previous day levels
    if df_1m is not None:
        pd_levels = get_previous_day_levels(df_1m, current_time)
        levels.pdh = pd_levels.get('pdh', 0) or 0
        levels.pdl = pd_levels.get('pdl', 0) or 0
        levels.pdc = pd_levels.get('pdc', 0) or 0
        levels.dist_pdh = levels.pdh - current_price if levels.pdh else 0
        levels.dist_pdl = current_price - levels.pdl if levels.pdl else 0
    
    return levels

```

### src/features/pipeline.py

```python
"""
Feature Pipeline
Compose all features into a single bundle for model input.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from src.sim.stepper import MarketStepper
from src.features.state import MarketState, get_market_state, normalize_ohlcv
from src.features.indicators import IndicatorValues
from src.features.levels import LevelValues
from src.features.time_features import TimeFeatures, compute_time_features
from src.features.context import ContextFeatures, compute_context_features
from src.config import DEFAULT_LOOKBACK_1M, DEFAULT_LOOKBACK_5M, DEFAULT_LOOKBACK_15M


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    # Lookback windows (in bars)
    lookback_1m: int = DEFAULT_LOOKBACK_1M    # 120 bars = 2 hours
    lookback_5m: int = DEFAULT_LOOKBACK_5M    # 24 bars = 2 hours
    lookback_15m: int = DEFAULT_LOOKBACK_15M  # 8 bars = 2 hours
    
    # What to include
    include_ohlcv: bool = True
    include_indicators: bool = True
    include_levels: bool = True
    include_time: bool = True
    
    # Normalization
    price_norm: str = "zscore"  # 'zscore', 'minmax', 'none'
    
    def to_dict(self) -> dict:
        return {
            'lookback_1m': self.lookback_1m,
            'lookback_5m': self.lookback_5m,
            'lookback_15m': self.lookback_15m,
            'include_ohlcv': self.include_ohlcv,
            'include_indicators': self.include_indicators,
            'include_levels': self.include_levels,
            'include_time': self.include_time,
            'price_norm': self.price_norm,
        }


@dataclass
class FeatureBundle:
    """
    Complete feature bundle for a single decision point.
    
    Separates price windows (for CNN) from context vector (for MLP).
    """
    # Price windows - (lookback, channels) for CNN
    x_price_1m: np.ndarray      # (120, 5) default
    x_price_5m: np.ndarray      # (24, 5) default  
    x_price_15m: np.ndarray     # (8, 5) default
    
    # Context vector - (dim,) for MLP
    x_context: np.ndarray       # (20,) default
    
    # Raw components (for debugging/logging)
    market_state: Optional[MarketState] = None
    indicators: Optional[IndicatorValues] = None
    levels: Optional[LevelValues] = None
    time_features: Optional[TimeFeatures] = None
    context_features: Optional[ContextFeatures] = None
    
    # Metadata
    bar_idx: int = 0
    timestamp: Optional[pd.Timestamp] = None
    current_price: float = 0.0
    atr: float = 0.0


def compute_features(
    stepper: MarketStepper,
    config: FeatureConfig,
    df_5m: pd.DataFrame = None,
    df_15m: pd.DataFrame = None,
    df_1h: pd.DataFrame = None,
    df_4h: pd.DataFrame = None,
    precomputed_indicators: Dict[int, IndicatorValues] = None,
    precomputed_levels: Dict[int, LevelValues] = None,
) -> FeatureBundle:
    """
    Compute all features at current stepper position.
    
    All data access is CAUSAL (via stepper.get_history only).
    
    Args:
        stepper: MarketStepper at current position
        config: Feature configuration
        df_5m, df_15m, df_1h, df_4h: Pre-resampled higher timeframe data
        precomputed_indicators: Optional pre-computed indicator values by bar_idx
        precomputed_levels: Optional pre-computed level values by bar_idx
        
    Returns:
        FeatureBundle with all features
    """
    # Get market state
    state = get_market_state(
        stepper,
        df_5m=df_5m,
        df_15m=df_15m,
        lookback_1m=config.lookback_1m,
        lookback_5m=config.lookback_5m,
        lookback_15m=config.lookback_15m,
    )
    
    # Normalize price windows
    x_price_1m = normalize_ohlcv(state.ohlcv_1m, config.price_norm)
    x_price_5m = normalize_ohlcv(state.ohlcv_5m, config.price_norm)
    x_price_15m = normalize_ohlcv(state.ohlcv_15m, config.price_norm)
    
    # Get indicator values (from precomputed or compute)
    bar_idx = state.current_bar_idx
    if precomputed_indicators and bar_idx in precomputed_indicators:
        indicators = precomputed_indicators[bar_idx]
    else:
        indicators = IndicatorValues()  # Empty if not provided
    
    # Get level values
    if precomputed_levels and bar_idx in precomputed_levels:
        levels = precomputed_levels[bar_idx]
    else:
        levels = LevelValues()
    
    # Time features
    time_feats = TimeFeatures()
    if state.current_time:
        time_feats = compute_time_features(state.current_time)
    
    # Context features (combines everything)
    atr = indicators.atr_5m_14 if indicators.atr_5m_14 > 0 else 1.0
    context_feats = compute_context_features(
        state.current_price,
        indicators,
        levels,
        time_feats,
        atr=atr
    )
    
    x_context = context_feats.to_array()
    
    return FeatureBundle(
        x_price_1m=x_price_1m,
        x_price_5m=x_price_5m,
        x_price_15m=x_price_15m,
        x_context=x_context,
        market_state=state,
        indicators=indicators,
        levels=levels,
        time_features=time_feats,
        context_features=context_feats,
        bar_idx=bar_idx,
        timestamp=state.current_time,
        current_price=state.current_price,
        atr=atr,
    )


def precompute_indicators(
