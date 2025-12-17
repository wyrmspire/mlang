"""
Multi-Timeframe Resampling
Explicit alignment rules for higher timeframes.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Optional
from dataclasses import dataclass


class ResampleRule(Enum):
    """Bar alignment rule."""
    RIGHT_CLOSED = "right"   # Bar timestamp = end of period (bar just closed)
    LEFT_CLOSED = "left"     # Bar timestamp = start of period


@dataclass
class MTFConfig:
    """Multi-timeframe configuration."""
    base_tf: str = "1min"
    higher_tfs: tuple = ("5min", "15min", "1h", "4h")
    resample_rule: ResampleRule = ResampleRule.RIGHT_CLOSED


# Map user-friendly names to pandas offset strings
TF_MAP = {
    '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
    '1h': '1h', '4h': '4h', '1d': '1D',
}


def resample_ohlcv(
    df: pd.DataFrame,
    target_tf: str,
    rule: ResampleRule = ResampleRule.RIGHT_CLOSED,
    time_col: str = 'time'
) -> pd.DataFrame:
    """
    Resample OHLCV data to higher timeframe.
    
    Args:
        df: DataFrame with time, open, high, low, close, volume columns
        target_tf: Target timeframe ('5m', '15m', '1h', '4h', etc.)
        rule: Alignment rule
        time_col: Name of time column
        
    Returns:
        Resampled DataFrame with same column structure.
        
    Note:
        RIGHT_CLOSED means at time T, the bar returned is the one
        that JUST CLOSED (contains data from T-tf to T).
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found")
    
    # Set time as index for resampling
    df_indexed = df.set_index(time_col)
    
    # Resample
    if rule == ResampleRule.RIGHT_CLOSED:
        # label='right' means bar timestamp = end of period
        resampled = df_indexed.resample(target_tf, label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    else:
        # label='left' means bar timestamp = start of period
        resampled = df_indexed.resample(target_tf, label='left', closed='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    
    # Drop NaN rows (incomplete bars at ends)
    resampled = resampled.dropna()
    
    # Reset index to have time as column
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={'index': time_col})
    
    return resampled


def get_current_htf_bar(
    df_htf: pd.DataFrame,
    current_time: pd.Timestamp,
    time_col: str = 'time'
) -> Optional[pd.Series]:
    """
    Get the most recent COMPLETED bar for a higher timeframe.
    
    At time T, returns the HTF bar that has closed at or before T.
    This is CAUSAL - no future leaking.
    
    Args:
        df_htf: Higher timeframe DataFrame (already resampled)
        current_time: Current simulation time
        time_col: Name of time column
        
    Returns:
        Most recent completed bar, or None if no bars before current_time.
    """
    mask = df_htf[time_col] <= current_time
    if not mask.any():
        return None
    
    idx = df_htf.loc[mask, time_col].idxmax()
    return df_htf.loc[idx]


def get_htf_window(
    df_htf: pd.DataFrame,
    current_time: pd.Timestamp,
    lookback: int,
    time_col: str = 'time'
) -> pd.DataFrame:
    """
    Get last N completed bars for a higher timeframe.
    
    CAUSAL - only returns bars that closed at or before current_time.
    """
    mask = df_htf[time_col] <= current_time
    available = df_htf.loc[mask]
    
    if len(available) < lookback:
        return available
    
    return available.tail(lookback)


def resample_all_timeframes(
    df_1m: pd.DataFrame,
    config: MTFConfig = MTFConfig()
) -> dict:
    """
    Resample 1m data to all configured higher timeframes.
    
    Returns:
        Dict mapping timeframe string to DataFrame.
        e.g., {'1m': df_1m, '5m': df_5m, '15m': df_15m, ...}
    """
    result = {'1m': df_1m}
    
    for tf in config.higher_tfs:
        # Convert to pandas-compatible format if needed
        pandas_tf = TF_MAP.get(tf, tf)
        result[tf.replace('min', 'm')] = resample_ohlcv(df_1m, pandas_tf, config.resample_rule)
    
    return result
