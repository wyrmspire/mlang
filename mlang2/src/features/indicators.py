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
