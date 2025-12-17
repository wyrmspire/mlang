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
