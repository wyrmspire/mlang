"""
Market State
Raw OHLCV windows from the stepper.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict

from src.sim.stepper import MarketStepper
from src.config import DEFAULT_LOOKBACK_1M, DEFAULT_LOOKBACK_5M, DEFAULT_LOOKBACK_15M


@dataclass
class MarketState:
    """
    Raw market state at a point in time.
    All data is CAUSAL (from past only).
    """
    # Price windows (lookback, 5) for OHLCV
    ohlcv_1m: np.ndarray     # (120, 5) default - 2 hours
    ohlcv_5m: np.ndarray     # (24, 5) default - 2 hours
    ohlcv_15m: np.ndarray    # (8, 5) default - 2 hours
    
    # Current bar info
    current_price: float
    current_time: pd.Timestamp
    current_bar_idx: int
    
    # Additional context
    current_bar: Optional[pd.Series] = None


def get_market_state(
    stepper: MarketStepper,
    df_5m: pd.DataFrame = None,
    df_15m: pd.DataFrame = None,
    lookback_1m: int = DEFAULT_LOOKBACK_1M,
    lookback_5m: int = DEFAULT_LOOKBACK_5M,
    lookback_15m: int = DEFAULT_LOOKBACK_15M,
) -> MarketState:
    """
    Extract market state from stepper.
    
    CAUSAL ONLY - uses stepper.get_history().
    
    Args:
        stepper: MarketStepper positioned at current bar
        df_5m: Pre-resampled 5m data (optional, for efficiency)
        df_15m: Pre-resampled 15m data (optional)
        lookback_*: Number of bars for each timeframe
        
    Returns:
        MarketState with all price windows
    """
    # Get 1m history
    ohlcv_1m = stepper.get_history_array(
        lookback_1m, 
        columns=['open', 'high', 'low', 'close', 'volume']
    )
    
    # Get current bar info
    current_bar = stepper.get_current_bar()
    current_time = stepper.get_current_time()
    current_bar_idx = stepper.get_current_idx()
    
    if current_bar is not None:
        current_price = current_bar['close']
    else:
        current_price = 0.0
    
    # Get higher timeframe windows if data provided
    if df_5m is not None and current_time is not None:
        ohlcv_5m = _get_htf_window(df_5m, current_time, lookback_5m)
    else:
        ohlcv_5m = np.zeros((lookback_5m, 5), dtype=np.float32)
    
    if df_15m is not None and current_time is not None:
        ohlcv_15m = _get_htf_window(df_15m, current_time, lookback_15m)
    else:
        ohlcv_15m = np.zeros((lookback_15m, 5), dtype=np.float32)
    
    return MarketState(
        ohlcv_1m=ohlcv_1m,
        ohlcv_5m=ohlcv_5m,
        ohlcv_15m=ohlcv_15m,
        current_price=current_price,
        current_time=current_time,
        current_bar_idx=current_bar_idx,
        current_bar=current_bar,
    )


def _get_htf_window(
    df_htf: pd.DataFrame,
    current_time: pd.Timestamp,
    lookback: int
) -> np.ndarray:
    """Get higher timeframe window as numpy array."""
    from src.data.resample import get_htf_window
    
    window = get_htf_window(df_htf, current_time, lookback)
    
    if len(window) == 0:
        return np.zeros((lookback, 5), dtype=np.float32)
    
    values = window[['open', 'high', 'low', 'close', 'volume']].values
    
    # Pad if insufficient
    if len(values) < lookback:
        padding = np.zeros((lookback - len(values), 5), dtype=np.float32)
        values = np.vstack([padding, values])
    
    return values.astype(np.float32)


def normalize_ohlcv(
    ohlcv: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize OHLCV window for model input.
    
    Args:
        ohlcv: (lookback, 5) array
        method: 'zscore', 'minmax', or 'none'
        
    Returns:
        Normalized array
    """
    if method == 'none':
        return ohlcv
    
    # Normalize price columns (0-3), leave volume separate
    prices = ohlcv[:, :4]
    volume = ohlcv[:, 4:5]
    
    if method == 'zscore':
        mean = np.mean(prices)
        std = np.std(prices)
        if std < 1e-8:
            std = 1.0
        prices_norm = (prices - mean) / std
        
        vol_mean = np.mean(volume)
        vol_std = np.std(volume)
        if vol_std < 1e-8:
            vol_std = 1.0
        vol_norm = (volume - vol_mean) / vol_std
        
    elif method == 'minmax':
        p_min, p_max = np.min(prices), np.max(prices)
        if p_max - p_min < 1e-8:
            prices_norm = np.zeros_like(prices)
        else:
            prices_norm = (prices - p_min) / (p_max - p_min)
        
        v_min, v_max = np.min(volume), np.max(volume)
        if v_max - v_min < 1e-8:
            vol_norm = np.zeros_like(volume)
        else:
            vol_norm = (volume - v_min) / (v_max - v_min)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return np.hstack([prices_norm, vol_norm]).astype(np.float32)
