"""
Stop Calculator
Flexible stop loss calculation based on different reference points.

Stop types:
- CANDLE_OPEN: Previous candle open (5m, 15m, etc.)
- CANDLE_LOW/HIGH: Previous candle low/high
- RANGE_LOW/HIGH: Low/high of a time range (e.g., OR)
- SWING_LOW/HIGH: Previous swing point on higher TF
- ATR_OFFSET: Simple ATR offset from entry (legacy)

ATR is used as PADDING, not as the stop level itself.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np


class StopType(Enum):
    """Type of stop level calculation."""
    ATR_OFFSET = "atr_offset"         # Legacy: stop = entry +/- atr * mult
    CANDLE_OPEN = "candle_open"       # Stop at previous candle open
    CANDLE_LOW = "candle_low"         # Stop at previous candle low (LONG)
    CANDLE_HIGH = "candle_high"       # Stop at previous candle high (SHORT)
    RANGE_LOW = "range_low"           # Stop at range low (e.g., OR low for LONG)
    RANGE_HIGH = "range_high"         # Stop at range high (e.g., OR high for SHORT)
    SWING_LOW = "swing_low"           # Previous swing low on HTF (LONG)
    SWING_HIGH = "swing_high"         # Previous swing high on HTF (SHORT)


@dataclass
class StopConfig:
    """Configuration for stop calculation."""
    stop_type: StopType = StopType.CANDLE_LOW
    timeframe: str = "5m"             # Timeframe for candle-based stops
    lookback: int = 1                 # How many candles back to look
    atr_padding: float = 0.25         # ATR padding beyond the stop level
    range_start_time: str = "09:30"   # For range-based stops
    range_end_time: str = "09:45"     # For range-based stops
    swing_lookback: int = 20          # Bars to look for swing points
    
    def to_dict(self) -> dict:
        return {
            'stop_type': self.stop_type.value,
            'timeframe': self.timeframe,
            'lookback': self.lookback,
            'atr_padding': self.atr_padding,
        }


def calculate_stop(
    direction: str,
    entry_price: float,
    atr: float,
    config: StopConfig,
    df_1m: Optional[pd.DataFrame] = None,
    df_htf: Optional[pd.DataFrame] = None,
    current_idx: int = 0,
    range_high: float = 0.0,
    range_low: float = 0.0,
) -> Tuple[float, str]:
    """
    Calculate stop price based on configuration.
    
    Args:
        direction: 'LONG' or 'SHORT'
        entry_price: Entry price for reference
        atr: Current ATR for padding
        config: Stop configuration
        df_1m: 1-minute data (for finding HTF bars)
        df_htf: Higher timeframe data (5m, 15m, etc.)
        current_idx: Current bar index in df_1m
        range_high: Pre-calculated range high (for RANGE_* stops)
        range_low: Pre-calculated range low (for RANGE_* stops)
        
    Returns:
        (stop_price, reason_string)
    """
    padding = config.atr_padding * atr
    
    if config.stop_type == StopType.ATR_OFFSET:
        # Legacy: simple ATR offset from entry
        if direction == "LONG":
            stop = entry_price - atr
        else:
            stop = entry_price + atr
        return (stop, f"ATR offset from entry")
    
    elif config.stop_type == StopType.CANDLE_OPEN:
        # Stop at previous candle open
        if df_htf is not None and len(df_htf) > config.lookback:
            candle = df_htf.iloc[-(config.lookback + 1)]
            base_stop = candle['open']
            if direction == "LONG":
                stop = base_stop - padding
            else:
                stop = base_stop + padding
            return (stop, f"{config.timeframe} candle open - padding")
    
    elif config.stop_type == StopType.CANDLE_LOW:
        # Stop at previous candle low (for LONG)
        if df_htf is not None and len(df_htf) > config.lookback:
            candle = df_htf.iloc[-(config.lookback + 1)]
            base_stop = candle['low']
            stop = base_stop - padding
            return (stop, f"{config.timeframe} candle low - padding")
    
    elif config.stop_type == StopType.CANDLE_HIGH:
        # Stop at previous candle high (for SHORT)
        if df_htf is not None and len(df_htf) > config.lookback:
            candle = df_htf.iloc[-(config.lookback + 1)]
            base_stop = candle['high']
            stop = base_stop + padding
            return (stop, f"{config.timeframe} candle high + padding")
    
    elif config.stop_type == StopType.RANGE_LOW:
        # Stop at range low (for LONG trades)
        if range_low > 0:
            stop = range_low - padding
            return (stop, f"Range low - padding")
    
    elif config.stop_type == StopType.RANGE_HIGH:
        # Stop at range high (for SHORT trades)
        if range_high > 0:
            stop = range_high + padding
            return (stop, f"Range high + padding")
    
    elif config.stop_type == StopType.SWING_LOW:
        # Previous swing low on HTF
        if df_htf is not None and len(df_htf) > config.swing_lookback:
            lookback_df = df_htf.iloc[-config.swing_lookback:]
            swing_low = lookback_df['low'].min()
            stop = swing_low - padding
            return (stop, f"{config.timeframe} swing low - padding")
    
    elif config.stop_type == StopType.SWING_HIGH:
        # Previous swing high on HTF
        if df_htf is not None and len(df_htf) > config.swing_lookback:
            lookback_df = df_htf.iloc[-config.swing_lookback:]
            swing_high = lookback_df['high'].max()
            stop = swing_high + padding
            return (stop, f"{config.timeframe} swing high + padding")
    
    # Fallback: ATR offset
    if direction == "LONG":
        stop = entry_price - atr
    else:
        stop = entry_price + atr
    return (stop, "Fallback ATR offset")


def get_stop_for_or_retest(
    direction: str,
    entry_price: float,
    atr: float,
    or_high: float,
    or_low: float,
    padding_atr: float = 0.25
) -> Tuple[float, str]:
    """
    Convenience function for Opening Range retest strategy.
    
    For LONG (retest of OR low): stop is OR low - padding
    For SHORT (retest of OR high): stop is OR high + padding
    """
    padding = padding_atr * atr
    
    if direction == "LONG":
        stop = or_low - padding
        return (stop, f"OR low ({or_low:.2f}) - {padding:.2f} padding")
    else:
        stop = or_high + padding
        return (stop, f"OR high ({or_high:.2f}) + {padding:.2f} padding")


def calculate_risk(entry_price: float, stop_price: float, direction: str) -> float:
    """Calculate risk in points (always positive)."""
    if direction == "LONG":
        return abs(entry_price - stop_price)
    else:
        return abs(stop_price - entry_price)


def calculate_tp_from_risk(
    entry_price: float,
    stop_price: float,
    direction: str,
    r_multiple: float
) -> float:
    """Calculate take profit price from risk and R multiple."""
    risk = calculate_risk(entry_price, stop_price, direction)
    
    if direction == "LONG":
        return entry_price + (risk * r_multiple)
    else:
        return entry_price - (risk * r_multiple)
