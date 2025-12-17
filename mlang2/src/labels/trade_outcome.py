"""
Trade Outcome
Compute trade outcomes from future data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from src.labels.future_window import FutureWindowProvider
from src.sim.bar_fill_model import BarFillEngine, BarFillConfig


@dataclass
class TradeOutcome:
    """Outcome of a simulated trade."""
    outcome: str          # 'WIN', 'LOSS', 'TIMEOUT'
    pnl: float           # Points of profit/loss
    exit_bar_offset: int  # Bars from entry to exit
    exit_price: float
    mae: float           # Max Adverse Excursion (points)
    mfe: float           # Max Favorable Excursion (points)
    bars_held: int


def compute_trade_outcome(
    future_provider: FutureWindowProvider,
    entry_price: float,
    direction: str,
    stop_loss: float,
    take_profit: float,
    max_bars: int = 200,
    fill_config: BarFillConfig = None
) -> TradeOutcome:
    """
    Simulate a trade to completion using future data.
    
    This is the core labeling function - uses future data
    to determine what WOULD have happened.
    
    Args:
        future_provider: Provider for future bars
        entry_price: Entry price
        direction: 'LONG' or 'SHORT'
        stop_loss: Stop loss price
        take_profit: Take profit price
        max_bars: Maximum bars before timeout
        fill_config: Bar fill configuration
        
    Returns:
        TradeOutcome with all metrics
    """
    fill_config = fill_config or BarFillConfig()
    future = future_provider.get_future(max_bars)
    
    if len(future) == 0:
        return TradeOutcome(
            outcome='TIMEOUT',
            pnl=0.0,
            exit_bar_offset=0,
            exit_price=entry_price,
            mae=0.0,
            mfe=0.0,
            bars_held=0
        )
    
    # Track excursions
    highs = future['high'].values
    lows = future['low'].values
    
    if direction == 'LONG':
        # LONG: adverse = low below entry, favorable = high above entry
        mae = max(0, entry_price - lows.min())
        mfe = max(0, highs.max() - entry_price)
        
        # Find first SL or TP hit
        sl_hits = np.where(lows <= stop_loss)[0]
        tp_hits = np.where(highs >= take_profit)[0]
        
    else:  # SHORT
        # SHORT: adverse = high above entry, favorable = low below entry
        mae = max(0, highs.max() - entry_price)
        mfe = max(0, entry_price - lows.min())
        
        sl_hits = np.where(highs >= stop_loss)[0]
        tp_hits = np.where(lows <= take_profit)[0]
    
    sl_bar = sl_hits[0] if len(sl_hits) > 0 else max_bars + 1
    tp_bar = tp_hits[0] if len(tp_hits) > 0 else max_bars + 1
    
    # Determine outcome
    if tp_bar < sl_bar:
        outcome = 'WIN'
        exit_bar = tp_bar
        exit_price = take_profit
    elif sl_bar < max_bars:
        outcome = 'LOSS'
        exit_bar = sl_bar
        exit_price = stop_loss
    else:
        outcome = 'TIMEOUT'
        exit_bar = min(len(future) - 1, max_bars - 1)
        exit_price = future.iloc[exit_bar]['close']
    
    # Handle same-bar hits (both SL and TP)
    if sl_bar == tp_bar and sl_bar < max_bars:
        # Apply tie-break rule from fill config
        from src.sim.bar_fill_model import SLTPTieBreak
        
        if fill_config.sl_tp_tiebreak == SLTPTieBreak.CONSERVATIVE:
            outcome = 'LOSS'
            exit_price = stop_loss
        elif fill_config.sl_tp_tiebreak == SLTPTieBreak.OPTIMISTIC:
            outcome = 'WIN'
            exit_price = take_profit
        else:
            # Open proximity
            bar = future.iloc[sl_bar]
            sl_dist = abs(bar['open'] - stop_loss)
            tp_dist = abs(bar['open'] - take_profit)
            if sl_dist <= tp_dist:
                outcome = 'LOSS'
                exit_price = stop_loss
            else:
                outcome = 'WIN'
                exit_price = take_profit
    
    # Calculate PnL
    if direction == 'LONG':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    
    return TradeOutcome(
        outcome=outcome,
        pnl=pnl,
        exit_bar_offset=exit_bar + 1,  # +1 because futures start at entry+1
        exit_price=exit_price,
        mae=mae,
        mfe=mfe,
        bars_held=exit_bar + 1
    )


def compute_price_target_reached(
    future_provider: FutureWindowProvider,
    target_price: float,
    direction: str,  # 'UP' or 'DOWN'
    within_bars: int
) -> Tuple[bool, int]:
    """
    Check if price reaches target within N bars.
    
    Returns:
        (reached: bool, bars_to_reach: int or -1 if not reached)
    """
    future = future_provider.get_future(within_bars)
    
    if len(future) == 0:
        return (False, -1)
    
    if direction == 'UP':
        hits = np.where(future['high'].values >= target_price)[0]
    else:
        hits = np.where(future['low'].values <= target_price)[0]
    
    if len(hits) > 0:
        return (True, int(hits[0]) + 1)
    
    return (False, -1)
