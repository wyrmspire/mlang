"""
Bar Fill Model
Explicit rules for same-bar fill behavior.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd

from src.sim.costs import CostModel, DEFAULT_COSTS


class EntryModel(Enum):
    """How entries are filled."""
    NEXT_BAR_OPEN = "next_open"     # Market fills at next bar open
    THIS_BAR_CLOSE = "this_close"   # Can fill at current bar close
    LIMIT_INTRABAR = "limit_intra"  # Limit can fill intrabar if touched


class SLTPTieBreak(Enum):
    """How to handle SL and TP both touched in same bar."""
    CONSERVATIVE = "conservative"    # Assume SL hit first (worst case)
    OPTIMISTIC = "optimistic"        # Assume TP hit first (best case)
    OPEN_PROXIMITY = "open_prox"     # Whichever is closer to open


class SameBarExit(Enum):
    """Can entry and exit happen same bar?"""
    ALLOWED = "allowed"     # Can exit same bar as entry
    BLOCKED = "blocked"     # Must wait at least 1 bar


@dataclass
class BarFillConfig:
    """
    Complete bar fill model configuration.
    
    Since OHLC bars don't reveal price path, we must choose
    consistent conventions for all same-bar scenarios.
    """
    entry_model: EntryModel = EntryModel.NEXT_BAR_OPEN
    sl_tp_tiebreak: SLTPTieBreak = SLTPTieBreak.CONSERVATIVE
    same_bar_exit: SameBarExit = SameBarExit.BLOCKED
    
    def to_dict(self) -> dict:
        return {
            'entry_model': self.entry_model.value,
            'sl_tp_tiebreak': self.sl_tp_tiebreak.value,
            'same_bar_exit': self.same_bar_exit.value,
        }


class BarFillEngine:
    """
    Applies BarFillConfig rules consistently to all order types.
    """
    
    def __init__(
        self,
        config: BarFillConfig = None,
        costs: CostModel = None
    ):
        self.config = config or BarFillConfig()
        self.costs = costs or DEFAULT_COSTS
    
    def can_fill_limit_entry(
        self,
        limit_price: float,
        direction: str,
        bar: pd.Series
    ) -> bool:
        """
        Check if limit entry would fill on this bar.
        
        LONG limit fills if low <= limit_price
        SHORT limit fills if high >= limit_price
        """
        if direction == 'LONG':
            return bar['low'] <= limit_price
        else:
            return bar['high'] >= limit_price
    
    def get_limit_entry_fill_price(
        self,
        limit_price: float,
        direction: str,
        bar: pd.Series
    ) -> Optional[float]:
        """
        Get fill price for limit entry.
        
        Returns limit price if filled (or better if gap).
        """
        if not self.can_fill_limit_entry(limit_price, direction, bar):
            return None
        
        if direction == 'LONG':
            # Could fill at limit or better (lower)
            # If bar opens below limit, fill at open
            if bar['open'] <= limit_price:
                return bar['open']
            return limit_price
        else:
            # SHORT - fill at limit or better (higher)
            if bar['open'] >= limit_price:
                return bar['open']
            return limit_price
    
    def check_exit(
        self,
        position_direction: str,
        stop_price: float,
        tp_price: float,
        bar: pd.Series,
        entry_bar_idx: int,
        current_bar_idx: int
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Check if SL or TP is hit on this bar.
        
        Returns:
            (outcome, fill_price) where outcome is 'SL', 'TP', or None
        """
        # Check same-bar exit rule
        if self.config.same_bar_exit == SameBarExit.BLOCKED:
            if current_bar_idx <= entry_bar_idx:
                return (None, None)
        
        # Check if exits are touched
        if position_direction == 'LONG':
            # LONG: SL hit if low <= stop, TP hit if high >= tp
            sl_touched = bar['low'] <= stop_price
            tp_touched = bar['high'] >= tp_price
        else:
            # SHORT: SL hit if high >= stop, TP hit if low <= tp
            sl_touched = bar['high'] >= stop_price
            tp_touched = bar['low'] <= tp_price
        
        if sl_touched and tp_touched:
            # Both touched - apply tie-break
            return self._resolve_tie(
                position_direction, stop_price, tp_price, bar
            )
        elif sl_touched:
            return ('SL', stop_price)
        elif tp_touched:
            return ('TP', tp_price)
        else:
            return (None, None)
    
    def _resolve_tie(
        self,
        direction: str,
        stop_price: float,
        tp_price: float,
        bar: pd.Series
    ) -> Tuple[str, float]:
        """Resolve SL/TP same-bar tie."""
        
        if self.config.sl_tp_tiebreak == SLTPTieBreak.CONSERVATIVE:
            # Assume worst case - SL first
            return ('SL', stop_price)
        
        elif self.config.sl_tp_tiebreak == SLTPTieBreak.OPTIMISTIC:
            # Assume best case - TP first
            return ('TP', tp_price)
        
        else:  # OPEN_PROXIMITY
            # Whichever is closer to open price wins
            sl_dist = abs(bar['open'] - stop_price)
            tp_dist = abs(bar['open'] - tp_price)
            
            if sl_dist <= tp_dist:
                return ('SL', stop_price)
            else:
                return ('TP', tp_price)


# Default fill engine
DEFAULT_FILL_ENGINE = BarFillEngine()
