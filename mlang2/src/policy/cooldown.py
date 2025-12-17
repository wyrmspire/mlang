"""
Cooldown
Manage trade cooldown periods to prevent overtrading.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class CooldownConfig:
    """Cooldown configuration."""
    min_bars_between_trades: int = 15  # Minimum bars between entries
    min_bars_after_loss: int = 30      # Extra cooldown after a loss
    max_trades_per_day: int = 10       # Maximum trades per trading day


class CooldownManager:
    """
    Manage trade cooldown state.
    """
    
    def __init__(self, config: CooldownConfig = None):
        self.config = config or CooldownConfig()
        self._last_trade_bar: int = -999
        self._last_outcome: str = ""
        self._trades_today: int = 0
        self._current_date: Optional[pd.Timestamp] = None
    
    def reset(self):
        """Reset cooldown state."""
        self._last_trade_bar = -999
        self._last_outcome = ""
        self._trades_today = 0
        self._current_date = None
    
    def record_trade(
        self,
        bar_idx: int,
        outcome: str = "",
        timestamp: pd.Timestamp = None
    ):
        """Record that a trade was taken."""
        self._last_trade_bar = bar_idx
        self._last_outcome = outcome
        
        # Track trades per day
        if timestamp:
            trade_date = timestamp.date()
            if self._current_date != trade_date:
                self._current_date = trade_date
                self._trades_today = 0
            self._trades_today += 1
    
    def is_on_cooldown(
        self,
        current_bar: int,
        timestamp: pd.Timestamp = None
    ) -> tuple:
        """
        Check if currently on cooldown.
        
        Returns:
            (on_cooldown: bool, reason: str)
        """
        # Check max trades per day
        if timestamp:
            current_date = timestamp.date()
            if current_date == self._current_date:
                if self._trades_today >= self.config.max_trades_per_day:
                    return (True, f"Max trades per day ({self.config.max_trades_per_day}) reached")
        
        # Check bars since last trade
        bars_since = current_bar - self._last_trade_bar
        
        # Extra cooldown after loss
        if self._last_outcome == 'LOSS':
            min_bars = self.config.min_bars_after_loss
        else:
            min_bars = self.config.min_bars_between_trades
        
        if bars_since < min_bars:
            return (True, f"Cooldown: {bars_since}/{min_bars} bars since last trade")
        
        return (False, "")
    
    def bars_remaining(self, current_bar: int) -> int:
        """Get bars remaining in cooldown."""
        bars_since = current_bar - self._last_trade_bar
        
        if self._last_outcome == 'LOSS':
            min_bars = self.config.min_bars_after_loss
        else:
            min_bars = self.config.min_bars_between_trades
        
        return max(0, min_bars - bars_since)
