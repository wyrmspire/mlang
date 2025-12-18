"""
Opening Range Scanner
Identifies 15m Opening Range and triggers on retest.
"""

import pandas as pd
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from src.policy.scanners import Scanner, ScannerResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.config import NY_TZ


@dataclass
class OpeningRangeState:
    """Tracks the opening range for the current day."""
    date: Optional[pd.Timestamp] = None
    or_high: float = 0.0
    or_low: float = 0.0
    or_established: bool = False
    last_trigger_bar: int = -1000  # Cooldown tracking


class OpeningRangeScanner(Scanner):
    """
    Scanner that triggers on retest of 15m Opening Range.
    
    Logic:
    1. At 9:45 NY (after first 15m bar closes), establish OR high/low.
    2. Trigger when price retests OR high (for LONG) or OR low (for SHORT).
    3. Only trigger once per level per day (with cooldown).
    
    Config:
        or_timeframe_minutes: How many minutes to establish OR (default 15).
        retest_threshold_atr: How close price must be to OR level (default 0.25 ATR).
        cooldown_bars: Minimum bars between triggers (default 30).
    """
    
    def __init__(
        self,
        or_timeframe_minutes: int = 15,
        retest_threshold_atr: float = 0.25,
        cooldown_bars: int = 30,
        direction: str = "BOTH",  # 'LONG', 'SHORT', or 'BOTH'
    ):
        self.or_timeframe_minutes = or_timeframe_minutes
        self.retest_threshold_atr = retest_threshold_atr
        self.cooldown_bars = cooldown_bars
        self.direction = direction
        
        # State tracking (per day)
        self._state = OpeningRangeState()
        self._or_bars: list = []  # Collect bars during OR period
    
    @property
    def scanner_id(self) -> str:
        return f"opening_range_{self.or_timeframe_minutes}m"
    
    def _is_or_period(self, t: pd.Timestamp) -> bool:
        """Check if timestamp is within the OR establishment period (9:30-9:45 NY)."""
        ny_time = t.astimezone(NY_TZ)
        hour = ny_time.hour
        minute = ny_time.minute
        
        # 9:30 to 9:30 + or_timeframe_minutes
        if hour == 9 and 30 <= minute < (30 + self.or_timeframe_minutes):
            return True
        return False
    
    def _is_after_or(self, t: pd.Timestamp) -> bool:
        """Check if we're past the OR period (9:45+ for 15m OR)."""
        ny_time = t.astimezone(NY_TZ)
        hour = ny_time.hour
        minute = ny_time.minute
        
        or_end_minute = 30 + self.or_timeframe_minutes
        if hour == 9 and minute >= or_end_minute:
            return True
        if hour > 9:
            return True
        return False
    
    def _is_new_day(self, t: pd.Timestamp) -> bool:
        """Check if this is a new trading day."""
        if self._state.date is None:
            return True
        ny_time = t.astimezone(NY_TZ)
        return ny_time.date() != self._state.date.date()
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        t = features.timestamp
        if t is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Reset on new day
        if self._is_new_day(t):
            self._state = OpeningRangeState(date=t.astimezone(NY_TZ))
            self._or_bars = []
        
        # Collect bars during OR period
        if self._is_or_period(t) and not self._state.or_established:
            bar_data = {
                'high': state.current_high if hasattr(state, 'current_high') else features.current_price + 1,
                'low': state.current_low if hasattr(state, 'current_low') else features.current_price - 1,
            }
            self._or_bars.append(bar_data)
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Establish OR after period ends
        if self._is_after_or(t) and not self._state.or_established and len(self._or_bars) > 0:
            self._state.or_high = max(b['high'] for b in self._or_bars)
            self._state.or_low = min(b['low'] for b in self._or_bars)
            self._state.or_established = True
        
        # Can't trigger if OR not established
        if not self._state.or_established:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Cooldown check
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # Check for retest
        price = features.current_price
        atr = features.atr if features.atr > 0 else 1.0
        threshold = self.retest_threshold_atr * atr
        
        or_high = self._state.or_high
        or_low = self._state.or_low
        
        # Check LONG retest (price near OR low)
        long_triggered = (
            self.direction in ("LONG", "BOTH") and
            abs(price - or_low) <= threshold
        )
        
        # Check SHORT retest (price near OR high)
        short_triggered = (
            self.direction in ("SHORT", "BOTH") and
            abs(price - or_high) <= threshold
        )
        
        if long_triggered or short_triggered:
            self._state.last_trigger_bar = features.bar_idx
            return ScannerResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    'or_high': or_high,
                    'or_low': or_low,
                    'direction': 'LONG' if long_triggered else 'SHORT',
                    'retest_level': or_low if long_triggered else or_high,
                    'distance_atr': abs(price - (or_low if long_triggered else or_high)) / atr,
                },
                score=1.0
            )
        
        return ScannerResult(scanner_id=self.scanner_id, triggered=False)
