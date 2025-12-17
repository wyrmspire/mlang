"""
Filters
Pre-trade filters that block decisions before reaching policy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from src.features.pipeline import FeatureBundle
from src.features.time_features import Session


@dataclass
class FilterResult:
    """Result from a filter check."""
    passed: bool
    filter_id: str
    reason: str = ""


class Filter(ABC):
    """Base class for pre-trade filters."""
    
    @property
    @abstractmethod
    def filter_id(self) -> str:
        pass
    
    @abstractmethod
    def check(self, features: FeatureBundle) -> FilterResult:
        """Check if filter passes."""
        pass


class SessionFilter(Filter):
    """Only trade during specific sessions."""
    
    def __init__(self, allowed_sessions: List[str] = None):
        self.allowed_sessions = allowed_sessions or ['RTH']
    
    @property
    def filter_id(self) -> str:
        return f"session_{'_'.join(self.allowed_sessions)}"
    
    def check(self, features: FeatureBundle) -> FilterResult:
        if features.time_features is None:
            return FilterResult(passed=False, filter_id=self.filter_id, reason="No time features")
        
        session = features.time_features.session
        passed = session in self.allowed_sessions
        
        return FilterResult(
            passed=passed,
            filter_id=self.filter_id,
            reason="" if passed else f"Session {session} not in {self.allowed_sessions}"
        )


class TimeFilter(Filter):
    """Only trade during specific hours."""
    
    def __init__(
        self,
        allowed_hours: List[int] = None,
        excluded_hours: List[int] = None
    ):
        self.allowed_hours = allowed_hours  # If set, only these hours
        self.excluded_hours = excluded_hours or []  # Always exclude these
    
    @property
    def filter_id(self) -> str:
        return "time_filter"
    
    def check(self, features: FeatureBundle) -> FilterResult:
        if features.time_features is None:
            return FilterResult(passed=False, filter_id=self.filter_id, reason="No time features")
        
        hour = features.time_features.hour_ny
        
        if hour in self.excluded_hours:
            return FilterResult(
                passed=False,
                filter_id=self.filter_id,
                reason=f"Hour {hour} is excluded"
            )
        
        if self.allowed_hours and hour not in self.allowed_hours:
            return FilterResult(
                passed=False,
                filter_id=self.filter_id,
                reason=f"Hour {hour} not in allowed hours"
            )
        
        return FilterResult(passed=True, filter_id=self.filter_id)


class VolatilityFilter(Filter):
    """Filter based on ATR or volatility conditions."""
    
    def __init__(
        self,
        min_atr: float = 0.0,
        max_adr_pct: float = 1.5
    ):
        self.min_atr = min_atr
        self.max_adr_pct = max_adr_pct
    
    @property
    def filter_id(self) -> str:
        return f"volatility_{self.min_atr}_{self.max_adr_pct}"
    
    def check(self, features: FeatureBundle) -> FilterResult:
        # Check minimum ATR
        if self.min_atr > 0 and features.atr < self.min_atr:
            return FilterResult(
                passed=False,
                filter_id=self.filter_id,
                reason=f"ATR {features.atr:.2f} below minimum {self.min_atr}"
            )
        
        # Check ADR consumption
        if features.indicators and features.indicators.adr_pct_used > self.max_adr_pct:
            return FilterResult(
                passed=False,
                filter_id=self.filter_id,
                reason=f"ADR {features.indicators.adr_pct_used:.1%} exceeds max {self.max_adr_pct:.1%}"
            )
        
        return FilterResult(passed=True, filter_id=self.filter_id)


class FilterChain:
    """Run multiple filters in sequence."""
    
    def __init__(self, filters: List[Filter] = None):
        self.filters = filters or []
    
    def add(self, f: Filter) -> 'FilterChain':
        self.filters.append(f)
        return self
    
    def check(self, features: FeatureBundle) -> FilterResult:
        """
        Run all filters. Returns first failure or final pass.
        """
        for f in self.filters:
            result = f.check(features)
            if not result.passed:
                return result
        
        return FilterResult(passed=True, filter_id="all", reason="All filters passed")


# Default filter chain for RTH trading
DEFAULT_FILTERS = FilterChain([
    SessionFilter(['RTH']),
    TimeFilter(excluded_hours=[12]),  # Exclude lunch
])
