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

```

### src/policy/library/__init__.py

```python
"""
Strategy Library
Modular implementations of setup scanners.
"""

```

### src/policy/library/mid_day_reversal.py

```python
"""
Mid-Day Reversal Strategy
Modular scanner that looks for reversals during lunch/mid-day.
"""

from src.policy.scanners import Scanner, ScannerResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle

class MidDayReversalScanner(Scanner):
    """
    Scanner that triggers for mid-day reversal setups.
    
    Logic:
    1. Must be in RTH (Regular Trading Hours).
    2. Must be Mid-day (11:00 AM - 1:30 PM NY).
    3. Price must show an extreme or RSI must be at an extreme.
    """
    
    def __init__(
        self, 
        start_hour: int = 11, 
        end_hour: int = 13, 
        rsi_extreme: float = 30.0
    ):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.rsi_extreme = rsi_extreme

    @property
    def scanner_id(self) -> str:
        return f"midday_reversal_{self.start_hour}_{self.end_hour}"

    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        t = features.time_features
        if not t or not t.is_rth:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # 1. Check time window
        is_midday = self.start_hour <= t.hour_ny <= self.end_hour
        if not is_midday:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # 2. Check for reversal signal (Simple RSI extreme for now)
        if features.indicators is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        rsi = features.indicators.rsi_5m_14
        oversold = rsi <= self.rsi_extreme
        overbought = rsi >= (100 - self.rsi_extreme)
        
        triggered = oversold or overbought
        
        return ScannerResult(
            scanner_id=self.scanner_id,
            triggered=triggered,
            context={
                'hour': t.hour_ny,
                'rsi': rsi,
                'condition': 'oversold' if oversold else 'overbought' if overbought else 'neutral'
            },
            score=1.0 if triggered else 0.0
        )

```

### src/policy/scanners.py

```python
"""
Scanners
Setup detection - determines when a decision point occurs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from src.features.state import MarketState
from src.features.pipeline import FeatureBundle


@dataclass
class ScannerResult:
    """Result from a scanner check."""
    scanner_id: str
    triggered: bool
    context: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0   # Confidence/strength of signal


class Scanner(ABC):
    """
    Base class for setup scanners.
    
    Scanners define WHEN a decision point occurs.
    They don't decide the action - just whether to evaluate.
    """
    
    @property
    @abstractmethod
    def scanner_id(self) -> str:
        """Unique identifier for this scanner."""
        pass
    
    @abstractmethod
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        """
        Check if current state triggers this scanner.
        
        Args:
            state: Current market state
            features: Computed features
            
        Returns:
            ScannerResult with triggered flag and context
        """
        pass


class AlwaysScanner(Scanner):
    """
    Scanner that always triggers.
    Useful for testing or fixed-interval strategies.
    """
    
    @property
    def scanner_id(self) -> str:
        return "always"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        return ScannerResult(
            scanner_id=self.scanner_id,
            triggered=True,
            score=1.0
        )


class IntervalScanner(Scanner):
    """
    Scanner that triggers every N bars.
    """
    
    def __init__(self, interval: int = 5):
        self.interval = interval
        self._last_triggered = -interval
    
    @property
    def scanner_id(self) -> str:
        return f"interval_{self.interval}"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        bar_idx = features.bar_idx
        
        if bar_idx - self._last_triggered >= self.interval:
            self._last_triggered = bar_idx
            return ScannerResult(
                scanner_id=self.scanner_id,
                triggered=True,
                score=1.0
            )
        
        return ScannerResult(
            scanner_id=self.scanner_id,
            triggered=False
        )


class LevelProximityScanner(Scanner):
    """
    Scanner that triggers when price is near key levels.
    """
    
    def __init__(
        self,
        atr_threshold: float = 0.5,
        level_types: List[str] = None
    ):
        self.atr_threshold = atr_threshold
        self.level_types = level_types or ['1h', '4h', 'pdh', 'pdl']
    
    @property
    def scanner_id(self) -> str:
        return f"level_proximity_{self.atr_threshold}"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        if features.levels is None or features.atr <= 0:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        levels = features.levels
        atr = features.atr
        price = features.current_price
        
        # Check distances to each level
        min_dist_atr = float('inf')
        nearest_level = None
        
        checks = [
            ('1h_high', levels.dist_1h_high),
            ('1h_low', levels.dist_1h_low),
            ('4h_high', levels.dist_4h_high),
            ('4h_low', levels.dist_4h_low),
            ('pdh', levels.dist_pdh),
            ('pdl', levels.dist_pdl),
        ]
        
        for name, dist in checks:
            dist_atr = abs(dist) / atr if atr > 0 else float('inf')
            if dist_atr < min_dist_atr:
                min_dist_atr = dist_atr
                nearest_level = name
        
        triggered = min_dist_atr <= self.atr_threshold
        
        return ScannerResult(
            scanner_id=self.scanner_id,
            triggered=triggered,
            context={
                'nearest_level': nearest_level,
                'distance_atr': min_dist_atr,
            },
            score=max(0, 1 - min_dist_atr / self.atr_threshold) if triggered else 0
        )


class RSIExtremeScanner(Scanner):
    """
    Scanner that triggers at RSI extremes.
    """
    
    def __init__(
        self,
        oversold: float = 30.0,
        overbought: float = 70.0
    ):
        self.oversold = oversold
        self.overbought = overbought
    
    @property
    def scanner_id(self) -> str:
        return f"rsi_extreme_{int(self.oversold)}_{int(self.overbought)}"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        if features.indicators is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        rsi = features.indicators.rsi_5m_14
        
        oversold = rsi <= self.oversold
        overbought = rsi >= self.overbought
        triggered = oversold or overbought
        
        return ScannerResult(
            scanner_id=self.scanner_id,
            triggered=triggered,
            context={
                'rsi': rsi,
                'condition': 'oversold' if oversold else 'overbought' if overbought else 'neutral',
            },
            score=1.0 if triggered else 0.0
        )


def _discover_library_scanners() -> Dict[str, type]:
    """Helper to find all Scanner classes in the library."""
    import importlib
    import pkgutil
    import inspect
    from src.policy import library
    
    found = {}
    
    # Iterate over modules in the library package
    for loader, name, is_pkg in pkgutil.iter_modules(library.__path__):
        full_name = f"src.policy.library.{name}"
        module = importlib.import_module(full_name)
        
        # Find all classes that inherit from Scanner
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Scanner) and cls is not Scanner:
                # Use a slugified name or something generic
                # For discovery, we'll try to use a 'base' name or the class name lowercased
                key = name.lower().replace('_', '')
                found[key] = cls
                
    return found


def get_scanner(scanner_id: str, **kwargs) -> Scanner:
    """Factory function to get scanner by ID."""
    scanners = {
        'always': AlwaysScanner,
        'interval': IntervalScanner,
        'level_proximity': LevelProximityScanner,
        'rsi_extreme': RSIExtremeScanner,
    }
    
    # Add discovered library scanners
    scanners.update(_discover_library_scanners())
    
    # Extract base name
    # We support both 'rsi_extreme' and just 'rsi' if we wanted
    base = scanner_id.split('_')[0].lower().replace('_', '')
    
    if base in scanners:
        return scanners[base](**kwargs)
    
    # Also check full name matches in case of library scanners
    # e.g. middayreversal
    clean_id = scanner_id.replace('_', '').lower()
    for name, cls in scanners.items():
        if clean_id.startswith(name):
            return cls(**kwargs)
    
    raise ValueError(f"Unknown scanner: {scanner_id}. Available: {list(scanners.keys())}")

```

### src/sim/__init__.py

```python
# Simulation module
"""Deterministic trade simulation engine."""

```

### src/sim/account.py

```python
"""
Account
Position and PnL tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd

from src.sim.execution import Fill
from src.sim.costs import CostModel, DEFAULT_COSTS


@dataclass
class Position:
    """Active position."""
    direction: str
    entry_price: float
    size: int
    entry_bar: int
    entry_time: Optional[pd.Timestamp] = None
    
    # Tracking
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0


@dataclass
class TradeRecord:
    """Completed trade record."""
    direction: str
    entry_price: float
    exit_price: float
    size: int
    entry_bar: int
    exit_bar: int
    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    
    # Outcome
    outcome: str = ""  # 'WIN', 'LOSS', 'TIMEOUT'
    pnl: float = 0.0
    gross_pnl: float = 0.0
    commission: float = 0.0
    
    # Analytics
    bars_held: int = 0
    mae: float = 0.0
    mfe: float = 0.0
    r_multiple: float = 0.0  # PnL / initial risk


class Account:
    """
    Trading account with position and PnL tracking.
    """
    
    def __init__(
        self,
        starting_balance: float = 50000.0,
        costs: CostModel = None
    ):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.costs = costs or DEFAULT_COSTS
        
        self.positions: List[Position] = []
        self.trades: List[TradeRecord] = []
        
        # Running stats
        self.realized_pnl = 0.0
        self.peak_balance = starting_balance
        self.max_drawdown = 0.0
    
    def open_position(
        self,
        fill: Fill,
        stop_loss: float = None,
        take_profit: float = None,
        time: pd.Timestamp = None
    ) -> Position:
        """Open new position from fill."""
        position = Position(
            direction=fill.direction,
            entry_price=fill.fill_price,
            size=fill.size,
            entry_bar=fill.fill_bar,
            entry_time=time,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.positions.append(position)
        return position
    
    def close_position(
        self,
        position: Position,
        fill: Fill,
        outcome: str = "",
        mae: float = 0.0,
        mfe: float = 0.0,
        time: pd.Timestamp = None
    ) -> TradeRecord:
        """Close position and record trade."""
        # Calculate PnL
        gross_pnl = self.costs.calculate_pnl(
            position.entry_price,
            fill.fill_price,
            position.direction,
            position.size,
            include_commission=False
        )
        
        commission = self.costs.calculate_commission(position.size, round_trip=True)
        net_pnl = gross_pnl - commission
        
        # Calculate R-multiple if we have stop loss
        r_multiple = 0.0
        if position.stop_loss:
            initial_risk = abs(position.entry_price - position.stop_loss) * self.costs.point_value * position.size
            if initial_risk > 0:
                r_multiple = net_pnl / initial_risk
        
        # Determine outcome if not provided
        if not outcome:
            if net_pnl > 0:
                outcome = 'WIN'
            elif net_pnl < 0:
                outcome = 'LOSS'
            else:
                outcome = 'BREAKEVEN'
        
        # Create trade record
        trade = TradeRecord(
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=fill.fill_price,
            size=position.size,
            entry_bar=position.entry_bar,
            exit_bar=fill.fill_bar,
            entry_time=position.entry_time,
            exit_time=time,
            outcome=outcome,
            pnl=net_pnl,
            gross_pnl=gross_pnl,
            commission=commission,
            bars_held=fill.fill_bar - position.entry_bar,
            mae=mae,
            mfe=mfe,
            r_multiple=r_multiple,
        )
        
        # Update account
        self.trades.append(trade)
        self.positions.remove(position)
        self.balance += net_pnl
        self.realized_pnl += net_pnl
        
        # Update drawdown tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = self.peak_balance - self.balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        return trade
    
    def get_equity(self, current_price: float) -> float:
        """Get current equity (balance + unrealized)."""
        unrealized = 0.0
        for pos in self.positions:
            unrealized += self.costs.calculate_pnl(
                pos.entry_price,
                current_price,
                pos.direction,
                pos.size,
                include_commission=False
            )
        return self.balance + unrealized
    
    def has_open_position(self) -> bool:
        """Check if any position is open."""
        return len(self.positions) > 0
    
    def get_stats(self) -> dict:
        """Get account statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
            }
        
        wins = sum(1 for t in self.trades if t.outcome == 'WIN')
        total = len(self.trades)
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total if total > 0 else 0.0,
            'total_pnl': self.realized_pnl,
            'avg_pnl': self.realized_pnl / total if total > 0 else 0.0,
            'max_drawdown': self.max_drawdown,
            'final_balance': self.balance,
        }

```

### src/sim/bar_fill_model.py

```python
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

```

### src/sim/costs.py

```python
"""
Cost Model
Fees, slippage, and tick rounding.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.config import TICK_SIZE, POINT_VALUE, COMMISSION_PER_SIDE, DEFAULT_SLIPPAGE_TICKS


@dataclass
class CostModel:
    """
    Trading cost model for realistic simulation.
    """
    commission_per_side: float = COMMISSION_PER_SIDE  # Per contract per side
    slippage_ticks: float = DEFAULT_SLIPPAGE_TICKS    # Average slippage
    tick_size: float = TICK_SIZE                       # MES = 0.25
    point_value: float = POINT_VALUE                   # MES = $5
    
    def round_to_tick(self, price: float, direction: str = 'nearest') -> float:
        """
        Round price to valid tick.
        
        Args:
            price: Raw price
            direction: 'nearest', 'up', or 'down'
        """
        if direction == 'up':
            return np.ceil(price / self.tick_size) * self.tick_size
        elif direction == 'down':
            return np.floor(price / self.tick_size) * self.tick_size
        else:
            return round(price / self.tick_size) * self.tick_size
    
    def apply_slippage(
        self,
        price: float,
        direction: str,
        order_type: str = 'MARKET'
    ) -> float:
        """
        Apply slippage to fill price.
        
        Slippage is adverse: 
        - BUY market fills ABOVE quoted price
        - SELL market fills BELOW quoted price
        
        Limit orders have no slippage (fill at limit or better).
        """
        if order_type == 'LIMIT':
            return price
        
        slippage_points = self.slippage_ticks * self.tick_size
        
        if direction == 'LONG':
            # Buying - slip up
            return self.round_to_tick(price + slippage_points, 'up')
        else:
            # Selling - slip down
            return self.round_to_tick(price - slippage_points, 'down')
    
    def calculate_commission(self, contracts: int, round_trip: bool = True) -> float:
        """Calculate commission in dollars."""
        sides = 2 if round_trip else 1
        return contracts * self.commission_per_side * sides
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        direction: str,
        contracts: int,
        include_commission: bool = True
    ) -> float:
        """
        Calculate trade PnL in dollars.
        
        Args:
            entry_price: Entry fill price
            exit_price: Exit fill price
            direction: 'LONG' or 'SHORT'
            contracts: Number of contracts
            include_commission: Whether to subtract commission
        """
        if direction == 'LONG':
            points = exit_price - entry_price
        else:
            points = entry_price - exit_price
        
        gross_pnl = points * self.point_value * contracts
        
        if include_commission:
            commission = self.calculate_commission(contracts, round_trip=True)
            return gross_pnl - commission
        
        return gross_pnl
    
    def calculate_risk(
        self,
        entry_price: float,
        stop_price: float,
        contracts: int
    ) -> float:
        """Calculate risk in dollars (not including commission)."""
