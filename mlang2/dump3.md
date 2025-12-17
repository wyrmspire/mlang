    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
) -> Dict[int, IndicatorValues]:
    """
    Pre-compute all indicators for efficiency.
    
    Returns dict mapping bar_idx to IndicatorValues.
    This should be cached via FeatureStore.
    """
    from src.features.indicators import calculate_ema, calculate_rsi, calculate_atr, calculate_vwap, calculate_adr
    
    # Compute on each timeframe
    df_5m = df_5m.copy()
    df_5m['ema_200'] = calculate_ema(df_5m['close'], 200)
    df_5m['rsi_14'] = calculate_rsi(df_5m['close'], 14)
    df_5m['atr_14'] = calculate_atr(df_5m, 14)
    
    df_15m = df_15m.copy()
    df_15m['ema_200'] = calculate_ema(df_15m['close'], 200)
    df_15m['rsi_14'] = calculate_rsi(df_15m['close'], 14)
    df_15m['atr_14'] = calculate_atr(df_15m, 14)
    
    # VWAP on 1m
    df_1m = df_1m.copy()
    df_1m['vwap_session'] = calculate_vwap(df_1m, period='session')
    df_1m['vwap_weekly'] = calculate_vwap(df_1m, period='weekly')
    
    # ADR
    df_1m['adr'] = calculate_adr(df_1m, 14)
    
    # Map back to 1m bar indices
    # This is a simplified version - in practice, need proper alignment
    result = {}
    
    for idx in range(len(df_1m)):
        result[idx] = IndicatorValues()
        # ... (would do proper lookups here)
    
    return result

```

### src/features/state.py

```python
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

```

### src/features/time_features.py

```python
"""
Time Features
Session, time of day, and calendar features.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from zoneinfo import ZoneInfo

from src.config import NY_TZ, SESSION_RTH_START, SESSION_RTH_END


class Session(Enum):
    """Trading session."""
    RTH = "RTH"          # Regular Trading Hours (9:30-16:00 NY)
    GLOBEX = "GLOBEX"    # Overnight session (18:00-9:30 NY)
    PRE = "PRE"          # Pre-market (could be 4:00-9:30)
    POST = "POST"        # After-hours
    CLOSED = "CLOSED"    # Market closed


@dataclass
class TimeFeatures:
    """Bundle of time-based features."""
    # Raw values
    hour_ny: int = 0              # Hour in NY time (0-23)
    minute: int = 0               # Minute (0-59)
    day_of_week: int = 0          # 0=Monday, 4=Friday
    
    # Session info
    session: str = "UNKNOWN"      # RTH, GLOBEX, etc.
    mins_into_session: int = 0    # Minutes since session start
    mins_to_session_end: int = 0  # Minutes until session end
    
    # Flags
    is_rth: bool = False          # In regular trading hours
    is_first_hour: bool = False   # First hour of RTH (9:30-10:30)
    is_last_hour: bool = False    # Last hour of RTH (15:00-16:00)
    is_lunch: bool = False        # Lunch (12:00-13:00 NY)
    
    # Cyclical encodings (for neural nets)
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    dow_sin: float = 0.0
    dow_cos: float = 0.0


def get_session(
    timestamp: pd.Timestamp,
    tz: ZoneInfo = NY_TZ
) -> Session:
    """
    Determine which trading session a timestamp falls in.
    
    Args:
        timestamp: Timezone-aware timestamp
        tz: Reference timezone (NY for CME)
        
    Returns:
        Session enum value
    """
    # Convert to NY time
    t = timestamp.astimezone(tz)
    
    # Check if weekend
    if t.weekday() >= 5:  # Saturday=5, Sunday=6
        return Session.CLOSED
    
    hour = t.hour
    minute = t.minute
    time_mins = hour * 60 + minute
    
    # RTH: 9:30 - 16:00 (570 - 960 mins)
    rth_start_mins = 9 * 60 + 30
    rth_end_mins = 16 * 60
    
    # Globex: 18:00 - 9:30 (next day)
    globex_start_mins = 18 * 60
    
    if rth_start_mins <= time_mins < rth_end_mins:
        return Session.RTH
    elif time_mins >= globex_start_mins or time_mins < rth_start_mins:
        return Session.GLOBEX
    else:
        return Session.CLOSED


def compute_time_features(
    timestamp: pd.Timestamp,
    tz: ZoneInfo = NY_TZ
) -> TimeFeatures:
    """
    Compute all time-based features for a timestamp.
    """
    t = timestamp.astimezone(tz)
    
    hour = t.hour
    minute = t.minute
    dow = t.weekday()
    
    session = get_session(timestamp, tz)
    
    # RTH boundaries
    rth_start_mins = 9 * 60 + 30
    rth_end_mins = 16 * 60
    current_mins = hour * 60 + minute
    
    # Session timing
    is_rth = session == Session.RTH
    if is_rth:
        mins_into = current_mins - rth_start_mins
        mins_to_end = rth_end_mins - current_mins
    else:
        mins_into = 0
        mins_to_end = 0
    
    # Cyclical encodings
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 5)  # 5 trading days
    dow_cos = np.cos(2 * np.pi * dow / 5)
    
    return TimeFeatures(
        hour_ny=hour,
        minute=minute,
        day_of_week=dow,
        session=session.value,
        mins_into_session=mins_into,
        mins_to_session_end=mins_to_end,
        is_rth=is_rth,
        is_first_hour=(is_rth and hour == 9) or (is_rth and hour == 10 and minute < 30),
        is_last_hour=is_rth and hour == 15,
        is_lunch=is_rth and hour == 12,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        dow_sin=dow_sin,
        dow_cos=dow_cos,
    )


def add_time_features(
    df: pd.DataFrame,
    time_col: str = 'time',
    tz: ZoneInfo = NY_TZ
) -> pd.DataFrame:
    """
    Add time features as columns to a DataFrame.
    """
    df = df.copy()
    
    times = pd.to_datetime(df[time_col])
    if times.dt.tz is None:
        times = times.tz_localize('UTC').tz_convert(tz)
    else:
        times = times.dt.tz_convert(tz)
    
    df['hour_ny'] = times.dt.hour
    df['minute'] = times.dt.minute
    df['day_of_week'] = times.dt.weekday
    
    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_ny'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_ny'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    
    # Session
    df['is_rth'] = ((df['hour_ny'] > 9) | ((df['hour_ny'] == 9) & (df['minute'] >= 30))) & (df['hour_ny'] < 16)
    df['is_first_hour'] = df['is_rth'] & ((df['hour_ny'] == 9) | ((df['hour_ny'] == 10) & (df['minute'] < 30)))
    df['is_last_hour'] = df['is_rth'] & (df['hour_ny'] == 15)
    df['is_lunch'] = df['is_rth'] & (df['hour_ny'] == 12)
    
    return df

```

### src/labels/__init__.py

```python
# Labels module
"""Future-aware outcome computation - QUARANTINED from features/policy."""

```

### src/labels/counterfactual.py

```python
"""
Counterfactual Labeler
Label ALL decision points with "what would have happened".
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from src.labels.future_window import FutureWindowProvider
from src.labels.trade_outcome import TradeOutcome, compute_trade_outcome
from src.sim.oco import OCOConfig, create_oco_bracket
from src.sim.bar_fill_model import BarFillConfig, BarFillEngine
from src.sim.costs import CostModel, DEFAULT_COSTS


@dataclass
class CounterfactualLabel:
    """
    Counterfactual outcome label.
    
    "What WOULD have happened if we traded here?"
    """
    # Primary outcome
    outcome: str          # WIN, LOSS, TIMEOUT
    pnl: float           # Points
    pnl_dollars: float   # Actual dollars (with costs)
    
    # Excursions
    mae: float           # Max Adverse Excursion (points)
    mfe: float           # Max Favorable Excursion (points)
    mae_atr: float       # MAE normalized by ATR
    mfe_atr: float       # MFE normalized by ATR
    
    # Timing
    bars_held: int
    
    # Prices
    entry_price: float
    exit_price: float
    stop_price: float
    tp_price: float
    
    # OCO config used
    oco_config: OCOConfig = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cf_outcome': self.outcome,
            'cf_pnl': self.pnl,
            'cf_pnl_dollars': self.pnl_dollars,
            'cf_mae': self.mae,
            'cf_mfe': self.mfe,
            'cf_mae_atr': self.mae_atr,
            'cf_mfe_atr': self.mfe_atr,
            'cf_bars_held': self.bars_held,
            'cf_entry_price': self.entry_price,
            'cf_exit_price': self.exit_price,
        }


def compute_counterfactual(
    df: pd.DataFrame,
    entry_idx: int,
    oco_config: OCOConfig,
    atr: float,
    fill_config: BarFillConfig = None,
    costs: CostModel = None,
    max_bars: int = 200
) -> CounterfactualLabel:
    """
    Compute counterfactual outcome for a decision point.
    
    "If we entered here with this OCO, what would happen?"
    
    Args:
        df: Full dataframe
        entry_idx: Bar index of decision point
        oco_config: OCO configuration to simulate
        atr: ATR at decision point (for bracket calculation)
        fill_config: Bar fill configuration
        costs: Cost model
        max_bars: Max bars to simulate
        
    Returns:
        CounterfactualLabel with complete outcome info
    """
    costs = costs or DEFAULT_COSTS
    fill_config = fill_config or BarFillConfig()
    
    # Create OCO bracket
    entry_bar = df.iloc[entry_idx]
    base_price = entry_bar['close']
    
    bracket = create_oco_bracket(
        config=oco_config,
        base_price=base_price,
        atr=atr,
        costs=costs
    )
    
    # Create future provider
    future_provider = FutureWindowProvider(df, entry_idx)
    
    # Compute outcome
    outcome = compute_trade_outcome(
        future_provider=future_provider,
        entry_price=bracket.entry_price,
        direction=oco_config.direction,
        stop_loss=bracket.stop_price,
        take_profit=bracket.tp_price,
        max_bars=max_bars,
        fill_config=fill_config
    )
    
    # Calculate dollar PnL
    pnl_dollars = costs.calculate_pnl(
        bracket.entry_price,
        outcome.exit_price,
        oco_config.direction,
        contracts=1,
        include_commission=True
    )
    
    return CounterfactualLabel(
        outcome=outcome.outcome,
        pnl=outcome.pnl,
        pnl_dollars=pnl_dollars,
        mae=outcome.mae,
        mfe=outcome.mfe,
        mae_atr=outcome.mae / atr if atr > 0 else 0,
        mfe_atr=outcome.mfe / atr if atr > 0 else 0,
        bars_held=outcome.bars_held,
        entry_price=bracket.entry_price,
        exit_price=outcome.exit_price,
        stop_price=bracket.stop_price,
        tp_price=bracket.tp_price,
        oco_config=oco_config,
    )


def compute_multi_oco_counterfactuals(
    df: pd.DataFrame,
    entry_idx: int,
    oco_configs: List[OCOConfig],
    atr: float,
    fill_config: BarFillConfig = None,
    costs: CostModel = None
) -> Dict[str, CounterfactualLabel]:
    """
    Compute counterfactual outcomes for multiple OCO variants.
    
    Enables "which OCO would have worked best?" analysis.
    
    Returns:
        Dict mapping oco_config.name to CounterfactualLabel
    """
    results = {}
    
    for oco in oco_configs:
        label = compute_counterfactual(
            df=df,
            entry_idx=entry_idx,
            oco_config=oco,
            atr=atr,
            fill_config=fill_config,
            costs=costs
        )
        name = oco.name or f"{oco.direction}_{oco.tp_multiple}R"
        results[name] = label
    
    return results


def label_is_good_skip(cf_label: CounterfactualLabel) -> bool:
    """
    Determine if skipping this trade was a good decision.
    
    "Skipped good" = would have lost
    "Skipped bad" = would have won
    
    From the perspective of improving the model, we want to
    learn to skip the losses.
    """
    return cf_label.outcome == 'LOSS'


def label_is_bad_skip(cf_label: CounterfactualLabel) -> bool:
    """Trade we skipped but should have taken (would have won)."""
    return cf_label.outcome == 'WIN'

```

### src/labels/future_window.py

```python
"""
Future Window Provider
QUARANTINED future access - only used in labels/ module.
"""

import pandas as pd
import numpy as np
from typing import Optional


class FutureWindowProvider:
    """
    Provides access to future data for labeling.
    
    This class is ONLY instantiated inside the labels/ module.
    It takes the full dataframe and an entry index, then provides
    controlled access to future bars.
    
    Physical separation ensures features/ and policy/ cannot
    accidentally access future data.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        time_col: str = 'time'
    ):
        """
        Initialize future window provider.
        
        Args:
            df: Full dataframe with all bars
            entry_idx: Index of the entry bar (decision point)
            time_col: Name of time column
        """
        self.df = df
        self.entry_idx = entry_idx
        self.time_col = time_col
        self._validate()
    
    def _validate(self):
        """Validate inputs."""
        if self.entry_idx < 0:
            raise ValueError("entry_idx must be >= 0")
        if self.entry_idx >= len(self.df):
            raise ValueError(f"entry_idx {self.entry_idx} >= data length {len(self.df)}")
    
    def get_future(self, lookahead: int) -> pd.DataFrame:
        """
        Get next N bars AFTER entry.
        
        Does NOT include the entry bar itself.
        """
        start = self.entry_idx + 1
        end = min(start + lookahead, len(self.df))
        return self.df.iloc[start:end].copy()
    
    def get_future_array(
        self,
        lookahead: int,
        columns: list = None
    ) -> np.ndarray:
        """Get future as numpy array."""
        columns = columns or ['open', 'high', 'low', 'close']
        future = self.get_future(lookahead)
        
        if len(future) == 0:
            return np.array([]).reshape(0, len(columns))
        
        return future[columns].values
    
    def get_entry_bar(self) -> pd.Series:
        """Get the entry bar."""
        return self.df.iloc[self.entry_idx]
    
    def get_entry_price(self, price_col: str = 'close') -> float:
        """Get entry price (default: close of entry bar)."""
        return self.df.iloc[self.entry_idx][price_col]
    
    def get_entry_time(self) -> Optional[pd.Timestamp]:
        """Get entry timestamp."""
        bar = self.get_entry_bar()
        if self.time_col in bar:
            return bar[self.time_col]
        return None
    
    def bars_available(self) -> int:
        """Number of future bars available."""
        return len(self.df) - self.entry_idx - 1
    
    def has_sufficient_future(self, required: int) -> bool:
        """Check if enough future bars exist."""
        return self.bars_available() >= required

```

### src/labels/labeler.py

```python
"""
Labeler
Main labeling pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pandas as pd

from src.labels.counterfactual import (
    CounterfactualLabel, 
    compute_counterfactual,
    compute_multi_oco_counterfactuals
)
from src.sim.oco import OCOConfig
from src.sim.bar_fill_model import BarFillConfig
from src.sim.costs import CostModel, DEFAULT_COSTS


@dataclass
class LabelConfig:
    """Configuration for labeling."""
    # Primary OCO for counterfactual
    oco_config: OCOConfig = field(default_factory=OCOConfig)
    
    # Optional: additional OCO variants for multi-armed bandit
    oco_variants: List[OCOConfig] = field(default_factory=list)
    
    # Fill model
    fill_config: BarFillConfig = field(default_factory=BarFillConfig)
    
    # Cost model
    cost_model: CostModel = field(default_factory=lambda: DEFAULT_COSTS)
    
    # Simulation
    max_bars: int = 200
    
    def to_dict(self) -> dict:
        return {
            'oco_config': self.oco_config.to_dict(),
            'oco_variants': [o.to_dict() for o in self.oco_variants],
            'fill_config': self.fill_config.to_dict(),
            'max_bars': self.max_bars,
        }


class Labeler:
    """
    Main labeling class.
    
    Takes decision points and adds counterfactual labels.
    """
    
    def __init__(self, config: LabelConfig):
        self.config = config
    
    def label_decision_point(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        atr: float
    ) -> CounterfactualLabel:
        """
        Label a single decision point.
        
        Args:
            df: Full dataframe
            entry_idx: Index of decision point
            atr: ATR at decision point
            
        Returns:
            CounterfactualLabel
        """
        return compute_counterfactual(
            df=df,
            entry_idx=entry_idx,
            oco_config=self.config.oco_config,
            atr=atr,
            fill_config=self.config.fill_config,
            costs=self.config.cost_model,
            max_bars=self.config.max_bars
        )
    
    def label_decision_point_multi(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        atr: float
    ) -> Dict[str, CounterfactualLabel]:
        """
        Label with multiple OCO variants.
        """
        all_ocos = [self.config.oco_config] + self.config.oco_variants
        
        return compute_multi_oco_counterfactuals(
            df=df,
            entry_idx=entry_idx,
            oco_configs=all_ocos,
            atr=atr,
            fill_config=self.config.fill_config,
            costs=self.config.cost_model
        )
    
    def label_batch(
        self,
        df: pd.DataFrame,
        entry_indices: List[int],
        atrs: List[float]
    ) -> List[CounterfactualLabel]:
        """
        Label a batch of decision points.
        """
        results = []
        for idx, atr in zip(entry_indices, atrs):
            label = self.label_decision_point(df, idx, atr)
            results.append(label)
        return results


# Convenience function
def create_default_labeler(
    direction: str = "LONG",
    tp_multiple: float = 1.4,
    stop_atr: float = 1.0
) -> Labeler:
    """Create labeler with common defaults."""
    oco = OCOConfig(
        direction=direction,
        tp_multiple=tp_multiple,
        stop_atr=stop_atr,
        name=f"{direction}_{tp_multiple}R"
    )
    config = LabelConfig(oco_config=oco)
    return Labeler(config)

```

### src/labels/trade_outcome.py

```python
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
