"""
Market Stepper
Bar-by-bar market simulation with CAUSAL data access only.
NO peek_future() method - future access is quarantined in labels/.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class StepResult:
    """Result of a single step."""
    bar: pd.Series
    bar_idx: int
    is_done: bool


class MarketStepper:
    """
    Bar-by-bar market simulation.
    
    CAUSAL ONLY - no future access.
    Same inputs → same outputs (deterministic).
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        time_col: str = 'time'
    ):
        """
        Initialize stepper.
        
        Args:
            df: DataFrame with OHLCV data (must have time column)
            start_idx: Starting bar index
            end_idx: Ending bar index (exclusive). None = end of data.
            time_col: Name of time column
        """
        self.df = df.reset_index(drop=True)
        self.time_col = time_col
        self.start_idx = start_idx
        self.end_idx = end_idx or len(df)
        self.current_idx = start_idx
        
        if self.start_idx < 0:
            raise ValueError("start_idx must be >= 0")
        if self.end_idx > len(df):
            raise ValueError(f"end_idx {self.end_idx} > data length {len(df)}")
        if self.start_idx >= self.end_idx:
            raise ValueError("start_idx must be < end_idx")
    
    def reset(self, start_idx: Optional[int] = None):
        """Reset stepper to start position."""
        self.current_idx = start_idx if start_idx is not None else self.start_idx
    
    def step(self) -> StepResult:
        """
        Advance one bar.
        
        Returns:
            StepResult with current bar, index, and done flag.
        """
        if self.current_idx >= self.end_idx:
            return StepResult(
                bar=None,
                bar_idx=self.current_idx,
                is_done=True
            )
        
        bar = self.df.iloc[self.current_idx]
        bar_idx = self.current_idx
        self.current_idx += 1
        
        return StepResult(
            bar=bar,
            bar_idx=bar_idx,
            is_done=self.current_idx >= self.end_idx
        )
    
    def get_current_bar(self) -> Optional[pd.Series]:
        """Get current bar (the one just returned by step)."""
        idx = self.current_idx - 1
        if idx < 0 or idx >= len(self.df):
            return None
        return self.df.iloc[idx]
    
    def get_current_idx(self) -> int:
        """Get current bar index."""
        return self.current_idx - 1
    
    def get_current_time(self) -> Optional[pd.Timestamp]:
        """Get current bar timestamp."""
        bar = self.get_current_bar()
        if bar is None:
            return None
        return bar[self.time_col]
    
    def get_history(self, lookback: int) -> pd.DataFrame:
        """
        Get past N bars (CAUSAL - no future leak).
        
        Returns bars from [current_idx - lookback, current_idx).
        If not enough history, returns what's available.
        """
        end_idx = self.current_idx
        start_idx = max(0, end_idx - lookback)
        return self.df.iloc[start_idx:end_idx].copy()
    
    def get_history_array(
        self,
        lookback: int,
        columns: list = None
    ) -> np.ndarray:
        """
        Get history as numpy array for model input.
        
        Args:
            lookback: Number of bars
            columns: Columns to include. Default: ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Array of shape (lookback, n_columns). Padded with zeros if insufficient history.
        """
        columns = columns or ['open', 'high', 'low', 'close', 'volume']
        history = self.get_history(lookback)
        
        # Extract values
        values = history[columns].values
        
        # Pad if insufficient history
        if len(values) < lookback:
            padding = np.zeros((lookback - len(values), len(columns)))
            values = np.vstack([padding, values])
        
        return values.astype(np.float32)
    
    def bars_remaining(self) -> int:
        """Number of bars remaining in simulation."""
        return max(0, self.end_idx - self.current_idx)
    
    def progress(self) -> float:
        """Progress as fraction [0, 1]."""
        total = self.end_idx - self.start_idx
        done = self.current_idx - self.start_idx
        return done / total if total > 0 else 1.0
    
    # NOTE: No peek_future() method exists!
    # Future access is only available via FutureWindowProvider in labels/
