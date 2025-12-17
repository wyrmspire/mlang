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
