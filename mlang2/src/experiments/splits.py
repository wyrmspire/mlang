"""
Walk-Forward Splits
Time-series cross-validation with embargo.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WalkForwardConfig:
    """Walk-forward split configuration."""
    train_weeks: int = 3
    test_weeks: int = 1
    embargo_bars: int = 100  # Gap to prevent feature leakage
    min_train_records: int = 1000


@dataclass
class Split:
    """Single train/test split."""
    split_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    def __repr__(self):
        return (f"Split(train={self.train_start.date()}→{self.train_end.date()}, "
                f"test={self.test_start.date()}→{self.test_end.date()})")


def generate_walk_forward_splits(
    df: pd.DataFrame,
    config: WalkForwardConfig,
    time_col: str = 'time'
) -> List[Split]:
    """
    Generate walk-forward splits with embargo gaps.
    
    Layout:
    |---train---|--embargo--|---test---|---train---|--embargo--|---test---|
    
    Args:
        df: DataFrame with time column
        config: Split configuration
        time_col: Name of time column
        
    Returns:
        List of Split objects
    """
    times = pd.to_datetime(df[time_col]).sort_values()
    start_time = times.min()
    end_time = times.max()
    
    train_duration = pd.Timedelta(weeks=config.train_weeks)
    test_duration = pd.Timedelta(weeks=config.test_weeks)
    embargo_duration = pd.Timedelta(minutes=config.embargo_bars)  # Assuming 1m bars
    
    splits = []
    current_start = start_time
    split_idx = 0
    
    while True:
        train_end = current_start + train_duration
        embargo_start = train_end
        embargo_end = embargo_start + embargo_duration
        test_start = embargo_end
        test_end = test_start + test_duration
        
        # Check if we have enough data for this split
        if test_end > end_time:
            break
        
        split = Split(
            split_idx=split_idx,
            train_start=current_start,
            train_end=train_end,
            embargo_start=embargo_start,
            embargo_end=embargo_end,
            test_start=test_start,
            test_end=test_end,
        )
        splits.append(split)
        
        # Move to next window
        current_start = test_start  # Or test_end for non-overlapping
        split_idx += 1
    
    return splits


def apply_split(
    df: pd.DataFrame,
    split: Split,
    time_col: str = 'time'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a split to get train and test DataFrames.
    """
    times = pd.to_datetime(df[time_col])
    
    train_mask = (times >= split.train_start) & (times < split.train_end)
    test_mask = (times >= split.test_start) & (times < split.test_end)
    
    return df[train_mask].copy(), df[test_mask].copy()


def apply_embargo_to_records(
    train_records: list,
    test_start: pd.Timestamp,
    embargo_bars: int
) -> list:
    """
    Remove training records within embargo window of test start.
    
    Prevents information leakage from rolling features.
    """
    # Calculate cutoff time
    embargo_duration = pd.Timedelta(minutes=embargo_bars)
    cutoff = test_start - embargo_duration
    
    # Filter records
    filtered = [r for r in train_records if r.timestamp < cutoff]
    
    removed = len(train_records) - len(filtered)
    if removed > 0:
        print(f"Embargo: removed {removed} records within {embargo_bars} bars of test start")
    
    return filtered
