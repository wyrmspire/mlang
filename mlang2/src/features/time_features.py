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
