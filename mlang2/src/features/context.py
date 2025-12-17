"""
Context Features
Derived context vector (x_context) for MLP input.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.features.state import MarketState
from src.features.indicators import IndicatorValues
from src.features.levels import LevelValues
from src.features.time_features import TimeFeatures


@dataclass
class ContextFeatures:
    """
    Context feature vector for MLP input.
    
    These are scalar/low-dim features derived from indicators and state,
    separate from the raw OHLCV windows used by CNN.
    """
    # EMA distances (normalized by ATR)
    dist_ema_5m_200_atr: float = 0.0
    dist_ema_15m_200_atr: float = 0.0
    
    # VWAP distances
    dist_vwap_session_atr: float = 0.0
    dist_vwap_weekly_atr: float = 0.0
    
    # Level distances
    dist_nearest_1h_level_atr: float = 0.0
    dist_nearest_4h_level_atr: float = 0.0
    dist_pdh_atr: float = 0.0
    dist_pdl_atr: float = 0.0
    
    # Volatility
    adr_pct_used: float = 0.0
    
    # Momentum
    rsi_5m_14: float = 50.0
    rsi_15m_14: float = 50.0
    
    # Volume
    relative_volume: float = 1.0
    
    # Time (cyclical)
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    dow_sin: float = 0.0
    dow_cos: float = 0.0
    
    # Time (flags)
    is_rth: float = 0.0
    is_first_hour: float = 0.0
    is_last_hour: float = 0.0
    mins_into_session: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.dist_ema_5m_200_atr,
            self.dist_ema_15m_200_atr,
            self.dist_vwap_session_atr,
            self.dist_vwap_weekly_atr,
            self.dist_nearest_1h_level_atr,
            self.dist_nearest_4h_level_atr,
            self.dist_pdh_atr,
            self.dist_pdl_atr,
            self.adr_pct_used,
            self.rsi_5m_14 / 100.0,  # Normalize to [0, 1]
            self.rsi_15m_14 / 100.0,
            self.relative_volume,
            self.hour_sin,
            self.hour_cos,
            self.dow_sin,
            self.dow_cos,
            self.is_rth,
            self.is_first_hour,
            self.is_last_hour,
            self.mins_into_session / 390.0,  # Normalize by RTH length
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered feature names."""
        return [
            'dist_ema_5m_200_atr',
            'dist_ema_15m_200_atr',
            'dist_vwap_session_atr',
            'dist_vwap_weekly_atr',
            'dist_nearest_1h_level_atr',
            'dist_nearest_4h_level_atr',
            'dist_pdh_atr',
            'dist_pdl_atr',
            'adr_pct_used',
            'rsi_5m_14_norm',
            'rsi_15m_14_norm',
            'relative_volume',
            'hour_sin',
            'hour_cos',
            'dow_sin',
            'dow_cos',
            'is_rth',
            'is_first_hour',
            'is_last_hour',
            'mins_into_session_norm',
        ]
    
    @staticmethod
    def dim() -> int:
        """Get feature dimension."""
        return 20


def compute_context_features(
    current_price: float,
    indicators: IndicatorValues,
    levels: LevelValues,
    time_features: TimeFeatures,
    atr: float = 1.0
) -> ContextFeatures:
    """
    Compute context feature vector from component features.
    
    All distances are normalized by ATR for scale-independence.
    """
    if atr <= 0:
        atr = 1.0
    
    # EMA distances
    dist_ema_5m = (current_price - indicators.ema_5m_200) / atr if indicators.ema_5m_200 else 0
    dist_ema_15m = (current_price - indicators.ema_15m_200) / atr if indicators.ema_15m_200 else 0
    
    # VWAP distances
    dist_vwap_session = (current_price - indicators.vwap_session) / atr if indicators.vwap_session else 0
    dist_vwap_weekly = (current_price - indicators.vwap_weekly) / atr if indicators.vwap_weekly else 0
    
    # Level distances (use nearest, sign indicates above/below)
    dist_1h = min(abs(levels.dist_1h_high), abs(levels.dist_1h_low)) / atr
    if levels.dist_1h_high < levels.dist_1h_low:
        dist_1h = -dist_1h  # Closer to resistance (above)
    
    dist_4h = min(abs(levels.dist_4h_high), abs(levels.dist_4h_low)) / atr
    if levels.dist_4h_high < levels.dist_4h_low:
        dist_4h = -dist_4h
    
    return ContextFeatures(
        dist_ema_5m_200_atr=dist_ema_5m,
        dist_ema_15m_200_atr=dist_ema_15m,
        dist_vwap_session_atr=dist_vwap_session,
        dist_vwap_weekly_atr=dist_vwap_weekly,
        dist_nearest_1h_level_atr=dist_1h,
        dist_nearest_4h_level_atr=dist_4h,
        dist_pdh_atr=levels.dist_pdh / atr if levels.pdh else 0,
        dist_pdl_atr=levels.dist_pdl / atr if levels.pdl else 0,
        adr_pct_used=indicators.adr_pct_used,
        rsi_5m_14=indicators.rsi_5m_14,
        rsi_15m_14=indicators.rsi_15m_14,
        relative_volume=indicators.relative_volume,
        hour_sin=time_features.hour_sin,
        hour_cos=time_features.hour_cos,
        dow_sin=time_features.dow_sin,
        dow_cos=time_features.dow_cos,
        is_rth=float(time_features.is_rth),
        is_first_hour=float(time_features.is_first_hour),
        is_last_hour=float(time_features.is_last_hour),
        mins_into_session=float(time_features.mins_into_session),
    )
