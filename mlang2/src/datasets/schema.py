"""
Dataset Schema
Explicit separation of x_price (CNN) from x_context (MLP).
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DatasetSchema:
    """
    Defines the structure of training data.
    
    Separates:
    - x_price: Raw OHLCV windows for CNN
    - x_context: Derived features for MLP
    - y: Labels (classification and regression)
    """
    
    # =========================================================================
    # Price Windows (for CNN)
    # =========================================================================
    # Shape: (lookback, channels) where channels = OHLCV = 5
    x_price_2h_1m: Tuple[int, int] = (120, 5)   # 120 1m bars = 2 hours
    x_price_2h_5m: Tuple[int, int] = (24, 5)    # 24 5m bars = 2 hours
    x_price_2h_15m: Tuple[int, int] = (8, 5)    # 8 15m bars = 2 hours
    
    # =========================================================================
    # Context Vector (for MLP)
    # =========================================================================
    x_context_dim: int = 20
    
    x_context_features: List[str] = field(default_factory=lambda: [
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
    ])
    
    # =========================================================================
    # Labels
    # =========================================================================
    # Classification: Counterfactual outcome
    # NOTE: NO_TRADE is NOT a label class - it's an action
    y_classification: List[str] = field(default_factory=lambda: [
        'WIN',
        'LOSS',
        'TIMEOUT'
    ])
    
    # Regression targets
    y_regression: List[str] = field(default_factory=lambda: [
        'cf_pnl',
        'cf_mae',
        'cf_mfe',
        'cf_bars_held'
    ])
    
    def to_dict(self) -> dict:
        return {
            'x_price_2h_1m': self.x_price_2h_1m,
            'x_price_2h_5m': self.x_price_2h_5m,
            'x_price_2h_15m': self.x_price_2h_15m,
            'x_context_dim': self.x_context_dim,
            'x_context_features': self.x_context_features,
            'y_classification': self.y_classification,
            'y_regression': self.y_regression,
        }
    
    def get_label_idx(self, label: str) -> int:
        """Get index for classification label."""
        return self.y_classification.index(label)
    
    def label_from_idx(self, idx: int) -> str:
        """Get label name from index."""
        return self.y_classification[idx]


# Default schema
DEFAULT_SCHEMA = DatasetSchema()


def validate_record_schema(record, schema: DatasetSchema = None) -> bool:
    """
    Validate that a DecisionRecord matches the schema.
    """
    schema = schema or DEFAULT_SCHEMA
    
    # Check price windows
    if record.x_price_1m is not None:
        expected = schema.x_price_2h_1m
        actual = record.x_price_1m.shape
        if actual != expected:
            return False
    
    # Check context vector
    if record.x_context is not None:
        if len(record.x_context) != schema.x_context_dim:
            return False
    
    # Check label is valid
    if record.cf_outcome and record.cf_outcome not in schema.y_classification:
        return False
    
    return True
