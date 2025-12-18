"""
Viz Config
Configuration for visualization export.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VizConfig:
    """Configuration for viz export."""
    
    # What to include
    include_full_series: bool = False  # Full OHLCV for overview mode
    include_windows: bool = True       # x_price windows at decision time
    include_model_outputs: bool = True # Logits/probabilities
    
    # Window settings
    window_lookback_1m: int = 120
    window_lookback_5m: int = 24
    window_lookback_15m: int = 8
    
    # Output format
    output_format: str = "jsonl"  # 'json' or 'jsonl'
    compress: bool = False
    
    def to_dict(self) -> dict:
        return {
            'include_full_series': self.include_full_series,
            'include_windows': self.include_windows,
            'include_model_outputs': self.include_model_outputs,
            'window_lookback_1m': self.window_lookback_1m,
            'window_lookback_5m': self.window_lookback_5m,
            'window_lookback_15m': self.window_lookback_15m,
            'output_format': self.output_format,
            'compress': self.compress,
        }
