"""
Sweep Configuration Dataclasses
Defines all tunable parameters for the Shotgun Sweep pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class PatternSweepConfig:
    """Configuration for pattern mining geometry."""
    rise_ratio_min: float = 2.5      # Min (Peak - Start) / (Start - Trigger)
    rise_ratio_max: float = 4.0      # Max ratio (invalidation threshold)
    min_drop: float = 1.0            # Minimum $ drop to qualify
    atr_buffer: float = 0.2          # ATR multiplier for stop placement
    validation_distance: float = 1.0  # Distance for pattern validation
    lookback_bars: int = 120         # How far back to scan for patterns
    
    # Unique identifier for this config
    config_id: str = ""
    
    def to_cli_args(self) -> List[str]:
        """Convert to CLI argument list."""
        return [
            "--rise-min", str(self.rise_ratio_min),
            "--rise-max", str(self.rise_ratio_max),
            "--min-drop", str(self.min_drop),
            "--atr-buffer", str(self.atr_buffer),
            "--validation-dist", str(self.validation_distance),
            "--lookback", str(self.lookback_bars),
        ]
    
    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "rise_ratio_min": self.rise_ratio_min,
            "rise_ratio_max": self.rise_ratio_max,
            "min_drop": self.min_drop,
            "atr_buffer": self.atr_buffer,
            "validation_distance": self.validation_distance,
            "lookback_bars": self.lookback_bars,
        }


@dataclass
class CandleComposition:
    """Defines the mix of candle timeframes for model input."""
    candles_1m: int = 30    # Number of 1-minute candles
    candles_3m: int = 20    # Number of 3-minute candles  
    candles_5m: int = 10    # Number of 5-minute candles
    candles_15m: int = 0    # Number of 15-minute candles
    
    @property
    def total_features(self) -> int:
        """Total number of candle input features."""
        return (self.candles_1m + self.candles_3m + 
                self.candles_5m + self.candles_15m) * 4  # OHLC
    
    @property
    def label(self) -> str:
        """Human-readable label for this composition."""
        parts = []
        if self.candles_1m: parts.append(f"{self.candles_1m}x1m")
        if self.candles_3m: parts.append(f"{self.candles_3m}x3m")
        if self.candles_5m: parts.append(f"{self.candles_5m}x5m")
        if self.candles_15m: parts.append(f"{self.candles_15m}x15m")
        return "+".join(parts) if parts else "empty"
    
    def to_cli_args(self) -> List[str]:
        return [
            "--candles-1m", str(self.candles_1m),
            "--candles-3m", str(self.candles_3m),
            "--candles-5m", str(self.candles_5m),
            "--candles-15m", str(self.candles_15m),
        ]


@dataclass
class OCOBracketConfig:
    """Configuration for OCO (One-Cancels-Other) bracket testing."""
    direction: str = "SHORT"          # 'LONG', 'SHORT', 'BOTH'
    r_multiple: float = 1.4           # Take profit as multiple of risk
    stop_atr_pct: float = 0.5         # Stop distance as % of ATR
    stop_type: str = "WICK"           # 'WICK', 'OPEN', 'ATR'
    
    # Unique identifier
    config_id: str = ""
    
    @property
    def label(self) -> str:
        return f"{self.direction}_{self.r_multiple}R_{self.stop_type}_{int(self.stop_atr_pct*100)}atr"
    
    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "direction": self.direction,
            "r_multiple": self.r_multiple,
            "stop_atr_pct": self.stop_atr_pct,
            "stop_type": self.stop_type,
        }


@dataclass
class ModelSweepConfig:
    """Configuration for model architecture and training."""
    architecture: str = "CNN_Classic"  # 'CNN_Classic', 'CNN_Wide', 'LSTM', 'MLP'
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    dropout: float = 0.3
    
    # Input configuration
    candle_composition: CandleComposition = field(default_factory=CandleComposition)
    
    # Unique identifier
    config_id: str = ""
    
    def to_cli_args(self) -> List[str]:
        args = [
            "--architecture", self.architecture,
            "--epochs", str(self.epochs),
            "--lr", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--dropout", str(self.dropout),
        ]
        args.extend(self.candle_composition.to_cli_args())
        return args
    
    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "architecture": self.architecture,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "candle_composition": self.candle_composition.label,
        }


# ============================================================
# Default Sweep Ranges
# ============================================================

PATTERN_SWEEP_RANGES = {
    "rise_ratio_min": (1.5, 3.5),
    "rise_ratio_max": (2.5, 6.0),
    "min_drop": (0.5, 2.0),
    "atr_buffer": (0.1, 0.5),
    "validation_distance": (0.5, 2.0),
    "lookback_bars": (60, 180),
}

OCO_SWEEP_VALUES = {
    "direction": ["LONG", "SHORT"],
    "r_multiple": [1.0, 1.4, 1.8, 2.0, 2.5, 3.0],
    "stop_atr_pct": [0.25, 0.5, 0.75, 1.0],
    "stop_type": ["WICK", "OPEN", "ATR"],
}

MODEL_ARCHITECTURES = ["CNN_Classic", "CNN_Wide", "LSTM_Seq", "Feature_MLP"]

# Pre-defined candle compositions to sweep
CANDLE_COMPOSITIONS = [
    CandleComposition(30, 0, 0, 0),      # Pure 30x1m
    CandleComposition(60, 0, 0, 0),      # Pure 60x1m
    CandleComposition(20, 0, 0, 0),      # Minimal 20x1m
    CandleComposition(30, 20, 10, 0),    # Mixed: 30x1m + 20x3m + 10x5m
    CandleComposition(20, 10, 5, 0),     # Light mixed
    CandleComposition(40, 10, 0, 0),     # Heavy 1m + some 3m
]
