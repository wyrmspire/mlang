"""
Real-time model inference for live trading signals.
Uses the trained rejection CNN model to generate trading signals during playback.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn_model import TradeCNN
from src.config import MODELS_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("model_inference")


class ModelInference:
    """Handles model loading and inference for trading signals."""
    
    # Model parameters
    WINDOW_SIZE = 20
    ATR_PERIOD = 14
    EXTENSION_LOOKBACK = 5
    MIN_ATR_THRESHOLD = 5.0
    EXTENSION_MULTIPLIER = 1.5
    PULLBACK_THRESHOLD = 0.3
    SL_BUFFER = 0.5
    TP_BUFFER = 0.5
    MIN_CONFIDENCE = 0.5
    
    def __init__(self, model_name: str = "rejection_cnn_v1"):
        """
        Initialize model inference.
        
        Args:
            model_name: Name of the model file (without .pth extension)
        """
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk."""
        model_path = MODELS_DIR / f"{self.model_name}.pth"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return
        
        try:
            self.model = TradeCNN().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            logger.info(f"Model loaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def check_for_setup(self, candles_1m: pd.DataFrame, current_idx: int) -> Optional[Dict[str, Any]]:
        """
        Check if current market conditions match a trading setup.
        Uses the inverse strategy logic: looking for continuation patterns.
        
        Args:
            candles_1m: DataFrame with 1-minute candles
            current_idx: Current position in the data (we can only look back from here)
            
        Returns:
            Dict with setup details if found, None otherwise
        """
        if self.model is None:
            return None
        
        # Need at least window_size + ATR period + extension check period
        min_required = self.WINDOW_SIZE + self.ATR_PERIOD + self.EXTENSION_LOOKBACK
        if current_idx < min_required:
            return None
        
        # Get recent candles (5-minute resampled for pattern detection)
        recent_5m = self._resample_to_5m(candles_1m.iloc[:current_idx])
        
        if len(recent_5m) < 20:  # Need at least 20 5m candles
            return None
        
        # Calculate ATR on 5m data
        atr = self._calculate_atr(recent_5m.tail(15), period=self.ATR_PERIOD)
        
        if atr < self.MIN_ATR_THRESHOLD:  # Minimum volatility filter
            return None
        
        # Check for extension pattern (price moved threshold x ATR from recent pivot)
        setup = self._check_extension_pattern(recent_5m, atr, candles_1m, current_idx)
        
        return setup
    
    def _resample_to_5m(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """Resample 1m data to 5m candles."""
        if df_1m.empty or 'time' not in df_1m.columns:
            return pd.DataFrame()
        
        df_copy = df_1m.copy()
        df_copy = df_copy.set_index('time')
        
        resampled = df_copy.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled.reset_index()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period + 1:
            return 0.0
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr_list = []
        for i in range(1, len(df)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        if len(tr_list) < period:
            return 0.0
        
        return np.mean(tr_list[-period:])
    
    def _check_extension_pattern(self, df_5m: pd.DataFrame, atr: float, 
                                   df_1m: pd.DataFrame, current_idx: int) -> Optional[Dict[str, Any]]:
        """
        Check for extension pattern (pullback setup for continuation).
        This implements the inverse strategy logic.
        """
        if len(df_5m) < 10:
            return None
        
        recent = df_5m.tail(10)
        
        # Find pivot point (local min/max in last 10 candles)
        highs = recent['high'].values
        lows = recent['low'].values
        closes = recent['close'].values
        
        # Check for upward extension (looking for long setup)
        max_high = np.max(highs)
        min_low = np.min(lows)
        range_move = max_high - min_low
        
        # Extension threshold: configurable multiplier x ATR
        if range_move < self.EXTENSION_MULTIPLIER * atr:
            return None
        
        # Determine direction based on where price is now vs pivot
        current_price = closes[-1]
        
        # Upward extension (price went up, pulled back - potential LONG)
        if max_high - current_price > self.PULLBACK_THRESHOLD * range_move:
            direction = "LONG"
            entry_price = current_price
            sl_price = min_low - self.SL_BUFFER * atr  # SL below the pivot
            tp_price = max_high + self.TP_BUFFER * atr  # TP at extension high
            
        # Downward extension (price went down, pulled back - potential SHORT)
        elif current_price - min_low > self.PULLBACK_THRESHOLD * range_move:
            direction = "SHORT"
            entry_price = current_price
            sl_price = max_high + self.SL_BUFFER * atr  # SL above the pivot
            tp_price = min_low - self.TP_BUFFER * atr  # TP at extension low
        else:
            return None
        
        # Get model confidence using last 20 1m candles
        confidence = self._get_model_confidence(df_1m, current_idx)
        
        # Only take trades with > threshold confidence
        if confidence < self.MIN_CONFIDENCE:
            return None
        
        return {
            "direction": direction,
            "entry_price": float(entry_price),
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
            "confidence": float(confidence),
            "atr": float(atr),
            "setup_type": "continuation"
        }
    
    def _get_model_confidence(self, df_1m: pd.DataFrame, current_idx: int) -> float:
        """
        Get model confidence score for current setup.
        Uses last WINDOW_SIZE 1m candles before current position.
        """
        if self.model is None:
            return 0.5  # Neutral if no model
        
        # Get last WINDOW_SIZE candles
        start_idx = max(0, current_idx - self.WINDOW_SIZE)
        window = df_1m.iloc[start_idx:current_idx]
        
        if len(window) < self.WINDOW_SIZE:
            return 0.5
        
        # Take exactly last WINDOW_SIZE
        window = window.tail(self.WINDOW_SIZE)
        
        # Extract OHLC features
        feats = window[['open', 'high', 'low', 'close']].values
        
        # Z-score normalization (per window)
        mean = np.mean(feats)
        std = np.std(feats)
        if std == 0:
            std = 1.0
        
        feats_norm = (feats - mean) / std
        
        # Convert to tensor
        X = torch.FloatTensor(feats_norm).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                output = self.model(X)
                confidence = output.item()
            return confidence
        except Exception as e:
            logger.error(f"Error in model inference: {e}")
            return 0.5


def get_available_models() -> list:
    """Get list of available trained models."""
    models = []
    for model_file in MODELS_DIR.glob("*.pth"):
        models.append(model_file.stem)
    return models
