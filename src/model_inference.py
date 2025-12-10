"""
Real-time model inference for Limit Order Prediction.
Uses the trained CNN_Predictive model (Proactive Strategy).
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.variants import CNN_Predictive
from src.config import MODELS_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("model_inference")

class ModelInference:
    """Handles model loading and inference for predictive limit orders."""
    
    # Model parameters
    WINDOW_SIZE = 20
    ATR_PERIOD_15M = 14
    THRESHOLD = 0.15 # Tuned in Phase 10
    
    def __init__(self, model_name: str = "CNN_Predictive_5m"):
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        
    def _load_model(self):
        """Load the trained model."""
        # Clean up extension if present
        clean_name = self.model_name.replace(".pth", "")
        model_path = MODELS_DIR / f"{clean_name}.pth"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return
            
        try:
            # Input Dim = 5 (OHLC + ATR)
            self.model = CNN_Predictive(input_dim=5).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Model loaded: {clean_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None

    def calculate_15m_atr(self, df_1m: pd.DataFrame) -> pd.Series:
        """
        Calculate 15m ATR from 1m data.
        Returns a Series indexed by timestamp, SHIFTED by 1 (no lookahead).
        """
        if df_1m.empty: return pd.Series()
        
        # Resample to 15m
        df_15m = df_1m.set_index('time').resample('15T').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        if len(df_15m) < self.ATR_PERIOD_15M + 1:
            return pd.Series()
            
        high = df_15m['high']
        low = df_15m['low']
        close = df_15m['close']
        
        tr_list = []
        for i in range(len(df_15m)):
             if i==0: 
                 tr = high.iloc[i]-low.iloc[i]
             else:
                tr = max(high.iloc[i]-low.iloc[i], 
                         abs(high.iloc[i]-close.iloc[i-1]), 
                         abs(low.iloc[i]-close.iloc[i-1]))
             tr_list.append(tr)
             
        atr = pd.Series(tr_list, index=df_15m.index).rolling(self.ATR_PERIOD_15M).mean()
        
        # SHIFT 1: The ATR known at 10:00 is the one calculated from data UP TO 10:00.
        # But `resample` labels 10:00-10:15 as 10:00. 
        # So at 10:15 (start of next bar), we can see the 10:00 ATR.
        
        # We want: At 10:05 (inside 10:00-10:15 bar), we use ATR from 09:45 (closed 10:00).
        # Shift 1 means row 10:00 gets 09:45 value.
        return atr.shift(1)

    def analyze(self, candle_index: int, df_1m: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze the market at `candle_index` (1m index).
        We need to form the feature vector from PREVIOUS data.
        """
        if self.model is None or candle_index < 200: # Need enough buffer for resample
            return None
            
        # 1. Slice data up to current point (simulating 'now')
        # We need enough history to calc ATR and Resample
        # Let's say we take everything up to candle_index (inclusive of the 'current' close?)
        # For prediction, we use PAST 20 bars. So if now is T, we use T-20 to T.
        
        # Current time
        current_time = df_1m.iloc[candle_index]['time']
        
        # Get ATR Series (calculated on whole history available so far)
        # In strictly live mode, we'd only have history up to now.
        history_df = df_1m.iloc[:candle_index+1].copy()
        
        # We need at least enough data for 15m resampling + 14 period ATR
        # ~ 15 * 15 = 225 bars min.
        if len(history_df) < 300: 
            return None
            
        atr_series = self.calculate_15m_atr(history_df)
        
        # Get current effective ATR
        # We want the ATR associated with the *current* 5m bar context.
        # Since we shifted, we can just ffill.
        atr_lookup = atr_series.reindex(history_df['time'], method='ffill')
        current_atr = atr_lookup.iloc[-1]
        
        if pd.isna(current_atr) or current_atr <= 0:
            return None
            
        # 2. Prepare 5m Features (Last 20 bars)
        # Resample history to 5m
        df_5m = history_df.set_index('time').resample('5T').agg({
            'open':'first', 'high':'max', 'low':'min', 'close':'last'
        }).dropna()
        
        if len(df_5m) < self.WINDOW_SIZE:
            return None
            
        # Take last 20
        recent = df_5m.tail(self.WINDOW_SIZE)
        
        # Attach ATR to these 5m bars
        # For the feature block, we need the ATR that was known at each bar's time.
        # reindex handles this if we use the Shifted series.
        recent_atr = atr_series.reindex(recent.index, method='ffill')
        
        # Construct Feature Block: O,H,L,C,ATR
        o = recent['open'].values
        h = recent['high'].values
        l = recent['low'].values
        c = recent['close'].values
        a = recent_atr.values
        
        block = np.stack([o, h, l, c, a], axis=1) # (20, 5)
        
        # Normalize (Z-Score)
        mean = np.nanmean(block, axis=0)
        std = np.nanstd(block, axis=0)
        std[std == 0] = 1e-6
        block = (block - mean) / std
        
        # Inference
        inp = torch.tensor(block, dtype=torch.float32).unsqueeze(0).to(self.device).transpose(1, 2)
        # CNN expects (Batch, Channels, Seq) -> (1, 5, 20)
        # Transpose done in model.forward? 
        # In variants.py: x = x.transpose(1, 2). So input should be (Batch, Seq, Dim).
        # My variants.py logic says "x = x.transpose(1, 2)" to convert (B,S,D) -> (B,D,S).
        # So input should be (B, S, D) = (1, 20, 5).
        
        # Wait, I changed `variants.py` to:
        # def forward(self, x): x = x.transpose(1, 2) ...
        # So I should pass (1, 20, 5).
        inp = torch.tensor(block, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob = self.model(inp).item()
            
        if prob > -1.0: # Always return
            
            # Place Limit Orders
            limit_dist = 1.5 * current_atr
            current_close = float(history_df.iloc[-1]['close']) 
            
            return {
                "signal": {
                    "type": "OCO_LIMIT",
                    "prob": prob,
                    "atr": current_atr,
                    "limit_dist": limit_dist,
                    "current_price": current_close,
                    "sell_limit": current_close + limit_dist,
                    "buy_limit": current_close - limit_dist,
                    "sl_dist": 1.0 * current_atr, 
                    "validity": 15 * 60 
                }
            }


def get_available_models():
    return [f.stem for f in MODELS_DIR.glob("*.pth")]
