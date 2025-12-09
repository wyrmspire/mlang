import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR, MODELS_DIR, LOCAL_TZ
from src.utils.logging_utils import get_logger

logger = get_logger("smart_cnn")

# --- Architecture (Must match training!) ---
class TradeCNN(nn.Module):
    def __init__(self, input_len=20, input_channels=4):
        super(TradeCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * 5, 32) 
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str 
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = 0.0
    outcome: str = None 

class SmartCNNStrategy:
    def __init__(self, 
                 model_path: Path = MODELS_DIR / "setup_cnn_v1.pth",
                 tp_ticks: int = 20, 
                 sl_ticks: int = 10,
                 threshold: float = 0.6): # Confidence threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradeCNN().to(self.device)
        
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.error(f"Model not found at {model_path}")
            
        self.tp_dist = tp_ticks * 0.25
        self.sl_dist = sl_ticks * 0.25
        self.threshold = threshold
        self.trades = []

    def get_prediction(self, df_window):
        # Prepare Input
        # Needs 20 bars. 
        if len(df_window) < 20: 
            return 0.0, 0.0 # Prob Long, Prob Short
            
        # Normalize
        base_price = df_window.iloc[0]['open']
        feats = df_window[['open', 'high', 'low', 'close']].values
        feats_norm = (feats / base_price) - 1.0
        
        # Ensure exact 20
        feats_norm = feats_norm[-20:]
        
        # Create Batch (1, 20, 4) -> (1, 4, 20) handled by model
        # Input Long
        input_long = torch.FloatTensor(feats_norm).unsqueeze(0).to(self.device)
        # Input Short (Inverted)
        input_short = torch.FloatTensor(-feats_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob_long = self.model(input_long).item()
            prob_short = self.model(input_short).item()
            
        return prob_long, prob_short

    def run_simulation(self, start_date_str: str = "2025-07-07 16:40:00"):
        input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if not input_path.exists(): return
        
        logger.info(f"Simulating Smart Strategy (Test Set starting {start_date_str})...")
        df_1m = pd.read_parquet(input_path)
        df_1m['time'] = pd.to_datetime(df_1m['time'])
        df_1m = df_1m.sort_values('time').set_index('time')
        
        # Filter for Test Period
        start_ts = pd.Timestamp(start_date_str).tz_localize('UTC') if 'UTC' not in start_date_str else pd.Timestamp(start_date_str)
        # Check tz awareness of df
        if df_1m.index.tz is None:
            # Assume UTC if data is UTC
            pass 
        else:
            if start_ts.tz is None: start_ts = start_ts.tz_localize('UTC')
        
        # We need context Before start_ts, so slice generously then filter triggers
        df_1m_test = df_1m.loc[start_ts - pd.Timedelta(hours=1):]
        
        # Resample for 20m triggers
        # We need "Last 5m candle" reference.
        df_5m = df_1m_test.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        triggers = df_5m[df_5m.index.minute % 20 == 0]
        triggers = triggers[triggers.index >= start_ts]
        
        logger.info(f"Found {len(triggers)} test opportunities.")
        
        count = 0
        for current_time, row in triggers.iterrows():
            trigger_time = current_time
            
            # Context for Model: 20m before trigger
            context_end = trigger_time
            context_start = context_end - pd.Timedelta(minutes=20)
            
            # Fetch 1m context
            context_window = df_1m.loc[context_start:context_end]
            # Precise slice: strictly < trigger_time?
            context_window = context_window[context_window.index < trigger_time]
            
            if len(context_window) < 15: continue # Skip if missing data
            
            p_long, p_short = self.get_prediction(context_window)
            
            if count < 20: 
                 logger.info(f"Pred: L={p_long:.4f} S={p_short:.4f}")
            
            # Decision
            direction = None
            # Relaxed logic: If both meet threshold, pick higher or random tie-break
            if p_long > self.threshold and p_long >= p_short:
                direction = 'LONG'
                confidence = p_long
            elif p_short > self.threshold and p_short > p_long:
                direction = 'SHORT'
                confidence = p_short
                
            if not direction:
                continue
                
            # EXECUTION (Dynamic Sizing)
            # Need previous 5m candle for sizing
            prev_time = trigger_time - pd.Timedelta(minutes=5)
            if prev_time not in df_5m.index: continue
            prev_bar = df_5m.loc[prev_time]
            candle_range = prev_bar['high'] - prev_bar['low']
            if candle_range == 0: candle_range = 0.25
            
            sl_dist = 2.0 * candle_range
            tp_dist = 3.0 * candle_range
            
            entry_price = prev_bar['close'] # Approx fill at close of prev bar (Open of current)
            if trigger_time in df_1m.index:
                 entry_price = df_1m.loc[trigger_time]['open']
            
            if direction == 'LONG':
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist
                
            # Simulate Outcome
            future = df_1m.loc[trigger_time:]
            outcome = 'TIMEOUT'
            exit_px = entry_price
            exit_t = trigger_time
            
            # Vectorized Check (Subset 2000 bars)
            subset = future.iloc[:2000]
            if subset.empty: continue
            
            times = subset.index.values
            highs = subset['high'].values
            lows = subset['low'].values
            closes = subset['close'].values
            
            if direction == 'LONG':
                 mask_win = highs >= tp_price
                 mask_loss = lows <= sl_price
            else:
                 mask_win = lows <= tp_price
                 mask_loss = highs >= sl_price
                 
            idx_win = np.argmax(mask_win) if mask_win.any() else 999999
            idx_loss = np.argmax(mask_loss) if mask_loss.any() else 999999
            
            if idx_win == 999999 and idx_loss == 999999:
                outcome = 'TIMEOUT'
                exit_px = closes[-1]
                exit_t = times[-1]
            elif idx_win < idx_loss:
                outcome = 'WIN'
                exit_px = tp_price
                exit_t = times[idx_win]
            else:
                outcome = 'LOSS'
                exit_px = sl_price
                exit_t = times[idx_loss]
            
            pnl = (exit_px - entry_price) * (1 if direction == 'LONG' else -1)
            
            self.trades.append({
                'entry_time': trigger_time,
                'direction': direction,
                'pnl': pnl,
                'outcome': outcome,
                'confidence': confidence
            })
            
            count += 1
            if count % 100 == 0:
                logger.info(f"Simulated {count} trades... Last PnL: {pnl:.2f}")

        logger.info(f"Smart Simulation Complete. Trades: {len(self.trades)}")
        if self.trades:
            df_res = pd.DataFrame(self.trades)
            wins = df_res[df_res['outcome'] == 'WIN']
            wr = len(wins) / len(df_res)
            logger.info(f"Win Rate: {wr:.2f} | Avg PnL: {df_res['pnl'].mean():.2f} | Total PnL: {df_res['pnl'].sum():.2f}")
            out_path = PROCESSED_DIR / "smart_verification_trades.parquet"
            df_res.to_parquet(out_path)

if __name__ == "__main__":
    # Threshold 0.38 since model output is around 0.40
    strat = SmartCNNStrategy(threshold=0.38) 
    strat.run_simulation()
