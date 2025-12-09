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

    def calculate_atr(self, df):
        # Rolling 14 period
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr = pd.concat([high-low, (high-close).abs(), (low-close).abs()], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    def run_simulation(self, start_date_str: str = "2025-07-04 08:00:00", initial_balance: float = 50000.0, position_size: int = 1):
        input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if not input_path.exists(): return
        
        logger.info(f"Simulating Continuous Smart Scan (Test Set {start_date_str})...")
        logger.info(f"Assumptions: Balance ${initial_balance:,.0f} | Size {position_size} | Scan: 5m")
        logger.info("Logic: Predicting 'Pre-Conditions' for Win/Loss. Fading high-prob Losers.")
        
        df_1m = pd.read_parquet(input_path)
        df_1m['time'] = pd.to_datetime(df_1m['time'])
        df_1m = df_1m.sort_values('time').set_index('time')
        
        if 'UTC' not in start_date_str:
             start_ts = pd.Timestamp(start_date_str).tz_localize('UTC')
        else:
             start_ts = pd.Timestamp(start_date_str)
             
        # Need context
        df_1m_test = df_1m.loc[start_ts - pd.Timedelta(hours=2):]
        
        # Calculate ATR on 15m for sizing consistency
        df_15m_atr = df_1m_test.resample('15min').agg({'high':'max', 'low':'min', 'close':'last'}).dropna()
        df_15m_atr['atr'] = self.calculate_atr(df_15m_atr)
        
        # Merge ATR to 1m - Use Reindex with FFill
        # df_1m_test is the master index
        df_1m_test['atr'] = df_15m_atr['atr'].reindex(df_1m_test.index, method='ffill')
        
        triggers = df_1m_test[df_1m_test.index >= start_ts]
        triggers = triggers.iloc[:5000] # Limit for fast feedback
        logger.info(f"Scanning {len(triggers)} 1m intervals (Sample)...")
        
        results = []
        
        for ts, row in triggers.iterrows():
            atr = row['atr']
            if pd.isna(atr): continue
            
            entry_time = ts 
            
            # Context: 20m window BEFORE entry
            context_end = entry_time
            # DataFrame slice by index is inclusive/exclusive depending on usage.
            # We want strictly [end-20m, end)
            context_start = context_end - pd.Timedelta(minutes=20)
            
            # Optimization: fast slice
            # Assuming dataframe is sorted
            # To avoid slow .loc every time, we could use rolling window iterator if we were strict.
            # But with GPU the bottleneck might be slice creation.
            # df_1m has full data.
            
            context_df = df_1m.loc[context_start:entry_time] # Includes entry_time usually?
            # Adjust to be strict < entry_time
            if len(context_df) > 0 and context_df.index[-1] == entry_time:
                 context_df = context_df.iloc[:-1]
                 
            # Predict
            # Returns (LongProb, ShortProb)
            # Batched inference would be better but keeping simple for now. 
            # 70k inferences might take 1-2 mins.
            p_long, p_short = self.get_prediction(context_df) 
            
            entry_price = row['open'] # Enter at Open of NEXT candle? 
            # Row `ts` is the candle 'Time' (Start of candle). 
            # We have closed candle `ts`. 
            # We make decision. We enter at Open of `ts + 1m`.
            # OR we assume we trade at CLOSE of `ts`.
            # Standard backtest: decision at Close, Entry at Next Open.
            # `triggers` iterates `df_1m_test`. `row` is the candle `ts`.
            # We assume this candle just closed? 
            # Actually iterrows gives the row. If we read parquet, time is usually Open Time.
            # So `row` is the candle starting at `ts`.
            # We can't know the future of this candle.
            # We must use context UP TO `ts`?
            # "train on 20 1m candles BEFORE...".
            # If we are at 10:00 (Open), previous bars are 09:59, etc.
            # So context is 09:40 to 09:59.
            # Decision made at 10:00 Open.
            # Trade entry at 10:00 Open.
            
            # My logic in `get_prediction`: uses `df_window`.
            # Context here: `context_start` to `entry_time`.
            # If entry_time is 10:00. context is 09:40-10:00.
            # `context_df[index < entry_time]` -> 09:40..09:59. Correct.
            
            # Sizing
            risk_dist = 1.35 * atr
            
            # Simulate LONG
            sl_long = entry_price - risk_dist
            tp_long = entry_price + (1.4 * risk_dist)
            res_long = self.simulate_trade(df_1m, entry_time, entry_price, sl_long, tp_long, 'LONG')
            
            # Simulate SHORT
            sl_short = entry_price + risk_dist
            tp_short = entry_price - (1.4 * risk_dist)
            res_short = self.simulate_trade(df_1m, entry_time, entry_price, sl_short, tp_short, 'SHORT')
            
            results.append({
                'time': entry_time,
                'p_long': p_long,
                'p_short': p_short,
                'pnl_long': res_long['pnl'],
                'pnl_short': res_short['pnl'],
                'outcome_long': res_long['outcome'],
                'outcome_short': res_short['outcome']
            })
            
            if len(results) % 2000 == 0:
                logger.info(f"Scanned {len(results)} intervals...")
                
        # --- ANALYSIS ---
        df = pd.DataFrame(results)
        logger.info(f"Simulation Complete. {len(df)} Intervals.")
        
        # Define Strategies
        
        # 1. Follow Strongest Signal (If > Threshold)
        # 2. Fade Strongest Signal (If < Low Threshold) - "Reversing the Loss"
        
        hybrid_pnl = 0.0
        hybrid_trades = 0
        
        follow_pnl = 0.0
        follow_trades = 0
        
        fade_losers_pnl = 0.0
        fade_losers_trades = 0
        
        fade_winners_pnl = 0.0
        fade_winners_trades = 0
        
        for _, row in df.iterrows():
            pl = row['p_long']
            ps = row['p_short']
            
            # Decision Logic
            action = None
            direction = None
            
            # Prioritize High Confidence
            # If Long > High -> Long (Follow)
            # If Short > High -> Short (Follow)
            # If Long < Low -> Short (Fade)
            # If Short < Low -> Long (Fade)
            
            # Conflict resolution? 
            # If Long > 0.6 AND Short < 0.4? -> Both say Long is good/Short is bad. Strong Long.
            
            score_long = pl - ps # High = Long good. Low = Short good.
            # Actually, pl is Prob(Long Win). ps is Prob(Short Win).
            # If pl is high, Long is good.
            # If ps is high, Short is good.
            # They shouldn't both be high (market can't win both ways usually, but can chop).
            
            # Let's simple check:
            # Max Prob drives direction
            
            best_prob = max(pl, ps)
            is_long_best = pl >= ps
            
            # FOLLOW Logic
            if best_prob > self.threshold: # e.g. 0.55
                # We would Follow
                pnl = row['pnl_long'] if is_long_best else row['pnl_short']
                follow_pnl += pnl
                follow_trades += 1
                
                # FADE WINNERS Logic (Inverse of Follow)
                fade_winners_pnl += (-1 * pnl)
                fade_winners_trades += 1
                
                # Hybrid: Follow Winners
                hybrid_pnl += pnl
                hybrid_trades += 1
                
            # FADE LOSERS Logic (Reversing the Signal)
            # If predictions are LOW?
            # worst_prob = min(pl, ps)
            # But "Signal" usually implies something the model thought was a setup.
            # If we continuous scan, everything is a signal?
            # User said: "look for the loss triggers learned... place an opposite trade"
            # This implies if Model predicts LOW Win Prob (High Loss Prob), we take opposite.
            
            # If pl < 0.40 -> Fade Long (Go Short)
            if pl < self.thresh_low:
                # Signal is "Long is Bad". Action: Go Short.
                profit = row['pnl_short']
                fade_losers_pnl += profit
                fade_losers_trades += 1
                
                # Hybrid: Fade Losers
                hybrid_pnl += profit
                hybrid_trades += 1
                
            # If ps < 0.40 -> Fade Short (Go Long)
            if ps < self.thresh_low:
                 profit = row['pnl_long']
                 fade_losers_pnl += profit
                 fade_losers_trades += 1
                 
                 hybrid_pnl += profit
                 hybrid_trades += 1

        logger.info("-" * 60)
        logger.info(f"REPORT | Balance: ${initial_balance:,.0f} | Size: {position_size}")
        logger.info("-" * 60)
        logger.info(f"1. Follow Only (Conf > {self.threshold}):")
        logger.info(f"   Trades: {follow_trades} | PnL: {follow_pnl:.2f}")
        logger.info("-" * 60)
        logger.info(f"2. Fade Losers (Conf < {self.thresh_low}):")
        logger.info(f"   Trades: {fade_losers_trades} | PnL: {fade_losers_pnl:.2f}")
        logger.info("-" * 60)
        logger.info(f"3. Hybrid (Follow Winners + Fade Losers):")
        logger.info(f"   Trades: {hybrid_trades} | PnL: {hybrid_pnl:.2f} | Final: ${initial_balance + hybrid_pnl:,.2f}")
        logger.info("-" * 60)
        logger.info(f"4. Fade Winners (Contrarian):")
        logger.info(f"   Trades: {fade_winners_trades} | PnL: {fade_winners_pnl:.2f}")
        logger.info("-" * 60)
        
        df.to_parquet(PROCESSED_DIR / "smart_reverse_trades.parquet")
        logger.info("Saved detailed results to smart_reverse_trades.parquet")
        
    def simulate_trade(self, df, entry_time, entry_price, sl, tp, direction):
        subset = df.loc[entry_time:].iloc[:2000]
        if subset.empty: return {'outcome': 'TIMEOUT', 'pnl': 0.0}
        
        times = subset.index.values
        highs = subset['high'].values
        lows = subset['low'].values
        closes = subset['close'].values
        
        if direction == 'LONG':
             mask_win = highs >= tp
             mask_loss = lows <= sl
        else:
             mask_win = lows <= tp
             mask_loss = highs >= sl
             
        idx_win = np.argmax(mask_win) if mask_win.any() else 999999
        idx_loss = np.argmax(mask_loss) if mask_loss.any() else 999999
        
        if idx_win == 999999 and idx_loss == 999999:
            outcome = 'TIMEOUT'
            pnl = (closes[-1] - entry_price) * (1 if direction == 'LONG' else -1)
        elif idx_win < idx_loss:
            outcome = 'WIN'
            pnl = (tp - entry_price) * (1 if direction == 'LONG' else -1)
        else:
            outcome = 'LOSS'
            pnl = (sl - entry_price) * (1 if direction == 'LONG' else -1)
            
        return {'outcome': outcome, 'pnl': pnl}

if __name__ == "__main__":
    # Model output is centered ~0.41. 
    # Follow > 0.415? Fade < 0.405?
    # Let's try to capture top/bottom 20%.
    s = SmartCNNStrategy(threshold=0.43) 
    s.thresh_low = 0.39 
    s.run_simulation()
