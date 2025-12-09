
import unittest
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import MODELS_DIR
from src.models.variants import CNN_Classic, CNN_Wide, LSTM_Seq, Feature_MLP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase2Test")

# Re-use metrics logic
@dataclass
class Trade:
    id: int
    model: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    status: str = "OPEN"
    pnl: float = 0.0
    exit_time: Optional[pd.Timestamp] = None

# Feature Helper
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(df):
    df['rsi'] = compute_rsi(df['close'])
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['dist_ma20'] = (df['close'] - df['ma20']) / df['ma20']
    df['dist_ma50'] = (df['close'] - df['ma50']) / df['ma50']
    df['atr'] = (df['high'] - df['low']).rolling(14).mean() / df['close']
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    df = df.fillna(0.0)
    feats = ['rsi', 'dist_ma20', 'dist_ma50', 'atr', 'ret_1', 'ret_5', 'volume']
    for f in feats:
        m = df[f].mean()
        s = df[f].std()
        if s == 0: s=1
        df[f] = (df[f]-m)/s
    return df, feats

class ModelStrategy:
    def __init__(self, name, model, mode='classic', lookback=20, risk_amount=300.0, df_full=None):
        self.name = name
        self.model = model
        self.mode = mode
        self.lookback = lookback
        self.risk_amount = risk_amount
        self.df_full = df_full # Needed for MLP lookup
        
        self.bars_5m = []
        self.current_5m_candle = None
        self.triggers = []
        self.active_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.min_atr = 5.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def calculate_atr(self):
        if len(self.bars_5m) < 15: return np.nan
        df = pd.DataFrame(self.bars_5m, columns=['time','open','high','low','close','atr'])
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr_list = []
        for i in range(len(df)):
            if i==0: tr_list.append(high[i]-low[i])
            else:
               tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
               tr_list.append(tr)
        return pd.Series(tr_list).rolling(14).mean().iloc[-1]

    def predict(self, timestamp):
        # Prepare Input
        # Find index in full DF (1m)
        try:
            current_idx = self.df_full.index.get_loc(timestamp)
        except:
            return 0.5 # Default neutral
            
        start_idx = current_idx - self.lookback
        if start_idx < 0: return 0.5
        
        subset = self.df_full.iloc[start_idx:current_idx]
        
        if self.mode == 'mlp':
            feats = subset[['rsi', 'dist_ma20', 'dist_ma50', 'atr', 'ret_1', 'ret_5', 'volume']].iloc[-1].values
            inp = torch.FloatTensor(feats).to(self.device).unsqueeze(0)
        elif self.mode == 'hybrid': # LSTM
             vals = subset[['open','high','low','close']].values
             mean = vals.mean()
             std = vals.std()
             if std==0: std=1
             vals=(vals-mean)/std
             inp = torch.FloatTensor(vals).to(self.device).unsqueeze(0) # (1, Seq, Dim)
        else: # Classic/Wide CNN
             vals = subset[['open','high','low','close']].values
             mean = vals.mean()
             std = vals.std()
             if std==0: std=1
             vals=(vals-mean)/std
             inp = torch.FloatTensor(vals).to(self.device).unsqueeze(0) # (1, Seq, Dim)

        with torch.no_grad():
            prob = self.model(inp).item()
        return prob

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        timestamp = pd.Timestamp(self.to_ts(timestamp)) # Helper
        
        # 1. Update Trades
        for t in self.active_trades:
            if t.status != "OPEN": continue
            dist = abs(t.entry_price - t.stop_loss)
            size = self.risk_amount/dist if dist>0 else 0
            
            if t.direction == "LONG":
                if high_p >= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.take_profit - t.entry_price)
                    t.exit_time = timestamp
                elif low_p <= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
            else:
                if low_p <= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.entry_price - t.take_profit)
                    t.exit_time = timestamp
                elif high_p >= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
        
        active = [t for t in self.active_trades if t.status == "OPEN"]
        # Closed move to self.closed_trades
        for t in self.active_trades:
            if t.status != "OPEN": self.closed_trades.append(t)
        self.active_trades = active
        
        # 2. Aggregation 5m
        if self.current_5m_candle is None:
            fl = (timestamp.minute // 5) * 5
            st = timestamp.replace(minute=fl, second=0, microsecond=0)
            self.current_5m_candle = {'time':st, 'open':open_p, 'high':high_p, 'low':low_p, 'close':close_p}
        else:
            c = self.current_5m_candle
            c['high'] = max(c['high'], high_p)
            c['low'] = min(c['low'], low_p)
            c['close'] = close_p
            
        is_close = (timestamp.minute % 5 == 4)
        
        # 3. Triggers
        # Standard Rejection Logic
        triggers_to_remove = []
        for cand in self.triggers:
            if (timestamp - cand['start_time']).total_seconds() > 3600:
                triggers_to_remove.append(cand)
                continue
            cand['max_high'] = max(cand['max_high'], high_p)
            cand['min_low'] = min(cand['min_low'], low_p)
            
            triggered = False
            direction = None
            stop = 0.0
            
            if cand['max_high'] >= cand['short_tgt'] and low_p <= cand['open']:
                # Short Rejection Set
                stop = cand['max_high']
                # Check Model for INVERSE (LONG)
                prob = self.predict(timestamp)
                if prob > 0.5: # Model says Inverse Win
                    triggered = True
                    direction = "LONG"
                    # Inverse Params
                    risk_dist = stop - cand['open']
                    tp = cand['max_high'] # Extreme
                    sl = cand['open'] - (1.4 * risk_dist) # Rejection TP
            elif cand['min_low'] <= cand['long_tgt'] and high_p >= cand['open']:
                # Long Rejection Set
                stop = cand['min_low']
                prob = self.predict(timestamp)
                if prob > 0.5:
                    triggered = True
                    direction = "SHORT"
                    risk_dist = cand['open'] - stop
                    tp = cand['min_low']
                    sl = cand['open'] + (1.4 * risk_dist)

            if triggered and len(self.active_trades) < 1:
                self.trade_counter += 1
                t = Trade(id=self.trade_counter, model=self.name, direction=direction, entry_time=timestamp, 
                          entry_price=cand['open'], stop_loss=sl, take_profit=tp, risk=self.risk_amount)
                self.active_trades.append(t)
                triggers_to_remove.append(cand)
        
        for t in triggers_to_remove: 
            if t in self.triggers: self.triggers.remove(t)
            
        if is_close:
            atr = self.calculate_atr()
            c = self.current_5m_candle
            self.bars_5m.append((c['time'],c['open'],c['high'],c['low'],c['close'],atr))
            if not np.isnan(atr) and atr >= self.min_atr:
                self.triggers.append({'start_time':c['time'], 'open':c['open'], 'atr':atr, 
                                      'short_tgt':c['open']+1.5*atr, 'long_tgt':c['open']-1.5*atr, 
                                      'max_high':c['high'], 'min_low':c['low']})
            self.current_5m_candle = None

    def to_ts(self, ts):
        if hasattr(ts, 'to_pydatetime'): return ts
        return pd.Timestamp(ts)

class TestPhase2(unittest.TestCase):
    def test_compare_models(self):
        # 1. Load Data
        ticker = "MES=F"
        start = "2025-12-01"
        end = "2025-12-08"
        print(f"Downloading {ticker} from {start} to {end}...")
        df = yf.download(ticker, start=start, end=end, interval="1m", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        # Rename date/datetime
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        
        # Prepare Features (for MLP)
        df, _ = prepare_features(df)
        
        # 2. Load Models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        models = []
        
        # Baseline (No Model - Always Trade)
        # We simulate this by a "Model" that always returns 1.0
        class AlwaysYes(nn.Module):
            def forward(self, x): return torch.tensor([1.0])
        models.append(ModelStrategy("Baseline (Logic Only)", AlwaysYes(), df_full=df))
        
        # Trained Models
        model_defs = [
            ('CNN_Classic', CNN_Classic(), 'classic', 20),
            ('CNN_Wide', CNN_Wide(), 'classic', 60),
            ('LSTM_Seq', LSTM_Seq(), 'hybrid', 20),
            ('Feature_MLP', Feature_MLP(12), 'mlp', 20) # 12 dims? verify feat len
        ]
        # Recalc feat len
        # feats = ['rsi', 'dist_ma20', 'dist_ma50', 'atr', 'ret_1', 'ret_5', 'volume'] -> 7 dims
        
        for name, net, mode, lb in model_defs:
            path = MODELS_DIR / f"{name}.pth"
            if path.exists():
                # Adjust input dim for MLP if needed
                if name == 'Feature_MLP': net = Feature_MLP(7)
                
                net.load_state_dict(torch.load(path, map_location=device))
                net.to(device)
                net.eval()
                models.append(ModelStrategy(name, net, mode=mode, lookback=lb, df_full=df))
            else:
                print(f"Warning: {path} not found.")

        # 3. Run Comparisons
        results = []
        
        for strat in models:
            print(f"Running {strat.name}...")
            # Reset
            strat.active_trades = []
            strat.closed_trades = []
            strat.triggers = []
            strat.bars_5m = []
            strat.current_5m_candle = None
            
            for ts, row in df.iterrows():
                strat.on_bar(ts, row['open'], row['high'], row['low'], row['close'])
            
            # Collect Stats
            n_trades = len(strat.closed_trades)
            if n_trades > 0:
                wins = len([t for t in strat.closed_trades if t.status=='WIN'])
                wr = wins / n_trades * 100
                pnl = sum([t.pnl for t in strat.closed_trades])
                
                results.append({
                    'Model': strat.name,
                    'Trades': n_trades,
                    'WinRate': wr,
                    'PnL': pnl
                })
            else:
                results.append({'Model': strat.name, 'Trades': 0, 'WinRate': 0, 'PnL': 0})
        
        # 4. Report
        res_df = pd.DataFrame(results).sort_values('PnL', ascending=False)
        print("\n=== PHASE 2 MODEL COMPARISON (Single Position, Fixed Dates) ===")
        print(res_df.to_string(index=False))
        
        # Plot
        if not res_df.empty:
            plt.figure(figsize=(10,6))
            plt.bar(res_df['Model'], res_df['PnL'], color=['green' if x>0 else 'red' for x in res_df['PnL']])
            plt.title("Model PnL Comparison (Out of Sample)")
            plt.ylabel("Total PnL ($)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("phase2_comparison.png")
            print("Saved phase2_comparison.png")

if __name__ == '__main__':
    unittest.main()
