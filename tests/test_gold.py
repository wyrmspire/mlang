
import unittest
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import MODELS_DIR
from src.models.variants import CNN_Classic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GoldTest")

@dataclass
class Trade:
    id: int
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    status: str = "OPEN"
    pnl: float = 0.0
    exit_time: Optional[pd.Timestamp] = None

class GoldStrategy:
    def __init__(self, model, lookback=20, risk_amount=300.0, df_full=None):
        self.model = model
        self.lookback = lookback
        self.risk_amount = risk_amount
        self.df_full = df_full
        
        self.bars_5m = []
        self.current_5m_candle = None
        self.triggers = []
        self.active_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.min_atr = 2.0 # Adjusted for Gold (might need tuning, but 2.0 is safe for 5m Gold)
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
        try:
            current_idx = self.df_full.index.get_loc(timestamp)
        except:
            return 0.5 
            
        start_idx = current_idx - self.lookback
        if start_idx < 0: return 0.5
        
        subset = self.df_full.iloc[start_idx:current_idx]
        vals = subset[['open','high','low','close']].values
        mean = vals.mean()
        std = vals.std()
        if std==0: std=1
        vals=(vals-mean)/std
        inp = torch.FloatTensor(vals).to(self.device).unsqueeze(0) 

        with torch.no_grad():
            prob = self.model(inp).item()
        return prob

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        timestamp = pd.Timestamp(timestamp)
        
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
        for t in self.active_trades:
            if t.status != "OPEN": self.closed_trades.append(t)
        self.active_trades = active
        
        # 2. Aggregation
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
                stop = cand['max_high']
                prob = self.predict(timestamp)
                if prob > 0.5:
                    triggered = True
                    direction = "LONG"
                    risk_dist = stop - cand['open']
                    tp = cand['max_high'] 
                    sl = cand['open'] - (1.4 * risk_dist) 
            elif cand['min_low'] <= cand['long_tgt'] and high_p >= cand['open']:
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
                t = Trade(id=self.trade_counter, direction=direction, entry_time=timestamp, 
                          entry_price=cand['open'], stop_loss=sl, take_profit=tp, risk=self.risk_amount)
                self.active_trades.append(t)
                triggers_to_remove.append(cand)
        
        for t in triggers_to_remove: 
            if t in self.triggers: self.triggers.remove(t)
            
        if is_close:
            atr = self.calculate_atr()
            c = self.current_5m_candle
            self.bars_5m.append((c['time'],c['open'],c['high'],c['low'],c['close'],atr))
            # Gold ATR likely > 0.5, but min_atr set to 2.0. Let's adjust dynamically or log
            if not np.isnan(atr) and atr >= self.min_atr:
                self.triggers.append({'start_time':c['time'], 'open':c['open'], 'atr':atr, 
                                      'short_tgt':c['open']+1.5*atr, 'long_tgt':c['open']-1.5*atr, 
                                      'max_high':c['high'], 'min_low':c['low']})
            self.current_5m_candle = None


class TestGold(unittest.TestCase):
    def test_gold_logic(self):
        ticker = "MGC=F"
        # YFinance restriction: 1m data only available for last 7 days.
        logger.info(f"Downloading {ticker} (Last 5 Days)...")
        
        df = yf.download(ticker, period="5d", interval="1m", progress=False)
        if df.empty:
            logger.error("No data returned for Gold.")
            return

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        
        logger.info(f"Loaded {len(df)} candles for Gold.")
        
        # Load Model (CNN Classic - Trained on MES, Applying to Gold)
        # NOTE: WE ARE TRANSFERRING A MODEL TRAINED ON MES TO GOLD.
        # This is a bold test of generalization.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN_Classic()
        path = MODELS_DIR / "CNN_Classic.pth"
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            logger.info("Loaded CNN_Classic model (MES-trained) for Gold test.")
        else:
            logger.warning("Model weights not found! Using untrained model (random).")
            
        strategy = GoldStrategy(model, df_full=df)
        
        logger.info("Running strategy specific to Gold...")
        for ts, row in df.iterrows():
            strategy.on_bar(ts, row['open'], row['high'], row['low'], row['close'])
            
        n_trades = len(strategy.closed_trades)
        logger.info(f"Total Trades: {n_trades}")
        
        if n_trades > 0:
            wins = len([t for t in strategy.closed_trades if t.status=='WIN'])
            wr = wins / n_trades * 100
            pnl = sum([t.pnl for t in strategy.closed_trades])
            
            logger.info("=" * 40)
            logger.info(f"GOLD ({ticker}) PERFORMANCE REPORT")
            logger.info(f"Period:    Last 5 Days")
            logger.info(f"Trades:    {n_trades}")
            logger.info(f"Win Rate:  {wr:.1f}%")
            logger.info(f"Total PnL: ${pnl:.2f}")
            logger.info("=" * 40)
            
            # Save basic plot
            res_df = pd.DataFrame([{'t': t.exit_time, 'pnl': t.pnl} for t in strategy.closed_trades])
            res_df = res_df.sort_values('t')
            res_df['cum'] = res_df['pnl'].cumsum()
            
            plt.figure(figsize=(10,5))
            plt.plot(res_df['t'], res_df['cum'], marker='o')
            plt.title(f"Gold Strategy Performance | PnL: ${pnl:.0f}")
            plt.tight_layout()
            plt.savefig("gold_performance.png")
            logger.info("Saved gold_performance.png")
            
        else:
            logger.warning("No trades triggered. Check ATR thresholds or Volatility.")

if __name__ == '__main__':
    unittest.main()
