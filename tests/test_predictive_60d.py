
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
from src.train_predictive_5m import CNN_Predictive

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictiveTest60d")

@dataclass
class LimitOrder:
    id: int
    direction: str # "SELL" or "BUY"
    limit_price: float
    stop_loss: float
    take_profit: float
    created_time: pd.Timestamp
    expiry_time: pd.Timestamp

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

class PredictiveStrategy60d:
    def __init__(self, model, risk_amount=300.0, df_full=None):
        self.model = model
        self.risk_amount = risk_amount
        self.df_full = df_full
        
        # We need ATR logic. 
        # Calculate ATR on 15m re-sampled version of full_df
        self.df_15m = df_full.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
        high = self.df_15m['high']
        low = self.df_15m['low']
        close = self.df_15m['close']
        tr_list = []
        for i in range(len(self.df_15m)):
             if i==0: tr_list.append(high.iloc[i]-low.iloc[i])
             else:
                tr = max(high.iloc[i]-low.iloc[i], abs(high.iloc[i]-close.iloc[i-1]), abs(low.iloc[i]-close.iloc[i-1]))
                tr_list.append(tr)
        self.atr_series = pd.Series(tr_list, index=self.df_15m.index).rolling(14).mean().shift(1)
        # Shift 1 means 10:15 row gets 10:00 ATR. Accessing it at 10:15 (start of bar?)
        # Wait. In test:
        # 10:05 5m.
        # df_full.at[10:05, 'atr']
        # ffill from 10:00 label.
        # If 10:00 label in 15m is "10:00 start", then shift(1) means it comes from 09:45.
        # Is that too conservative? 
        # 10:00 15m candle ends 10:15.
        # At 10:05, the 10:00 ATR is NOT known (candle open).
        # We only know 09:45 ATR.
        # So YES, we must rely on 09:45 ATR for 10:00, 10:05, 10:10.
        # If `label='left'`, 15m labels are `09:45`, `10:00`.
        # At 10:05 we ffill from `10:00`. The row `10:00` must contain `09:45` data.
        # So `shift(1)` is correct. input[10:00] = ATR[09:45].
        
        # Merge ATR back to 5m DF for easy lookup?
        # ffill to propagate last known 15m ATR to 5m bars
        self.df_full['atr_15m_lookup'] = self.atr_series.reindex(df_full.index, method='ffill')
        
        self.active_limits = []
        self.active_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lookback = 20
        self.threshold = 0.15 # Adjusted for imbalance (base rate ~0.07) 
        
    def predict(self, timestamp):
        try:
            current_idx = self.df_full.index.get_loc(timestamp)
        except:
            return 0.5 
            
        start_idx = current_idx - self.lookback
        if start_idx < 0: return 0.5
        
        subset = self.df_full.iloc[start_idx:current_idx] 
        # Features: o,h,l,c, atr_15m
        
        # We need the 15m ATR associated with each past bar for the Feature Block?
        # In training we input (o,h,l,c, atr). 
        # Yes, we added 'atr_15m_lookup' to df_full.
        
        block = subset[['open','high','low','close','atr_15m_lookup']].values
        
        # Normalize
        mean = block.mean(axis=0)
        std = block.std(axis=0)
        std[std==0] = 1e-6
        block = (block - mean) / std
        
        inp = torch.FloatTensor(block).to(self.device).unsqueeze(0) 

        with torch.no_grad():
            prob = self.model(inp).item()
        return prob

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        timestamp = pd.Timestamp(timestamp)
        
        # 1. Check Limits (First Check)
        limits_to_remove = []
        for order in self.active_limits:
            if timestamp > order.expiry_time:
                limits_to_remove.append(order)
                continue
            
            filled = False
            fill_px = 0.0
            
            if order.direction == "SELL":
                # Check Fill
                if high_p >= order.limit_price:
                    # Fill
                    fill_px = max(open_p, order.limit_price)
                    filled = True
            elif order.direction == "BUY":
                if low_p <= order.limit_price:
                    fill_px = min(open_p, order.limit_price)
                    filled = True
            
            if filled:
                self.trade_counter += 1
                dist = abs(fill_px - order.stop_loss)
                size = self.risk_amount/dist if dist>0 else 0
                
                t = Trade(id=self.trade_counter, direction=("SHORT" if order.direction=="SELL" else "LONG"),
                          entry_time=timestamp, entry_price=fill_px, stop_loss=order.stop_loss, 
                          take_profit=order.take_profit, risk=self.risk_amount)
                self.active_trades.append(t)
                limits_to_remove.append(order)
        
        for l in limits_to_remove:
            if l in self.active_limits: self.active_limits.remove(l)

        # 2. Update Trades
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
        
        # 3. New Orders (Predictive)
        # Use Lookahead ATR? No, use current known ATR.
        try:
            atr = self.df_full.at[timestamp, 'atr_15m_lookup']
        except: 
            atr = np.nan
            
        if np.isnan(atr) or atr <= 0: return # No ATR context yet
        
        prob = self.predict(timestamp)
        if self.trade_counter == 0 and np.random.rand() < 0.01:
             logger.info(f"Sample Prob: {prob:.4f} | ATR: {atr:.2f}")

        if prob > self.threshold:
            # Place OCO Limits
            # Valid for 15 minutes (3 x 5m bars)
            expiry = timestamp + pd.Timedelta(minutes=15)
            
            limit_dist = 1.5 * atr
            stop_dist = 0.5 * atr # Tighter stop? Or 1.0? 
            # In Phase 7 we used 1.5 Limit distance.
            # Stop was 1.0 ATR beyond.
            
            # Sell Limit
            sell_limit = close_p + limit_dist
            sell_tp = close_p 
            sell_sl = sell_limit + (1.0 * atr)
            
            self.active_limits.append(LimitOrder(id=len(self.active_limits), direction="SELL",
                                                 limit_price=sell_limit, stop_loss=sell_sl, take_profit=sell_tp,
                                                 created_time=timestamp, expiry_time=expiry))
                                                 
            # Buy Limit
            buy_limit = close_p - limit_dist
            buy_tp = close_p
            buy_sl = buy_limit - (1.0 * atr)
            
            self.active_limits.append(LimitOrder(id=len(self.active_limits), direction="BUY",
                                                 limit_price=buy_limit, stop_loss=buy_sl, take_profit=buy_tp,
                                                 created_time=timestamp, expiry_time=expiry))

class TestPredictive60d(unittest.TestCase):
    def test_mes_60d(self):
        ticker = "MES=F"
        logger.info(f"Downloading {ticker} (Last 60 Days, 5m)...")
        # 59d to be safe
        df = yf.download(ticker, period="59d", interval="5m", progress=False)
        if df.empty: 
            logger.error("No data.")
            return

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        
        # Load Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN_Predictive(input_dim=5).to(device)
        path = MODELS_DIR / "CNN_Predictive_5m.pth"
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
        else:
            logger.error("No model found!")
            return

        strategy = PredictiveStrategy60d(model, df_full=df)
        
        logger.info("Running 60-Day Simulation (5m)...")
        for ts, row in df.iterrows():
            strategy.on_bar(ts, row['open'], row['high'], row['low'], row['close'])
            
        n_trades = len(strategy.closed_trades)
        
        if n_trades > 0:
            wins = len([t for t in strategy.closed_trades if t.status=='WIN'])
            wr = wins / n_trades * 100
            pnl = sum([t.pnl for t in strategy.closed_trades])
            
            logger.info("=" * 40)
            logger.info(f"MES 60-DAY RESULT (5m/15m)")
            logger.info(f"Trades:    {n_trades}")
            logger.info(f"Win Rate:  {wr:.1f}%")
            logger.info(f"Total PnL: ${pnl:.2f}")
            logger.info("=" * 40)
            
            # Plot
            res_df = pd.DataFrame([{'t': t.exit_time, 'pnl': t.pnl} for t in strategy.closed_trades])
            res_df = res_df.sort_values('t')
            res_df['cum'] = res_df['pnl'].cumsum()
            
            plt.figure(figsize=(10,5))
            plt.plot(res_df['t'], res_df['cum'], marker='o', color='purple')
            plt.title(f"MES 60-Day Performance (Predictive 15m) | PnL: ${pnl:.0f}", fontsize=12)
            plt.xlabel("Date")
            plt.ylabel("PnL ($)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("mes_60d_performance.png")
            logger.info("Saved mes_60d_performance.png")
        else:
            logger.warning("No trades executed.")

if __name__ == '__main__':
    unittest.main()
