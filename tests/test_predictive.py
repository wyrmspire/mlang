
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
from src.train_predictive import CNN_Predictive

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictiveTest")

@dataclass
class LimitOrder:
    id: int
    direction: str # "SELL" (Short) or "BUY" (Long)
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

class PredictiveStrategy:
    def __init__(self, model, risk_amount=300.0, df_full=None):
        self.model = model
        self.risk_amount = risk_amount
        self.df_full = df_full
        
        self.bars_5m = [] # Used for ATR calc
        self.current_5m_candle = None
        self.active_limits = []
        self.active_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lookback = 20
        self.threshold = 0.6 # tunable
        
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
        # Normalize same as training
        mean = vals.mean()
        std = vals.std()
        if std==0: std=1e-6
        vals=(vals-mean)/std
        inp = torch.FloatTensor(vals).to(self.device).unsqueeze(0) 

        with torch.no_grad():
            prob = self.model(inp).item()
        return prob

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        timestamp = pd.Timestamp(timestamp)
        
        # 1. Check Limits (First Check)
        # Using High/Low of THIS bar to check fills for limits created BEFORE this bar
        # In reality, we shouldn't fill limits created ON this bar (Lookahead). 
        # But we create limits AFTER Processing this bar, so they apply to NEXT bar.
        # So here we check limits from Previous bars.
        
        limits_to_remove = []
        for order in self.active_limits:
            if timestamp > order.expiry_time:
                limits_to_remove.append(order)
                continue
            
            filled = False
            if order.direction == "SELL":
                # Limit Sell filled if Price touches Limit
                if high_p >= order.limit_price:
                    # Fill!
                    # Check Assumption: Did Price gap over? 
                    # If Open > Limit, we fill at Open (Better Price).
                    # If Open < Limit and High > Limit, we fill at Limit.
                    fill_px = max(open_p, order.limit_price)
                    filled = True
                    
            elif order.direction == "BUY":
                if low_p <= order.limit_price:
                    fill_px = min(open_p, order.limit_price)
                    filled = True
            
            if filled:
                # Create Trade
                self.trade_counter += 1
                dist = abs(fill_px - order.stop_loss)
                size = self.risk_amount/dist if dist>0 else 0
                
                t = Trade(id=self.trade_counter, direction=("SHORT" if order.direction=="SELL" else "LONG"),
                          entry_time=timestamp, entry_price=fill_px, stop_loss=order.stop_loss, 
                          take_profit=order.take_profit, risk=self.risk_amount)
                self.active_trades.append(t)
                limits_to_remove.append(order)
                # OCO: If we support OCO we would remove the pair. For now assume independent.
        
        for l in limits_to_remove:
            if l in self.active_limits: self.active_limits.remove(l)

        # 2. Update Trades (Stop/Target Check)
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
        # 5m Aggregation for ATR
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
        
        if is_close:
            atr = self.calculate_atr()
            c = self.current_5m_candle
            self.bars_5m.append((c['time'],c['open'],c['high'],c['low'],c['close'],atr))
            self.current_5m_candle = None
            
            # PREDICTION LOGIC (At 5m Close, or every 1m?)
            # Model trained on 1m windows. Let's predict every 1m? 
            # Ideally align with training. Training used sliding window on 1m.
            # So calculating ATR on 5m but predicting on 1m is mixed. 
            # Let's use 5m ATR for target distance, but Predict every 1m?
            # For simplicity, let's predict every 1m using the last KNOWN 5m ATR.
            
            # Predict
            prob = self.predict(timestamp)
            if prob > self.threshold and not np.isnan(atr) and atr > 0:
                # Signal!
                # Place OCO Limits
                expiry = timestamp + pd.Timedelta(minutes=15)
                
                # Sell Limit
                sell_limit = close_p + (1.5 * atr)
                # SL/TP for Sell
                # Target: Revert to current price (close_p)
                # Stop: Another 1.0 ATR higher
                sell_tp = close_p 
                sell_sl = sell_limit + (1.0 * atr)
                
                self.active_limits.append(LimitOrder(id=len(self.active_limits), direction="SELL",
                                                     limit_price=sell_limit, stop_loss=sell_sl, take_profit=sell_tp,
                                                     created_time=timestamp, expiry_time=expiry))
                                                     
                # Buy Limit
                buy_limit = close_p - (1.5 * atr)
                buy_tp = close_p
                buy_sl = buy_limit - (1.0 * atr)
                
                self.active_limits.append(LimitOrder(id=len(self.active_limits), direction="BUY",
                                                     limit_price=buy_limit, stop_loss=buy_sl, take_profit=buy_tp,
                                                     created_time=timestamp, expiry_time=expiry))

class TestPredictive(unittest.TestCase):
    def test_predictive_mnq(self):
        ticker = "MES=F"
        logger.info(f"Downloading {ticker} (Last 5 Days)...")
        df = yf.download(ticker, period="5d", interval="1m", progress=False)
        if df.empty: return

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        
        # Load Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN_Predictive().to(device)
        path = MODELS_DIR / "CNN_Predictive.pth"
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
        else:
            logger.warning("No model found!")

        strategy = PredictiveStrategy(model, df_full=df)
        
        logger.info("Running Predictive Limit Strategy...")
        for ts, row in df.iterrows():
            strategy.on_bar(ts, row['open'], row['high'], row['low'], row['close'])
            
        n_trades = len(strategy.closed_trades)
        logger.info(f"Total Trades: {n_trades}")
        
        if n_trades > 0:
            wins = len([t for t in strategy.closed_trades if t.status=='WIN'])
            wr = wins / n_trades * 100
            pnl = sum([t.pnl for t in strategy.closed_trades])
            
            logger.info("=" * 40)
            logger.info(f"PREDICTIVE MNQ RESULT")
            logger.info(f"Trades:    {n_trades}")
            logger.info(f"Win Rate:  {wr:.1f}%")
            logger.info(f"Total PnL: ${pnl:.2f}")
            logger.info("=" * 40)
            
            # Plot
            res_df = pd.DataFrame([{'t': t.exit_time, 'pnl': t.pnl} for t in strategy.closed_trades])
            res_df = res_df.sort_values('t')
            res_df['cum'] = res_df['pnl'].cumsum()
            
            plt.figure(figsize=(10,5))
            plt.plot(res_df['t'], res_df['cum'], marker='o', color='blue')
            plt.title(f"Predictive Strategy Performance | PnL: ${pnl:.0f}")
            plt.tight_layout()
            plt.savefig("predictive_performance.png")
            logger.info("Saved predictive_performance.png")

if __name__ == '__main__':
    unittest.main()
