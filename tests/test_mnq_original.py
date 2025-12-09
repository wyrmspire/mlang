
import unittest
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MNQOriginalTest")

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

class MNQOriginalStrategy:
    def __init__(self, risk_amount=300.0, df_full=None):
        self.risk_amount = risk_amount
        self.df_full = df_full
        
        self.bars_5m = []
        self.current_5m_candle = None
        self.triggers = []
        self.active_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.min_atr = 5.0 
        
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
        
        # 3. Triggers - ORIGINAL LOGIC
        # Logic: If price hits target extension and returns to open -> ENTER IN REJECTION DIRECTION.
        # i.e., Price went UP -> We go SHORT. Price went DOWN -> We go LONG.
        
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
            
            # SHORT SETUP (Rejection of High)
            if cand['max_high'] >= cand['short_tgt'] and low_p <= cand['open']:
                # The "Rejection" implies price went up and came back down.
                # ORIGINAL Trade: Sell the rejection.
                triggered = True
                direction = "SHORT"
                stop = cand['max_high'] # Stop at the wick high
                risk_dist = stop - cand['open']
                sl = stop
                tp = cand['open'] - (1.4 * risk_dist) # Target Down
            
            # LONG SETUP (Rejection of Low)
            elif cand['min_low'] <= cand['long_tgt'] and high_p >= cand['open']:
                # The "Rejection" implies price went down and came back up.
                # ORIGINAL Trade: Buy the rejection.
                triggered = True
                direction = "LONG"
                stop = cand['min_low'] # Stop at wick low
                risk_dist = cand['open'] - stop
                sl = stop
                tp = cand['open'] + (1.4 * risk_dist) # Target Up

            if triggered and len(self.active_trades) < 1:
                self.trade_counter += 1
                t = Trade(id=self.trade_counter, direction=direction, entry_time=timestamp, 
                          entry_price=close_p, stop_loss=sl, take_profit=tp, risk=self.risk_amount)
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


class TestMNQOriginal(unittest.TestCase):
    def test_mnq_original_logic(self):
        ticker = "MNQ=F"
        # YFinance restriction: 1m data only available for last 7 days.
        logger.info(f"Downloading {ticker} (Last 5 Days)...")
        
        df = yf.download(ticker, period="5d", interval="1m", progress=False)
        if df.empty:
            logger.error("No data returned for MNQ.")
            return

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        
        logger.info(f"Loaded {len(df)} candles for MNQ.")
        
        strategy = MNQOriginalStrategy(risk_amount=300.0, df_full=df)
        
        logger.info("Running ORIGINAL Rejection Strategy on MNQ...")
        for ts, row in df.iterrows():
            strategy.on_bar(ts, row['open'], row['high'], row['low'], row['close'])
            
        n_trades = len(strategy.closed_trades)
        logger.info(f"Total Trades: {n_trades}")
        
        if n_trades > 0:
            wins = len([t for t in strategy.closed_trades if t.status=='WIN'])
            wr = wins / n_trades * 100
            pnl = sum([t.pnl for t in strategy.closed_trades])
            
            logger.info("=" * 40)
            logger.info(f"MNQ ORIGINAL ({ticker}) PERFORMANCE REPORT")
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
            plt.plot(res_df['t'], res_df['cum'], marker='o', color='orange')
            plt.title(f"MNQ Original Strategy Performance | PnL: ${pnl:.0f}")
            plt.tight_layout()
            plt.savefig("mnq_original_performance.png")
            logger.info("Saved mnq_original_performance.png")
            
        else:
            logger.warning("No trades triggered. Check ATR thresholds or Volatility.")

if __name__ == '__main__':
    unittest.main()
