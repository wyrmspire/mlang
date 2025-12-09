
import unittest
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamingInverseTest")

@dataclass
class Trade:
    id: int
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    atr: float
    status: str = "OPEN" # OPEN, WIN, LOSS
    pnl: float = 0.0
    exit_time: Optional[pd.Timestamp] = None

class StreamingInverseStrategy:
    def __init__(self, risk_amount=300.0):
        self.bars_1m = [] # List of (time, o, h, l, c)
        self.bars_5m = [] # List of (time, o, h, l, c, atr)
        
        self.risk_amount = risk_amount
        
        self.current_5m_candle = None
        self.last_5m_close_time = None
        
        # Generator State
        self.triggers = [] 
        
        self.active_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        
        self.min_atr = 5.0
        self.lookahead = 12 
        
    def calculate_atr(self, period=14):
        if len(self.bars_5m) < period + 1:
            return np.nan
            
        # We need the last N candles
        df = pd.DataFrame(self.bars_5m, columns=['time', 'open', 'high', 'low', 'close', 'atr'])
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # TR calculation
        # TR = max(H-L, |H-Cp|, |L-Cp|)
        tr_list = []
        for i in range(len(df)):
            if i == 0:
                tr_list.append(high[i] - low[i])
                continue
                
            h = high[i]
            l = low[i]
            cp = close[i-1]
            
            tr = max(h - l, abs(h - cp), abs(l - cp))
            tr_list.append(tr)
            
        tr_series = pd.Series(tr_list)
        atr = tr_series.rolling(window=period).mean().iloc[-1]
        return atr

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        """
        Process a single 1m bar
        """
        timestamp = pd.Timestamp(timestamp)
        
        # 1. Update Active Trades (1m Check)
        for trade in self.active_trades:
            if trade.status != "OPEN": continue
            
            # Position Size Calculation:
            # Risk Amount = $300
            # Risk Distance = |Entry - SL|
            # Size = Risk Amount / Risk Distance
            dist = abs(trade.entry_price - trade.stop_loss)
            if dist == 0: size = 0
            else: size = self.risk_amount / dist
            
            # Check Exit
            if trade.direction == "LONG":
                # Hit TP (Extreme)?
                if high_p >= trade.take_profit:
                    trade.status = "WIN"
                    # Reward Distance = TP - Entry
                    reward_dist = trade.take_profit - trade.entry_price
                    trade.pnl = size * reward_dist
                    trade.exit_time = timestamp
                    logger.info(f"✅ Trade {trade.id} WIN (Hit TP)")
                # Hit SL (Rejection Win)?
                elif low_p <= trade.stop_loss:
                    trade.status = "LOSS"
                    # Risk Distance is 'dist'
                    trade.pnl = -size * dist
                    trade.exit_time = timestamp
                    logger.info(f"❌ Trade {trade.id} LOSS (Hit SL)")
            else: # SHORT
                if low_p <= trade.take_profit:
                    trade.status = "WIN"
                    reward_dist = trade.entry_price - trade.take_profit
                    trade.pnl = size * reward_dist
                    trade.exit_time = timestamp
                    logger.info(f"✅ Trade {trade.id} WIN (Hit TP)")
                elif high_p >= trade.stop_loss:
                    trade.status = "LOSS"
                    trade.pnl = -size * dist
                    trade.exit_time = timestamp
                    logger.info(f"❌ Trade {trade.id} LOSS (Hit SL)")

        # Cleanup closed trades
        active = []
        for t in self.active_trades:
            if t.status != "OPEN":
                self.closed_trades.append(t)
            else:
                active.append(t)
        self.active_trades = active

        # 2. Aggregation to 5m
        if self.current_5m_candle is None:
            # Start new candle
            # Align time to 5m floor
            floor_minute = (timestamp.minute // 5) * 5
            candle_start_time = timestamp.replace(minute=floor_minute, second=0, microsecond=0)
            
            self.current_5m_candle = {
                'time': candle_start_time,
                'open': open_p,
                'high': high_p,
                'low': low_p,
                'close': close_p
            }
        else:
            # Update current candle
            c = self.current_5m_candle
            c['high'] = max(c['high'], high_p)
            c['low'] = min(c['low'], low_p)
            c['close'] = close_p
            
        # Check if 5m candle is complete
        is_candle_close = (timestamp.minute % 5 == 4)
        
        # Update Triggers based on 1m price movement
        
        # Update candidates
        candidates_to_remove = []
        for cand in self.triggers:
            # Check Timeout (12 x 5m candles = 60 mins)
            # Roughly Check time difference
            if (timestamp - cand['start_time']).total_seconds() > 3600:
                candidates_to_remove.append(cand)
                continue
                
            # Update High/Low seen since start
            cand['max_high'] = max(cand['max_high'], high_p)
            cand['min_low'] = min(cand['min_low'], low_p)
            
            # Check SHORT Setup (Price goes UP to Target, then returns DOWN to Open)
            # 1. Did we hit the Up Target?
            if cand['max_high'] >= cand['short_tgt']:
                # 2. Did we Return to Open?
                if low_p <= cand['open']:
                    
                    stop_loss_price = cand['max_high']
                    risk_dist = stop_loss_price - cand['open']
                    
                    # Inverse Trade
                    # Rejection TP (our SL) is 1.4x Rejection Risk
                    rejection_risk = risk_dist # Risk of rejection trade
                    rejection_tp_dist = 1.4 * rejection_risk
                    
                    # Our Trade (Inverse LONG)
                    # Entry: Open
                    # Stop: Open - rejection_tp_dist
                    # Target: Open + risk_dist (The Extreme)
                    
                    self.entry_trade(
                        direction="LONG",
                        timestamp=timestamp,
                        price=cand['open'],
                        stop_loss=cand['open'] - rejection_tp_dist, 
                        take_profit=cand['max_high'], 
                        risk=rejection_tp_dist, # Our visual risk, used for display?
                        atr=cand['atr']
                    )
                    candidates_to_remove.append(cand) # Triggered, consume it
                    continue
                    
            # Check LONG Setup (Price goes DOWN to Target, then returns UP to Open)
            if cand['min_low'] <= cand['long_tgt']:
                if high_p >= cand['open']:
                    
                    stop_loss_price = cand['min_low']
                    risk_dist = cand['open'] - stop_loss_price
                    
                    rejection_risk = risk_dist
                    rejection_tp_dist = 1.4 * rejection_risk
                    
                    # Our Trade (Inverse SHORT)
                    # Entry: Open
                    # Stop: Open + rejection_tp_dist
                    # Target: Open - risk_dist (The Extreme)
                    
                    self.entry_trade(
                        direction="SHORT",
                        timestamp=timestamp,
                        price=cand['open'],
                        stop_loss=cand['open'] + rejection_tp_dist, 
                        take_profit=cand['min_low'], 
                        risk=rejection_tp_dist,
                        atr=cand['atr']
                    )
                    candidates_to_remove.append(cand)
                    continue
        
        # Remove matched/expired
        for c in candidates_to_remove:
            if c in self.triggers:
                self.triggers.remove(c)
        
        # 3. Handle 5m Close
        if is_candle_close:
            
            current_atr = self.calculate_atr() # Uses existing bars_5m
            
            c = self.current_5m_candle
            c['atr'] = current_atr
            
            self.bars_5m.append((
                c['time'], c['open'], c['high'], c['low'], c['close'], c['atr']
            ))
            
            # Add NEW Candidate (The candle just closed is a "Start Candle" for potential setups)
            # Only if ATR is valid
            if not np.isnan(current_atr) and current_atr >= self.min_atr:
                # Add to triggers list
                self.triggers.append({
                    'start_time': c['time'],
                    'open': c['open'],
                    'atr': current_atr,
                    'short_tgt': c['open'] + (1.5 * current_atr),
                    'long_tgt': c['open'] - (1.5 * current_atr),
                    'max_high': c['high'], # Init with own high
                    'min_low': c['low']   # Init with own low
                })
                
            # Reset current candle
            self.current_5m_candle = None
            
    def entry_trade(self, direction, timestamp, price, stop_loss, take_profit, risk, atr):
        self.trade_counter += 1
        t = Trade(
            id=self.trade_counter,
            direction=direction,
            entry_time=timestamp,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk=risk,
            atr=atr
        )
        self.active_trades.append(t)
        logger.info(f"🚀 ENTER {direction} at {price:.2f} | Time: {timestamp}")
        logger.info(f"    📝 Params: TP={take_profit:.2f} | SL={stop_loss:.2f} | RiskDist={risk:.2f} | ATR={atr:.2f}")

class TestInverseStrategy(unittest.TestCase):
    def test_streaming_logic(self):
        # 1. Fetch Data
        ticker = "MES=F" # Micro E-mini S&P 500
        logger.info(f"Downloading {ticker} data 1m...")
        df = yf.download(ticker, interval="1m", period="5d", progress=False)
        
        if df.empty:
            logger.warning("No data returned from yfinance. Test cannot proceed.")
            return

        # Flatten columns if MultiIndex (yfinance updated recently)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        # Rename columns to lower
        df.columns = [c.lower() for c in df.columns]
        
        logger.info(f"Loaded {len(df)} 1m candles.")
        
        # 2. Init Strategy
        strategy = StreamingInverseStrategy()
        
        # 3. Stream Data
        logger.info("Streaming data...")
        for i, row in df.iterrows():
            # If yfinance returns 'Datetime' or 'Date'
            ts = row.get('datetime', row.get('date'))
            strategy.on_bar(
                timestamp=ts,
                open_p=float(row['open']),
                high_p=float(row['high']),
                low_p=float(row['low']),
                close_p=float(row['close'])
            )
            
        # 4. Results
        logger.info(f"Total Trades Taken: {len(strategy.closed_trades) + len(strategy.active_trades)}")
        
        if len(strategy.closed_trades) == 0:
            logger.warning("No trades triggered in the last 5 days. Market might be low volatility or logic mismatch.")
        else:
            logger.info("TRADES DETECTED!")
            
            # Create DataFrame for Analysis
            trades_data = []
            for t in strategy.closed_trades:
                trades_data.append({
                    'id': t.id,
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'pnl': t.pnl,
                    'outcome': t.status
                })
            
            res_df = pd.DataFrame(trades_data)
            res_df = res_df.sort_values('exit_time')
            res_df['cum_pnl'] = res_df['pnl'].cumsum()
            res_df['drawdown'] = res_df['cum_pnl'] - res_df['cum_pnl'].cummax()
            
            total_trades = len(res_df)
            wins = len(res_df[res_df['outcome'] == 'WIN'])
            win_rate = (wins / total_trades) * 100
            total_pnl = res_df['pnl'].sum()
            max_drawdown = res_df['drawdown'].min()

            # Detailed Stats
            # Ensure exit_time and entry_time are datetime
            res_df['exit_time'] = pd.to_datetime(res_df['exit_time'])
            res_df['entry_time'] = pd.to_datetime(res_df['entry_time'])
            res_df['duration'] = res_df['exit_time'] - pd.to_datetime(res_df['entry_time'])
            res_df['duration_mins'] = res_df['duration'].dt.total_seconds() / 60.0
            res_df['hour'] = pd.to_datetime(res_df['entry_time']).dt.hour
            
            avg_duration = res_df['duration_mins'].mean()
            avg_win_duration = res_df[res_df['outcome'] == 'WIN']['duration_mins'].mean()
            avg_loss_duration = res_df[res_df['outcome'] == 'LOSS']['duration_mins'].mean()
            
            # Hourly Stats
            hourly_stats = res_df.groupby('hour').apply(
                lambda x: pd.Series({
                    'count': len(x),
                    'win_rate': (len(x[x['outcome'] == 'WIN']) / len(x) * 100) if len(x) > 0 else 0,
                    'pnl': x['pnl'].sum()
                })
            )
            
            # Console Report
            logger.info("=" * 40)
            logger.info(f"STRATEGY PERFORMANCE REPORT (5 Days)")
            logger.info(f"Total Trades:      {total_trades}")
            logger.info(f"Win Rate:          {win_rate:.2f}%")
            logger.info(f"Total PnL:         ${total_pnl:.2f}")
            logger.info(f"Max Drawdown:      ${max_drawdown:.2f}")
            logger.info("-" * 40)
            logger.info(f"Avg Duration:      {avg_duration:.1f} mins")
            logger.info(f"Avg Win Duration:  {avg_win_duration:.1f} mins")
            logger.info(f"Avg Loss Duration: {avg_loss_duration:.1f} mins")
            logger.info("-" * 40)
            logger.info("Hourly Performance:")
            for hour, row in hourly_stats.iterrows():
                logger.info(f"  Hour {int(hour):02d}: {int(row['count']):3d} trades | WR: {row['win_rate']:5.1f}% | PnL: ${row['pnl']:.0f}")
            logger.info("=" * 40)
            
            # Plotting
            plt.figure(figsize=(12, 8))
            
            # Subplot 1: Equity Curve
            plt.subplot(2, 1, 1)
            plt.plot(res_df['exit_time'], res_df['cum_pnl'], label='Cumulative PnL', color='green')
            plt.fill_between(res_df['exit_time'], res_df['cum_pnl'], 0, alpha=0.1, color='green')
            plt.title(f"Inverse Strategy (5 Days) | Total PnL: ${total_pnl:.0f} | Win Rate: {win_rate:.1f}%")
            plt.ylabel("PnL ($)")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            # Subplot 2: Hourly Performance
            plt.subplot(2, 1, 2)
            colors = ['g' if p > 0 else 'r' for p in hourly_stats['pnl']]
            plt.bar(hourly_stats.index, hourly_stats['pnl'], color=colors, alpha=0.7)
            plt.title("PnL by Hour of Day")
            plt.xlabel("Hour (UTC)")
            plt.ylabel("PnL ($)")
            plt.xticks(hourly_stats.index)
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            out_file = "inverse_strategy_performance.png"
            plt.savefig(out_file)
            logger.info(f"Performance plot saved to {out_file}")
            plt.close()

if __name__ == '__main__':
    unittest.main()
