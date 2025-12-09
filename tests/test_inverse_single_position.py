
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
logger = logging.getLogger("SinglePosStrategy")

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
    
    # Deep Analysis Metrics
    max_unrealized_pnl: float = -np.inf # Peak profit during trade (MFE)
    min_unrealized_pnl: float = np.inf  # Max drawdown during trade (MAE)
    max_adverse_excursion: float = 0.0 # Absolute distance moved against
    max_favorable_excursion: float = 0.0 # Absolute distance moved in favor

class SinglePositionInverseStrategy:
    def __init__(self, risk_amount=300.0, max_active_trades=1):
        self.bars_1m = [] 
        self.bars_5m = [] 
        
        self.risk_amount = risk_amount
        self.max_active_trades = max_active_trades
        
        self.current_5m_candle = None
        
        self.triggers = [] 
        self.active_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        
        self.min_atr = 5.0
        self.lookahead = 12 
        
    def calculate_atr(self, period=14):
        if len(self.bars_5m) < period + 1:
            return np.nan
        
        df = pd.DataFrame(self.bars_5m, columns=['time', 'open', 'high', 'low', 'close', 'atr'])
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
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
        timestamp = pd.Timestamp(timestamp)
        
        # 1. Update Active Trades (1m Check)
        for trade in self.active_trades:
            if trade.status != "OPEN": continue
            
            # Position Size Calculation
            dist = abs(trade.entry_price - trade.stop_loss)
            size = self.risk_amount / dist if dist > 0 else 0
            
            # Update MAE/MFE (Intra-candle conservative estimation)
            # Conservatively assume High happened first or Low? We don't know intra-res tick.
            # We track the EXTREMES of this candle relative to entry.
            
            if trade.direction == "LONG":
                # Current Candle Range: low_p to high_p
                # Max potential profit this bar: high_p - entry
                # Max potential loss this bar: low_p - entry
                curr_max_pnl = (high_p - trade.entry_price) * size
                curr_min_pnl = (low_p - trade.entry_price) * size
                
                trade.max_unrealized_pnl = max(trade.max_unrealized_pnl, curr_max_pnl)
                trade.min_unrealized_pnl = min(trade.min_unrealized_pnl, curr_min_pnl)
                
                trade.max_favorable_excursion = max(trade.max_favorable_excursion, high_p - trade.entry_price)
                trade.max_adverse_excursion = max(trade.max_adverse_excursion, trade.entry_price - low_p)

                # Check Exit
                if high_p >= trade.take_profit:
                    trade.status = "WIN"
                    reward_dist = trade.take_profit - trade.entry_price
                    trade.pnl = size * reward_dist
                    trade.exit_time = timestamp
                    logger.info(f"✅ Trade {trade.id} WIN | PnL: ${trade.pnl:.2f} | MAE: ${trade.min_unrealized_pnl:.2f} (Drawdown) | MFE: ${trade.max_unrealized_pnl:.2f}")
                elif low_p <= trade.stop_loss:
                    trade.status = "LOSS"
                    trade.pnl = -size * dist
                    trade.exit_time = timestamp
                    logger.info(f"❌ Trade {trade.id} LOSS | PnL: ${trade.pnl:.2f} | MAE: ${trade.min_unrealized_pnl:.2f} (Drawdown)")

            else: # SHORT
                # Short Profit: Entry - Low
                # Short Loss: High - Entry
                curr_max_pnl = (trade.entry_price - low_p) * size
                curr_min_pnl = (trade.entry_price - high_p) * size
                
                trade.max_unrealized_pnl = max(trade.max_unrealized_pnl, curr_max_pnl)
                trade.min_unrealized_pnl = min(trade.min_unrealized_pnl, curr_min_pnl)
                
                trade.max_favorable_excursion = max(trade.max_favorable_excursion, trade.entry_price - low_p)
                trade.max_adverse_excursion = max(trade.max_adverse_excursion, high_p - trade.entry_price)

                if low_p <= trade.take_profit:
                    trade.status = "WIN"
                    reward_dist = trade.entry_price - trade.take_profit
                    trade.pnl = size * reward_dist
                    trade.exit_time = timestamp
                    logger.info(f"✅ Trade {trade.id} WIN | PnL: ${trade.pnl:.2f} | MAE: ${trade.min_unrealized_pnl:.2f} (Drawdown) | MFE: ${trade.max_unrealized_pnl:.2f}")
                elif high_p >= trade.stop_loss:
                    trade.status = "LOSS"
                    trade.pnl = -size * dist
                    trade.exit_time = timestamp
                    logger.info(f"❌ Trade {trade.id} LOSS | PnL: ${trade.pnl:.2f} | MAE: ${trade.min_unrealized_pnl:.2f} (Drawdown)")

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
            floor_minute = (timestamp.minute // 5) * 5
            candle_start_time = timestamp.replace(minute=floor_minute, second=0, microsecond=0)
            self.current_5m_candle = {
                'time': candle_start_time,
                'open': open_p, 'high': high_p, 'low': low_p, 'close': close_p
            }
        else:
            c = self.current_5m_candle
            c['high'] = max(c['high'], high_p)
            c['low'] = min(c['low'], low_p)
            c['close'] = close_p
            
        is_candle_close = (timestamp.minute % 5 == 4)
        
        # 3. Update Triggers & Check Entries
        # IMPORTANT: "Single Position" Check
        # If we already have a trade, we DO NOT look for new entries.
        # But we DO need to update existing triggers? 
        # Actually, if we ignore new setups while in a trade, we might miss the start of the pattern.
        # But logically, "Simple" mode = "Don't enter if in trade".
        # So we can keep tracking triggers, but just gating the `entry_trade` call.
        
        candidates_to_remove = []
        for cand in self.triggers:
            if (timestamp - cand['start_time']).total_seconds() > 3600:
                candidates_to_remove.append(cand)
                continue
                
            cand['max_high'] = max(cand['max_high'], high_p)
            cand['min_low'] = min(cand['min_low'], low_p)
            
            # SHORT REJECTION -> INVERSE LONG
            if cand['max_high'] >= cand['short_tgt']:
                if low_p <= cand['open']:
                    
                    stop_loss_price = cand['max_high']
                    risk_dist = stop_loss_price - cand['open']
                    rejection_risk = risk_dist
                    rejection_tp_dist = 1.4 * rejection_risk
                    
                    # GATE: Only enter if no active trades
                    if len(self.active_trades) < self.max_active_trades:
                        self.entry_trade(
                            direction="LONG",
                            timestamp=timestamp,
                            price=cand['open'],
                            stop_loss=cand['open'] - rejection_tp_dist, 
                            take_profit=cand['max_high'], 
                            risk=rejection_tp_dist,
                            atr=cand['atr']
                        )
                    candidates_to_remove.append(cand)
                    continue
                    
            # LONG REJECTION -> INVERSE SHORT
            if cand['min_low'] <= cand['long_tgt']:
                if high_p >= cand['open']:
                    
                    stop_loss_price = cand['min_low']
                    risk_dist = cand['open'] - stop_loss_price
                    rejection_risk = risk_dist
                    rejection_tp_dist = 1.4 * rejection_risk
                    
                    if len(self.active_trades) < self.max_active_trades:
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
        
        for c in candidates_to_remove:
            if c in self.triggers:
                self.triggers.remove(c)
        
        # 4. Handle 5m Close
        if is_candle_close:
            current_atr = self.calculate_atr() 
            c = self.current_5m_candle
            c['atr'] = current_atr
            self.bars_5m.append((c['time'], c['open'], c['high'], c['low'], c['close'], c['atr']))
            
            if not np.isnan(current_atr) and current_atr >= self.min_atr:
                self.triggers.append({
                    'start_time': c['time'],
                    'open': c['open'],
                    'atr': current_atr,
                    'short_tgt': c['open'] + (1.5 * current_atr),
                    'long_tgt': c['open'] - (1.5 * current_atr),
                    'max_high': c['high'], 
                    'min_low': c['low']   
                })
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
        logger.info(f"🚀 ENTER {direction} at {price:.2f} | Time: {timestamp} | Risk: ${self.risk_amount:.0f}")

class TestInverseSinglePos(unittest.TestCase):
    def test_single_position_logic(self):
        # 1. Fetch Data (FIXED DATES)
        ticker = "MES=F" 
        start_date = "2025-12-01"
        end_date = "2025-12-08" # 1 week
        logger.info(f"Downloading {ticker} data from {start_date} to {end_date}...")
        
        df = yf.download(ticker, start=start_date, end=end_date, interval="1m", progress=False)
        
        if df.empty:
            logger.warning("No data returned. Test cannot proceed.")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        
        logger.info(f"Loaded {len(df)} 1m candles.")
        
        # 2. Init Strategy (Single Position)
        strategy = SinglePositionInverseStrategy(risk_amount=300.0, max_active_trades=1)
        
        # 3. Stream Data
        logger.info("Streaming data...")
        for i, row in df.iterrows():
            ts = row.get('datetime', row.get('date'))
            strategy.on_bar(
                timestamp=ts,
                open_p=float(row['open']),
                high_p=float(row['high']),
                low_p=float(row['low']),
                close_p=float(row['close'])
            )
            
        # 4. Analysis
        if len(strategy.closed_trades) == 0:
            logger.warning("No trades triggered.")
        else:
            logger.info("TRADES DETECTED!")
            
            trades_data = []
            for t in strategy.closed_trades:
                trades_data.append({
                    'id': t.id,
                    'direction': t.direction,
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'pnl': t.pnl,
                    'outcome': t.status,
                    'max_mae': t.min_unrealized_pnl, # Max Drawdown during trade
                    'max_mfe': t.max_unrealized_pnl  # Max Profit during trade
                })
            
            res_df = pd.DataFrame(trades_data)
            res_df = res_df.sort_values('exit_time')
            res_df['cum_pnl'] = res_df['pnl'].cumsum()
            res_df['drawdown'] = res_df['cum_pnl'] - res_df['cum_pnl'].cummax()
            
            total_trades = len(res_df)
            wins = len(res_df[res_df['outcome'] == 'WIN'])
            win_rate = (wins / total_trades) * 100
            total_pnl = res_df['pnl'].sum()
            max_dd = res_df['drawdown'].min()
            
            avg_mae_win = res_df[res_df['outcome'] == 'WIN']['max_mae'].mean()
            avg_mae_loss = res_df[res_df['outcome'] == 'LOSS']['max_mae'].mean()
            
            logger.info("=" * 60)
            logger.info(f"SINGLE POSITION STRATEGY REPORT ({start_date} to {end_date})")
            logger.info(f"Total Trades:      {total_trades}")
            logger.info(f"Win Rate:          {win_rate:.2f}%")
            logger.info(f"Total PnL:         ${total_pnl:.2f}")
            logger.info(f"Max Equity DD:     ${max_dd:.2f}")
            logger.info("-" * 60)
            logger.info(f"Avg Drawdown (MAE) in Winners: ${avg_mae_win:.2f}")
            logger.info(f"Avg Drawdown (MAE) in Losers:  ${avg_mae_loss:.2f}")
            logger.info("=" * 60)
            
            # Plotting
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(res_df['exit_time'], res_df['cum_pnl'], label='Cumulative PnL', color='blue')
            plt.fill_between(res_df['exit_time'], res_df['cum_pnl'], 0, alpha=0.1, color='blue')
            plt.title(f"Single Position Inverse Strategy | Total PnL: ${total_pnl:.0f} | WR: {win_rate:.1f}%")
            plt.ylabel("PnL ($)")
            plt.grid(True)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            # Scatter of MAE vs PnL
            colors = ['g' if o == 'WIN' else 'r' for o in res_df['outcome']]
            plt.scatter(res_df['max_mae'], res_df['pnl'], c=colors, alpha=0.6)
            plt.title("Trade Outcome vs. Max Drawdown (MAE) Experienced")
            plt.xlabel("Max Unrealized Loss during Trade ($)")
            plt.ylabel("Final PnL ($)")
            plt.grid(True)
            
            plt.tight_layout()
            out_file = "inverse_single_pos_metrics.png"
            plt.savefig(out_file)
            logger.info(f"Saved analysis to {out_file}")
            plt.close()

if __name__ == '__main__':
    unittest.main()
