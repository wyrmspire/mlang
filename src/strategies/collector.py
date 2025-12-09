import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("collector")

@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str
    stop_loss: float
    take_profit: float
    atr_val: float
    buffer_mult: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None 
    pnl: float = 0.0

class TradeCollector:
    def __init__(self):
        self.trades: List[TradeRecord] = []
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        # DF must be 15m resampled
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def identify_engulfing(self, df: pd.DataFrame):
        # Bullish Engulfing: Prev Red, Curr Green, Curr Body > Prev Body (Overlap)
        # Prev Red: Close < Open
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        
        curr_open = df['open']
        curr_close = df['close']
        
        # Vectorized conditions
        prev_red = prev_close < prev_open
        curr_green = curr_close > curr_open
        
        # Engulfing Body:
        # Bullish: Curr Open <= Prev Close AND Curr Close >= Prev Open
        bull_engulf = (
            prev_red & curr_green & 
            (curr_open <= prev_close) & 
            (curr_close >= prev_open)
        )
        
        # Bearish Engulfing: Prev Green, Curr Red
        # Bearish: Curr Open >= Prev Close AND Curr Close <= Prev Open
        prev_green = prev_close > prev_open
        curr_red = curr_close < curr_open
        
        bear_engulf = (
            prev_green & curr_red &
            (curr_open >= prev_close) &
            (curr_close <= prev_open)
        )
        
        return bull_engulf, bear_engulf

    def run(self):
        input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if not input_path.exists():
            logger.error("Real data not found.")
            return

        logger.info("Loading Real Data...")
        df_1m = pd.read_parquet(input_path)
        df_1m['time'] = pd.to_datetime(df_1m['time'])
        df_1m = df_1m.sort_values('time').set_index('time')
        
        # Resample to 15m
        df_15m = df_1m.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        logger.info("Calculating ATR and Patterns on 15m data...")
        df_15m['atr'] = self.calculate_atr(df_15m)
        
        bull_mask, bear_mask = self.identify_engulfing(df_15m)
        
        # Collect Setup Candidates (Entry at CLOSE of the Engulfing Bar)
        # Entry Time = Timestamp of 15m bar + 15m (Close time)
        # Wait, the index is usually left-bound (start time).
        # Boolean masks align with the row.
        # If row 10:00 is Bull Engulfing, it finishes at 10:15.
        # We enter at Open of 10:15 bar (approx Close of 10:00 bar).
        
        candidates = []
        
        # We iterate only the engulfing bars
        triggers = df_15m[bull_mask | bear_mask]
        logger.info(f"Found {len(triggers)} Engulfing patterns.")
        
        # Pre-process 1m data for simulation lookup
        # Optimization: Don't pass full df to simulation every time if possible.
        # But we need granular lookup.
        
        # SWEEP PARAMETERS
        best_pnl = -float('inf')
        best_trades = []
        best_mult = 0.0
        
        sweep_values = np.arange(0.00, 0.55, 0.05) # 0.0 to 0.5
        
        for mult in sweep_values:
            logger.info(f"Simulating ATR Buffer Multiplier: {mult:.2f}")
            current_trades = []
            
            # Run simulation for this setting
            # We can re-use the candidate list but outcomes differ due to SL width
            
            for ts, row in triggers.iterrows():
                # Data check
                atr = row['atr']
                if pd.isna(atr) or atr == 0: continue
                
                # Determine Direction
                is_bull = bull_mask.loc[ts]
                direction = 'LONG' if is_bull else 'SHORT'
                
                # Pattern High/Low for Base SL
                pattern_high = row['high'] # Is SL usually below the SIGNAL candle? Yes.
                pattern_low = row['low']
                
                atr_buffer = mult * atr
                
                entry_price = row['close'] # Market on Close
                entry_time = ts + pd.Timedelta(minutes=15) # Resolution time
                
                if direction == 'LONG':
                    # SL below Low
                    sl_price = pattern_low - atr_buffer
                    risk = entry_price - sl_price
                    tp_price = entry_price + (1.4 * risk)
                else:
                    # SL above High
                    sl_price = pattern_high + atr_buffer
                    risk = sl_price - entry_price
                    tp_price = entry_price - (1.4 * risk)
                    
                # Sanity: negative risk?
                if risk <= 0: continue
                
                # Simulate Outcome using 1m data
                outcome, pnl, exit_px, exit_t = self.simulate_trade_vectorized(
                    df_1m, entry_time, entry_price, sl_price, tp_price, direction
                )
                
                current_trades.append({
                    'entry_time': entry_time,
                    'direction': direction,
                    'outcome': outcome,
                    'pnl': pnl,
                    'exit_time': exit_t,
                    'exit_price': exit_px,
                    'buffer_mult': mult
                })
                
            # Stats for this sweep
            if not current_trades: continue
            
            df_curr = pd.DataFrame(current_trades)
            total_pnl = df_curr['pnl'].sum()
            win_rate = len(df_curr[df_curr['outcome'] == 'WIN']) / len(df_curr)
            
            logger.info(f"Mult {mult:.2f} | Trades: {len(df_curr)} | WR: {win_rate:.2f} | PnL: {total_pnl:.2f}")
            
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_trades = current_trades
                best_mult = mult
                
        logger.info(f"--- SWEEP COMPLETE ---")
        logger.info(f"Best Multiplier: {best_mult:.2f} | PnL: {best_pnl:.2f}")
        
        # Save Best
        if best_trades:
            df_out = pd.DataFrame(best_trades)
            out_path = PROCESSED_DIR / "engulfing_trades.parquet"
            df_out.to_parquet(out_path)
            logger.info(f"Saved {len(df_out)} best trades to {out_path}")

    def simulate_trade_vectorized(self, df_1m, entry_time, entry_price, sl_price, tp_price, direction):
        # Subset
        subset = df_1m.loc[entry_time:].iloc[:2880] # Max 2 days (1440 * 2)
        if subset.empty:
            return 'TIMEOUT', 0.0, entry_price, entry_time
            
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
            pnl = (exit_px - entry_price) * (1 if direction == 'LONG' else -1)
        elif idx_win < idx_loss:
            outcome = 'WIN'
            exit_px = tp_price
            exit_t = times[idx_win]
            pnl = (tp_price - entry_price) * (1 if direction == 'LONG' else -1) # Exact TP PnL
        else:
            outcome = 'LOSS'
            exit_px = sl_price
            exit_t = times[idx_loss]
            pnl = (sl_price - entry_price) * (1 if direction == 'LONG' else -1) # Exact SL PnL
            
        return outcome, pnl, exit_px, exit_t

if __name__ == "__main__":
    c = TradeCollector()
    c.run()
