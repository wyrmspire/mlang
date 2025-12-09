import pandas as pd
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import sys
# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR, LOCAL_TZ
from src.utils.logging_utils import get_logger

logger = get_logger("random_tilt")

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str # 'LONG' or 'SHORT'
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = 0.0
    outcome: str = None # 'WIN', 'LOSS'
    setup_window_start: pd.Timestamp = None # For CNN mapping

class RandomTiltStrategy:
    def __init__(self, 
                 tp_ticks: int = 20, 
                 sl_ticks: int = 10, 
                 tick_size: float = 0.25,
                 base_prob: float = 0.5):
        self.tp_dist = tp_ticks * tick_size
        self.sl_dist = sl_ticks * tick_size
        self.tick_size = tick_size
        self.base_prob = base_prob # Base probability for Long
        self.trades: List[Trade] = []
        
    def calculate_tilt(self, df_window: pd.DataFrame) -> float:
        """
        Determine probability of going LONG based on recent price action.
        Simple logic: 
        - If recent trend is up, tilt long (e.g. 0.6).
        - If range is tight, keep near 0.5.
        """
        if len(df_window) < 2:
            return 0.5
            
        last_close = df_window.iloc[-1]['close']
        prev_close = df_window.iloc[-2]['close']
        
        # Simple Momentum Tilt
        # If last 5m candle was Green, tilt Long slightly?
        tilt = 0.5
        
        # Trend Component (last 3 bars)
        closes = df_window['close'].values
        if len(closes) >= 3:
            slope = (closes[-1] - closes[-3]) / closes[-3]
            # Max tilt +/- 0.2
            # e.g. 0.1% move -> 0.1 tilt?
            dir_tilt = np.clip(slope * 100, -0.2, 0.2)
            tilt += dir_tilt
            
        return np.clip(tilt, 0.1, 0.9)

    def run_simulation(self, start_date: str = "2024-01-01", days: int = 20):
        """
        Run simulation over 1-min data (resampled to 5m for decisions).
        """
        input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if not input_path.exists():
            logger.error("Data not found.")
            return

        logger.info(f"Loading data for simulation ({days} days)...")
        df_1m = pd.read_parquet(input_path)
        df_1m['time'] = pd.to_datetime(df_1m['time'])
        df_1m = df_1m.sort_values('time')
        
        # Filter date range? For now just take tail if needed or all
        # Let's resample to 5m for decision points
        df_5m = df_1m.set_index('time').resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna().reset_index()
        
        logger.info(f"Simulating on {len(df_5m)} 5-min bars...")
        
        # Simulation Loop
        # We need 1-min granularity for realistic SL/TP fills
        # But decisions happen at 5m close.
        
        # Map 5m decision times to 1m indices for lookup
        # Optimization: Iterate 5m bars, then look ahead in 1m data for outcome.
        
        # Build lookup for 1m data
        df_1m_indexed = df_1m.set_index('time').sort_index()
        
        for i in range(4, len(df_5m)-1): 
            # Context window (e.g. last 4 bars for calculating tilt)
            window = df_5m.iloc[i-4:i+1]
            current_bar = df_5m.iloc[i]
            
            # Decision Time: Close of current_bar
            decision_time = current_bar['time'] + pd.Timedelta(minutes=5)
            
            # 1. Calc Tilt
            prob_long = self.calculate_tilt(window)
            
            # 2. Decision
            # Random roll
            is_long = random.random() < prob_long
            
            direction = 'LONG' if is_long else 'SHORT'
            entry_price = current_bar['close'] # Market on Close (theoretical)
            # Actually next bar Open is more realistic, let's assume fill at Close for simplicity/speed
            
            # 3. Resolve Outcome (Look ahead in 1m data)
            # We look up to X hours ahead or until fill
            future_1m = df_1m_indexed.loc[decision_time : decision_time + pd.Timedelta(hours=2)]
            
            if future_1m.empty:
                continue
                
            tp_price = entry_price + self.tp_dist if is_long else entry_price - self.tp_dist
            sl_price = entry_price - self.sl_dist if is_long else entry_price + self.sl_dist
            
            outcome = 'TIMEOUT'
            exit_px = future_1m.iloc[-1]['close']
            exit_time = future_1m.iloc[-1].name
            
            for _, row in future_1m.iterrows():
                h, l = row['high'], row['low']
                
                # Check stops
                passed_tp = (h >= tp_price) if is_long else (l <= tp_price)
                passed_sl = (l <= sl_price) if is_long else (h >= sl_price)
                
                if passed_sl and passed_tp:
                    # Conflict: Both hit in same minute. Assume SL (stats conservatism)
                    outcome = 'LOSS'
                    exit_px = sl_price
                    exit_time = row.name
                    break
                elif passed_sl:
                    outcome = 'LOSS'
                    exit_px = sl_price
                    exit_time = row.name
                    break
                elif passed_tp:
                    outcome = 'WIN'
                    exit_px = tp_price
                    exit_time = row.name
                    break
            
            # Record Trade
            pnl = (exit_px - entry_price) * (1 if is_long else -1)
            
            # Save 20m window start (for CNN)
            # 20m before decision time
            setup_start = decision_time - pd.Timedelta(minutes=20)
            
            t = Trade(
                entry_time=decision_time,
                entry_price=entry_price,
                direction=direction,
                exit_time=exit_time,
                exit_price=exit_px,
                pnl=pnl,
                outcome=outcome,
                setup_window_start=setup_start
            )
            self.trades.append(t)
            
        logger.info(f"Simulation complete. Generated {len(self.trades)} trades.")
        self.save_trades()

    def save_trades(self):
        records = [
            {
                'entry_time': t.entry_time,
                'entry_price': t.entry_price,
                'direction': t.direction,
                'exit_time': t.exit_time,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'outcome': t.outcome,
                'setup_window_start': t.setup_window_start
            }
            for t in self.trades
        ]
        df = pd.DataFrame(records)
        out_path = PROCESSED_DIR / "random_tilt_trades.parquet"
        df.to_parquet(out_path)
        logger.info(f"Saved trades to {out_path}")

if __name__ == "__main__":
    # Test Run
    strat = RandomTiltStrategy()
    strat.run_simulation(days=5)
