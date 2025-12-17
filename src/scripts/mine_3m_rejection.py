import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import PROCESSED_DIR

def mine_3m_continuous():
    print("Loading data...")
    df_1m = pd.read_parquet(PROCESSED_DIR / "continuous_1m.parquet")
    
    # Handle Index
    if isinstance(df_1m.index, pd.DatetimeIndex) or 'time' not in df_1m.columns:
        df_1m = df_1m.reset_index()
        
    # Standardize 'time' column name
    # Usually reset_index creates 'index' or 'time' depending on name
    # We force lowercase 'time'
    kw = [c for c in df_1m.columns if 'time' in c.lower() or 'date' in c.lower() or c == 'index']
    if kw:
        # Prefer 'time' or 'datetime'
        target = kw[0]
        for k in kw:
            if k == 'time': target = k
        df_1m = df_1m.rename(columns={target: 'time'})
        
    df_1m = df_1m.sort_values('time').reset_index(drop=True)
    
    # Ensure time is datetime
    df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
    
    # 1. Resample to 3m for Context (ATR, Start Candle definition)
    df_3m = df_1m.set_index('time').resample('3T').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    
    # Calculate ATR (14 period on 3m)
    df_3m['tr'] = np.maximum(
        df_3m['high'] - df_3m['low'],
        np.maximum(
            abs(df_3m['high'] - df_3m['close'].shift(1)),
            abs(df_3m['low'] - df_3m['close'].shift(1))
        )
    )
    df_3m['atr'] = df_3m['tr'].rolling(14).mean()
    
    # Map ATR back to 1m (ffill)
    # We want the ATR known AT the time, so we align by time.
    # 3m ATR available at 10:03 is calculated from 09:xx..10:00-10:03.
    # So for 1m bars inside the next candle, we can use the previous finished 3m ATR.
    df_atr = df_3m[['time', 'atr', 'high']].rename(columns={'high': '3m_high'})
    df_1m = pd.merge_asof(df_1m, df_atr, on='time', direction='backward') 
    
    # 2. Continuous Scanning
    # We look for: Start (Low) -> Peak (High) -> Current (Trigger)
    # R = (Peak - Start) / (Start - Trigger)
    # Valid if 1.5 <= R < 2.5
    # Trigger condition: Start - Current >= 1.0 (Min Unit $1)
    
    labeled_trades = []
    
    # Optimizing the loop:
    # Instead of O(N^2), strict scanning might be slow.
    # We'll use a sliding window approach or limited lookback.
    # Limit lookback for "Start" to e.g., 2 hours (40*3m = 120 bars)?
    
    closes = df_1m['close'].values
    times = df_1m['time'].values
    highs = df_1m['high'].values
    atrs = df_1m['atr'].values
    prices_3m_high = df_1m['3m_high'].values # High of the 3m candle associated with this time
    
    n = len(df_1m)
    print(f"Scanning {n} 1m bars...")
    
    # We need to jump forward to avoid overlapping the exact same trade?
    # Or capture all valid triggers? Let's capture distinct triggers.
    
    last_trade_time = times[0] - np.timedelta64(1, 'D')
    
    for i in range(200, n): # Start after warmup
        curr_time = times[i]
        curr_price = closes[i]
        
        # Optimization: Only check if price is dropping? 
        # Actually we need to find IF there was a valid Start/Peak before.
        
        # Lookback for Start (Minima)
        # We can look back say 120 bars (2 hours)
        lookback = 120
        start_scan = max(0, i - lookback)
        
        # Vectorized check for ratio?
        # Let's just loop backwards for simplicity first, optimize if slow.
        # We need a Start point such that:
        # Start < Peak
        # Current < Start
        
        if i % 10000 == 0:
            print(f"Processed {i}/{n}...")
            
        # Only consider triggers if we haven't traded very recently?
        if (curr_time - last_trade_time) < np.timedelta64(15, 'm'):
             continue
             
        # Find potential starts (local lows)
        # We iterate backwards from i-1
        for j in range(i-1, start_scan, -1):
            start_price = closes[j]
            
            # Condition: Current must be below Start
            drop_size = start_price - curr_price
            if drop_size < 1.0: # Min Unit $1
                continue
                
            # Find Peak in between [j, i]
            # Since j and i are close, just slice
            peaks = highs[j:i]
            peak_price = np.max(peaks)
            
            rise_size = peak_price - start_price
            
            if rise_size <= 0: continue
            
            ratio = rise_size / drop_size
            
            # Geometric Logic
            # 2.5 <= (Peak - Start) / (Start - Current) < 4.0
            # Equivalent to: 2.5 * Drop <= Rise < 4.0 * Drop
            
            if 2.5 <= ratio < 4.0:
                # Setup Found!
                
                # Check Invalidation: Did it exceed 4.0 BEFORE this drop?
                # The logic "Start + 4.0 * Unit" < Peak?
                # Wait, "Start + 4.0 * Unit" corresponds to Ratio 4.0.
                # If current Peak is within < 4.0 range, then it NEVER exceeded 4.0?
                # Correct. If peak_price was higher, ratio would be higher.
                # So simply checking ratio < 4.0 ensures it wasn't a runner.
                
                # Check "Start" Validity - usually a local low?
                # User didn't strictly specify start must be a swing low.
                # But it implies "Beginning of the ascent".
                # Let's assume any point that fits the geometry is a valid perspective.
                
                # DEFINE STOP
                # "stop goes .2 atr above the first candle that began the ascent"
                # We identify the 3m candle period for `times[j]` (Start Time)
                if pd.isna(atrs[j]): continue
                
                # Find the 3m bucket high for the start time
                # We joined '3m_high' previously
                stop_ref_high = prices_3m_high[j]
                stop_level = stop_ref_high + 0.2 * atrs[j]
                
                # Entry & Targets
                entry_price = curr_price
                risk_dist = stop_level - entry_price
                
                if risk_dist <= 0: continue # Invalid logical stop (price already above stop?)
                
                take_profit = entry_price - (1.4 * risk_dist)
                
                # Determine Outcome
                # Look forward from i
                future = df_1m.iloc[i+1:]
                outcome = 'OPEN'
                
                # Fast forward search
                # Hits SL or TP first?
                # Vector search
                sl_hit = future[future['high'] >= stop_level]
                tp_hit = future[future['low'] <= take_profit]
                
                sl_idx = sl_hit.index[0] if not sl_hit.empty else 999999999
                tp_idx = tp_hit.index[0] if not tp_hit.empty else 999999999
                
                if sl_idx == 999999999 and tp_idx == 999999999:
                    outcome = 'Inconclusive'
                elif tp_idx < sl_idx:
                    outcome = 'WIN'
                else:
                    outcome = 'LOSS'
                    
                labeled_trades.append({
                    'start_time': times[j],
                    'trigger_time': curr_time,
                    'unit_size': drop_size,
                    'entry': entry_price,
                    'stop': stop_level,
                    'tp': take_profit,
                    'outcome': outcome,
                    'ratio': ratio,
                    'peak': peak_price
                })
                
                last_trade_time = curr_time
                break # Move to next current time (Greedy: take first valid start for this moment)
                
    print(f"Found {len(labeled_trades)} patterns.")
    if len(labeled_trades) > 0:
        df_res = pd.DataFrame(labeled_trades)
        out_path = PROCESSED_DIR / "labeled_3m_rejection.parquet"
        df_res.to_parquet(out_path)
        print(f"Saved to {out_path}")
        print(df_res['outcome'].value_counts())

if __name__ == "__main__":
    mine_3m_continuous()
