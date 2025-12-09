
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR

def mine_patterns():
    input_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not input_path.exists():
        print(f"Error: {input_path} missing.")
        return

    print("Loading continuous 1m data...")
    df = pd.read_parquet(input_path)
    
    # Resample 5m
    print("Resampling to 5m...")
    df_5m = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # ATR
    high_low = df_5m['high'] - df_5m['low']
    high_close = np.abs(df_5m['high'] - df_5m['close'].shift())
    low_close = np.abs(df_5m['low'] - df_5m['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_5m['atr'] = tr.rolling(window=14).mean().shift(1)
    
    # Scan logic (Vectorized / Fast Loop)
    triggers = []
    opens = df_5m['open'].values
    highs = df_5m['high'].values
    lows = df_5m['low'].values
    times = df_5m.index.values
    atrs = df_5m['atr'].values
    
    n = len(df_5m)
    expansion_ratio = 1.5
    min_atr = 5.0
    max_lookahead = 12 
    
    print(f"Scanning {n} candles for patterns...")
    
    for i in range(14, n - max_lookahead):
        R = atrs[i]
        if np.isnan(R) or R < min_atr: continue
        
        start_open = opens[i]
        short_tgt = start_open + (expansion_ratio * R)
        long_tgt = start_open - (expansion_ratio * R)
        
        max_r = -99999.0
        min_r = 99999.0
        
        for j in range(i, i + max_lookahead):
            curr_h = highs[j]
            curr_l = lows[j]
            if curr_h > max_r: max_r = curr_h
            if curr_l < min_r: min_r = curr_l
            
            # Short Setup (Rejection Logic)
            if max_r >= short_tgt:
                if curr_l <= start_open:
                    stop_loss = max_r
                    risk = stop_loss - start_open
                    triggers.append({
                        'start_time': times[i],
                        'trigger_time': times[j],
                        'direction': 'SHORT',
                        'entry_price': start_open,
                        'stop_loss': stop_loss,
                        'take_profit': start_open - (1.4 * risk),
                        'risk': risk,
                        'atr': R
                    })
                    break
            
            # Long Setup
            if min_r <= long_tgt:
                if curr_h >= start_open:
                    stop_loss = min_r
                    risk = start_open - stop_loss
                    triggers.append({
                        'start_time': times[i],
                        'trigger_time': times[j],
                        'direction': 'LONG',
                        'entry_price': start_open,
                        'stop_loss': stop_loss,
                        'take_profit': start_open + (1.4 * risk),
                        'risk': risk,
                        'atr': R
                    })
                    break

    print(f"Found {len(triggers)} potential triggers.")
    if not triggers: return

    trig_df = pd.DataFrame(triggers)
    
    # Label Outcomes (Precision Mode) using 1m data
    print("Labeling outcomes using 1m data...")
    trig_df['trigger_time'] = pd.to_datetime(trig_df['trigger_time'], utc=True)
    
    outcomes = []
    
    # Ensure 1m index is sorted
    df = df.sort_index()
    
    for idx, row in trig_df.iterrows():
        # Check from trigger time + 5 mins (Allow fill? No, assume instant fill at Entry)
        # Actually logic says "trigger_time" is the 5m close time? No, times[j] is the 5m candle timestamp.
        # But the fill happens intra-candle. 
        # For simplicity, let's start checking from the *next* 1m candle after trigger_time (which aligns to 5m start).
        
        start_check = row['trigger_time'] + pd.Timedelta(minutes=5)
        
        future = df.loc[start_check:].iloc[:5000] # Check next 5000 mins ~3 days
        if future.empty:
            outcomes.append('UNKNOWN')
            continue
            
        highs_f = future['high'].values
        lows_f = future['low'].values
        tp = row['take_profit'] # Rejection TP
        sl = row['stop_loss']   # Rejection SL
        
        outcome = 'TIMEOUT'
        
        if row['direction'] == 'LONG':
            # Rejection LONG: Target Up, Stop Down.
            wins = np.where(highs_f >= tp)[0]
            losses = np.where(lows_f <= sl)[0]
        else:
            # Rejection SHORT: Target Down, Stop Up
            wins = np.where(lows_f <= tp)[0]
            losses = np.where(highs_f >= sl)[0]
            
        w_idx = wins[0] if len(wins) > 0 else 999999
        l_idx = losses[0] if len(losses) > 0 else 999999
        
        if w_idx < l_idx: outcome = 'WIN'
        elif l_idx < w_idx: outcome = 'LOSS'
        
        outcomes.append(outcome)
        
    trig_df['outcome'] = outcomes
    valid = trig_df[trig_df['outcome'].isin(['WIN', 'LOSS'])]
    
    print(f"Valid Labeled Trades: {len(valid)}")
    print(valid['outcome'].value_counts())
    
    out_path = PROCESSED_DIR / "labeled_continuous.parquet"
    valid.to_parquet(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    mine_patterns()
