import pandas as pd
import numpy as np
from pathlib import Path
import sys
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR

def run():
    print("Starting Miner...", flush=True)
    input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    # Load
    df = pd.read_parquet(input_path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time').sort_index()
    print(f"Loaded {len(df)} 1m rows.", flush=True)
    
    # Resample 5m
    df_5m = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    print(f"Resampled to {len(df_5m)} 5m rows.", flush=True)
    
    # ATR
    high_low = df_5m['high'] - df_5m['low']
    high_close = np.abs(df_5m['high'] - df_5m['close'].shift())
    low_close = np.abs(df_5m['low'] - df_5m['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_5m['atr'] = tr.rolling(window=14).mean().shift(1)
    
    # Scan
    triggers = []
    opens = df_5m['open'].values
    highs = df_5m['high'].values
    lows = df_5m['low'].values
    times = df_5m.index.values
    atrs = df_5m['atr'].values
    
    n = len(df_5m)
    expansion_ratio = 1.5
    min_atr = 5.0
    max_lookahead = 12 # 60 mins
    
    print("Scanning for patterns...", flush=True)
    for i in range(14, n - max_lookahead):
        start_open = opens[i]
        R = atrs[i]
        
        if np.isnan(R) or R < min_atr: continue
        
        short_tgt = start_open + (expansion_ratio * R)
        long_tgt = start_open - (expansion_ratio * R)
        
        max_r = -99999.0
        min_r = 99999.0
        
        for j in range(i, i + max_lookahead):
            curr_h = highs[j]
            curr_l = lows[j]
            if curr_h > max_r: max_r = curr_h
            if curr_l < min_r: min_r = curr_l
            
            # Short Setup
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
                    
    print(f"Found {len(triggers)} triggers.", flush=True)
    if not triggers: return
    
    trig_df = pd.DataFrame(triggers)
    
    # Label Outcomes (Precision Mode)
    print("Labeling outcomes...", flush=True)
    trig_df['start_time'] = pd.to_datetime(trig_df['start_time'], utc=True)
    trig_df['trigger_time'] = pd.to_datetime(trig_df['trigger_time'], utc=True)
    
    outcomes = []
    
    for idx, row in trig_df.iterrows():
        # Check from trigger time + 5 mins
        start_check = row['trigger_time'] + pd.Timedelta(minutes=5)
        future = df.loc[start_check:].iloc[:2000] # Check next 2000 1m candles
        
        if future.empty:
            outcomes.append('UNKNOWN')
            continue
            
        highs_f = future['high'].values
        lows_f = future['low'].values
        tp = row['take_profit']
        sl = row['stop_loss']
        outcome = 'TIMEOUT'
        
        if row['direction'] == 'LONG':
            wins = np.where(highs_f >= tp)[0]
            losses = np.where(lows_f <= sl)[0]
        else:
            wins = np.where(lows_f <= tp)[0]
            losses = np.where(highs_f >= sl)[0]
            
        w_idx = wins[0] if len(wins) > 0 else 999999
        l_idx = losses[0] if len(losses) > 0 else 999999
        
        if w_idx < l_idx: outcome = 'WIN'
        elif l_idx < w_idx: outcome = 'LOSS'
        
        outcomes.append(outcome)
        
    trig_df['outcome'] = outcomes
    valid = trig_df[trig_df['outcome'].isin(['WIN', 'LOSS'])]
    print(f"Valid Trades: {len(valid)}", flush=True)
    
    out_path = PROCESSED_DIR / "labeled_rejections_5m.parquet"
    valid.to_parquet(out_path)
    print(f"Saved to {out_path}", flush=True)

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"CRITICAL: {e}")
        traceback.print_exc()
