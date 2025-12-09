
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR

PROCESSED_DATA_FILE = PROCESSED_DIR / "continuous_1m.parquet"

def calculate_atr(df, period=14):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    tr_list = [high[0]-low[0]]
    for i in range(1, len(df)):
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        tr_list.append(tr)
    return pd.Series(tr_list, index=df.index).rolling(period).mean()

def mine_predictive_labels():
    print("Loading data...")
    df = pd.read_parquet(PROCESSED_DATA_FILE)
    df = df.sort_index()
    
    # Calculate ATR (rolling 14)
    df['atr'] = calculate_atr(df)
    
    # Parameters
    LOOKBACK = 20
    LOOKAHEAD = 15
    ATR_MULT = 1.5
    
    data = []
    
    # Vectorized approach or rolling? 
    # Labeling loop is safer for complex logic
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['atr'].values
    times = df.index
    
    print(f"Scanning {len(df)} candles for predictive setups...")
    
    count_pos = 0
    count_neg = 0
    
    # We step through 1m data
    # At index i (current time), we look at [i-LOOKBACK : i] as Input
    # We look at [i+1 : i+LOOKAHEAD] for Pattern
    
    for i in range(LOOKBACK, len(df) - LOOKAHEAD):
        curr_open = opens[i]
        curr_atr = atrs[i]
        
        if np.isnan(curr_atr) or curr_atr == 0: continue
        
        # Define Targets based on CURRENT info (at time i)
        # We want to place a Limit Sell at: curr_open + 1.5*ATR
        # We want to place a Limit Buy at: curr_open - 1.5*ATR
        short_target = curr_open + (ATR_MULT * curr_atr)
        long_target = curr_open - (ATR_MULT * curr_atr)
        
        # Check Lookahead Window for rejection
        short_success = 0
        long_success = 0
        
        # We want price to HIT the target but REJECT (Close below it for short)
        # Actually, for the "Original" Strategy, we just want price to HIT the target and then revert?
        # Rejection Definition:
        # High >= Target AND Close < Target (Wick created)
        
        for j in range(1, LOOKAHEAD+1):
            future_idx = i + j
            
            # Check Short Rejection
            if highs[future_idx] >= short_target and closes[future_idx] < short_target:
                short_success = 1
                break # Found one
                
        # Check Long Rejection
        for j in range(1, LOOKAHEAD+1):
            future_idx = i + j
            if lows[future_idx] <= long_target and closes[future_idx] > long_target:
                long_success = 1
                break
        
        # Store Feature Window (Indices)
        # We store just the index and label, will load features during training to save space
        
        if short_success:
            data.append({'index': i, 'label': 1, 'type': 'SHORT'})
            count_pos += 1
        elif long_success:
            data.append({'index': i, 'label': 1, 'type': 'LONG'})
            count_pos += 1
        else:
            # Downsample Negatives? 
            # Let's save them all for now, handle balancing in Dataset
            data.append({'index': i, 'label': 0, 'type': 'NONE'})
            count_neg += 1
            
    print(f"Mined {len(data)} samples.")
    print(f"Positives: {count_pos}")
    print(f"Negatives: {count_neg}")
    
    out_df = pd.DataFrame(data)
    out_path = PROCESSED_DIR / "labeled_predictive.parquet"
    out_df.to_parquet(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    mine_predictive_labels()
