
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR

PROCESSED_DATA_FILE = PROCESSED_DIR / "continuous_1m.parquet"

def calculate_atr(df, period=14):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    tr_list = []
    for i in range(len(df)):
        if i==0: tr_list.append(high[i]-low[i])
        else:
            tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
            tr_list.append(tr)
    return pd.Series(tr_list, index=df.index).rolling(period).mean()

def mine_5m_predictive():
    print("Loading data...")
    df = pd.read_parquet(PROCESSED_DATA_FILE).sort_index()
    
    # 1. Resample to 5m (Features)
    print("Resampling to 5m...")
    df_5m = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    # 2. Resample to 15m (ATR)
    print("Resampling to 15m for ATR...")
    df_15m = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    df_15m['atr_15m'] = calculate_atr(df_15m)
    
    # SHIFT ATR to avoid lookahead. 
    # The ATR for 10:00-10:15 (labeled 10:00) is known at 10:15.
    # We want it available for bars starting 10:15 onwards.
    # If we simply join on index, 10:00 5m gets 10:00 15m (Future).
    # Shifting by 1 means 10:15 15m row gets 10:00 15m data.
    # So when 10:15 5m joins to 10:15 15m, it gets the 10:00 ATR. Correct.
    df_15m['atr_15m'] = df_15m['atr_15m'].shift(1)
    
    # 3. Merge ATR back to 5m
    # We want the most recent COMPLETED 15m ATR available at the 5m timestamp
    # ffill allows 5:05 and 5:10 to use the ATR from 5:00 (closed 4:45-5:00 or similar depending on label)
    # Actually, if we use 'close' time, at 5:00 we know ATR of 4:45-5:00.
    
    joined = df_5m.join(df_15m['atr_15m'], how='left')
    joined['atr_15m'] = joined['atr_15m'].ffill()
    
    # Drop initial NaNs
    joined = joined.dropna()
    
    print("Scanning for setups (5m bars, 15m ATR targets)...")
    
    lookback = 20 # 20 x 5m bars = 100m context
    horizon = 3   # 3 x 5m bars = 15m execution window
    
    data = []
    indices = []
    
    opens = joined['open'].values
    highs = joined['high'].values
    lows = joined['low'].values
    closes = joined['close'].values
    atrs = joined['atr_15m'].values
    times = joined.index
    
    pos_count = 0
    neg_count = 0
    
    # Vectorized check might be hard due to horizon logic, doing it loop for clarity
    for i in tqdm(range(lookback, len(joined) - horizon)):
        
        current_close = closes[i]
        atr = atrs[i]
        
        if atr == 0 or np.isnan(atr): continue
        
        # Targets
        short_trigger = current_close + (1.5 * atr)
        long_trigger = current_close - (1.5 * atr)
        
        # Check Horizon (Next 3 bars) for REJECTION
        # Rejection = Hit Trigger AND Revert to Open (Close < Trigger)
        # Actually our strategy is Limit Order:
        # Sell Limit filled if High >= short_trigger.
        # Win if subsequent Low <= Target? 
        # Standard Rejection Strategy definition:
        # Win if we fill and then price moves in our favor.
        # Let's simplify: Label = 1 if High >= ShortTrigger (FILL) AND Price does NOT stop out (ShortTrigger + 1 ATR) before it hits (ShortTrigger - 1.5ATR)?
        #   OR simply: Price hits ShortTrigger and ends up LOWER?
        
        # Let's stick to the definition that worked: 
        # "Rejection" = Price extends 1.5 ATR and CLOSES back below. (Candle color flip logic)
        # But here we have 15m window.
        # Let's verify if a Limit Order would win.
        
        is_positive = False
        
        # Look ahead 3 bars
        for k in range(1, horizon + 1):
            fut_idx = i + k
            h = highs[fut_idx]
            l = lows[fut_idx]
            
            # SHORT setup check
            if h >= short_trigger:
                # Filled short.
                # Did it hit stop?
                sl_price = short_trigger + (1.0 * atr)
                if h >= sl_price:
                    # Stopped out in same specific bar?
                    # Assume worst case: Hit SL first if High touches both
                    # Actually standard assumption: if Close < SL, maybe survived?
                    # Let's just say: Label 1 if "Good Rejection".
                    pass 
                else:
                    # Not stopped.
                    # Did it revert?
                    # Profitable if price comes back down.
                    # Current strategy: TP is 'current_close' (Entry - 1.5 ATR)
                    # So can we hit 'current_close' after hitting 'short_trigger'?
                    if l <= current_close:
                        is_positive = True
                        break
            
            # LONG setup check
            if l <= long_trigger:
                sl_price = long_trigger - (1.0 * atr)
                if l <= sl_price:
                    pass
                else:
                    if h >= current_close:
                        is_positive = True
                        break
                        
        if is_positive:
            indices.append({'index': i, 'label': 1.0, 'time': times[i]})
            pos_count += 1
        elif np.random.rand() < 0.3: # Downsample negatives
            indices.append({'index': i, 'label': 0.0, 'time': times[i]})
            neg_count += 1
            
    print(f"Mined {len(indices)} samples. Pos: {pos_count}, Neg: {neg_count}")
    
    # Save
    labels_df = pd.DataFrame(indices)
    labels_df.to_parquet(PROCESSED_DIR / "labeled_predictive_5m.parquet")
    
    # Save the 5m features for training reader
    joined.to_parquet(PROCESSED_DIR / "features_5m_atr15m.parquet")
    print(f"Saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    mine_5m_predictive()
