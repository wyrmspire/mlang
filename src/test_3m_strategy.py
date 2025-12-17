import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, MODELS_DIR
from train_3m_cnn import CNN_Rejection

def test_3m_strategy():
    print("Loading data...")
    df_1m = pd.read_parquet(PROCESSED_DIR / "continuous_1m.parquet")
    
    # Handle Index & Time
    if isinstance(df_1m.index, pd.DatetimeIndex) or 'time' not in df_1m.columns:
        df_1m = df_1m.reset_index()
    
    kw = [c for c in df_1m.columns if 'time' in c.lower() or 'date' in c.lower() or c == 'index']
    if kw:
        target = kw[0]
        for k in kw:
            if k == 'time': target = k
        df_1m = df_1m.rename(columns={target: 'time'})
        
    df_1m = df_1m.sort_values('time').reset_index(drop=True)
    df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
    
    # --- Split (Test on final 30%) ---
    # We use the same split logic as training to ensure no leakage
    # Ideally we'd use the explicit date, but index split is consistent if source is same
    # Wait, the miner used all data. The trainer used first 70% of TRADES.
    # The backtest should run on the last 30% of TIME (or roughly same period).
    # Let's approximate: simple index split of dataframe.
    
    split_idx = int(len(df_1m) * 0.70)
    df_test = df_1m.iloc[split_idx:].reset_index(drop=True)
    
    # --- 15m Resampling for Execution Logic ---
    print("Resampling to 15m...")
    df_15m = df_test.set_index('time').resample('15T').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    
    # Calculate 15m ATR for Stops
    df_15m['tr'] = np.maximum(
        df_15m['high'] - df_15m['low'],
        np.maximum(
            abs(df_15m['high'] - df_15m['close'].shift(1)),
            abs(df_15m['low'] - df_15m['close'].shift(1))
        )
    )
    df_15m['atr'] = df_15m['tr'].rolling(14).mean()
    
    # We need to map 15m ATR back to the decision time.
    # We iterate through 15m bars.
    
    # --- Load Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CNN_Rejection().to(device)
    model_path = MODELS_DIR / "cnn_3m_rejection.pth"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # --- Pre-calculate 15m context (Define df_sim) ---
    print("Pre-calculating 15m context...")
    df_context = df_test[['time', 'open', 'close']].copy()
    df_context['bucket_time'] = df_context['time'].dt.floor('15T')
    
    # Merge with 15m ATR and Open
    df_15m_lookup = df_15m[['time', 'open', 'atr']].rename(columns={'time': 'bucket_time', 'open': 'open_15m', 'atr': 'atr_15m'})
    
    df_sim = pd.merge(df_test, df_context[['time', 'bucket_time']], on='time')
    df_sim = pd.merge(df_sim, df_15m_lookup, on='bucket_time', how='left')
    
    # --- Batch Inference ---
    print("Preparing 30m context windows for batch inference...")
    
    vals = df_sim[['open', 'high', 'low', 'close']].values.astype(np.float32)
    
    # We need to normalize EACH window independently (standard scaler) ??
    # The training used per-window normalization. We must match it.
    # Vectorized per-window normalization is tricky without expanding memory.
    # We can use a customized Dataset/DataLoader with num_workers=0 (or higher) to fetch batches on the fly.
    
    from torch.utils.data import TensorDataset
    
    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, data_array, lookback=30):
            self.data = data_array
            self.lookback = lookback
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            if idx < self.lookback:
                # Pad
                return torch.zeros(4, self.lookback)
                
            window = self.data[idx-self.lookback : idx]
            
            # Normalize
            mean = window.mean()
            std = window.std()
            if std == 0: std = 1
            window = (window - mean) / std
            
            # (Seq, Dim) -> (Dim, Seq)
            return torch.FloatTensor(window).T
            
    inf_ds = InferenceDataset(vals)
    inf_loader = torch.utils.data.DataLoader(inf_ds, batch_size=4096, shuffle=False)
    
    print("Running Inference on GPU...")
    all_probs = []
    
    with torch.no_grad():
        for X in inf_loader:
             X = X.to(device)
             probs = model(X)
             all_probs.append(probs.cpu().numpy())
             
    all_probs = np.concatenate(all_probs).flatten()
    print("Inference Complete.")
    
    # --- Simulation Loop (1m Resolution) ---
    print(f"Starting Simulation on {len(df_sim)} 1m bars...")
    
    account_balance = 2000.0
    risk_per_trade = 75.0
    max_trades = 3
    
    open_trades = [] 
    closed_trades = []
    
    # We Loop 1m bars
    for i in range(50, len(df_sim)):
        curr_bar = df_sim.iloc[i]
        curr_time = curr_bar['time']
        
        # 1. Manage Open Trades
        remaining_trades = []
        for trade in open_trades:
            # SHORT Trade Logic
            # SL Hit if High >= Stop
            # TP Hit if Low <= TP
            
            sl_hit = (curr_bar['high'] >= trade['stop'])
            tp_hit = (curr_bar['low'] <= trade['tp'])
            
            status = 'OPEN'
            pnl = 0
            exit_price = 0
            
            if sl_hit and tp_hit:
                 # Conservative: Loss
                 status = 'LOSS'
                 pnl = -risk_per_trade
                 exit_price = trade['stop']
            elif sl_hit:
                 status = 'LOSS'
                 pnl = -risk_per_trade
                 exit_price = trade['stop']
            elif tp_hit:
                 status = 'WIN'
                 pnl = trade['reward']
                 exit_price = trade['tp']
                 
            if status != 'OPEN':
                trade['status'] = status
                trade['pnl'] = pnl
                trade['exit_time'] = curr_time
                trade['exit_price'] = exit_price
                closed_trades.append(trade)
                account_balance += pnl
            else:
                remaining_trades.append(trade)
        open_trades = remaining_trades
        
        # 2. Check for New Entry
        if len(open_trades) < max_trades:
             # Use pre-calculated probability
             prob = all_probs[i]
             
             if prob > 0.5:
                 # SIGNAL -> SHORT
                 
                 # Hypothetical 15m Candle Logic (Last 15m: i-14 to i)
                 # Wick of the window (Low for Long, High for Short)
                 start_scan = i - 14
                 if start_scan < 0: continue
                 
                 # Find Highest High in the 15m window
                 hyp_high = df_sim.iloc[start_scan : i+1]['high'].max()
                 
                 atr = curr_bar['atr_15m']
                 if pd.isna(atr): continue
                 
                 entry_price = curr_bar['close']
                 
                 # Logic: "stop .2 atr ABOVE the Wick (High)"
                 stop_price = hyp_high + 0.2 * atr
                 
                 # For Short: Stop must be ABOVE Entry.
                 if entry_price >= stop_price:
                     continue 
                     
                 dist = stop_price - entry_price
                 risk_amt = risk_per_trade
                 take_profit = entry_price - (2.2 * dist)
                 reward_amt = 2.2 * risk_amt
                 
                 new_trade = {
                    'entry_time': curr_time,
                    'entry_price': entry_price,
                    'stop': stop_price,
                    'tp': take_profit,
                    'reward': reward_amt,
                    'status': 'OPEN',
                    'direction': 'SHORT'
                 }
                 open_trades.append(new_trade)
                
    # End Simulation
    print(f"\nSimulation Complete.")
    print(f"Total Trades: {len(closed_trades)}")
    print(f"Final Balance: ${account_balance:.2f} (Start: $2000)")
    
    if len(closed_trades) > 0:
        wins = len([t for t in closed_trades if t['status'] == 'WIN'])
        wr = wins / len(closed_trades)
        print(f"Win Rate: {wr:.2%}")
        
    # Save log
    pd.DataFrame(closed_trades).to_csv("rejection_backtest_results.csv")
    print("Saved trades to rejection_backtest_results.csv")

if __name__ == "__main__":
    test_3m_strategy()
