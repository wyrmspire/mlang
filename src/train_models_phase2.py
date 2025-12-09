
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import copy

sys.path.append(str(Path(__file__).parent.parent))
from src.config import PROCESSED_DIR, MODELS_DIR
from src.models.variants import CNN_Classic, CNN_Wide, LSTM_Seq, Feature_MLP

# Utils
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(df):
    # Engineer features for MLP
    # RSI, ATR, Dist from MA(20), MA(50), Volatility
    df['rsi'] = compute_rsi(df['close'])
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['dist_ma20'] = (df['close'] - df['ma20']) / df['ma20']
    df['dist_ma50'] = (df['close'] - df['ma50']) / df['ma50']
    df['atr'] = (df['high'] - df['low']).rolling(14).mean() / df['close'] # Normalized ATR
    
    # Lagged returns
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    
    # Drop NaNs
    df = df.fillna(0.0)
    
    # Features list
    feats = ['rsi', 'dist_ma20', 'dist_ma50', 'atr', 'ret_1', 'ret_5', 'volume']
    # Normalize
    for f in feats:
        m = df[f].mean()
        s = df[f].std()
        if s == 0: s = 1
        df[f] = (df[f] - m) / s
        
    return df, feats

class TradeDataset(Dataset):
    def __init__(self, trades, candles, mode='classic', lookback=20, feature_cols=None):
        self.trades = trades
        self.candles = candles
        self.mode = mode
        self.lookback = lookback
        self.feature_cols = feature_cols
        
    def __len__(self):
        return len(self.trades)
        
    def __getitem__(self, idx):
        row = self.trades.iloc[idx]
        trigger_time = row['trigger_time'] # 5m trigger time
        
        # Determine Target (1 = Inverse WIN = Rejection LOSS)
        target = 1.0 if row['outcome'] == 'LOSS' else 0.0
        
        # Get Window (ending at trigger_time)
        # We need 1m candles
        # trigger_time is 5m bucket. E.g. 09:05.
        # We want data BEFORE 09:05.
        
        end_idx = self.candles.index.searchsorted(trigger_time)
        start_idx = end_idx - self.lookback
        
        if start_idx < 0:
            # Padding? or just fail safe
            start_idx = 0
            
        subset = self.candles.iloc[start_idx:end_idx]
        
        # Pad if short
        if len(subset) < self.lookback:
            # This shouldn't happen often if we filter trades
            # Create zeros
            pass 
            
        # Prepare Input Tensor
        if self.mode == 'mlp':
            # Take features from the LAST row (closest to decision time)
            feats = subset[self.feature_cols].iloc[-1].values
            return torch.FloatTensor(feats), torch.FloatTensor([target])
        else:
            # Image/Sequence (O/H/L/C)
            # Normalize window
            vals = subset[['open', 'high', 'low', 'close']].values
            mean = vals.mean()
            std = vals.std()
            if std == 0: std = 1
            vals = (vals - mean) / std
            
            return torch.FloatTensor(vals), torch.FloatTensor([target])

def train_phase2():
    # 1. Load Data
    print("Loading data...")
    df_1m = pd.read_parquet(PROCESSED_DIR / "continuous_1m.parquet")
    trades = pd.read_parquet(PROCESSED_DIR / "labeled_continuous.parquet")
    
    # 2. Features for MLP
    df_1m, fe_cols = prepare_features(df_1m)
    print(f"Engineered {len(fe_cols)} features for MLP.")
    
    # 3. Split
    trades = trades.sort_values('trigger_time')
    split = int(0.8 * len(trades))
    train_trades = trades.iloc[:split]
    val_trades = trades.iloc[split:]
    print(f"Train: {len(train_trades)}, Val: {len(val_trades)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # 4. Train Configs
    configs = [
        {'name': 'CNN_Classic', 'model': CNN_Classic(), 'mode': 'classic', 'lookback': 20},
        {'name': 'CNN_Wide',    'model': CNN_Wide(),    'mode': 'classic', 'lookback': 60},
        {'name': 'LSTM_Seq',    'model': LSTM_Seq(),    'mode': 'classic', 'lookback': 20},
        {'name': 'Feature_MLP', 'model': Feature_MLP(input_dim=len(fe_cols)), 'mode': 'mlp', 'features': fe_cols}
    ]
    
    for cfg in configs:
        print(f"\n--- Training {cfg['name']} ---")
        model = cfg['model'].to(device)
        
        # Datasets
        train_ds = TradeDataset(train_trades, df_1m, mode=cfg['mode'], lookback=cfg.get('lookback', 20), feature_cols=cfg.get('features'))
        val_ds = TradeDataset(val_trades, df_1m, mode=cfg['mode'], lookback=cfg.get('lookback', 20), feature_cols=cfg.get('features'))
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        best_acc = 0.0
        
        for epoch in range(10): # 10 epochs
            model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            # Val
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    predicted = (pred > 0.5).float()
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
            
            acc = correct / total
            print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), MODELS_DIR / f"{cfg['name']}.pth")
                
        print(f"Best Val Acc for {cfg['name']}: {best_acc:.4f}")

if __name__ == "__main__":
    train_phase2()
