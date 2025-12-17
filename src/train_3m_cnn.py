import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, MODELS_DIR

# --- Dataset Definition ---
class RejectionDataset(Dataset):
    def __init__(self, trades, candles, lookback=30):
        self.trades = trades
        self.candles = candles # 1m dataframe
        self.lookback = lookback
        
        # Prepare valid indices mapping
        # We need fast access to candle data by timestamp
        # Ensure candles are sorted
        if not self.candles.index.name == 'time':
            self.candles = self.candles.set_index('time').sort_index()
            
    def __len__(self):
        return len(self.trades)
        
    def __getitem__(self, idx):
        trade = self.trades.iloc[idx]
        start_time = trade['start_time']
        target = 1.0 if trade['outcome'] == 'WIN' else 0.0
        
        # Input: 30m context BEFORE start_time
        # We want data strictly < start_time
        # Slice: [start_time - 30m, start_time)
        
        # We need integer indexing or slicing on the datetime index
        # searchsorted returns the insertion point to maintain order
        # "left" means index of the first element >= value
        
        # We want the index of 'start_time' in candles
        # If start_time exists, we take prior 30 bars.
        
        end_idx = self.candles.index.searchsorted(start_time)
        start_idx = end_idx - self.lookback
        
        if start_idx < 0:
            # Padding
            # Create zeros (Lookback, 4)
            data = np.zeros((self.lookback, 4), dtype=np.float32)
        else:
            subset = self.candles.iloc[start_idx:end_idx]
            vals = subset[['open', 'high', 'low', 'close']].values
            
            # Normalize Standard Scaler per window
            mean = vals.mean()
            std = vals.std()
            if std == 0: std = 1
            data = (vals - mean) / std
            
            # Pad if short (e.g. gaps)
            if len(data) < self.lookback:
                 p = np.zeros((self.lookback - len(data), 4), dtype=np.float32)
                 data = np.vstack([p, data])
                 
        # To Tensor (Seq, Dim) -> (Dim, Seq) for Conv1d
        # Input: (4, 30)
        t_data = torch.FloatTensor(data).transpose(0, 1)
        t_target = torch.FloatTensor([target])
        
        return t_data, t_target

# --- Model Definition ---
class CNN_Rejection(nn.Module):
    def __init__(self, input_dim=4, seq_len=30):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # 30 -> 15
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # 15 -> 7
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 7 -> 3
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (Batch, Dim, Seq)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_3m_model():
    # 1. Load Data
    print("Loading data...")
    df_1m = pd.read_parquet(PROCESSED_DIR / "continuous_1m.parquet")
    # Ensure time index for Dataset
    if 'time' in df_1m.columns:
        df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
        # df_1m = df_1m.set_index('time').sort_index() # Done in Dataset init
    
    trades = pd.read_parquet(PROCESSED_DIR / "labeled_3m_rejection.parquet")
    # Ensure trades start_time is UTC
    trades['start_time'] = pd.to_datetime(trades['start_time'], utc=True)
    
    # 2. Time Split (70% Train)
    trades = trades.sort_values('start_time')
    split_idx = int(len(trades) * 0.70)
    train_trades = trades.iloc[:split_idx]
    
    # We DO NOT touch the test set here (final 30%).
    # We can split Train into Train/Val for validation during training.
    # Let's do 80/20 split of the "Train" set for validation.
    val_split = int(len(train_trades) * 0.80)
    val_subset = train_trades.iloc[val_split:]
    train_subset = train_trades.iloc[:val_split]
    
    print(f"Total Trades: {len(trades)}")
    print(f"Training Set: {len(train_subset)} | Validation Set: {len(val_subset)}")
    print(f"Target Distribution (Train): \n{train_subset['outcome'].value_counts(normalize=True)}")
    
    # 3. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_ds = RejectionDataset(train_subset, df_1m)
    val_ds = RejectionDataset(val_subset, df_1m)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0) # workers=0 for windows compat
    val_loader = DataLoader(val_ds, batch_size=64)
    
    model = CNN_Rejection().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Loop
    epochs = 20
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (out > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item()
                preds = (out > 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
                
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODELS_DIR / "cnn_3m_rejection.pth")
            print("  [Saved Best Model]")
            
    print("\nTraining Complete.")
    print(f"Model saved to {MODELS_DIR / 'cnn_3m_rejection.pth'}")

if __name__ == "__main__":
    train_3m_model()
