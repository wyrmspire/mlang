import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger
from src.models.cnn_model import TradeCNN

logger = get_logger("train_rejection_cnn")

# GPU Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    logger.warning("GPU NOT DETECTED! Training will be slow.")
else:
    logger.info(f"Using device: {device}")

class TradeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(window_size=20):
    trades_path = PROCESSED_DIR / "labeled_rejections_5m.parquet"
    if not trades_path.exists():
        logger.error("No labeled rejections found. Run pattern_miner.py first.")
        return None, None
        
    logger.info("Loading rejections and raw data...")
    trades = pd.read_parquet(trades_path)
    trades = trades.sort_values('start_time') # Critical for chronological split
    
    df_1m = pd.read_parquet(ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet")
    df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
    df_1m = df_1m.set_index('time').sort_index()
    
    X = []
    y = []
    
    valid_trades = trades[trades['outcome'].isin(['WIN', 'LOSS'])]
    logger.info(f"Processing {len(valid_trades)} valid trades...")
    
    for idx, trade in valid_trades.iterrows():
        # User Logic: Train on 20 1m candles BEFORE the 'First Move' (Start Time).
        end_time = trade['start_time']
        start_time = end_time - pd.Timedelta(minutes=window_size)
        
        # Slice [start_time, end_time)
        slice_df = df_1m.loc[start_time:end_time]
        slice_df = slice_df[slice_df.index < end_time] # Strict inequality
        
        if len(slice_df) < window_size:
            continue
            
        base_price = slice_df.iloc[0]['open']
        if base_price == 0: continue
            
        feats = slice_df[['open', 'high', 'low', 'close']].values
        
        # Normalize: Z-Score per window
        mean = np.mean(feats)
        std = np.std(feats)
        if std == 0: std = 1.0 # Prevent div/0
        
        feats_norm = (feats - mean) / std
        
        # Ensure exact size
        if len(feats_norm) > window_size:
            feats_norm = feats_norm[-window_size:]
        elif len(feats_norm) < window_size:
             continue
             
        # Invert logic for SHORT trades?
        # If the pattern is Short, we want the model to learn "Bearish Context".
        # If Long, "Bullish Context".
        # We can either train separate models or FLIP the data for Shorts so the model learns "Setup Quality" regardless of direction.
        # Given "Proportions" request, flipping is smart.
        # But wait, User said: "price is at 5000 and goes up... we train on...".
        # The pattern miner finds both.
        # Standard approach: Invert price action for Shorts so "Up" always means "Win direction" (or Long setup).
        # But here, 'Win' means Rejection worked.
        # If Short: Price went UP (Extension), then Down.
        # If we Invert Short input: Price went DOWN (Extension), then Up. -> Looks like Long input.
        # So YES, we should invert Shorts to make them look like Longs, or vice versa, to unify the dataset.
        
        if trade['direction'] == 'SHORT':
            feats_norm = -feats_norm
            
        X.append(feats_norm)
        label = 1 if trade['outcome'] == 'WIN' else 0
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Prepared {len(X)} samples. Win Rate: {np.mean(y):.2f}")
    return X, y

def train():
    X, y = prepare_data()
    if X is None or len(X) == 0: 
        logger.error("No data prepared.")
        return
        
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # Datasets
    train_ds = TradeDataset(X_train, y_train)
    test_ds = TradeDataset(X_test, y_test)
    
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64)
    
    model = TradeCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    best_acc = 0.0
    
    logger.info("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct/total
        
        # Eval
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = (outputs > 0.5).float()
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        
        test_acc = test_correct / test_total if test_total > 0 else 0
        
        logger.info(f"Epoch {epoch+1}: Loss {running_loss/len(train_dl):.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODELS_DIR / "rejection_cnn_v1.pth")
            
    logger.info(f"Training Complete. Best Test Acc: {best_acc:.3f}")

if __name__ == "__main__":
    train()
