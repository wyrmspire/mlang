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

logger = get_logger("train_cnn")

# GPU Check - Strict
if not torch.cuda.is_available():
    logger.error("GPU NOT DETECTED! User required GPU for training.")
    sys.exit(1)

device = torch.device("cuda")
logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

class TradeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1) # Binary classification
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TradeCNN(nn.Module):
    def __init__(self, input_len=20, input_channels=4):
        super(TradeCNN, self).__init__()
        # Input shape: (Batch, Channels, Length) -> (B, 4, 20)
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * 5, 32) 
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (Batch, Length, Channels) from preparation?
        # PyTorch Conv1d expects (Batch, Channels, Length)
        x = x.permute(0, 2, 1) 
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def prepare_data(window_size=20):
    trades_path = PROCESSED_DIR / "engulfing_trades.parquet"
    if not trades_path.exists():
        logger.error("No trades found.")
        return None, None, None

    logger.info("Loading collected trades and raw data...")
    trades = pd.read_parquet(trades_path)
    trades = trades.sort_values('entry_time') # Ensure chronological order
    
    df_1m = pd.read_parquet(ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet")
    df_1m['time'] = pd.to_datetime(df_1m['time'])
    df_1m = df_1m.set_index('time').sort_index()
    
    X = []
    y = []
    
    # We only care about VALID outcomes for training (Win/Loss)
    valid_trades = trades[trades['outcome'].isin(['WIN', 'LOSS'])]
    logger.info(f"Processing {len(valid_trades)} valid trades for training data...")
    
    for idx, trade in valid_trades.iterrows():
        # User Requirement: Train on 20 1m candles BEFORE the trade setup (Engulfing candle).
        # entry_time is the CLOSE of the 15m Engulfing bar.
        # So we need to shift back 15m to get to the OPEN of the Engulfing bar.
        # And then take 20m before THAT.
        
        setup_start_time = trade['entry_time'] - pd.Timedelta(minutes=15)
        end_time = setup_start_time
        start_time = end_time - pd.Timedelta(minutes=window_size)
        
        slice_df = df_1m.loc[start_time:end_time]
        
        # Check size (rough check, might include end_time bar depending on slice exactness)
        # We want strict window.
        slice_df = slice_df[slice_df.index < end_time]
        
        if len(slice_df) < window_size:
            continue
            
        base_price = slice_df.iloc[0]['open']
        if base_price == 0: continue
            
        feats = slice_df[['open', 'high', 'low', 'close']].values
        # Normalize
        feats_norm = (feats / base_price) - 1.0
        
        if len(feats_norm) > window_size:
            feats_norm = feats_norm[-window_size:]
        elif len(feats_norm) < window_size:
             continue
             
        # Label: Win = 1
        label = 1 if trade['outcome'] == 'WIN' else 0
        
        if trade['direction'] == 'SHORT':
             # Invert returns actions
             feats_norm = -feats_norm
             
        X.append(feats_norm)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Prepared {len(X)} samples. Win Rate: {np.mean(y):.2f}")
    return X, y, trades # Return trades to identify split time

def train():
    X, y, trades_df = prepare_data()
    if X is None: return
    
    # 60 / 40 Split (Sequential)
    split_idx = int(0.6 * len(X))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Identify Split Timestamp for Smart Strategy
    # Valid trades were filtered, so finding exact time mapping requires mapping X back to trades
    # But approximate is fine, or we use the trades_df assuming X follows valid_trades order
    # X and y were built iterating `valid_trades`. So split_idx in X corresponds to split_idx in valid_trades.
    
    valid_trades = trades_df[trades_df['outcome'].isin(['WIN', 'LOSS'])]
    split_time = valid_trades.iloc[split_idx]['entry_time']
    
    logger.info(f"Split Index: {split_idx} | Split Time: {split_time}")
    logger.info(f"Train Set: {len(X_train)} | Test Set: {len(X_test)}")
    
    dataset = TradeDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # Shuffle TRAIN only
    
    model = TradeCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    logger.info("Training PyTorch CNN on GPU...")
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f} - Acc: {correct/total:.2f}")
    
    # Save
    model_path = MODELS_DIR / "setup_cnn_v1.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Evaluate on Test Set (One pass)
    test_ds = TradeDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=32)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    logger.info(f"Test Set Accuracy (First Pass): {correct/total:.2f}")

if __name__ == "__main__":
    train()
