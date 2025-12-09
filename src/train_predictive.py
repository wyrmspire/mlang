
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import copy

sys.path.append(str(Path(__file__).parent.parent))
from src.config import PROCESSED_DIR, MODELS_DIR

# Define Model Architecture
class CNN_Predictive(nn.Module):
    def __init__(self, input_dim=4, seq_len=20):
        super(CNN_Predictive, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (seq_len // 2), 64)
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Time, Feature) -> Permute to (Feature, Time) for Conv1d
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.relu(x)
        # No pool after first
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) # (20 -> 10)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

class PredictiveDataset(Dataset):
    def __init__(self, indices_df, full_df, lookback=20):
        self.indices = indices_df
        self.full_df = full_df
        self.lookback = lookback
        
        # Optimize access
        self.opens = full_df['open'].values.astype(np.float32)
        self.highs = full_df['high'].values.astype(np.float32)
        self.lows = full_df['low'].values.astype(np.float32)
        self.closes = full_df['close'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        row = self.indices.iloc[idx]
        target_idx = int(row['index'])
        label = float(row['label'])
        
        start_idx = target_idx - self.lookback
        
        # Features
        o = self.opens[start_idx:target_idx]
        h = self.highs[start_idx:target_idx]
        l = self.lows[start_idx:target_idx]
        c = self.closes[start_idx:target_idx]
        
        # Normalize
        # Simple Z-score relative to window mean
        block = np.stack([o, h, l, c], axis=1)
        mean = block.mean()
        std = block.std()
        if std == 0: std = 1e-6
        block = (block - mean) / std
        
        return torch.tensor(block, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train_predictive():
    print("Loading data...")
    PROCESSED_DATA_FILE = PROCESSED_DIR / "continuous_1m.parquet"
    full_df = pd.read_parquet(PROCESSED_DATA_FILE).sort_index()
    
    labels_path = PROCESSED_DIR / "labeled_predictive.parquet"
    labels_df = pd.read_parquet(labels_path)
    
    # Split
    split_idx = int(len(labels_df) * 0.8)
    train_df = labels_df.iloc[:split_idx]
    val_df = labels_df.iloc[split_idx:]
    
    train_ds = PredictiveDataset(train_df, full_df)
    val_ds = PredictiveDataset(val_df, full_df)
    
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model = CNN_Predictive().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        sum_loss = 0
        correct = 0
        total = 0
        
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.item()
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
        train_acc = correct / total
        train_loss = sum_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                pred = model(X)
                loss = criterion(pred, y)
                val_loss += loss.item()
                predicted = (pred > 0.5).float()
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)
        
        val_epoch_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_epoch_loss:.4f} Val Acc {val_acc:.4f}")
        
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), MODELS_DIR / "CNN_Predictive.pth")
            print(f"Saved Best Model (Loss {best_loss:.4f})")
            
    print("Training Complete.")

if __name__ == "__main__":
    train_predictive()
