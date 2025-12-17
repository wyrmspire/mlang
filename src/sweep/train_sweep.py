"""
CLI Training Script for Sweep Pipeline
Trains models with different architectures and candle compositions.

Usage:
    python src/sweep/train_sweep.py \
        --architecture CNN_Classic \
        --input-data labeled_sweep_001.parquet \
        --candles-1m 30 --candles-3m 20 --candles-5m 10 \
        --epochs 10 --lr 0.001 \
        --output-suffix "cnn_001"
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, ONE_MIN_PARQUET_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger
from src.models.variants import CNN_Classic, CNN_Wide, LSTM_Seq, Feature_MLP

logger = get_logger("train_sweep")

# Enforce GPU
if not torch.cuda.is_available():
    logger.error("GPU NOT DETECTED! This script requires CUDA.")
    sys.exit(1)

device = torch.device("cuda")
logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")


class TradeDataset(Dataset):
    """Dataset for trade pattern training."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Model Training with CLI parameters")
    
    # Model architecture
    parser.add_argument("--architecture", type=str, default="CNN_Classic",
                        choices=["CNN_Classic", "CNN_Wide", "LSTM_Seq", "Feature_MLP"],
                        help="Model architecture to use")
    
    # Input data
    parser.add_argument("--input-data", type=str, required=True,
                        help="Path to labeled pattern data (parquet)")
    
    # Candle composition
    parser.add_argument("--candles-1m", type=int, default=30)
    parser.add_argument("--candles-3m", type=int, default=0)
    parser.add_argument("--candles-5m", type=int, default=0)
    parser.add_argument("--candles-15m", type=int, default=0)
    
    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Split ratios
    parser.add_argument("--train-ratio", type=float, default=0.56)
    parser.add_argument("--val-ratio", type=float, default=0.14)
    
    # OCO config for labeling (use outcome column from sweep)
    parser.add_argument("--oco-config", type=str, default="SHORT_2.0R_50ATR",
                        help="OCO config to use for outcome labels (e.g. SHORT_2.0R_50ATR)")
    
    # Output
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    
    return parser.parse_args()


def prepare_data(
    patterns: pd.DataFrame,
    df_1m: pd.DataFrame,
    window_size: int = 30,
    oco_config: str = "SHORT_2.0R_50ATR",
) -> tuple:
    """
    Prepare training data from patterns.
    Uses specified OCO config for outcome labels.
    
    Returns:
        X, y, timestamps
    """
    X = []
    y = []
    timestamps = []
    
    # Determine outcome column
    outcome_col = f"outcome_{oco_config}"
    if outcome_col not in patterns.columns:
        # Fallback to legacy 'outcome' column
        if 'outcome' in patterns.columns:
            outcome_col = 'outcome'
        else:
            logger.error(f"Outcome column not found: {outcome_col}")
            return np.array([]), np.array([]), []
    
    valid_patterns = patterns[patterns[outcome_col].isin(['WIN', 'LOSS'])].copy()
    valid_patterns = valid_patterns.sort_values('trigger_time')
    
    logger.info(f"Processing {len(valid_patterns)} valid patterns using {outcome_col}...")
    
    # Ensure index is UTC
    if df_1m.index.tz is None:
        df_1m.index = df_1m.index.tz_localize('UTC')
    else:
        df_1m.index = df_1m.index.tz_convert('UTC')
        
    for idx, pattern in valid_patterns.iterrows():
        trigger_time = pattern['trigger_time']
        
        # Ensure trigger_time is UTC
        if pd.Timestamp(trigger_time).tz is None:
            trigger_time = pd.Timestamp(trigger_time).tz_localize('UTC')
        else:
            trigger_time = pd.Timestamp(trigger_time).tz_convert('UTC')
        
        # Get window before trigger
        end_time = trigger_time
        start_time = end_time - pd.Timedelta(minutes=window_size)
        
        try:
            window = df_1m.loc[start_time:end_time]
        except KeyError:
            continue
            
        window = window[window.index < end_time]
        
        if len(window) < window_size:
            continue
        
        # Z-Score Normalization per window (per success_study.md)
        feats = window[['open', 'high', 'low', 'close']].values
        mean = np.mean(feats)
        std = np.std(feats)
        if std == 0:
            std = 1.0  # Prevent div/0
        
        feats_norm = (feats - mean) / std
        
        # Take last window_size bars
        if len(feats_norm) > window_size:
            feats_norm = feats_norm[-window_size:]
        elif len(feats_norm) < window_size:
            continue
        
        # Invert for SHORT direction to unify dataset (per success_study.md)
        pattern_direction = pattern.get('direction', 'SHORT')
        if pattern_direction == "SHORT":
            feats_norm = -feats_norm
        
        # Label using OCO-specific outcome
        label = 1 if pattern[outcome_col] == 'WIN' else 0
        
        X.append(feats_norm)
        y.append(label)
        timestamps.append(trigger_time)
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Prepared {len(X)} samples. Win rate: {np.mean(y):.2f}")
    
    return X, y, timestamps


def get_model(architecture: str, seq_len: int, input_dim: int = 4):
    """Get model instance by architecture name."""
    
    if architecture == "CNN_Classic":
        return CNN_Classic(input_dim=input_dim, seq_len=seq_len)
    elif architecture == "CNN_Wide":
        return CNN_Wide(input_dim=input_dim, seq_len=seq_len)
    elif architecture == "LSTM_Seq":
        return LSTM_Seq(input_dim=input_dim)
    elif architecture == "Feature_MLP":
        # MLP expects flattened features
        return Feature_MLP(input_dim=seq_len * input_dim)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
) -> dict:
    """
    Train model and return metrics.
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {train_loss/len(train_loader):.4f}, "
                    f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
    
    return {
        "best_val_acc": best_val_acc,
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
        "history": history,
    }


def main():
    args = parse_args()
    
    # Calculate window size from candle composition
    window_size = args.candles_1m  # Primary window (we'll handle multi-TF later)
    
    logger.info(f"Training {args.architecture} with {window_size} bar window")
    
    # Load pattern data
    pattern_path = Path(args.input_data)
    if not pattern_path.is_absolute():
        pattern_path = PROCESSED_DIR / args.input_data
    
    if not pattern_path.exists():
        logger.error(f"Pattern data not found: {pattern_path}")
        return {"error": "No data"}
    
    patterns = pd.read_parquet(pattern_path)
    logger.info(f"Loaded {len(patterns)} patterns")
    
    # Load 1m data
    data_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not data_path.exists():
        data_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    df_1m = pd.read_parquet(data_path)
    if 'time' in df_1m.columns:
        df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
        df_1m = df_1m.set_index('time')
    df_1m = df_1m.sort_index()
    
    # Prepare data
    X, y, timestamps = prepare_data(patterns, df_1m, window_size, args.oco_config)
    
    if len(X) < 50:
        logger.error(f"Not enough samples: {len(X)}")
        return {"error": "Insufficient data"}
    
    # Split chronologically
    n = len(X)
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + args.val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Create dataloaders
    train_ds = TradeDataset(X_train, y_train)
    val_ds = TradeDataset(X_val, y_val)
    test_ds = TradeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Get model
    model = get_model(args.architecture, window_size)
    logger.info(f"Model: {model.__class__.__name__}")
    
    # Train
    train_result = train_model(model, train_loader, val_loader, args.epochs, args.lr)
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    # Summary
    summary = {
        "architecture": args.architecture,
        "window_size": window_size,
        "candle_composition": f"{args.candles_1m}x1m+{args.candles_3m}x3m+"
                              f"{args.candles_5m}x5m+{args.candles_15m}x15m",
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "best_val_acc": train_result["best_val_acc"],
        "test_acc": test_acc,
        "train_win_rate": float(np.mean(y_train)),
        "test_win_rate": float(np.mean(y_test)),
    }
    
    logger.info("=" * 50)
    logger.info(f"Training Complete!")
    logger.info(f"Best Val Acc: {train_result['best_val_acc']:.3f}")
    logger.info(f"Test Acc: {test_acc:.3f}")
    logger.info("=" * 50)
    
    # Save model
    if not args.dry_run:
        suffix = args.output_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODELS_DIR / f"sweep_{args.architecture}_{suffix}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        summary["model_path"] = str(model_path)
        
        # Save summary
        summary_path = MODELS_DIR / f"sweep_{args.architecture}_{suffix}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
