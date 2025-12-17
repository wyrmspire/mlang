"""
Train CNN and Test Model-Filtered Trades
Train on 6 weeks, test on week 7 with model predictions.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper
from src.sim.oco import OCOConfig
from src.features.pipeline import compute_features, FeatureConfig
from src.features.indicators import calculate_atr
from src.labels.counterfactual import compute_counterfactual
from src.sim.bar_fill_model import BarFillConfig
from src.sim.costs import DEFAULT_COSTS
from src.models.fusion import SimpleCNN

print("=" * 60)
print("MLang2 CNN Training + Model-Filtered Testing")
print("GPU:", "CUDA" if torch.cuda.is_available() else "CPU")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 1. Load and prepare data
# ============================================================================
print("\n[1] Loading data...")
df = load_continuous_contract()

train_start = "2025-03-17"
train_end = "2025-04-27"
test_start = "2025-04-28"
test_end = "2025-05-04"

df_train = df[(df['time'] >= train_start) & (df['time'] < train_end)].reset_index(drop=True)
df_test = df[(df['time'] >= test_start) & (df['time'] < test_end)].reset_index(drop=True)

print(f"Train: {len(df_train)} bars")
print(f"Test: {len(df_test)} bars")

# Resample
htf_train = resample_all_timeframes(df_train)
htf_test = resample_all_timeframes(df_test)

# Compute ATR
df_5m_train = htf_train['5m'].copy()
df_5m_train['atr'] = calculate_atr(df_5m_train, 14)
avg_atr = df_5m_train['atr'].dropna().mean()
print(f"Average 5m ATR: {avg_atr:.2f}")

# OCO config
oco = OCOConfig(direction="LONG", entry_type="MARKET", stop_atr=1.0, tp_multiple=1.4, max_bars=200)
fill_config = BarFillConfig()

# ============================================================================
# 2. Generate training data with price windows
# ============================================================================
print("\n[2] Generating training data with price windows...")

LOOKBACK = 120  # 2 hours of 1m bars

def get_price_window(df, idx, lookback=LOOKBACK):
    """Get normalized price window for CNN input."""
    start = max(0, idx - lookback)
    window = df.iloc[start:idx][['open', 'high', 'low', 'close']].values
    
    # Pad if needed
    if len(window) < lookback:
        pad = np.zeros((lookback - len(window), 4))
        window = np.vstack([pad, window])
    
    # Z-score normalize
    mean = window.mean()
    std = window.std()
    if std < 1e-8:
        std = 1.0
    window = (window - mean) / std
    
    return window.astype(np.float32)


train_samples = []
stepper = MarketStepper(df_train, start_idx=LOOKBACK + 10, end_idx=len(df_train) - 200)

sample_count = 0
while True:
    step = stepper.step()
    if step.is_done:
        break
    
    # Every hour
    if step.bar_idx % 60 != 0:
        continue
    
    # Get price window
    x = get_price_window(df_train, step.bar_idx)
    
    # Get label
    cf = compute_counterfactual(
        df=df_train, entry_idx=step.bar_idx, oco_config=oco,
        atr=avg_atr, fill_config=fill_config, costs=DEFAULT_COSTS, max_bars=200
    )
    
    # Skip timeouts for binary classification
    if cf.outcome == 'TIMEOUT':
        continue
    
    y = 1 if cf.outcome == 'WIN' else 0
    train_samples.append({'x': x, 'y': y, 'pnl': cf.pnl_dollars})
    sample_count += 1

print(f"Generated {len(train_samples)} training samples (excluding timeouts)")
wins = sum(1 for s in train_samples if s['y'] == 1)
print(f"Class balance: {wins} WIN ({wins/len(train_samples):.1%}), {len(train_samples)-wins} LOSS")


# ============================================================================
# 3. Create PyTorch Dataset and train CNN
# ============================================================================
print("\n[3] Training CNN...")

class TradeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        # (channels, length) for Conv1d
        x = torch.FloatTensor(s['x'].T)  # (4, 120)
        y = torch.LongTensor([s['y']])
        return x, y

# Split train/val
dataset = TradeDataset(train_samples)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Model
model = SimpleCNN(input_channels=4, seq_length=LOOKBACK, num_classes=2, dropout=0.3).to(device)

# Class weights for imbalance
loss_weights = torch.FloatTensor([1.0, len(train_samples) / wins - 1]).to(device)
criterion = nn.CrossEntropyLoss(weight=loss_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
best_val_loss = float('inf')
epochs = 15

for epoch in range(epochs):
    # Train
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.squeeze().to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.squeeze().to(device)
            out = model(x)
            val_loss += criterion(out, y).item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    val_acc = correct / total
    print(f"Epoch {epoch+1:2d} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.1%}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/cnn_filter.pth')

print("\nBest model saved to models/cnn_filter.pth")


# ============================================================================
# 4. Test on Week 7 with model filtering
# ============================================================================
print("\n[4] Testing on Week 7 with model filtering...")

model.load_state_dict(torch.load('models/cnn_filter.pth'))
model.eval()

test_results = []
stepper_test = MarketStepper(df_test, start_idx=LOOKBACK + 10, end_idx=len(df_test) - 100)

while True:
    step = stepper_test.step()
    if step.is_done:
        break
    
    if step.bar_idx % 60 != 0:
        continue
    
    # Get price window
    x = get_price_window(df_test, step.bar_idx)
    
    # Get model prediction
    x_tensor = torch.FloatTensor(x.T).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        p_win = probs[0, 1].item()
    
    # Get actual outcome
    cf = compute_counterfactual(
        df=df_test, entry_idx=step.bar_idx, oco_config=oco,
        atr=avg_atr, fill_config=fill_config, costs=DEFAULT_COSTS, max_bars=200
    )
    
    test_results.append({
        'p_win': p_win,
        'outcome': cf.outcome,
        'pnl': cf.pnl_dollars,
    })

test_df = pd.DataFrame(test_results)
print(f"\nTotal Week 7 opportunities: {len(test_df)}")

# ============================================================================
# 5. Compare results with different thresholds
# ============================================================================
print("\n[5] Results by threshold:")
print("-" * 60)
print(f"{'Threshold':<12} {'Trades':<8} {'Wins':<6} {'Win Rate':<10} {'PnL':<12} {'Avg PnL'}")
print("-" * 60)

# Baseline (take all)
for thresh in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:
    filtered = test_df[test_df['p_win'] >= thresh]
    trades = len(filtered)
    if trades == 0:
        continue
    wins = (filtered['outcome'] == 'WIN').sum()
    losses = (filtered['outcome'] == 'LOSS').sum()
    wr = wins / (wins + losses) if (wins + losses) > 0 else 0
    pnl = filtered['pnl'].sum()
    avg_pnl = pnl / trades
    print(f">= {thresh:<9.1f} {trades:<8} {wins:<6} {wr:<10.1%} ${pnl:<11.2f} ${avg_pnl:.2f}")

print("-" * 60)
print("\n✅ KEY: If higher thresholds improve WR and PnL, the model is working!")
print("=" * 60)
