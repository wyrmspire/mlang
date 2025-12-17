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

```

### test_walkforward.py

```python
"""
Test Script - 6 Week Train, Week 7 Test
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper
from src.sim.oco import OCOConfig, create_oco_bracket
from src.features.pipeline import compute_features, FeatureConfig
from src.features.indicators import calculate_atr
from src.policy.scanners import IntervalScanner
from src.labels.counterfactual import compute_counterfactual
from src.sim.bar_fill_model import BarFillConfig
from src.sim.costs import DEFAULT_COSTS

print("=" * 60)
print("MLang2 Walk-Forward Test")
print("Train: 6 weeks, Test: Week 7")
print("=" * 60)

# Load data
print("\n[1] Loading data...")
df = load_continuous_contract()
print(f"Total bars: {len(df)}")

# Set dates
train_start = "2025-03-17"
train_end = "2025-04-27"
test_start = "2025-04-28"
test_end = "2025-05-04"

df_train = df[(df['time'] >= train_start) & (df['time'] < train_end)].reset_index(drop=True)
df_test = df[(df['time'] >= test_start) & (df['time'] < test_end)].reset_index(drop=True)

print(f"Train: {len(df_train)} bars ({train_start} to {train_end})")
print(f"Test: {len(df_test)} bars ({test_start} to {test_end})")

# Resample
print("\n[2] Resampling to higher timeframes...")
htf_train = resample_all_timeframes(df_train)
htf_test = resample_all_timeframes(df_test)
print(f"5m bars (train): {len(htf_train['5m'])}")

# Pre-compute ATR on train data
print("\n[3] Computing ATR...")
df_5m_train = htf_train['5m'].copy()
df_5m_train['atr'] = calculate_atr(df_5m_train, 14)
avg_atr = df_5m_train['atr'].dropna().mean()
print(f"Average 5m ATR: {avg_atr:.2f} points")

# OCO config
oco = OCOConfig(
    direction="LONG",
    entry_type="MARKET",
    stop_atr=1.0,
    tp_multiple=1.4,
    max_bars=200,
)
print(f"\n[4] OCO Config: {oco.direction}, {oco.tp_multiple}R, stop={oco.stop_atr}ATR")

# Generate training decision points
print("\n[5] Generating training decision points...")
scanner = IntervalScanner(interval=60)  # Every 60 bars = every hour
fill_config = BarFillConfig()

train_records = []
stepper = MarketStepper(df_train, start_idx=200, end_idx=len(df_train) - 200)

while True:
    step = stepper.step()
    if step.is_done:
        break
    
    # Simple interval trigger
    if step.bar_idx % 60 != 0:
        continue
    
    # Use fixed ATR for simplicity
    atr = avg_atr
    
    # Compute counterfactual label
    cf = compute_counterfactual(
        df=df_train,
        entry_idx=step.bar_idx,
        oco_config=oco,
        atr=atr,
        fill_config=fill_config,
        costs=DEFAULT_COSTS,
        max_bars=200
    )
    
    train_records.append({
        'bar_idx': step.bar_idx,
        'outcome': cf.outcome,
        'pnl': cf.pnl,
        'pnl_dollars': cf.pnl_dollars,
        'mae': cf.mae,
        'mfe': cf.mfe,
        'bars_held': cf.bars_held,
    })

print(f"Generated {len(train_records)} training decision points")

# Analyze training outcomes
train_df = pd.DataFrame(train_records)
wins = (train_df['outcome'] == 'WIN').sum()
losses = (train_df['outcome'] == 'LOSS').sum()
timeouts = (train_df['outcome'] == 'TIMEOUT').sum()
total_pnl = train_df['pnl_dollars'].sum()

print(f"\n[6] Training Outcomes:")
print(f"  Wins: {wins} ({wins/len(train_df):.1%})")
print(f"  Losses: {losses} ({losses/len(train_df):.1%})")
print(f"  Timeouts: {timeouts} ({timeouts/len(train_df):.1%})")
print(f"  Total PnL: ${total_pnl:.2f}")

# Now test on week 7
print("\n[7] Testing on Week 7...")
test_records = []
stepper_test = MarketStepper(df_test, start_idx=100, end_idx=len(df_test) - 100)

while True:
    step = stepper_test.step()
    if step.is_done:
        break
    
    if step.bar_idx % 60 != 0:
        continue
    
    cf = compute_counterfactual(
        df=df_test,
        entry_idx=step.bar_idx,
        oco_config=oco,
        atr=avg_atr,
        fill_config=fill_config,
        costs=DEFAULT_COSTS,
        max_bars=200
    )
    
    test_records.append({
        'bar_idx': step.bar_idx,
        'outcome': cf.outcome,
        'pnl': cf.pnl,
        'pnl_dollars': cf.pnl_dollars,
    })

test_df = pd.DataFrame(test_records)
test_wins = (test_df['outcome'] == 'WIN').sum()
test_losses = (test_df['outcome'] == 'LOSS').sum()
test_pnl = test_df['pnl_dollars'].sum()

print(f"\n[8] Test Week 7 Results:")
print(f"  Total trades: {len(test_df)}")
print(f"  Wins: {test_wins} ({test_wins/len(test_df):.1%})")
print(f"  Losses: {test_losses} ({test_losses/len(test_df):.1%})")
print(f"  Total PnL: ${test_pnl:.2f}")
print(f"  Avg PnL per trade: ${test_pnl/len(test_df):.2f}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)

```

### verify_modular_strategies.py

```python
"""
Verification script for Modular Strategy Discovery.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.policy.scanners import get_scanner

def test_modular_discovery():
    print("--- Testing Modular Discovery ---")
    try:
        # Try to get the mid_day_reversal scanner
        scanner = get_scanner("midday_reversal", start_hour=11, end_hour=13)
        print(f"Successfully instantiated: {scanner.scanner_id}")
        
        # Check if always scanner still works
        always = get_scanner("always")
        print(f"Successfully instantiated: {always.scanner_id}")
        
    except Exception as e:
        print(f"Error: {e}")
    print("\n")

if __name__ == "__main__":
    test_modular_discovery()

```

### verify_skills.py

```python
"""
Verification script for mlang2 Agent Skills.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.skills.registry import list_available_skills, registry

def test_discovery():
    print("--- Testing Skill Discovery ---")
    print(list_available_skills())
    print("\n")

def test_data_skill():
    print("--- Testing Data Skill (Summary) ---")
    summary = registry.get_skill("get_data_summary")()
    print(summary)
    print("\n")

if __name__ == "__main__":
    test_discovery()
    test_data_skill()

```

