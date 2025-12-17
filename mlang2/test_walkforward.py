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
