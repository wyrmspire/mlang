# CNN Pattern Detection Pipeline - Final Report

## Overview

This pipeline uses a CNN to detect price action patterns and generate trade signals with configurable entry/exit parameters.

---

## Key Learnings (From Building This)

### 1. Normalization Matters
- **Problem**: CNN output constant values (~0.27) regardless of input
- **Cause**: Percentage normalization `(x/base)-1` yields tiny values for 1m data
- **Fix**: Z-Score per window: `(x - mean) / std`

### 2. Train Both Directions
- Pattern = price moves X, then returns
- **SHORT pattern**: Price UP then returns (short the top)
- **LONG pattern**: Price DOWN then returns (buy the bottom)
- Training only SHORT caused bias. Fixed by mining both.

### 3. Use Higher Timeframe ATR for Stops
| ATR Source | Avg Stop | Contracts for $300 risk |
|------------|----------|------------------------|
| 1m | 1.35 pts | 44 (unrealistic) |
| 5m | 3.13 pts | 19 |
| **15m** | **5.48 pts** | **10 (realistic)** |

### 4. Tick Alignment Required
- MES tick = 0.25
- MNQ tick = 0.25
- Gold tick = 0.10
- All entries/stops/TPs must align to tick size

### 5. CNN = Pattern Detector, Not Predictor
- CNN recognizes pre-pattern price action
- Trade direction/TP determined by testing OCO configurations
- Best config on MES: LONG limit 0.5 ATR above, TP 1.4x

---

## Strategy Configuration

```
Entry:     LONG limit order 0.5 ATR(15m) above trigger price
Stop:      1 ATR(15m) below entry (tick-aligned)
TP:        1.4x risk distance
Risk:      $300 max per trade
Threshold: CNN prob > 0.15
Cooldown:  15 bars (15 min) between trades
```

---

## MES Results (Dec 5-10, 2025)

| Metric | Value |
|--------|-------|
| Trades | 14 |
| Win Rate | 57% |
| Net PnL | **+$2,078** |
| Max Drawdown | $581 |
| Avg Duration | 66 min |
| Avg Contracts | 14 |

### By Hour (UTC)
- Best: 00:00-01:00, 05:00-06:00, 18:00-19:00 (100% WR)
- Worst: 10:00-11:00 (0% WR)

---

## Multi-Asset Test Results

| Asset | Trades | WR | Net PnL | Notes |
|-------|--------|-----|---------|-------|
| **MES** | 14 | 57% | +$2,078 | Training asset |
| MNQ | 11 | 40% | -$62 | Near breakeven |
| Gold | 24 | 17% | -$3,554 | Does not transfer |

**Conclusion**: Model trained on MES does not transfer to other assets. Each asset needs its own trained model.

---

## Files

- `src/sweep/pattern_miner_v2.py` - Bidirectional pattern mining
- `src/sweep/train_sweep.py` - Z-Score normalized training
- `src/sweep/continuous_scanner.py` - Live scanner
- `models/sweep_CNN_Classic_v3_bidirectional.pth` - Trained model

---

## Recommended Next Steps

1. **More OCO Testing** - Sweep limit entry offsets (0.25 to 1.0 ATR)
2. **Time Filters** - Avoid 10:00-11:00 UTC based on results
3. **Asset-Specific Models** - Train separate models for MNQ/Gold
4. **Walk-Forward Validation** - Test on random historical weeks
5. **Slippage/Commission** - Add realistic costs

---

## Technical Details

### Position Sizing
```python
risk_per_contract = atr_15m * point_value  # e.g., 5.48 * $5 = $27.40/contract
contracts = min(max_contracts, int(max_risk / risk_per_contract))
```

### Entry Logic
1. CNN scans every 5 bars
2. If prob > 0.15, trigger signal
3. Place LONG limit order at `price + 0.5 * ATR(15m)`
4. Wait for fill (up to 200 bars)
5. Set stop at `entry - ATR(15m)`, TP at `entry + 1.4 * risk_dist`
