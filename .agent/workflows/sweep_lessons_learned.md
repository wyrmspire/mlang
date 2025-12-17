# Sweep Pipeline - Lessons Learned

## Overview

This document captures the challenges, mistakes, and solutions discovered while building the CNN-based pattern detection and trade simulation pipeline.

---

## Key Mistakes and Solutions

### 1. Model Collapse (Constant Output)

**Problem:** CNN output constant probability (~0.27) for all inputs.

**Root Cause:** Using percentage normalization `(x / base) - 1` which produces tiny values for 1m price data.

**Solution:** Z-Score normalization per window from `success_study.md`:
```python
mean = np.mean(feats)
std = np.std(feats)
feats_norm = (feats - mean) / std
```

---

### 2. Single Direction Training Bias

**Problem:** Model trained only on SHORT patterns (price UP → return), causing inherent SHORT bias in results.

**Solution:** Updated `pattern_miner_v2.py` to detect BOTH:
- SHORT: Price rises, then returns (short the reversal)
- LONG: Price drops, then returns (long the reversal)

---

### 3. Stops Too Tight for 1m Data

**Problem:** Using 0.5 ATR stops caused both LONG and SHORT to show <50% win rate (mathematically impossible).

**Root Cause:** 1m bar ranges often exceed 0.5 ATR, causing both stops to hit on same bar.

**Solution:** Use 1.0 ATR for stop distance.

---

### 4. Confusing R-Multiples with ATR

**Terminology Clarification:**
- **ATR** = Average True Range (volatility measure in points)
- **Stop distance** = Usually 1 ATR from entry (in points)
- **R-Multiple** = Take Profit as multiple of risk (1.4R = TP at 1.4× stop distance)

**Example (MES at 6000, ATR = 4 points):**
- Stop = 6000 - 4 = 5996 (1 ATR below for LONG)
- TP at 1.4R = 6000 + (4 × 1.4) = 6005.6

---

### 5. CNN Role Misunderstanding

**Wrong thinking:** CNN predicts WIN/LOSS outcome.

**Correct thinking:** CNN is a PATTERN DETECTOR:
1. CNN triggers when it recognizes price action shape
2. On trigger, test ALL OCO configurations
3. Simulations determine which direction/TP works best

---

### 6. Position Sizing for Futures

**MES Contract Math:**
- 1 point = $5 per contract
- To risk $75 with a 3-point stop: $75 / ($5 × 3) = 5 contracts

**Simplification for simulation:** Assume fractional contracts so risk = exactly $75 per trade.

---

## Working Pipeline

1. **Pattern Miner V2** (`pattern_miner_v2.py`)
   - Detects bidirectional patterns (SHORT + LONG)
   - Uses proportional ratios, not dollar amounts

2. **Training** (`train_sweep.py`)
   - Z-Score normalization per window
   - Direction inversion for unified dataset
   - GPU-accelerated

3. **Scanner** (`continuous_scanner.py`)
   - Scans every 5m candle
   - CNN detection with threshold
   - Tests all OCO configs per trigger

4. **Best Config Found:**
   - LONG limit order 0.5 ATR above trigger price
   - Stop: 1 ATR, TP: 1:1
   - 75% win rate on 5-day live data

---

## Files Modified

- `src/sweep/pattern_miner_v2.py` - Bidirectional pattern detection
- `src/sweep/train_sweep.py` - Z-Score normalization
- `src/sweep/continuous_scanner.py` - Full OCO testing

## Model Path

`models/sweep_CNN_Classic_v3_bidirectional.pth`
