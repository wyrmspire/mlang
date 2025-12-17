# PyRules: CNN Pattern Detection & Sweep Methodology

This document captures the methodology for scanning market data, training CNNs, and running parameter sweeps. It is written as transferable knowledge for any agent working on this system.

---

## 1. Data Normalization (Critical)

### The Problem
Raw price data (e.g., 5800.25, 5801.50) produces tiny percentage changes when normalized as `(x/base) - 1`. These micro-values cause CNNs to output near-constant predictions (~0.27 probability for everything).

### The Solution: Z-Score Normalization
For each rolling window of N candles:
```
normalized = (price - window_mean) / window_std
```

Apply Z-Score to OHLC independently within each window. This produces values roughly in the range [-3, +3], which neural networks handle well.

**Key Insight**: Always normalize *per window*, not globally across the dataset.

---

## 2. Pattern Mining (Label Generation)

### What Makes a Tradeable Pattern
We look for "rejection" or "round-trip" patterns where:
1. Price moves away from a candle's open (extension)
2. Price returns to (or through) the original open level (rejection)

### Bidirectional Detection
Train on BOTH directions:
- **SHORT setup**: Price rises (wick up), then returns down
- **LONG setup**: Price drops (wick down), then returns up

Single-direction training creates bias and poor generalization.

### Labeling Logic (WIN/LOSS)
For each detected pattern:
1. Set **Entry** at the return-to-open level
2. Set **Stop Loss** at the pattern's extreme (high for SHORT, low for LONG)
3. Set **Take Profit** at Entry ± (Risk × R-Multiple)
4. Simulate forward: which hits first, TP or SL?
5. Label: WIN if TP hit first, LOSS otherwise

Use 1-minute granularity for outcome simulation even if patterns are detected on higher timeframes.

---

## 3. CNN Architecture

### Input Shape
- 20 time steps × 4 features (OHLC)
- Shape: `(batch, 4, 20)` for Conv1D

### Architecture
```
Conv1D(4 → 32, kernel=3) → ReLU → BatchNorm
Conv1D(32 → 64, kernel=3) → ReLU → BatchNorm  
Conv1D(64 → 128, kernel=3) → ReLU → BatchNorm
AdaptiveAvgPool1D(1)
Flatten
Linear(128 → 64) → ReLU → Dropout(0.5)
Linear(64 → 1) → Sigmoid
```

### Training Details
- Loss: Binary Cross Entropy
- Optimizer: Adam, lr=0.001
- Epochs: 50-100 with early stopping
- Train/Val split: 80/20
- Batch size: 64

### GPU Acceleration
Always use CUDA if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

---

## 4. Scanning Methodology

### Continuous Scan Approach
On every 1-minute bar:
1. Extract the previous N candles (e.g., 20)
2. Normalize using Z-Score
3. Pass through CNN
4. If probability > threshold, trigger trade signal

### Threshold Selection
| Threshold | Behavior |
|-----------|----------|
| 0.15 | Conservative: fewer trades, lower drawdown |
| 0.10 | Moderate: balance of frequency and quality |
| 0.066 | Aggressive: max trades, highest drawdown |

**Recommendation**: Start with 0.15 for safety. Scale by increasing position size, not lowering threshold.

---

## 5. Trade Execution Logic

### Limit Order Direction (Critical)
This was a major breakthrough:
- **LONG**: Place limit order BELOW current close (pullback entry)
- **SHORT**: Place limit order ABOVE current close (pullback entry)

Placing limits in the "breakout" direction (LONG above, SHORT below) significantly underperforms.

### ATR Timeframe for Stops
Use higher timeframe ATR for stop calculation:
- 5-minute ATR: More responsive, tighter stops (preferred)
- 15-minute ATR: Wider stops, fewer whipsaws

1-minute ATR creates stops too tight—both SL and TP can hit on the same bar.

### Entry Bar Timing
To avoid look-ahead bias:
- Wait 5 bars after CNN trigger for 5m ATR
- Wait 15 bars after CNN trigger for 15m ATR

This ensures ATR calculation uses only completed bars.

---

## 6. Parameter Sweep (Supersweep)

### Parameters to Test
| Parameter | Values to Test |
|-----------|----------------|
| Direction | LONG, SHORT |
| Entry Offset | 0.25, 0.5, 0.75, 1.0 × ATR |
| Take Profit | 1.0R, 1.4R, 2.0R |
| ATR Timeframe | 5m, 15m |
| Exit Mode | SIMPLE, HALF_1R_REST_2R |

### Sweep Output
For each configuration, track:
- Trade count
- Win rate
- Total PnL
- Maximum drawdown
- Per-trade records with timestamps

### Evaluation Metrics
Primary: **Risk-adjusted returns** (PnL / MaxDD ratio)
Secondary: Win rate, trade frequency

---

## 7. Exit Strategies

### Simple Exit (Recommended)
- Full position closes at TP (1R, 1.4R, or 2.0R)
- Full position closes at SL

### Partial Exit (Underperforms)
- Close HALF at 1R
- Move stop to breakeven
- Let remainder run to 2R/3R

**Why partial exits fail**: Breakeven stops get hit too often. The "runner" portion rarely reaches target. Simple exits outperform on this strategy.

---

## 8. Filter Analysis

### Filters Tested
- Above/Below PDH/PDL (Previous Day High/Low)
- Above/Below VWAP
- Time of day
- Day of week

### Key Finding
All filters produced subsets that were profitable, but no filter improved expected value per trade. Filtering reduces trade count without improving edge.

**Recommendation**: Trade all signals that meet threshold. Don't filter.

---

## 9. Multi-Asset Considerations

### Asset Specificity
Models trained on one asset do NOT transfer to others:
- MES model fails on MNQ
- MES model fails on Gold

Each asset requires separate training on its own data.

### Tick Alignment
Different assets have different tick sizes:
- MES/MNQ: 0.25
- Gold: 0.10

Round all prices (entry, SL, TP) to valid tick increments.

---

## 10. Common Pitfalls

1. **Wrong normalization**: Using percentage change instead of Z-Score
2. **Single direction bias**: Only training on SHORT or LONG patterns
3. **Wrong limit direction**: LONG limits above close (should be below)
4. **ATR timeframe too low**: 1m ATR creates too-tight stops
5. **Look-ahead bias**: Using future bars for ATR or entry calculation
6. **Overfitting to historical**: Best historical config may differ from forward performance
7. **Ignoring drawdown**: Chasing highest PnL without considering risk

---

## 11. Reality Checks

### Not Modeled
- Slippage (real fills may differ from limit price)
- Commissions (~$2.50 per MES round trip)
- Overnight holds
- Low volatility periods (strategy may underperform)

### Historical vs Forward
Best config on historical data (+$17k) was different from fresh data performance (+$1.5k). Always validate on out-of-sample data.

---

## Summary: The Winning Recipe

1. **Normalize**: Z-Score per window, not percentage
2. **Train bidirectionally**: Both LONG and SHORT patterns
3. **Place limits correctly**: LONG below close, SHORT above
4. **Use 5m ATR**: For stop calculation, not 1m
5. **Wait for bars**: Avoid look-ahead bias
6. **Keep exits simple**: No partial exits, no BE stops
7. **Don't over-filter**: Trade all threshold-meeting signals
8. **Validate forward**: Historical != future
