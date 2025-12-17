# CNN Sweep Pipeline - Complete Guide

## What This Is

A CNN-based pattern detection system that:
1. Scans 1-minute candles for patterns
2. Triggers trades when CNN probability > threshold
3. Places limit orders with configurable entries/exits
4. Tests 96 different configurations to find optimal settings

---

## How We Got Here (Evolution of the Strategy)

### Phase 1: Initial Broken State
- CNN outputting constant ~0.27 probability for all inputs
- **Root cause**: Using percentage normalization `(x/base)-1` which produces tiny values
- **Fix**: Z-Score normalization `(x - mean) / std` per window

### Phase 2: Directional Bias
- Model only trained on SHORT patterns (price up → return)
- All results showed SHORT bias
- **Fix**: Updated `pattern_miner_v2.py` to detect BOTH:
  - SHORT: price rises then returns
  - LONG: price drops then returns

### Phase 3: Stops Too Tight  
- 0.5 ATR stops on 1m data caused both SL and TP to hit same bar
- Win rates impossibly low (<50% on 1:1 R/R)
- **Fix**: Use higher timeframe ATR (5m or 15m) for stops

### Phase 4: Wrong Limit Order Direction
- Initially placed LONG limits ABOVE close (breakout entry)
- Results were mediocre
- **Fix**: LONG limits BELOW close (pullback entry), SHORT ABOVE
- This **dramatically improved results** (+$17k vs +$6k)

### Phase 5: Entry Bar Confusion
- Tested various entry bar timings
- Originally used trigger bar directly (potential look-ahead)
- Then tested previous bar
- **Final**: Wait 5 bars for 5m ATR, 15 bars for 15m ATR

### Phase 6: Partial Exits
- Tested HALF at 1R, rest rides to 2R/3R with BE stop
- Result: BE stop gets hit too often, simple exit outperforms

### Phase 7: Sensitivity & Risk Tuning
- Tested lower thresholds (down to 0.066) to increase frequency
- **Result**: Lower thresholds increase PnL but drastically increase Drawdown
- **Recommendation**: Stick to **0.15 threshold** for safety (lowest DD)
- **Tuning**: Better to increase Risk per Trade on high-quality setup (0.15) than to lower quality bar (0.066)

| Threshold | Trades | WR | Net PnL | Max DD |
|-----------|--------|----|---------|--------|
| **0.15** (Default) | 328 | 57% | +$14,635 | **$2,859** |
| 0.10 | 833 | 55% | +$24,095 | $4,638 |
| 0.066 | 1753 | 54% | **+$50,526** | $7,712 |

*Note: 0.15 gives best risk-adjusted returns. To scale, increase position size, not sensitivity.*

---

## Current Best Config

### `LONG_0.25_1.0R_5m_SIMPLE`
| Parameter | Value |
|-----------|-------|
| Direction | LONG only |
| Entry | Limit 0.25 ATR(5m) **BELOW** close |
| Wait | 5 bars after CNN trigger |
| Stop | 1 ATR(5m) below entry |
| TP | 1R (1:1) |
| Risk | $300 per trade |

### Why This Works
- Pullback entry (limit below) catches retracements
- 1:1 R/R with 67% WR = consistent profits
- 5m ATR gives realistic stop distances (3-5 points)
- Low drawdown ($580 vs $1,768 on 2R targets)

---

## Quick Start

### Download Fresh Data
```python
import yfinance as yf
from datetime import datetime, timedelta

# MES (use ES=F as proxy)
df = yf.Ticker('ES=F').history(
    start=datetime.now() - timedelta(days=7),
    end=datetime.now(),
    interval='1m'
)
df.to_parquet('data/processed/fresh_5day_1m.parquet')

# MNQ (use NQ=F)
yf.Ticker('NQ=F').history(...).to_parquet('fresh_5day_mnq.parquet')

# Gold (use GC=F)
yf.Ticker('GC=F').history(...).to_parquet('fresh_5day_gold.parquet')
```

---

## Best Config Found

### `LONG_0.25_1.0R_5m_SIMPLE`
| Parameter | Value |
|-----------|-------|
| Direction | LONG only |
| Entry | Limit 0.25 ATR(5m) **BELOW** close |
| Wait | 5 bars after CNN trigger |
| Stop | 1 ATR(5m) below entry |
| TP | 1R (1:1) |
| Risk | $300 per trade |

### Performance
| Dataset | Trades | WR | PnL | MaxDD |
|---------|--------|-----|-----|-------|
| MES Historical | 351 | 58% | +$16,538 | - |
| MES Fresh 5-day | 15 | 67% | +$1,512 | $580 |
| MNQ Fresh | 13 | 23% | -$1,134 | - |
| Gold Fresh | 30 | 20% | -$3,127 | - |

**Model only works on MES** - each asset needs separate training.

---

## All Configs Tested (96 Total)

### Parameters Swept
- **Direction**: LONG, SHORT
- **Entry Offset**: 0.25, 0.5, 0.75, 1.0 ATR (below for LONG, above for SHORT)
- **TP**: 1.0R, 1.4R, 2.0R
- **ATR Timeframe**: 5m, 15m
- **Exit Mode**: SIMPLE, HALF_1R_REST_2R

### Top 5 on Historical MES
| Config | WR | PnL |
|--------|-----|-----|
| LONG_0.5_2.0R_5m | 39% | +$17,340 |
| LONG_0.25_1.0R_5m | 58% | +$16,538 |
| LONG_0.5_1.4R_5m | 48% | +$14,304 |
| LONG_1.0_1.0R_5m | 58% | +$14,219 |
| LONG_0.25_1.4R_5m | 47% | +$13,622 |

---

## Key Lessons Learned

1. **Normalization**: Z-Score `(x-mean)/std` per window, not percentage
2. **Train Both Directions**: Bidirectional pattern mining (LONG + SHORT)
3. **Limit Order Direction**: LONG limit BELOW close (pullback), SHORT ABOVE
4. **ATR Timeframe**: 5m ATR beats 15m for stops
5. **Tick Alignment**: MES/MNQ=0.25, Gold=0.10
6. **HALF+Runner underperforms**: BE stop gets hit before runner target
7. **Model Asset-Specific**: MES model doesn't transfer to MNQ/Gold

---

## Filter Analysis (None Help)

All filters profitable - filtering reduces trades without improving EV:
- above_pdh=True: 63% WR
- above_vwap: No improvement
- Best hours: 09, 14, 16, 22
- Best days: Mon, Wed, Thu

---

## Files Created

| File | Purpose |
|------|---------|
| `src/sweep/supersweep.py` | 96-config sweep engine |
| `src/sweep/pattern_miner_v2.py` | Bidirectional pattern detection |
| `src/sweep/train_sweep.py` | Z-Score normalized training |
| `models/sweep_CNN_Classic_v3_bidirectional.pth` | Trained model |
| `results/supersweep_results.parquet` | All trade records with filters |
| `docs/best_config.md` | Best config summary |
| `.agent/workflows/sweep_lessons_learned.md` | Detailed lessons |

---

## Running Supersweep

```bash
python src/sweep/supersweep.py --output results/supersweep_results.parquet
```

Options:
- `--risk 300` - Risk per trade
- `--threshold 0.15` - CNN probability threshold
- `--model models/sweep_CNN_Classic_v3_bidirectional.pth`

---

## What Still Needs Work

### Dynamic Risk Sizing (Not Implemented)
- $50k account, $2k max DD
- Base risk $300 (0.6%)
- Increase in $50 increments as account grows
- Fail at $48k balance

### Account Simulation Script (Needed)
- Track running balance
- Calculate drawdown
- Position sizing based on account size
- Win streaks / losing streaks analysis

### Multi-Asset Training
- Current model only works on MES
- Need to retrain on MNQ data
- Need to retrain on Gold data

### Filter Optimization
- Current filters don't improve EV (all subsets profitable)
- May need more granular time-of-day analysis
- Consider session-based filters (RTH vs ETH)

---

## Important Caveats

1. **Historical ≠ Future**: Best config on historical (+$17k) was different from fresh data (+$1.5k)
2. **Slippage not modeled**: Real fills may differ from limit price
3. **Commission not included**: ~$2.50 per MES round trip
4. **Overnight holds**: Some trades may hold overnight (not modeled)
5. **Consolidating markets**: Strategy may underperform in low-volatility periods

---

## Terminology Reference

| Term | Meaning |
|------|---------|
| ATR | Average True Range (volatility measure) |
| R | Risk unit (1R = stop distance) |
| TP | Take Profit |
| WR | Win Rate |
| DD | Drawdown |
| PDH/PDL | Previous Day High/Low |
| ONH/ONL | Overnight High/Low |
| PDC | Previous Day Close |
| VWAP | Volume Weighted Average Price |
| EMA | Exponential Moving Average |
