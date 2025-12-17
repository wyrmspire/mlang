# Best CNN Strategy Configuration

## Optimal Setup: LONG_0.5_2.0R_5m

### Entry Rules
1. **CNN Trigger**: Probability > 0.15 on 20-bar 1m window (Z-Score normalized)
2. **Wait**: 5 bars after trigger for 5m candle to close
3. **Limit Order**: **BELOW** close by 0.5 × ATR(5m,14)
4. **Direction**: LONG only

### Trade Management
- **Stop**: 1 ATR(5m) below entry
- **Take Profit**: 2R (2× risk distance)
- **Position Size**: $300 risk / (stop_dist × $5)

### Performance (Full MES Data - 179,587 bars)
| Metric | Value |
|--------|-------|
| Triggers | 368 |
| Filled Trades | 327 |
| Win Rate | 39.4% |
| **Net PnL** | **+$17,340** |

### Why It Works
- 39% WR seems low, but 2R reward means:
  - 39 wins × 2R = 78R
  - 61 losses × 1R = -61R
  - Net = +17R per 100 trades

### Filter Analysis (no filters improve EV)
All filter subsets profitable - filtering reduces total trades without improving expectancy.

### Tick Alignment
- MES: 0.25
- MNQ: 0.25
- Gold: 0.10

### Files
- Model: `models/sweep_CNN_Classic_v3_bidirectional.pth`
- Supersweep: `src/sweep/supersweep.py`
- Results: `results/supersweep_results.parquet`
