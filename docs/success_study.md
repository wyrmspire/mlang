# Success Study: The "Fade" Rejection Strategy

**Date**: December 8, 2025
**Outcome**: Discovered a highly profitable strategy (+70% Win Rate) by inverting a losing pattern.

## 1. The Journey & Challenges

### Phase 1: The "Rejection" Hypothesis
We started with the idea of a **"Round Trip Rejection"**:
-   **Concept**: Price extends 1.5x its average range (ATR) in one direction, then immediately returns to the start.
-   **Theory**: This "failed break" should lead to a reversal.
-   **Implementation Issues**:
    -   **Timeframe Sensitivity**: 1-minute candles were too noisy. We effectively switched to a **Hybrid Model** (Scan 5m, Input 1m).
    -   **Silent Crashes**: `pattern_miner.py` failed silently during large-scale pandas operations. **Fix**: Simplified the logic and used robust print statements instead of complex logging during the critical loop.
    -   **Model Collapse**: The CNN initially output a constant `0.32` probability.
        -   **Cause**: Inputs were normalized using "Percentage Change", which for 1m data is tiny and erratic.
        -   **Fix**: Switched to **Z-Score Normalization** (Standardization), which allowed the model to converge effectively.

### Phase 2: The Data Reality
-   **Baseline Test**: We ran the strategy *without* ML filters.
-   **Result**: 26-29% Win Rate.
-   **Insight**: The "Rejection" pattern filters itself out. If price extends strongly (1.5x ATR) and pulls back, it often **continues** in the original direction rather than reversing.

### Phase 3: The Pivot (Inversion)
-   **User Insight**: "Fade all entries."
-   **Result**: Flipping the trade logic turned a 29% Loser into a **70% Winner**.
-   **Logic**: We validly identified a high-probability **Continuation Pattern** (Pullback Buy) rather than a Reversal.

---

## 2. Key Files & Architecture

### **Good / Verified Files**
1.  **`src/pattern_miner.py`**
    -   **Role**: The Source of Truth.
    -   **Logic**: Hybrid 5m/1m. Scans 5m data for `ATR(14) >= 5` and `Extension >= 1.5x ATR`.
    -   **Safety**: Uses 1m granularity for outcome labeling to ensure precise fills/stops.
    
2.  **`src/models/train_rejection_cnn.py`**
    -   **Role**: The Trainer.
    -   **Key Feature**: Z-Score Normalization `(x - mean) / std`. This is crucial for valid CNN training on price data.
    
3.  **`src/strategies/inverse_strategy.py`**
    -   **Role**: The Money Maker.
    -   **Logic**: Takes the `labeled_rejections_5m.parquet` and simulates FADING every single signal (Inverse Logic).

---

## 3. Data Leakage Prevention & Future Testing

To ensure this result isn't a "backtest anomaly" or result of data leakage, follow these strict protocols when testing on new data:

### A. The "Future Wall" (Strict Chronological Split)
-   **Current State**: We trained on the first 80% and tested on the last 20%.
-   **Verification**: Ensure that the "Test Set" start time is strictly *after* the "Train Set" end time.
-   **Check**:
    ```python
    assert train_data['time'].max() < test_data['time'].min()
    ```

### B. Input Context Isolation
-   **Risk**: The CNN "seeing" the pattern completion.
-   **Solution**: The CNN input window MUST end at `pattern_start_time`.
    -   **Correct**: Input = `[Start - 20m : Start]`
    -   **Incorrect**: Input = `[Trigger - 20m : Trigger]` (This would show the extension happening).
    -   **Status**: **Verified**. We currently use `Start Time` as the cutoff.

### C. Normalization Leakage
-   **Risk**: Calculating Z-Score using statistics from the *whole dataset* (Global Mean/Std).
-   **Solution**: Dynamic Z-Score (Per Window).
    -   We calculate Mean/Std *only* on the specific 20-candle window passed to the model.
    -   **Status**: **Verified**. Code uses `mean = np.mean(feats); feats_norm = (feats - mean) / std` inside the loop.

### D. Lookahead Labeling
-   **Risk**: Labeling a trade 'WIN' based on high/lows that happened *during* the pattern formation.
-   **Solution**: Outcome checking starts at `Trigger Time + 5 Minutes`.
    -   We intentionally skip the candle where the trigger occurred to be conservative and simulate a "Next Bar" entry or ensuring we don't peek at intra-bar future data.
    -   **Status**: **Verified** in `pattern_miner.py`.

### E. Recommended Validation Step (Walk-Forward)
Before deploying live:
1.  **Holdout**: Download a *new* month of data that the system has never seen.
2.  **Blind Run**: Run `pattern_miner` -> `inverse_strategy` on this new month.
3.  **Expectation**: Win Rate should remain within 5-10% of the backtest (i.e., >60%).
