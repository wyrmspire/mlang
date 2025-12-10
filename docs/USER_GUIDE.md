# MES Trading System - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Pattern Generator Mode](#pattern-generator-mode)
3. [Model Training](#model-training)
4. [YFinance Playback Mode](#yfinance-playback-mode)
5. [Creating Custom Setups](#creating-custom-setups)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

#### Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run data pipeline (if using MES historical data)
python -m src.preprocess
python -m src.feature_engineering
python -m src.state_features
python -m src.pattern_library

# Start API server
uvicorn src.api:app --reload --port 8000
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:5173`

### First Time Setup

1. **Start the Backend**: Make sure the API is running on port 8000
2. **Open the Frontend**: Navigate to the local URL
3. **Choose a Mode**: Select Pattern Generator or YFinance Playback from the top tabs

---

## Pattern Generator Mode

### Purpose
Generate synthetic MES price data based on historical patterns. Useful for:
- Understanding price behavior
- Creating training data
- Testing generation algorithms
- Visualizing pattern clusters

### Using Single Day Mode

1. **Select a Date**: Choose from the dropdown (sorted newest first)
2. **Choose Timeframe**: 1m, 5m, 15m, or 60m
3. **Set Generator Parameters**:
   - **Day of Week**: 0=Monday through 6=Sunday
   - **Session Type**: RTH (Regular Trading Hours) or Globex
4. **Click "Generate Session"**: Synthetic data overlays on historical data

### Using Multi-Day Mode

1. **Toggle "Multi-Day"** in the sidebar
2. **Select Start Date**: Earlier date for the sequence
3. **Set Number of Days**: How many days to generate (5-20 typical)
4. **Click "Generate Sequence"**: See real vs synthetic comparison

### Understanding the Output

- **Real Data (Top Chart)**: Historical price action
- **Synthetic Data (Bottom Chart)**: AI-generated sequence
- **Comparison**: Look for similar behavior patterns (trends, ranges, volatility)

### Tips
- Start with RTH sessions - they have cleaner patterns
- Compare multiple generations to see pattern variety
- Use this mode to familiarize yourself with MES price structure

---

## Model Training

### Overview
Training creates CNN models that detect trading setups by learning from historical price patterns.

### Available Training Scripts

#### 1. Predictive Model (Mean Reversion)
```bash
python -m src.train_predictive
```
**Purpose**: Detects mean reversion setups for OCO limit order strategies  
**Data**: 1-minute bars, 20-bar lookback  
**Output**: `models/CNN_Predictive.pth`

#### 2. Predictive Model (5-minute)
```bash
python -m src.train_predictive_5m
```
**Purpose**: Same as above but on 5-minute aggregated data  
**Data**: 5-minute bars, 20-bar lookback  
**Output**: `models/CNN_Predictive_5m.pth`

### Training Process

1. **Data Preparation**:
   - Historical data is loaded from `data/processed/`
   - Labels are created based on strategy rules
   - Features are normalized (Z-score)

2. **Model Training**:
   - 80/20 train/validation split
   - Binary cross-entropy loss
   - Adam optimizer
   - Early stopping on validation loss
   - Best model saved to `models/` directory

3. **Validation**:
   - Win rate and accuracy reported
   - Validation loss monitored
   - Model checkpoint saved when performance improves

### Creating Training Data

#### Method 1: Use Existing MES Data
If you have historical MES data in `data/processed/continuous_1m.parquet`:
```bash
python -m src.train_predictive
```

#### Method 2: Use YFinance Data
Modify the training script to use YFinance:
```python
import yfinance as yf

ticker = yf.Ticker("MES=F")
df = ticker.history(period="60d", interval="1m")
# Continue with training...
```

### Training Parameters

Edit the training script to adjust:
- `batch_size = 64` - Samples per training step
- `epochs = 10` - Number of training passes
- `lr = 1e-3` - Learning rate
- `lookback = 20` - Number of bars for model input

### Expected Results

Good training shows:
- Training loss decreasing steadily
- Validation loss decreasing (if it increases, model is overfitting)
- Win rate > 50% on validation set
- Saved model file in `models/` directory

---

## YFinance Playback Mode

### Purpose
Visually validate trading strategies by replaying trades on real market data. This is the MOST IMPORTANT mode for verifying your strategy works without future leaking.

### Getting Started

1. **Click "YFinance Playback"** tab at the top
2. **Configure Data Source**:
   - Symbol: MES=F, ES=F, or any ticker
   - Source Granularity: 1m or 5m
   - Days to Load: 5-7 for 1m, 30-60 for 5m
   - **Mock Data Checkbox**: Enable for testing (generates fake but realistic data)

3. **Click "Load Data"**: Wait for data to load

### Configuring the Strategy

#### Select Model
Choose from dropdown (models found in `models/*.pth`):
- `CNN_Predictive` - 1m mean reversion
- `CNN_Predictive_5m` - 5m mean reversion
- Others as available

#### Chart Timeframe
Display aggregation (doesn't affect model):
- 5m, 15m, or 1 Hour
- Choose what's easiest for you to visualize

#### Entry Mechanism
**Predictive Limit** (Current Implementation):
- **Sensitivity**: Model confidence threshold (15% = aggressive, 50% = conservative)
- **Limit Factor**: Distance for limit orders (1.5× ATR typical)
- **Stop Factor**: Distance for stop loss (1.0× ATR typical)

When signal fires:
- Sell Limit placed at Price + (Limit × ATR)
- Buy Limit placed at Price - (Limit × ATR)
- First to fill cancels the other (OCO)
- Both have 15-minute expiry

### Running the Playback

1. **Click Play**: Start the tick-by-tick simulation
2. **Adjust Speed**: Slider controls milliseconds per tick (200ms default)
3. **Watch the Magic**:
   - Green light shows model confidence
   - Orders appear as dashed lines
   - Fills convert to solid lines
   - Exits show PnL

4. **Pause Anytime**: Inspect current state
5. **Click Reset**: Start over from beginning

### Understanding the Display

#### Main Chart
- **Candlesticks**: Aggregated to display timeframe
- **Current Price Line**: Latest tick price
- **Pending Orders**: Dashed horizontal lines (yellow/blue)
- **Open Positions**: Solid lines with stop/target markers
- **Trade Results**: Green (win) or red (loss) markers

#### Stats Panel (Left Side)
- **Realized PnL**: Closed trades profit/loss
- **Floating PnL**: Open positions marked to market
- **Open Count**: Active positions
- **Pending Count**: Unfilled orders

#### Signal Indicator (Bottom Left)
- **Green Light**: Intensity = model confidence
- **Percentage**: Probability score from model
- **ATR**: Current market volatility

### Reading the Signals

**High Confidence (>50%)**:
- Bright green light
- Model strongly predicts mean reversion
- More likely to place orders (if above sensitivity threshold)

**Low Confidence (<20%)**:
- Dim light
- Model uncertain
- No orders placed (unless sensitivity is very low)

### Trade Lifecycle Example

```
1. Time 10:00:00 - Model analyzes, confidence = 65%
   ↓
2. Time 10:00:01 - Orders placed: Buy @ 5795, Sell @ 5805
   ↓
3. Time 10:00:15 - Price touches 5805
   ↓
4. Time 10:00:15 - Sell limit fills, SHORT position opened
   ↓
5. Time 10:02:00 - Price returns to 5800 (target hit)
   ↓
6. Time 10:02:00 - Position closed, PnL = +$300
```

### Best Practices

1. **Start with Mock Data**: Test the interface without rate limits
2. **Use Lower Speed**: 500-1000ms per tick for detailed observation
3. **Pause Often**: Inspect signals and order placement
4. **Watch for Patterns**: Do signals make sense? Do they occur at logical times?
5. **Check Fills**: Do orders fill realistically?

### Validation Checklist

Use playback mode to verify:
- [ ] Signals fire BEFORE orders are placed
- [ ] Orders fill on FUTURE bars (not instant)
- [ ] Stop loss and take profit levels are reasonable
- [ ] PnL calculations match expectations
- [ ] Expired orders get cancelled
- [ ] OCO logic works (one fill cancels the other)

---

## Creating Custom Setups

### Concept: What is a Setup?

A **setup** is a specific market condition you want to trade. Examples:
- **Mean Reversion**: Price moves too far, likely to snap back
- **Breakout**: Price breaks through resistance, continues higher
- **Rejection**: Price gets rejected at a key level, reverses

### Steps to Create a New Setup

#### 1. Define Your Criteria

What makes this setup appear? Write it out:
```
Example: "Rejection Setup"
- Price makes a sharp move (>1.5× ATR in 3 bars)
- Then reverses completely within 5 bars
- Enter on retest of the reversal level
```

#### 2. Create Labeling Logic

Label historical bars as setup/no-setup:
```python
# Example: Label mean reversion setups
def label_mean_reversion(df, atr, lookforward=10):
    labels = []
    for i in range(len(df) - lookforward):
        current_price = df.iloc[i]['close']
        future_prices = df.iloc[i+1:i+lookforward+1]['close']
        
        # Did price revert to mean?
        mean_price = current_price
        touched_mean = any(abs(p - mean_price) < 0.5 * atr for p in future_prices)
        
        labels.append(1 if touched_mean else 0)
    
    return labels
```

#### 3. Create Training Script

Copy `src/train_predictive.py` as template:
```python
# src/train_my_setup.py

from src.train_predictive import CNN_Predictive, PredictiveDataset
import pandas as pd

# Load your data
df = pd.read_parquet("data/processed/continuous_1m.parquet")

# Label data with your criteria
labels = label_my_setup(df)

# Create labeled dataframe
labeled_df = pd.DataFrame({
    'index': range(len(labels)),
    'label': labels
})

# Train model (same as predictive)
# ... training code ...

# Save with descriptive name
torch.save(model.state_dict(), "models/MySetup_CNN.pth")
```

#### 4. Test in Playback

1. Train your model: `python -m src.train_my_setup`
2. Open YFinance Playback Mode
3. Select your model from dropdown
4. Load data and play
5. Observe if signals match your setup criteria

#### 5. Iterate

- Adjust labeling criteria
- Try different lookforward periods
- Modify model architecture if needed
- Re-train and re-test

### Example Setups to Implement

**Trend Following**:
```python
# Label: Price makes new high, continues higher
# Entry: Breakout above pivot
# Target: 2×ATR extension
# Stop: Below pivot
```

**Support/Resistance Bounce**:
```python
# Label: Price touches key level, bounces
# Entry: Limit at level
# Target: Previous swing
# Stop: Below level
```

**Gap Fill**:
```python
# Label: Large overnight gap that fills during session
# Entry: Gap midpoint
# Target: Complete fill
# Stop: New extreme
```

---

## Troubleshooting

### "No models available" in Playback

**Cause**: No .pth files in `models/` directory  
**Solution**: Run a training script first
```bash
python -m src.train_predictive
```

### "Data not cached for symbol"

**Cause**: Backend lost cache between fetch and analysis  
**Solution**: Click "Load Data" again before playing

### Playback is too fast/slow

**Solution**: Adjust the speed slider (10-1000ms)
- 10ms = Very fast (for long backtests)
- 200ms = Default
- 1000ms = Slow (for detailed inspection)

### No trades happening

**Check**:
1. Model confidence - Is the green light lighting up?
2. Sensitivity threshold - Lower it to see more signals
3. Data quality - Is ATR > 0?
4. Model loaded - Check browser console for errors

### "Failed to fetch data" from YFinance

**Cause**: YFinance rate limits or invalid symbol  
**Solution**: 
1. Enable "Use Mock Data" for testing
2. Wait a few minutes and retry
3. Verify symbol is correct (e.g., "MES=F" not "MES")
4. Reduce days loaded (max 7 for 1m, 60 for 5m)

### Training shows no improvement

**Possible Issues**:
1. **Labels are wrong**: Check your labeling logic
2. **Model too simple**: Try CNN_Wide architecture
3. **Not enough data**: Need at least 10,000+ labeled samples
4. **Learning rate**: Try 1e-4 instead of 1e-3

**Solution**: Add logging to see distribution of labels:
```python
print(f"Positive labels: {sum(labels)}")
print(f"Negative labels: {len(labels) - sum(labels)}")
```
Should be somewhat balanced (30-70% positive is OK)

### PnL calculations seem wrong

**Verify**:
1. Risk amount = $300 per trade (hardcoded)
2. Position size = Risk / (Entry - Stop)
3. PnL = Size × (Exit - Entry) for longs
4. PnL = Size × (Entry - Exit) for shorts

**Check**: Pause playback and manually calculate for one trade

### Unclear if model is "cheating"

**Visual Verification Steps**:
1. Pause when signal fires (green light)
2. Note the time and price
3. Check that orders are placed AFTER analysis
4. Step forward one tick at a time
5. Confirm fill happens on a FUTURE bar
6. Check that model input window was BEFORE signal

---

## Advanced Topics

### Custom Entry Mechanisms

To add a new entry mechanism (e.g., "Market Close"):

1. Add to `YFinanceMode.tsx` in `entryMechanism` state
2. Implement logic in `checkSignal()` function
3. Handle order creation differently:
```typescript
if (entryMechanism === 'Market Close') {
    // Enter immediately at close
    const position: Trade = {
        id: `pos_${idx}`,
        direction: 'LONG', // or 'SHORT' based on signal
        entry_price: currentPrice,
        status: 'open',
        // ... stops and targets
    };
    setTrades(prev => [...prev, position]);
}
```

### Batch Testing

To test multiple models/parameters:

1. Create test script based on `tests/test_predictive.py`
2. Loop over models and parameter combinations
3. Save results to CSV for comparison
4. Analyze which configurations work best

### Data Exporting

To export trade results:
```typescript
// In YFinanceMode.tsx, add export button
const exportTrades = () => {
    const csv = trades.map(t => ({
        id: t.id,
        entry: t.entry_time,
        exit: t.exit_time,
        pnl: t.pnl,
        direction: t.direction
    }));
    
    // Convert to CSV and download
    const csvContent = "data:text/csv;charset=utf-8," 
        + Object.keys(csv[0]).join(",") + "\n"
        + csv.map(e => Object.values(e).join(",")).join("\n");
    
    const link = document.createElement("a");
    link.setAttribute("href", encodeURI(csvContent));
    link.setAttribute("download", "trades.csv");
    link.click();
};
```

---

## Best Practices Summary

### For Learning
1. Start with Pattern Generator to understand data
2. Review ARCHITECTURE.md for system overview
3. Use Mock Data in Playback for safe experimentation

### For Development
1. Always start with small datasets
2. Validate visually before running long backtests
3. Save models with descriptive names
4. Document your setup criteria

### For Trading Validation
1. Use Playback Mode extensively
2. Verify no future leaking by pausing and inspecting
3. Test on different time periods
4. Check edge cases (gaps, low liquidity, high volatility)

---

**Need More Help?**
- Check `docs/ARCHITECTURE.md` for technical details
- Review code comments in source files
- Run test scripts to see working examples

**Last Updated**: 2025-12-10
