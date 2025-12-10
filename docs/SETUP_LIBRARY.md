# Trading Setup Library

## Overview

This document catalogs different trading setups (patterns/strategies) that can be trained and tested in the MES Trading System. Each setup includes:
- **Description**: What the setup looks for
- **Entry Logic**: How to enter the trade
- **Exit Logic**: Stop loss and take profit rules
- **Labeling Criteria**: How to create training labels
- **Implementation Status**: Whether it's implemented

---

## Currently Implemented Setups

### 1. Mean Reversion (Predictive Limit)

**Status**: ✅ Fully Implemented

**Description**: 
Detects when price has moved too far from its mean and is likely to snap back. Uses OCO (One-Cancels-Other) bracket orders to catch the reversal in either direction.

**Market Conditions**:
- Works best in ranging markets
- Requires sufficient ATR (volatility)
- Avoid during strong trends

**Entry Logic**:
- Model signals high probability of mean reversion
- Place BUY limit below current price: Price - (1.5 × ATR)
- Place SELL limit above current price: Price + (1.5 × ATR)
- First to fill cancels the other
- Orders expire after 15 minutes

**Exit Logic**:
- Take Profit: Return to mean (original signal price)
- Stop Loss: 1.0 × ATR beyond entry price
- R-Multiple: Varies based on fill location

**Labeling Criteria**:
```python
def label_mean_reversion(df, atr, lookforward=10):
    """
    Label bars where price reverts to mean within lookforward period.
    
    Args:
        df: DataFrame with OHLC data
        atr: Current ATR value
        lookforward: Bars to look ahead
        
    Returns:
        1 if price reverts to mean, 0 otherwise
    """
    labels = []
    for i in range(len(df) - lookforward):
        current = df.iloc[i]['close']
        future = df.iloc[i+1:i+lookforward+1]
        
        # Check if any future bar gets within 0.5×ATR of current
        reverted = any(abs(future['close'] - current) < 0.5 * atr)
        labels.append(1 if reverted else 0)
    
    return labels
```

**Training Script**: `src/train_predictive.py`, `src/train_predictive_5m.py`

**Model Files**: 
- `models/CNN_Predictive.pth` (1m data)
- `models/CNN_Predictive_5m.pth` (5m data)

**Parameters**:
- Lookback: 20 bars
- Input features: OHLC + ATR
- Threshold: 0.15 (15% confidence minimum)

**Performance Tips**:
- Lower sensitivity (higher threshold) for cleaner signals
- Increase limit factor in choppy markets
- Decrease stop factor for tighter risk management

---

## Setups To Implement

### 2. Breakout (Momentum)

**Status**: 📋 Planned

**Description**: 
Detects when price breaks through a significant level and continues in that direction. Enters in the direction of the breakout.

**Market Conditions**:
- Works best during trending periods
- Needs clear resistance/support levels
- Volume confirmation helpful

**Entry Logic** (Proposed):
- Model detects high probability breakout
- Wait for price to break above resistance (or below support)
- Enter market order on break + 1 bar confirmation
- Or use limit order at breakout level + 0.25×ATR

**Exit Logic** (Proposed):
- Take Profit: 2.0 × ATR extension from breakout
- Stop Loss: Back below breakout level (1.0 × ATR)
- Trailing stop option: Move stop to breakeven after 1×ATR

**Labeling Criteria** (Proposed):
```python
def label_breakout(df, atr, lookback=20, lookforward=10):
    """
    Label bars where price breaks recent high/low and continues.
    
    Args:
        df: DataFrame with OHLC data
        atr: Current ATR value
        lookback: Bars to identify resistance/support
        lookforward: Bars to confirm continuation
        
    Returns:
        1 if breakout and continuation, 0 otherwise
    """
    labels = []
    for i in range(lookback, len(df) - lookforward):
        # Find recent high
        recent_high = df.iloc[i-lookback:i]['high'].max()
        current = df.iloc[i]['close']
        
        # Check if breaking high
        if current > recent_high + 0.5 * atr:
            # Check if continues higher
            future_high = df.iloc[i+1:i+lookforward+1]['high'].max()
            if future_high > current + 1.0 * atr:
                labels.append(1)
                continue
        
        labels.append(0)
    
    return labels
```

**Implementation Steps**:
1. Create `src/train_breakout.py`
2. Implement labeling logic
3. Train CNN_Breakout model
4. Add to YFinanceMode as entry option
5. Test in playback mode

---

### 3. Rejection (Reversal)

**Status**: 🔧 Partially Implemented (model exists)

**Description**: 
Detects when price gets rejected at a key level and reverses sharply. Often forms a long wick candle.

**Market Conditions**:
- Works at established support/resistance
- Best during range-bound markets
- Requires quick reversal (within 1-3 bars)

**Entry Logic** (Proposed):
- Model detects rejection pattern
- Enter counter-trend after rejection confirmed
- Limit order at rejection level (50% retracement of wick)
- Or market order on next bar after rejection candle

**Exit Logic** (Proposed):
- Take Profit: Opposite end of range (2-3× ATR)
- Stop Loss: Beyond rejection wick (0.5 × ATR)
- Quick exit if price re-tests and breaks level

**Labeling Criteria** (Proposed):
```python
def label_rejection(df, atr, lookforward=5):
    """
    Label bars where price gets rejected and reverses.
    
    Args:
        df: DataFrame with OHLC data
        atr: Current ATR value
        lookforward: Bars to confirm reversal
        
    Returns:
        1 if rejection and reversal, 0 otherwise
    """
    labels = []
    for i in range(len(df) - lookforward):
        current = df.iloc[i]
        
        # Check for long wick (rejection)
        body = abs(current['close'] - current['open'])
        upper_wick = current['high'] - max(current['open'], current['close'])
        lower_wick = min(current['open'], current['close']) - current['low']
        
        # Upper rejection (sell signal)
        if upper_wick > 2 * body and upper_wick > 0.75 * atr:
            future = df.iloc[i+1:i+lookforward+1]
            # Price moves down?
            if future['low'].min() < current['close'] - 1.0 * atr:
                labels.append(1)
                continue
        
        # Lower rejection (buy signal)
        if lower_wick > 2 * body and lower_wick > 0.75 * atr:
            future = df.iloc[i+1:i+lookforward+1]
            # Price moves up?
            if future['high'].max() > current['close'] + 1.0 * atr:
                labels.append(1)
                continue
        
        labels.append(0)
    
    return labels
```

**Model File**: `models/rejection_cnn_v1.pth` (exists but needs integration)

**Next Steps**:
1. Update model inference to support rejection signals
2. Add entry mechanism to YFinanceMode
3. Test with existing model
4. Refine as needed

---

### 4. Trend Following (Pullback Entry)

**Status**: 📋 Planned

**Description**: 
Enters in the direction of an established trend after a brief pullback. Combines trend strength with mean reversion timing.

**Market Conditions**:
- Requires clear trending market
- Look for "higher highs, higher lows" (uptrend)
- Or "lower highs, lower lows" (downtrend)

**Entry Logic** (Proposed):
- Identify trend direction (moving average, higher highs/lows)
- Wait for pullback against trend (0.5-1.0× ATR)
- Model signals high probability of trend resumption
- Enter in trend direction on pullback

**Exit Logic** (Proposed):
- Take Profit: New swing high/low (3-5× ATR potential)
- Stop Loss: Below pullback low (1.5 × ATR)
- Trailing stop: Move up as new swing highs made

**Labeling Criteria** (Proposed):
```python
def label_trend_pullback(df, atr, ma_period=50, lookforward=15):
    """
    Label bars where trend continues after pullback.
    
    Args:
        df: DataFrame with OHLC data
        atr: Current ATR value
        ma_period: Period for trend identification
        lookforward: Bars to confirm resumption
        
    Returns:
        1 if trend resumes, 0 otherwise
    """
    df['ma'] = df['close'].rolling(ma_period).mean()
    
    labels = []
    for i in range(ma_period, len(df) - lookforward):
        current = df.iloc[i]
        ma = current['ma']
        
        # Uptrend: Price above MA
        if current['close'] > ma:
            # Pullback: Price touches MA
            if abs(current['close'] - ma) < 0.5 * atr:
                # Resumes up?
                future = df.iloc[i+1:i+lookforward+1]
                if future['high'].max() > current['close'] + 2.0 * atr:
                    labels.append(1)
                    continue
        
        # Downtrend: Similar logic
        if current['close'] < ma:
            if abs(current['close'] - ma) < 0.5 * atr:
                future = df.iloc[i+1:i+lookforward+1]
                if future['low'].min() < current['close'] - 2.0 * atr:
                    labels.append(1)
                    continue
        
        labels.append(0)
    
    return labels
```

---

### 5. Gap Fill

**Status**: 📋 Planned

**Description**: 
Detects overnight gaps and predicts whether they will fill during the session.

**Market Conditions**:
- Requires gap at market open
- Works best with moderate gaps (0.5-2.0× ATR)
- Avoid on major news days

**Entry Logic** (Proposed):
- Identify gap at open (current open vs previous close)
- Model predicts probability of fill
- Enter at gap midpoint with limit order
- Or enter at open with target at previous close

**Exit Logic** (Proposed):
- Take Profit: Complete gap fill (previous close)
- Stop Loss: New extreme beyond gap (1.0× ATR)
- Time-based: Exit at session close if not filled

**Labeling Criteria** (Proposed):
```python
def label_gap_fill(df, atr):
    """
    Label bars where overnight gap fills during session.
    
    Args:
        df: DataFrame with OHLC data (must have date)
        atr: Current ATR value
        
    Returns:
        1 if gap fills, 0 otherwise
    """
    labels = []
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        current = df.iloc[i]
        
        # Check for day change (new session)
        if current['date'] != prev['date']:
            gap = current['open'] - prev['close']
            
            # Only label if meaningful gap
            if abs(gap) > 0.5 * atr:
                # Find session bars
                session_start = i
                session_end = i
                while (session_end < len(df) and 
                       df.iloc[session_end]['date'] == current['date']):
                    session_end += 1
                
                session = df.iloc[session_start:session_end]
                
                # Up gap: Check if low touches prev close
                if gap > 0:
                    filled = session['low'].min() <= prev['close']
                # Down gap: Check if high touches prev close
                else:
                    filled = session['high'].max() >= prev['close']
                
                labels.append(1 if filled else 0)
                continue
        
        labels.append(0)  # Not a gap bar
    
    return labels
```

---

### 6. Range Expansion

**Status**: 📋 Planned

**Description**: 
Detects when market breaks out of a tight range and expands volatility. Often leads to sustained moves.

**Market Conditions**:
- Follows period of low volatility (ATR compression)
- Volume often expands on breakout
- Works best on intraday timeframes

**Entry Logic** (Proposed):
- Monitor ATR for compression (current < 0.7× 20-period average)
- Model detects setup for expansion
- Enter on first strong directional bar (>1.5× recent ATR)
- Direction determined by break (above/below range)

**Exit Logic** (Proposed):
- Take Profit: 3-4× recent ATR (volatile target)
- Stop Loss: Back into range (1.0× ATR)
- Partial profits: Scale out as extensions hit

---

## Setup Comparison Matrix

| Setup | Market Type | Timeframe | Win Rate | R-Multiple | Complexity |
|-------|-------------|-----------|----------|------------|------------|
| Mean Reversion | Range | 1m-15m | 55-65% | 0.8-1.5 | Low |
| Breakout | Trend | 5m-60m | 45-55% | 2.0-3.0 | Medium |
| Rejection | Range/Support | 1m-5m | 60-70% | 1.0-1.5 | Low |
| Trend Pullback | Strong Trend | 15m-60m | 50-60% | 2.0-4.0 | High |
| Gap Fill | Session Open | 5m-15m | 65-75% | 1.0-2.0 | Low |
| Range Expansion | Compression | 5m-15m | 40-50% | 3.0-5.0 | Medium |

**Legend**:
- Win Rate: Historical percentage of winning trades
- R-Multiple: Average reward/risk ratio
- Complexity: Difficulty of implementation and parameter tuning

---

## Creating Your Own Setup

### Step-by-Step Process

1. **Observe Market Behavior**
   - Watch charts and note recurring patterns
   - Identify what happens before/after the pattern
   - Define the setup in plain English

2. **Define Objective Criteria**
   - What measurements define this setup?
   - Can it be calculated from OHLC + indicators?
   - Is it consistent across different timeframes?

3. **Write Labeling Function**
   - Use historical data to create labels
   - Label = 1 when setup occurs and works
   - Label = 0 otherwise
   - Include lookforward period to check "works"

4. **Create Training Script**
   - Load data and apply labels
   - Split train/validation
   - Train CNN model
   - Save with descriptive name

5. **Implement Entry Mechanism**
   - Add to YFinanceMode.tsx as new option
   - Define order placement logic
   - Set stops and targets
   - Handle special cases

6. **Test and Validate**
   - Use playback mode extensively
   - Try different market conditions
   - Measure win rate and PnL
   - Refine parameters

7. **Document**
   - Add to this library
   - Include example code
   - Note market conditions that work best

---

## Setup Selection Guide

### For Beginners
Start with:
1. **Mean Reversion** - Simple logic, well-documented
2. **Rejection** - Visual pattern, easy to understand

### For Intermediate
Try:
3. **Breakout** - Requires trend identification
4. **Gap Fill** - Time-specific, different logic

### For Advanced
Implement:
5. **Trend Pullback** - Complex multi-factor analysis
6. **Range Expansion** - Volatility-based, needs tuning

---

## Multi-Setup Systems

### Combining Setups

You can run multiple setups simultaneously:

```typescript
// Example: Run both mean reversion and breakout
const strategies = ['MeanReversion', 'Breakout'];

strategies.forEach(strat => {
    const signal = analyzeWithModel(models[strat]);
    if (signal.confidence > threshold[strat]) {
        placeOrders(strat, signal);
    }
});
```

**Benefits**:
- Diversification across market conditions
- Higher opportunity frequency
- Reduced correlation of trades

**Risks**:
- Increased complexity
- Conflicting signals
- More parameters to manage

### Setup Portfolio Allocation

Allocate risk across setups:
- 40% Mean Reversion (stable base)
- 30% Trend Following (capture big moves)
- 20% Breakout (momentum)
- 10% Experimental (new setups)

---

## Performance Tracking

### Metrics to Monitor

For each setup, track:
- **Total Trades**: Number of signals taken
- **Win Rate**: Percentage of profitable trades
- **Average R**: Average reward/risk ratio
- **Profit Factor**: Gross profit / Gross loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns

### Analysis by Conditions

Break down performance by:
- Time of day (market open, midday, close)
- Day of week (Monday effect, Friday profit-taking)
- Volatility regime (high ATR, low ATR)
- Trend environment (trending up/down/range)

This helps identify optimal conditions for each setup.

---

## Advanced Concepts

### Adaptive Parameters

Instead of fixed parameters, adapt to conditions:

```python
# Example: Adjust limit factor based on ATR regime
atr_20 = df['atr'].rolling(20).mean()
current_atr = df['atr'].iloc[-1]

if current_atr > 1.5 * atr_20:  # High volatility
    limit_factor = 2.0  # Wider limits
else:
    limit_factor = 1.0  # Tighter limits
```

### Machine Learning for Parameter Optimization

Use ML to find optimal parameters:
1. Grid search over parameter ranges
2. Evaluate each combination in backtest
3. Select based on Sharpe ratio or other metric
4. Validate on out-of-sample data

---

## Resources

### Recommended Reading
- "Trading in the Zone" by Mark Douglas
- "Technical Analysis of the Financial Markets" by John Murphy
- "Algorithmic Trading" by Ernest Chan

### Useful Tools
- TradingView for manual pattern observation
- Python libraries: pandas, numpy, scikit-learn
- Jupyter notebooks for experimentation

---

**Last Updated**: 2025-12-10  
**Version**: 1.0

**Contributing**: Have a setup that works? Add it to this library!
