# MES Trading System Architecture

## Overview

This is a professional quantitative trading development platform that combines pattern generation, CNN-based signal detection, and visual trade playback. The system allows you to:

1. **Generate synthetic price data** based on historical MES patterns
2. **Train CNN models** to detect trading setups and predict price behavior
3. **Backtest strategies** using historical YFinance data
4. **Visually validate trades** in playback mode to verify signal integrity

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MES Trading Platform                     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Pattern    │    │    Model     │    │   Trading    │
│  Generator   │    │   Training   │    │   Playback   │
│    Mode      │    │    Mode      │    │     Mode     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Synthetic   │    │     CNN      │    │  YFinance    │
│    Data      │───▶│   Models     │───▶│  Real Data   │
│  Generation  │    │  (*.pth)     │    │   Testing    │
└──────────────┘    └──────────────┘    └──────────────┘
```

## Core Components

### 1. Pattern Generator Mode (`frontend/src/App.tsx`)

**Purpose**: Generate synthetic MES price data based on historical patterns

**Key Features**:
- Single-day generation with specific session patterns (RTH/Globex)
- Multi-day generation with state-aware transitions
- Visual comparison of real vs synthetic data
- Cluster-based pattern library

**Workflow**:
1. Select historical date or date range
2. Configure generation parameters (day of week, session type)
3. Generate synthetic overlay or multi-day sequence
4. Visual validation on charts

**Backend**: 
- `src/generator.py` - Core pattern generation logic
- `src/pattern_library.py` - K-means clustering of hour patterns
- `src/state_features.py` - Daily state extraction for continuity

### 2. Model Training (Backend Only - CLI)

**Purpose**: Train CNN models to detect trading setups and predict price movements

**Available Models**:
- `CNN_Predictive` - Mean reversion signal detector (OCO limit orders)
- `CNN_Classic` - Basic pattern classifier
- `CNN_Wide` - Wider architecture for complex patterns
- `rejection_cnn` - Rejection bar detector
- `setup_cnn` - General setup detector

**Training Scripts**:
- `src/train_predictive.py` - Train predictive models on 1m data
- `src/train_predictive_5m.py` - Train on 5m aggregated data
- `src/train_models_phase2.py` - Multi-model training pipeline

**Data Preparation**:
1. Label data using strategy-specific criteria
2. Create features (OHLC, ATR, volume, etc.)
3. Train with validation split
4. Save best model to `models/` directory

**Command**:
```bash
python -m src.train_predictive
python -m src.train_predictive_5m
```

### 3. YFinance Playback Mode (`frontend/src/components/YFinanceMode.tsx`)

**Purpose**: Visually replay trades on real market data to validate signal integrity and verify no future leaking

**Key Features**:
- Tick-by-tick playback with adjustable speed
- Real-time CNN signal analysis
- Live order management (pending → open → closed)
- OCO (One-Cancels-Other) bracket orders
- PnL tracking (realized + unrealized)
- Visual trade markers on chart

**Workflow**:
1. Load real YFinance data (or use mock data for testing)
2. Select trained CNN model
3. Configure entry mechanism and parameters
4. Play forward tick-by-tick
5. Watch model sense setups and execute trades
6. Validate results visually

**Entry Mechanisms**:
- **Predictive Limit**: OCO bracket orders at ±N×ATR from current price
- **Market Close** (Not Yet Implemented): Enter at signal bar close

**Backend API Endpoints**:
- `GET /api/yfinance/candles` - Fetch real market data
- `GET /api/yfinance/candles/mock` - Generate mock data for testing
- `POST /api/yfinance/playback/analyze` - Analyze candle with model
- `GET /api/yfinance/models` - List available trained models

### 4. Model Inference Engine (`src/model_inference.py`)

**Purpose**: Real-time signal generation for playback mode

**Features**:
- Loads trained CNN models
- Calculates 15m ATR from 1m data
- Generates signals with probability scores
- Returns trade parameters (limit prices, stops, targets)

**Parameters** (Configurable):
- `WINDOW_SIZE = 20` - Lookback period for model input
- `ATR_PERIOD = 14` - ATR calculation period
- `THRESHOLD = 0.15` - Minimum probability to generate signal

**Signal Output**:
```python
{
    "type": "OCO_LIMIT",
    "prob": 0.65,           # Model confidence
    "atr": 3.2,            # Current ATR
    "current_price": 5800,
    "sell_limit": 5804.8,  # Price + 1.5×ATR
    "buy_limit": 5795.2,   # Price - 1.5×ATR
    "sl_dist": 3.2,        # 1.0×ATR
    "validity": 900        # 15 minutes
}
```

## Data Flow

### Training Flow (No Future Leaking)
```
Historical Data → Feature Engineering → Label Generation → Train Model → Save .pth
                                                              ↓
                                            (Features use ONLY past data)
```

### Testing Flow (Strict Time-Series)
```
Load Data → Initialize Model → Playback Loop:
                                ├─ Update current candle
                                ├─ Manage existing trades/orders
                                ├─ Analyze with model (past data only)
                                └─ Place new orders (future execution)
```

### Key Anti-Leak Protections:

1. **ATR Calculation**: Shifted by 1 period (value at T uses data up to T-1)
2. **Model Input**: Uses PREVIOUS N bars, not including current bar
3. **Order Execution**: Limits placed AFTER analysis, filled on NEXT ticks
4. **Trade Management**: Uses bar high/low for fills (realistic intrabar)

## File Organization

```
mlang/
├── frontend/               # React + TypeScript UI
│   ├── src/
│   │   ├── App.tsx        # Main app with mode tabs
│   │   ├── components/
│   │   │   ├── YFinanceMode.tsx      # Playback interface
│   │   │   ├── YFinanceChart.tsx     # Chart with trades
│   │   │   ├── YFinancePage.tsx      # (Legacy - simple view)
│   │   │   ├── ChartPanel.tsx        # Pattern gen chart
│   │   │   └── SidebarControls.tsx   # Pattern controls
│   │   └── api/
│   │       ├── client.ts             # Pattern API
│   │       └── yfinance.ts           # YFinance API
│   └── package.json
├── src/                    # Python backend
│   ├── api.py             # FastAPI server
│   ├── model_inference.py # Real-time model execution
│   ├── train_predictive.py    # Model training
│   ├── train_predictive_5m.py
│   ├── generator.py       # Pattern generation
│   ├── pattern_library.py
│   ├── yfinance_loader.py
│   ├── config.py
│   ├── models/            # Model architectures
│   │   └── variants.py
│   ├── strategies/        # Strategy implementations
│   │   ├── collector.py
│   │   ├── inverse_strategy.py
│   │   └── ...
│   └── utils/
├── models/                # Trained model files (.pth)
├── data/                  # Data storage
│   ├── raw/              # Raw MES JSON
│   └── processed/        # Parquet files
├── tests/                # Test scripts
│   ├── test_predictive.py
│   └── ...
└── docs/                 # Documentation
    └── ARCHITECTURE.md   # This file
```

## Configuration Files

### Backend (`src/config.py`)
- Directory paths
- Model hyperparameters
- Data constants

### Frontend (`frontend/src/components/YFinanceMode.tsx`)
- Order expiry time
- Risk per trade
- Default model selection

## Trading Concepts Clarified

### Setup vs Signal vs Trade

1. **Setup**: Market condition that might lead to a trade
   - Example: Price approaching key level with low ATR
   - Detected by CNN model (probability score)

2. **Signal**: Decision point to place orders
   - Generated when model confidence > threshold
   - Includes specific parameters (limit prices, stops, targets)

3. **Trade**: Actual position after order fills
   - Has entry price, direction (LONG/SHORT)
   - Managed with stop loss and take profit
   - Tracks PnL until exit

### Order Lifecycle

```
Signal Generated → Pending Orders (OCO) → Order Filled → Open Position → Exit (TP/SL)
                   ↓                       ↓              ↓
                   [15min expiry]         [Cancel OCO]   [Track PnL]
```

### Entry Mechanisms Explained

**Predictive Limit (Current)**:
- Model predicts mean reversion
- Places buy limit BELOW and sell limit ABOVE
- First to fill cancels the other (OCO)
- Targets return to mean (current price)

**Market Close (TODO)**:
- Enter immediately at bar close when signal fires
- Simpler but less selective
- No limit order management needed

## Future Leaking Prevention

### What is Future Leaking?
Using information from the future (data not yet available) during backtesting, leading to unrealistic results.

### How We Prevent It:

1. **Model Training**:
   - Labels created from FUTURE data (this is OK - we're learning patterns)
   - Features created from PAST data only

2. **Playback Testing**:
   - Process data tick-by-tick chronologically
   - Model sees only past candles (windowed lookback)
   - Orders placed AFTER analysis, executed on FUTURE ticks
   - ATR shifted to prevent using current bar's ATR

3. **Trade Execution**:
   - Limit orders fill using bar high/low (realistic intrabar)
   - Gap fills handled conservatively (better price = immediate fill)
   - No peeking at future bars for order management

### Visual Validation

The playback mode EXISTS specifically to visually confirm no leaking:
- You see the signal fire
- You see orders placed
- You see fills happen on subsequent bars
- You can pause and inspect each step

## Performance Metrics

### Win Rate
Percentage of closed trades that hit take profit (vs stop loss)

### PnL (Profit and Loss)
- **Realized**: Closed trades
- **Unrealized**: Open positions marked to market

### R-Multiple
Ratio of reward to risk per trade
- Risk = Entry - Stop Loss
- Reward = Take Profit - Entry
- R = Reward / Risk

## Common Questions

**Q: Why do I see good results but no actual trades?**  
A: Check if test scripts are using the correct model and data. Some scripts analyze patterns without executing trade logic. Use YFinance Playback Mode for visual trade validation.

**Q: How do I know the model isn't cheating?**  
A: Use Playback Mode - watch trades happen in real-time. Pause and verify that signals use only past data. Check that orders fill on future bars, not current.

**Q: Can I create my own setups?**  
A: Yes! Create a new training script in `src/`, define labeling logic for your setup, train a model, then use it in Playback Mode.

**Q: What's the difference between 1m and 5m models?**  
A: Source data granularity. 1m models train on 1-minute bars, 5m on 5-minute. Finer granularity = more data but slower training. Choose based on your trading timeframe.

## Next Steps

1. **For New Users**: Start with Pattern Generator to understand data structure
2. **For Modelers**: Review training scripts, create custom setups
3. **For Traders**: Use YFinance Playback to validate strategies visually
4. **For Developers**: See API documentation for integration options

## Getting Help

- Review this architecture doc
- Check inline code comments
- Run test scripts to see examples
- Use mock data mode for safe experimentation

---

**Last Updated**: 2025-12-10  
**Version**: 1.0
