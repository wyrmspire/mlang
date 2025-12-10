# MES Trading System

A professional quantitative trading development platform for pattern generation, CNN-based signal detection, and visual trade validation.

## What This System Does

This platform combines three powerful capabilities:

1. **Pattern Generator**: Generate synthetic MES price data based on historical patterns using K-means clustering
2. **Model Training**: Train CNN models to detect trading setups (mean reversion, breakouts, etc.)
3. **Visual Playback**: Validate strategies on real market data with tick-by-tick replay to verify no future leaking

## Key Features

- ✅ **Dual-mode UI**: Pattern generation OR live trading playback
- ✅ **CNN Models**: Pre-trained models for mean reversion setups
- ✅ **Real Data Integration**: YFinance API for live market data
- ✅ **Visual Trade Validation**: See signals fire and trades execute in real-time
- ✅ **Order Management**: OCO bracket orders, stop loss, take profit
- ✅ **PnL Tracking**: Real-time profit/loss calculation
- ✅ **Anti-Leak Protection**: Strict time-series processing prevents future data leaking

## Quick Start

### 1. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn src.api:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### 3. Try It Out

**Pattern Generator Mode**:
- Select a date and timeframe
- Click "Generate Session" to see synthetic data overlay

**YFinance Playback Mode**:
- Switch to "YFinance Playback" tab
- Enable "Use Mock Data" for testing
- Click "Load Data" then "Play"
- Watch the CNN model detect setups and execute trades

## Documentation

📚 **[Architecture Guide](docs/ARCHITECTURE.md)** - System overview and component details  
📖 **[User Guide](docs/USER_GUIDE.md)** - Step-by-step instructions for all features  
📊 **[Setup Library](docs/SETUP_LIBRARY.md)** - Trading setup catalog with implementation examples  
📈 **[Success Study](docs/success_study.md)** - Performance analysis and results

## Architecture Overview

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
│    Mode      │    │    (CLI)     │    │     Mode     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Synthetic   │    │     CNN      │    │  YFinance    │
│    Data      │───▶│   Models     │───▶│  Real Data   │
│  Generation  │    │  (*.pth)     │    │   Testing    │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Component Details

- **Data Pipeline**: Python scripts to ingest, process, and engineer features from raw price data
- **Pattern Learning**: K-Means clustering of intraday hours by Session, Day of Week, and Hour
- **Generator**: Non-parametric bootstrapper that creates realistic synthetic multi-day sequences
- **API**: FastAPI backend serving data, generation, and model inference
- **Frontend**: React + TypeScript UI with dual modes (Pattern Gen / Trading Playback)
- **Model Training**: CNN architectures for detecting trading setups
- **Inference Engine**: Real-time signal generation with anti-leak protections

## Workflow

### 1. Pattern Generation (Optional)
Use if you have historical MES data and want to understand patterns:

```bash
# Process historical data
python -m src.preprocess           # Raw JSON → Parquet
python -m src.feature_engineering  # Extract features
python -m src.state_features       # Daily state extraction
python -m src.pattern_library      # Build clusters
```

Then use the **Pattern Generator** tab in the UI to explore synthetic data generation.

### 2. Model Training
Train CNN models to detect trading setups:

```bash
# Train mean reversion model
python -m src.train_predictive      # 1-minute data
# OR
python -m src.train_predictive_5m   # 5-minute data
```

Models are saved to `models/*.pth` and automatically available in playback mode.

### 3. Visual Validation
Use **YFinance Playback** mode to validate strategies:

1. Switch to "YFinance Playback" tab
2. Configure data source and model
3. Click "Load Data" → "Play"
4. Watch trades execute in real-time
5. Verify signals, entries, and exits visually

## Usage Examples

### Pattern Generator Mode

**Single Day**:
1. Select a date from dropdown
2. Choose timeframe (1m, 5m, 15m, 60m)
3. Click "Generate Session"
4. See synthetic overlay on historical data

**Multi-Day**:
1. Toggle "Multi-Day" mode
2. Select start date and duration
3. Click "Generate Sequence"
4. Compare real vs synthetic charts

### Trading Playback Mode

**Basic Usage**:
1. Enable "Use Mock Data" for testing
2. Click "Load Data"
3. Select model from dropdown
4. Adjust sensitivity slider
5. Click "Play" and watch

**Advanced**:
- Adjust entry mechanism parameters (Limit Factor, Stop Factor)
- Change playback speed for detailed inspection
- Pause to examine specific signals
- Monitor PnL and trade statistics in real-time

## Model Training Workflow

### Creating Custom Setups

1. **Define your setup criteria** (e.g., breakout, rejection, trend following)
2. **Create labeling logic** to identify setup occurrences in historical data
3. **Copy a training script** as template (`src/train_predictive.py`)
4. **Implement your labeling function**
5. **Train the model** and save to `models/`
6. **Test in playback mode** to validate

See [Setup Library](docs/SETUP_LIBRARY.md) for detailed examples.

## Project Structure

```
mlang/
├── frontend/                      # React + TypeScript UI
│   ├── src/
│   │   ├── App.tsx               # Main app with mode tabs
│   │   ├── components/
│   │   │   ├── YFinanceMode.tsx  # Trading playback interface
│   │   │   ├── YFinanceChart.tsx # Chart with trade markers
│   │   │   ├── ChartPanel.tsx    # Pattern generator chart
│   │   │   └── SidebarControls.tsx
│   │   └── api/
│   │       ├── client.ts         # Pattern API client
│   │       └── yfinance.ts       # YFinance API client
│   └── package.json
├── src/                          # Python backend
│   ├── api.py                    # FastAPI server
│   ├── model_inference.py        # Real-time model inference
│   ├── train_predictive.py       # Model training scripts
│   ├── train_predictive_5m.py
│   ├── generator.py              # Pattern generation
│   ├── pattern_library.py        # K-means clustering
│   ├── yfinance_loader.py        # Data fetching
│   ├── config.py                 # Configuration
│   ├── models/                   # Model architectures
│   │   └── variants.py
│   ├── strategies/               # Trading strategies
│   └── utils/                    # Utilities
├── models/                       # Trained models (.pth files)
│   ├── CNN_Predictive.pth
│   ├── CNN_Predictive_5m.pth
│   └── ...
├── data/                         # Data storage
│   ├── raw/                      # Raw MES JSON (if using)
│   └── processed/                # Parquet files
├── tests/                        # Test scripts
│   ├── test_predictive.py
│   └── ...
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md           # System architecture
│   ├── USER_GUIDE.md             # User guide
│   ├── SETUP_LIBRARY.md          # Trading setups
│   └── success_study.md          # Performance analysis
└── README.md                     # This file
```

## Available Models

Pre-trained models in `models/` directory:

- **CNN_Predictive** - Mean reversion on 1m data
- **CNN_Predictive_5m** - Mean reversion on 5m data  
- **CNN_Classic** - Basic pattern classifier
- **CNN_Wide** - Wider architecture for complex patterns
- **rejection_cnn_v1** - Rejection bar detector
- **setup_cnn_v1** - General setup detector

Models are automatically detected and available in the playback mode dropdown.

## API Endpoints

### Pattern Generator
- `GET /api/dates` - Available dates
- `GET /api/candles` - Historical data for date/timeframe
- `POST /api/generate/session` - Generate single session
- `POST /api/generate/multi-day` - Generate multi-day sequence

### YFinance Playback
- `GET /api/yfinance/candles` - Fetch real market data
- `GET /api/yfinance/candles/mock` - Generate mock data for testing
- `POST /api/yfinance/playback/analyze` - Analyze with model
- `GET /api/yfinance/models` - List available models

## Configuration

### Backend (`src/config.py`)
```python
WINDOW_SIZE = 20          # Model lookback period
ATR_PERIOD = 14          # ATR calculation period  
THRESHOLD = 0.15         # Signal confidence threshold
```

### Frontend (`frontend/src/components/YFinanceMode.tsx`)
```typescript
ORDER_EXPIRY_SECONDS = 900  // 15 minutes
riskAmount = 300            // Risk per trade ($)
modelThreshold = 0.15       // Signal sensitivity
limitFactor = 1.5           // Limit order distance (×ATR)
slFactor = 1.0             // Stop loss distance (×ATR)
```

## Testing

### Unit Tests
```bash
# Run specific test
python -m pytest tests/test_predictive.py

# Run all tests
python -m pytest tests/
```

### Manual Validation
Use YFinance Playback mode with mock data:
1. Enable "Use Mock Data" 
2. Load data and play
3. Verify signals and trade execution visually
4. Check PnL calculations

## Troubleshooting

### "No models available"
**Solution**: Run a training script first
```bash
python -m src.train_predictive
```

### YFinance rate limits
**Solution**: Enable "Use Mock Data" for testing

### No trades executing
**Check**:
- Model confidence (green light indicator)
- Sensitivity threshold (lower = more signals)
- ATR > 0 (volatility present)

See [User Guide](docs/USER_GUIDE.md) for detailed troubleshooting.

## Performance & Results

See [docs/success_study.md](docs/success_study.md) for detailed performance analysis.

**Mean Reversion Strategy (Predictive Limit)**:
- Win Rate: 55-65%
- Average R-Multiple: 1.0-1.5
- Best Conditions: Ranging markets with moderate ATR

## Development

### Adding New Setups

1. Create labeling function in new training script
2. Train CNN model with your criteria
3. Save to `models/MySetup_CNN.pth`
4. Model automatically available in playback

See [Setup Library](docs/SETUP_LIBRARY.md) for examples.

### Adding Entry Mechanisms

Edit `frontend/src/components/YFinanceMode.tsx`:
1. Add option to `entryMechanism` state
2. Implement logic in `checkSignal()` function
3. Handle order creation for your mechanism

## Future Enhancements

- [ ] Market Close entry mechanism
- [ ] Additional setup types (breakout, gap fill)
- [ ] Multi-timeframe analysis
- [ ] Portfolio-level risk management
- [ ] Trade export and reporting
- [ ] Parameter optimization tools
- [ ] Real-time deployment infrastructure

## Contributing

This is a professional quant development platform. Contributions welcome:

1. Document your setup criteria clearly
2. Test thoroughly with playback mode
3. Verify no future leaking
4. Add to Setup Library documentation

## Key Files

- `src/config.py` - Configuration constants
- `src/generator.py` - Core generation logic (single and multi-day)
- `src/state_features.py` - Daily state extraction
- `src/pattern_library.py` - Clustering logic
- `src/model_inference.py` - Real-time model inference
- `frontend/src/App.tsx` - Main UI
- `frontend/src/components/YFinanceMode.tsx` - Trading playback interface

## Credits

Built for professional quantitative trading development with focus on:
- Strict time-series processing (no future leaking)
- Visual trade validation
- Clean separation of concerns
- Professional documentation

## License

See LICENSE file for details.

---

**Need Help?** 
- Read the [User Guide](docs/USER_GUIDE.md)
- Check [Architecture Guide](docs/ARCHITECTURE.md)  
- Review [Setup Library](docs/SETUP_LIBRARY.md)
- Explore example test scripts in `tests/`

**Last Updated**: 2025-12-10

