# API Documentation

## Overview

The MES Trading System provides a RESTful API built with FastAPI. This document covers all available endpoints for both Pattern Generation and YFinance Playback modes.

**Base URL**: `http://localhost:8000`

---

## General Endpoints

### Health Check

**GET** `/health`

Check if the API is running.

**Response**:
```json
{
  "status": "ok"
}
```

---

## Pattern Generator Endpoints

### Get Available Dates

**GET** `/api/dates`

Returns list of available dates in the historical MES data.

**Response**:
```json
["2024-01-15", "2024-01-14", "2024-01-13", ...]
```

**Notes**:
- Dates are sorted newest first
- Only returns dates with sufficient data

---

### Get Candles for Date

**GET** `/api/candles`

Fetch historical OHLC data for a specific date and timeframe.

**Query Parameters**:
- `date` (string, required): Date in YYYY-MM-DD format
- `timeframe` (string, required): One of: "1m", "5m", "15m", "60m"

**Response**:
```json
{
  "data": [
    {
      "time": 1705305600,
      "open": 5800.25,
      "high": 5801.50,
      "low": 5799.75,
      "close": 5800.00,
      "volume": 1250
    },
    ...
  ]
}
```

---

### Get Candles for Date Range

**GET** `/api/candles-range`

Fetch continuous OHLC data for a date range.

**Query Parameters**:
- `start_date` (string, required): Start date (YYYY-MM-DD)
- `end_date` (string, required): End date (YYYY-MM-DD)
- `timeframe` (string, required): "1m", "5m", "15m", "60m"

**Response**: Same format as `/api/candles`

**Notes**:
- Includes weekends and gaps
- Maximum 60 days range recommended

---

### Generate Single Session

**POST** `/api/generate/session`

Generate synthetic price data for a single trading session.

**Request Body**:
```json
{
  "day_of_week": 0,
  "session_type": "RTH",
  "start_price": 5800.0,
  "date": "2024-01-15",
  "timeframe": "5m"
}
```

**Parameters**:
- `day_of_week` (int, required): 0=Monday through 6=Sunday
- `session_type` (string, required): "RTH" or "Globex"
- `start_price` (float, required): Starting price for generation
- `date` (string, optional): Reference date for context
- `timeframe` (string, required): Output timeframe

**Response**: Same format as `/api/candles`

---

### Generate Multi-Day Sequence

**POST** `/api/generate/multi-day`

Generate continuous multi-day synthetic sequence.

**Request Body**:
```json
{
  "num_days": 5,
  "session_type": "RTH",
  "start_price": 5800.0,
  "start_date": "2024-01-15",
  "timeframe": "5m"
}
```

**Parameters**:
- `num_days` (int, required): Number of days to generate (1-60)
- `session_type` (string, required): "RTH" or "Globex"
- `start_price` (float, required): Initial price
- `start_date` (string, required): Starting date
- `timeframe` (string, required): Output timeframe

**Response**: Continuous candle array spanning multiple days

---

## YFinance Playback Endpoints

### Get YFinance Candles

**GET** `/api/yfinance/candles`

Fetch real market data from Yahoo Finance.

**Query Parameters**:
- `symbol` (string, default="MES=F"): Ticker symbol
- `days` (int, default=5): Number of days to fetch (1-60)
- `timeframe` (string, default="1m"): "1m", "5m", "15m", "60m", "1h", "1d"

**Response**:
```json
{
  "data": [
    {
      "time": 1705305600,
      "open": 5800.25,
      "high": 5801.50,
      "low": 5799.75,
      "close": 5800.00,
      "volume": 1250
    },
    ...
  ],
  "symbol": "MES=F",
  "timeframe": "1m"
}
```

**Notes**:
- Data is cached for subsequent analysis calls
- 1m data limited to ~7 days by Yahoo Finance
- 5m data available for ~60 days
- Rate limits apply (use mock data for testing)

**Errors**:
- 400: Invalid symbol or no data returned
- 500: YFinance API error

---

### Get Mock Candles

**GET** `/api/yfinance/candles/mock`

Generate realistic mock OHLC data for testing.

**Query Parameters**:
- `bars` (int, default=500): Number of bars to generate
- `timeframe` (string, default="1m"): "1m", "5m", "15m", "60m", "1h"

**Response**: Same format as `/api/yfinance/candles`

**Notes**:
- No rate limits
- Generates realistic trends and volatility
- Perfect for testing without API calls
- Symbol returned as "MOCK"

---

### Analyze Candle with Model

**POST** `/api/yfinance/playback/analyze`

Run CNN model inference on a specific candle index.

**Query Parameters**:
- `candle_index` (int, required): Index in the loaded data array
- `model_name` (string, required): Model to use (e.g., "CNN_Predictive_5m")
- `symbol` (string, required): Symbol of loaded data
- `date` (string, required): Date in ISO format

**Request**: None (uses cached data from previous fetch/mock call)

**Response**:
```json
{
  "signal": {
    "type": "OCO_LIMIT",
    "prob": 0.65,
    "atr": 3.2,
    "limit_dist": 4.8,
    "current_price": 5800.0,
    "sell_limit": 5804.8,
    "buy_limit": 5795.2,
    "sl_dist": 3.2,
    "validity": 900
  }
}
```

**Signal Object**:
- `type` (string): Signal type ("OCO_LIMIT")
- `prob` (float): Model confidence (0.0-1.0)
- `atr` (float): Current ATR value
- `limit_dist` (float): Distance for limit orders
- `current_price` (float): Current closing price
- `sell_limit` (float): Sell limit price
- `buy_limit` (float): Buy limit price
- `sl_dist` (float): Stop loss distance
- `validity` (int): Order validity in seconds

**Response when no signal**:
```json
{
  "signal": null
}
```

**Errors**:
- 400: Data not cached (call fetch first), or symbol mismatch
- 400: Invalid candle index

**Notes**:
- Requires prior call to `/api/yfinance/candles` or `/api/yfinance/candles/mock`
- Uses last cached data for the symbol (or MOCK)
- Model must exist in `models/` directory
- Returns null if insufficient history for analysis

---

### Get Available Models

**GET** `/api/yfinance/models`

List all available trained models.

**Response**:
```json
{
  "models": [
    "CNN_Predictive",
    "CNN_Predictive_5m",
    "CNN_Classic",
    "CNN_Wide",
    "rejection_cnn_v1",
    "setup_cnn_v1"
  ]
}
```

**Notes**:
- Scans `models/` directory for .pth files
- Models are automatically available after training

---

## Model Training (CLI Only)

Training is performed via command-line scripts, not HTTP API.

### Train Predictive Model (1m)

```bash
python -m src.train_predictive
```

**Output**: `models/CNN_Predictive.pth`

### Train Predictive Model (5m)

```bash
python -m src.train_predictive_5m
```

**Output**: `models/CNN_Predictive_5m.pth`

---

## Data Models

### Candle Object

```typescript
interface Candle {
  time: number;        // Unix timestamp (seconds)
  open: number;        // Opening price
  high: number;        // High price
  low: number;         // Low price
  close: number;       // Closing price
  volume: number;      // Volume (optional, default 0)
  synthetic_day?: number;  // Day index for multi-day generation
}
```

### Trade Object (Frontend Only)

```typescript
interface Trade {
  id: string;
  direction: 'BUY' | 'SELL' | 'LONG' | 'SHORT';
  entry_price: number;
  sl_price: number;
  tp_price: number;
  entry_time: number;
  status: 'pending' | 'open' | 'closed';
  risk_amount: number;
  pnl?: number;
  exit_price?: number;
  exit_time?: number;
}
```

---

## Error Responses

All endpoints follow consistent error format:

```json
{
  "detail": "Error message describing the issue"
}
```

**Common HTTP Status Codes**:
- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found (date/model not found)
- 422: Validation Error (malformed request)
- 500: Internal Server Error

---

## Rate Limits

### YFinance API
- Fetching real data has rate limits imposed by Yahoo Finance
- Recommended: Use mock data for development/testing
- Limit requests to once every few seconds in production

### Mock Data
- No rate limits
- Can generate unlimited synthetic data

---

## Caching

### YFinance Data Cache
The API maintains an in-memory cache:

```python
_last_yf_cache = {
    "symbol": "MES=F",
    "data": DataFrame,
    "interval": "1m"
}
```

**Behavior**:
- Cache populated by `/api/yfinance/candles` or `/api/yfinance/candles/mock`
- Cache used by `/api/yfinance/playback/analyze`
- Cache accepts "MOCK" as a special symbol (always valid)
- Cache is lost on server restart

---

## Usage Examples

### Example 1: Pattern Generator Workflow

```bash
# 1. Get available dates
curl http://localhost:8000/api/dates

# 2. Fetch historical data
curl "http://localhost:8000/api/candles?date=2024-01-15&timeframe=5m"

# 3. Generate synthetic overlay
curl -X POST http://localhost:8000/api/generate/session \
  -H "Content-Type: application/json" \
  -d '{
    "day_of_week": 0,
    "session_type": "RTH",
    "start_price": 5800.0,
    "date": "2024-01-15",
    "timeframe": "5m"
  }'
```

### Example 2: YFinance Playback Workflow

```bash
# 1. Get available models
curl http://localhost:8000/api/yfinance/models

# 2. Load mock data (for testing)
curl "http://localhost:8000/api/yfinance/candles/mock?bars=500&timeframe=1m"

# 3. Analyze candle at index 250
curl -X POST "http://localhost:8000/api/yfinance/playback/analyze?candle_index=250&model_name=CNN_Predictive_5m&symbol=MOCK&date=2024-01-15T10:00:00"

# 4. Continue analyzing subsequent candles...
curl -X POST "http://localhost:8000/api/yfinance/playback/analyze?candle_index=251&model_name=CNN_Predictive_5m&symbol=MOCK&date=2024-01-15T10:01:00"
```

### Example 3: Real Market Data

```bash
# Fetch real MES data
curl "http://localhost:8000/api/yfinance/candles?symbol=MES=F&days=5&timeframe=1m"

# Analyze with model
curl -X POST "http://localhost:8000/api/yfinance/playback/analyze?candle_index=300&model_name=CNN_Predictive_5m&symbol=MES=F&date=2024-01-15T10:00:00"
```

---

## Frontend Integration

### Pattern Generator Client

```typescript
import { api } from './api/client';

// Get dates
const dates = await api.getDates();

// Get candles
const candles = await api.getCandles('2024-01-15', '5m');

// Generate session
const synthetic = await api.generateSession(0, 'RTH', 5800, '2024-01-15', '5m');
```

### YFinance Client

```typescript
import { yfinanceApi } from './api/yfinance';

// Fetch data
const result = await yfinanceApi.fetchData('MES=F', 5, '1m', false);

// Get models
const models = await yfinanceApi.getAvailableModels();

// Analyze candle
const signal = await yfinanceApi.analyzeCandle(250, 'CNN_Predictive_5m', 'MES=F', '2024-01-15');
```

---

## WebSocket Support

Currently not implemented. All operations are synchronous HTTP requests.

**Future Enhancement**: Real-time streaming for live playback.

---

## Authentication

Currently not required. All endpoints are public.

**Future Enhancement**: API key authentication for production deployments.

---

## Best Practices

### For Development
1. Use mock data (`/api/yfinance/candles/mock`) to avoid rate limits
2. Enable CORS for local frontend development (already configured)
3. Check health endpoint before starting work

### For Testing
1. Load small datasets first (100-500 bars)
2. Test with multiple models to compare
3. Verify cache is populated before analyzing

### For Production
1. Implement rate limiting on YFinance calls
2. Consider Redis for distributed caching
3. Add authentication layer
4. Monitor API usage and performance

---

## Troubleshooting

### "Data not cached" error
**Cause**: Tried to analyze before loading data  
**Solution**: Call `/api/yfinance/candles` or `/api/yfinance/candles/mock` first

### "Model not found" error
**Cause**: Model file doesn't exist  
**Solution**: Train the model first using CLI scripts

### YFinance timeout/rate limit
**Cause**: Too many requests to Yahoo Finance  
**Solution**: Use mock data or wait before retrying

### Empty data returned
**Cause**: Invalid symbol or date range  
**Solution**: Verify symbol format (e.g., "MES=F" not "MES")

---

## API Changelog

### v1.0 (Current)
- Pattern generation endpoints
- YFinance data fetching
- Model inference
- Mock data generation
- Basic caching

### Future (Planned)
- WebSocket streaming
- Authentication
- Advanced caching (Redis)
- Batch analysis endpoint
- Results export endpoint

---

**Last Updated**: 2025-12-10  
**Version**: 1.0
