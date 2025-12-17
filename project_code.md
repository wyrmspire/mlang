# Project Code Dump

## Project Structure
```
.
|____.git
|____.gitignore
|____analysis
| |____baseline_results.txt
| |____check_wicks.py
| |____final_report.py
| |____new_results.txt
| |____smoke_test_generator.py
| |____wick_report.txt
|____data
| |____processed
| | |____collected_trades.parquet
| | |____continuous_1m.parquet
| | |____engulfing_trades.parquet
| | |____features_5m_atr15m.parquet
| | |____labeled_3m_rejection.parquet
| | |____labeled_continuous.parquet
| | |____labeled_predictive.parquet
| | |____labeled_predictive_5m.parquet
| | |____labeled_rejections.parquet
| | |____labeled_rejections_5m.parquet
| | |____mes_1min_parquet
| | | |____mes_1min_all.parquet
| | |____mes_day_state.parquet
| | |____mes_hour_features_parquet
| | | |____mes_hour_features.parquet
| | |____mes_patterns
| | | |____cluster_metadata.json
| | | |____mes_pattern_library.parquet
| | |____mes_setup_clusters.parquet
| | |____mes_setup_decision_tree.json
| | |____mes_setup_features.parquet
| | |____mes_setup_rules.json
| | |____random_tilt_trades.parquet
| | |____rejection_strategy_results.csv
| | |____smart_reverse_trades.parquet
| | |____smart_verification_trades.parquet
| |____raw
| | |____continuous_contract.json
|____diff.sh
|____docs
| |____API_REFERENCE.md
| |____ARCHITECTURE.md
| |____QUICK_START.md
| |____rejection_strategy_spec.md
| |____REORGANIZATION_SUMMARY.md
| |____SETUP_LIBRARY.md
| |____success_study.md
| |____USER_GUIDE.md
|____frontend
| |____index.html
| |____node_modules
| |____package-lock.json
| |____package.json
| |____src
| | |____api
| | | |____client.ts
| | | |____yfinance.ts
| | |____App.tsx
| | |____components
| | | |____ChartPanel.tsx
| | | |____SidebarControls.tsx
| | | |____YFinanceChart.tsx
| | | |____YFinanceMode.tsx
| | | |____YFinancePage.tsx
| | |____main.tsx
| |____tsconfig.json
| |____tsconfig.node.json
| |____vite.config.ts
|____gitr.sh
|____inverse_results.log
|____inverse_strategy_performance.png
|____limits_test_output.txt
|____logs
|____miner_5m.log
|____miner_error.log
|____models
| |____cnn_3m_rejection.pth
| |____CNN_Classic.pth
| |____CNN_Predictive.pth
| |____CNN_Predictive_5m.pth
| |____CNN_Wide.pth
| |____Feature_MLP.pth
| |____LSTM_Seq.pth
| |____rejection_cnn_v1.pth
| |____setup_cnn_v1.pth
|____phase2_comparison.png
|____printcode.sh
|____project_code.md
|____README.md
|____rejection_backtest_results.csv
|____requirements.txt
|____src
| |____api.py
| |____config.py
| |____data_loader.py
| |____debug_inference_check.py
| |____debug_miner.py
| |____feature_engineering.py
| |____generator.py
| |____models
| | |____cnn_model.py
| | |____train_cnn.py
| | |____train_rejection_cnn.py
| | |____variants.py
| | |______pycache__
| |____model_inference.py
| |____pattern_library.py
| |____pattern_miner.py
| |____preprocess.py
| |____sanity_check.py
| |____scripts
| | |____inspect_parquet.py
| | |____mine_3m_rejection.py
| | |____mine_continuous.py
| | |____mine_predictive.py
| | |____mine_predictive_5m.py
| | |____process_continuous.py
| | |____test_yfinance_limits.py
| |____setup_miner.py
| |____state_features.py
| |____strategies
| | |____collector.py
| | |____inverse_strategy.py
| | |____random_tilt.py
| | |____rejection_strategy.py
| | |____smart_cnn.py
| | |____smart_reverse.py
| |____test_3m_strategy.py
| |____train_3m_cnn.py
| |____train_models_phase2.py
| |____train_predictive.py
| |____train_predictive_5m.py
| |____utils
| | |____logging_utils.py
| | |______init__.py
| | |______pycache__
| |____yfinance_loader.py
| |______init__.py
| |______pycache__
|____start.sh
|____stderr.trace
|____stderr_5m.trace
|____stderr_5m_sliced.trace
|____stderr_clean.trace
|____stderr_debug.trace
|____stderr_robust.trace
|____stderr_unbuffered.trace
|____strategy_log.txt
|____strategy_output.log
|____strat_5m.log
|____strat_run.log
|____tests
| |____test_gold.py
| |____test_inverse_single_position.py
| |____test_inverse_streaming.py
| |____test_mnq.py
| |____test_mnq_original.py
| |____test_phase2_models.py
| |____test_predictive.py
| |____test_predictive_60d.py
| |____test_smoke.py
|____uvicorn.pid
|____uvicorn_8001.pid
```

## File Contents

### ./analysis/check_wicks.py
```py
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.generator import get_generator
from src.config import ONE_MIN_PARQUET_DIR

def calc_wick_stats(df):
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['wick'] = df['range'] - df['body']
    # Avoid div by zero
    df['wick_ratio'] = np.where(df['range'] > 1e-6, df['wick'] / df['range'], 0)
    
    return {
        "mean_range": df['range'].mean(),
        "mean_body": df['body'].mean(),
        "mean_wick": df['wick'].mean(),
        "mean_wick_ratio": df['wick_ratio'].mean(),
        "std_wick_ratio": df['wick_ratio'].std()
    }

def resample_data(df, interval):
    resampled = df.set_index('time').resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna().reset_index()
    return resampled

def main():
    # Redirect print to a file
    with open("analysis/wick_report.txt", "w") as f:
        # Helper to print to both
        def log(msg):
            print(msg)
            f.write(msg + "\n")
            
        log("--- Wick Analysis (Multi-Timeframe) ---")
        
        # 1. Real Data
        path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if path.exists():
            log("Loading Real Data...")
            df_real_1m = pd.read_parquet(path)
        else:
            log("Real data not found.")
            df_real_1m = None

        # 2. Synthetic Data
        log("Generating Synthetic Data (20 days)...")
        gen = get_generator()
        try:
            df_synth_1m = gen.generate_multi_day(num_days=20, start_date="2024-01-01")
        except Exception as e:
            log(f"Generation failed: {e}")
            df_synth_1m = pd.DataFrame()

        if df_synth_1m.empty:
            log("No synthetic data generated.")
            return

        timeframes = ['1min', '5min', '15min', '1h']
        
        for tf in timeframes:
            log(f"\n[{tf} Timeframe]")
            
            # Prepare Real
            if df_real_1m is not None:
                if tf == '1min':
                    dfr = df_real_1m.copy()
                else:
                    dfr = resample_data(df_real_1m, tf)
                real_stats = calc_wick_stats(dfr)
            else:
                real_stats = None
                
            # Prepare Synth
            if tf == '1min':
                dfs = df_synth_1m.copy()
            else:
                dfs = resample_data(df_synth_1m, tf)
            synth_stats = calc_wick_stats(dfs)
            
            # Print
            if real_stats:
                log(f"{'Metric':<20} | {'Real':<10} | {'Synth':<10} | {'Ratio'}")
                log("-" * 50)
                for k in ['mean_wick_ratio', 'mean_range']:
                    r = real_stats.get(k, 0)
                    s = synth_stats.get(k, 0)
                    ratio = s/r if r!=0 else 0
                    log(f"{k:<20} | {r:<10.4f} | {s:<10.4f} | {ratio:.2f}x")
            else:
                log(f"Synth Stats ({tf}):")
                log(str(synth_stats))


if __name__ == "__main__":
    main()
```

### ./analysis/final_report.py
```py
import pandas as pd
import numpy as np

path = 'data/processed/smart_reverse_trades.parquet'
df = pd.read_parquet(path)

print(f"Total Intervals: {len(df)}")
p_follow = 0.43 # Threshold used
p_fade = 0.39   # Threshold used

# 1. Follow Only
# If p_long > p_follow -> PnL Long
# If p_short > p_follow -> PnL Short
# Tie-break? Max freq.
# df columns: time, p_long, p_short, pnl_long, pnl_short, outcome_long, outcome_short

# Vectorized Analysis
# Best Prob
best_prob = df[['p_long', 'p_short']].max(axis=1)
long_best = df['p_long'] >= df['p_short']

# Follow
mask_follow = best_prob > p_follow
pnl_follow = np.where(long_best, df['pnl_long'], df['pnl_short'])
total_follow = pnl_follow[mask_follow].sum()
count_follow = mask_follow.sum()

# Fade Losers (Prob < p_fade)
# If p_long < p_fade -> Short (Take Short PnL)
# If p_short < p_fade -> Long (Take Long PnL)
# Prioritize? If both bad? 
# Logic in script: 
# If p_long < p_fade: Fade Long (Go Short).
# If p_short < p_fade: Fade Short (Go Long).
# Both? Add both? (Script likely added both to hybrid_pnl or distinct).
# Script logic:
# if pl < low: fade_losers_pnl += pnl_short
# if ps < low: fade_losers_pnl += pnl_long

mask_fade_long = df['p_long'] < p_fade
pnl_fade_long = df['pnl_short'][mask_fade_long].sum()
count_fade_long = mask_fade_long.sum()

mask_fade_short = df['p_short'] < p_fade
pnl_fade_short = df['pnl_long'][mask_fade_short].sum()
count_fade_short = mask_fade_short.sum()

total_fade = pnl_fade_long + pnl_fade_short
count_fade = count_fade_long + count_fade_short

# Hybrid
total_hybrid = total_follow + total_fade
count_hybrid = count_follow + count_fade

# Fade Winners (Inverse of Follow)
# If Follow would go Long, we go Short.
pnl_fade_winners_arr = np.where(long_best, df['pnl_short'], df['pnl_long'])
total_fade_winners = pnl_fade_winners_arr[mask_follow].sum()

print("-" * 30)
print(f"1. Follow Only (> {p_follow}):")
print(f"   Trades: {count_follow} | PnL: {total_follow:.2f}")
print("-" * 30)
print(f"2. Fade Losers (< {p_fade}):")
print(f"   Trades: {count_fade} | PnL: {total_fade:.2f}")
print("-" * 30)
print(f"3. Hybrid (Follow + Fade Losers):")
print(f"   Trades: {count_hybrid} | PnL: {total_hybrid:.2f}")
print("-" * 30)
print(f"4. Fade Winners (Contrarian):")
print(f"   Trades: {count_follow} | PnL: {total_fade_winners:.2f}")
```

### ./analysis/smoke_test_generator.py
```py
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.generator import get_generator
from src.config import ONE_MIN_PARQUET_DIR

def get_real_stats():
    path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    if not path.exists():
        print("Real data not found.")
        return None
    
    df = pd.read_parquet(path)
    df['date'] = df['time'].dt.date
    
    # 1. Daily Range
    daily = df.groupby('date').agg({
        'high': 'max', 
        'low': 'min', 
        'open': 'first', 
        'close': 'last'
    })
    daily['range'] = daily['high'] - daily['low']
    daily['abs_ret'] = (daily['close'] - daily['open']).abs()
    
    # 2. Consecutive Candles (Resampled to 5m for "streakiness")
    df_5m = df.set_index('time').resample('5min').agg({
        'open': 'first', 'close': 'last'
    }).dropna()
    
    df_5m['up'] = df_5m['close'] > df_5m['open']
    # Group consecutive
    # logic: compare vs shift, cumsum to make groups
    df_5m['grp'] = (df_5m['up'] != df_5m['up'].shift()).cumsum()
    streaks = df_5m.groupby('grp')['up'].count()
    
    return {
        "daily_range_mean": daily['range'].mean(),
        "daily_range_std": daily['range'].std(),
        "daily_move_mean": daily['abs_ret'].mean(),
        "max_streak_5m": streaks.max(),
        "avg_streak_5m": streaks.mean(),
        "streak_95pct": streaks.quantile(0.95)
    }

def get_synth_stats(num_days=10):
    gen = get_generator()
    # Generate 10 days
    # To get stable stats, maybe more? 10 is fine for speed.
    try:
        df = gen.generate_multi_day(num_days=num_days, start_date="2024-01-01") # Arbitrary start if needed, or None
    except Exception as e:
        print(f"Generation failed: {e}")
        return None
        
    if df.empty:
        return None

    # Synth data doesn't have 'date' column usually, but has 'synthetic_day' (0..N)
    # create synthetic date
    df['date'] = df['synthetic_day']
    
    # 1. Daily Range
    daily = df.groupby('date').agg({
        'high': 'max', 
        'low': 'min', 
        'open': 'first', 
        'close': 'last'
    })
    daily['range'] = daily['high'] - daily['low']
    daily['abs_ret'] = (daily['close'] - daily['open']).abs()

    # 2. Streaks (5m)
    # We need to respect day boundaries for accurate resampling? 
    # Or just treat as continuous stream? Continuous is approx ok for streaks.
    df_5m = df.set_index('time').resample('5min').agg({
        'open': 'first', 'close': 'last'
    }).dropna()
    
    df_5m['up'] = df_5m['close'] > df_5m['open']
    df_5m['grp'] = (df_5m['up'] != df_5m['up'].shift()).cumsum()
    streaks = df_5m.groupby('grp')['up'].count()

    return {
        "daily_range_mean": daily['range'].mean(),
        "daily_range_std": daily['range'].std(),
        "daily_move_mean": daily['abs_ret'].mean(),
        "max_streak_5m": streaks.max(),
        "avg_streak_5m": streaks.mean(),
        "streak_95pct": streaks.quantile(0.95),
        "total_drift": (df['close'].iloc[-1] - df['open'].iloc[0])
    }

def print_comparison(real, synth):
    print(f"{'Metric':<25} | {'Real':<15} | {'Synthetic':<15} | {'Diff'}")
    print("-" * 70)
    
    metrics = ['daily_range_mean', 'daily_range_std', 'daily_move_mean', 'max_streak_5m', 'avg_streak_5m', 'streak_95pct']
    
    for m in metrics:
        r_val = real.get(m, 0)
        s_val = synth.get(m, 0)
        diff = ((s_val - r_val) / r_val) * 100 if r_val != 0 else 0
        print(f"{m:<25} | {r_val:<15.2f} | {s_val:<15.2f} | {diff:+.1f}%")

    print("-" * 70)
    print(f"Total Drift (10d): {synth.get('total_drift', 0):.2f}")

if __name__ == "__main__":
    print("Computing Real Stats...")
    real = get_real_stats()
    
    print("Generating Synthetic Data...")
    synth = get_synth_stats(20) # 20 days for better average
    
    if real and synth:
        print_comparison(real, synth)
```

### ./frontend/package.json
```json
{
    "name": "frontend",
    "private": true,
    "version": "0.0.0",
    "type": "module",
    "scripts": {
        "dev": "vite",
        "build": "tsc && vite build",
        "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
        "preview": "vite preview"
    },
    "dependencies": {
        "axios": "^1.6.0",
        "date-fns": "^2.30.0",
        "lightweight-charts": "^4.1.0",
        "lucide-react": "^0.290.0",
        "react": "^18.2.0",
        "react-dom": "^18.2.0"
    },
    "devDependencies": {
        "@types/react": "^18.2.15",
        "@types/react-dom": "^18.2.7",
        "@vitejs/plugin-react": "^4.0.3",
        "typescript": "^5.0.2",
        "vite": "^4.4.5"
    }
}```

### ./frontend/src/api/client.ts
```ts
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export interface Candle {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    is_synthetic?: boolean;
    synthetic_day?: number;
}

export const api = {
    getDates: async () => {
        const res = await axios.get<string[]>(`${API_URL}/api/dates`);
        return res.data;
    },

    getCandles: async (date: string, timeframe: string = '1m', sessionType: string = 'all') => {
        const res = await axios.get<{ data: Candle[] }>(`${API_URL}/api/candles`, {
            params: { date, timeframe, session_type: sessionType }
        });
        return res.data.data;
    },

    getCandlesRange: async (startDate: string, endDate: string, timeframe: string, sessionType: string = 'all') => {
        const res = await axios.get<{ data: Candle[] }>(`${API_URL}/api/candles-range`, {
            params: { start_date: startDate, end_date: endDate, timeframe, session_type: sessionType }
        });
        return res.data.data;
    },

    generateSession: async (dayOfWeek: number, sessionType: string, startPrice: number, date: string, timeframe: string) => {
        const res = await axios.post<{ data: Candle[] }>(`${API_URL}/api/generate/session`, {
            day_of_week: dayOfWeek,
            session_type: sessionType,
            start_price: startPrice,
            date: date,
            timeframe: timeframe
        });
        return res.data.data;
    },

    generateMultiDay: async (numDays: number, sessionType: string, startPrice: number, startDate: string, timeframe: string) => {
        const res = await axios.post<{ data: Candle[] }>(`${API_URL}/api/generate/multi-day`, {
            num_days: numDays,
            session_type: sessionType,
            initial_price: startPrice,
            start_date: startDate,
            timeframe: timeframe
        });
        return res.data.data;
    },

    getYFinanceCandles: async (symbol: string = "MES=F", days: number = 5, timeframe: string = "1m") => {
        const res = await axios.get<{ data: Candle[], symbol: string, timeframe: string }>(`${API_URL}/api/yfinance/candles`, {
            params: { symbol, days, timeframe }
        });
        return res.data.data;
    }
};
```

### ./frontend/src/api/yfinance.ts
```ts
import axios from 'axios';
import { Candle } from './client';

const API_URL = 'http://localhost:8000';

export interface TradeSignal {
    type?: string; // 'OCO_LIMIT' | 'MARKET'
    direction: string;
    entry_price: number;
    sl_price: number; // For direct market orders
    tp_price: number; // For direct market orders
    confidence: number;
    entry_time: number;
    atr?: number;
    setup_type?: string;

    // OCO Fields
    sell_limit?: number;
    buy_limit?: number;
    sl_dist?: number;
    limit_dist?: number;
    current_price?: number;
    prob?: number;
}

export interface YFinanceData {
    success: boolean;
    data: Candle[];
    dates: string[];
    symbol: string;
    message?: string;
}

export interface Trade {
    id: string;
    direction: string;
    entry_price: number;
    entry_time: number;
    sl_price: number;
    tp_price: number;
    exit_price?: number;
    exit_time?: number;
    pnl?: number;
    status: 'open' | 'closed' | 'pending';
    risk_amount: number;
}

export const yfinanceApi = {
    fetchData: async (symbol: string = 'ES=F', daysBack: number = 7, interval: string = '1m', useMock: boolean = false): Promise<YFinanceData> => {
        const BARS_PER_DAY_1M = 390; // Trading hours: 6.5 hours * 60 minutes
        const BARS_PER_DAY_5M = 78;  // Trading hours: 6.5 hours * 12 (5-min bars per hour)

        const endpoint = useMock ? '/api/yfinance/candles/mock' : '/api/yfinance/candles';
        const params = useMock
            ? { bars: daysBack * (interval === '1m' ? BARS_PER_DAY_1M : BARS_PER_DAY_5M), timeframe: interval }
            : { symbol, days: daysBack, timeframe: interval };

        const res = await axios.get<YFinanceData>(`${API_URL}${endpoint}`, { params });
        return res.data;
    },

    getAvailableModels: async (): Promise<string[]> => {
        const res = await axios.get<{ models: string[] }>(`${API_URL}/api/yfinance/models`);
        return res.data.models;
    },

    analyzeCandle: async (
        candleIndex: number,
        modelName: string,
        symbol: string,
        date: string
    ): Promise<{ signal: TradeSignal | null }> => {
        const res = await axios.post<{ signal: TradeSignal | null }>(
            `${API_URL}/api/yfinance/playback/analyze`,
            null,
            {
                params: {
                    candle_index: candleIndex,
                    model_name: modelName,
                    symbol,
                    date
                }
            }
        );
        // Return matching the structure expected by YFinanceMode (wrapper object)
        // Or change YFinanceMode. currently YFinanceMode expects `resp.signal`.
        // If I return `res.data`, it contains `signal`.
        return res.data;
    }
};
```

### ./frontend/src/App.tsx
```tsx
import React, { useState, useEffect } from 'react';
import { SidebarControls } from './components/SidebarControls';
import { ChartPanel } from './components/ChartPanel';
import { YFinanceMode } from './components/YFinanceMode';
import { api, Candle } from './api/client';

type AppMode = 'pattern' | 'yfinance';

const App: React.FC = () => {
    // Mode selection
    const [mode, setMode] = useState<AppMode>('pattern');

    // Data State
    const [dates, setDates] = useState<string[]>([]);

    // Single Day State
    const [selectedDate, setSelectedDate] = useState<string>('');
    const [candles, setCandles] = useState<Candle[]>([]);
    const [syntheticData, setSyntheticData] = useState<Candle[]>([]); // Single day overlay

    // Multi Day State
    const [isMultiDay, setIsMultiDay] = useState(false);
    const [startDate, setStartDate] = useState<string>('');
    const [numDays, setNumDays] = useState<number>(5);
    const [multiDayReal, setMultiDayReal] = useState<Candle[]>([]);
    const [multiDaySynth, setMultiDaySynth] = useState<Candle[]>([]);

    // UI State
    const [timeframe, setTimeframe] = useState<string>('5m');
    const [isGenerating, setIsGenerating] = useState(false);

    // Gen Params
    const [genDayOfWeek, setGenDayOfWeek] = useState<number>(0);
    const [genSessionType, setGenSessionType] = useState<string>('RTH');

    // Initial Load
    useEffect(() => {
        if (mode === 'pattern') {
            loadDates();
        }
    }, [mode]);

    const loadDates = async () => {
        try {
            const d = await api.getDates();
            setDates(d);
            if (d.length > 0) {
                // Default to newest for Single Day
                setSelectedDate(d[0]);

                // Default to ~30 days back for Multi-Day so we have forward history
                const offset = Math.min(30, d.length - 1);
                setStartDate(d[offset]);
            }
        } catch (e) {
            console.error(e);
        }
    };

    // Load Data Effect
    useEffect(() => {
        if (!selectedDate && !startDate && dates.length === 0) return;

        const fetchData = async () => {
            try {
                if (!isMultiDay) {
                    // Single Day Mode
                    if (selectedDate) {
                        const data = await api.getCandles(selectedDate, timeframe);
                        setCandles(data);
                    }
                } else {
                    // Multi Day Mode
                    if (startDate && dates.length > 0) {
                        const startIdx = dates.indexOf(startDate);
                        if (startIdx >= 0) {
                            // Calculate "Future" End Date (chronologically later => smaller index)
                            let endIdx = startIdx - numDays + 1;
                            if (endIdx < 0) endIdx = 0; // Clamp to newest available

                            const endDate = dates[endIdx];

                            if (startIdx >= endIdx) {
                                const data = await api.getCandlesRange(startDate, endDate, timeframe);
                                setMultiDayReal(data);
                            }
                        }
                    }
                }
            } catch (e) {
                console.error(e);
            }
        };

        fetchData();

        // Auto-set generator Day of Week to match selected date (Single Mode only)
        if (!isMultiDay && selectedDate) {
            try {
                const [y, m, d] = selectedDate.split('-').map(Number);
                const dateObj = new Date(y, m - 1, d);
                let dow = dateObj.getDay();
                dow = dow === 0 ? 6 : dow - 1;
                setGenDayOfWeek(dow);
            } catch (e) { console.error(e); }
        }

    }, [selectedDate, startDate, timeframe, isMultiDay, numDays, dates]);

    // Clear synthetic data when timeframe changes
    useEffect(() => {
        setSyntheticData([]);
        setMultiDaySynth([]);
    }, [timeframe]);


    const handleGenerate = async () => {
        setIsGenerating(true);
        try {
            if (!isMultiDay) {
                // Single Day
                const startPrice = candles.length > 0 ? candles[0].open : 5800;
                const data = await api.generateSession(
                    genDayOfWeek,
                    genSessionType,
                    startPrice,
                    selectedDate,
                    timeframe
                );
                setSyntheticData(data);
            } else {
                // Multi Day
                const startPrice = multiDayReal.length > 0 ? multiDayReal[0].open : 5800;
                const data = await api.generateMultiDay(
                    numDays,
                    genSessionType,
                    startPrice,
                    startDate,
                    timeframe
                );
                setMultiDaySynth(data);
            }
        } catch (e) {
            console.error("Generation failed", e);
            alert("Generation failed. Check console/backend.");
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#1E1E1E', color: '#EEE' }}>
            {/* Top Menu Bar */}
            <div style={{
                display: 'flex',
                gap: '0',
                background: '#252526',
                borderBottom: '1px solid #333',
                padding: '0'
            }}>
                <button
                    onClick={() => setMode('pattern')}
                    style={{
                        padding: '15px 30px',
                        background: mode === 'pattern' ? '#1E1E1E' : 'transparent',
                        color: mode === 'pattern' ? '#4FC3F7' : '#AAA',
                        border: 'none',
                        borderBottom: mode === 'pattern' ? '2px solid #4FC3F7' : '2px solid transparent',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: mode === 'pattern' ? 'bold' : 'normal',
                        transition: 'all 0.2s'
                    }}
                >
                    Pattern Generator
                </button>
                <button
                    onClick={() => setMode('yfinance')}
                    style={{
                        padding: '15px 30px',
                        background: mode === 'yfinance' ? '#1E1E1E' : 'transparent',
                        color: mode === 'yfinance' ? '#4FC3F7' : '#AAA',
                        border: 'none',
                        borderBottom: mode === 'yfinance' ? '2px solid #4FC3F7' : '2px solid transparent',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: mode === 'yfinance' ? 'bold' : 'normal',
                        transition: 'all 0.2s'
                    }}
                >
                    YFinance Playback
                </button>
            </div>

            {/* Content */}
            {mode === 'yfinance' ? (
                <YFinanceMode onBack={() => setMode('pattern')} />
            ) : (
                <div style={{ display: 'flex', height: 'calc(100vh - 52px)' }}>
                    <SidebarControls
                        dates={dates}
                        selectedDate={selectedDate}
                        onDateChange={setSelectedDate}
                        timeframe={timeframe}
                        onTimeframeChange={setTimeframe}
                        onGenerate={handleGenerate}
                        isGenerating={isGenerating}
                        dayOfWeek={genDayOfWeek}
                        setDayOfWeek={setGenDayOfWeek}
                        sessionType={genSessionType}
                        setSessionType={setGenSessionType}
                        onClearSynthetic={() => {
                            setSyntheticData([]);
                            setMultiDaySynth([]);
                        }}

                        // Multi
                        isMultiDay={isMultiDay}
                        setIsMultiDay={setIsMultiDay}
                        numDays={numDays}
                        setNumDays={setNumDays}
                        startDate={startDate}
                        setStartDate={setStartDate}
                    />

                    <div style={{ flex: 1, padding: '10px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {!isMultiDay ? (
                            // Single Pane with Overlay
                            <div style={{ flex: 1, overflow: 'hidden' }}>
                                <ChartPanel data={candles} syntheticData={syntheticData} title={`Historical: ${selectedDate}`} />
                            </div>
                        ) : (
                            // Split Pane
                            <>
                                <div style={{ flex: 1, border: '1px solid #444', overflow: 'hidden' }}>
                                    <ChartPanel data={multiDayReal} title={`Real Data (${startDate} + ${numDays} days)`} />
                                </div>
                                <div style={{ flex: 1, border: '1px solid #444', overflow: 'hidden' }}>
                                    <ChartPanel data={multiDaySynth} title="Synthetic Generated Path" />
                                </div>
                            </>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default App;
```

### ./frontend/src/components/ChartPanel.tsx
```tsx
import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts';
import { Candle } from '../api/client';

interface ChartPanelProps {
    data: Candle[];
    syntheticData?: Candle[];
    title?: string;
}

export const ChartPanel: React.FC<ChartPanelProps> = ({ data, syntheticData, title }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const syntheticSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight, // Use actual height
            layout: {
                background: { type: ColorType.Solid, color: '#1E1E1E' },
                textColor: '#DDD',
            },
            grid: {
                vertLines: { color: '#333' },
                horzLines: { color: '#333' },
            },
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            }
        });

        // ... (series creation remains same)
        const series = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });

        const syntheticSeries = chart.addCandlestickSeries({
            upColor: '#4caf50',
            downColor: '#ff9800',
            borderVisible: true,
            borderColor: '#ffff00',
            wickUpColor: '#4caf50',
            wickDownColor: '#ff9800',
        });

        chartRef.current = chart;
        seriesRef.current = series;
        syntheticSeriesRef.current = syntheticSeries;

        // Resize Observer
        const resizeObserver = new ResizeObserver(entries => {
            if (!entries || entries.length === 0) return;
            const { width, height } = entries[0].contentRect;
            chart.applyOptions({ width, height });
            chart.timeScale().fitContent();
        });

        resizeObserver.observe(containerRef.current);

        return () => {
            resizeObserver.disconnect();
            chart.remove();
        };
    }, []);

    // ... (rest of effects)

    useEffect(() => {
        if (seriesRef.current && data.length > 0) {
            // Sort just in case
            const sorted = [...data].sort((a, b) => a.time - b.time);
            seriesRef.current.setData(sorted as any);
            chartRef.current?.timeScale().fitContent();
        } else if (seriesRef.current) {
            seriesRef.current.setData([]);
        }
    }, [data]);

    useEffect(() => {
        if (syntheticSeriesRef.current) {
            if (syntheticData && syntheticData.length > 0) {
                const sorted = [...syntheticData].sort((a, b) => a.time - b.time);
                syntheticSeriesRef.current.setData(sorted as any);
            } else {
                syntheticSeriesRef.current.setData([]);
            }
        }
    }, [syntheticData]);

    return (
        <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
            {title && <div style={{ padding: '5px', background: '#333', fontSize: '12px' }}>{title}</div>}
            <div ref={containerRef} style={{ width: '100%', flex: 1, minHeight: '0', overflow: 'hidden' }} />
        </div>
    );
};
```

### ./frontend/src/components/SidebarControls.tsx
```tsx
import React from 'react';

interface SidebarControlsProps {
    dates: string[];
    selectedDate: string;
    onDateChange: (d: string) => void;
    timeframe: string;
    onTimeframeChange: (t: string) => void;
    onGenerate: () => void;
    isGenerating: boolean;
    dayOfWeek: number;
    setDayOfWeek: (d: number) => void;
    sessionType: string;
    setSessionType: (s: string) => void;
    onClearSynthetic: () => void;
    onOpenYFinance?: () => void;

    // Multi-Day Props
    isMultiDay: boolean;
    setIsMultiDay: (b: boolean) => void;
    numDays: number;
    setNumDays: (n: number) => void;
    startDate: string;         // For multi-day range start
    setStartDate: (d: string) => void;
}

export const SidebarControls: React.FC<SidebarControlsProps> = ({
    dates, selectedDate, onDateChange,
    timeframe, onTimeframeChange,
    onGenerate, isGenerating,
    dayOfWeek, setDayOfWeek,
    sessionType, setSessionType,
    onClearSynthetic,
    onOpenYFinance,
    isMultiDay, setIsMultiDay,
    numDays, setNumDays,
    startDate, setStartDate
}) => {
    return (
        <div style={{ width: '250px', background: '#252526', padding: '10px', borderRight: '1px solid #333' }}>
            <h3>Data Controls</h3>

            <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>Mode</label>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                        onClick={() => setIsMultiDay(false)}
                        style={{ flex: 1, background: !isMultiDay ? '#007ACC' : '#444', color: '#FFF', border: 'none', padding: '5px' }}
                    >
                        Single
                    </button>
                    <button
                        onClick={() => setIsMultiDay(true)}
                        style={{ flex: 1, background: isMultiDay ? '#007ACC' : '#444', color: '#FFF', border: 'none', padding: '5px' }}
                    >
                        Multi-Day
                    </button>
                </div>
            </div>

            {!isMultiDay ? (
                // SINGLE DAY CONTROLS
                <>
                    <div style={{ marginBottom: '15px' }}>
                        <label style={{ display: 'block', marginBottom: '5px' }}>Date</label>
                        <select
                            value={selectedDate}
                            onChange={e => onDateChange(e.target.value)}
                            style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}
                        >
                            {dates.map(d => <option key={d} value={d}>{d}</option>)}
                        </select>
                    </div>
                </>
            ) : (
                // MULTI DAY CONTROLS
                <>
                    <div style={{ marginBottom: '15px' }}>
                        <label style={{ display: 'block', marginBottom: '5px' }}>Start Date</label>
                        <select
                            value={startDate}
                            onChange={e => setStartDate(e.target.value)}
                            style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}
                        >
                            {dates.map(d => <option key={d} value={d}>{d}</option>)}
                        </select>
                    </div>
                    <div style={{ marginBottom: '15px' }}>
                        <label style={{ display: 'block', marginBottom: '5px' }}>Num Days</label>
                        <input
                            type="number"
                            value={numDays}
                            onChange={e => setNumDays(Number(e.target.value))}
                            min={2} max={90}
                            style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}
                        />
                    </div>
                </>
            )}

            <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>Timeframe</label>
                <select
                    value={timeframe}
                    onChange={e => onTimeframeChange(e.target.value)}
                    style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}
                >
                    <option value="1m">1 Minute</option>
                    <option value="5m">5 Minutes</option>
                    <option value="15m">15 Minutes</option>
                    <option value="30m">30 Minutes</option>
                    <option value="60m">1 Hour</option>
                </select>
            </div>

            <hr style={{ borderColor: '#444', margin: '15px 0' }} />

            <h3>Generator</h3>

            <div style={{ marginBottom: '10px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>Session Type</label>
                <select value={sessionType} onChange={e => setSessionType(e.target.value)} style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}>
                    <option value="RTH">RTH (08:30-15:15)</option>
                    <option value="overnight">Overnight (17:00-08:30)</option>
                </select>
            </div>

            {!isMultiDay && (
                <div style={{ marginBottom: '10px' }}>
                    <label style={{ display: 'block', marginBottom: '5px' }}>Day of Week</label>
                    <select value={dayOfWeek} onChange={e => setDayOfWeek(Number(e.target.value))} style={{ width: '100%', padding: '5px', background: '#333', color: '#EEE', border: '1px solid #555' }}>
                        <option value={0}>Monday</option>
                        <option value={1}>Tuesday</option>
                        <option value={2}>Wednesday</option>
                        <option value={3}>Thursday</option>
                        <option value={4}>Friday</option>
                        <option value={5}>Saturday</option>
                        <option value={6}>Sunday</option>
                    </select>
                </div>
            )}

            <button
                onClick={onGenerate}
                disabled={isGenerating}
                style={{
                    width: '100%', padding: '10px',
                    background: isGenerating ? '#555' : '#0E639C',
                    color: 'white', border: 'none', cursor: isGenerating ? 'wait' : 'pointer',
                    marginTop: '10px'
                }}
            >
                {isGenerating ? 'Generating...' : isMultiDay ? 'Generate Sequence' : 'Generate Session'}
            </button>

            <button
                onClick={onClearSynthetic}
                style={{
                    width: '100%', padding: '10px',
                    background: '#333',
                    color: '#CCC', border: '1px solid #555', cursor: 'pointer',
                    marginTop: '5px'
                }}
            >
                Clear Synthetic
            </button>

            {onOpenYFinance && (
                <button
                    onClick={onOpenYFinance}
                    style={{
                        width: '100%', padding: '10px',
                        background: '#D4A500',
                        color: '#000', border: 'none', cursor: 'pointer',
                        marginTop: '10px',
                        fontWeight: 'bold'
                    }}
                >
                    📈 YFinance Data
                </button>
            )}
        </div>
    );
};
```

### ./frontend/src/components/YFinanceChart.tsx
```tsx
import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts';
import { Candle } from '../api/client';
import { Trade } from '../api/yfinance';

interface YFinanceChartProps {
    data: Candle[];
    trades: Trade[];
    currentPrice: number;
}

export const YFinanceChart: React.FC<YFinanceChartProps> = ({ data, trades, currentPrice }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const markersRef = useRef<any[]>([]);

    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight,
            layout: {
                background: { type: ColorType.Solid, color: '#1E1E1E' },
                textColor: '#DDD',
            },
            grid: {
                vertLines: { color: '#333' },
                horzLines: { color: '#333' },
            },
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
            crosshair: {
                mode: 1,
            }
        });

        const series = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });

        chartRef.current = chart;
        seriesRef.current = series;

        // Resize Observer
        const resizeObserver = new ResizeObserver(entries => {
            if (!entries || entries.length === 0) return;
            const { width, height } = entries[0].contentRect;
            chart.applyOptions({ width, height });
            chart.timeScale().fitContent();
        });

        resizeObserver.observe(containerRef.current);

        return () => {
            resizeObserver.disconnect();
            chart.remove();
        };
    }, []);

    // Update candle data
    useEffect(() => {
        if (seriesRef.current && data.length > 0) {
            const sorted = [...data].sort((a, b) => a.time - b.time);
            seriesRef.current.setData(sorted as any);
            
            // Auto-scroll to latest
            if (chartRef.current) {
                chartRef.current.timeScale().scrollToRealTime();
            }
        }
    }, [data]);

    // Update markers for trades
    useEffect(() => {
        if (!seriesRef.current) return;

        const markers: any[] = [];

        trades.forEach(trade => {
            // Entry marker
            markers.push({
                time: trade.entry_time,
                position: trade.direction === 'LONG' ? 'belowBar' : 'aboveBar',
                color: trade.direction === 'LONG' ? '#26a69a' : '#ef5350',
                shape: 'arrowUp',
                text: `${trade.direction} @ ${trade.entry_price.toFixed(2)}`
            });

            // Exit marker (if closed)
            if (trade.status === 'closed' && trade.exit_time && trade.exit_price) {
                const isWin = (trade.pnl || 0) > 0;
                markers.push({
                    time: trade.exit_time,
                    position: trade.direction === 'LONG' ? 'aboveBar' : 'belowBar',
                    color: isWin ? '#4caf50' : '#ff5252',
                    shape: 'arrowDown',
                    text: `Exit @ ${trade.exit_price.toFixed(2)} | ${isWin ? 'WIN' : 'LOSS'} $${Math.abs(trade.pnl || 0).toFixed(0)}`
                });
            }
        });

        seriesRef.current.setMarkers(markers);
        markersRef.current = markers;
    }, [trades]);

    // Draw SL/TP lines for open trades
    useEffect(() => {
        if (!chartRef.current || !data.length) return;

        // Note: lightweight-charts doesn't have easy way to remove all lines,
        // so we'll rely on component remount for now
        // Lines are shown in the sidebar and overlays which is sufficient
        
    }, [trades, data]);

    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
            
            {/* Current Price Overlay */}
            {currentPrice > 0 && (
                <div style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    background: 'rgba(0, 0, 0, 0.7)',
                    padding: '8px 12px',
                    borderRadius: '4px',
                    fontSize: '14px',
                    fontWeight: 'bold',
                    color: '#4caf50'
                }}>
                    Current: ${currentPrice.toFixed(2)}
                </div>
            )}

            {/* Active Trades Overlay */}
            {trades.filter(t => t.status === 'open').length > 0 && (
                <div style={{
                    position: 'absolute',
                    top: '50px',
                    right: '10px',
                    background: 'rgba(0, 0, 0, 0.7)',
                    padding: '8px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    maxWidth: '200px'
                }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Active Positions</div>
                    {trades.filter(t => t.status === 'open').map(trade => (
                        <div key={trade.id} style={{ 
                            marginBottom: '5px', 
                            paddingBottom: '5px', 
                            borderBottom: '1px solid #444' 
                        }}>
                            <div style={{ color: trade.direction === 'LONG' ? '#26a69a' : '#ef5350' }}>
                                {trade.direction}
                            </div>
                            <div>Entry: {trade.entry_price.toFixed(2)}</div>
                            <div style={{ fontSize: '10px', color: '#999' }}>
                                SL: {trade.sl_price.toFixed(2)} | TP: {trade.tp_price.toFixed(2)}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
```

### ./frontend/src/components/YFinanceMode.tsx
```tsx
import React, { useState, useEffect, useRef } from 'react';
import { yfinanceApi, Trade } from '../api/yfinance';
import { Candle } from '../api/client';
import { YFinanceChart } from './YFinanceChart';
import { Play, Pause, RotateCcw, ArrowLeft, HelpCircle, ChevronDown, ChevronRight } from 'lucide-react';

interface YFinanceModeProps {
    onBack: () => void;
}

export const YFinanceMode: React.FC<YFinanceModeProps> = ({ onBack }) => {
    // Data settings
    const [sourceInterval, setSourceInterval] = useState<'1m' | '5m'>('1m');
    const [loadDays, setLoadDays] = useState<number>(5);
    const [useMockData, setUseMockData] = useState<boolean>(true); // Default to mock for testing

    // Data state
    const [allData, setAllData] = useState<Candle[]>([]); // Source candles (1m or 5m)
    const [symbol, setSymbol] = useState<string>('MES=F');

    // Playback state
    const [currentIndex, setCurrentIndex] = useState<number>(0);
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const [playbackSpeed, setPlaybackSpeed] = useState<number>(200);
    const [displayTimeframe, setDisplayTimeframe] = useState<number>(15); // Minutes

    // Chart Data (Aggregated)
    const [chartCandles, setChartCandles] = useState<Candle[]>([]);
    const [currentPrice, setCurrentPrice] = useState<number>(0);

    // Model & Trading state
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [selectedModel, setSelectedModel] = useState<string>('CNN_Predictive_5m');
    const riskAmount = 300; // Fixed risk per trade
    const [trades, setTrades] = useState<Trade[]>([]);
    const [totalPnL, setTotalPnL] = useState<number>(0);
    const [unrealizedPnL, setUnrealizedPnL] = useState<number>(0);

    // Analysis State
    const [lastAnalysis, setLastAnalysis] = useState<{ prob: number, atr: number, time: number } | null>(null);

    // Entry Control State
    const [modelThreshold, setModelThreshold] = useState<number>(0.15);
    const [limitFactor, setLimitFactor] = useState<number>(1.5);
    const [slFactor, setSlFactor] = useState<number>(1.0);
    const [entryMechanism, setEntryMechanism] = useState<'Predictive Limit' | 'Market Close'>('Predictive Limit');

    // UI state
    const [isLoading, setIsLoading] = useState<boolean>(false);
    
    // Collapsible sections
    const [dataExpanded, setDataExpanded] = useState<boolean>(true);
    const [modelExpanded, setModelExpanded] = useState<boolean>(true);
    const [entryExpanded, setEntryExpanded] = useState<boolean>(true);
    const [statsExpanded, setStatsExpanded] = useState<boolean>(true);

    // Tooltip state
    const [showHelp, setShowHelp] = useState<{ [key: string]: boolean }>({});

    // Constants
    const ORDER_EXPIRY_SECONDS = 15 * 60; // 15 minutes

    // Refs
    const playbackIntervalRef = useRef<number | null>(null);

    // Initial Load
    useEffect(() => {
        loadModels();
        // Adjust defaults based on source interval
        if (sourceInterval === '1m') {
            setLoadDays(5);
            setDisplayTimeframe(15);
        } else {
            setLoadDays(30);
            setDisplayTimeframe(60);
        }
    }, [sourceInterval]);

    const loadModels = async () => {
        try {
            const models = await yfinanceApi.getAvailableModels();
            setAvailableModels(models);
            if (models.length > 0 && !models.includes(selectedModel)) {
                // Prefer predictive
                const pred = models.find(m => m.includes('Predictive'));
                setSelectedModel(pred || models[0]);
            }
        } catch (e) {
            console.error('Error loading models:', e);
        }
    };

    const loadData = async () => {
        setIsLoading(true);
        try {
            // Using loadDays and sourceInterval
            const result = await yfinanceApi.fetchData(symbol, loadDays, sourceInterval, useMockData);

            if (result.data && result.data.length > 0) {
                setAllData(result.data);
                // Reset everything
                resetPlayback(result.data);
            } else {
                alert(result.message || 'Failed to load data');
            }
        } catch (e) {
            console.error('Error loading data:', e);
            alert('Failed to load data. Try enabling Mock Data mode.');
        } finally {
            setIsLoading(false);
        }
    };

    const resetPlayback = (data: Candle[] = allData) => {
        if (data.length === 0) return;

        // Start from beginning of loaded buffer? Or 200 bars in?
        // Let's start a bit in so we have history for the chart/model
        const startIdx = Math.min(200, data.length - 1);
        setCurrentIndex(startIdx);

        setIsPlaying(false);
        setChartCandles([]);
        setTrades([]);
        setTotalPnL(0);
        setUnrealizedPnL(0);
    };

    // Playback Loop
    useEffect(() => {
        if (isPlaying && allData.length > 0 && currentIndex < allData.length) {
            playbackIntervalRef.current = window.setInterval(() => {
                setCurrentIndex(prev => {
                    if (prev + 1 >= allData.length) {
                        setIsPlaying(false);
                        return prev;
                    }
                    return prev + 1;
                });
            }, playbackSpeed);
        } else {
            if (playbackIntervalRef.current !== null) {
                clearInterval(playbackIntervalRef.current);
                playbackIntervalRef.current = null;
            }
        }
        return () => {
            if (playbackIntervalRef.current !== null) clearInterval(playbackIntervalRef.current);
        };
    }, [isPlaying, currentIndex, allData.length, playbackSpeed]);

    // Update Chart & Logic on Update
    useEffect(() => {
        if (allData.length === 0 || currentIndex >= allData.length) return;

        const tick = allData[currentIndex];
        setCurrentPrice(tick.close);

        // 1. Accumulate Candle
        setChartCandles(prev => {
            if (prev.length === 0) {
                // Start with first tick aligned to timeframe boundary
                const tfSeconds = displayTimeframe * 60;
                const alignedTime = Math.floor(tick.time / tfSeconds) * tfSeconds;
                return [{
                    time: alignedTime,
                    open: tick.open,
                    high: tick.high,
                    low: tick.low,
                    close: tick.close,
                    volume: tick.volume
                }];
            }

            const lastCandle = prev[prev.length - 1];
            const tfSeconds = displayTimeframe * 60;
            const tickAlignedTime = Math.floor(tick.time / tfSeconds) * tfSeconds;

            // Check if tick belongs to current candle or starts new one
            if (tickAlignedTime === lastCandle.time) {
                // Update existing candle
                const updated = {
                    ...lastCandle,
                    high: Math.max(lastCandle.high, tick.high),
                    low: Math.min(lastCandle.low, tick.low),
                    close: tick.close,
                    volume: lastCandle.volume + tick.volume
                };
                return [...prev.slice(0, -1), updated];
            } else {
                // New candle starts
                return [...prev, {
                    time: tickAlignedTime,
                    open: tick.open,
                    high: tick.high,
                    low: tick.low,
                    close: tick.close,
                    volume: tick.volume
                }];
            }
        });

        // 2. Manage Trades
        updateTrades(tick);

        // 3. Check Signals - Only check when we complete a candle of the MODEL's input timeframe
        // For 5m mode: sourceInterval='1m', we check every 5 bars (5 minutes)
        // For 15m mode: sourceInterval='5m', we check every 3 bars (15 minutes)
        if (isPlaying && shouldCheckSignal(currentIndex)) {
            checkSignal(currentIndex);
        }

    }, [currentIndex]); // Only on index change

    // Helper to determine if we should check for signals at this index
    const shouldCheckSignal = (idx: number): boolean => {
        if (idx < 200) return false; // Need sufficient history

        // Check based on model's required interval
        // Models expect 20 candles of their input timeframe
        const checkInterval = sourceInterval === '1m' ? 5 : 3; // 5m for 1m data, 15m for 5m data

        // Check every N bars where N = minutes in target TF / minutes in source TF
        return idx % checkInterval === 0;
    };

    // Trade Management
    const updateTrades = (tick: Candle) => {
        setTrades(prev => {
            let pnlRealized = 0;
            const nextTrades = prev.map(t => {
                // Open Positions
                if (t.status === 'open') {
                    let hitSL = false, hitTP = false;
                    // MATCHING TEST SCRIPT: TP Check FIRST (Optimistic)
                    if (t.direction === 'LONG') {
                        if (tick.high >= t.tp_price) hitTP = true;
                        else if (tick.low <= t.sl_price) hitSL = true;
                    } else if (t.direction === 'SHORT') {
                        if (tick.low <= t.tp_price) hitTP = true;
                        else if (tick.high >= t.sl_price) hitSL = true;
                    } else {
                        console.error(`Unknown position direction: ${t.direction}`);
                        return { ...t, status: 'closed', pnl: 0, exit_time: tick.time } as Trade;
                    }

                    if (hitTP) {
                        // Win
                        const risk = Math.abs(t.entry_price - t.sl_price);
                        const reward = Math.abs(t.entry_price - t.tp_price);
                        const rMultiple = reward / (risk || 1);
                        const win = t.risk_amount * rMultiple;

                        pnlRealized += win;
                        return { ...t, status: 'closed', pnl: win, exit_price: t.tp_price, exit_time: tick.time } as Trade;
                    }
                    if (hitSL) {
                        const loss = -Math.abs(t.risk_amount);
                        pnlRealized += loss;
                        return { ...t, status: 'closed', pnl: loss, exit_price: t.sl_price, exit_time: tick.time } as Trade;
                    }
                }
                return t;
            });

            // Handle Pending Orders - separate pass to handle OCO logic
            let ocoGroupFilled: Record<string, boolean> = {};
            const finalTrades = nextTrades.map(t => {
                if (t.status === 'pending') {
                    // Check if order is before entry time
                    if (tick.time < t.entry_time) return t;

                    // Extract OCO group from ID (format: "s_123" or "b_123")
                    const groupId = t.id.substring(2); // Remove "s_" or "b_" prefix

                    // Check if other side of OCO was already filled
                    if (ocoGroupFilled[groupId]) {
                        // Cancel this pending order (OCO)
                        return { ...t, status: 'closed', pnl: 0, exit_time: tick.time } as Trade;
                    }

                    // Check for expiry (15 minutes)
                    const elapsedTime = tick.time - t.entry_time;
                    if (elapsedTime > ORDER_EXPIRY_SECONDS) {
                        // Expired
                        return { ...t, status: 'closed', pnl: 0, exit_time: tick.time } as Trade;
                    }

                    // Check if filled
                    let filled = false;
                    let fillPrice = t.entry_price;

                    if (t.direction === 'SELL') {
                        if (tick.high >= t.entry_price) {
                            filled = true;
                            // Gap Check: If Open is HIGHER than limit, we get filled at Open (Better Price)
                            fillPrice = Math.max(tick.open, t.entry_price);
                        }
                    } else if (t.direction === 'BUY') {
                        if (tick.low <= t.entry_price) {
                            filled = true;
                            // Gap Check: If Open is LOWER than limit, we get filled at Open (Better Price)
                            fillPrice = Math.min(tick.open, t.entry_price);
                        }
                    }

                    if (filled) {
                        // Mark this OCO group as filled
                        ocoGroupFilled[groupId] = true;
                        // Convert to position: BUY -> LONG, SELL -> SHORT
                        const positionDirection = t.direction === 'BUY' ? 'LONG' : 'SHORT';
                        // Update entry_price to the actual fill price
                        return { ...t, status: 'open', direction: positionDirection, entry_price: fillPrice } as Trade;
                    }
                }

                return t;
            });

            if (pnlRealized !== 0) setTotalPnL(s => s + pnlRealized);

            // Unrealized
            let float = 0;
            finalTrades.forEach(t => {
                if (t.status === 'open') {
                    let dist = 0;
                    if (t.direction === 'LONG') {
                        dist = tick.close - t.entry_price;
                    } else if (t.direction === 'SHORT') {
                        dist = t.entry_price - tick.close;
                    }
                    const risk = Math.abs(t.entry_price - t.sl_price);
                    const size = t.risk_amount / (risk || 1);
                    float += dist * size;
                }
            });
            setUnrealizedPnL(float);

            return finalTrades;
        });
    };

    const checkSignal = async (idx: number) => {
        try {
            const dateStr = new Date(allData[idx].time * 1000).toISOString().split('T')[0];
            const resp = await yfinanceApi.analyzeCandle(idx, selectedModel, symbol, dateStr);

            if (resp && resp.signal) {
                const s = resp.signal;
                const prob = s.prob || 0;
                const atr = s.atr || 0;

                setLastAnalysis({ prob, atr, time: allData[idx].time });

                // Frontend Decision Logic (Phase 12)
                if (prob < modelThreshold) return; // Sensitivity Check

                const currentPrice = allData[idx].close;

                if (entryMechanism === 'Predictive Limit') {
                    // Create Limit Orders based on Factors
                    const limitDist = limitFactor * atr;
                    const slDist = slFactor * atr;

                    const sellLimit = currentPrice + limitDist;
                    const buyLimit = currentPrice - limitDist;

                    const sell: Trade = {
                        id: `s_${idx}`,
                        direction: 'SELL',
                        entry_price: sellLimit,
                        sl_price: sellLimit + slDist,
                        tp_price: currentPrice, // Target Mean
                        entry_time: allData[idx].time,
                        status: 'pending',
                        risk_amount: riskAmount
                    };
                    const buy: Trade = {
                        id: `b_${idx}`,
                        direction: 'BUY',
                        entry_price: buyLimit,
                        sl_price: buyLimit - slDist,
                        tp_price: currentPrice,
                        entry_time: allData[idx].time,
                        status: 'pending',
                        risk_amount: riskAmount
                    };

                    setTrades(prev => [...prev, sell, buy]);
                }
            } else {
                // No signal (Warmup or Error)
                setLastAnalysis({ prob: 0, atr: 0, time: allData[idx].time });
            }
        } catch (e) {
            console.error(e);
            setLastAnalysis({ prob: -1, atr: 0, time: allData[idx].time }); // Error state
        }
    };

    const openTrades = trades.filter(t => t.status === 'open');
    const pendingTrades = trades.filter(t => t.status === 'pending');
    const closedTrades = trades.filter(t => t.status === 'closed');
    
    // Helper to render section header with collapse toggle
    const SectionHeader: React.FC<{ 
        title: string; 
        expanded: boolean; 
        onToggle: () => void;
        helpKey?: string;
        helpText?: string;
    }> = ({ title, expanded, onToggle, helpKey, helpText }) => (
        <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            marginBottom: '10px',
            paddingBottom: '5px',
            borderBottom: '1px solid #444'
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <button 
                    onClick={onToggle}
                    style={{ 
                        background: 'transparent', 
                        border: 'none', 
                        color: '#aaa', 
                        cursor: 'pointer',
                        padding: '0',
                        display: 'flex',
                        alignItems: 'center'
                    }}
                >
                    {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                </button>
                <h3 style={{ fontSize: '14px', fontWeight: 'bold', margin: 0 }}>{title}</h3>
                {helpKey && helpText && (
                    <button
                        onClick={() => setShowHelp(prev => ({ ...prev, [helpKey]: !prev[helpKey] }))}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#777',
                            cursor: 'pointer',
                            padding: '2px',
                            display: 'flex'
                        }}
                        title="Show help"
                    >
                        <HelpCircle size={14} />
                    </button>
                )}
            </div>
        </div>
    );
    
    // Helper to render help text
    const HelpText: React.FC<{ helpKey: string; text: string }> = ({ helpKey, text }) => (
        showHelp[helpKey] ? (
            <div style={{
                background: '#2a2a2a',
                padding: '8px',
                borderRadius: '4px',
                fontSize: '11px',
                color: '#aaa',
                marginBottom: '10px',
                border: '1px solid #444'
            }}>
                {text}
            </div>
        ) : null
    );

    return (
        <div style={{ display: 'flex', height: '100vh', background: '#1E1E1E', color: '#EEE' }}>
            {/* Sidebar */}
            <div style={{ width: '320px', background: '#252526', padding: '15px', borderRight: '1px solid #333', overflowY: 'auto' }}>
                <button onClick={onBack} style={{ display: 'flex', alignItems: 'center', gap: '5px', padding: '5px 10px', marginBottom: '20px', background: 'transparent', color: '#aaa', border: 'none', cursor: 'pointer' }}>
                    <ArrowLeft size={16} /> Back
                </button>

                <h2 style={{ margin: '0 0 20px 0', fontSize: '18px' }}>YFinance Playback</h2>
                
                {/* Data Source Section */}
                <div style={{ marginBottom: '20px' }}>
                    <SectionHeader 
                        title="Data Source" 
                        expanded={dataExpanded} 
                        onToggle={() => setDataExpanded(!dataExpanded)}
                        helpKey="dataSource"
                        helpText="Load real market data from YFinance or use mock data for testing. Mock data is recommended for initial testing."
                    />
                    <HelpText 
                        helpKey="dataSource" 
                        text="Choose your data source: Real market data from YFinance (rate-limited), or generated mock data (unlimited). 1m data is limited to ~7 days, 5m data up to ~60 days."
                    />
                    
                    {dataExpanded && (
                        <div>
                            <label style={{ fontSize: '12px', display: 'block', marginBottom: '5px' }}>Symbol</label>
                            <input 
                                value={symbol} 
                                onChange={e => setSymbol(e.target.value)} 
                                style={{ width: '100%', padding: '6px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px' }}
                                placeholder="MES=F, ES=F, AAPL"
                            />

                            <label style={{ marginTop: '10px', display: 'block', fontSize: '12px', marginBottom: '5px' }}>Source Granularity</label>
                            <div style={{ display: 'flex', gap: '5px' }}>
                                <button
                                    onClick={() => setSourceInterval('1m')}
                                    style={{ 
                                        flex: 1, 
                                        padding: '6px', 
                                        background: sourceInterval === '1m' ? '#007acc' : '#444', 
                                        border: 'none', 
                                        color: 'white',
                                        borderRadius: '3px',
                                        cursor: 'pointer',
                                        fontSize: '12px'
                                    }}>
                                    1 Minute
                                </button>
                                <button
                                    onClick={() => setSourceInterval('5m')}
                                    style={{ 
                                        flex: 1, 
                                        padding: '6px', 
                                        background: sourceInterval === '5m' ? '#007acc' : '#444', 
                                        border: 'none', 
                                        color: 'white',
                                        borderRadius: '3px',
                                        cursor: 'pointer',
                                        fontSize: '12px'
                                    }}>
                                    5 Minute
                                </button>
                            </div>

                            <label style={{ marginTop: '10px', display: 'block', fontSize: '12px', marginBottom: '5px' }}>Days to Load</label>
                            <input 
                                type="number" 
                                value={loadDays} 
                                onChange={e => setLoadDays(Number(e.target.value))} 
                                min={1} 
                                max={60} 
                                style={{ width: '100%', padding: '6px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px' }}
                            />
                            <div style={{ fontSize: '10px', color: '#777', marginTop: '2px' }}>Max: ~7d (1m), ~60d (5m)</div>

                            <label style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer' }}>
                                <input
                                    type="checkbox"
                                    checked={useMockData}
                                    onChange={e => setUseMockData(e.target.checked)}
                                    style={{ cursor: 'pointer' }}
                                />
                                <span style={{ fontSize: '12px' }}>Use Mock Data (testing)</span>
                            </label>

                            <button 
                                onClick={loadData} 
                                disabled={isLoading} 
                                style={{ 
                                    width: '100%', 
                                    marginTop: '12px', 
                                    padding: '8px', 
                                    background: isLoading ? '#555' : '#007acc', 
                                    color: 'white', 
                                    border: 'none', 
                                    cursor: isLoading ? 'default' : 'pointer',
                                    borderRadius: '3px',
                                    fontWeight: 'bold',
                                    fontSize: '13px'
                                }}>
                                {isLoading ? 'Loading...' : 'Load Data'}
                            </button>
                            
                            {allData.length > 0 && (
                                <div style={{ marginTop: '8px', fontSize: '11px', color: '#888', textAlign: 'center' }}>
                                    ✓ Loaded {allData.length} bars
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Model Configuration Section */}
                <div style={{ marginBottom: '20px' }}>
                    <SectionHeader 
                        title="Model Configuration" 
                        expanded={modelExpanded} 
                        onToggle={() => setModelExpanded(!modelExpanded)}
                        helpKey="modelConfig"
                        helpText="Select the trained CNN model and chart display settings."
                    />
                    <HelpText 
                        helpKey="modelConfig" 
                        text="CNN models are trained to detect specific setups. Predictive models work for mean reversion. Chart timeframe is for display only and doesn't affect model analysis."
                    />
                    
                    {modelExpanded && (
                        <div>
                            <label style={{ fontSize: '12px', display: 'block', marginBottom: '5px' }}>Model</label>
                            <select 
                                value={selectedModel} 
                                onChange={e => setSelectedModel(e.target.value)} 
                                style={{ width: '100%', padding: '6px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px' }}
                            >
                                {availableModels.map(m => <option key={m}>{m}</option>)}
                            </select>
                            
                            <label style={{ marginTop: '10px', fontSize: '12px', display: 'block', marginBottom: '5px' }}>Chart Timeframe (Display)</label>
                            <select 
                                value={displayTimeframe} 
                                onChange={e => {
                                    setDisplayTimeframe(Number(e.target.value));
                                    setChartCandles([]);
                                }} 
                                style={{ width: '100%', padding: '6px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px' }}
                            >
                                <option value={5} disabled={sourceInterval === '5m'}>5 Minutes</option>
                                <option value={15}>15 Minutes</option>
                                <option value={60}>1 Hour</option>
                            </select>
                            <div style={{ fontSize: '10px', color: '#777', marginTop: '2px' }}>For visualization only</div>
                        </div>
                    )}
                </div>

                {/* Entry Mechanism Section */}
                <div style={{ marginBottom: '20px' }}>
                    <SectionHeader 
                        title="Entry Mechanism" 
                        expanded={entryExpanded} 
                        onToggle={() => setEntryExpanded(!entryExpanded)}
                        helpKey="entryMech"
                        helpText="Configure how trades are entered when signals fire."
                    />
                    <HelpText 
                        helpKey="entryMech" 
                        text="Sensitivity: Lower = more selective (fewer trades). Predictive Limit: Places OCO bracket orders at ±N×ATR. First to fill cancels the other. 15min expiry."
                    />

                    {entryExpanded && (
                        <div>
                            <label style={{ fontSize: '12px', display: 'block', marginBottom: '5px' }}>
                                Sensitivity: {(modelThreshold * 100).toFixed(0)}% 
                                <span style={{ fontSize: '10px', color: '#888', marginLeft: '5px' }}>
                                    (min confidence)
                                </span>
                            </label>
                            <input
                                type="range" min="0" max="100" step="1"
                                value={modelThreshold * 100}
                                onChange={e => setModelThreshold(Number(e.target.value) / 100)}
                                style={{ width: '100%' }}
                            />
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#777' }}>
                                <span>Aggressive</span>
                                <span>Conservative</span>
                            </div>

                            <label style={{ display: 'block', marginTop: '10px', fontSize: '12px', marginBottom: '5px' }}>Method</label>
                            <select 
                                value={entryMechanism} 
                                onChange={e => setEntryMechanism(e.target.value as any)} 
                                style={{ width: '100%', padding: '6px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px' }}
                            >
                                <option>Predictive Limit</option>
                                <option disabled>Market Close (Coming Soon)</option>
                            </select>

                            {entryMechanism === 'Predictive Limit' && (
                                <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                                    <div style={{ flex: 1 }}>
                                        <label style={{ fontSize: '11px', display: 'block', marginBottom: '3px' }}>Limit (×ATR)</label>
                                        <input 
                                            type="number" 
                                            step="0.1" 
                                            value={limitFactor} 
                                            onChange={e => setLimitFactor(Number(e.target.value))} 
                                            style={{ width: '100%', padding: '4px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px', fontSize: '12px' }}
                                        />
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <label style={{ fontSize: '11px', display: 'block', marginBottom: '3px' }}>Stop (×ATR)</label>
                                        <input 
                                            type="number" 
                                            step="0.1" 
                                            value={slFactor} 
                                            onChange={e => setSlFactor(Number(e.target.value))} 
                                            style={{ width: '100%', padding: '4px', background: '#333', border: '1px solid #555', color: '#fff', borderRadius: '3px', fontSize: '12px' }}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Performance Stats Section */}
                <div>
                    <SectionHeader 
                        title="Performance" 
                        expanded={statsExpanded} 
                        onToggle={() => setStatsExpanded(!statsExpanded)}
                    />
                    
                    {statsExpanded && (
                        <div>
                            <div style={{ background: '#333', padding: '12px', borderRadius: '4px', marginBottom: '10px' }}>
                                <div style={{ marginBottom: '8px' }}>
                                    <div style={{ fontSize: '11px', color: '#888' }}>Realized PnL</div>
                                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: totalPnL >= 0 ? '#4caf50' : '#f44336' }}>
                                        ${totalPnL.toFixed(2)}
                                    </div>
                                </div>
                                <div style={{ marginBottom: '8px' }}>
                                    <div style={{ fontSize: '11px', color: '#888' }}>Floating PnL</div>
                                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: unrealizedPnL >= 0 ? '#4caf50' : '#f44336' }}>
                                        ${unrealizedPnL.toFixed(2)}
                                    </div>
                                </div>
                                <div style={{ paddingTop: '8px', borderTop: '1px solid #444', fontSize: '12px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                        <span>Open Positions:</span>
                                        <strong>{openTrades.length}</strong>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                        <span>Pending Orders:</span>
                                        <strong>{pendingTrades.length}</strong>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span>Closed Trades:</span>
                                        <strong>{closedTrades.length}</strong>
                                    </div>
                                </div>
                            </div>
                            
                            {closedTrades.length > 0 && (
                                <div style={{ background: '#2a2a2a', padding: '8px', borderRadius: '4px', fontSize: '11px' }}>
                                    <div style={{ marginBottom: '4px' }}>
                                        Win Rate: <strong>
                                            {((closedTrades.filter(t => (t.pnl ?? 0) > 0).length / closedTrades.length) * 100).toFixed(1)}%
                                        </strong>
                                    </div>
                                    <div>
                                        Avg PnL: <strong style={{ color: (totalPnL / closedTrades.length) >= 0 ? '#4caf50' : '#f44336' }}>
                                            ${(totalPnL / closedTrades.length).toFixed(2)}
                                        </strong>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Main Chart */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                {/* Control Bar */}
                <div style={{ 
                    padding: '12px 15px', 
                    display: 'flex', 
                    gap: '15px', 
                    alignItems: 'center', 
                    background: '#2d2d2d',
                    borderBottom: '1px solid #444'
                }}>
                    <button 
                        onClick={() => setIsPlaying(!isPlaying)} 
                        style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '5px', 
                            padding: '8px 16px', 
                            cursor: 'pointer',
                            background: isPlaying ? '#f44336' : '#4caf50',
                            border: 'none',
                            color: 'white',
                            borderRadius: '4px',
                            fontWeight: 'bold',
                            fontSize: '13px'
                        }}
                        disabled={allData.length === 0}
                    >
                        {isPlaying ? <><Pause size={16} /> Pause</> : <><Play size={16} /> Play</>}
                    </button>
                    <button 
                        onClick={() => resetPlayback()} 
                        style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '5px', 
                            padding: '8px 16px', 
                            cursor: 'pointer',
                            background: '#555',
                            border: 'none',
                            color: 'white',
                            borderRadius: '4px',
                            fontSize: '13px'
                        }}
                        disabled={allData.length === 0}
                    >
                        <RotateCcw size={16} /> Reset
                    </button>
                    
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginLeft: '10px' }}>
                        <span style={{ fontSize: '12px', color: '#aaa' }}>Speed:</span>
                        <input 
                            type="range" 
                            min="10" 
                            max="1000" 
                            step="10" 
                            value={playbackSpeed} 
                            onChange={e => setPlaybackSpeed(Number(e.target.value))}
                            style={{ width: '120px' }}
                        />
                        <span style={{ fontSize: '12px', color: '#ddd', minWidth: '60px' }}>{playbackSpeed}ms</span>
                    </div>
                    
                    <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '15px' }}>
                        <div style={{ fontSize: '12px', color: '#aaa' }}>
                            Progress: <strong style={{ color: '#fff' }}>
                                {allData.length > 0 ? `${currentIndex} / ${allData.length}` : '--'}
                            </strong>
                        </div>
                        <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#fff' }}>
                            {allData[currentIndex]?.time ? new Date(allData[currentIndex].time * 1000).toLocaleString() : '--'}
                        </div>
                    </div>
                </div>

                {/* Signal Indicator */}
                <div style={{ 
                    position: 'absolute', 
                    bottom: '20px', 
                    left: '340px', 
                    background: 'rgba(0,0,0,0.85)', 
                    padding: '15px', 
                    borderRadius: '8px', 
                    display: 'flex', 
                    gap: '20px', 
                    alignItems: 'center', 
                    zIndex: 100, 
                    border: '1px solid #555',
                    backdropFilter: 'blur(10px)'
                }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '5px' }}>
                        <div style={{
                            width: '40px', 
                            height: '40px', 
                            borderRadius: '50%',
                            background: lastAnalysis && lastAnalysis.prob > 0 
                                ? `radial-gradient(circle, rgba(0, 255, 0, ${lastAnalysis.prob}), rgba(0, 128, 0, ${lastAnalysis.prob * 0.5}))` 
                                : '#333',
                            border: '2px solid #555',
                            boxShadow: lastAnalysis && lastAnalysis.prob > modelThreshold 
                                ? `0 0 20px rgba(0,255,0,${lastAnalysis.prob})` 
                                : 'none',
                            transition: 'all 0.3s ease'
                        }} />
                        <div style={{ fontSize: '9px', color: '#888', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            Signal
                        </div>
                    </div>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div>
                            <div style={{ fontSize: '10px', color: '#aaa', marginBottom: '2px' }}>Model Confidence</div>
                            <div style={{ 
                                fontSize: '18px', 
                                fontWeight: 'bold', 
                                color: lastAnalysis && lastAnalysis.prob > modelThreshold ? '#4caf50' : '#fff'
                            }}>
                                {lastAnalysis ? `${(lastAnalysis.prob * 100).toFixed(1)}%` : '--'}
                            </div>
                        </div>
                        
                        <div>
                            <div style={{ fontSize: '10px', color: '#aaa', marginBottom: '2px' }}>ATR (15m)</div>
                            <div style={{ fontSize: '14px', color: '#ddd', fontWeight: 'bold' }}>
                                {lastAnalysis ? lastAnalysis.atr.toFixed(2) : '--'}
                            </div>
                        </div>
                    </div>
                    
                    {lastAnalysis && lastAnalysis.prob > modelThreshold && (
                        <div style={{
                            padding: '4px 8px',
                            background: 'rgba(76, 175, 80, 0.2)',
                            border: '1px solid #4caf50',
                            borderRadius: '4px',
                            fontSize: '11px',
                            color: '#4caf50',
                            fontWeight: 'bold',
                            animation: 'pulse 2s infinite'
                        }}>
                            ACTIVE
                        </div>
                    )}
                </div>

                <div style={{ flex: 1 }}>
                    <YFinanceChart
                        data={chartCandles}
                        trades={trades}
                        currentPrice={currentPrice}
                    />
                </div>
            </div>
        </div>
    );
}
```

### ./frontend/src/components/YFinancePage.tsx
```tsx
import React, { useState } from 'react';
import { ChartPanel } from './ChartPanel';
import { api, Candle } from '../api/client';

interface YFinancePageProps {
    onBack: () => void;
}

export const YFinancePage: React.FC<YFinancePageProps> = ({ onBack }) => {
    const [symbol, setSymbol] = useState<string>('MES=F');
    const [days, setDays] = useState<number>(5);
    const [timeframe, setTimeframe] = useState<string>('1m');
    const [candles, setCandles] = useState<Candle[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFetch = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const data = await api.getYFinanceCandles(symbol, days, timeframe);
            setCandles(data);
        } catch (e: any) {
            setError(e.response?.data?.detail || e.message || 'Failed to fetch data');
            setCandles([]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={{ display: 'flex', height: '100vh', background: '#1E1E1E', color: '#EEE', flexDirection: 'column' }}>
            {/* Header Bar */}
            <div style={{ 
                padding: '15px', 
                background: '#2D2D2D', 
                borderBottom: '1px solid #444',
                display: 'flex',
                gap: '15px',
                alignItems: 'center',
                flexWrap: 'wrap'
            }}>
                <button
                    onClick={onBack}
                    style={{
                        padding: '8px 15px',
                        background: '#555',
                        color: '#FFF',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '14px'
                    }}
                >
                    ← Back to Generator
                </button>

                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <label style={{ fontSize: '14px' }}>
                        Symbol:
                        <input
                            type="text"
                            value={symbol}
                            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                            placeholder="MES=F"
                            style={{
                                marginLeft: '5px',
                                padding: '6px 10px',
                                background: '#333',
                                color: '#FFF',
                                border: '1px solid #555',
                                borderRadius: '4px',
                                width: '120px'
                            }}
                        />
                    </label>

                    <label style={{ fontSize: '14px' }}>
                        Days:
                        <input
                            type="number"
                            value={days}
                            onChange={(e) => setDays(Math.max(1, parseInt(e.target.value) || 1))}
                            min="1"
                            max="30"
                            style={{
                                marginLeft: '5px',
                                padding: '6px 10px',
                                background: '#333',
                                color: '#FFF',
                                border: '1px solid #555',
                                borderRadius: '4px',
                                width: '60px'
                            }}
                        />
                    </label>

                    <label style={{ fontSize: '14px' }}>
                        Timeframe:
                        <select
                            value={timeframe}
                            onChange={(e) => setTimeframe(e.target.value)}
                            style={{
                                marginLeft: '5px',
                                padding: '6px 10px',
                                background: '#333',
                                color: '#FFF',
                                border: '1px solid #555',
                                borderRadius: '4px'
                            }}
                        >
                            <option value="1m">1 Minute</option>
                            <option value="5m">5 Minutes</option>
                            <option value="15m">15 Minutes</option>
                            <option value="60m">1 Hour</option>
                        </select>
                    </label>

                    <button
                        onClick={handleFetch}
                        disabled={isLoading}
                        style={{
                            padding: '8px 15px',
                            background: isLoading ? '#666' : '#4CAF50',
                            color: '#FFF',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: isLoading ? 'default' : 'pointer',
                            fontSize: '14px',
                            fontWeight: 'bold'
                        }}
                    >
                        {isLoading ? 'Loading...' : 'Fetch Data'}
                    </button>
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div style={{
                    padding: '10px 15px',
                    background: '#C33',
                    color: '#FFF',
                    fontSize: '14px',
                    borderBottom: '1px solid #900'
                }}>
                    ❌ {error}
                </div>
            )}

            {/* Chart Panel */}
            <div style={{ flex: 1, padding: '10px', overflow: 'hidden' }}>
                <ChartPanel 
                    data={candles} 
                    title={candles.length > 0 ? `${symbol} - ${timeframe} (${candles.length} candles)` : 'No data loaded'} 
                />
            </div>

            {/* Footer Info */}
            {candles.length > 0 && (
                <div style={{
                    padding: '10px 15px',
                    background: '#2D2D2D',
                    borderTop: '1px solid #444',
                    fontSize: '12px',
                    color: '#AAA'
                }}>
                    Loaded {candles.length} candles • 
                    {' '}First: {new Date(candles[0].time * 1000).toLocaleString()} • 
                    {' '}Last: {new Date(candles[candles.length - 1].time * 1000).toLocaleString()}
                </div>
            )}
        </div>
    );
};
```

### ./frontend/src/main.tsx
```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>,
)
```

### ./frontend/vite.config.ts
```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
})
```

### ./requirements.txt
```txt
fastapi
uvicorn
pandas
numpy
scikit-learn
pyarrow
pydantic
python-multipart
yfinance
```

### ./src/api.py
```py
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import yfinance as yf
from datetime import datetime, timedelta

from src.config import ONE_MIN_PARQUET_DIR, PATTERNS_DIR, LOCAL_TZ
from src.generator import get_generator
from src.utils.logging_utils import get_logger

logger = get_logger("api")

app = FastAPI(title="MES Pattern Generator API")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(f"Validation Error: {exc.errors()}")
    logger.error(f"Request Body: {body.decode()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode()},
    )

# Allow CORS for frontend

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data cache
_data_cache = {
    "1min": None,
    "last_loaded": None
}

def get_1min_data():
    """
    Load or return cached 1-minute data.
    """
    parquet_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    if not parquet_path.exists():
        return pd.DataFrame()
    
    # Reload if null or if file changed? For now simple cache
    if _data_cache["1min"] is None:
        logger.info("Loading 1-min parquet into memory...")
        df = pd.read_parquet(parquet_path)
        # Ensure date is string for JSON
        df['time_str'] = df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S') 
        # Convert date column to string
        df['date_str'] = df['date'].astype(str)
        _data_cache["1min"] = df
        
    return _data_cache["1min"]

# --- Models ---
class GenerateSessionRequest(BaseModel):
    day_of_week: int
    session_type: str = "RTH"
    start_price: float = 5800.0
    date: Optional[str] = None # YYYY-MM-DD
    timeframe: str = "1m"

class Candle(BaseModel):
    time: float # Unix timestamp (seconds)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0
    synthetic_day: Optional[int] = None

class CandleResponse(BaseModel):
    data: List[Candle]

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/dates")
def get_available_dates():
    df = get_1min_data()
    if df.empty:
        return []
    # Get unique dates
    dates = df['date_str'].unique().tolist()
    # Sort descending
    dates.sort(reverse=True)
    return dates

def _resample_df(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "1m":
        return df
        
    rule_map = {"1m": "1T", "5m": "5T", "15m": "15T", "30m": "30T", "60m": "60T", "1h": "60T"}
    rule = rule_map.get(timeframe, "1T")
    
    # Check if 'time' is index or column
    if 'time' in df.columns:
        df = df.set_index('time')
        
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Handle synthetic_day if present (take mode? or first?)
    # Usually synthetic day is constant for a day, but across boundary...
    # For resampling, 'first' is safe enough for day index
    if 'synthetic_day' in df.columns:
        agg_dict['synthetic_day'] = 'first'
        
    resampled = df.resample(rule).agg(agg_dict).dropna()
    return resampled.reset_index()

@app.get("/api/candles")
def get_candles(
    date: str = Query(..., description="Date YYYY-MM-DD"),
    timeframe: str = "1m",
    session_type: Optional[str] = None
):
    df = get_1min_data()
    if df.empty:
        return {"data": []}
    
    # Filter by date
    subset = df[df['date_str'] == date].copy()
    if subset.empty:
        return {"data": []}
        
    if session_type and session_type != "all":
        subset = subset[subset['session_type'] == session_type]
        
    subset = _resample_df(subset, timeframe)
        
    results = []
    for _, row in subset.iterrows():
        results.append({
            "time": int(row['time'].timestamp()),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row.get('volume', 0)
        })
        
    return {"data": results}

@app.post("/api/generate/session")
def generate_session(req: GenerateSessionRequest):
    gen = get_generator()
    
    synthetic_df = gen.generate_session(
        day_of_week=req.day_of_week,
        session_type=req.session_type,
        start_price=req.start_price,
        date=req.date
    )
    
    if synthetic_df.empty:
        raise HTTPException(status_code=400, detail="Generation failed (no patterns?)")
    
    # Resample
    synthetic_df = _resample_df(synthetic_df, req.timeframe)
        
    # Return candles
    results = []
    for _, row in synthetic_df.iterrows():
        results.append({
            "time": int(row['time'].timestamp()),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume']
        })
        
    return {"data": results}

class GenerateMultiDayRequest(BaseModel):
    num_days: int
    session_type: str = "RTH"
    initial_price: float = 5800.0
    start_date: Optional[str] = None # YYYY-MM-DD
    timeframe: str = "1m"

@app.get("/api/candles-range")
def get_candles_range(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    timeframe: str = "1m",
    session_type: Optional[str] = None
):
    df = get_1min_data()
    if df.empty: return {"data": []}
    
    # Filter by date range (inclusive strings work for ISO dates)
    subset = df[(df['date_str'] >= start_date) & (df['date_str'] <= end_date)].copy()
    
    if session_type and session_type != "all":
        subset = subset[subset['session_type'] == session_type]
        
    subset = _resample_df(subset, timeframe)

    results = []
    for _, row in subset.iterrows():
        results.append({
            "time": int(row['time'].timestamp()),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row.get('volume', 0)
        })
    return {"data": results}

@app.post("/api/generate/multi-day")
def generate_multi_day(req: GenerateMultiDayRequest):
    gen = get_generator()
    synthetic_df = gen.generate_multi_day(
        num_days=req.num_days,
        session_type=req.session_type,
        initial_price=req.initial_price,
        start_date=req.start_date
    )
    
    if synthetic_df.empty:
        raise HTTPException(status_code=400, detail="Generation failed")
    
    # Resample
    synthetic_df = _resample_df(synthetic_df, req.timeframe)
        
    results = []
    for _, row in synthetic_df.iterrows():
        results.append({
            "time": int(row['time'].timestamp()),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume'],
            "synthetic_day": row.get('synthetic_day', 0)
        })
    return {"data": results}

@app.get("/api/setup-stats")
def get_setup_stats(min_samples: int = 100, min_expansion_rate: float = 0.5):
    """
    Return setups (clusters) with basic statistics.
    """
    from src.config import PROCESSED_DIR
    
    meta_path = PROCESSED_DIR / "mes_setup_rules.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Setup metadata not found. Run setup_miner.py first.")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    setup_stats = meta.get("setup_stats", [])
    
    # Filter
    filtered = [
        s for s in setup_stats
        if s["count"] >= min_samples and s["exp_rate"] >= min_expansion_rate
    ]
    
    # Sort by expansion rate desc
    filtered.sort(key=lambda x: x["exp_rate"], reverse=True)
    
    return {
        "setups": filtered,
        "features": meta.get("feature_cols", []),
        "meta": meta
    }

@app.get("/api/setup-rules")
def get_setup_rules():
    """
    Return the decision tree rules text.
    """
    from src.config import PROCESSED_DIR
    path = PROCESSED_DIR / "mes_setup_decision_tree.json"
    if not path.exists():
         raise HTTPException(status_code=404, detail="Rules not found")
         
    with open(path, 'r') as f:
        data = json.load(f)
        
    return data

@app.get("/api/pattern-buckets")
def get_pattern_buckets():
    gen = get_generator()
    if not gen.cluster_meta:
        return []
        
    summary = []
    for m in gen.cluster_meta:
        summary.append({
            "session": m['session_type'],
            "dow": m['day_of_week'],
            "hour": m['hour_bucket'],
            "k": m['k'],
            "samples": m['total_samples']
        })
    return summary

# --- YFinance Endpoints ---

@app.get("/api/yfinance/candles/mock")
def get_mock_candles(
    bars: int = Query(500, description="Number of bars to generate"),
    timeframe: str = Query("1m", description="Timeframe interval")
):
    """
    Generate mock OHLC data for testing playback mode.
    Creates realistic price movement with trends and volatility.
    """
    import random
    
    # Starting parameters
    start_price = 5800.0
    volatility = 2.0
    trend = 0.05
    
    # Map timeframe to seconds
    interval_map = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '60m': 3600,
        '1h': 3600
    }
    interval_seconds = interval_map.get(timeframe, 60)  # Default to 1m if unknown
    
    # Generate timestamp (starting from now - bars * interval)
    end_time = int(datetime.now().timestamp())
    start_time = end_time - (bars * interval_seconds)
    
    results = []
    current_price = start_price
    
    for i in range(bars):
        timestamp = start_time + (i * interval_seconds)
        
        # Generate OHLC
        open_price = current_price
        
        # Random walk with slight trend
        change_pct = random.gauss(trend / bars, volatility / 100)
        close_price = open_price * (1 + change_pct)
        
        # High/Low with some intrabar movement
        intrabar_range = abs(close_price - open_price) * random.uniform(1.2, 2.0)
        high_price = max(open_price, close_price) + random.uniform(0, intrabar_range)
        low_price = min(open_price, close_price) - random.uniform(0, intrabar_range)
        
        volume = random.uniform(1000, 5000)
        
        results.append({
            "time": timestamp,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": round(volume, 0)
        })
        
        current_price = close_price
    
    # Update cache for playback analysis
    # Convert to DataFrame with time as column (not index) for model inference
    df_cache = pd.DataFrame(results)
    df_cache['time'] = pd.to_datetime(df_cache['time'], unit='s')
    
    _last_yf_cache['symbol'] = 'MOCK'
    _last_yf_cache['data'] = df_cache
    _last_yf_cache['interval'] = timeframe
    
    logger.info(f"Generated {len(results)} mock candles")
    return {"data": results, "symbol": "MOCK", "timeframe": timeframe}

@app.get("/api/yfinance/candles")
def get_yfinance_candles(
    symbol: str = Query("MES=F", description="Ticker symbol (e.g., MES=F, ES=F, AAPL)"),
    days: int = Query(5, description="Number of days of historical 1m data to fetch"),
    timeframe: str = Query("1m", description="Timeframe (1m, 5m, 15m, 60m, 1h, 1d)")
):
    """
    Fetch 1-minute OHLC data from yfinance for a given symbol.
    Default symbol is MES=F (Micro E-mini S&P 500).
    """
    try:
        # Calculate date range (yahoo finance 1m only goes back ~7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching {symbol} 1m data from {start_date} to {end_date}")
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        # Map timeframe to yfinance interval if needed, or just use timeframe if valid
        # Valid: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        # We use '1m' or '5m' generally.
        yf_interval = timeframe if timeframe in ['1m', '5m', '15m', '60m', '1h', '1d'] else "1m"
        
        df = ticker.history(start=start_date, end=end_date, interval=yf_interval)
        
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data returned for {symbol}. Check symbol spelling.")
            
        # Normalize Dataframe for Cache (Match test scripts)
        # 1. Reset Index
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        # 2. Lowercase cols
        df.columns = [c.lower() for c in df.columns]
        # 3. Rename date/datetime to 'time'
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        # 4. Ensure UTC
        # df['time'] = pd.to_datetime(df['time'], utc=True) # Usually already tz-aware
        
        # Update Cache
        _last_yf_cache['symbol'] = symbol
        _last_yf_cache['data'] = df
        _last_yf_cache['interval'] = yf_interval

        
        # No manual resampling needed if we fetched the correct interval
        # But we might want to standardize columns
        
        # Convert to candle format
        results = []
        for idx, row in df.iterrows():
            results.append({
                "time": int(idx.timestamp()),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row.get('Volume', 0))
            })
        
        logger.info(f"Returning {len(results)} candles for {symbol}")
        return {"data": results, "symbol": symbol, "timeframe": timeframe}
    
    except Exception as e:
        logger.error(f"YFinance fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"YFinance error: {str(e)}")

# --- Analysis Endpoint (Re-implemented) ---
from src.model_inference import ModelInference

# Global inference cache
_inference_engine = None

def get_inference_engine(model_name: str):
    global _inference_engine
    if _inference_engine is None or _inference_engine.model_name != model_name:
        _inference_engine = ModelInference(model_name)
    return _inference_engine

@app.post("/api/yfinance/playback/analyze")
def analyze_playback_candle(
    candle_index: int = Query(..., description="Index in the client's data array"),
    model_name: str = Query(..., description="Model to use"),
    symbol: str = Query("MES=F"),
    date: str = Query(...),
    # We need the full context. Client sends index.
    # But backend is stateless regarding the client's specific array? 
    # NO. The Backend `get_yfinance_candles` fetches data but doesn't persist it for this session uniquely?
    # Actually, `ModelInference.analyze` needs the DATAFRAME.
    # We can't easily re-fetch yfinance data on every tick safely/quickly.
    # SOLUTION: Client should probably send the last N candles? Or Backend caches the LAST fetched yfinance data?
    # Let's use a simple global cache for the last fetched symbol/data in memory.
    # See `_last_yf_cache` below.
): 
    # Need access to data
    # Allow mock data to be used for any symbol
    if _last_yf_cache['symbol'] != symbol and _last_yf_cache['symbol'] != 'MOCK':
         # Warn or try to re-fetch?
         # If client just called get_yfinance_candles, it should be here.
         raise HTTPException(status_code=400, detail=f"Data not cached for {symbol}. Call fetch first. Cached: {_last_yf_cache['symbol']}")
         
    df = _last_yf_cache['data']
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data cached.")
        
    engine = get_inference_engine(model_name)
    
    # Run analysis
    # Note: candle_index must align with the cached dataframe
    result = engine.analyze(candle_index, df)
    
    return {"signal": result["signal"] if result else None}

# Simple cache system for YFinance data to support playback analysis
_last_yf_cache = {
    "symbol": None,
    "data": None, # DataFrame
    "interval": None
}

# Update fetch to populate cache

@app.get("/api/yfinance/models")
def get_yfinance_models():
    """Get list of available models for playback."""
    from src.model_inference import get_available_models
    models = get_available_models()
    return {"models": models}

```

### ./src/config.py
```py
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Subdirectories for processed data
ONE_MIN_PARQUET_DIR = PROCESSED_DIR / "mes_1min_parquet"
HOUR_FEATURES_DIR = PROCESSED_DIR / "mes_hour_features_parquet"
PATTERNS_DIR = PROCESSED_DIR / "mes_patterns"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DIR, ONE_MIN_PARQUET_DIR, HOUR_FEATURES_DIR, PATTERNS_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data Constants
MES_PREFIX = "MES"
LOCAL_TZ = "America/Chicago"  # Central Time for CME session logic

# Generator Constants
MIN_HOURS_FOR_PATTERN = 5  # Minimum samples to form a cluster bucket
DEFAULT_CLUSTERS = 3        # Default k for k-means

# Physics / Wickiness Controls
GENERATOR_WICK_SCALE = 1.5      # Boost wick sizes
GENERATOR_NOISE_FACTOR = 1.5    # Boost noise/volatility
GENERATOR_REVERSION_PROB = 0.15 # Probability of counter-trend noise
```

### ./src/data_loader.py
```py
import json
import pandas as pd
from pathlib import Path
from typing import List, Union
from src.config import RAW_DATA_DIR, MES_PREFIX
from src.utils.logging_utils import get_logger

logger = get_logger("data_loader")

def load_mes_json_file(path: Path) -> pd.DataFrame:
    """
    Load a single JSON or NDJSON file and return a DataFrame of MES bars.
    """
    try:
        with open(path, 'r') as f:
            # Try loading as standard JSON array first
            try:
                data = json.load(f)
                if not isinstance(data, list):
                     # If it's a single object, wrap in list
                    data = [data]
            except json.JSONDecodeError:
                # Fallback to NDJSON (line delimited)
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]

        if not data:
            logger.warning(f"File {path.name} is empty or invalid.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # Filter for MES symbols
        if 'original_symbol' in df.columns:
            df = df[df['original_symbol'].str.startswith(MES_PREFIX, na=False)]
        
        # Ensure required columns exist
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"File {path.name} missing columns: {missing}")
            return pd.DataFrame()

        # Parse time (assumes ISO 8601 with Z)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        
        # Sort by time just in case
        df.sort_values('time', inplace=True)
        
        return df[['time', 'open', 'high', 'low', 'close', 'volume', 'original_symbol']]

    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

def load_all_mes_bars(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load all matching JSON files from the raw directory and combine them.
    """
    all_dfs = []
    files = list(raw_dir.glob("*.json"))
    
    if not files:
        logger.warning(f"No JSON files found in {raw_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(files)} raw data files.")
    
    for p in files:
        df = load_mes_json_file(p)
        if not df.empty:
            all_dfs.append(df)
            
    if not all_dfs:
        return pd.DataFrame()
        
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Drop duplicates if any (by time and symbol)
    before_dedup = len(combined)
    combined.drop_duplicates(subset=['time', 'original_symbol'], keep='last', inplace=True)
    if len(combined) < before_dedup:
        logger.info(f"Dropped {before_dedup - len(combined)} duplicate records.")
        
    # Sort globally
    combined.sort_values('time', inplace=True)
    combined.reset_index(drop=True, inplace=True)
    
    logger.info(f"Loaded total {len(combined)} MES bars.")
    return combined

if __name__ == "__main__":
    # Test run
    df = load_all_mes_bars()
    print(df.head())
    print(df.info())
```

### ./src/debug_inference_check.py
```py

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model_inference import ModelInference
from src.api import get_mock_candles

def test_inference():
    print("Initializing ModelInference...")
    try:
        # Initialize engine (loads model)
        # Ensure we have a model file. The user history implies one exists.
        engine = ModelInference(model_name="CNN_Predictive_5m") 
        if engine.model is None:
            print("ERROR: Model failed to load.")
            return

        print("Generating 1000 mock candles 1m...")
        # We need to simulate the API's cache structure
        mock_data = get_mock_candles(bars=1000, timeframe="1m")
        results = mock_data["data"]
        
        df = pd.DataFrame(results)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        print(f"Data shape: {df.shape}")
        
        print("Running inference loop...")
        signals_found = 0
        max_prob = 0.0
        
        # We need at least 200 bars
        for i in range(200, len(df)):
            # Force analyze even if not on 5m boundary just to see if ANYTHING fires
            # The analyze method does its own resampling, so calling it on every minute is fine for testing.
            
            # Note: analyze() requires 200 bars history.
            
            # We must be careful about the 'time' column which analyze expects
            res = engine.analyze(i, df)
            
            if res:
                print(f"Index {i}: SIGNAL FOUND! Prob: {res['signal']['prob']:.4f}")
                signals_found += 1
            
            # To check 'aliveness', we might want to peek at the raw probability even if it creates no signal
            # But the `analyze` method swallows the probability if < THRESHOLD
            # So let's monkey-patch or just trust that if we see NOTHING, the threshold is the issue.
            
            # Actually, let's temporarily modify the threshold in memory
            
        print(f"Total Signals with default threshold ({engine.THRESHOLD}): {signals_found}")

        # Test with lower threshold
        print("\nRetesting with threshold 0.01...")
        engine.THRESHOLD = 0.01
        low_thresh_signals = 0
        
        for i in range(200, len(df), 5): # Check every 5 mins
            res = engine.analyze(i, df)
            if res:
                 p = res['signal']['prob']
                 max_prob = max(max_prob, p)
                 if p > 0.10: # Only print interesting ones
                     print(f"Index {i}: Prob {p:.4f}")
                 low_thresh_signals += 1

        print(f"Max Probability seen: {max_prob:.4f}")
        print(f"Signals > 0.01: {low_thresh_signals}")

    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
```

### ./src/debug_miner.py
```py
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import ONE_MIN_PARQUET_DIR

def test():
    print("Start Loading...", flush=True)
    input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows.", flush=True)
    
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time').sort_index()
    print("Index set.", flush=True)
    
    print("Resampling...", flush=True)
    df_5m = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    print(f"Resampled to {len(df_5m)} rows.", flush=True)

if __name__ == "__main__":
    test()
```

### ./src/feature_engineering.py
```py
import pandas as pd
import numpy as np
from scipy.stats import linregress
from src.config import ONE_MIN_PARQUET_DIR, HOUR_FEATURES_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("feature_engineering")

def calc_hour_stats(group):
    """
    Compute features for a single hour group of 1-min bars.
    """
    if len(group) < 10:  # Skip incomplete hours
        return None
    
    # Sort just in case
    group = group.sort_values('time')
    
    # Returns (log returns often better, but simple returns ok for small intervals)
    # Using simple returns: p_t / p_{t-1} - 1
    closes = group['close'].values
    opens = group['open'].values
    highs = group['high'].values
    lows = group['low'].values
    
    returns = np.diff(closes) / closes[:-1]
    
    # Basic Price features
    open_start = opens[0]
    close_end = closes[-1]
    net_return = (close_end / open_start) - 1.0 # Return over the hour
    
    high_max = np.max(highs)
    low_min = np.min(lows)
    price_range = (high_max - low_min) / open_start
    
    # Volatility
    vol = np.std(returns) if len(returns) > 0 else 0
    
    # Skew
    skew = pd.Series(returns).skew() if len(returns) > 2 else 0
    
    # Trend (Slope of price vs time index)
    # Normalize price to start at 1.0 for comparability
    normalized_price = closes / open_start
    x = np.arange(len(normalized_price))
    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, normalized_price)
        trend_slope = slope
        trend_r2 = r_value ** 2
    except:
        trend_slope = 0
        trend_r2 = 0
        
    # Persistence: Directional consistency
    # How many bars closed in the direction of the hour's net move?
    hour_direction = np.sign(net_return)
    if hour_direction == 0:
        persistence = 0
    else:
        bar_returns = (closes - opens) / opens 
        # Fraction of bars with same sign as hour
        persistence = np.mean(np.sign(bar_returns) == hour_direction)
        
    # Metadata from first row
    first = group.iloc[0]
    
    return pd.Series({
        'net_return': net_return,
        'range': price_range,
        'vol': vol,
        'skew': skew,
        'trend_slope': trend_slope,
        'trend_r2': trend_r2,
        'persistence': persistence,
        'count': len(group),
        'start_time': first['time'],  # For reference
        'date': first['date'],
        'hour_bucket': first['hour_bucket'],
        'session_type': first['session_type'],
        'day_of_week': first['day_of_week']
    })

def build_hour_features():
    logger.info("Loading 1-min data...")
    input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    df = pd.read_parquet(input_path)
    
    logger.info("Grouping and computing hour features...")
    # Group by unique hour identifier keys
    # Key: date, hour_bucket (and implied session/day which are constant for that hour)
    
    # We can group by ['date', 'hour_bucket'] directly
    # Note: 'date' and 'hour_bucket' should uniquely identify an hour in the timeline
    
    hour_stats = df.groupby(['date', 'hour_bucket'], group_keys=False).apply(calc_hour_stats)
    
    if hour_stats.empty:
        logger.warning("No hour stats computed.")
        return

    hour_stats.dropna(how='all', inplace=True)
    hour_stats.reset_index(drop=True, inplace=True)
    
    # Standardization (Z-scores)
    # We typically want to cluster based on shape, so we normalize.
    # However, 'net_return' magnitude matters. 
    # Let's add Z-score columns for all numeric features
    feature_cols = ['net_return', 'range', 'vol', 'skew', 'trend_slope', 'trend_r2', 'persistence']
    
    for col in feature_cols:
        mean = hour_stats[col].mean()
        std = hour_stats[col].std()
        hour_stats[f'{col}_z'] = (hour_stats[col] - mean) / (std + 1e-8)
        
    output_path = HOUR_FEATURES_DIR / "mes_hour_features.parquet"
    hour_stats.to_parquet(output_path)
    logger.info(f"Saved {len(hour_stats)} hour feature rows to {output_path}")

if __name__ == "__main__":
    build_hour_features()
```

### ./src/generator.py
```py
import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
from src.config import ONE_MIN_PARQUET_DIR, PATTERNS_DIR, LOCAL_TZ, GENERATOR_WICK_SCALE, GENERATOR_NOISE_FACTOR, GENERATOR_REVERSION_PROB
from src.utils.logging_utils import get_logger

logger = get_logger("generator")

class PatternGenerator:
    _global_used_dates = set()

    def __init__(self):
        self.patterns_df = None
        self.cluster_meta = None
        self.raw_1min = None
        self._load_data()
        
    def _load_data(self):
        logger.info("Loading generator data...")
        
        # Load Pattern Library
        lib_path = PATTERNS_DIR / "mes_pattern_library.parquet"
        if lib_path.exists():
            self.patterns_df = pd.read_parquet(lib_path)
            # Create efficient lookup: (session, dow, hour_bucket) -> list of valid (hour_id / start_time) and their clusters
            # We want to pick a cluster, then pick a historical hour from that cluster.
        else:
            logger.warning("Pattern library not found. Generator will falter.")

        # Load Metadata (frequencies)
        meta_path = PATTERNS_DIR / "cluster_metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.cluster_meta = json.load(f)
                # Helper map: key -> distribution
                self.meta_map = {}
                self.hour_map = {} # (session, hour) -> list of meta
                for m in self.cluster_meta:
                    key = (m['session_type'], m['day_of_week'], m['hour_bucket'])
                    self.meta_map[key] = m
                    
                    h_key = (m['session_type'], m['hour_bucket'])
                    if h_key not in self.hour_map:
                        self.hour_map[h_key] = []
                    self.hour_map[h_key].append(m)
        
        # Load raw data for stitching
        # Ideally, we should have random access. For now, load all into memory (it's small enough for < 1GB local patterns)
        raw_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if raw_path.exists():
            self.raw_1min = pd.read_parquet(raw_path)
            self.raw_1min.set_index('time', inplace=True)
            self.raw_1min.sort_index(inplace=True)
            
            # Compute Physics Baselines
            # 1. Daily Range (approx)
            daily_high = self.raw_1min['high'].resample('D').max()
            daily_low = self.raw_1min['low'].resample('D').min()
            self.avg_daily_range = (daily_high - daily_low).mean()
            if np.isnan(self.avg_daily_range): self.avg_daily_range = 50.0 # fallback
            
            # 2. 1-min Return Volatility (Std Dev)
            self.vol_1min_std = self.raw_1min['close'].pct_change().std()
            if np.isnan(self.vol_1min_std): self.vol_1min_std = 0.0005
            
            logger.info(f"Physics Baseline: Range={self.avg_daily_range:.2f}, 1mVol={self.vol_1min_std:.6f}")
        else:
            logger.error("Raw 1-min data missing!")

    def _get_historical_hour(self, start_time: pd.Timestamp):
        """
        Fetch the DataFrame of 1-min bars for a specific historical hour.
        """
        if self.raw_1min is None:
            return pd.DataFrame()
        
        # Determine end time (start + 1 hour)
        end_time = start_time + pd.Timedelta(hours=1)
        
        # Slice (inclusive of start, exclusive of end usually, but exact hour slicing needs care)
        # Our hours are "08:00" -> covers 08:00 to 08:59 usually.
        # Let's just slice strictly.
        sub = self.raw_1min.loc[start_time:end_time - pd.Timedelta(seconds=1)].copy()
        
        if sub.empty:
            return pd.DataFrame()
            
        # Calc returns relative to the OPEN of the first bar
        # We need to stitch this shape onto a new start price.
        # Shape defined by: (Open_t / Open_0), (High_t / Open_0), etc?
        # Simpler: Use % returns from previous close.
        
        # For stitching:
        # We need the series of Intraday Returns: p_t / p_{start}
        # Or just use the raw Close series and rebase it.
        
        first_open = sub.iloc[0]['open']
        
        sub['rel_open'] = sub['open'] / first_open
        sub['rel_high'] = sub['high'] / first_open
        sub['rel_low'] = sub['low'] / first_open
        sub['rel_close'] = sub['close'] / first_open
        
        # Volume profile can be kept as is or scaled
        
        return sub

    # ... (existing methods above)

    def _calc_state_similarity(self, current_state: dict, cluster_stats: dict) -> float:
        """
        Compute similarity score between current_state and a cluster's historical state stats.
        Higher score = better match.
        """
        if not current_state or not cluster_stats:
            return 1.0 # Neutral
            
        # Simplified distance metric: 1 / (1 + weighted_euclidean_dist)
        # Features: prev_day_ret, trend_3d, vol_1m
        # We need to handle missing keys gracefully
        
        dist = 0.0
        keys = ['prev_day_ret', 'trend_3d', 'vol_1m'] # key features
        
        for k in keys:
            val = current_state.get(k, 0)
            c_stat = cluster_stats.get(k)
            if c_stat:
                mean = c_stat['mean']
                std = c_stat['std'] if c_stat['std'] > 1e-6 else 1.0 # avoid div/0
                
                # Z-score distance
                z = abs(val - mean) / std
                dist += z
        
        # Convert distance to a weight (similarity)
        # e.g. exp(-dist)
        return np.exp(-dist)

    def generate_multi_day(self, num_days: int, session_type: str = "RTH", initial_price: float = 5800.0, start_date: str = None):
        """
        Generate a multi-day synthetic path.
        Generates 'num_days' of TRADING sessions (skips weekends/no-pattern days).
        """
        logger.info(f"Generating {num_days} days starting from {start_date or 'now'}")
        
        # Initial State (default neutral)
        current_state = {
            'prev_day_ret': 0.0,
            'prev_day_range': 0.01,
            'trend_3d': 0.0,
            'vol_1m': 0.001 
        }
        
        current_price = initial_price
        
        # Base Date
        if start_date:
             # Look for start_date, but if it's weekend, we might start searching from there
             base_date = pd.to_datetime(start_date).tz_localize(LOCAL_TZ)
        else:
             base_date = pd.Timestamp.now(tz=LOCAL_TZ).normalize()
             
        full_history = []
        days_generated = 0
        search_offset = 0
        max_lookahead = num_days * 3 # Prevent infinite loop
        # Use simple global history to prevent repetitiveness across calls
        # We clear it if it gets too large relative to available history (e.g. > 50% of dates)
        if len(PatternGenerator._global_used_dates) > 100: 
             PatternGenerator._global_used_dates.clear()
             
        used_source_dates = PatternGenerator._global_used_dates
        
        while days_generated < num_days and search_offset < max_lookahead:
            # Target Date
            target_date = base_date + pd.Timedelta(days=search_offset)
            search_offset += 1
            
            day_of_week = target_date.dayofweek # 0=Mon
            
            # Simple Weekend Skip
            if day_of_week >= 5:
                continue
            
            # 1. Generate Session for this day
            # Determine buckets
            valid_hours = [
                m['hour_bucket'] 
                for m in self.cluster_meta 
                if m['session_type'] == session_type and m['day_of_week'] == day_of_week
            ]
            valid_hours = sorted(list(set(valid_hours)))
            
            if not valid_hours:
                logger.warning(f"No patterns for {target_date.date()} (DOW {day_of_week})")
                continue

            day_bars = []
            # Start time logic...
            start_hour_int = int(valid_hours[0].split(':')[0])
            current_time_cursor = target_date + pd.Timedelta(hours=start_hour_int)
            
            day_open = current_price
            
            for h_bucket in valid_hours:
                # OLD strict lookup: key = (session_type, day_of_week, h_bucket)
                # m = self.meta_map[key] ...
                
                # NEW Soft DOW lookup
                h_key = (session_type, h_bucket)
                if h_key not in self.hour_map:
                    continue
                    
                potential_metas = self.hour_map[h_key]
                
                # DOW Bias Weights
                meta_weights = []
                for m in potential_metas:
                    m_dow = m['day_of_week']
                    dist = abs(m_dow - day_of_week)
                    if dist > 3: dist = 7 - dist # Circular distance (Sun-Mon is 1)
                    
                    if dist == 0: weight = 10.0 # Match
                    elif dist == 1: weight = 3.0 # Neighbor
                    else: weight = 1.0
                    meta_weights.append(weight)
                
                # Pick a source meta (Source DOW)
                chosen_meta = random.choices(potential_metas, weights=meta_weights, k=1)[0]
                
                # Now proceed with this chosen meta (effectively borrowing that DOW's logic)
                meta = chosen_meta
                # IMPORTANT: We must sample from patterns of the CHOSEN DOW, not the target DOW
                source_dow = meta['day_of_week']
                
                clusters = [int(k) for k in meta['cluster_counts'].keys()]
                counts = list(meta['cluster_counts'].values())
                
                # BIASING LOGIC
                weights = []
                state_stats = meta.get('state_stats', {})
                
                for i, c_id in enumerate(clusters):
                    base_weight = counts[i]
                    if state_stats:
                        sim = self._calc_state_similarity(current_state, state_stats.get(str(c_id)))
                        weights.append(base_weight * sim)
                    else:
                        weights.append(base_weight)
                
                # Add Noise to weights (Variety)
                # Multiplicative noise: w * random(0.8, 1.2)
                weights = [w * random.uniform(0.8, 1.2) for w in weights]
                
                total_w = sum(weights)
                if total_w == 0: weights = [1]*len(weights)
                
                chosen_cluster = random.choices(clusters, weights=weights, k=1)[0]
                
                # Sample from CHOSEN DOW (source_dow)
                candidates = self.patterns_df[
                    (self.patterns_df['session_type'] == session_type) &
                    (self.patterns_df['day_of_week'] == source_dow) &
                    (self.patterns_df['hour_bucket'] == h_bucket) &
                    (self.patterns_df['cluster_id'] == chosen_cluster)
                ]
                
                if candidates.empty: continue
                
                # Uniqueness Check
                available_candidates = candidates.copy()
                if used_source_dates:
                     # Filter out rows where start_time date is in use
                     available_candidates['date_str'] = available_candidates['start_time'].astype(str).str.slice(0, 10)
                     available_candidates = available_candidates[~available_candidates['date_str'].isin(used_source_dates)]
                
                if not available_candidates.empty:
                    sample = available_candidates.sample(1).iloc[0]
                else:
                    logger.warning(f"  -> Forced to reuse date for DOW {day_of_week} Cluster {chosen_cluster}")
                    sample = candidates.sample(1).iloc[0]

                hist_start_time = sample['start_time']
                used_source_dates.add(str(hist_start_time).split(' ')[0])
                logger.info(f"  -> Day {days_generated} (DOW {day_of_week}): Cluster {chosen_cluster} | Source: {hist_start_time}")
                
                bars_df = self._get_historical_hour(hist_start_time)
                if bars_df.empty: continue
                
                # Stitching with Physics
                # 1. Calculate historical 1-min returns for this segment
                hist_closes = bars_df['close'].values
                hist_opens = bars_df['open'].values
                hist_highs = bars_df['high'].values
                hist_lows = bars_df['low'].values
                
                # Log returns are safer for compounding
                # but simple % diff is easier to reason about for 1min
                rets = np.diff(hist_closes, prepend=hist_closes[0]) / hist_closes[0] # Relative to start?
                # Actually, standard way: r_t = p_t / p_{t-1} - 1
                period_rets = np.diff(hist_closes) / hist_closes[:-1]
                period_rets = np.insert(period_rets, 0, (hist_closes[0] - bars_df.iloc[0]['open']) / bars_df.iloc[0]['open']) # First bar open-to-close?
                # Let's simplify: Take the shape of Close curve.
                
                # Volatility Scaling
                # Real data ~80 range. Synth was ~250. Scaling factor approx 0.35-0.4 based on observations.
                # Let's use a dynamic approach or fixed "physics scalar".
                # User asked to derive it. 
                # Our pattern library might be selecting high-vol days.
                # Let's enforce the average daily range constraint via a scalar.
                
                # Heuristic: 0.4 damping factor brings 250 -> 100 which is close to 82.
                VOL_SCALE = 0.4 
                
                # Noise Parameters
                # Boost noise with GENERATOR_NOISE_FACTOR to create more intraday volatility (15m wicks)
                NOISE_SCALE = self.vol_1min_std * 0.5 * GENERATOR_NOISE_FACTOR
                
                last_sim_close = current_price
                
                n = len(bars_df)
                times = [current_time_cursor + pd.Timedelta(minutes=i) for i in range(n)]
                
                for i in range(n):
                    # Original Move (Percent)
                    if i == 0:
                        raw_ret = (hist_closes[0] - bars_df.iloc[0]['open']) / bars_df.iloc[0]['open']
                        # Open jump? usually 0 if we stitch perfectly to close
                    else:
                        raw_ret = (hist_closes[i] - hist_closes[i-1]) / hist_closes[i-1]
                    
                    # 1. Scale Volatility
                    scaled_ret = raw_ret * VOL_SCALE
                    
                    # 2. Add Noise (AR(1) or White)
                    # White noise for now, simple
                    noise = np.random.normal(0, NOISE_SCALE)
                    
                    # 3. Anti-Persistence (Mean Reversion chance?)
                    # If the move is strong, there's a chance to revert (chop)
                    # This breaks straight line trends and creates 15m/1h wicks.
                    if abs(scaled_ret) > 0.0001 and random.random() < GENERATOR_REVERSION_PROB:
                         # Flip the sign of the deterministic return component
                         scaled_ret = -0.5 * scaled_ret 
                         # And maybe boost noise for this candle
                         noise *= 1.5
                    
                    final_ret = scaled_ret + noise
                    
                    # Reconstruct
                    sim_close = last_sim_close * (1 + final_ret)
                    sim_open = last_sim_close
                    
                    # Infer High/Low from the scaled body + original wick ratios
                    # Original candle body/wick
                    h_c = hist_closes[i]
                    h_o = hist_opens[i] if i > 0 else bars_df.iloc[0]['open']
                    h_h = hist_highs[i]
                    h_l = hist_lows[i]
                    
                    h_range = h_h - h_l
                    if h_range == 0: h_range = 1e-6
                    
                    # Ratios of wick to range
                    # This assumes shape preservation.
                    # Simpler: Just scale the High/Low deviations from Open/Close by VOL_SCALE too
                    # Apply GENERATOR_WICK_SCALE here to boost atomic 1m wicks
                    
                    sim_high = max(sim_open, sim_close) + (h_h - max(h_o, h_c)) * (current_price / h_c) * VOL_SCALE * GENERATOR_WICK_SCALE
                    sim_low = min(sim_open, sim_close) - (min(h_o, h_c) - h_l) * (current_price / h_c) * VOL_SCALE * GENERATOR_WICK_SCALE
                    
                    # Ensure consistency
                    sim_high = max(sim_high, sim_open, sim_close)
                    sim_low = min(sim_low, sim_open, sim_close)
                    
                    bar = {
                        'time': times[i],
                        'open': sim_open,
                        'high': sim_high,
                        'low': sim_low,
                        'close': sim_close,
                        'volume': bars_df['volume'].iloc[i],
                        'synthetic_day': days_generated
                    }
                    day_bars.append(bar)
                    full_history.append(bar)
                    
                    last_sim_close = sim_close
                
                current_price = last_sim_close
                current_time_cursor += pd.Timedelta(hours=1)
                
            # End of Day: Update State
            if day_bars:
                days_generated += 1 # Success
                
                day_closes = [b['close'] for b in day_bars]
                d_open = day_bars[0]['open']
                d_close = day_bars[-1]['close']
                d_high = max(b['high'] for b in day_bars)
                d_low = min(b['low'] for b in day_bars)
                
                prev_close = current_state.get('last_close', d_open)
                
                current_state['prev_day_ret'] = (d_close / prev_close) - 1 if prev_close else 0
                current_state['prev_day_range'] = (d_high - d_low) / prev_close if prev_close else 0.01
                
                rets = np.diff(day_closes) / day_closes[:-1]
                current_state['vol_1m'] = np.std(rets) if len(rets) > 0 else 0.001
                current_state['trend_3d'] = 0.9 * current_state['trend_3d'] + 0.1 * current_state['prev_day_ret']
                current_state['last_close'] = d_close

        return pd.DataFrame(full_history)

# Singleton instance
_generator = None
def get_generator():
    global _generator
    if _generator is None:
        _generator = PatternGenerator()
    return _generator

if __name__ == "__main__":
    # Test
    gen = get_generator()
    df = gen.generate_session(day_of_week=0, session_type='RTH', start_price=5800.0)
    print(df.head())
    print(df.tail())
```

### ./src/models/cnn_model.py
```py
import torch
import torch.nn as nn

class TradeCNN(nn.Module):
    def __init__(self, input_len=20, input_channels=4):
        super(TradeCNN, self).__init__()
        # Input shape: (Batch, Channels, Length) -> (B, 4, 20)
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2) # Output len -> 10
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Pool again? Output len -> 5
        
        self.fc1 = nn.Linear(64 * 5, 32) 
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape should vary but we expect (Batch, 4, 20) in training loop
        # Check if permute needed if input is (Batch, 20, 4)
        if x.shape[1] != 4:
            x = x.permute(0, 2, 1) 
            
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
```

### ./src/models/train_cnn.py
```py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("train_cnn")

# GPU Check - Strict
if not torch.cuda.is_available():
    logger.error("GPU NOT DETECTED! User required GPU for training.")
    sys.exit(1)

device = torch.device("cuda")
logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

class TradeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1) # Binary classification
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TradeCNN(nn.Module):
    def __init__(self, input_len=20, input_channels=4):
        super(TradeCNN, self).__init__()
        # Input shape: (Batch, Channels, Length) -> (B, 4, 20)
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * 5, 32) 
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (Batch, Length, Channels) from preparation?
        # PyTorch Conv1d expects (Batch, Channels, Length)
        x = x.permute(0, 2, 1) 
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def prepare_data(window_size=20):
    trades_path = PROCESSED_DIR / "engulfing_trades.parquet"
    if not trades_path.exists():
        logger.error("No trades found.")
        return None, None, None

    logger.info("Loading collected trades and raw data...")
    trades = pd.read_parquet(trades_path)
    trades = trades.sort_values('entry_time') # Ensure chronological order
    
    df_1m = pd.read_parquet(ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet")
    df_1m['time'] = pd.to_datetime(df_1m['time'])
    df_1m = df_1m.set_index('time').sort_index()
    
    X = []
    y = []
    
    # We only care about VALID outcomes for training (Win/Loss)
    valid_trades = trades[trades['outcome'].isin(['WIN', 'LOSS'])]
    logger.info(f"Processing {len(valid_trades)} valid trades for training data...")
    
    for idx, trade in valid_trades.iterrows():
        # User Requirement: Train on 20 1m candles BEFORE the trade setup (Engulfing candle).
        # entry_time is the CLOSE of the 15m Engulfing bar.
        # So we need to shift back 15m to get to the OPEN of the Engulfing bar.
        # And then take 20m before THAT.
        
        setup_start_time = trade['entry_time'] - pd.Timedelta(minutes=15)
        end_time = setup_start_time
        start_time = end_time - pd.Timedelta(minutes=window_size)
        
        slice_df = df_1m.loc[start_time:end_time]
        
        # Check size (rough check, might include end_time bar depending on slice exactness)
        # We want strict window.
        slice_df = slice_df[slice_df.index < end_time]
        
        if len(slice_df) < window_size:
            continue
            
        base_price = slice_df.iloc[0]['open']
        if base_price == 0: continue
            
        feats = slice_df[['open', 'high', 'low', 'close']].values
        # Normalize
        feats_norm = (feats / base_price) - 1.0
        
        if len(feats_norm) > window_size:
            feats_norm = feats_norm[-window_size:]
        elif len(feats_norm) < window_size:
             continue
             
        # Label: Win = 1
        label = 1 if trade['outcome'] == 'WIN' else 0
        
        if trade['direction'] == 'SHORT':
             # Invert returns actions
             feats_norm = -feats_norm
             
        X.append(feats_norm)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Prepared {len(X)} samples. Win Rate: {np.mean(y):.2f}")
    return X, y, trades # Return trades to identify split time

def train():
    X, y, trades_df = prepare_data()
    if X is None: return
    
    # 60 / 40 Split (Sequential)
    split_idx = int(0.6 * len(X))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Identify Split Timestamp for Smart Strategy
    # Valid trades were filtered, so finding exact time mapping requires mapping X back to trades
    # But approximate is fine, or we use the trades_df assuming X follows valid_trades order
    # X and y were built iterating `valid_trades`. So split_idx in X corresponds to split_idx in valid_trades.
    
    valid_trades = trades_df[trades_df['outcome'].isin(['WIN', 'LOSS'])]
    split_time = valid_trades.iloc[split_idx]['entry_time']
    
    logger.info(f"Split Index: {split_idx} | Split Time: {split_time}")
    logger.info(f"Train Set: {len(X_train)} | Test Set: {len(X_test)}")
    
    dataset = TradeDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # Shuffle TRAIN only
    
    model = TradeCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    logger.info("Training PyTorch CNN on GPU...")
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f} - Acc: {correct/total:.2f}")
    
    # Save
    model_path = MODELS_DIR / "setup_cnn_v1.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Evaluate on Test Set (One pass)
    test_ds = TradeDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=32)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    logger.info(f"Test Set Accuracy (First Pass): {correct/total:.2f}")

if __name__ == "__main__":
    train()
```

### ./src/models/train_rejection_cnn.py
```py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger
from src.models.cnn_model import TradeCNN

logger = get_logger("train_rejection_cnn")

# GPU Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    logger.warning("GPU NOT DETECTED! Training will be slow.")
else:
    logger.info(f"Using device: {device}")

class TradeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(window_size=20):
    trades_path = PROCESSED_DIR / "labeled_rejections_5m.parquet"
    if not trades_path.exists():
        logger.error("No labeled rejections found. Run pattern_miner.py first.")
        return None, None
        
    logger.info("Loading rejections and raw data...")
    trades = pd.read_parquet(trades_path)
    trades = trades.sort_values('start_time') # Critical for chronological split
    
    df_1m = pd.read_parquet(ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet")
    df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
    df_1m = df_1m.set_index('time').sort_index()
    
    X = []
    y = []
    
    valid_trades = trades[trades['outcome'].isin(['WIN', 'LOSS'])]
    logger.info(f"Processing {len(valid_trades)} valid trades...")
    
    for idx, trade in valid_trades.iterrows():
        # User Logic: Train on 20 1m candles BEFORE the 'First Move' (Start Time).
        end_time = trade['start_time']
        start_time = end_time - pd.Timedelta(minutes=window_size)
        
        # Slice [start_time, end_time)
        slice_df = df_1m.loc[start_time:end_time]
        slice_df = slice_df[slice_df.index < end_time] # Strict inequality
        
        if len(slice_df) < window_size:
            continue
            
        base_price = slice_df.iloc[0]['open']
        if base_price == 0: continue
            
        feats = slice_df[['open', 'high', 'low', 'close']].values
        
        # Normalize: Z-Score per window
        mean = np.mean(feats)
        std = np.std(feats)
        if std == 0: std = 1.0 # Prevent div/0
        
        feats_norm = (feats - mean) / std
        
        # Ensure exact size
        if len(feats_norm) > window_size:
            feats_norm = feats_norm[-window_size:]
        elif len(feats_norm) < window_size:
             continue
             
        # Invert logic for SHORT trades?
        # If the pattern is Short, we want the model to learn "Bearish Context".
        # If Long, "Bullish Context".
        # We can either train separate models or FLIP the data for Shorts so the model learns "Setup Quality" regardless of direction.
        # Given "Proportions" request, flipping is smart.
        # But wait, User said: "price is at 5000 and goes up... we train on...".
        # The pattern miner finds both.
        # Standard approach: Invert price action for Shorts so "Up" always means "Win direction" (or Long setup).
        # But here, 'Win' means Rejection worked.
        # If Short: Price went UP (Extension), then Down.
        # If we Invert Short input: Price went DOWN (Extension), then Up. -> Looks like Long input.
        # So YES, we should invert Shorts to make them look like Longs, or vice versa, to unify the dataset.
        
        if trade['direction'] == 'SHORT':
            feats_norm = -feats_norm
            
        X.append(feats_norm)
        label = 1 if trade['outcome'] == 'WIN' else 0
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Prepared {len(X)} samples. Win Rate: {np.mean(y):.2f}")
    return X, y

def train():
    X, y = prepare_data()
    if X is None or len(X) == 0: 
        logger.error("No data prepared.")
        return
        
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # Datasets
    train_ds = TradeDataset(X_train, y_train)
    test_ds = TradeDataset(X_test, y_test)
    
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64)
    
    model = TradeCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    best_acc = 0.0
    
    logger.info("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct/total
        
        # Eval
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = (outputs > 0.5).float()
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        
        test_acc = test_correct / test_total if test_total > 0 else 0
        
        logger.info(f"Epoch {epoch+1}: Loss {running_loss/len(train_dl):.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODELS_DIR / "rejection_cnn_v1.pth")
            
    logger.info(f"Training Complete. Best Test Acc: {best_acc:.3f}")

if __name__ == "__main__":
    train()
```

### ./src/models/variants.py
```py

import torch
import torch.nn as nn

class CNN_Classic(nn.Module):
    """
    Original Architecture: 20-bar lookback, 4 channels (O,H,L,C).
    2 Conv layers + FC.
    """
    def __init__(self, input_dim=4, seq_len=20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # 20 -> 10
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # 10 -> 5
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 5, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (Batch, Seq, Dim) -> (Batch, Dim, Seq)
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNN_Wide(nn.Module):
    """
    Wide Context Architecture: 60-bar lookback.
    Deeper Conv layers.
    """
    def __init__(self, input_dim=4, seq_len=60):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2), # 60 -> 30
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2), # 30 -> 15
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(3), # 15 -> 5
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LSTM_Seq(nn.Module):
    """
    Recurrent Architecture: 20-bar lookback.
    LSTM layer to capture temporal order.
    """
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (Batch, Seq, Dim)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step)

class Feature_MLP(nn.Module):
    """
    Feature-based Feed Forward.
    Input: Pre-calculated Technical Indicators (Simulated Dim=10 for now).
    """
    def __init__(self, input_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (Batch, Dim) - No sequence dim
        return self.net(x)

class CNN_Predictive(nn.Module):
    """
    Predictive Model: 20-bar lookback.
    Input-Dim: 5 (Open, High, Low, Close, ATR).
    Predicts probability of 15m Rejection.
    """
    def __init__(self, input_dim=5, seq_len=20): 
        super(CNN_Predictive, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (seq_len // 2), 64)
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Time, Feature) -> (Batch, Feature, Time) needed for Conv1d
        # Or (Batch, Feature, Time) already? 
        # Check training script: x = x.permute(0, 2, 1) 
        # In this file, other models assume x is (Batch, Seq, Dim), then transpose.
        # Let's standardize: assume input is (Batch, Seq, Dim)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dout(x)
        x = self.fc2(x)
        return self.sigmoid(x)
```

### ./src/model_inference.py
```py
"""
Real-time model inference for Limit Order Prediction.
Uses the trained CNN_Predictive model (Proactive Strategy).
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.variants import CNN_Predictive
from src.config import MODELS_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("model_inference")

class ModelInference:
    """Handles model loading and inference for predictive limit orders."""
    
    # Model parameters
    WINDOW_SIZE = 20
    ATR_PERIOD_15M = 14
    THRESHOLD = 0.15 # Tuned in Phase 10
    
    def __init__(self, model_name: str = "CNN_Predictive_5m"):
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        
    def _load_model(self):
        """Load the trained model."""
        # Clean up extension if present
        clean_name = self.model_name.replace(".pth", "")
        model_path = MODELS_DIR / f"{clean_name}.pth"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return
            
        try:
            # Input Dim = 5 (OHLC + ATR)
            self.model = CNN_Predictive(input_dim=5).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Model loaded: {clean_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None

    def calculate_15m_atr(self, df_1m: pd.DataFrame) -> pd.Series:
        """
        Calculate 15m ATR from 1m data.
        Returns a Series indexed by timestamp, SHIFTED by 1 (no lookahead).
        """
        if df_1m.empty: return pd.Series()
        
        # Resample to 15m
        df_15m = df_1m.set_index('time').resample('15T').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        if len(df_15m) < self.ATR_PERIOD_15M + 1:
            return pd.Series()
            
        high = df_15m['high']
        low = df_15m['low']
        close = df_15m['close']
        
        tr_list = []
        for i in range(len(df_15m)):
             if i==0: 
                 tr = high.iloc[i]-low.iloc[i]
             else:
                tr = max(high.iloc[i]-low.iloc[i], 
                         abs(high.iloc[i]-close.iloc[i-1]), 
                         abs(low.iloc[i]-close.iloc[i-1]))
             tr_list.append(tr)
             
        atr = pd.Series(tr_list, index=df_15m.index).rolling(self.ATR_PERIOD_15M).mean()
        
        # SHIFT 1: The ATR known at 10:00 is the one calculated from data UP TO 10:00.
        # But `resample` labels 10:00-10:15 as 10:00. 
        # So at 10:15 (start of next bar), we can see the 10:00 ATR.
        
        # We want: At 10:05 (inside 10:00-10:15 bar), we use ATR from 09:45 (closed 10:00).
        # Shift 1 means row 10:00 gets 09:45 value.
        return atr.shift(1)

    def analyze(self, candle_index: int, df_1m: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze the market at `candle_index` (1m index).
        We need to form the feature vector from PREVIOUS data.
        """
        if self.model is None or candle_index < 200: # Need enough buffer for resample
            return None
            
        # 1. Slice data up to current point (simulating 'now')
        # We need enough history to calc ATR and Resample
        # Let's say we take everything up to candle_index (inclusive of the 'current' close?)
        # For prediction, we use PAST 20 bars. So if now is T, we use T-20 to T.
        
        # Current time
        current_time = df_1m.iloc[candle_index]['time']
        
        # Get ATR Series (calculated on whole history available so far)
        # In strictly live mode, we'd only have history up to now.
        history_df = df_1m.iloc[:candle_index+1].copy()
        
        # We need at least enough data for 15m resampling + 14 period ATR
        # ~ 15 * 15 = 225 bars min.
        if len(history_df) < 300: 
            return None
            
        atr_series = self.calculate_15m_atr(history_df)
        
        # Get current effective ATR
        # We want the ATR associated with the *current* 5m bar context.
        # Since we shifted, we can just ffill.
        atr_lookup = atr_series.reindex(history_df['time'], method='ffill')
        current_atr = atr_lookup.iloc[-1]
        
        if pd.isna(current_atr) or current_atr <= 0:
            return None
            
        # 2. Prepare 5m Features (Last 20 bars)
        # Resample history to 5m
        df_5m = history_df.set_index('time').resample('5T').agg({
            'open':'first', 'high':'max', 'low':'min', 'close':'last'
        }).dropna()
        
        if len(df_5m) < self.WINDOW_SIZE:
            return None
            
        # Take last 20
        recent = df_5m.tail(self.WINDOW_SIZE)
        
        # Attach ATR to these 5m bars
        # For the feature block, we need the ATR that was known at each bar's time.
        # reindex handles this if we use the Shifted series.
        recent_atr = atr_series.reindex(recent.index, method='ffill')
        
        # Construct Feature Block: O,H,L,C,ATR
        o = recent['open'].values
        h = recent['high'].values
        l = recent['low'].values
        c = recent['close'].values
        a = recent_atr.values
        
        block = np.stack([o, h, l, c, a], axis=1) # (20, 5)
        
        # Normalize (Z-Score)
        mean = np.nanmean(block, axis=0)
        std = np.nanstd(block, axis=0)
        std[std == 0] = 1e-6
        block = (block - mean) / std
        
        # Inference
        inp = torch.tensor(block, dtype=torch.float32).unsqueeze(0).to(self.device).transpose(1, 2)
        # CNN expects (Batch, Channels, Seq) -> (1, 5, 20)
        # Transpose done in model.forward? 
        # In variants.py: x = x.transpose(1, 2). So input should be (Batch, Seq, Dim).
        # My variants.py logic says "x = x.transpose(1, 2)" to convert (B,S,D) -> (B,D,S).
        # So input should be (B, S, D) = (1, 20, 5).
        
        # Wait, I changed `variants.py` to:
        # def forward(self, x): x = x.transpose(1, 2) ...
        # So I should pass (1, 20, 5).
        inp = torch.tensor(block, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob = self.model(inp).item()
            
        if prob > -1.0: # Always return
            
            # Place Limit Orders
            limit_dist = 1.5 * current_atr
            current_close = float(history_df.iloc[-1]['close']) 
            
            return {
                "signal": {
                    "type": "OCO_LIMIT",
                    "prob": prob,
                    "atr": current_atr,
                    "limit_dist": limit_dist,
                    "current_price": current_close,
                    "sell_limit": current_close + limit_dist,
                    "buy_limit": current_close - limit_dist,
                    "sl_dist": 1.0 * current_atr, 
                    "validity": 15 * 60 
                }
            }


def get_available_models():
    return [f.stem for f in MODELS_DIR.glob("*.pth")]
```

### ./src/pattern_library.py
```py
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.cluster import KMeans
from src.config import HOUR_FEATURES_DIR, PATTERNS_DIR, PROCESSED_DIR, MIN_HOURS_FOR_PATTERN, DEFAULT_CLUSTERS
from src.utils.logging_utils import get_logger

logger = get_logger("pattern_library")

DAY_STATE_PATH = PROCESSED_DIR / "mes_day_state.parquet"

def build_pattern_library():
    logger.info("Loading hour features...")
    input_path = HOUR_FEATURES_DIR / "mes_hour_features.parquet"
    if not input_path.exists():
        logger.error(f"Features file not found: {input_path}")
        return

    df = pd.read_parquet(input_path)
    
    # Load and Join Day State
    if DAY_STATE_PATH.exists():
        logger.info("Joining day state features...")
        state_df = pd.read_parquet(DAY_STATE_PATH)
        # Ensure date type match
        # df['date'] is datetime.date object usually from pandas dt.date
        # state_df['date'] might be object (if loaded from parquet saved that way).
        # Let's force object/string coersion for join key
        df['date_key'] = df['date'].astype(str)
        state_df['date_key'] = state_df['date'].astype(str)
        
        # Merge
        # We want to keep all hour rows, join state info
        df = df.merge(state_df, on='date_key', how='left', suffixes=('', '_state'))
        # Drop temp key, and duplicate date columns if any
        if 'date_state' in df.columns:
            df.drop(columns=['date_state'], inplace=True)
            
        logger.info(f"Joined state. Columns: {df.columns.tolist()}")
    else:
        logger.warning("No day state file found. Continuing without state features.")

    # We cluster per bucket: (session_type, day_of_week, hour_bucket)
    # Features to use for clustering (Standardized versions)
    feature_cols = [c for c in df.columns if c.endswith('_z')]
    
    # Identify state columns (heuristically, ones we added)
    state_cols = ['prev_day_ret', 'prev_day_range', 'trend_3d', 'vol_1m']
    # Filter to only those present
    available_state_cols = [c for c in state_cols if c in df.columns]
    
    if not feature_cols:
        logger.error("No z-score columns found for clustering.")
        return

    buckets = df.groupby(['session_type', 'day_of_week', 'hour_bucket'])
    
    all_patterns = []
    cluster_metadata = []

    logger.info(f"Processing {len(buckets)} time buckets...")
    
    for (session, dow, hour), group in buckets:
        if len(group) < MIN_HOURS_FOR_PATTERN:
            continue
            
        X = group[feature_cols].values
        
        # dynamic k ? simplified -> fixed k for now, or minimal logic
        k = DEFAULT_CLUSTERS
        if len(group) < 50:
            k = 2
            
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        
        labels = kmeans.labels_
        
        # Save mapping back to the df rows
        # We need to identify each hour uniquely. 
        # The group index (original df index) is preserved? 
        # Yes, prompt says: "mapping: hour_id -> cluster_id"
        # Let's add cluster_id to the group subset
        
        group = group.copy()
        group['cluster_id'] = labels
        
        # Store metadata
        counts = pd.Series(labels).value_counts().to_dict()
        total = len(group)
        
        # Compute State Statistics for this cluster
        # e.g. What is the average "prev_day_ret" for hours in this cluster?
        state_stats = {}
        if available_state_cols:
            for c_id in range(k):
                sub = group[group['cluster_id'] == c_id]
                stats = {}
                for s_col in available_state_cols:
                    stats[s_col] = {
                        "mean": float(sub[s_col].mean()),
                        "std": float(sub[s_col].std())
                    }
                state_stats[int(c_id)] = stats

        meta = {
            "session_type": session,
            "day_of_week": int(dow),
            "hour_bucket": hour,
            "k": k,
            "total_samples": total,
            "cluster_counts": {int(c): int(n) for c,n in counts.items()},
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "feature_names": feature_cols,
            "state_stats": state_stats
        }
        cluster_metadata.append(meta)
        
        # Keep relevant columns for the "Pattern Library" dataframe
        # We need: date, hour_bucket, feature_cols, cluster_id + STATE cols
        cols_to_keep = ['date', 'hour_bucket', 'session_type', 'day_of_week', 'start_time', 'cluster_id'] + feature_cols + available_state_cols
        all_patterns.append(group[cols_to_keep])

    if not all_patterns:
        logger.warning("No patterns generated (maybe not enough data?).")
        return

    # 1. Save Pattern Assignments (The "Library")
    full_library = pd.concat(all_patterns, ignore_index=True)
    library_path = PATTERNS_DIR / "mes_pattern_library.parquet"
    full_library.to_parquet(library_path)
    logger.info(f"Saved pattern assignments to {library_path}")

    # 2. Save Metadata (Cluster Configs)
    meta_path = PATTERNS_DIR / "cluster_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(cluster_metadata, f, indent=2)
    logger.info(f"Saved cluster metadata to {meta_path}")

if __name__ == "__main__":
    build_pattern_library()
```

### ./src/pattern_miner.py
```py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR

def run():
    print("Starting Miner...", flush=True)
    input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    # Load
    df = pd.read_parquet(input_path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time').sort_index()
    print(f"Loaded {len(df)} 1m rows.", flush=True)
    
    # Resample 5m
    df_5m = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    print(f"Resampled to {len(df_5m)} 5m rows.", flush=True)
    
    # ATR
    high_low = df_5m['high'] - df_5m['low']
    high_close = np.abs(df_5m['high'] - df_5m['close'].shift())
    low_close = np.abs(df_5m['low'] - df_5m['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_5m['atr'] = tr.rolling(window=14).mean().shift(1)
    
    # Scan
    triggers = []
    opens = df_5m['open'].values
    highs = df_5m['high'].values
    lows = df_5m['low'].values
    times = df_5m.index.values
    atrs = df_5m['atr'].values
    
    n = len(df_5m)
    expansion_ratio = 1.5
    min_atr = 5.0
    max_lookahead = 12 # 60 mins
    
    print("Scanning for patterns...", flush=True)
    for i in range(14, n - max_lookahead):
        start_open = opens[i]
        R = atrs[i]
        
        if np.isnan(R) or R < min_atr: continue
        
        short_tgt = start_open + (expansion_ratio * R)
        long_tgt = start_open - (expansion_ratio * R)
        
        max_r = -99999.0
        min_r = 99999.0
        
        for j in range(i, i + max_lookahead):
            curr_h = highs[j]
            curr_l = lows[j]
            if curr_h > max_r: max_r = curr_h
            if curr_l < min_r: min_r = curr_l
            
            # Short Setup
            if max_r >= short_tgt:
                if curr_l <= start_open:
                    stop_loss = max_r
                    risk = stop_loss - start_open
                    triggers.append({
                        'start_time': times[i],
                        'trigger_time': times[j],
                        'direction': 'SHORT',
                        'entry_price': start_open,
                        'stop_loss': stop_loss,
                        'take_profit': start_open - (1.4 * risk),
                        'risk': risk,
                        'atr': R
                    })
                    break
            
            # Long Setup
            if min_r <= long_tgt:
                if curr_h >= start_open:
                    stop_loss = min_r
                    risk = start_open - stop_loss
                    triggers.append({
                        'start_time': times[i],
                        'trigger_time': times[j],
                        'direction': 'LONG',
                        'entry_price': start_open,
                        'stop_loss': stop_loss,
                        'take_profit': start_open + (1.4 * risk),
                        'risk': risk,
                        'atr': R
                    })
                    break
                    
    print(f"Found {len(triggers)} triggers.", flush=True)
    if not triggers: return
    
    trig_df = pd.DataFrame(triggers)
    
    # Label Outcomes (Precision Mode)
    print("Labeling outcomes...", flush=True)
    trig_df['start_time'] = pd.to_datetime(trig_df['start_time'], utc=True)
    trig_df['trigger_time'] = pd.to_datetime(trig_df['trigger_time'], utc=True)
    
    outcomes = []
    
    for idx, row in trig_df.iterrows():
        # Check from trigger time + 5 mins
        start_check = row['trigger_time'] + pd.Timedelta(minutes=5)
        future = df.loc[start_check:].iloc[:2000] # Check next 2000 1m candles
        
        if future.empty:
            outcomes.append('UNKNOWN')
            continue
            
        highs_f = future['high'].values
        lows_f = future['low'].values
        tp = row['take_profit']
        sl = row['stop_loss']
        outcome = 'TIMEOUT'
        
        if row['direction'] == 'LONG':
            wins = np.where(highs_f >= tp)[0]
            losses = np.where(lows_f <= sl)[0]
        else:
            wins = np.where(lows_f <= tp)[0]
            losses = np.where(highs_f >= sl)[0]
            
        w_idx = wins[0] if len(wins) > 0 else 999999
        l_idx = losses[0] if len(losses) > 0 else 999999
        
        if w_idx < l_idx: outcome = 'WIN'
        elif l_idx < w_idx: outcome = 'LOSS'
        
        outcomes.append(outcome)
        
    trig_df['outcome'] = outcomes
    valid = trig_df[trig_df['outcome'].isin(['WIN', 'LOSS'])]
    print(f"Valid Trades: {len(valid)}", flush=True)
    
    out_path = PROCESSED_DIR / "labeled_rejections_5m.parquet"
    valid.to_parquet(out_path)
    print(f"Saved to {out_path}", flush=True)

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"CRITICAL: {e}")
        traceback.print_exc()
```

### ./src/preprocess.py
```py
import pandas as pd
from src.config import RAW_DATA_DIR, ONE_MIN_PARQUET_DIR, LOCAL_TZ
from src.data_loader import load_all_mes_bars
from src.utils.logging_utils import get_logger

logger = get_logger("preprocess")

def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rich time metadata to the DataFrame:
    - time_local
    - date, day_of_week
    - hour, minute
    - session_type (RTH vs overnight)
    - hour_bucket
    """
    if df.empty:
        return df
        
    # Convert to local time
    df['time_local'] = df['time'].dt.tz_convert(LOCAL_TZ)
    
    df['date'] = df['time_local'].dt.date
    df['day_of_week'] = df['time_local'].dt.dayofweek # 0=Mon, 6=Sun
    df['hour'] = df['time_local'].dt.hour
    df['minute'] = df['time_local'].dt.minute
    
    # RTH Definition: 08:30 to 15:15 (CME Equity Index RTH commonly used, or 9:30-16:00 ET -> 8:30-15:00 CT)
    # Using 08:30 CT to 15:15 CT for broad RTH coverage (includes close). 
    # Adjust as per specific user definition if needed, defaulting to standard CME pit hours approx.
    
    # Vectorized RTH check
    # Minutes from midnight
    mins_from_midnight = df['hour'] * 60 + df['minute']
    
    # 8:30 AM = 510 mins
    # 3:15 PM = 15:15 = 915 mins (Equity closes 15:00 usually, futures trade till 15:15 then break)
    # Let's use 08:30 - 15:00 as Core RTH for pattern learning purpose
    
    rth_start = 8 * 60 + 30
    rth_end = 15 * 60 + 0 # 15:00 CT
    
    df['session_type'] = 'overnight'
    df.loc[(mins_from_midnight >= rth_start) & (mins_from_midnight < rth_end), 'session_type'] = 'RTH'
    
    # Hour bucket: "08:00", "09:00" etc.
    df['hour_bucket'] = df['time_local'].dt.strftime('%H:00')
    
    return df

def build_and_save_1min():
    """
    Load raw data, process features, and save to parquet.
    """
    logger.info("Starting 1-min data build...")
    df = load_all_mes_bars(RAW_DATA_DIR)
    
    if df.empty:
        logger.warning("No data found to process.")
        return
        
    df = add_time_columns(df)
    
    output_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    df.to_parquet(output_path)
    logger.info(f"Saved processed 1-min data to {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    build_and_save_1min()
```

### ./src/sanity_check.py
```py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, PROCESSED_DIR, ONE_MIN_PARQUET_DIR
from src.models.cnn_model import TradeCNN

def check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradeCNN().to(device)
    path = MODELS_DIR / "rejection_cnn_v1.pth"
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    print(f"Loaded model from {path}")
    
    # Check Bias
    zeros = torch.zeros(1, 4, 20).to(device)
    with torch.no_grad():
        out = model(zeros).item()
    print(f"Model(Zeros): {out:.4f}")
    
    # Reproduce One Sample from Strategy Logic
    trades = pd.read_parquet(PROCESSED_DIR / "labeled_rejections.parquet").sort_values('start_time')
    
    # Pick a random trade from the end
    trade = trades.iloc[-100]
    print(f"Checking Trade: {trade['start_time']} ({trade['direction']}) Winner? {trade['outcome']}")
    
    df_1m = pd.read_parquet(ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet")
    df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
    df_1m = df_1m.set_index('time').sort_index()
    
    end_time = trade['start_time']
    start_time = end_time - pd.Timedelta(minutes=20)
    window = df_1m.loc[start_time:end_time]
    window = window[window.index < end_time]
    
    if len(window) < 20:
        print("Window too short")
        return

    feats = window[['open', 'high', 'low', 'close']].values[-20:]
    
    # Z-Score
    mean = np.mean(feats)
    std = np.std(feats)
    if std == 0: std = 1.0
    feats_norm = (feats - mean) / std
    
    if trade['direction'] == 'SHORT':
        feats_norm = -feats_norm
        
    inp = torch.FloatTensor(feats_norm).unsqueeze(0).to(device)
    if inp.shape[1] != 4: inp = inp.permute(0, 2, 1)
        
    with torch.no_grad():
        p = model(inp).item()
        
    print(f"Prediction for Sample: {p:.4f}")

if __name__ == "__main__":
    check()
```

### ./src/scripts/inspect_parquet.py
```py
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import PROCESSED_DIR

df = pd.read_parquet(PROCESSED_DIR / "continuous_1m.parquet")
print(df.columns)
print(df.head())
```

### ./src/scripts/mine_3m_rejection.py
```py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import PROCESSED_DIR

def mine_3m_continuous():
    print("Loading data...")
    df_1m = pd.read_parquet(PROCESSED_DIR / "continuous_1m.parquet")
    
    # Handle Index
    if isinstance(df_1m.index, pd.DatetimeIndex) or 'time' not in df_1m.columns:
        df_1m = df_1m.reset_index()
        
    # Standardize 'time' column name
    # Usually reset_index creates 'index' or 'time' depending on name
    # We force lowercase 'time'
    kw = [c for c in df_1m.columns if 'time' in c.lower() or 'date' in c.lower() or c == 'index']
    if kw:
        # Prefer 'time' or 'datetime'
        target = kw[0]
        for k in kw:
            if k == 'time': target = k
        df_1m = df_1m.rename(columns={target: 'time'})
        
    df_1m = df_1m.sort_values('time').reset_index(drop=True)
    
    # Ensure time is datetime
    df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
    
    # 1. Resample to 3m for Context (ATR, Start Candle definition)
    df_3m = df_1m.set_index('time').resample('3T').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    
    # Calculate ATR (14 period on 3m)
    df_3m['tr'] = np.maximum(
        df_3m['high'] - df_3m['low'],
        np.maximum(
            abs(df_3m['high'] - df_3m['close'].shift(1)),
            abs(df_3m['low'] - df_3m['close'].shift(1))
        )
    )
    df_3m['atr'] = df_3m['tr'].rolling(14).mean()
    
    # Map ATR back to 1m (ffill)
    # We want the ATR known AT the time, so we align by time.
    # 3m ATR available at 10:03 is calculated from 09:xx..10:00-10:03.
    # So for 1m bars inside the next candle, we can use the previous finished 3m ATR.
    df_atr = df_3m[['time', 'atr', 'high']].rename(columns={'high': '3m_high'})
    df_1m = pd.merge_asof(df_1m, df_atr, on='time', direction='backward') 
    
    # 2. Continuous Scanning
    # We look for: Start (Low) -> Peak (High) -> Current (Trigger)
    # R = (Peak - Start) / (Start - Trigger)
    # Valid if 1.5 <= R < 2.5
    # Trigger condition: Start - Current >= 1.0 (Min Unit $1)
    
    labeled_trades = []
    
    # Optimizing the loop:
    # Instead of O(N^2), strict scanning might be slow.
    # We'll use a sliding window approach or limited lookback.
    # Limit lookback for "Start" to e.g., 2 hours (40*3m = 120 bars)?
    
    closes = df_1m['close'].values
    times = df_1m['time'].values
    highs = df_1m['high'].values
    atrs = df_1m['atr'].values
    prices_3m_high = df_1m['3m_high'].values # High of the 3m candle associated with this time
    
    n = len(df_1m)
    print(f"Scanning {n} 1m bars...")
    
    # We need to jump forward to avoid overlapping the exact same trade?
    # Or capture all valid triggers? Let's capture distinct triggers.
    
    last_trade_time = times[0] - np.timedelta64(1, 'D')
    
    for i in range(200, n): # Start after warmup
        curr_time = times[i]
        curr_price = closes[i]
        
        # Optimization: Only check if price is dropping? 
        # Actually we need to find IF there was a valid Start/Peak before.
        
        # Lookback for Start (Minima)
        # We can look back say 120 bars (2 hours)
        lookback = 120
        start_scan = max(0, i - lookback)
        
        # Vectorized check for ratio?
        # Let's just loop backwards for simplicity first, optimize if slow.
        # We need a Start point such that:
        # Start < Peak
        # Current < Start
        
        if i % 10000 == 0:
            print(f"Processed {i}/{n}...")
            
        # Only consider triggers if we haven't traded very recently?
        if (curr_time - last_trade_time) < np.timedelta64(15, 'm'):
             continue
             
        # Find potential starts (local lows)
        # We iterate backwards from i-1
        for j in range(i-1, start_scan, -1):
            start_price = closes[j]
            
            # Condition: Current must be below Start
            drop_size = start_price - curr_price
            if drop_size < 1.0: # Min Unit $1
                continue
                
            # Find Peak in between [j, i]
            # Since j and i are close, just slice
            peaks = highs[j:i]
            peak_price = np.max(peaks)
            
            rise_size = peak_price - start_price
            
            if rise_size <= 0: continue
            
            ratio = rise_size / drop_size
            
            # Geometric Logic
            # 2.5 <= (Peak - Start) / (Start - Current) < 4.0
            # Equivalent to: 2.5 * Drop <= Rise < 4.0 * Drop
            
            if 2.5 <= ratio < 4.0:
                # Setup Found!
                
                # Check Invalidation: Did it exceed 4.0 BEFORE this drop?
                # The logic "Start + 4.0 * Unit" < Peak?
                # Wait, "Start + 4.0 * Unit" corresponds to Ratio 4.0.
                # If current Peak is within < 4.0 range, then it NEVER exceeded 4.0?
                # Correct. If peak_price was higher, ratio would be higher.
                # So simply checking ratio < 4.0 ensures it wasn't a runner.
                
                # Check "Start" Validity - usually a local low?
                # User didn't strictly specify start must be a swing low.
                # But it implies "Beginning of the ascent".
                # Let's assume any point that fits the geometry is a valid perspective.
                
                # DEFINE STOP
                # "stop goes .2 atr above the first candle that began the ascent"
                # We identify the 3m candle period for `times[j]` (Start Time)
                if pd.isna(atrs[j]): continue
                
                # Find the 3m bucket high for the start time
                # We joined '3m_high' previously
                stop_ref_high = prices_3m_high[j]
                stop_level = stop_ref_high + 0.2 * atrs[j]
                
                # Entry & Targets
                entry_price = curr_price
                risk_dist = stop_level - entry_price
                
                if risk_dist <= 0: continue # Invalid logical stop (price already above stop?)
                
                take_profit = entry_price - (1.4 * risk_dist)
                
                # Determine Outcome
                # Look forward from i
                future = df_1m.iloc[i+1:]
                outcome = 'OPEN'
                
                # Fast forward search
                # Hits SL or TP first?
                # Vector search
                sl_hit = future[future['high'] >= stop_level]
                tp_hit = future[future['low'] <= take_profit]
                
                sl_idx = sl_hit.index[0] if not sl_hit.empty else 999999999
                tp_idx = tp_hit.index[0] if not tp_hit.empty else 999999999
                
                if sl_idx == 999999999 and tp_idx == 999999999:
                    outcome = 'Inconclusive'
                elif tp_idx < sl_idx:
                    outcome = 'WIN'
                else:
                    outcome = 'LOSS'
                    
                labeled_trades.append({
                    'start_time': times[j],
                    'trigger_time': curr_time,
                    'unit_size': drop_size,
                    'entry': entry_price,
                    'stop': stop_level,
                    'tp': take_profit,
                    'outcome': outcome,
                    'ratio': ratio,
                    'peak': peak_price
                })
                
                last_trade_time = curr_time
                break # Move to next current time (Greedy: take first valid start for this moment)
                
    print(f"Found {len(labeled_trades)} patterns.")
    if len(labeled_trades) > 0:
        df_res = pd.DataFrame(labeled_trades)
        out_path = PROCESSED_DIR / "labeled_3m_rejection.parquet"
        df_res.to_parquet(out_path)
        print(f"Saved to {out_path}")
        print(df_res['outcome'].value_counts())

if __name__ == "__main__":
    mine_3m_continuous()
```

### ./src/scripts/mine_continuous.py
```py

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR

def mine_patterns():
    input_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not input_path.exists():
        print(f"Error: {input_path} missing.")
        return

    print("Loading continuous 1m data...")
    df = pd.read_parquet(input_path)
    
    # Resample 5m
    print("Resampling to 5m...")
    df_5m = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # ATR
    high_low = df_5m['high'] - df_5m['low']
    high_close = np.abs(df_5m['high'] - df_5m['close'].shift())
    low_close = np.abs(df_5m['low'] - df_5m['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_5m['atr'] = tr.rolling(window=14).mean().shift(1)
    
    # Scan logic (Vectorized / Fast Loop)
    triggers = []
    opens = df_5m['open'].values
    highs = df_5m['high'].values
    lows = df_5m['low'].values
    times = df_5m.index.values
    atrs = df_5m['atr'].values
    
    n = len(df_5m)
    expansion_ratio = 1.5
    min_atr = 5.0
    max_lookahead = 12 
    
    print(f"Scanning {n} candles for patterns...")
    
    for i in range(14, n - max_lookahead):
        R = atrs[i]
        if np.isnan(R) or R < min_atr: continue
        
        start_open = opens[i]
        short_tgt = start_open + (expansion_ratio * R)
        long_tgt = start_open - (expansion_ratio * R)
        
        max_r = -99999.0
        min_r = 99999.0
        
        for j in range(i, i + max_lookahead):
            curr_h = highs[j]
            curr_l = lows[j]
            if curr_h > max_r: max_r = curr_h
            if curr_l < min_r: min_r = curr_l
            
            # Short Setup (Rejection Logic)
            if max_r >= short_tgt:
                if curr_l <= start_open:
                    stop_loss = max_r
                    risk = stop_loss - start_open
                    triggers.append({
                        'start_time': times[i],
                        'trigger_time': times[j],
                        'direction': 'SHORT',
                        'entry_price': start_open,
                        'stop_loss': stop_loss,
                        'take_profit': start_open - (1.4 * risk),
                        'risk': risk,
                        'atr': R
                    })
                    break
            
            # Long Setup
            if min_r <= long_tgt:
                if curr_h >= start_open:
                    stop_loss = min_r
                    risk = start_open - stop_loss
                    triggers.append({
                        'start_time': times[i],
                        'trigger_time': times[j],
                        'direction': 'LONG',
                        'entry_price': start_open,
                        'stop_loss': stop_loss,
                        'take_profit': start_open + (1.4 * risk),
                        'risk': risk,
                        'atr': R
                    })
                    break

    print(f"Found {len(triggers)} potential triggers.")
    if not triggers: return

    trig_df = pd.DataFrame(triggers)
    
    # Label Outcomes (Precision Mode) using 1m data
    print("Labeling outcomes using 1m data...")
    trig_df['trigger_time'] = pd.to_datetime(trig_df['trigger_time'], utc=True)
    
    outcomes = []
    
    # Ensure 1m index is sorted
    df = df.sort_index()
    
    for idx, row in trig_df.iterrows():
        # Check from trigger time + 5 mins (Allow fill? No, assume instant fill at Entry)
        # Actually logic says "trigger_time" is the 5m close time? No, times[j] is the 5m candle timestamp.
        # But the fill happens intra-candle. 
        # For simplicity, let's start checking from the *next* 1m candle after trigger_time (which aligns to 5m start).
        
        start_check = row['trigger_time'] + pd.Timedelta(minutes=5)
        
        future = df.loc[start_check:].iloc[:5000] # Check next 5000 mins ~3 days
        if future.empty:
            outcomes.append('UNKNOWN')
            continue
            
        highs_f = future['high'].values
        lows_f = future['low'].values
        tp = row['take_profit'] # Rejection TP
        sl = row['stop_loss']   # Rejection SL
        
        outcome = 'TIMEOUT'
        
        if row['direction'] == 'LONG':
            # Rejection LONG: Target Up, Stop Down.
            wins = np.where(highs_f >= tp)[0]
            losses = np.where(lows_f <= sl)[0]
        else:
            # Rejection SHORT: Target Down, Stop Up
            wins = np.where(lows_f <= tp)[0]
            losses = np.where(highs_f >= sl)[0]
            
        w_idx = wins[0] if len(wins) > 0 else 999999
        l_idx = losses[0] if len(losses) > 0 else 999999
        
        if w_idx < l_idx: outcome = 'WIN'
        elif l_idx < w_idx: outcome = 'LOSS'
        
        outcomes.append(outcome)
        
    trig_df['outcome'] = outcomes
    valid = trig_df[trig_df['outcome'].isin(['WIN', 'LOSS'])]
    
    print(f"Valid Labeled Trades: {len(valid)}")
    print(valid['outcome'].value_counts())
    
    out_path = PROCESSED_DIR / "labeled_continuous.parquet"
    valid.to_parquet(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    mine_patterns()
```

### ./src/scripts/mine_predictive.py
```py

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR

PROCESSED_DATA_FILE = PROCESSED_DIR / "continuous_1m.parquet"

def calculate_atr(df, period=14):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    tr_list = [high[0]-low[0]]
    for i in range(1, len(df)):
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        tr_list.append(tr)
    return pd.Series(tr_list, index=df.index).rolling(period).mean()

def mine_predictive_labels():
    print("Loading data...")
    df = pd.read_parquet(PROCESSED_DATA_FILE)
    df = df.sort_index()
    
    # Calculate ATR (rolling 14)
    df['atr'] = calculate_atr(df)
    
    # Parameters
    LOOKBACK = 20
    LOOKAHEAD = 15
    ATR_MULT = 1.5
    
    data = []
    
    # Vectorized approach or rolling? 
    # Labeling loop is safer for complex logic
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['atr'].values
    times = df.index
    
    print(f"Scanning {len(df)} candles for predictive setups...")
    
    count_pos = 0
    count_neg = 0
    
    # We step through 1m data
    # At index i (current time), we look at [i-LOOKBACK : i] as Input
    # We look at [i+1 : i+LOOKAHEAD] for Pattern
    
    for i in range(LOOKBACK, len(df) - LOOKAHEAD):
        curr_open = opens[i]
        curr_atr = atrs[i]
        
        if np.isnan(curr_atr) or curr_atr == 0: continue
        
        # Define Targets based on CURRENT info (at time i)
        # We want to place a Limit Sell at: curr_open + 1.5*ATR
        # We want to place a Limit Buy at: curr_open - 1.5*ATR
        short_target = curr_open + (ATR_MULT * curr_atr)
        long_target = curr_open - (ATR_MULT * curr_atr)
        
        # Check Lookahead Window for rejection
        short_success = 0
        long_success = 0
        
        # We want price to HIT the target but REJECT (Close below it for short)
        # Actually, for the "Original" Strategy, we just want price to HIT the target and then revert?
        # Rejection Definition:
        # High >= Target AND Close < Target (Wick created)
        
        for j in range(1, LOOKAHEAD+1):
            future_idx = i + j
            
            # Check Short Rejection
            if highs[future_idx] >= short_target and closes[future_idx] < short_target:
                short_success = 1
                break # Found one
                
        # Check Long Rejection
        for j in range(1, LOOKAHEAD+1):
            future_idx = i + j
            if lows[future_idx] <= long_target and closes[future_idx] > long_target:
                long_success = 1
                break
        
        # Store Feature Window (Indices)
        # We store just the index and label, will load features during training to save space
        
        if short_success:
            data.append({'index': i, 'label': 1, 'type': 'SHORT'})
            count_pos += 1
        elif long_success:
            data.append({'index': i, 'label': 1, 'type': 'LONG'})
            count_pos += 1
        else:
            # Downsample Negatives? 
            # Let's save them all for now, handle balancing in Dataset
            data.append({'index': i, 'label': 0, 'type': 'NONE'})
            count_neg += 1
            
    print(f"Mined {len(data)} samples.")
    print(f"Positives: {count_pos}")
    print(f"Negatives: {count_neg}")
    
    out_df = pd.DataFrame(data)
    out_path = PROCESSED_DIR / "labeled_predictive.parquet"
    out_df.to_parquet(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    mine_predictive_labels()
```

### ./src/scripts/mine_predictive_5m.py
```py

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR

PROCESSED_DATA_FILE = PROCESSED_DIR / "continuous_1m.parquet"

def calculate_atr(df, period=14):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    tr_list = []
    for i in range(len(df)):
        if i==0: tr_list.append(high[i]-low[i])
        else:
            tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
            tr_list.append(tr)
    return pd.Series(tr_list, index=df.index).rolling(period).mean()

def mine_5m_predictive():
    print("Loading data...")
    df = pd.read_parquet(PROCESSED_DATA_FILE).sort_index()
    
    # 1. Resample to 5m (Features)
    print("Resampling to 5m...")
    df_5m = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    # 2. Resample to 15m (ATR)
    print("Resampling to 15m for ATR...")
    df_15m = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    df_15m['atr_15m'] = calculate_atr(df_15m)
    
    # SHIFT ATR to avoid lookahead. 
    # The ATR for 10:00-10:15 (labeled 10:00) is known at 10:15.
    # We want it available for bars starting 10:15 onwards.
    # If we simply join on index, 10:00 5m gets 10:00 15m (Future).
    # Shifting by 1 means 10:15 15m row gets 10:00 15m data.
    # So when 10:15 5m joins to 10:15 15m, it gets the 10:00 ATR. Correct.
    df_15m['atr_15m'] = df_15m['atr_15m'].shift(1)
    
    # 3. Merge ATR back to 5m
    # We want the most recent COMPLETED 15m ATR available at the 5m timestamp
    # ffill allows 5:05 and 5:10 to use the ATR from 5:00 (closed 4:45-5:00 or similar depending on label)
    # Actually, if we use 'close' time, at 5:00 we know ATR of 4:45-5:00.
    
    joined = df_5m.join(df_15m['atr_15m'], how='left')
    joined['atr_15m'] = joined['atr_15m'].ffill()
    
    # Drop initial NaNs
    joined = joined.dropna()
    
    print("Scanning for setups (5m bars, 15m ATR targets)...")
    
    lookback = 20 # 20 x 5m bars = 100m context
    horizon = 3   # 3 x 5m bars = 15m execution window
    
    data = []
    indices = []
    
    opens = joined['open'].values
    highs = joined['high'].values
    lows = joined['low'].values
    closes = joined['close'].values
    atrs = joined['atr_15m'].values
    times = joined.index
    
    pos_count = 0
    neg_count = 0
    
    # Vectorized check might be hard due to horizon logic, doing it loop for clarity
    for i in tqdm(range(lookback, len(joined) - horizon)):
        
        current_close = closes[i]
        atr = atrs[i]
        
        if atr == 0 or np.isnan(atr): continue
        
        # Targets
        short_trigger = current_close + (1.5 * atr)
        long_trigger = current_close - (1.5 * atr)
        
        # Check Horizon (Next 3 bars) for REJECTION
        # Rejection = Hit Trigger AND Revert to Open (Close < Trigger)
        # Actually our strategy is Limit Order:
        # Sell Limit filled if High >= short_trigger.
        # Win if subsequent Low <= Target? 
        # Standard Rejection Strategy definition:
        # Win if we fill and then price moves in our favor.
        # Let's simplify: Label = 1 if High >= ShortTrigger (FILL) AND Price does NOT stop out (ShortTrigger + 1 ATR) before it hits (ShortTrigger - 1.5ATR)?
        #   OR simply: Price hits ShortTrigger and ends up LOWER?
        
        # Let's stick to the definition that worked: 
        # "Rejection" = Price extends 1.5 ATR and CLOSES back below. (Candle color flip logic)
        # But here we have 15m window.
        # Let's verify if a Limit Order would win.
        
        is_positive = False
        
        # Look ahead 3 bars
        for k in range(1, horizon + 1):
            fut_idx = i + k
            h = highs[fut_idx]
            l = lows[fut_idx]
            
            # SHORT setup check
            if h >= short_trigger:
                # Filled short.
                # Did it hit stop?
                sl_price = short_trigger + (1.0 * atr)
                if h >= sl_price:
                    # Stopped out in same specific bar?
                    # Assume worst case: Hit SL first if High touches both
                    # Actually standard assumption: if Close < SL, maybe survived?
                    # Let's just say: Label 1 if "Good Rejection".
                    pass 
                else:
                    # Not stopped.
                    # Did it revert?
                    # Profitable if price comes back down.
                    # Current strategy: TP is 'current_close' (Entry - 1.5 ATR)
                    # So can we hit 'current_close' after hitting 'short_trigger'?
                    if l <= current_close:
                        is_positive = True
                        break
            
            # LONG setup check
            if l <= long_trigger:
                sl_price = long_trigger - (1.0 * atr)
                if l <= sl_price:
                    pass
                else:
                    if h >= current_close:
                        is_positive = True
                        break
                        
        if is_positive:
            indices.append({'index': i, 'label': 1.0, 'time': times[i]})
            pos_count += 1
        elif np.random.rand() < 0.3: # Downsample negatives
            indices.append({'index': i, 'label': 0.0, 'time': times[i]})
            neg_count += 1
            
    print(f"Mined {len(indices)} samples. Pos: {pos_count}, Neg: {neg_count}")
    
    # Save
    labels_df = pd.DataFrame(indices)
    labels_df.to_parquet(PROCESSED_DIR / "labeled_predictive_5m.parquet")
    
    # Save the 5m features for training reader
    joined.to_parquet(PROCESSED_DIR / "features_5m_atr15m.parquet")
    print(f"Saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    mine_5m_predictive()
```

### ./src/scripts/process_continuous.py
```py

import pandas as pd
import json
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import DATA_DIR, PROCESSED_DIR

def process_continuous():
    raw_path = DATA_DIR / "raw" / "continuous_contract.json"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found.")
        return

    print(f"Loading {raw_path}...")
    with open(raw_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} records. Converting to DataFrame...")
    df = pd.DataFrame(data)
    
    # Convert time
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    
    # Basic Clean
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df.astype(float)
    
    # Check for gaps?
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    
    out_path = PROCESSED_DIR / "continuous_1m.parquet"
    df.to_parquet(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    process_continuous()
```

### ./src/scripts/test_yfinance_limits.py
```py

import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("YFLimitTest")

def test_limits():
    ticker = "MES=F"
    
    # Define test cases
    # (Interval, List of Periods to Try)
    test_matrix = [
        ("1m", ["5d", "7d", "8d", "29d", "30d"]),
        ("5m", ["55d", "59d", "60d", "61d", "89d", "90d"]),
        ("15m", ["55d", "59d", "60d", "61d", "89d", "90d"]),
        ("30m", ["55d", "59d", "60d", "61d", "89d", "90d"]),
        ("1h", ["89d", "90d", "1y", "729d", "730d", "731d", "2y"])
    ]
    
    results = []
    
    print(f"Testing YFinance API Limits for {ticker}...\n")
    print(f"{'Interval':<10} | {'Period':<10} | {'Result':<10} | {'Rows':<10} | {'Start Date':<25}")
    print("-" * 80)
    
    for interval, periods in test_matrix:
        for period in periods:
            try:
                # Suppress YFinance noise?
                df = yf.download(ticker, period=period, interval=interval, progress=False, ignore_tz=True)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                status = "OK"
                rows = len(df)
                start_date = "N/A"
                
                if df.empty:
                    status = "EMPTY"
                    rows = 0
                else:
                    df.reset_index(inplace=True)
                    # Check for date col
                    for col in ['Date', 'Datetime']:
                        if col in df.columns:
                            start_date = str(df[col].iloc[0])
                            break
                            
                print(f"{interval:<10} | {period:<10} | {status:<10} | {rows:<10} | {start_date:<25}")
                results.append({'Interval': interval, 'Period': period, 'Status': status, 'Rows': rows, 'Start': start_date})
                
            except Exception as e:
                print(f"{interval:<10} | {period:<10} | ERROR      | 0          | {str(e)[:25]}")
                results.append({'Interval': interval, 'Period': period, 'Status': 'ERROR', 'Rows': 0, 'Start': str(e)})

if __name__ == "__main__":
    test_limits()
```

### ./src/setup_miner.py
```py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
import json
import sys

# Ensure src can be imported
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR, LOCAL_TZ
from src.utils.logging_utils import get_logger

logger = get_logger("setup_miner")

SETUP_FEATURES_PATH = PROCESSED_DIR / "mes_setup_features.parquet"
SETUP_CLUSTERS_PATH = PROCESSED_DIR / "mes_setup_clusters.parquet"
SETUP_RULES_PATH = PROCESSED_DIR / "mes_setup_rules.json"
TREE_RULES_PATH = PROCESSED_DIR / "mes_setup_decision_tree.json"

def build_setup_features(bar_tf: str = "5T", fwd_horizon_bars: int = 12):
    """
    Build a bar-level feature table to mine setups.
    bar_tf: '5T' for 5m, '15T' for 15m, etc.
    fwd_horizon_bars: how many future bars to look at for outcomes (12 * 5m = 60m).
    """
    input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    if not input_path.exists():
        logger.error(f"1-min input not found: {input_path}")
        return

    logger.info("Loading 1-min data...")
    df = pd.read_parquet(input_path)
    df = df.sort_values('time')

    # Resample to bar_tf
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'date': 'first', # Keep date for merging
    }

    logger.info(f"Resampling to {bar_tf}...")
    # Group by date to strictly separate sessions? Or just continuous resample?
    # Continuous is usually better for flowing indicators, but we must handle gaps.
    # Grouping by date ensures no overnight candles if data is purely RTH.
    resampled = (
        df.set_index('time')
          .resample(bar_tf)
          .agg(agg)
          .dropna(subset=['open', 'close'])
          .reset_index()
    )

    # Rebuild time metadata
    resampled['time_local'] = resampled['time'].dt.tz_convert(LOCAL_TZ)
    resampled['hour'] = resampled['time_local'].dt.hour
    resampled['day_of_week'] = resampled['time_local'].dt.dayofweek

    # --- Location Features (PDH/PDL) ---
    logger.info("Computing Location Features...")
    daily = (
        df.groupby('date')
          .agg(
              day_high=('high', 'max'),
              day_low=('low', 'min'),
          )
          .reset_index()
    )
    daily['date_shifted'] = daily['date'] # Logic correction: We need PREVIOUS day
    # Actually, to get previous day stats aligned with current rows, we shift the stats forward
    # But merging is tricky with non-contiguous dates.
    # Simpler: Shift the daily DF indices
    
    # Let's stick to the prompt's logic but implement correctly:
    # We want today's row to have yesterday's High/Low
    # Sort distinct dates
    distinct_dates = sorted(df['date'].unique())
    # This is complex to do vectorized without a calendar. 
    # Pandas shift(-1)? No, we want shift(1).
    
    daily = daily.sort_values('date')
    daily['prev_day_high'] = daily['day_high'].shift(1)
    daily['prev_day_low'] = daily['day_low'].shift(1)
    
    # Merge back to resampled
    # Ensure date types match
    resampled['date'] = resampled['date'].astype(str)
    daily['date'] = daily['date'].astype(str)
    
    resampled = resampled.merge(
        daily[['date', 'prev_day_high', 'prev_day_low']], 
        on='date', 
        how='left'
    )
    
    # Distance in %
    resampled['dist_pdh'] = (resampled['close'] - resampled['prev_day_high']) / resampled['prev_day_high']
    resampled['dist_pdl'] = (resampled['close'] - resampled['prev_day_low']) / resampled['prev_day_low']
    
    # Fill NAs (first day) with 0 or drop? Drop is safer.
    # We will dropna at the end.

    # --- State Features ---
    logger.info("Merging State Features...")
    day_state_path = PROCESSED_DIR / "mes_day_state.parquet"
    if day_state_path.exists():
        state_df = pd.read_parquet(day_state_path)
        state_df['date_key'] = state_df['date'].astype(str)
        # resampled already has 'date' as str
        resampled = resampled.merge(
            state_df[['date_key', 'prev_day_ret', 'prev_day_range', 'trend_3d', 'vol_1m']],
            left_on='date',
            right_on='date_key',
            how='left'
        )
        resampled.drop(columns=['date_key'], inplace=True)
    else:
        logger.warning("No day state file found.")

    # --- Local Behavior Features ---
    logger.info("Computing Local Rolling Stats...")
    # For 5m bars, 30m window = 6 bars
    window_30 = 6 if bar_tf == "5T" else 2 # heuristic
    
    resampled['ret'] = resampled['close'].pct_change()
    
    # Roll Range (High-Low) / Close
    resampled['roll_range_30'] = (
        (resampled['high'].rolling(window_30).max() - resampled['low'].rolling(window_30).min())
        / resampled['close'].shift(1)
    )
    
    resampled['roll_vol_30'] = resampled['ret'].rolling(window_30).std()
    
    # Trend Slope
    # Basic linear slope of closes over window
    def calc_slope(y):
        if len(y) < 2: return 0
        x = np.arange(len(y))
        # Norm y
        y_norm = y / y[0]
        return np.polyfit(x, y_norm, 1)[0]
        
    # Vectorized or apply? apply is slow but fine for mining
    # Optimization: Just use (Close - Close_lag) / Lag / Window ?
    # Let's use simple momentum: (C_t / C_{t-n}) - 1
    resampled['roll_trend_30'] = (resampled['close'] / resampled['close'].shift(window_30)) - 1
    
    # --- Forward Outcomes ---
    logger.info("Computing Forward Outcomes...")
    
    # Shift FUTURE into CURRENT row
    # Next N bars
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=fwd_horizon_bars)
    
    # MFE: Max High in next N bars / Current Close - 1
    fwd_highs = resampled['high'].rolling(window=indexer).max().shift(-1) # shift(-1) to start from next bar
    fwd_lows = resampled['low'].rolling(window=indexer).min().shift(-1)
    fwd_closes = resampled['close'].shift(-fwd_horizon_bars)
    
    entry = resampled['close']
    resampled['fwd_return'] = (fwd_closes / entry) - 1
    resampled['fwd_mfe'] = (fwd_highs / entry) - 1
    resampled['fwd_mae'] = (fwd_lows / entry) - 1 # This will be negative approx
    
    # Labels
    # Expansion: > 0.4% excursion
    resampled['label_expansion'] = (resampled['fwd_mfe'] > 0.004).astype(int)
    
    # Trend Cont: Sign matches and move is decent
    resampled['label_trend_cont'] = (
        (np.sign(resampled['fwd_return']) == np.sign(resampled['roll_trend_30'])) &
        (resampled['fwd_return'].abs() > 0.002)
    ).astype(int)

    # Cleanup
    final_df = resampled.dropna()
    final_df.to_parquet(SETUP_FEATURES_PATH)
    logger.info(f"Saved setup features to {SETUP_FEATURES_PATH} ({len(final_df)} rows)")

def cluster_setups(n_clusters: int = 12):
    """
    Cluster the feature vectors to find recurring 'Setups'.
    """
    if not SETUP_FEATURES_PATH.exists():
        logger.error("No setup features found. Run build_setup_features first.")
        return

    logger.info("Clustering Setups...")
    df = pd.read_parquet(SETUP_FEATURES_PATH)
    
    feature_cols = [
        'hour', 'day_of_week',
        'dist_pdh', 'dist_pdl',
        'roll_range_30', 'roll_vol_30', 'roll_trend_30',
        'prev_day_ret', 'prev_day_range', 'trend_3d', 'vol_1m',
    ]
    
    # Ensure cols exist
    existing_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[existing_cols].values
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df['setup_id'] = kmeans.fit_predict(X_norm)
    
    # Stats
    stats = df.groupby('setup_id').agg({
        'label_expansion': 'mean',
        'fwd_return': 'mean',
        'fwd_mfe': 'mean',
        'fwd_mae': 'mean',
        'setup_id': 'count' # Count
    }).rename(columns={'setup_id': 'count', 'label_expansion': 'exp_rate'})
    
    stats = stats.reset_index()
    
    # Save Clusters (DF needs to be saved to map back if we want to tag charts)
    df.to_parquet(SETUP_CLUSTERS_PATH)
    
    # Metadata
    meta = {
        "feature_cols": existing_cols,
        "n_clusters": n_clusters,
        "setup_stats": stats.to_dict(orient='records')
    }
    
    with open(SETUP_RULES_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
        
    logger.info(f"Saved setup clusters and stats.")
    
def extract_rules(max_depth=4):
    """
    Train decision tree to explain 'label_expansion'
    """
    if not SETUP_FEATURES_PATH.exists():
        return
        
    logger.info("Extracting Rules...")
    df = pd.read_parquet(SETUP_FEATURES_PATH)
    
    feature_cols = [
        'hour', 'day_of_week',
        'dist_pdh', 'dist_pdl',
        'roll_range_30', 'roll_vol_30', 'roll_trend_30',
        'prev_day_ret', 'prev_day_range', 'trend_3d', 'vol_1m',
    ]
    existing_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[existing_cols].values
    y = df['label_expansion'].values
    
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=100, random_state=42)
    clf.fit(X, y)
    
    rules = export_text(clf, feature_names=existing_cols)
    
    output = {
        "target": "Expansion > 0.4%",
        "rules_text": rules
    }
    
    with open(TREE_RULES_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Expansion Rules:\n", rules)

if __name__ == "__main__":
    build_setup_features()
    cluster_setups()
    extract_rules()
```

### ./src/state_features.py
```py
import pandas as pd
import numpy as np
from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("state_features")

DAY_STATE_PATH = PROCESSED_DIR / "mes_day_state.parquet"

def compute_daily_state():
    logger.info("Loading 1-min data for state extraction...")
    input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    df = pd.read_parquet(input_path)
    
    # Needs purely date-based aggregation.
    # Group by date
    days = df.groupby('date')
    
    state_rows = []
    
    # We iterate chronologically to compute "prev day" stats.
    # Actually, simpler: compute daily stats first, then shift.
    
    daily_stats = []
    
    logger.info(f"Computing daily stats for {len(days)} days...")
    
    for date, group in days:
        group = group.sort_values('time')
        open_ = group.iloc[0]['open']
        close = group.iloc[-1]['close']
        high = group['high'].max()
        low = group['low'].min()
        
        # Volatility: std of 1-min returns
        # simple returns
        rets = group['close'].pct_change().dropna()
        daily_vol = rets.std()
        
        daily_stats.append({
            "date": date,  # date object
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "vol_1m": daily_vol,
        })
        
    daily_df = pd.DataFrame(daily_stats)
    daily_df.sort_values('date', inplace=True)
    daily_df.set_index('date', inplace=True)
    
    # Compute Features (Lagged)
    # The "State" for Day T is known at the START of Day T.
    # So it is based on info from Day T-1, T-2...
    
    daily_df['prev_close'] = daily_df['close'].shift(1)
    daily_df['prev_high'] = daily_df['high'].shift(1)
    daily_df['prev_low'] = daily_df['low'].shift(1)
    
    # Net Return (Day T-1)
    daily_df['prev_day_ret'] = (daily_df['close'].shift(1) / daily_df['close'].shift(2)) - 1
    
    # Range (Day T-1) relative to Close T-2
    daily_df['prev_day_range'] = (daily_df['high'].shift(1) - daily_df['low'].shift(1)) / daily_df['close'].shift(2)
    
    # Trend (Last 3 days slope) via regression or simple Close T-1 / Close T-4
    daily_df['trend_3d'] = (daily_df['close'].shift(1) / daily_df['close'].shift(4)) - 1
    
    # Gap (Open T vs Close T-1) - This is technically Day T info, but known at open.
    # We can treat Open T as part of the "Session Start State". 
    # But usually we want purely T-1 info to bias the whole day.
    # Let's keep it T-1 based for now.
    
    # 4h Level Proximity (Simplified)
    # Just use recent highs/lows.
    
    # Fill NA
    daily_df.fillna(0, inplace=True)
    
    # Save
    # Reset index to make date a column
    daily_df.reset_index(inplace=True)
    
    # Ensure date is just date type (it was index from groupby keys, which are datetime.date usually)
    # Convert to string for broader compat or just keep object
    
    logger.info(f"Saving {len(daily_df)} day states to {DAY_STATE_PATH}")
    daily_df.to_parquet(DAY_STATE_PATH)

if __name__ == "__main__":
    compute_daily_state()
```

### ./src/strategies/collector.py
```py
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("collector")

@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str
    stop_loss: float
    take_profit: float
    atr_val: float
    buffer_mult: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None 
    pnl: float = 0.0

class TradeCollector:
    def __init__(self):
        self.trades: List[TradeRecord] = []
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        # DF must be 15m resampled
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def identify_engulfing(self, df: pd.DataFrame):
        # Bullish Engulfing: Prev Red, Curr Green, Curr Body > Prev Body (Overlap)
        # Prev Red: Close < Open
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        
        curr_open = df['open']
        curr_close = df['close']
        
        # Vectorized conditions
        prev_red = prev_close < prev_open
        curr_green = curr_close > curr_open
        
        # Engulfing Body:
        # Bullish: Curr Open <= Prev Close AND Curr Close >= Prev Open
        bull_engulf = (
            prev_red & curr_green & 
            (curr_open <= prev_close) & 
            (curr_close >= prev_open)
        )
        
        # Bearish Engulfing: Prev Green, Curr Red
        # Bearish: Curr Open >= Prev Close AND Curr Close <= Prev Open
        prev_green = prev_close > prev_open
        curr_red = curr_close < curr_open
        
        bear_engulf = (
            prev_green & curr_red &
            (curr_open >= prev_close) &
            (curr_close <= prev_open)
        )
        
        return bull_engulf, bear_engulf

    def run(self):
        input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if not input_path.exists():
            logger.error("Real data not found.")
            return

        logger.info("Loading Real Data...")
        df_1m = pd.read_parquet(input_path)
        df_1m['time'] = pd.to_datetime(df_1m['time'])
        df_1m = df_1m.sort_values('time').set_index('time')
        
        # Resample to 15m
        df_15m = df_1m.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        logger.info("Calculating ATR and Patterns on 15m data...")
        df_15m['atr'] = self.calculate_atr(df_15m)
        
        bull_mask, bear_mask = self.identify_engulfing(df_15m)
        
        # Collect Setup Candidates (Entry at CLOSE of the Engulfing Bar)
        # Entry Time = Timestamp of 15m bar + 15m (Close time)
        # Wait, the index is usually left-bound (start time).
        # Boolean masks align with the row.
        # If row 10:00 is Bull Engulfing, it finishes at 10:15.
        # We enter at Open of 10:15 bar (approx Close of 10:00 bar).
        
        candidates = []
        
        # We iterate only the engulfing bars
        triggers = df_15m[bull_mask | bear_mask]
        logger.info(f"Found {len(triggers)} Engulfing patterns.")
        
        # Pre-process 1m data for simulation lookup
        # Optimization: Don't pass full df to simulation every time if possible.
        # But we need granular lookup.
        
        # SWEEP PARAMETERS
        best_pnl = -float('inf')
        best_trades = []
        best_mult = 0.0
        
        sweep_values = np.arange(0.00, 0.55, 0.05) # 0.0 to 0.5
        
        for mult in sweep_values:
            logger.info(f"Simulating ATR Buffer Multiplier: {mult:.2f}")
            current_trades = []
            
            # Run simulation for this setting
            # We can re-use the candidate list but outcomes differ due to SL width
            
            for ts, row in triggers.iterrows():
                # Data check
                atr = row['atr']
                if pd.isna(atr) or atr == 0: continue
                
                # Determine Direction
                is_bull = bull_mask.loc[ts]
                direction = 'LONG' if is_bull else 'SHORT'
                
                # Pattern High/Low for Base SL
                pattern_high = row['high'] # Is SL usually below the SIGNAL candle? Yes.
                pattern_low = row['low']
                
                atr_buffer = mult * atr
                
                entry_price = row['close'] # Market on Close
                entry_time = ts + pd.Timedelta(minutes=15) # Resolution time
                
                if direction == 'LONG':
                    # SL below Low
                    sl_price = pattern_low - atr_buffer
                    risk = entry_price - sl_price
                    tp_price = entry_price + (1.4 * risk)
                else:
                    # SL above High
                    sl_price = pattern_high + atr_buffer
                    risk = sl_price - entry_price
                    tp_price = entry_price - (1.4 * risk)
                    
                # Sanity: negative risk?
                if risk <= 0: continue
                
                # Simulate Outcome using 1m data
                outcome, pnl, exit_px, exit_t = self.simulate_trade_vectorized(
                    df_1m, entry_time, entry_price, sl_price, tp_price, direction
                )
                
                current_trades.append({
                    'entry_time': entry_time,
                    'direction': direction,
                    'outcome': outcome,
                    'pnl': pnl,
                    'exit_time': exit_t,
                    'exit_price': exit_px,
                    'buffer_mult': mult
                })
                
            # Stats for this sweep
            if not current_trades: continue
            
            df_curr = pd.DataFrame(current_trades)
            total_pnl = df_curr['pnl'].sum()
            win_rate = len(df_curr[df_curr['outcome'] == 'WIN']) / len(df_curr)
            
            logger.info(f"Mult {mult:.2f} | Trades: {len(df_curr)} | WR: {win_rate:.2f} | PnL: {total_pnl:.2f}")
            
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_trades = current_trades
                best_mult = mult
                
        logger.info(f"--- SWEEP COMPLETE ---")
        logger.info(f"Best Multiplier: {best_mult:.2f} | PnL: {best_pnl:.2f}")
        
        # Save Best
        if best_trades:
            df_out = pd.DataFrame(best_trades)
            out_path = PROCESSED_DIR / "engulfing_trades.parquet"
            df_out.to_parquet(out_path)
            logger.info(f"Saved {len(df_out)} best trades to {out_path}")

    def simulate_trade_vectorized(self, df_1m, entry_time, entry_price, sl_price, tp_price, direction):
        # Subset
        subset = df_1m.loc[entry_time:].iloc[:2880] # Max 2 days (1440 * 2)
        if subset.empty:
            return 'TIMEOUT', 0.0, entry_price, entry_time
            
        times = subset.index.values
        highs = subset['high'].values
        lows = subset['low'].values
        closes = subset['close'].values
        
        if direction == 'LONG':
             mask_win = highs >= tp_price
             mask_loss = lows <= sl_price
        else:
             mask_win = lows <= tp_price
             mask_loss = highs >= sl_price
             
        idx_win = np.argmax(mask_win) if mask_win.any() else 999999
        idx_loss = np.argmax(mask_loss) if mask_loss.any() else 999999
        
        if idx_win == 999999 and idx_loss == 999999:
            outcome = 'TIMEOUT'
            exit_px = closes[-1]
            exit_t = times[-1]
            pnl = (exit_px - entry_price) * (1 if direction == 'LONG' else -1)
        elif idx_win < idx_loss:
            outcome = 'WIN'
            exit_px = tp_price
            exit_t = times[idx_win]
            pnl = (tp_price - entry_price) * (1 if direction == 'LONG' else -1) # Exact TP PnL
        else:
            outcome = 'LOSS'
            exit_px = sl_price
            exit_t = times[idx_loss]
            pnl = (sl_price - entry_price) * (1 if direction == 'LONG' else -1) # Exact SL PnL
            
        return outcome, pnl, exit_px, exit_t

if __name__ == "__main__":
    c = TradeCollector()
    c.run()
```

### ./src/strategies/inverse_strategy.py
```py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import PROCESSED_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("inverse_strategy")

def run_inverse_backtest():
    logger.info("Running Inverse (Fade) Strategy Backtest...")
    
    trades_path = PROCESSED_DIR / "labeled_rejections_5m.parquet"
    if not trades_path.exists():
        logger.error("Trades file missing.")
        return
        
    trades = pd.read_parquet(trades_path).sort_values('start_time')
    
    # We use the whole dataset or just test? User said "fade all entries".
    # Let's use the same Test split (last 20%) to be comparable, 
    # OR since we aren't using a model (blind fade), we can run on ALL.
    # Let's run on ALL to see robust stats.
    
    logger.info(f"Total Triggers: {len(trades)}")
    
    initial_balance = 50000.0
    risk_per_trade = 300.0
    balance = initial_balance
    
    wins = 0
    losses = 0
    total_pnl = 0.0
    
    for idx, trade in trades.iterrows():
        # Rejection Strategy:
        # SHORT (Price went Up, we Sell). Target = Down. SL = Up (Extreme).
        # Outcome WIN = Price went Down.
        # Outcome LOSS = Price went Up (Hit SL/Extreme).
        
        # MEANING OF INVERSE (FADE THE ENTRY):
        # We see Price went Up. We see Return to Open.
        # Instead of Selling (Rejection), we BUY (Continuation).
        # We Target the Extreme (High).
        # We Stop Out if Price goes Down (Rejection Win).
        
        # So:
        # Unique mapping:
        # Rejection LOSS -> Price hit Extreme -> Inverse WIN.
        # Rejection WIN -> Price hit Rejection Target -> Inverse LOSS.
        
        original_outcome = trade['outcome']
        if original_outcome not in ['WIN', 'LOSS']: continue
        
        pnl = 0.0
        
        if original_outcome == 'LOSS':
            # Inverse WIN
            # We risk $300.
            # Reward? 
            # In Rejection: Risk = |Entry - Extreme|.
            # In Inverse: Target = Extreme. So Reward = |Entry - Extreme|.
            # So Reward = 1.0 * Risk (Distance to Extreme).
            # Wait, Rejection had SL at Extreme. So Risk distance = Extreme.
            # Inverse Target is Extreme. So Reward distance = Risk distance.
            # So Reward = 1R.
            
            pnl = risk_per_trade * 1.0 # 1:1 Reward
            wins += 1
            
        elif original_outcome == 'WIN':
            # Inverse LOSS
            # We Stop Out.
            # Risk = $300.
            pnl = -risk_per_trade
            losses += 1
            
        balance += pnl
        total_pnl += pnl
        
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    logger.info("--------------------------------------------------")
    logger.info("INVERSE STRATEGY RESULTS (Fading the Rejection)")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {win_rate*100:.2f}%")
    logger.info(f"Total PnL: ${total_pnl:.2f}")
    logger.info(f"Final Balance: ${balance:.2f}")
    logger.info("--------------------------------------------------")

if __name__ == "__main__":
    run_inverse_backtest()
```

### ./src/strategies/random_tilt.py
```py
import pandas as pd
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import sys
# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR, LOCAL_TZ
from src.utils.logging_utils import get_logger

logger = get_logger("random_tilt")

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str # 'LONG' or 'SHORT'
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = 0.0
    outcome: str = None # 'WIN', 'LOSS'
    setup_window_start: pd.Timestamp = None # For CNN mapping

class RandomTiltStrategy:
    def __init__(self, 
                 tp_ticks: int = 20, 
                 sl_ticks: int = 10, 
                 tick_size: float = 0.25,
                 base_prob: float = 0.5):
        self.tp_dist = tp_ticks * tick_size
        self.sl_dist = sl_ticks * tick_size
        self.tick_size = tick_size
        self.base_prob = base_prob # Base probability for Long
        self.trades: List[Trade] = []
        
    def calculate_tilt(self, df_window: pd.DataFrame) -> float:
        """
        Determine probability of going LONG based on recent price action.
        Simple logic: 
        - If recent trend is up, tilt long (e.g. 0.6).
        - If range is tight, keep near 0.5.
        """
        if len(df_window) < 2:
            return 0.5
            
        last_close = df_window.iloc[-1]['close']
        prev_close = df_window.iloc[-2]['close']
        
        # Simple Momentum Tilt
        # If last 5m candle was Green, tilt Long slightly?
        tilt = 0.5
        
        # Trend Component (last 3 bars)
        closes = df_window['close'].values
        if len(closes) >= 3:
            slope = (closes[-1] - closes[-3]) / closes[-3]
            # Max tilt +/- 0.2
            # e.g. 0.1% move -> 0.1 tilt?
            dir_tilt = np.clip(slope * 100, -0.2, 0.2)
            tilt += dir_tilt
            
        return np.clip(tilt, 0.1, 0.9)

    def run_simulation(self, start_date: str = "2024-01-01", days: int = 20):
        """
        Run simulation over 1-min data (resampled to 5m for decisions).
        """
        input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if not input_path.exists():
            logger.error("Data not found.")
            return

        logger.info(f"Loading data for simulation ({days} days)...")
        df_1m = pd.read_parquet(input_path)
        df_1m['time'] = pd.to_datetime(df_1m['time'])
        df_1m = df_1m.sort_values('time')
        
        # Filter date range? For now just take tail if needed or all
        # Let's resample to 5m for decision points
        df_5m = df_1m.set_index('time').resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna().reset_index()
        
        logger.info(f"Simulating on {len(df_5m)} 5-min bars...")
        
        # Simulation Loop
        # We need 1-min granularity for realistic SL/TP fills
        # But decisions happen at 5m close.
        
        # Map 5m decision times to 1m indices for lookup
        # Optimization: Iterate 5m bars, then look ahead in 1m data for outcome.
        
        # Build lookup for 1m data
        df_1m_indexed = df_1m.set_index('time').sort_index()
        
        for i in range(4, len(df_5m)-1): 
            # Context window (e.g. last 4 bars for calculating tilt)
            window = df_5m.iloc[i-4:i+1]
            current_bar = df_5m.iloc[i]
            
            # Decision Time: Close of current_bar
            decision_time = current_bar['time'] + pd.Timedelta(minutes=5)
            
            # 1. Calc Tilt
            prob_long = self.calculate_tilt(window)
            
            # 2. Decision
            # Random roll
            is_long = random.random() < prob_long
            
            direction = 'LONG' if is_long else 'SHORT'
            entry_price = current_bar['close'] # Market on Close (theoretical)
            # Actually next bar Open is more realistic, let's assume fill at Close for simplicity/speed
            
            # 3. Resolve Outcome (Look ahead in 1m data)
            # We look up to X hours ahead or until fill
            future_1m = df_1m_indexed.loc[decision_time : decision_time + pd.Timedelta(hours=2)]
            
            if future_1m.empty:
                continue
                
            tp_price = entry_price + self.tp_dist if is_long else entry_price - self.tp_dist
            sl_price = entry_price - self.sl_dist if is_long else entry_price + self.sl_dist
            
            outcome = 'TIMEOUT'
            exit_px = future_1m.iloc[-1]['close']
            exit_time = future_1m.iloc[-1].name
            
            for _, row in future_1m.iterrows():
                h, l = row['high'], row['low']
                
                # Check stops
                passed_tp = (h >= tp_price) if is_long else (l <= tp_price)
                passed_sl = (l <= sl_price) if is_long else (h >= sl_price)
                
                if passed_sl and passed_tp:
                    # Conflict: Both hit in same minute. Assume SL (stats conservatism)
                    outcome = 'LOSS'
                    exit_px = sl_price
                    exit_time = row.name
                    break
                elif passed_sl:
                    outcome = 'LOSS'
                    exit_px = sl_price
                    exit_time = row.name
                    break
                elif passed_tp:
                    outcome = 'WIN'
                    exit_px = tp_price
                    exit_time = row.name
                    break
            
            # Record Trade
            pnl = (exit_px - entry_price) * (1 if is_long else -1)
            
            # Save 20m window start (for CNN)
            # 20m before decision time
            setup_start = decision_time - pd.Timedelta(minutes=20)
            
            t = Trade(
                entry_time=decision_time,
                entry_price=entry_price,
                direction=direction,
                exit_time=exit_time,
                exit_price=exit_px,
                pnl=pnl,
                outcome=outcome,
                setup_window_start=setup_start
            )
            self.trades.append(t)
            
        logger.info(f"Simulation complete. Generated {len(self.trades)} trades.")
        self.save_trades()

    def save_trades(self):
        records = [
            {
                'entry_time': t.entry_time,
                'entry_price': t.entry_price,
                'direction': t.direction,
                'exit_time': t.exit_time,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'outcome': t.outcome,
                'setup_window_start': t.setup_window_start
            }
            for t in self.trades
        ]
        df = pd.DataFrame(records)
        out_path = PROCESSED_DIR / "random_tilt_trades.parquet"
        df.to_parquet(out_path)
        logger.info(f"Saved trades to {out_path}")

if __name__ == "__main__":
    # Test Run
    strat = RandomTiltStrategy()
    strat.run_simulation(days=5)
```

### ./src/strategies/rejection_strategy.py
```py
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger
from src.models.cnn_model import TradeCNN

logger = get_logger("rejection_strategy")

class RejectionStrategy:
    def __init__(self, 
                 model_path: Path = MODELS_DIR / "rejection_cnn_v1.pth",
                 risk_per_trade: float = 300.0,
                 initial_balance: float = 50000.0,
                 threshold: float = 0.6):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradeCNN().to(self.device)
        
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.error(f"Model not found at {model_path}")
            sys.exit(1)
            
        self.risk_per_trade = risk_per_trade
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.threshold = threshold
        self.trades = []
        
    def get_prediction(self, df_window, direction):
        # Prepare Input
        feats = df_window[['open', 'high', 'low', 'close']].values
        
        mean = np.mean(feats)
        std = np.std(feats)
        if std == 0: std = 1.0
        
        feats_norm = (feats - mean) / std
        
        # Add batch dim
        
        # Add batch dim
        inp = torch.FloatTensor(feats_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob = self.model(inp).item()
            
        return prob

    def run_backtest(self):
        logger.info("Starting Backtest on Test Set (Latest 20%)...")
        
        # Load Trades
        trades_path = PROCESSED_DIR / "labeled_rejections_5m.parquet"
        if not trades_path.exists():
            logger.error("Trades file missing.")
            return
            
        trades = pd.read_parquet(trades_path).sort_values('start_time')
        
        # Split - take last 20%
        split_idx = int(0.8 * len(trades))
        test_trades = trades.iloc[split_idx:].copy()
        
        logger.info(f"Test Set has {len(test_trades)} potential triggers.")
        
        # Load Candles
        df_1m = pd.read_parquet(ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet")
        df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
        df_1m = df_1m.set_index('time').sort_index()
        
        confidences = []
        executed_trades = []
        
        for idx, trade in test_trades.iterrows():
            result = trade['outcome']
            if result not in ['WIN', 'LOSS']: continue
            
            # Prepare Input Context
            start_time = trade['start_time'] - pd.Timedelta(minutes=20)
            end_time = trade['start_time']
            
            # Input window
            window = df_1m.loc[start_time:end_time]
            window = window[window.index < end_time]
            
            if len(window) < 20: continue
            if len(window) > 20: window = window.iloc[-20:]
            
            # Model Inference
            confidence = self.get_prediction(window, trade['direction'])
            confidences.append(confidence)
            
            # Execute Trade ALWAYS (Threshold 0.0 check essentially)
            if confidence >= -1.0: # Force Execute
                pnl = 0.0
                if result == 'WIN':
                    pnl = self.risk_per_trade * 1.4
                else: # LOSS
                    pnl = -self.risk_per_trade
                
                self.balance += pnl
                
                executed_trades.append({
                    'entry_time': trade['trigger_time'],
                    'direction': trade['direction'],
                    'confidence': confidence,
                    'outcome': result,
                    'pnl': pnl,
                    'balance': self.balance
                })
        
        # Analysis
        if not executed_trades:
            logger.warning("No trades executed.")
            return
            
        results_df = pd.DataFrame(executed_trades)
        wins = results_df[results_df['outcome'] == 'WIN']
        win_rate = len(wins) / len(results_df)
        total_pnl = results_df['pnl'].sum()
        
        conf_arr = np.array(confidences)
        logger.info(f"Confidence Stats: Min={conf_arr.min():.4f}, Max={conf_arr.max():.4f}, Mean={conf_arr.mean():.4f}")
        logger.info(f"Confidence Hist: <0.4: {np.sum(conf_arr < 0.4)}, 0.4-0.6: {np.sum((conf_arr >= 0.4) & (conf_arr < 0.6))}, >0.6: {np.sum(conf_arr >= 0.6)}")
        
        # Analysis
        if not executed_trades:
            logger.warning("No trades executed.")
            return
            
        results_df = pd.DataFrame(executed_trades)
        wins = results_df[results_df['outcome'] == 'WIN']
        win_rate = len(wins) / len(results_df)
        total_pnl = results_df['pnl'].sum()
        
        logger.info("--------------------------------------------------")
        logger.info(f"Backtest Complete.")
        logger.info(f"Trades Taken: {len(results_df)} / {len(test_trades)} potential")
        logger.info(f"Win Rate: {win_rate*100:.2f}%")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info(f"Final Balance: ${self.balance:.2f} (Start: ${self.initial_balance})")
        logger.info("--------------------------------------------------")
        
        out_path = PROCESSED_DIR / "rejection_strategy_results.csv"
        results_df.to_csv(out_path, index=False)
        logger.info(f"Results saved to {out_path}")

if __name__ == "__main__":
    strat = RejectionStrategy(threshold=0.6)
    strat.run_backtest()
```

### ./src/strategies/smart_cnn.py
```py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR, MODELS_DIR, LOCAL_TZ
from src.utils.logging_utils import get_logger

logger = get_logger("smart_cnn")

# --- Architecture (Must match training!) ---
class TradeCNN(nn.Module):
    def __init__(self, input_len=20, input_channels=4):
        super(TradeCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * 5, 32) 
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str 
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = 0.0
    outcome: str = None 

class SmartCNNStrategy:
    def __init__(self, 
                 model_path: Path = MODELS_DIR / "setup_cnn_v1.pth",
                 tp_ticks: int = 20, 
                 sl_ticks: int = 10,
                 threshold: float = 0.6): # Confidence threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradeCNN().to(self.device)
        
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.error(f"Model not found at {model_path}")
            
        self.tp_dist = tp_ticks * 0.25
        self.sl_dist = sl_ticks * 0.25
        self.threshold = threshold
        self.trades = []

    def get_prediction(self, df_window):
        # Prepare Input
        # Needs 20 bars. 
        if len(df_window) < 20: 
            return 0.0, 0.0 # Prob Long, Prob Short
            
        # Normalize
        base_price = df_window.iloc[0]['open']
        feats = df_window[['open', 'high', 'low', 'close']].values
        feats_norm = (feats / base_price) - 1.0
        
        # Ensure exact 20
        feats_norm = feats_norm[-20:]
        
        # Create Batch (1, 20, 4) -> (1, 4, 20) handled by model
        # Input Long
        input_long = torch.FloatTensor(feats_norm).unsqueeze(0).to(self.device)
        # Input Short (Inverted)
        input_short = torch.FloatTensor(-feats_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob_long = self.model(input_long).item()
            prob_short = self.model(input_short).item()
            
        return prob_long, prob_short

    def run_simulation(self, start_date_str: str = "2025-07-07 16:40:00"):
        input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if not input_path.exists(): return
        
        logger.info(f"Simulating Smart Strategy (Test Set starting {start_date_str})...")
        df_1m = pd.read_parquet(input_path)
        df_1m['time'] = pd.to_datetime(df_1m['time'])
        df_1m = df_1m.sort_values('time').set_index('time')
        
        # Filter for Test Period
        start_ts = pd.Timestamp(start_date_str).tz_localize('UTC') if 'UTC' not in start_date_str else pd.Timestamp(start_date_str)
        # Check tz awareness of df
        if df_1m.index.tz is None:
            # Assume UTC if data is UTC
            pass 
        else:
            if start_ts.tz is None: start_ts = start_ts.tz_localize('UTC')
        
        # We need context Before start_ts, so slice generously then filter triggers
        df_1m_test = df_1m.loc[start_ts - pd.Timedelta(hours=1):]
        
        # Resample for 20m triggers
        # We need "Last 5m candle" reference.
        df_5m = df_1m_test.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        triggers = df_5m[df_5m.index.minute % 20 == 0]
        triggers = triggers[triggers.index >= start_ts]
        
        logger.info(f"Found {len(triggers)} test opportunities.")
        
        count = 0
        for current_time, row in triggers.iterrows():
            trigger_time = current_time
            
            # Context for Model: 20m before trigger
            context_end = trigger_time
            context_start = context_end - pd.Timedelta(minutes=20)
            
            # Fetch 1m context
            context_window = df_1m.loc[context_start:context_end]
            # Precise slice: strictly < trigger_time?
            context_window = context_window[context_window.index < trigger_time]
            
            if len(context_window) < 15: continue # Skip if missing data
            
            p_long, p_short = self.get_prediction(context_window)
            
            if count < 20: 
                 logger.info(f"Pred: L={p_long:.4f} S={p_short:.4f}")
            
            # Decision
            direction = None
            # Relaxed logic: If both meet threshold, pick higher or random tie-break
            if p_long > self.threshold and p_long >= p_short:
                direction = 'LONG'
                confidence = p_long
            elif p_short > self.threshold and p_short > p_long:
                direction = 'SHORT'
                confidence = p_short
                
            if not direction:
                continue
                
            # EXECUTION (Dynamic Sizing)
            # Need previous 5m candle for sizing
            prev_time = trigger_time - pd.Timedelta(minutes=5)
            if prev_time not in df_5m.index: continue
            prev_bar = df_5m.loc[prev_time]
            candle_range = prev_bar['high'] - prev_bar['low']
            if candle_range == 0: candle_range = 0.25
            
            sl_dist = 2.0 * candle_range
            tp_dist = 3.0 * candle_range
            
            entry_price = prev_bar['close'] # Approx fill at close of prev bar (Open of current)
            if trigger_time in df_1m.index:
                 entry_price = df_1m.loc[trigger_time]['open']
            
            if direction == 'LONG':
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist
                
            # Simulate Outcome
            future = df_1m.loc[trigger_time:]
            outcome = 'TIMEOUT'
            exit_px = entry_price
            exit_t = trigger_time
            
            # Vectorized Check (Subset 2000 bars)
            subset = future.iloc[:2000]
            if subset.empty: continue
            
            times = subset.index.values
            highs = subset['high'].values
            lows = subset['low'].values
            closes = subset['close'].values
            
            if direction == 'LONG':
                 mask_win = highs >= tp_price
                 mask_loss = lows <= sl_price
            else:
                 mask_win = lows <= tp_price
                 mask_loss = highs >= sl_price
                 
            idx_win = np.argmax(mask_win) if mask_win.any() else 999999
            idx_loss = np.argmax(mask_loss) if mask_loss.any() else 999999
            
            if idx_win == 999999 and idx_loss == 999999:
                outcome = 'TIMEOUT'
                exit_px = closes[-1]
                exit_t = times[-1]
            elif idx_win < idx_loss:
                outcome = 'WIN'
                exit_px = tp_price
                exit_t = times[idx_win]
            else:
                outcome = 'LOSS'
                exit_px = sl_price
                exit_t = times[idx_loss]
            
            pnl = (exit_px - entry_price) * (1 if direction == 'LONG' else -1)
            
            self.trades.append({
                'entry_time': trigger_time,
                'direction': direction,
                'pnl': pnl,
                'outcome': outcome,
                'confidence': confidence
            })
            
            count += 1
            if count % 100 == 0:
                logger.info(f"Simulated {count} trades... Last PnL: {pnl:.2f}")

        logger.info(f"Smart Simulation Complete. Trades: {len(self.trades)}")
        if self.trades:
            df_res = pd.DataFrame(self.trades)
            wins = df_res[df_res['outcome'] == 'WIN']
            wr = len(wins) / len(df_res)
            logger.info(f"Win Rate: {wr:.2f} | Avg PnL: {df_res['pnl'].mean():.2f} | Total PnL: {df_res['pnl'].sum():.2f}")
            out_path = PROCESSED_DIR / "smart_verification_trades.parquet"
            df_res.to_parquet(out_path)

if __name__ == "__main__":
    # Threshold 0.38 since model output is around 0.40
    strat = SmartCNNStrategy(threshold=0.38) 
    strat.run_simulation()
```

### ./src/strategies/smart_reverse.py
```py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import ONE_MIN_PARQUET_DIR, PROCESSED_DIR, MODELS_DIR, LOCAL_TZ
from src.utils.logging_utils import get_logger

logger = get_logger("smart_cnn")

# --- Architecture (Must match training!) ---
class TradeCNN(nn.Module):
    def __init__(self, input_len=20, input_channels=4):
        super(TradeCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * 5, 32) 
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str 
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = 0.0
    outcome: str = None 

class SmartCNNStrategy:
    def __init__(self, 
                 model_path: Path = MODELS_DIR / "setup_cnn_v1.pth",
                 tp_ticks: int = 20, 
                 sl_ticks: int = 10,
                 threshold: float = 0.6): # Confidence threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradeCNN().to(self.device)
        
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.error(f"Model not found at {model_path}")
            
        self.tp_dist = tp_ticks * 0.25
        self.sl_dist = sl_ticks * 0.25
        self.threshold = threshold
        self.trades = []

    def get_prediction(self, df_window):
        # Prepare Input
        # Needs 20 bars. 
        if len(df_window) < 20: 
            return 0.0, 0.0 # Prob Long, Prob Short
            
        # Normalize
        base_price = df_window.iloc[0]['open']
        feats = df_window[['open', 'high', 'low', 'close']].values
        feats_norm = (feats / base_price) - 1.0
        
        # Ensure exact 20
        feats_norm = feats_norm[-20:]
        
        # Create Batch (1, 20, 4) -> (1, 4, 20) handled by model
        # Input Long
        input_long = torch.FloatTensor(feats_norm).unsqueeze(0).to(self.device)
        # Input Short (Inverted)
        input_short = torch.FloatTensor(-feats_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob_long = self.model(input_long).item()
            prob_short = self.model(input_short).item()
            
        return prob_long, prob_short

    def calculate_atr(self, df):
        # Rolling 14 period
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr = pd.concat([high-low, (high-close).abs(), (low-close).abs()], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    def run_simulation(self, start_date_str: str = "2025-07-04 08:00:00", initial_balance: float = 50000.0, position_size: int = 1):
        input_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if not input_path.exists(): return
        
        logger.info(f"Simulating Continuous Smart Scan (Test Set {start_date_str})...")
        logger.info(f"Assumptions: Balance ${initial_balance:,.0f} | Size {position_size} | Scan: 5m")
        logger.info("Logic: Predicting 'Pre-Conditions' for Win/Loss. Fading high-prob Losers.")
        
        df_1m = pd.read_parquet(input_path)
        df_1m['time'] = pd.to_datetime(df_1m['time'])
        df_1m = df_1m.sort_values('time').set_index('time')
        
        if 'UTC' not in start_date_str:
             start_ts = pd.Timestamp(start_date_str).tz_localize('UTC')
        else:
             start_ts = pd.Timestamp(start_date_str)
             
        # Need context
        df_1m_test = df_1m.loc[start_ts - pd.Timedelta(hours=2):]
        
        # Calculate ATR on 15m for sizing consistency
        df_15m_atr = df_1m_test.resample('15min').agg({'high':'max', 'low':'min', 'close':'last'}).dropna()
        df_15m_atr['atr'] = self.calculate_atr(df_15m_atr)
        
        # Merge ATR to 1m - Use Reindex with FFill
        # df_1m_test is the master index
        df_1m_test['atr'] = df_15m_atr['atr'].reindex(df_1m_test.index, method='ffill')
        
        triggers = df_1m_test[df_1m_test.index >= start_ts]
        triggers = triggers.iloc[:5000] # Limit for fast feedback
        logger.info(f"Scanning {len(triggers)} 1m intervals (Sample)...")
        
        results = []
        
        for ts, row in triggers.iterrows():
            atr = row['atr']
            if pd.isna(atr): continue
            
            entry_time = ts 
            
            # Context: 20m window BEFORE entry
            context_end = entry_time
            # DataFrame slice by index is inclusive/exclusive depending on usage.
            # We want strictly [end-20m, end)
            context_start = context_end - pd.Timedelta(minutes=20)
            
            # Optimization: fast slice
            # Assuming dataframe is sorted
            # To avoid slow .loc every time, we could use rolling window iterator if we were strict.
            # But with GPU the bottleneck might be slice creation.
            # df_1m has full data.
            
            context_df = df_1m.loc[context_start:entry_time] # Includes entry_time usually?
            # Adjust to be strict < entry_time
            if len(context_df) > 0 and context_df.index[-1] == entry_time:
                 context_df = context_df.iloc[:-1]
                 
            # Predict
            # Returns (LongProb, ShortProb)
            # Batched inference would be better but keeping simple for now. 
            # 70k inferences might take 1-2 mins.
            p_long, p_short = self.get_prediction(context_df) 
            
            entry_price = row['open'] # Enter at Open of NEXT candle? 
            # Row `ts` is the candle 'Time' (Start of candle). 
            # We have closed candle `ts`. 
            # We make decision. We enter at Open of `ts + 1m`.
            # OR we assume we trade at CLOSE of `ts`.
            # Standard backtest: decision at Close, Entry at Next Open.
            # `triggers` iterates `df_1m_test`. `row` is the candle `ts`.
            # We assume this candle just closed? 
            # Actually iterrows gives the row. If we read parquet, time is usually Open Time.
            # So `row` is the candle starting at `ts`.
            # We can't know the future of this candle.
            # We must use context UP TO `ts`?
            # "train on 20 1m candles BEFORE...".
            # If we are at 10:00 (Open), previous bars are 09:59, etc.
            # So context is 09:40 to 09:59.
            # Decision made at 10:00 Open.
            # Trade entry at 10:00 Open.
            
            # My logic in `get_prediction`: uses `df_window`.
            # Context here: `context_start` to `entry_time`.
            # If entry_time is 10:00. context is 09:40-10:00.
            # `context_df[index < entry_time]` -> 09:40..09:59. Correct.
            
            # Sizing
            risk_dist = 1.35 * atr
            
            # Simulate LONG
            sl_long = entry_price - risk_dist
            tp_long = entry_price + (1.4 * risk_dist)
            res_long = self.simulate_trade(df_1m, entry_time, entry_price, sl_long, tp_long, 'LONG')
            
            # Simulate SHORT
            sl_short = entry_price + risk_dist
            tp_short = entry_price - (1.4 * risk_dist)
            res_short = self.simulate_trade(df_1m, entry_time, entry_price, sl_short, tp_short, 'SHORT')
            
            results.append({
                'time': entry_time,
                'p_long': p_long,
                'p_short': p_short,
                'pnl_long': res_long['pnl'],
                'pnl_short': res_short['pnl'],
                'outcome_long': res_long['outcome'],
                'outcome_short': res_short['outcome']
            })
            
            if len(results) % 2000 == 0:
                logger.info(f"Scanned {len(results)} intervals...")
                
        # --- ANALYSIS ---
        df = pd.DataFrame(results)
        logger.info(f"Simulation Complete. {len(df)} Intervals.")
        
        # Define Strategies
        
        # 1. Follow Strongest Signal (If > Threshold)
        # 2. Fade Strongest Signal (If < Low Threshold) - "Reversing the Loss"
        
        hybrid_pnl = 0.0
        hybrid_trades = 0
        
        follow_pnl = 0.0
        follow_trades = 0
        
        fade_losers_pnl = 0.0
        fade_losers_trades = 0
        
        fade_winners_pnl = 0.0
        fade_winners_trades = 0
        
        for _, row in df.iterrows():
            pl = row['p_long']
            ps = row['p_short']
            
            # Decision Logic
            action = None
            direction = None
            
            # Prioritize High Confidence
            # If Long > High -> Long (Follow)
            # If Short > High -> Short (Follow)
            # If Long < Low -> Short (Fade)
            # If Short < Low -> Long (Fade)
            
            # Conflict resolution? 
            # If Long > 0.6 AND Short < 0.4? -> Both say Long is good/Short is bad. Strong Long.
            
            score_long = pl - ps # High = Long good. Low = Short good.
            # Actually, pl is Prob(Long Win). ps is Prob(Short Win).
            # If pl is high, Long is good.
            # If ps is high, Short is good.
            # They shouldn't both be high (market can't win both ways usually, but can chop).
            
            # Let's simple check:
            # Max Prob drives direction
            
            best_prob = max(pl, ps)
            is_long_best = pl >= ps
            
            # FOLLOW Logic
            if best_prob > self.threshold: # e.g. 0.55
                # We would Follow
                pnl = row['pnl_long'] if is_long_best else row['pnl_short']
                follow_pnl += pnl
                follow_trades += 1
                
                # FADE WINNERS Logic (Inverse of Follow)
                fade_winners_pnl += (-1 * pnl)
                fade_winners_trades += 1
                
                # Hybrid: Follow Winners
                hybrid_pnl += pnl
                hybrid_trades += 1
                
            # FADE LOSERS Logic (Reversing the Signal)
            # If predictions are LOW?
            # worst_prob = min(pl, ps)
            # But "Signal" usually implies something the model thought was a setup.
            # If we continuous scan, everything is a signal?
            # User said: "look for the loss triggers learned... place an opposite trade"
            # This implies if Model predicts LOW Win Prob (High Loss Prob), we take opposite.
            
            # If pl < 0.40 -> Fade Long (Go Short)
            if pl < self.thresh_low:
                # Signal is "Long is Bad". Action: Go Short.
                profit = row['pnl_short']
                fade_losers_pnl += profit
                fade_losers_trades += 1
                
                # Hybrid: Fade Losers
                hybrid_pnl += profit
                hybrid_trades += 1
                
            # If ps < 0.40 -> Fade Short (Go Long)
            if ps < self.thresh_low:
                 profit = row['pnl_long']
                 fade_losers_pnl += profit
                 fade_losers_trades += 1
                 
                 hybrid_pnl += profit
                 hybrid_trades += 1

        logger.info("-" * 60)
        logger.info(f"REPORT | Balance: ${initial_balance:,.0f} | Size: {position_size}")
        logger.info("-" * 60)
        logger.info(f"1. Follow Only (Conf > {self.threshold}):")
        logger.info(f"   Trades: {follow_trades} | PnL: {follow_pnl:.2f}")
        logger.info("-" * 60)
        logger.info(f"2. Fade Losers (Conf < {self.thresh_low}):")
        logger.info(f"   Trades: {fade_losers_trades} | PnL: {fade_losers_pnl:.2f}")
        logger.info("-" * 60)
        logger.info(f"3. Hybrid (Follow Winners + Fade Losers):")
        logger.info(f"   Trades: {hybrid_trades} | PnL: {hybrid_pnl:.2f} | Final: ${initial_balance + hybrid_pnl:,.2f}")
        logger.info("-" * 60)
        logger.info(f"4. Fade Winners (Contrarian):")
        logger.info(f"   Trades: {fade_winners_trades} | PnL: {fade_winners_pnl:.2f}")
        logger.info("-" * 60)
        
        df.to_parquet(PROCESSED_DIR / "smart_reverse_trades.parquet")
        logger.info("Saved detailed results to smart_reverse_trades.parquet")
        
    def simulate_trade(self, df, entry_time, entry_price, sl, tp, direction):
        subset = df.loc[entry_time:].iloc[:2000]
        if subset.empty: return {'outcome': 'TIMEOUT', 'pnl': 0.0}
        
        times = subset.index.values
        highs = subset['high'].values
        lows = subset['low'].values
        closes = subset['close'].values
        
        if direction == 'LONG':
             mask_win = highs >= tp
             mask_loss = lows <= sl
        else:
             mask_win = lows <= tp
             mask_loss = highs >= sl
             
        idx_win = np.argmax(mask_win) if mask_win.any() else 999999
        idx_loss = np.argmax(mask_loss) if mask_loss.any() else 999999
        
        if idx_win == 999999 and idx_loss == 999999:
            outcome = 'TIMEOUT'
            pnl = (closes[-1] - entry_price) * (1 if direction == 'LONG' else -1)
        elif idx_win < idx_loss:
            outcome = 'WIN'
            pnl = (tp - entry_price) * (1 if direction == 'LONG' else -1)
        else:
            outcome = 'LOSS'
            pnl = (sl - entry_price) * (1 if direction == 'LONG' else -1)
            
        return {'outcome': outcome, 'pnl': pnl}

if __name__ == "__main__":
    # Model output is centered ~0.41. 
    # Follow > 0.415? Fade < 0.405?
    # Let's try to capture top/bottom 20%.
    s = SmartCNNStrategy(threshold=0.43) 
    s.thresh_low = 0.39 
    s.run_simulation()
```

### ./src/test_3m_strategy.py
```py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, MODELS_DIR
from train_3m_cnn import CNN_Rejection

def test_3m_strategy():
    print("Loading data...")
    df_1m = pd.read_parquet(PROCESSED_DIR / "continuous_1m.parquet")
    
    # Handle Index & Time
    if isinstance(df_1m.index, pd.DatetimeIndex) or 'time' not in df_1m.columns:
        df_1m = df_1m.reset_index()
    
    kw = [c for c in df_1m.columns if 'time' in c.lower() or 'date' in c.lower() or c == 'index']
    if kw:
        target = kw[0]
        for k in kw:
            if k == 'time': target = k
        df_1m = df_1m.rename(columns={target: 'time'})
        
    df_1m = df_1m.sort_values('time').reset_index(drop=True)
    df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
    
    # --- Split (Test on final 30%) ---
    # We use the same split logic as training to ensure no leakage
    # Ideally we'd use the explicit date, but index split is consistent if source is same
    # Wait, the miner used all data. The trainer used first 70% of TRADES.
    # The backtest should run on the last 30% of TIME (or roughly same period).
    # Let's approximate: simple index split of dataframe.
    
    split_idx = int(len(df_1m) * 0.70)
    df_test = df_1m.iloc[split_idx:].reset_index(drop=True)
    
    # --- 15m Resampling for Execution Logic ---
    print("Resampling to 15m...")
    df_15m = df_test.set_index('time').resample('15T').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    
    # Calculate 15m ATR for Stops
    df_15m['tr'] = np.maximum(
        df_15m['high'] - df_15m['low'],
        np.maximum(
            abs(df_15m['high'] - df_15m['close'].shift(1)),
            abs(df_15m['low'] - df_15m['close'].shift(1))
        )
    )
    df_15m['atr'] = df_15m['tr'].rolling(14).mean()
    
    # We need to map 15m ATR back to the decision time.
    # We iterate through 15m bars.
    
    # --- Load Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CNN_Rejection().to(device)
    model_path = MODELS_DIR / "cnn_3m_rejection.pth"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # --- Pre-calculate 15m context (Define df_sim) ---
    print("Pre-calculating 15m context...")
    df_context = df_test[['time', 'open', 'close']].copy()
    df_context['bucket_time'] = df_context['time'].dt.floor('15T')
    
    # Merge with 15m ATR and Open
    df_15m_lookup = df_15m[['time', 'open', 'atr']].rename(columns={'time': 'bucket_time', 'open': 'open_15m', 'atr': 'atr_15m'})
    
    df_sim = pd.merge(df_test, df_context[['time', 'bucket_time']], on='time')
    df_sim = pd.merge(df_sim, df_15m_lookup, on='bucket_time', how='left')
    
    # --- Batch Inference ---
    print("Preparing 30m context windows for batch inference...")
    
    vals = df_sim[['open', 'high', 'low', 'close']].values.astype(np.float32)
    
    # We need to normalize EACH window independently (standard scaler) ??
    # The training used per-window normalization. We must match it.
    # Vectorized per-window normalization is tricky without expanding memory.
    # We can use a customized Dataset/DataLoader with num_workers=0 (or higher) to fetch batches on the fly.
    
    from torch.utils.data import TensorDataset
    
    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, data_array, lookback=30):
            self.data = data_array
            self.lookback = lookback
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            if idx < self.lookback:
                # Pad
                return torch.zeros(4, self.lookback)
                
            window = self.data[idx-self.lookback : idx]
            
            # Normalize
            mean = window.mean()
            std = window.std()
            if std == 0: std = 1
            window = (window - mean) / std
            
            # (Seq, Dim) -> (Dim, Seq)
            return torch.FloatTensor(window).T
            
    inf_ds = InferenceDataset(vals)
    inf_loader = torch.utils.data.DataLoader(inf_ds, batch_size=4096, shuffle=False)
    
    print("Running Inference on GPU...")
    all_probs = []
    
    with torch.no_grad():
        for X in inf_loader:
             X = X.to(device)
             probs = model(X)
             all_probs.append(probs.cpu().numpy())
             
    all_probs = np.concatenate(all_probs).flatten()
    print("Inference Complete.")
    
    # --- Simulation Loop (1m Resolution) ---
    print(f"Starting Simulation on {len(df_sim)} 1m bars...")
    
    account_balance = 2000.0
    risk_per_trade = 75.0
    max_trades = 3
    
    open_trades = [] 
    closed_trades = []
    
    # We Loop 1m bars
    for i in range(50, len(df_sim)):
        curr_bar = df_sim.iloc[i]
        curr_time = curr_bar['time']
        
        # 1. Manage Open Trades
        remaining_trades = []
        for trade in open_trades:
            # SHORT Trade Logic
            # SL Hit if High >= Stop
            # TP Hit if Low <= TP
            
            sl_hit = (curr_bar['high'] >= trade['stop'])
            tp_hit = (curr_bar['low'] <= trade['tp'])
            
            status = 'OPEN'
            pnl = 0
            exit_price = 0
            
            if sl_hit and tp_hit:
                 # Conservative: Loss
                 status = 'LOSS'
                 pnl = -risk_per_trade
                 exit_price = trade['stop']
            elif sl_hit:
                 status = 'LOSS'
                 pnl = -risk_per_trade
                 exit_price = trade['stop']
            elif tp_hit:
                 status = 'WIN'
                 pnl = trade['reward']
                 exit_price = trade['tp']
                 
            if status != 'OPEN':
                trade['status'] = status
                trade['pnl'] = pnl
                trade['exit_time'] = curr_time
                trade['exit_price'] = exit_price
                closed_trades.append(trade)
                account_balance += pnl
            else:
                remaining_trades.append(trade)
        open_trades = remaining_trades
        
        # 2. Check for New Entry
        if len(open_trades) < max_trades:
             # Use pre-calculated probability
             prob = all_probs[i]
             
             if prob > 0.5:
                 # SIGNAL -> SHORT
                 
                 # Hypothetical 15m Candle Logic (Last 15m: i-14 to i)
                 # Wick of the window (Low for Long, High for Short)
                 start_scan = i - 14
                 if start_scan < 0: continue
                 
                 # Find Highest High in the 15m window
                 hyp_high = df_sim.iloc[start_scan : i+1]['high'].max()
                 
                 atr = curr_bar['atr_15m']
                 if pd.isna(atr): continue
                 
                 entry_price = curr_bar['close']
                 
                 # Logic: "stop .2 atr ABOVE the Wick (High)"
                 stop_price = hyp_high + 0.2 * atr
                 
                 # For Short: Stop must be ABOVE Entry.
                 if entry_price >= stop_price:
                     continue 
                     
                 dist = stop_price - entry_price
                 risk_amt = risk_per_trade
                 take_profit = entry_price - (2.2 * dist)
                 reward_amt = 2.2 * risk_amt
                 
                 new_trade = {
                    'entry_time': curr_time,
                    'entry_price': entry_price,
                    'stop': stop_price,
                    'tp': take_profit,
                    'reward': reward_amt,
                    'status': 'OPEN',
                    'direction': 'SHORT'
                 }
                 open_trades.append(new_trade)
                
    # End Simulation
    print(f"\nSimulation Complete.")
    print(f"Total Trades: {len(closed_trades)}")
    print(f"Final Balance: ${account_balance:.2f} (Start: $2000)")
    
    if len(closed_trades) > 0:
        wins = len([t for t in closed_trades if t['status'] == 'WIN'])
        wr = wins / len(closed_trades)
        print(f"Win Rate: {wr:.2%}")
        
    # Save log
    pd.DataFrame(closed_trades).to_csv("rejection_backtest_results.csv")
    print("Saved trades to rejection_backtest_results.csv")

if __name__ == "__main__":
    test_3m_strategy()
```

### ./src/train_3m_cnn.py
```py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, MODELS_DIR

# --- Dataset Definition ---
class RejectionDataset(Dataset):
    def __init__(self, trades, candles, lookback=30):
        self.trades = trades
        self.candles = candles # 1m dataframe
        self.lookback = lookback
        
        # Prepare valid indices mapping
        # We need fast access to candle data by timestamp
        # Ensure candles are sorted
        if not self.candles.index.name == 'time':
            self.candles = self.candles.set_index('time').sort_index()
            
    def __len__(self):
        return len(self.trades)
        
    def __getitem__(self, idx):
        trade = self.trades.iloc[idx]
        start_time = trade['start_time']
        target = 1.0 if trade['outcome'] == 'WIN' else 0.0
        
        # Input: 30m context BEFORE start_time
        # We want data strictly < start_time
        # Slice: [start_time - 30m, start_time)
        
        # We need integer indexing or slicing on the datetime index
        # searchsorted returns the insertion point to maintain order
        # "left" means index of the first element >= value
        
        # We want the index of 'start_time' in candles
        # If start_time exists, we take prior 30 bars.
        
        end_idx = self.candles.index.searchsorted(start_time)
        start_idx = end_idx - self.lookback
        
        if start_idx < 0:
            # Padding
            # Create zeros (Lookback, 4)
            data = np.zeros((self.lookback, 4), dtype=np.float32)
        else:
            subset = self.candles.iloc[start_idx:end_idx]
            vals = subset[['open', 'high', 'low', 'close']].values
            
            # Normalize Standard Scaler per window
            mean = vals.mean()
            std = vals.std()
            if std == 0: std = 1
            data = (vals - mean) / std
            
            # Pad if short (e.g. gaps)
            if len(data) < self.lookback:
                 p = np.zeros((self.lookback - len(data), 4), dtype=np.float32)
                 data = np.vstack([p, data])
                 
        # To Tensor (Seq, Dim) -> (Dim, Seq) for Conv1d
        # Input: (4, 30)
        t_data = torch.FloatTensor(data).transpose(0, 1)
        t_target = torch.FloatTensor([target])
        
        return t_data, t_target

# --- Model Definition ---
class CNN_Rejection(nn.Module):
    def __init__(self, input_dim=4, seq_len=30):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # 30 -> 15
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # 15 -> 7
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 7 -> 3
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (Batch, Dim, Seq)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_3m_model():
    # 1. Load Data
    print("Loading data...")
    df_1m = pd.read_parquet(PROCESSED_DIR / "continuous_1m.parquet")
    # Ensure time index for Dataset
    if 'time' in df_1m.columns:
        df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
        # df_1m = df_1m.set_index('time').sort_index() # Done in Dataset init
    
    trades = pd.read_parquet(PROCESSED_DIR / "labeled_3m_rejection.parquet")
    # Ensure trades start_time is UTC
    trades['start_time'] = pd.to_datetime(trades['start_time'], utc=True)
    
    # 2. Time Split (70% Train)
    trades = trades.sort_values('start_time')
    split_idx = int(len(trades) * 0.70)
    train_trades = trades.iloc[:split_idx]
    
    # We DO NOT touch the test set here (final 30%).
    # We can split Train into Train/Val for validation during training.
    # Let's do 80/20 split of the "Train" set for validation.
    val_split = int(len(train_trades) * 0.80)
    val_subset = train_trades.iloc[val_split:]
    train_subset = train_trades.iloc[:val_split]
    
    print(f"Total Trades: {len(trades)}")
    print(f"Training Set: {len(train_subset)} | Validation Set: {len(val_subset)}")
    print(f"Target Distribution (Train): \n{train_subset['outcome'].value_counts(normalize=True)}")
    
    # 3. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_ds = RejectionDataset(train_subset, df_1m)
    val_ds = RejectionDataset(val_subset, df_1m)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0) # workers=0 for windows compat
    val_loader = DataLoader(val_ds, batch_size=64)
    
    model = CNN_Rejection().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Loop
    epochs = 20
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (out > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item()
                preds = (out > 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
                
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODELS_DIR / "cnn_3m_rejection.pth")
            print("  [Saved Best Model]")
            
    print("\nTraining Complete.")
    print(f"Model saved to {MODELS_DIR / 'cnn_3m_rejection.pth'}")

if __name__ == "__main__":
    train_3m_model()
```

### ./src/train_models_phase2.py
```py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import copy

sys.path.append(str(Path(__file__).parent.parent))
from src.config import PROCESSED_DIR, MODELS_DIR
from src.models.variants import CNN_Classic, CNN_Wide, LSTM_Seq, Feature_MLP

# Utils
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(df):
    # Engineer features for MLP
    # RSI, ATR, Dist from MA(20), MA(50), Volatility
    df['rsi'] = compute_rsi(df['close'])
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['dist_ma20'] = (df['close'] - df['ma20']) / df['ma20']
    df['dist_ma50'] = (df['close'] - df['ma50']) / df['ma50']
    df['atr'] = (df['high'] - df['low']).rolling(14).mean() / df['close'] # Normalized ATR
    
    # Lagged returns
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    
    # Drop NaNs
    df = df.fillna(0.0)
    
    # Features list
    feats = ['rsi', 'dist_ma20', 'dist_ma50', 'atr', 'ret_1', 'ret_5', 'volume']
    # Normalize
    for f in feats:
        m = df[f].mean()
        s = df[f].std()
        if s == 0: s = 1
        df[f] = (df[f] - m) / s
        
    return df, feats

class TradeDataset(Dataset):
    def __init__(self, trades, candles, mode='classic', lookback=20, feature_cols=None):
        self.trades = trades
        self.candles = candles
        self.mode = mode
        self.lookback = lookback
        self.feature_cols = feature_cols
        
    def __len__(self):
        return len(self.trades)
        
    def __getitem__(self, idx):
        row = self.trades.iloc[idx]
        trigger_time = row['trigger_time'] # 5m trigger time
        
        # Determine Target (1 = Inverse WIN = Rejection LOSS)
        target = 1.0 if row['outcome'] == 'LOSS' else 0.0
        
        # Get Window (ending at trigger_time)
        # We need 1m candles
        # trigger_time is 5m bucket. E.g. 09:05.
        # We want data BEFORE 09:05.
        
        end_idx = self.candles.index.searchsorted(trigger_time)
        start_idx = end_idx - self.lookback
        
        if start_idx < 0:
            # Padding? or just fail safe
            start_idx = 0
            
        subset = self.candles.iloc[start_idx:end_idx]
        
        # Pad if short
        if len(subset) < self.lookback:
            # This shouldn't happen often if we filter trades
            # Create zeros
            pass 
            
        # Prepare Input Tensor
        if self.mode == 'mlp':
            # Take features from the LAST row (closest to decision time)
            feats = subset[self.feature_cols].iloc[-1].values
            return torch.FloatTensor(feats), torch.FloatTensor([target])
        else:
            # Image/Sequence (O/H/L/C)
            # Normalize window
            vals = subset[['open', 'high', 'low', 'close']].values
            mean = vals.mean()
            std = vals.std()
            if std == 0: std = 1
            vals = (vals - mean) / std
            
            return torch.FloatTensor(vals), torch.FloatTensor([target])

def train_phase2():
    # 1. Load Data
    print("Loading data...")
    df_1m = pd.read_parquet(PROCESSED_DIR / "continuous_1m.parquet")
    trades = pd.read_parquet(PROCESSED_DIR / "labeled_continuous.parquet")
    
    # 2. Features for MLP
    df_1m, fe_cols = prepare_features(df_1m)
    print(f"Engineered {len(fe_cols)} features for MLP.")
    
    # 3. Split
    trades = trades.sort_values('trigger_time')
    split = int(0.8 * len(trades))
    train_trades = trades.iloc[:split]
    val_trades = trades.iloc[split:]
    print(f"Train: {len(train_trades)}, Val: {len(val_trades)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # 4. Train Configs
    configs = [
        {'name': 'CNN_Classic', 'model': CNN_Classic(), 'mode': 'classic', 'lookback': 20},
        {'name': 'CNN_Wide',    'model': CNN_Wide(),    'mode': 'classic', 'lookback': 60},
        {'name': 'LSTM_Seq',    'model': LSTM_Seq(),    'mode': 'classic', 'lookback': 20},
        {'name': 'Feature_MLP', 'model': Feature_MLP(input_dim=len(fe_cols)), 'mode': 'mlp', 'features': fe_cols}
    ]
    
    for cfg in configs:
        print(f"\n--- Training {cfg['name']} ---")
        model = cfg['model'].to(device)
        
        # Datasets
        train_ds = TradeDataset(train_trades, df_1m, mode=cfg['mode'], lookback=cfg.get('lookback', 20), feature_cols=cfg.get('features'))
        val_ds = TradeDataset(val_trades, df_1m, mode=cfg['mode'], lookback=cfg.get('lookback', 20), feature_cols=cfg.get('features'))
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        best_acc = 0.0
        
        for epoch in range(10): # 10 epochs
            model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            # Val
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    predicted = (pred > 0.5).float()
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
            
            acc = correct / total
            print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), MODELS_DIR / f"{cfg['name']}.pth")
                
        print(f"Best Val Acc for {cfg['name']}: {best_acc:.4f}")

if __name__ == "__main__":
    train_phase2()
```

### ./src/train_predictive.py
```py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import copy

sys.path.append(str(Path(__file__).parent.parent))
from src.config import PROCESSED_DIR, MODELS_DIR

# Define Model Architecture
class CNN_Predictive(nn.Module):
    def __init__(self, input_dim=4, seq_len=20):
        super(CNN_Predictive, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (seq_len // 2), 64)
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Time, Feature) -> Permute to (Feature, Time) for Conv1d
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.relu(x)
        # No pool after first
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) # (20 -> 10)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

class PredictiveDataset(Dataset):
    def __init__(self, indices_df, full_df, lookback=20):
        self.indices = indices_df
        self.full_df = full_df
        self.lookback = lookback
        
        # Optimize access
        self.opens = full_df['open'].values.astype(np.float32)
        self.highs = full_df['high'].values.astype(np.float32)
        self.lows = full_df['low'].values.astype(np.float32)
        self.closes = full_df['close'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        row = self.indices.iloc[idx]
        target_idx = int(row['index'])
        label = float(row['label'])
        
        start_idx = target_idx - self.lookback
        
        # Features
        o = self.opens[start_idx:target_idx]
        h = self.highs[start_idx:target_idx]
        l = self.lows[start_idx:target_idx]
        c = self.closes[start_idx:target_idx]
        
        # Normalize
        # Simple Z-score relative to window mean
        block = np.stack([o, h, l, c], axis=1)
        mean = block.mean()
        std = block.std()
        if std == 0: std = 1e-6
        block = (block - mean) / std
        
        return torch.tensor(block, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train_predictive():
    print("Loading data...")
    PROCESSED_DATA_FILE = PROCESSED_DIR / "continuous_1m.parquet"
    full_df = pd.read_parquet(PROCESSED_DATA_FILE).sort_index()
    
    labels_path = PROCESSED_DIR / "labeled_predictive.parquet"
    labels_df = pd.read_parquet(labels_path)
    
    # Split
    split_idx = int(len(labels_df) * 0.8)
    train_df = labels_df.iloc[:split_idx]
    val_df = labels_df.iloc[split_idx:]
    
    train_ds = PredictiveDataset(train_df, full_df)
    val_ds = PredictiveDataset(val_df, full_df)
    
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model = CNN_Predictive().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        sum_loss = 0
        correct = 0
        total = 0
        
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.item()
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
        train_acc = correct / total
        train_loss = sum_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                pred = model(X)
                loss = criterion(pred, y)
                val_loss += loss.item()
                predicted = (pred > 0.5).float()
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)
        
        val_epoch_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_epoch_loss:.4f} Val Acc {val_acc:.4f}")
        
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), MODELS_DIR / "CNN_Predictive.pth")
            print(f"Saved Best Model (Loss {best_loss:.4f})")
            
    print("Training Complete.")

if __name__ == "__main__":
    train_predictive()
```

### ./src/train_predictive_5m.py
```py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.config import PROCESSED_DIR, MODELS_DIR

# Define Model Architecture
class CNN_Predictive(nn.Module):
    def __init__(self, input_dim=5, seq_len=20): # input=5 (o,h,l,c,atr)
        super(CNN_Predictive, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (seq_len // 2), 64)
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Time, Feature) -> (Feature, Time)
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

class PredictiveDataset(Dataset):
    def __init__(self, indices_df, features_df, lookback=20):
        self.indices = indices_df
        self.features_df = features_df
        self.lookback = lookback
        
        # Optimize access
        self.opens = features_df['open'].values.astype(np.float32)
        self.highs = features_df['high'].values.astype(np.float32)
        self.lows = features_df['low'].values.astype(np.float32)
        self.closes = features_df['close'].values.astype(np.float32)
        self.atrs = features_df['atr_15m'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        row = self.indices.iloc[idx]
        target_idx = int(row['index'])
        label = float(row['label'])
        
        start_idx = target_idx - self.lookback
        
        # Features
        o = self.opens[start_idx:target_idx]
        h = self.highs[start_idx:target_idx]
        l = self.lows[start_idx:target_idx]
        c = self.closes[start_idx:target_idx]
        a = self.atrs[start_idx:target_idx]
        
        # Normalize
        block = np.stack([o, h, l, c, a], axis=1)
        # Normalize OHLC separately from ATR?
        # Z-score local
        mean = block.mean(axis=0)
        std = block.std(axis=0)
        # Avoid div 0
        std[std == 0] = 1e-6
        block = (block - mean) / std
        
        return torch.tensor(block, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train_predictive_5m():
    print("Loading data...")
    features_path = PROCESSED_DIR / "features_5m_atr15m.parquet"
    labels_path = PROCESSED_DIR / "labeled_predictive_5m.parquet"
    
    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)
    
    # Check dataset size
    print(f"Total samples: {len(labels_df)}")
    
    # Split
    split_idx = int(len(labels_df) * 0.8)
    train_df = labels_df.iloc[:split_idx]
    val_df = labels_df.iloc[split_idx:]
    
    train_ds = PredictiveDataset(train_df, features_df)
    val_ds = PredictiveDataset(val_df, features_df)
    
    if len(train_ds) == 0:
        print("Error: Empty training set.")
        return

    # Weighted Sampler for Imbalance?
    # Positives are ~7% of data
    
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    model = CNN_Predictive(input_dim=5).to(device)
    
    # Class weight?
    # pos_weight = torch.tensor([13.0]).to(device) # ~ 10541 / 760
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    # But model output is sigmoid, so use BCELoss.
    # Let's trust standard BCELoss first, maybe the strong signal shines through.
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 15
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        sum_loss = 0
        correct = 0
        total = 0
        
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.item()
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
        train_acc = correct / total if total > 0 else 0
        train_loss = sum_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                pred = model(X)
                loss = criterion(pred, y)
                val_loss += loss.item()
                predicted = (pred > 0.5).float()
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)
        
        val_epoch_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1}: Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_epoch_loss:.4f} Val Acc {val_acc:.4f}")
        
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), MODELS_DIR / "CNN_Predictive_5m.pth")
            print(f"Saved Best Model (Loss {best_loss:.4f})")
            
    print("Training Complete.")

if __name__ == "__main__":
    train_predictive_5m()
```

### ./src/utils/logging_utils.py
```py
import logging
import sys
from pathlib import Path
from src.config import LOGS_DIR

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)
        
        # File Handler
        log_file = LOGS_DIR / "app.log"
        f_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
        
        # Ensure immediate flush for file handler (often helpful in dev)
        # f_handler.flush = lambda: super(logging.FileHandler, f_handler).flush()
        
    return logger

# Configure Root logger to capture third party logs if needed
# But for now, just ensuring our named loggers work is usually enough.
```

### ./src/utils/__init__.py
```py
```

### ./src/yfinance_loader.py
```py
"""
YFinance data loader for historical playback.
Fetches 1-minute data from Yahoo Finance and prepares it for playback simulation.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_utils import get_logger

logger = get_logger("yfinance_loader")


class YFinanceLoader:
    """Loads and manages YFinance data for historical playback."""
    
    def __init__(self, symbol: str = "ES=F"):
        """
        Initialize YFinance loader.
        
        Args:
            symbol: Yahoo Finance symbol (default: ES=F for E-mini S&P 500 futures)
        """
        self.symbol = symbol
        self._cache = {}
        
    def fetch_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, 
                   days_back: int = 7) -> pd.DataFrame:
        """
        Fetch 1-minute data from YFinance.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            days_back: Number of days to fetch if dates not specified (default: 7)
            
        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        try:
            # Determine date range
            if end_date is None:
                end_dt = datetime.now()
            else:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                
            if start_date is None:
                start_dt = end_dt - timedelta(days=days_back)
            else:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            
            cache_key = f"{self.symbol}_{start_dt.date()}_{end_dt.date()}"
            
            # Check cache
            if cache_key in self._cache:
                logger.info(f"Using cached data for {cache_key}")
                return self._cache[cache_key].copy()
            
            logger.info(f"Fetching {self.symbol} data from {start_dt.date()} to {end_dt.date()}")
            
            # Fetch data with 1-minute interval
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=start_dt, end=end_dt, interval="1m")
            
            if df.empty:
                logger.warning(f"No data fetched for {self.symbol}")
                return pd.DataFrame()
            
            # Process and rename columns
            df = df.reset_index()
            df = df.rename(columns={
                'Datetime': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only required columns
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            # Ensure time is timezone-aware
            if df['time'].dt.tz is None:
                df['time'] = pd.to_datetime(df['time'], utc=True)
            else:
                df['time'] = df['time'].dt.tz_convert('UTC')
            
            # Add date column for filtering
            df['date'] = df['time'].dt.date.astype(str)
            
            # Cache the result
            self._cache[cache_key] = df.copy()
            
            logger.info(f"Fetched {len(df)} 1-minute candles")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching YFinance data: {e}")
            return pd.DataFrame()
    
    def get_available_dates(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of available dates from dataframe.
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            List of date strings in YYYY-MM-DD format, sorted descending
        """
        if df.empty or 'date' not in df.columns:
            return []
        
        dates = df['date'].unique().tolist()
        dates.sort(reverse=True)
        return dates
    
    def filter_by_date(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        Filter dataframe to a specific date.
        
        Args:
            df: DataFrame with 'date' column
            date: Date string in YYYY-MM-DD format
            
        Returns:
            Filtered DataFrame
        """
        if df.empty or 'date' not in df.columns:
            return pd.DataFrame()
        
        return df[df['date'] == date].copy()
    
    def resample_5m(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 1-minute data to 5-minute candles.
        
        Args:
            df: DataFrame with 1-minute data
            
        Returns:
            DataFrame with 5-minute candles
        """
        if df.empty:
            return pd.DataFrame()
        
        df_copy = df.copy()
        df_copy = df_copy.set_index('time')
        
        resampled = df_copy.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        resampled = resampled.reset_index()
        resampled['date'] = resampled['time'].dt.date.astype(str)
        
        return resampled


# Global instance
_yf_loader = None

def get_yfinance_loader(symbol: str = "ES=F") -> YFinanceLoader:
    """Get or create global YFinance loader instance."""
    global _yf_loader
    if _yf_loader is None or _yf_loader.symbol != symbol:
        _yf_loader = YFinanceLoader(symbol)
    return _yf_loader
```

### ./src/__init__.py
```py
```

### ./tests/test_gold.py
```py

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import MODELS_DIR
from src.models.variants import CNN_Classic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GoldTest")

@dataclass
class Trade:
    id: int
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    status: str = "OPEN"
    pnl: float = 0.0
    exit_time: Optional[pd.Timestamp] = None

class GoldStrategy:
    def __init__(self, model, lookback=20, risk_amount=300.0, df_full=None):
        self.model = model
        self.lookback = lookback
        self.risk_amount = risk_amount
        self.df_full = df_full
        
        self.bars_5m = []
        self.current_5m_candle = None
        self.triggers = []
        self.active_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.min_atr = 2.0 # Adjusted for Gold (might need tuning, but 2.0 is safe for 5m Gold)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def calculate_atr(self):
        if len(self.bars_5m) < 15: return np.nan
        df = pd.DataFrame(self.bars_5m, columns=['time','open','high','low','close','atr'])
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr_list = []
        for i in range(len(df)):
            if i==0: tr_list.append(high[i]-low[i])
            else:
               tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
               tr_list.append(tr)
        return pd.Series(tr_list).rolling(14).mean().iloc[-1]

    def predict(self, timestamp):
        try:
            current_idx = self.df_full.index.get_loc(timestamp)
        except:
            return 0.5 
            
        start_idx = current_idx - self.lookback
        if start_idx < 0: return 0.5
        
        subset = self.df_full.iloc[start_idx:current_idx]
        vals = subset[['open','high','low','close']].values
        mean = vals.mean()
        std = vals.std()
        if std==0: std=1
        vals=(vals-mean)/std
        inp = torch.FloatTensor(vals).to(self.device).unsqueeze(0) 

        with torch.no_grad():
            prob = self.model(inp).item()
        return prob

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        timestamp = pd.Timestamp(timestamp)
        
        # 1. Update Trades
        for t in self.active_trades:
            if t.status != "OPEN": continue
            dist = abs(t.entry_price - t.stop_loss)
            size = self.risk_amount/dist if dist>0 else 0
            
            if t.direction == "LONG":
                if high_p >= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.take_profit - t.entry_price)
                    t.exit_time = timestamp
                elif low_p <= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
            else:
                if low_p <= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.entry_price - t.take_profit)
                    t.exit_time = timestamp
                elif high_p >= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
        
        active = [t for t in self.active_trades if t.status == "OPEN"]
        for t in self.active_trades:
            if t.status != "OPEN": self.closed_trades.append(t)
        self.active_trades = active
        
        # 2. Aggregation
        if self.current_5m_candle is None:
            fl = (timestamp.minute // 5) * 5
            st = timestamp.replace(minute=fl, second=0, microsecond=0)
            self.current_5m_candle = {'time':st, 'open':open_p, 'high':high_p, 'low':low_p, 'close':close_p}
        else:
            c = self.current_5m_candle
            c['high'] = max(c['high'], high_p)
            c['low'] = min(c['low'], low_p)
            c['close'] = close_p
            
        is_close = (timestamp.minute % 5 == 4)
        
        # 3. Triggers
        triggers_to_remove = []
        for cand in self.triggers:
            if (timestamp - cand['start_time']).total_seconds() > 3600:
                triggers_to_remove.append(cand)
                continue
            cand['max_high'] = max(cand['max_high'], high_p)
            cand['min_low'] = min(cand['min_low'], low_p)
            
            triggered = False
            direction = None
            stop = 0.0
            
            if cand['max_high'] >= cand['short_tgt'] and low_p <= cand['open']:
                stop = cand['max_high']
                prob = self.predict(timestamp)
                if prob > 0.5:
                    triggered = True
                    direction = "LONG"
                    risk_dist = stop - cand['open']
                    tp = cand['max_high'] 
                    sl = cand['open'] - (1.4 * risk_dist) 
            elif cand['min_low'] <= cand['long_tgt'] and high_p >= cand['open']:
                stop = cand['min_low']
                prob = self.predict(timestamp)
                if prob > 0.5:
                    triggered = True
                    direction = "SHORT"
                    risk_dist = cand['open'] - stop
                    tp = cand['min_low']
                    sl = cand['open'] + (1.4 * risk_dist)

            if triggered and len(self.active_trades) < 1:
                self.trade_counter += 1
                t = Trade(id=self.trade_counter, direction=direction, entry_time=timestamp, 
                          entry_price=cand['open'], stop_loss=sl, take_profit=tp, risk=self.risk_amount)
                self.active_trades.append(t)
                triggers_to_remove.append(cand)
        
        for t in triggers_to_remove: 
            if t in self.triggers: self.triggers.remove(t)
            
        if is_close:
            atr = self.calculate_atr()
            c = self.current_5m_candle
            self.bars_5m.append((c['time'],c['open'],c['high'],c['low'],c['close'],atr))
            # Gold ATR likely > 0.5, but min_atr set to 2.0. Let's adjust dynamically or log
            if not np.isnan(atr) and atr >= self.min_atr:
                self.triggers.append({'start_time':c['time'], 'open':c['open'], 'atr':atr, 
                                      'short_tgt':c['open']+1.5*atr, 'long_tgt':c['open']-1.5*atr, 
                                      'max_high':c['high'], 'min_low':c['low']})
            self.current_5m_candle = None


class TestGold(unittest.TestCase):
    def test_gold_logic(self):
        ticker = "MGC=F"
        # YFinance restriction: 1m data only available for last 7 days.
        logger.info(f"Downloading {ticker} (Last 5 Days)...")
        
        df = yf.download(ticker, period="5d", interval="1m", progress=False)
        if df.empty:
            logger.error("No data returned for Gold.")
            return

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        
        logger.info(f"Loaded {len(df)} candles for Gold.")
        
        # Load Model (CNN Classic - Trained on MES, Applying to Gold)
        # NOTE: WE ARE TRANSFERRING A MODEL TRAINED ON MES TO GOLD.
        # This is a bold test of generalization.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN_Classic()
        path = MODELS_DIR / "CNN_Classic.pth"
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            logger.info("Loaded CNN_Classic model (MES-trained) for Gold test.")
        else:
            logger.warning("Model weights not found! Using untrained model (random).")
            
        strategy = GoldStrategy(model, df_full=df)
        
        logger.info("Running strategy specific to Gold...")
        for ts, row in df.iterrows():
            strategy.on_bar(ts, row['open'], row['high'], row['low'], row['close'])
            
        n_trades = len(strategy.closed_trades)
        logger.info(f"Total Trades: {n_trades}")
        
        if n_trades > 0:
            wins = len([t for t in strategy.closed_trades if t.status=='WIN'])
            wr = wins / n_trades * 100
            pnl = sum([t.pnl for t in strategy.closed_trades])
            
            logger.info("=" * 40)
            logger.info(f"GOLD ({ticker}) PERFORMANCE REPORT")
            logger.info(f"Period:    Last 5 Days")
            logger.info(f"Trades:    {n_trades}")
            logger.info(f"Win Rate:  {wr:.1f}%")
            logger.info(f"Total PnL: ${pnl:.2f}")
            logger.info("=" * 40)
            
            # Save basic plot
            res_df = pd.DataFrame([{'t': t.exit_time, 'pnl': t.pnl} for t in strategy.closed_trades])
            res_df = res_df.sort_values('t')
            res_df['cum'] = res_df['pnl'].cumsum()
            
            plt.figure(figsize=(10,5))
            plt.plot(res_df['t'], res_df['cum'], marker='o')
            plt.title(f"Gold Strategy Performance | PnL: ${pnl:.0f}")
            plt.tight_layout()
            plt.savefig("gold_performance.png")
            logger.info("Saved gold_performance.png")
            
        else:
            logger.warning("No trades triggered. Check ATR thresholds or Volatility.")

if __name__ == '__main__':
    unittest.main()
```

### ./tests/test_inverse_single_position.py
```py

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SinglePosStrategy")

@dataclass
class Trade:
    id: int
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    atr: float
    status: str = "OPEN" # OPEN, WIN, LOSS
    pnl: float = 0.0
    exit_time: Optional[pd.Timestamp] = None
    
    # Deep Analysis Metrics
    max_unrealized_pnl: float = -np.inf # Peak profit during trade (MFE)
    min_unrealized_pnl: float = np.inf  # Max drawdown during trade (MAE)
    max_adverse_excursion: float = 0.0 # Absolute distance moved against
    max_favorable_excursion: float = 0.0 # Absolute distance moved in favor

class SinglePositionInverseStrategy:
    def __init__(self, risk_amount=300.0, max_active_trades=1):
        self.bars_1m = [] 
        self.bars_5m = [] 
        
        self.risk_amount = risk_amount
        self.max_active_trades = max_active_trades
        
        self.current_5m_candle = None
        
        self.triggers = [] 
        self.active_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        
        self.min_atr = 5.0
        self.lookahead = 12 
        
    def calculate_atr(self, period=14):
        if len(self.bars_5m) < period + 1:
            return np.nan
        
        df = pd.DataFrame(self.bars_5m, columns=['time', 'open', 'high', 'low', 'close', 'atr'])
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr_list = []
        for i in range(len(df)):
            if i == 0:
                tr_list.append(high[i] - low[i])
                continue
            h = high[i]
            l = low[i]
            cp = close[i-1]
            tr = max(h - l, abs(h - cp), abs(l - cp))
            tr_list.append(tr)
            
        tr_series = pd.Series(tr_list)
        atr = tr_series.rolling(window=period).mean().iloc[-1]
        return atr

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        timestamp = pd.Timestamp(timestamp)
        
        # 1. Update Active Trades (1m Check)
        for trade in self.active_trades:
            if trade.status != "OPEN": continue
            
            # Position Size Calculation
            dist = abs(trade.entry_price - trade.stop_loss)
            size = self.risk_amount / dist if dist > 0 else 0
            
            # Update MAE/MFE (Intra-candle conservative estimation)
            # Conservatively assume High happened first or Low? We don't know intra-res tick.
            # We track the EXTREMES of this candle relative to entry.
            
            if trade.direction == "LONG":
                # Current Candle Range: low_p to high_p
                # Max potential profit this bar: high_p - entry
                # Max potential loss this bar: low_p - entry
                curr_max_pnl = (high_p - trade.entry_price) * size
                curr_min_pnl = (low_p - trade.entry_price) * size
                
                trade.max_unrealized_pnl = max(trade.max_unrealized_pnl, curr_max_pnl)
                trade.min_unrealized_pnl = min(trade.min_unrealized_pnl, curr_min_pnl)
                
                trade.max_favorable_excursion = max(trade.max_favorable_excursion, high_p - trade.entry_price)
                trade.max_adverse_excursion = max(trade.max_adverse_excursion, trade.entry_price - low_p)

                # Check Exit
                if high_p >= trade.take_profit:
                    trade.status = "WIN"
                    reward_dist = trade.take_profit - trade.entry_price
                    trade.pnl = size * reward_dist
                    trade.exit_time = timestamp
                    logger.info(f"✅ Trade {trade.id} WIN | PnL: ${trade.pnl:.2f} | MAE: ${trade.min_unrealized_pnl:.2f} (Drawdown) | MFE: ${trade.max_unrealized_pnl:.2f}")
                elif low_p <= trade.stop_loss:
                    trade.status = "LOSS"
                    trade.pnl = -size * dist
                    trade.exit_time = timestamp
                    logger.info(f"❌ Trade {trade.id} LOSS | PnL: ${trade.pnl:.2f} | MAE: ${trade.min_unrealized_pnl:.2f} (Drawdown)")

            else: # SHORT
                # Short Profit: Entry - Low
                # Short Loss: High - Entry
                curr_max_pnl = (trade.entry_price - low_p) * size
                curr_min_pnl = (trade.entry_price - high_p) * size
                
                trade.max_unrealized_pnl = max(trade.max_unrealized_pnl, curr_max_pnl)
                trade.min_unrealized_pnl = min(trade.min_unrealized_pnl, curr_min_pnl)
                
                trade.max_favorable_excursion = max(trade.max_favorable_excursion, trade.entry_price - low_p)
                trade.max_adverse_excursion = max(trade.max_adverse_excursion, high_p - trade.entry_price)

                if low_p <= trade.take_profit:
                    trade.status = "WIN"
                    reward_dist = trade.entry_price - trade.take_profit
                    trade.pnl = size * reward_dist
                    trade.exit_time = timestamp
                    logger.info(f"✅ Trade {trade.id} WIN | PnL: ${trade.pnl:.2f} | MAE: ${trade.min_unrealized_pnl:.2f} (Drawdown) | MFE: ${trade.max_unrealized_pnl:.2f}")
                elif high_p >= trade.stop_loss:
                    trade.status = "LOSS"
                    trade.pnl = -size * dist
                    trade.exit_time = timestamp
                    logger.info(f"❌ Trade {trade.id} LOSS | PnL: ${trade.pnl:.2f} | MAE: ${trade.min_unrealized_pnl:.2f} (Drawdown)")

        # Cleanup closed trades
        active = []
        for t in self.active_trades:
            if t.status != "OPEN":
                self.closed_trades.append(t)
            else:
                active.append(t)
        self.active_trades = active

        # 2. Aggregation to 5m
        if self.current_5m_candle is None:
            floor_minute = (timestamp.minute // 5) * 5
            candle_start_time = timestamp.replace(minute=floor_minute, second=0, microsecond=0)
            self.current_5m_candle = {
                'time': candle_start_time,
                'open': open_p, 'high': high_p, 'low': low_p, 'close': close_p
            }
        else:
            c = self.current_5m_candle
            c['high'] = max(c['high'], high_p)
            c['low'] = min(c['low'], low_p)
            c['close'] = close_p
            
        is_candle_close = (timestamp.minute % 5 == 4)
        
        # 3. Update Triggers & Check Entries
        # IMPORTANT: "Single Position" Check
        # If we already have a trade, we DO NOT look for new entries.
        # But we DO need to update existing triggers? 
        # Actually, if we ignore new setups while in a trade, we might miss the start of the pattern.
        # But logically, "Simple" mode = "Don't enter if in trade".
        # So we can keep tracking triggers, but just gating the `entry_trade` call.
        
        candidates_to_remove = []
        for cand in self.triggers:
            if (timestamp - cand['start_time']).total_seconds() > 3600:
                candidates_to_remove.append(cand)
                continue
                
            cand['max_high'] = max(cand['max_high'], high_p)
            cand['min_low'] = min(cand['min_low'], low_p)
            
            # SHORT REJECTION -> INVERSE LONG
            if cand['max_high'] >= cand['short_tgt']:
                if low_p <= cand['open']:
                    
                    stop_loss_price = cand['max_high']
                    risk_dist = stop_loss_price - cand['open']
                    rejection_risk = risk_dist
                    rejection_tp_dist = 1.4 * rejection_risk
                    
                    # GATE: Only enter if no active trades
                    if len(self.active_trades) < self.max_active_trades:
                        self.entry_trade(
                            direction="LONG",
                            timestamp=timestamp,
                            price=cand['open'],
                            stop_loss=cand['open'] - rejection_tp_dist, 
                            take_profit=cand['max_high'], 
                            risk=rejection_tp_dist,
                            atr=cand['atr']
                        )
                    candidates_to_remove.append(cand)
                    continue
                    
            # LONG REJECTION -> INVERSE SHORT
            if cand['min_low'] <= cand['long_tgt']:
                if high_p >= cand['open']:
                    
                    stop_loss_price = cand['min_low']
                    risk_dist = cand['open'] - stop_loss_price
                    rejection_risk = risk_dist
                    rejection_tp_dist = 1.4 * rejection_risk
                    
                    if len(self.active_trades) < self.max_active_trades:
                        self.entry_trade(
                            direction="SHORT",
                            timestamp=timestamp,
                            price=cand['open'],
                            stop_loss=cand['open'] + rejection_tp_dist, 
                            take_profit=cand['min_low'], 
                            risk=rejection_tp_dist,
                            atr=cand['atr']
                        )
                    candidates_to_remove.append(cand)
                    continue
        
        for c in candidates_to_remove:
            if c in self.triggers:
                self.triggers.remove(c)
        
        # 4. Handle 5m Close
        if is_candle_close:
            current_atr = self.calculate_atr() 
            c = self.current_5m_candle
            c['atr'] = current_atr
            self.bars_5m.append((c['time'], c['open'], c['high'], c['low'], c['close'], c['atr']))
            
            if not np.isnan(current_atr) and current_atr >= self.min_atr:
                self.triggers.append({
                    'start_time': c['time'],
                    'open': c['open'],
                    'atr': current_atr,
                    'short_tgt': c['open'] + (1.5 * current_atr),
                    'long_tgt': c['open'] - (1.5 * current_atr),
                    'max_high': c['high'], 
                    'min_low': c['low']   
                })
            self.current_5m_candle = None
            
    def entry_trade(self, direction, timestamp, price, stop_loss, take_profit, risk, atr):
        self.trade_counter += 1
        t = Trade(
            id=self.trade_counter,
            direction=direction,
            entry_time=timestamp,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk=risk,
            atr=atr
        )
        self.active_trades.append(t)
        logger.info(f"🚀 ENTER {direction} at {price:.2f} | Time: {timestamp} | Risk: ${self.risk_amount:.0f}")

class TestInverseSinglePos(unittest.TestCase):
    def test_single_position_logic(self):
        # 1. Fetch Data (FIXED DATES)
        ticker = "MES=F" 
        start_date = "2025-12-01"
        end_date = "2025-12-08" # 1 week
        logger.info(f"Downloading {ticker} data from {start_date} to {end_date}...")
        
        df = yf.download(ticker, start=start_date, end=end_date, interval="1m", progress=False)
        
        if df.empty:
            logger.warning("No data returned. Test cannot proceed.")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        
        logger.info(f"Loaded {len(df)} 1m candles.")
        
        # 2. Init Strategy (Single Position)
        strategy = SinglePositionInverseStrategy(risk_amount=300.0, max_active_trades=1)
        
        # 3. Stream Data
        logger.info("Streaming data...")
        for i, row in df.iterrows():
            ts = row.get('datetime', row.get('date'))
            strategy.on_bar(
                timestamp=ts,
                open_p=float(row['open']),
                high_p=float(row['high']),
                low_p=float(row['low']),
                close_p=float(row['close'])
            )
            
        # 4. Analysis
        if len(strategy.closed_trades) == 0:
            logger.warning("No trades triggered.")
        else:
            logger.info("TRADES DETECTED!")
            
            trades_data = []
            for t in strategy.closed_trades:
                trades_data.append({
                    'id': t.id,
                    'direction': t.direction,
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'pnl': t.pnl,
                    'outcome': t.status,
                    'max_mae': t.min_unrealized_pnl, # Max Drawdown during trade
                    'max_mfe': t.max_unrealized_pnl  # Max Profit during trade
                })
            
            res_df = pd.DataFrame(trades_data)
            res_df = res_df.sort_values('exit_time')
            res_df['cum_pnl'] = res_df['pnl'].cumsum()
            res_df['drawdown'] = res_df['cum_pnl'] - res_df['cum_pnl'].cummax()
            
            total_trades = len(res_df)
            wins = len(res_df[res_df['outcome'] == 'WIN'])
            win_rate = (wins / total_trades) * 100
            total_pnl = res_df['pnl'].sum()
            max_dd = res_df['drawdown'].min()
            
            avg_mae_win = res_df[res_df['outcome'] == 'WIN']['max_mae'].mean()
            avg_mae_loss = res_df[res_df['outcome'] == 'LOSS']['max_mae'].mean()
            
            logger.info("=" * 60)
            logger.info(f"SINGLE POSITION STRATEGY REPORT ({start_date} to {end_date})")
            logger.info(f"Total Trades:      {total_trades}")
            logger.info(f"Win Rate:          {win_rate:.2f}%")
            logger.info(f"Total PnL:         ${total_pnl:.2f}")
            logger.info(f"Max Equity DD:     ${max_dd:.2f}")
            logger.info("-" * 60)
            logger.info(f"Avg Drawdown (MAE) in Winners: ${avg_mae_win:.2f}")
            logger.info(f"Avg Drawdown (MAE) in Losers:  ${avg_mae_loss:.2f}")
            logger.info("=" * 60)
            
            # Plotting
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(res_df['exit_time'], res_df['cum_pnl'], label='Cumulative PnL', color='blue')
            plt.fill_between(res_df['exit_time'], res_df['cum_pnl'], 0, alpha=0.1, color='blue')
            plt.title(f"Single Position Inverse Strategy | Total PnL: ${total_pnl:.0f} | WR: {win_rate:.1f}%")
            plt.ylabel("PnL ($)")
            plt.grid(True)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            # Scatter of MAE vs PnL
            colors = ['g' if o == 'WIN' else 'r' for o in res_df['outcome']]
            plt.scatter(res_df['max_mae'], res_df['pnl'], c=colors, alpha=0.6)
            plt.title("Trade Outcome vs. Max Drawdown (MAE) Experienced")
            plt.xlabel("Max Unrealized Loss during Trade ($)")
            plt.ylabel("Final PnL ($)")
            plt.grid(True)
            
            plt.tight_layout()
            out_file = "inverse_single_pos_metrics.png"
            plt.savefig(out_file)
            logger.info(f"Saved analysis to {out_file}")
            plt.close()

if __name__ == '__main__':
    unittest.main()
```

### ./tests/test_inverse_streaming.py
```py

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamingInverseTest")

@dataclass
class Trade:
    id: int
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    atr: float
    status: str = "OPEN" # OPEN, WIN, LOSS
    pnl: float = 0.0
    exit_time: Optional[pd.Timestamp] = None

class StreamingInverseStrategy:
    def __init__(self, risk_amount=300.0):
        self.bars_1m = [] # List of (time, o, h, l, c)
        self.bars_5m = [] # List of (time, o, h, l, c, atr)
        
        self.risk_amount = risk_amount
        
        self.current_5m_candle = None
        self.last_5m_close_time = None
        
        # Generator State
        self.triggers = [] 
        
        self.active_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        
        self.min_atr = 5.0
        self.lookahead = 12 
        
    def calculate_atr(self, period=14):
        if len(self.bars_5m) < period + 1:
            return np.nan
            
        # We need the last N candles
        df = pd.DataFrame(self.bars_5m, columns=['time', 'open', 'high', 'low', 'close', 'atr'])
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # TR calculation
        # TR = max(H-L, |H-Cp|, |L-Cp|)
        tr_list = []
        for i in range(len(df)):
            if i == 0:
                tr_list.append(high[i] - low[i])
                continue
                
            h = high[i]
            l = low[i]
            cp = close[i-1]
            
            tr = max(h - l, abs(h - cp), abs(l - cp))
            tr_list.append(tr)
            
        tr_series = pd.Series(tr_list)
        atr = tr_series.rolling(window=period).mean().iloc[-1]
        return atr

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        """
        Process a single 1m bar
        """
        timestamp = pd.Timestamp(timestamp)
        
        # 1. Update Active Trades (1m Check)
        for trade in self.active_trades:
            if trade.status != "OPEN": continue
            
            # Position Size Calculation:
            # Risk Amount = $300
            # Risk Distance = |Entry - SL|
            # Size = Risk Amount / Risk Distance
            dist = abs(trade.entry_price - trade.stop_loss)
            if dist == 0: size = 0
            else: size = self.risk_amount / dist
            
            # Check Exit
            if trade.direction == "LONG":
                # Hit TP (Extreme)?
                if high_p >= trade.take_profit:
                    trade.status = "WIN"
                    # Reward Distance = TP - Entry
                    reward_dist = trade.take_profit - trade.entry_price
                    trade.pnl = size * reward_dist
                    trade.exit_time = timestamp
                    logger.info(f"✅ Trade {trade.id} WIN (Hit TP)")
                # Hit SL (Rejection Win)?
                elif low_p <= trade.stop_loss:
                    trade.status = "LOSS"
                    # Risk Distance is 'dist'
                    trade.pnl = -size * dist
                    trade.exit_time = timestamp
                    logger.info(f"❌ Trade {trade.id} LOSS (Hit SL)")
            else: # SHORT
                if low_p <= trade.take_profit:
                    trade.status = "WIN"
                    reward_dist = trade.entry_price - trade.take_profit
                    trade.pnl = size * reward_dist
                    trade.exit_time = timestamp
                    logger.info(f"✅ Trade {trade.id} WIN (Hit TP)")
                elif high_p >= trade.stop_loss:
                    trade.status = "LOSS"
                    trade.pnl = -size * dist
                    trade.exit_time = timestamp
                    logger.info(f"❌ Trade {trade.id} LOSS (Hit SL)")

        # Cleanup closed trades
        active = []
        for t in self.active_trades:
            if t.status != "OPEN":
                self.closed_trades.append(t)
            else:
                active.append(t)
        self.active_trades = active

        # 2. Aggregation to 5m
        if self.current_5m_candle is None:
            # Start new candle
            # Align time to 5m floor
            floor_minute = (timestamp.minute // 5) * 5
            candle_start_time = timestamp.replace(minute=floor_minute, second=0, microsecond=0)
            
            self.current_5m_candle = {
                'time': candle_start_time,
                'open': open_p,
                'high': high_p,
                'low': low_p,
                'close': close_p
            }
        else:
            # Update current candle
            c = self.current_5m_candle
            c['high'] = max(c['high'], high_p)
            c['low'] = min(c['low'], low_p)
            c['close'] = close_p
            
        # Check if 5m candle is complete
        is_candle_close = (timestamp.minute % 5 == 4)
        
        # Update Triggers based on 1m price movement
        
        # Update candidates
        candidates_to_remove = []
        for cand in self.triggers:
            # Check Timeout (12 x 5m candles = 60 mins)
            # Roughly Check time difference
            if (timestamp - cand['start_time']).total_seconds() > 3600:
                candidates_to_remove.append(cand)
                continue
                
            # Update High/Low seen since start
            cand['max_high'] = max(cand['max_high'], high_p)
            cand['min_low'] = min(cand['min_low'], low_p)
            
            # Check SHORT Setup (Price goes UP to Target, then returns DOWN to Open)
            # 1. Did we hit the Up Target?
            if cand['max_high'] >= cand['short_tgt']:
                # 2. Did we Return to Open?
                if low_p <= cand['open']:
                    
                    stop_loss_price = cand['max_high']
                    risk_dist = stop_loss_price - cand['open']
                    
                    # Inverse Trade
                    # Rejection TP (our SL) is 1.4x Rejection Risk
                    rejection_risk = risk_dist # Risk of rejection trade
                    rejection_tp_dist = 1.4 * rejection_risk
                    
                    # Our Trade (Inverse LONG)
                    # Entry: Open
                    # Stop: Open - rejection_tp_dist
                    # Target: Open + risk_dist (The Extreme)
                    
                    self.entry_trade(
                        direction="LONG",
                        timestamp=timestamp,
                        price=cand['open'],
                        stop_loss=cand['open'] - rejection_tp_dist, 
                        take_profit=cand['max_high'], 
                        risk=rejection_tp_dist, # Our visual risk, used for display?
                        atr=cand['atr']
                    )
                    candidates_to_remove.append(cand) # Triggered, consume it
                    continue
                    
            # Check LONG Setup (Price goes DOWN to Target, then returns UP to Open)
            if cand['min_low'] <= cand['long_tgt']:
                if high_p >= cand['open']:
                    
                    stop_loss_price = cand['min_low']
                    risk_dist = cand['open'] - stop_loss_price
                    
                    rejection_risk = risk_dist
                    rejection_tp_dist = 1.4 * rejection_risk
                    
                    # Our Trade (Inverse SHORT)
                    # Entry: Open
                    # Stop: Open + rejection_tp_dist
                    # Target: Open - risk_dist (The Extreme)
                    
                    self.entry_trade(
                        direction="SHORT",
                        timestamp=timestamp,
                        price=cand['open'],
                        stop_loss=cand['open'] + rejection_tp_dist, 
                        take_profit=cand['min_low'], 
                        risk=rejection_tp_dist,
                        atr=cand['atr']
                    )
                    candidates_to_remove.append(cand)
                    continue
        
        # Remove matched/expired
        for c in candidates_to_remove:
            if c in self.triggers:
                self.triggers.remove(c)
        
        # 3. Handle 5m Close
        if is_candle_close:
            
            current_atr = self.calculate_atr() # Uses existing bars_5m
            
            c = self.current_5m_candle
            c['atr'] = current_atr
            
            self.bars_5m.append((
                c['time'], c['open'], c['high'], c['low'], c['close'], c['atr']
            ))
            
            # Add NEW Candidate (The candle just closed is a "Start Candle" for potential setups)
            # Only if ATR is valid
            if not np.isnan(current_atr) and current_atr >= self.min_atr:
                # Add to triggers list
                self.triggers.append({
                    'start_time': c['time'],
                    'open': c['open'],
                    'atr': current_atr,
                    'short_tgt': c['open'] + (1.5 * current_atr),
                    'long_tgt': c['open'] - (1.5 * current_atr),
                    'max_high': c['high'], # Init with own high
                    'min_low': c['low']   # Init with own low
                })
                
            # Reset current candle
            self.current_5m_candle = None
            
    def entry_trade(self, direction, timestamp, price, stop_loss, take_profit, risk, atr):
        self.trade_counter += 1
        t = Trade(
            id=self.trade_counter,
            direction=direction,
            entry_time=timestamp,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk=risk,
            atr=atr
        )
        self.active_trades.append(t)
        logger.info(f"🚀 ENTER {direction} at {price:.2f} | Time: {timestamp}")
        logger.info(f"    📝 Params: TP={take_profit:.2f} | SL={stop_loss:.2f} | RiskDist={risk:.2f} | ATR={atr:.2f}")

class TestInverseStrategy(unittest.TestCase):
    def test_streaming_logic(self):
        # 1. Fetch Data
        ticker = "MES=F" # Micro E-mini S&P 500
        logger.info(f"Downloading {ticker} data 1m...")
        df = yf.download(ticker, interval="1m", period="5d", progress=False)
        
        if df.empty:
            logger.warning("No data returned from yfinance. Test cannot proceed.")
            return

        # Flatten columns if MultiIndex (yfinance updated recently)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        # Rename columns to lower
        df.columns = [c.lower() for c in df.columns]
        
        logger.info(f"Loaded {len(df)} 1m candles.")
        
        # 2. Init Strategy
        strategy = StreamingInverseStrategy()
        
        # 3. Stream Data
        logger.info("Streaming data...")
        for i, row in df.iterrows():
            # If yfinance returns 'Datetime' or 'Date'
            ts = row.get('datetime', row.get('date'))
            strategy.on_bar(
                timestamp=ts,
                open_p=float(row['open']),
                high_p=float(row['high']),
                low_p=float(row['low']),
                close_p=float(row['close'])
            )
            
        # 4. Results
        logger.info(f"Total Trades Taken: {len(strategy.closed_trades) + len(strategy.active_trades)}")
        
        if len(strategy.closed_trades) == 0:
            logger.warning("No trades triggered in the last 5 days. Market might be low volatility or logic mismatch.")
        else:
            logger.info("TRADES DETECTED!")
            
            # Create DataFrame for Analysis
            trades_data = []
            for t in strategy.closed_trades:
                trades_data.append({
                    'id': t.id,
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'pnl': t.pnl,
                    'outcome': t.status
                })
            
            res_df = pd.DataFrame(trades_data)
            res_df = res_df.sort_values('exit_time')
            res_df['cum_pnl'] = res_df['pnl'].cumsum()
            res_df['drawdown'] = res_df['cum_pnl'] - res_df['cum_pnl'].cummax()
            
            total_trades = len(res_df)
            wins = len(res_df[res_df['outcome'] == 'WIN'])
            win_rate = (wins / total_trades) * 100
            total_pnl = res_df['pnl'].sum()
            max_drawdown = res_df['drawdown'].min()

            # Detailed Stats
            # Ensure exit_time and entry_time are datetime
            res_df['exit_time'] = pd.to_datetime(res_df['exit_time'])
            res_df['entry_time'] = pd.to_datetime(res_df['entry_time'])
            res_df['duration'] = res_df['exit_time'] - pd.to_datetime(res_df['entry_time'])
            res_df['duration_mins'] = res_df['duration'].dt.total_seconds() / 60.0
            res_df['hour'] = pd.to_datetime(res_df['entry_time']).dt.hour
            
            avg_duration = res_df['duration_mins'].mean()
            avg_win_duration = res_df[res_df['outcome'] == 'WIN']['duration_mins'].mean()
            avg_loss_duration = res_df[res_df['outcome'] == 'LOSS']['duration_mins'].mean()
            
            # Hourly Stats
            hourly_stats = res_df.groupby('hour').apply(
                lambda x: pd.Series({
                    'count': len(x),
                    'win_rate': (len(x[x['outcome'] == 'WIN']) / len(x) * 100) if len(x) > 0 else 0,
                    'pnl': x['pnl'].sum()
                })
            )
            
            # Console Report
            logger.info("=" * 40)
            logger.info(f"STRATEGY PERFORMANCE REPORT (5 Days)")
            logger.info(f"Total Trades:      {total_trades}")
            logger.info(f"Win Rate:          {win_rate:.2f}%")
            logger.info(f"Total PnL:         ${total_pnl:.2f}")
            logger.info(f"Max Drawdown:      ${max_drawdown:.2f}")
            logger.info("-" * 40)
            logger.info(f"Avg Duration:      {avg_duration:.1f} mins")
            logger.info(f"Avg Win Duration:  {avg_win_duration:.1f} mins")
            logger.info(f"Avg Loss Duration: {avg_loss_duration:.1f} mins")
            logger.info("-" * 40)
            logger.info("Hourly Performance:")
            for hour, row in hourly_stats.iterrows():
                logger.info(f"  Hour {int(hour):02d}: {int(row['count']):3d} trades | WR: {row['win_rate']:5.1f}% | PnL: ${row['pnl']:.0f}")
            logger.info("=" * 40)
            
            # Plotting
            plt.figure(figsize=(12, 8))
            
            # Subplot 1: Equity Curve
            plt.subplot(2, 1, 1)
            plt.plot(res_df['exit_time'], res_df['cum_pnl'], label='Cumulative PnL', color='green')
            plt.fill_between(res_df['exit_time'], res_df['cum_pnl'], 0, alpha=0.1, color='green')
            plt.title(f"Inverse Strategy (5 Days) | Total PnL: ${total_pnl:.0f} | Win Rate: {win_rate:.1f}%")
            plt.ylabel("PnL ($)")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            # Subplot 2: Hourly Performance
            plt.subplot(2, 1, 2)
            colors = ['g' if p > 0 else 'r' for p in hourly_stats['pnl']]
            plt.bar(hourly_stats.index, hourly_stats['pnl'], color=colors, alpha=0.7)
            plt.title("PnL by Hour of Day")
            plt.xlabel("Hour (UTC)")
            plt.ylabel("PnL ($)")
            plt.xticks(hourly_stats.index)
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            out_file = "inverse_strategy_performance.png"
            plt.savefig(out_file)
            logger.info(f"Performance plot saved to {out_file}")
            plt.close()

if __name__ == '__main__':
    unittest.main()
```

### ./tests/test_mnq.py
```py

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import MODELS_DIR
from src.models.variants import CNN_Classic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MNQTest")

@dataclass
class Trade:
    id: int
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    status: str = "OPEN"
    pnl: float = 0.0
    exit_time: Optional[pd.Timestamp] = None

class MNQStrategy:
    def __init__(self, model, lookback=20, risk_amount=300.0, df_full=None):
        self.model = model
        self.lookback = lookback
        self.risk_amount = risk_amount
        self.df_full = df_full
        
        self.bars_5m = []
        self.current_5m_candle = None
        self.triggers = []
        self.active_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.min_atr = 5.0 # MNQ is similar to MES, 5.0 is reasonable
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def calculate_atr(self):
        if len(self.bars_5m) < 15: return np.nan
        df = pd.DataFrame(self.bars_5m, columns=['time','open','high','low','close','atr'])
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr_list = []
        for i in range(len(df)):
            if i==0: tr_list.append(high[i]-low[i])
            else:
               tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
               tr_list.append(tr)
        return pd.Series(tr_list).rolling(14).mean().iloc[-1]

    def predict(self, timestamp):
        try:
            current_idx = self.df_full.index.get_loc(timestamp)
        except:
            return 0.5 
            
        start_idx = current_idx - self.lookback
        if start_idx < 0: return 0.5
        
        subset = self.df_full.iloc[start_idx:current_idx]
        vals = subset[['open','high','low','close']].values
        mean = vals.mean()
        std = vals.std()
        if std==0: std=1
        vals=(vals-mean)/std
        inp = torch.FloatTensor(vals).to(self.device).unsqueeze(0) 

        with torch.no_grad():
            prob = self.model(inp).item()
        return prob

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        timestamp = pd.Timestamp(timestamp)
        
        # 1. Update Trades
        for t in self.active_trades:
            if t.status != "OPEN": continue
            dist = abs(t.entry_price - t.stop_loss)
            size = self.risk_amount/dist if dist>0 else 0
            
            if t.direction == "LONG":
                if high_p >= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.take_profit - t.entry_price)
                    t.exit_time = timestamp
                elif low_p <= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
            else:
                if low_p <= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.entry_price - t.take_profit)
                    t.exit_time = timestamp
                elif high_p >= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
        
        active = [t for t in self.active_trades if t.status == "OPEN"]
        for t in self.active_trades:
            if t.status != "OPEN": self.closed_trades.append(t)
        self.active_trades = active
        
        # 2. Aggregation
        if self.current_5m_candle is None:
            fl = (timestamp.minute // 5) * 5
            st = timestamp.replace(minute=fl, second=0, microsecond=0)
            self.current_5m_candle = {'time':st, 'open':open_p, 'high':high_p, 'low':low_p, 'close':close_p}
        else:
            c = self.current_5m_candle
            c['high'] = max(c['high'], high_p)
            c['low'] = min(c['low'], low_p)
            c['close'] = close_p
            
        is_close = (timestamp.minute % 5 == 4)
        
        # 3. Triggers
        triggers_to_remove = []
        for cand in self.triggers:
            if (timestamp - cand['start_time']).total_seconds() > 3600:
                triggers_to_remove.append(cand)
                continue
            cand['max_high'] = max(cand['max_high'], high_p)
            cand['min_low'] = min(cand['min_low'], low_p)
            
            triggered = False
            direction = None
            stop = 0.0
            
            if cand['max_high'] >= cand['short_tgt'] and low_p <= cand['open']:
                stop = cand['max_high']
                prob = self.predict(timestamp)
                if prob > 0.5:
                    triggered = True
                    direction = "LONG"
                    risk_dist = stop - cand['open']
                    tp = cand['max_high'] 
                    sl = cand['open'] - (1.4 * risk_dist) 
            elif cand['min_low'] <= cand['long_tgt'] and high_p >= cand['open']:
                stop = cand['min_low']
                prob = self.predict(timestamp)
                if prob > 0.5:
                    triggered = True
                    direction = "SHORT"
                    risk_dist = cand['open'] - stop
                    tp = cand['min_low']
                    sl = cand['open'] + (1.4 * risk_dist)

            if triggered and len(self.active_trades) < 1:
                self.trade_counter += 1
                t = Trade(id=self.trade_counter, direction=direction, entry_time=timestamp, 
                          entry_price=cand['open'], stop_loss=sl, take_profit=tp, risk=self.risk_amount)
                self.active_trades.append(t)
                triggers_to_remove.append(cand)
        
        for t in triggers_to_remove: 
            if t in self.triggers: self.triggers.remove(t)
            
        if is_close:
            atr = self.calculate_atr()
            c = self.current_5m_candle
            self.bars_5m.append((c['time'],c['open'],c['high'],c['low'],c['close'],atr))
            if not np.isnan(atr) and atr >= self.min_atr:
                self.triggers.append({'start_time':c['time'], 'open':c['open'], 'atr':atr, 
                                      'short_tgt':c['open']+1.5*atr, 'long_tgt':c['open']-1.5*atr, 
                                      'max_high':c['high'], 'min_low':c['low']})
            self.current_5m_candle = None


class TestMNQ(unittest.TestCase):
    def test_mnq_logic(self):
        ticker = "MNQ=F"
        # YFinance restriction: 1m data only available for last 7 days.
        logger.info(f"Downloading {ticker} (Last 5 Days)...")
        
        df = yf.download(ticker, period="5d", interval="1m", progress=False)
        if df.empty:
            logger.error("No data returned for MNQ.")
            return

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        
        logger.info(f"Loaded {len(df)} candles for MNQ.")
        
        # Load Model (CNN Classic - Trained on MES, Applying to MNQ)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN_Classic()
        path = MODELS_DIR / "CNN_Classic.pth"
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            logger.info("Loaded CNN_Classic model (MES-trained) for MNQ test.")
        else:
            logger.warning("Model weights not found! Using untrained model (random).")
            
        strategy = MNQStrategy(model, df_full=df)
        
        logger.info("Running strategy specific to MNQ...")
        for ts, row in df.iterrows():
            strategy.on_bar(ts, row['open'], row['high'], row['low'], row['close'])
            
        n_trades = len(strategy.closed_trades)
        logger.info(f"Total Trades: {n_trades}")
        
        if n_trades > 0:
            wins = len([t for t in strategy.closed_trades if t.status=='WIN'])
            wr = wins / n_trades * 100
            pnl = sum([t.pnl for t in strategy.closed_trades])
            
            logger.info("=" * 40)
            logger.info(f"MNQ ({ticker}) PERFORMANCE REPORT")
            logger.info(f"Period:    Last 5 Days")
            logger.info(f"Trades:    {n_trades}")
            logger.info(f"Win Rate:  {wr:.1f}%")
            logger.info(f"Total PnL: ${pnl:.2f}")
            logger.info("=" * 40)
            
            # Save basic plot
            res_df = pd.DataFrame([{'t': t.exit_time, 'pnl': t.pnl} for t in strategy.closed_trades])
            res_df = res_df.sort_values('t')
            res_df['cum'] = res_df['pnl'].cumsum()
            
            plt.figure(figsize=(10,5))
            plt.plot(res_df['t'], res_df['cum'], marker='o', color='purple')
            plt.title(f"MNQ Strategy Performance | PnL: ${pnl:.0f}")
            plt.tight_layout()
            plt.savefig("mnq_performance.png")
            logger.info("Saved mnq_performance.png")
            
        else:
            logger.warning("No trades triggered. Check ATR thresholds or Volatility.")

if __name__ == '__main__':
    unittest.main()
```

### ./tests/test_mnq_original.py
```py

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MNQOriginalTest")

@dataclass
class Trade:
    id: int
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    status: str = "OPEN"
    pnl: float = 0.0
    exit_time: Optional[pd.Timestamp] = None

class MNQOriginalStrategy:
    def __init__(self, risk_amount=300.0, df_full=None):
        self.risk_amount = risk_amount
        self.df_full = df_full
        
        self.bars_5m = []
        self.current_5m_candle = None
        self.triggers = []
        self.active_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.min_atr = 5.0 
        
    def calculate_atr(self):
        if len(self.bars_5m) < 15: return np.nan
        df = pd.DataFrame(self.bars_5m, columns=['time','open','high','low','close','atr'])
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr_list = []
        for i in range(len(df)):
            if i==0: tr_list.append(high[i]-low[i])
            else:
               tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
               tr_list.append(tr)
        return pd.Series(tr_list).rolling(14).mean().iloc[-1]

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        timestamp = pd.Timestamp(timestamp)
        
        # 1. Update Trades
        for t in self.active_trades:
            if t.status != "OPEN": continue
            dist = abs(t.entry_price - t.stop_loss)
            size = self.risk_amount/dist if dist>0 else 0
            
            if t.direction == "LONG":
                if high_p >= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.take_profit - t.entry_price)
                    t.exit_time = timestamp
                elif low_p <= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
            else:
                if low_p <= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.entry_price - t.take_profit)
                    t.exit_time = timestamp
                elif high_p >= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
        
        active = [t for t in self.active_trades if t.status == "OPEN"]
        for t in self.active_trades:
            if t.status != "OPEN": self.closed_trades.append(t)
        self.active_trades = active
        
        # 2. Aggregation
        if self.current_5m_candle is None:
            fl = (timestamp.minute // 5) * 5
            st = timestamp.replace(minute=fl, second=0, microsecond=0)
            self.current_5m_candle = {'time':st, 'open':open_p, 'high':high_p, 'low':low_p, 'close':close_p}
        else:
            c = self.current_5m_candle
            c['high'] = max(c['high'], high_p)
            c['low'] = min(c['low'], low_p)
            c['close'] = close_p
            
        is_close = (timestamp.minute % 5 == 4)
        
        # 3. Triggers - ORIGINAL LOGIC
        # Logic: If price hits target extension and returns to open -> ENTER IN REJECTION DIRECTION.
        # i.e., Price went UP -> We go SHORT. Price went DOWN -> We go LONG.
        
        triggers_to_remove = []
        for cand in self.triggers:
            if (timestamp - cand['start_time']).total_seconds() > 3600:
                triggers_to_remove.append(cand)
                continue
            cand['max_high'] = max(cand['max_high'], high_p)
            cand['min_low'] = min(cand['min_low'], low_p)
            
            triggered = False
            direction = None
            stop = 0.0
            
            # SHORT SETUP (Rejection of High)
            if cand['max_high'] >= cand['short_tgt'] and low_p <= cand['open']:
                # The "Rejection" implies price went up and came back down.
                # ORIGINAL Trade: Sell the rejection.
                triggered = True
                direction = "SHORT"
                stop = cand['max_high'] # Stop at the wick high
                risk_dist = stop - cand['open']
                sl = stop
                tp = cand['open'] - (1.4 * risk_dist) # Target Down
            
            # LONG SETUP (Rejection of Low)
            elif cand['min_low'] <= cand['long_tgt'] and high_p >= cand['open']:
                # The "Rejection" implies price went down and came back up.
                # ORIGINAL Trade: Buy the rejection.
                triggered = True
                direction = "LONG"
                stop = cand['min_low'] # Stop at wick low
                risk_dist = cand['open'] - stop
                sl = stop
                tp = cand['open'] + (1.4 * risk_dist) # Target Up

            if triggered and len(self.active_trades) < 1:
                self.trade_counter += 1
                t = Trade(id=self.trade_counter, direction=direction, entry_time=timestamp, 
                          entry_price=close_p, stop_loss=sl, take_profit=tp, risk=self.risk_amount)
                self.active_trades.append(t)
                triggers_to_remove.append(cand)
        
        for t in triggers_to_remove: 
            if t in self.triggers: self.triggers.remove(t)
            
        if is_close:
            atr = self.calculate_atr()
            c = self.current_5m_candle
            self.bars_5m.append((c['time'],c['open'],c['high'],c['low'],c['close'],atr))
            if not np.isnan(atr) and atr >= self.min_atr:
                self.triggers.append({'start_time':c['time'], 'open':c['open'], 'atr':atr, 
                                      'short_tgt':c['open']+1.5*atr, 'long_tgt':c['open']-1.5*atr, 
                                      'max_high':c['high'], 'min_low':c['low']})
            self.current_5m_candle = None


class TestMNQOriginal(unittest.TestCase):
    def test_mnq_original_logic(self):
        ticker = "MNQ=F"
        # YFinance restriction: 1m data only available for last 7 days.
        logger.info(f"Downloading {ticker} (Last 5 Days)...")
        
        df = yf.download(ticker, period="5d", interval="1m", progress=False)
        if df.empty:
            logger.error("No data returned for MNQ.")
            return

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        
        logger.info(f"Loaded {len(df)} candles for MNQ.")
        
        strategy = MNQOriginalStrategy(risk_amount=300.0, df_full=df)
        
        logger.info("Running ORIGINAL Rejection Strategy on MNQ...")
        for ts, row in df.iterrows():
            strategy.on_bar(ts, row['open'], row['high'], row['low'], row['close'])
            
        n_trades = len(strategy.closed_trades)
        logger.info(f"Total Trades: {n_trades}")
        
        if n_trades > 0:
            wins = len([t for t in strategy.closed_trades if t.status=='WIN'])
            wr = wins / n_trades * 100
            pnl = sum([t.pnl for t in strategy.closed_trades])
            
            logger.info("=" * 40)
            logger.info(f"MNQ ORIGINAL ({ticker}) PERFORMANCE REPORT")
            logger.info(f"Period:    Last 5 Days")
            logger.info(f"Trades:    {n_trades}")
            logger.info(f"Win Rate:  {wr:.1f}%")
            logger.info(f"Total PnL: ${pnl:.2f}")
            logger.info("=" * 40)
            
            # Save basic plot
            res_df = pd.DataFrame([{'t': t.exit_time, 'pnl': t.pnl} for t in strategy.closed_trades])
            res_df = res_df.sort_values('t')
            res_df['cum'] = res_df['pnl'].cumsum()
            
            plt.figure(figsize=(10,5))
            plt.plot(res_df['t'], res_df['cum'], marker='o', color='orange')
            plt.title(f"MNQ Original Strategy Performance | PnL: ${pnl:.0f}")
            plt.tight_layout()
            plt.savefig("mnq_original_performance.png")
            logger.info("Saved mnq_original_performance.png")
            
        else:
            logger.warning("No trades triggered. Check ATR thresholds or Volatility.")

if __name__ == '__main__':
    unittest.main()
```

### ./tests/test_phase2_models.py
```py

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import MODELS_DIR
from src.models.variants import CNN_Classic, CNN_Wide, LSTM_Seq, Feature_MLP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase2Test")

# Re-use metrics logic
@dataclass
class Trade:
    id: int
    model: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    status: str = "OPEN"
    pnl: float = 0.0
    exit_time: Optional[pd.Timestamp] = None

# Feature Helper
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(df):
    df['rsi'] = compute_rsi(df['close'])
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['dist_ma20'] = (df['close'] - df['ma20']) / df['ma20']
    df['dist_ma50'] = (df['close'] - df['ma50']) / df['ma50']
    df['atr'] = (df['high'] - df['low']).rolling(14).mean() / df['close']
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    df = df.fillna(0.0)
    feats = ['rsi', 'dist_ma20', 'dist_ma50', 'atr', 'ret_1', 'ret_5', 'volume']
    for f in feats:
        m = df[f].mean()
        s = df[f].std()
        if s == 0: s=1
        df[f] = (df[f]-m)/s
    return df, feats

class ModelStrategy:
    def __init__(self, name, model, mode='classic', lookback=20, risk_amount=300.0, df_full=None):
        self.name = name
        self.model = model
        self.mode = mode
        self.lookback = lookback
        self.risk_amount = risk_amount
        self.df_full = df_full # Needed for MLP lookup
        
        self.bars_5m = []
        self.current_5m_candle = None
        self.triggers = []
        self.active_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.min_atr = 5.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def calculate_atr(self):
        if len(self.bars_5m) < 15: return np.nan
        df = pd.DataFrame(self.bars_5m, columns=['time','open','high','low','close','atr'])
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr_list = []
        for i in range(len(df)):
            if i==0: tr_list.append(high[i]-low[i])
            else:
               tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
               tr_list.append(tr)
        return pd.Series(tr_list).rolling(14).mean().iloc[-1]

    def predict(self, timestamp):
        # Prepare Input
        # Find index in full DF (1m)
        try:
            current_idx = self.df_full.index.get_loc(timestamp)
        except:
            return 0.5 # Default neutral
            
        start_idx = current_idx - self.lookback
        if start_idx < 0: return 0.5
        
        subset = self.df_full.iloc[start_idx:current_idx]
        
        if self.mode == 'mlp':
            feats = subset[['rsi', 'dist_ma20', 'dist_ma50', 'atr', 'ret_1', 'ret_5', 'volume']].iloc[-1].values
            inp = torch.FloatTensor(feats).to(self.device).unsqueeze(0)
        elif self.mode == 'hybrid': # LSTM
             vals = subset[['open','high','low','close']].values
             mean = vals.mean()
             std = vals.std()
             if std==0: std=1
             vals=(vals-mean)/std
             inp = torch.FloatTensor(vals).to(self.device).unsqueeze(0) # (1, Seq, Dim)
        else: # Classic/Wide CNN
             vals = subset[['open','high','low','close']].values
             mean = vals.mean()
             std = vals.std()
             if std==0: std=1
             vals=(vals-mean)/std
             inp = torch.FloatTensor(vals).to(self.device).unsqueeze(0) # (1, Seq, Dim)

        with torch.no_grad():
            prob = self.model(inp).item()
        return prob

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        timestamp = pd.Timestamp(self.to_ts(timestamp)) # Helper
        
        # 1. Update Trades
        for t in self.active_trades:
            if t.status != "OPEN": continue
            dist = abs(t.entry_price - t.stop_loss)
            size = self.risk_amount/dist if dist>0 else 0
            
            if t.direction == "LONG":
                if high_p >= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.take_profit - t.entry_price)
                    t.exit_time = timestamp
                elif low_p <= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
            else:
                if low_p <= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.entry_price - t.take_profit)
                    t.exit_time = timestamp
                elif high_p >= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
        
        active = [t for t in self.active_trades if t.status == "OPEN"]
        # Closed move to self.closed_trades
        for t in self.active_trades:
            if t.status != "OPEN": self.closed_trades.append(t)
        self.active_trades = active
        
        # 2. Aggregation 5m
        if self.current_5m_candle is None:
            fl = (timestamp.minute // 5) * 5
            st = timestamp.replace(minute=fl, second=0, microsecond=0)
            self.current_5m_candle = {'time':st, 'open':open_p, 'high':high_p, 'low':low_p, 'close':close_p}
        else:
            c = self.current_5m_candle
            c['high'] = max(c['high'], high_p)
            c['low'] = min(c['low'], low_p)
            c['close'] = close_p
            
        is_close = (timestamp.minute % 5 == 4)
        
        # 3. Triggers
        # Standard Rejection Logic
        triggers_to_remove = []
        for cand in self.triggers:
            if (timestamp - cand['start_time']).total_seconds() > 3600:
                triggers_to_remove.append(cand)
                continue
            cand['max_high'] = max(cand['max_high'], high_p)
            cand['min_low'] = min(cand['min_low'], low_p)
            
            triggered = False
            direction = None
            stop = 0.0
            
            if cand['max_high'] >= cand['short_tgt'] and low_p <= cand['open']:
                # Short Rejection Set
                stop = cand['max_high']
                # Check Model for INVERSE (LONG)
                prob = self.predict(timestamp)
                if prob > 0.5: # Model says Inverse Win
                    triggered = True
                    direction = "LONG"
                    # Inverse Params
                    risk_dist = stop - cand['open']
                    tp = cand['max_high'] # Extreme
                    sl = cand['open'] - (1.4 * risk_dist) # Rejection TP
            elif cand['min_low'] <= cand['long_tgt'] and high_p >= cand['open']:
                # Long Rejection Set
                stop = cand['min_low']
                prob = self.predict(timestamp)
                if prob > 0.5:
                    triggered = True
                    direction = "SHORT"
                    risk_dist = cand['open'] - stop
                    tp = cand['min_low']
                    sl = cand['open'] + (1.4 * risk_dist)

            if triggered and len(self.active_trades) < 1:
                self.trade_counter += 1
                t = Trade(id=self.trade_counter, model=self.name, direction=direction, entry_time=timestamp, 
                          entry_price=cand['open'], stop_loss=sl, take_profit=tp, risk=self.risk_amount)
                self.active_trades.append(t)
                triggers_to_remove.append(cand)
        
        for t in triggers_to_remove: 
            if t in self.triggers: self.triggers.remove(t)
            
        if is_close:
            atr = self.calculate_atr()
            c = self.current_5m_candle
            self.bars_5m.append((c['time'],c['open'],c['high'],c['low'],c['close'],atr))
            if not np.isnan(atr) and atr >= self.min_atr:
                self.triggers.append({'start_time':c['time'], 'open':c['open'], 'atr':atr, 
                                      'short_tgt':c['open']+1.5*atr, 'long_tgt':c['open']-1.5*atr, 
                                      'max_high':c['high'], 'min_low':c['low']})
            self.current_5m_candle = None

    def to_ts(self, ts):
        if hasattr(ts, 'to_pydatetime'): return ts
        return pd.Timestamp(ts)

class TestPhase2(unittest.TestCase):
    def test_compare_models(self):
        # 1. Load Data
        ticker = "MES=F"
        start = "2025-12-01"
        end = "2025-12-08"
        print(f"Downloading {ticker} from {start} to {end}...")
        df = yf.download(ticker, start=start, end=end, interval="1m", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        # Rename date/datetime
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        
        # Prepare Features (for MLP)
        df, _ = prepare_features(df)
        
        # 2. Load Models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        models = []
        
        # Baseline (No Model - Always Trade)
        # We simulate this by a "Model" that always returns 1.0
        class AlwaysYes(nn.Module):
            def forward(self, x): return torch.tensor([1.0])
        models.append(ModelStrategy("Baseline (Logic Only)", AlwaysYes(), df_full=df))
        
        # Trained Models
        model_defs = [
            ('CNN_Classic', CNN_Classic(), 'classic', 20),
            ('CNN_Wide', CNN_Wide(), 'classic', 60),
            ('LSTM_Seq', LSTM_Seq(), 'hybrid', 20),
            ('Feature_MLP', Feature_MLP(12), 'mlp', 20) # 12 dims? verify feat len
        ]
        # Recalc feat len
        # feats = ['rsi', 'dist_ma20', 'dist_ma50', 'atr', 'ret_1', 'ret_5', 'volume'] -> 7 dims
        
        for name, net, mode, lb in model_defs:
            path = MODELS_DIR / f"{name}.pth"
            if path.exists():
                # Adjust input dim for MLP if needed
                if name == 'Feature_MLP': net = Feature_MLP(7)
                
                net.load_state_dict(torch.load(path, map_location=device))
                net.to(device)
                net.eval()
                models.append(ModelStrategy(name, net, mode=mode, lookback=lb, df_full=df))
            else:
                print(f"Warning: {path} not found.")

        # 3. Run Comparisons
        results = []
        
        for strat in models:
            print(f"Running {strat.name}...")
            # Reset
            strat.active_trades = []
            strat.closed_trades = []
            strat.triggers = []
            strat.bars_5m = []
            strat.current_5m_candle = None
            
            for ts, row in df.iterrows():
                strat.on_bar(ts, row['open'], row['high'], row['low'], row['close'])
            
            # Collect Stats
            n_trades = len(strat.closed_trades)
            if n_trades > 0:
                wins = len([t for t in strat.closed_trades if t.status=='WIN'])
                wr = wins / n_trades * 100
                pnl = sum([t.pnl for t in strat.closed_trades])
                
                results.append({
                    'Model': strat.name,
                    'Trades': n_trades,
                    'WinRate': wr,
                    'PnL': pnl
                })
            else:
                results.append({'Model': strat.name, 'Trades': 0, 'WinRate': 0, 'PnL': 0})
        
        # 4. Report
        res_df = pd.DataFrame(results).sort_values('PnL', ascending=False)
        print("\n=== PHASE 2 MODEL COMPARISON (Single Position, Fixed Dates) ===")
        print(res_df.to_string(index=False))
        
        # Plot
        if not res_df.empty:
            plt.figure(figsize=(10,6))
            plt.bar(res_df['Model'], res_df['PnL'], color=['green' if x>0 else 'red' for x in res_df['PnL']])
            plt.title("Model PnL Comparison (Out of Sample)")
            plt.ylabel("Total PnL ($)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("phase2_comparison.png")
            print("Saved phase2_comparison.png")

if __name__ == '__main__':
    unittest.main()
```

### ./tests/test_predictive.py
```py

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import MODELS_DIR
from src.train_predictive import CNN_Predictive

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictiveTest")

@dataclass
class LimitOrder:
    id: int
    direction: str # "SELL" (Short) or "BUY" (Long)
    limit_price: float
    stop_loss: float
    take_profit: float
    created_time: pd.Timestamp
    expiry_time: pd.Timestamp

@dataclass
class Trade:
    id: int
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    status: str = "OPEN"
    pnl: float = 0.0
    exit_time: Optional[pd.Timestamp] = None

class PredictiveStrategy:
    def __init__(self, model, risk_amount=300.0, df_full=None):
        self.model = model
        self.risk_amount = risk_amount
        self.df_full = df_full
        
        self.bars_5m = [] # Used for ATR calc
        self.current_5m_candle = None
        self.active_limits = []
        self.active_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lookback = 20
        self.threshold = 0.6 # tunable
        
    def calculate_atr(self):
        if len(self.bars_5m) < 15: return np.nan
        df = pd.DataFrame(self.bars_5m, columns=['time','open','high','low','close','atr'])
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr_list = []
        for i in range(len(df)):
            if i==0: tr_list.append(high[i]-low[i])
            else:
               tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
               tr_list.append(tr)
        return pd.Series(tr_list).rolling(14).mean().iloc[-1]

    def predict(self, timestamp):
        try:
            current_idx = self.df_full.index.get_loc(timestamp)
        except:
            return 0.5 
        start_idx = current_idx - self.lookback
        if start_idx < 0: return 0.5
        
        subset = self.df_full.iloc[start_idx:current_idx]
        vals = subset[['open','high','low','close']].values
        # Normalize same as training
        mean = vals.mean()
        std = vals.std()
        if std==0: std=1e-6
        vals=(vals-mean)/std
        inp = torch.FloatTensor(vals).to(self.device).unsqueeze(0) 

        with torch.no_grad():
            prob = self.model(inp).item()
        return prob

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        timestamp = pd.Timestamp(timestamp)
        
        # 1. Check Limits (First Check)
        # Using High/Low of THIS bar to check fills for limits created BEFORE this bar
        # In reality, we shouldn't fill limits created ON this bar (Lookahead). 
        # But we create limits AFTER Processing this bar, so they apply to NEXT bar.
        # So here we check limits from Previous bars.
        
        limits_to_remove = []
        for order in self.active_limits:
            if timestamp > order.expiry_time:
                limits_to_remove.append(order)
                continue
            
            filled = False
            if order.direction == "SELL":
                # Limit Sell filled if Price touches Limit
                if high_p >= order.limit_price:
                    # Fill!
                    # Check Assumption: Did Price gap over? 
                    # If Open > Limit, we fill at Open (Better Price).
                    # If Open < Limit and High > Limit, we fill at Limit.
                    fill_px = max(open_p, order.limit_price)
                    filled = True
                    
            elif order.direction == "BUY":
                if low_p <= order.limit_price:
                    fill_px = min(open_p, order.limit_price)
                    filled = True
            
            if filled:
                # Create Trade
                self.trade_counter += 1
                dist = abs(fill_px - order.stop_loss)
                size = self.risk_amount/dist if dist>0 else 0
                
                t = Trade(id=self.trade_counter, direction=("SHORT" if order.direction=="SELL" else "LONG"),
                          entry_time=timestamp, entry_price=fill_px, stop_loss=order.stop_loss, 
                          take_profit=order.take_profit, risk=self.risk_amount)
                self.active_trades.append(t)
                limits_to_remove.append(order)
                # OCO: If we support OCO we would remove the pair. For now assume independent.
        
        for l in limits_to_remove:
            if l in self.active_limits: self.active_limits.remove(l)

        # 2. Update Trades (Stop/Target Check)
        for t in self.active_trades:
            if t.status != "OPEN": continue
            dist = abs(t.entry_price - t.stop_loss)
            size = self.risk_amount/dist if dist>0 else 0
            
            if t.direction == "LONG":
                if high_p >= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.take_profit - t.entry_price)
                    t.exit_time = timestamp
                elif low_p <= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
            else:
                if low_p <= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.entry_price - t.take_profit)
                    t.exit_time = timestamp
                elif high_p >= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp

        active = [t for t in self.active_trades if t.status == "OPEN"]
        for t in self.active_trades:
            if t.status != "OPEN": self.closed_trades.append(t)
        self.active_trades = active
        
        # 3. New Orders (Predictive)
        # 5m Aggregation for ATR
        if self.current_5m_candle is None:
            fl = (timestamp.minute // 5) * 5
            st = timestamp.replace(minute=fl, second=0, microsecond=0)
            self.current_5m_candle = {'time':st, 'open':open_p, 'high':high_p, 'low':low_p, 'close':close_p}
        else:
            c = self.current_5m_candle
            c['high'] = max(c['high'], high_p)
            c['low'] = min(c['low'], low_p)
            c['close'] = close_p
            
        is_close = (timestamp.minute % 5 == 4)
        
        if is_close:
            atr = self.calculate_atr()
            c = self.current_5m_candle
            self.bars_5m.append((c['time'],c['open'],c['high'],c['low'],c['close'],atr))
            self.current_5m_candle = None
            
            # PREDICTION LOGIC (At 5m Close, or every 1m?)
            # Model trained on 1m windows. Let's predict every 1m? 
            # Ideally align with training. Training used sliding window on 1m.
            # So calculating ATR on 5m but predicting on 1m is mixed. 
            # Let's use 5m ATR for target distance, but Predict every 1m?
            # For simplicity, let's predict every 1m using the last KNOWN 5m ATR.
            
            # Predict
            prob = self.predict(timestamp)
            if prob > self.threshold and not np.isnan(atr) and atr > 0:
                # Signal!
                # Place OCO Limits
                expiry = timestamp + pd.Timedelta(minutes=15)
                
                # Sell Limit
                sell_limit = close_p + (1.5 * atr)
                # SL/TP for Sell
                # Target: Revert to current price (close_p)
                # Stop: Another 1.0 ATR higher
                sell_tp = close_p 
                sell_sl = sell_limit + (1.0 * atr)
                
                self.active_limits.append(LimitOrder(id=len(self.active_limits), direction="SELL",
                                                     limit_price=sell_limit, stop_loss=sell_sl, take_profit=sell_tp,
                                                     created_time=timestamp, expiry_time=expiry))
                                                     
                # Buy Limit
                buy_limit = close_p - (1.5 * atr)
                buy_tp = close_p
                buy_sl = buy_limit - (1.0 * atr)
                
                self.active_limits.append(LimitOrder(id=len(self.active_limits), direction="BUY",
                                                     limit_price=buy_limit, stop_loss=buy_sl, take_profit=buy_tp,
                                                     created_time=timestamp, expiry_time=expiry))

class TestPredictive(unittest.TestCase):
    def test_predictive_mnq(self):
        ticker = "MES=F"
        logger.info(f"Downloading {ticker} (Last 5 Days)...")
        df = yf.download(ticker, period="5d", interval="1m", progress=False)
        if df.empty: return

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        
        # Load Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN_Predictive().to(device)
        path = MODELS_DIR / "CNN_Predictive.pth"
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
        else:
            logger.warning("No model found!")

        strategy = PredictiveStrategy(model, df_full=df)
        
        logger.info("Running Predictive Limit Strategy...")
        for ts, row in df.iterrows():
            strategy.on_bar(ts, row['open'], row['high'], row['low'], row['close'])
            
        n_trades = len(strategy.closed_trades)
        logger.info(f"Total Trades: {n_trades}")
        
        if n_trades > 0:
            wins = len([t for t in strategy.closed_trades if t.status=='WIN'])
            wr = wins / n_trades * 100
            pnl = sum([t.pnl for t in strategy.closed_trades])
            
            logger.info("=" * 40)
            logger.info(f"PREDICTIVE MNQ RESULT")
            logger.info(f"Trades:    {n_trades}")
            logger.info(f"Win Rate:  {wr:.1f}%")
            logger.info(f"Total PnL: ${pnl:.2f}")
            logger.info("=" * 40)
            
            # Plot
            res_df = pd.DataFrame([{'t': t.exit_time, 'pnl': t.pnl} for t in strategy.closed_trades])
            res_df = res_df.sort_values('t')
            res_df['cum'] = res_df['pnl'].cumsum()
            
            plt.figure(figsize=(10,5))
            plt.plot(res_df['t'], res_df['cum'], marker='o', color='blue')
            plt.title(f"Predictive Strategy Performance | PnL: ${pnl:.0f}")
            plt.tight_layout()
            plt.savefig("predictive_performance.png")
            logger.info("Saved predictive_performance.png")

if __name__ == '__main__':
    unittest.main()
```

### ./tests/test_predictive_60d.py
```py

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import MODELS_DIR
from src.train_predictive_5m import CNN_Predictive

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictiveTest60d")

@dataclass
class LimitOrder:
    id: int
    direction: str # "SELL" or "BUY"
    limit_price: float
    stop_loss: float
    take_profit: float
    created_time: pd.Timestamp
    expiry_time: pd.Timestamp

@dataclass
class Trade:
    id: int
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk: float
    status: str = "OPEN"
    pnl: float = 0.0
    exit_time: Optional[pd.Timestamp] = None

class PredictiveStrategy60d:
    def __init__(self, model, risk_amount=300.0, df_full=None):
        self.model = model
        self.risk_amount = risk_amount
        self.df_full = df_full
        
        # We need ATR logic. 
        # Calculate ATR on 15m re-sampled version of full_df
        self.df_15m = df_full.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
        high = self.df_15m['high']
        low = self.df_15m['low']
        close = self.df_15m['close']
        tr_list = []
        for i in range(len(self.df_15m)):
             if i==0: tr_list.append(high.iloc[i]-low.iloc[i])
             else:
                tr = max(high.iloc[i]-low.iloc[i], abs(high.iloc[i]-close.iloc[i-1]), abs(low.iloc[i]-close.iloc[i-1]))
                tr_list.append(tr)
        self.atr_series = pd.Series(tr_list, index=self.df_15m.index).rolling(14).mean().shift(1)
        # Shift 1 means 10:15 row gets 10:00 ATR. Accessing it at 10:15 (start of bar?)
        # Wait. In test:
        # 10:05 5m.
        # df_full.at[10:05, 'atr']
        # ffill from 10:00 label.
        # If 10:00 label in 15m is "10:00 start", then shift(1) means it comes from 09:45.
        # Is that too conservative? 
        # 10:00 15m candle ends 10:15.
        # At 10:05, the 10:00 ATR is NOT known (candle open).
        # We only know 09:45 ATR.
        # So YES, we must rely on 09:45 ATR for 10:00, 10:05, 10:10.
        # If `label='left'`, 15m labels are `09:45`, `10:00`.
        # At 10:05 we ffill from `10:00`. The row `10:00` must contain `09:45` data.
        # So `shift(1)` is correct. input[10:00] = ATR[09:45].
        
        # Merge ATR back to 5m DF for easy lookup?
        # ffill to propagate last known 15m ATR to 5m bars
        self.df_full['atr_15m_lookup'] = self.atr_series.reindex(df_full.index, method='ffill')
        
        self.active_limits = []
        self.active_trades = []
        self.closed_trades = []
        self.trade_counter = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lookback = 20
        self.threshold = 0.15 # Adjusted for imbalance (base rate ~0.07) 
        
    def predict(self, timestamp):
        try:
            current_idx = self.df_full.index.get_loc(timestamp)
        except:
            return 0.5 
            
        start_idx = current_idx - self.lookback
        if start_idx < 0: return 0.5
        
        subset = self.df_full.iloc[start_idx:current_idx] 
        # Features: o,h,l,c, atr_15m
        
        # We need the 15m ATR associated with each past bar for the Feature Block?
        # In training we input (o,h,l,c, atr). 
        # Yes, we added 'atr_15m_lookup' to df_full.
        
        block = subset[['open','high','low','close','atr_15m_lookup']].values
        
        # Normalize
        mean = block.mean(axis=0)
        std = block.std(axis=0)
        std[std==0] = 1e-6
        block = (block - mean) / std
        
        inp = torch.FloatTensor(block).to(self.device).unsqueeze(0) 

        with torch.no_grad():
            prob = self.model(inp).item()
        return prob

    def on_bar(self, timestamp, open_p, high_p, low_p, close_p):
        timestamp = pd.Timestamp(timestamp)
        
        # 1. Check Limits (First Check)
        limits_to_remove = []
        for order in self.active_limits:
            if timestamp > order.expiry_time:
                limits_to_remove.append(order)
                continue
            
            filled = False
            fill_px = 0.0
            
            if order.direction == "SELL":
                # Check Fill
                if high_p >= order.limit_price:
                    # Fill
                    fill_px = max(open_p, order.limit_price)
                    filled = True
            elif order.direction == "BUY":
                if low_p <= order.limit_price:
                    fill_px = min(open_p, order.limit_price)
                    filled = True
            
            if filled:
                self.trade_counter += 1
                dist = abs(fill_px - order.stop_loss)
                size = self.risk_amount/dist if dist>0 else 0
                
                t = Trade(id=self.trade_counter, direction=("SHORT" if order.direction=="SELL" else "LONG"),
                          entry_time=timestamp, entry_price=fill_px, stop_loss=order.stop_loss, 
                          take_profit=order.take_profit, risk=self.risk_amount)
                self.active_trades.append(t)
                limits_to_remove.append(order)
        
        for l in limits_to_remove:
            if l in self.active_limits: self.active_limits.remove(l)

        # 2. Update Trades
        for t in self.active_trades:
            if t.status != "OPEN": continue
            dist = abs(t.entry_price - t.stop_loss)
            size = self.risk_amount/dist if dist>0 else 0
            
            if t.direction == "LONG":
                if high_p >= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.take_profit - t.entry_price)
                    t.exit_time = timestamp
                elif low_p <= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp
            else:
                if low_p <= t.take_profit:
                    t.status = "WIN"
                    t.pnl = size * (t.entry_price - t.take_profit)
                    t.exit_time = timestamp
                elif high_p >= t.stop_loss:
                    t.status = "LOSS"
                    t.pnl = -size * dist
                    t.exit_time = timestamp

        active = [t for t in self.active_trades if t.status == "OPEN"]
        for t in self.active_trades:
            if t.status != "OPEN": self.closed_trades.append(t)
        self.active_trades = active
        
        # 3. New Orders (Predictive)
        # Use Lookahead ATR? No, use current known ATR.
        try:
            atr = self.df_full.at[timestamp, 'atr_15m_lookup']
        except: 
            atr = np.nan
            
        if np.isnan(atr) or atr <= 0: return # No ATR context yet
        
        prob = self.predict(timestamp)
        if self.trade_counter == 0 and np.random.rand() < 0.01:
             logger.info(f"Sample Prob: {prob:.4f} | ATR: {atr:.2f}")

        if prob > self.threshold:
            # Place OCO Limits
            # Valid for 15 minutes (3 x 5m bars)
            expiry = timestamp + pd.Timedelta(minutes=15)
            
            limit_dist = 1.5 * atr
            stop_dist = 0.5 * atr # Tighter stop? Or 1.0? 
            # In Phase 7 we used 1.5 Limit distance.
            # Stop was 1.0 ATR beyond.
            
            # Sell Limit
            sell_limit = close_p + limit_dist
            sell_tp = close_p 
            sell_sl = sell_limit + (1.0 * atr)
            
            self.active_limits.append(LimitOrder(id=len(self.active_limits), direction="SELL",
                                                 limit_price=sell_limit, stop_loss=sell_sl, take_profit=sell_tp,
                                                 created_time=timestamp, expiry_time=expiry))
                                                 
            # Buy Limit
            buy_limit = close_p - limit_dist
            buy_tp = close_p
            buy_sl = buy_limit - (1.0 * atr)
            
            self.active_limits.append(LimitOrder(id=len(self.active_limits), direction="BUY",
                                                 limit_price=buy_limit, stop_loss=buy_sl, take_profit=buy_tp,
                                                 created_time=timestamp, expiry_time=expiry))

class TestPredictive60d(unittest.TestCase):
    def test_mes_60d(self):
        ticker = "MES=F"
        logger.info(f"Downloading {ticker} (Last 60 Days, 5m)...")
        # 59d to be safe
        df = yf.download(ticker, period="59d", interval="5m", progress=False)
        if df.empty: 
            logger.error("No data.")
            return

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'datetime' in df.columns: df.rename(columns={'datetime':'time'}, inplace=True)
        elif 'date' in df.columns: df.rename(columns={'date':'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        
        # Load Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN_Predictive(input_dim=5).to(device)
        path = MODELS_DIR / "CNN_Predictive_5m.pth"
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
        else:
            logger.error("No model found!")
            return

        strategy = PredictiveStrategy60d(model, df_full=df)
        
        logger.info("Running 60-Day Simulation (5m)...")
        for ts, row in df.iterrows():
            strategy.on_bar(ts, row['open'], row['high'], row['low'], row['close'])
            
        n_trades = len(strategy.closed_trades)
        
        if n_trades > 0:
            wins = len([t for t in strategy.closed_trades if t.status=='WIN'])
            wr = wins / n_trades * 100
            pnl = sum([t.pnl for t in strategy.closed_trades])
            
            logger.info("=" * 40)
            logger.info(f"MES 60-DAY RESULT (5m/15m)")
            logger.info(f"Trades:    {n_trades}")
            logger.info(f"Win Rate:  {wr:.1f}%")
            logger.info(f"Total PnL: ${pnl:.2f}")
            logger.info("=" * 40)
            
            # Plot
            res_df = pd.DataFrame([{'t': t.exit_time, 'pnl': t.pnl} for t in strategy.closed_trades])
            res_df = res_df.sort_values('t')
            res_df['cum'] = res_df['pnl'].cumsum()
            
            plt.figure(figsize=(10,5))
            plt.plot(res_df['t'], res_df['cum'], marker='o', color='purple')
            plt.title(f"MES 60-Day Performance (Predictive 15m) | PnL: ${pnl:.0f}", fontsize=12)
            plt.xlabel("Date")
            plt.ylabel("PnL ($)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("mes_60d_performance.png")
            logger.info("Saved mes_60d_performance.png")
        else:
            logger.warning("No trades executed.")

if __name__ == '__main__':
    unittest.main()
```

### ./tests/test_smoke.py
```py

import requests
import json
import sys
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def log(msg, status="INFO"):
    print(f"[{status}] {msg}")

def test_health():
    try:
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code == 200 and resp.json()['status'] == 'ok':
            log("Health Check Passed", "PASS")
            return True
        else:
            log(f"Health Check Failed: {resp.text}", "FAIL")
            return False
    except Exception as e:
        log(f"Health Check Exception: {e}", "FAIL")
        return False

def test_dates():
    try:
        resp = requests.get(f"{BASE_URL}/api/dates")
        dates = resp.json()
        if isinstance(dates, list) and len(dates) > 0:
            log(f"Dates Endpoint Passed (Found {len(dates)} dates)", "PASS")
            return dates[0] # Return newest date for other tests
        else:
            log(f"Dates Endpoint returned empty or invalid: {dates}", "WARN")
            return None
    except Exception as e:
        log(f"Dates Endpoint Exception: {e}", "FAIL")
        return None

def test_generate_session(date_str):
    try:
        payload = {
            "day_of_week": 0,
            "session_type": "RTH",
            "start_price": 5800.0,
            "date": date_str,
            "timeframe": "5m"
        }
        resp = requests.post(f"{BASE_URL}/api/generate/session", json=payload)
        if resp.status_code == 200:
            data = resp.json()['data']
            if len(data) > 0:
                log(f"Single Session Generation Passed ({len(data)} bars)", "PASS")
                return True
            else:
                log("Single Session Generation returned empty data", "FAIL")
                return False
        else:
            log(f"Single Session Generation Failed: {resp.text}", "FAIL")
            return False
    except Exception as e:
        log(f"Single Session Gen Exception: {e}", "FAIL")
        return False

def test_generate_multiday_weekend_crossover():
    # Test 5 days starting from a Friday to ensure it crosses weekend
    # 2024-01-05 is a Friday
    start_date = "2024-01-05" 
    try:
        payload = {
            "num_days": 5,
            "session_type": "RTH",
            "initial_price": 5800.0,
            "start_date": start_date,
            "timeframe": "15m"
        }
        resp = requests.post(f"{BASE_URL}/api/generate/multi-day", json=payload)
        if resp.status_code == 200:
            data = resp.json()['data']
            if len(data) > 0:
                # Check for distinct synthetic days
                days = set(d.get('synthetic_day') for d in data)
                log(f"Multi-Day Generation Passed ({len(data)} bars, {len(days)} distinct trading days)", "PASS")
                return True
            else:
                log("Multi-Day Generation returned empty data", "FAIL")
                return False
        else:
            log(f"Multi-Day Generation Failed: {resp.status_code} {resp.text}", "FAIL")
            return False
    except Exception as e:
        log(f"Multi-Day Gen Exception: {e}", "FAIL")
        return False

def run_all():
    log("Starting Smoke Tests...")
    
    if not test_health():
        sys.exit(1)
        
    latest_date = test_dates()
    
    # Use strict date for session if available, else omit
    test_generate_session(latest_date)
    
    test_generate_multiday_weekend_crossover()
    
    log("Smoke Tests Complete.")

if __name__ == "__main__":
    run_all()
```
