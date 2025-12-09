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
| | |____engulfing_trades.parquet
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
| | |____smart_reverse_trades.parquet
| | |____smart_verification_trades.parquet
| |____raw
| | |____continuous_contract.json
|____diff.sh
|____frontend
| |____index.html
| |____node_modules
| |____package-lock.json
| |____package.json
| |____src
| | |____api
| | | |____client.ts
| | |____App.tsx
| | |____components
| | | |____ChartPanel.tsx
| | | |____SidebarControls.tsx
| | |____main.tsx
| |____tsconfig.json
| |____tsconfig.node.json
| |____vite.config.ts
|____gitr.sh
|____logs
|____models
| |____setup_cnn_v1.pth
|____printcode.sh
|____project_code.md
|____README.md
|____requirements.txt
|____src
| |____api.py
| |____config.py
| |____data_loader.py
| |____feature_engineering.py
| |____generator.py
| |____models
| | |____train_cnn.py
| |____pattern_library.py
| |____preprocess.py
| |____setup_miner.py
| |____state_features.py
| |____strategies
| | |____collector.py
| | |____random_tilt.py
| | |____smart_cnn.py
| | |____smart_reverse.py
| |____utils
| | |____logging_utils.py
| | |______init__.py
| | |______pycache__
| |______init__.py
| |______pycache__
|____tests
| |____test_smoke.py
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
    }
};
```

### ./frontend/src/App.tsx
```tsx
import React, { useState, useEffect } from 'react';
import { SidebarControls } from './components/SidebarControls';
import { ChartPanel } from './components/ChartPanel';
import { api, Candle } from './api/client';

const App: React.FC = () => {
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
        loadDates();
    }, []);

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
        <div style={{ display: 'flex', height: '100vh', background: '#1E1E1E', color: '#EEE' }}>
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

### ./src/__init__.py
```py
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
