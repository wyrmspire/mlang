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
