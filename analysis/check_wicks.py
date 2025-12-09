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
