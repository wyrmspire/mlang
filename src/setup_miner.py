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
