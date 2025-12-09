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
