import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
from src.config import ONE_MIN_PARQUET_DIR, PATTERNS_DIR, LOCAL_TZ
from src.utils.logging_utils import get_logger

logger = get_logger("generator")

class PatternGenerator:
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
                for m in self.cluster_meta:
                    key = (m['session_type'], m['day_of_week'], m['hour_bucket'])
                    self.meta_map[key] = m
        
        # Load raw data for stitching
        # Ideally, we should have random access. For now, load all into memory (it's small enough for < 1GB local patterns)
        raw_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
        if raw_path.exists():
            self.raw_1min = pd.read_parquet(raw_path)
            self.raw_1min.set_index('time', inplace=True)
            self.raw_1min.sort_index(inplace=True)
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
                key = (session_type, day_of_week, h_bucket)
                if key not in self.meta_map:
                    continue
                    
                meta = self.meta_map[key]
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
                
                total_w = sum(weights)
                if total_w == 0: weights = [1]*len(weights)
                
                chosen_cluster = random.choices(clusters, weights=weights, k=1)[0]
                
                candidates = self.patterns_df[
                    (self.patterns_df['session_type'] == session_type) &
                    (self.patterns_df['day_of_week'] == day_of_week) &
                    (self.patterns_df['hour_bucket'] == h_bucket) &
                    (self.patterns_df['cluster_id'] == chosen_cluster)
                ]
                
                if candidates.empty: continue
                
                sample = candidates.sample(1).iloc[0]
                hist_start_time = sample['start_time']
                
                bars_df = self._get_historical_hour(hist_start_time)
                if bars_df.empty: continue
                
                # Stitch
                scale_factor = current_price / bars_df.iloc[0]['open']
                
                scaled_open = bars_df['open'] * scale_factor
                scaled_high = bars_df['high'] * scale_factor
                scaled_low = bars_df['low'] * scale_factor
                scaled_close = bars_df['close'] * scale_factor
                
                n = len(bars_df)
                times = [current_time_cursor + pd.Timedelta(minutes=i) for i in range(n)]
                
                for i in range(n):
                    bar = {
                        'time': times[i],
                        'open': scaled_open.iloc[i],
                        'high': scaled_high.iloc[i],
                        'low': scaled_low.iloc[i],
                        'close': scaled_close.iloc[i],
                        'volume': bars_df['volume'].iloc[i],
                        'synthetic_day': days_generated # Use index 0..N-1
                    }
                    day_bars.append(bar)
                    full_history.append(bar)
                
                current_price = scaled_close.iloc[-1]
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
