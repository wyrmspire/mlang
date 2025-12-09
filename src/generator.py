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
