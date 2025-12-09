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
