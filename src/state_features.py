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
