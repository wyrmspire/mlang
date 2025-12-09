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
