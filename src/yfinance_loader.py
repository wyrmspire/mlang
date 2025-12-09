"""
YFinance data loader for historical playback.
Fetches 1-minute data from Yahoo Finance and prepares it for playback simulation.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_utils import get_logger

logger = get_logger("yfinance_loader")


class YFinanceLoader:
    """Loads and manages YFinance data for historical playback."""
    
    def __init__(self, symbol: str = "ES=F"):
        """
        Initialize YFinance loader.
        
        Args:
            symbol: Yahoo Finance symbol (default: ES=F for E-mini S&P 500 futures)
        """
        self.symbol = symbol
        self._cache = {}
        
    def fetch_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, 
                   days_back: int = 7) -> pd.DataFrame:
        """
        Fetch 1-minute data from YFinance.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            days_back: Number of days to fetch if dates not specified (default: 7)
            
        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        try:
            # Determine date range
            if end_date is None:
                end_dt = datetime.now()
            else:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                
            if start_date is None:
                start_dt = end_dt - timedelta(days=days_back)
            else:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            
            cache_key = f"{self.symbol}_{start_dt.date()}_{end_dt.date()}"
            
            # Check cache
            if cache_key in self._cache:
                logger.info(f"Using cached data for {cache_key}")
                return self._cache[cache_key].copy()
            
            logger.info(f"Fetching {self.symbol} data from {start_dt.date()} to {end_dt.date()}")
            
            # Fetch data with 1-minute interval
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=start_dt, end=end_dt, interval="1m")
            
            if df.empty:
                logger.warning(f"No data fetched for {self.symbol}")
                return pd.DataFrame()
            
            # Process and rename columns
            df = df.reset_index()
            df = df.rename(columns={
                'Datetime': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only required columns
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            # Ensure time is timezone-aware
            if df['time'].dt.tz is None:
                df['time'] = pd.to_datetime(df['time'], utc=True)
            else:
                df['time'] = df['time'].dt.tz_convert('UTC')
            
            # Add date column for filtering
            df['date'] = df['time'].dt.date.astype(str)
            
            # Cache the result
            self._cache[cache_key] = df.copy()
            
            logger.info(f"Fetched {len(df)} 1-minute candles")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching YFinance data: {e}")
            return pd.DataFrame()
    
    def get_available_dates(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of available dates from dataframe.
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            List of date strings in YYYY-MM-DD format, sorted descending
        """
        if df.empty or 'date' not in df.columns:
            return []
        
        dates = df['date'].unique().tolist()
        dates.sort(reverse=True)
        return dates
    
    def filter_by_date(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        Filter dataframe to a specific date.
        
        Args:
            df: DataFrame with 'date' column
            date: Date string in YYYY-MM-DD format
            
        Returns:
            Filtered DataFrame
        """
        if df.empty or 'date' not in df.columns:
            return pd.DataFrame()
        
        return df[df['date'] == date].copy()
    
    def resample_5m(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 1-minute data to 5-minute candles.
        
        Args:
            df: DataFrame with 1-minute data
            
        Returns:
            DataFrame with 5-minute candles
        """
        if df.empty:
            return pd.DataFrame()
        
        df_copy = df.copy()
        df_copy = df_copy.set_index('time')
        
        resampled = df_copy.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        resampled = resampled.reset_index()
        resampled['date'] = resampled['time'].dt.date.astype(str)
        
        return resampled


# Global instance
_yf_loader = None

def get_yfinance_loader(symbol: str = "ES=F") -> YFinanceLoader:
    """Get or create global YFinance loader instance."""
    global _yf_loader
    if _yf_loader is None or _yf_loader.symbol != symbol:
        _yf_loader = YFinanceLoader(symbol)
    return _yf_loader
