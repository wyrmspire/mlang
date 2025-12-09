
import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("YFLimitTest")

def test_limits():
    ticker = "MES=F"
    
    # Define test cases
    # (Interval, List of Periods to Try)
    test_matrix = [
        ("1m", ["5d", "7d", "8d", "29d", "30d"]),
        ("5m", ["55d", "59d", "60d", "61d", "89d", "90d"]),
        ("15m", ["55d", "59d", "60d", "61d", "89d", "90d"]),
        ("30m", ["55d", "59d", "60d", "61d", "89d", "90d"]),
        ("1h", ["89d", "90d", "1y", "729d", "730d", "731d", "2y"])
    ]
    
    results = []
    
    print(f"Testing YFinance API Limits for {ticker}...\n")
    print(f"{'Interval':<10} | {'Period':<10} | {'Result':<10} | {'Rows':<10} | {'Start Date':<25}")
    print("-" * 80)
    
    for interval, periods in test_matrix:
        for period in periods:
            try:
                # Suppress YFinance noise?
                df = yf.download(ticker, period=period, interval=interval, progress=False, ignore_tz=True)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                status = "OK"
                rows = len(df)
                start_date = "N/A"
                
                if df.empty:
                    status = "EMPTY"
                    rows = 0
                else:
                    df.reset_index(inplace=True)
                    # Check for date col
                    for col in ['Date', 'Datetime']:
                        if col in df.columns:
                            start_date = str(df[col].iloc[0])
                            break
                            
                print(f"{interval:<10} | {period:<10} | {status:<10} | {rows:<10} | {start_date:<25}")
                results.append({'Interval': interval, 'Period': period, 'Status': status, 'Rows': rows, 'Start': start_date})
                
            except Exception as e:
                print(f"{interval:<10} | {period:<10} | ERROR      | 0          | {str(e)[:25]}")
                results.append({'Interval': interval, 'Period': period, 'Status': 'ERROR', 'Rows': 0, 'Start': str(e)})

if __name__ == "__main__":
    test_limits()
