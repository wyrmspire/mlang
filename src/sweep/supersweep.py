"""
SUPERSWEEP - Comprehensive Strategy Testing

Tests 30 OCO/limit/ATR configurations across all MES data with market context filters:
- Time of day, Day of week
- Above/below weekly VWAP
- 200 EMA on 5m, 15m
- PDH/PDL (previous day high/low)
- ONH/ONL (overnight high/low)
- Previous day close

Usage:
    python src/sweep/supersweep.py --output results/supersweep_results.parquet
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
from itertools import product

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.variants import CNN_Classic
from src.config import PROCESSED_DIR, ONE_MIN_PARQUET_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("supersweep")

# GPU check
if not torch.cuda.is_available():
    logger.error("GPU NOT DETECTED!")
    sys.exit(1)
device = torch.device("cuda")
logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


# ============ CONFIGURATION GRID ============

# Entry offsets (multiple of ATR)
ENTRY_OFFSETS = [0, 0.25, 0.5, 0.75, 1.0]

# ATR timeframes for stop calculation
ATR_TIMEFRAMES = ['5m', '15m']

# TP multiples
TP_MULTS = [1.0, 1.4, 2.0]

# 30 configurations: 5 offsets × 2 ATR × 3 TP = 30
def generate_configs():
    configs = []
    for offset, atr_tf, tp in product(ENTRY_OFFSETS, ATR_TIMEFRAMES, TP_MULTS):
        configs.append({
            'name': f'LONG_off{offset}_atr{atr_tf}_tp{tp}',
            'direction': 'LONG',
            'entry_offset': offset,
            'atr_tf': atr_tf,
            'tp_mult': tp,
        })
    return configs

CONFIGS = generate_configs()
logger.info(f"Generated {len(CONFIGS)} configurations")


# ============ HELPER FUNCTIONS ============

TICK = 0.25
PV = 5.0  # MES point value

def round_tick(p, d='n'):
    if d == 'u':
        return np.ceil(p / TICK) * TICK
    elif d == 'd':
        return np.floor(p / TICK) * TICK
    return round(p / TICK) * TICK


def calculate_vwap(df, period='W'):
    """Calculate VWAP for given period."""
    df = df.copy()
    df['typical'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical'] * df['volume']
    
    if period == 'W':
        df['period'] = df['time_dt'].dt.isocalendar().week
    else:
        df['period'] = df['time_dt'].dt.date
    
    vwap = df.groupby('period').apply(
        lambda x: x['tp_vol'].cumsum() / x['volume'].cumsum()
    ).reset_index(level=0, drop=True)
    return vwap


def calculate_ema(series, period):
    """Calculate EMA."""
    return series.ewm(span=period, adjust=False).mean()


def get_session_levels(df, trigger_time):
    """Get PDH, PDL, PDC, ONH, ONL for given trigger time."""
    try:
        trigger_date = trigger_time.date()
        
        # Previous day
        prev_day = trigger_date - pd.Timedelta(days=1)
        while prev_day.weekday() >= 5:  # Skip weekends
            prev_day -= pd.Timedelta(days=1)
        
        # Convert to string for date comparison
        prev_day_str = str(prev_day)
        prev_day_data = df[df['time_dt'].dt.strftime('%Y-%m-%d') == prev_day_str]
        
        if len(prev_day_data) == 0:
            return None
        
        pdh = prev_day_data['high'].max()
        pdl = prev_day_data['low'].min()
        pdc = prev_day_data['close'].iloc[-1]
        
        # Overnight - simplified: just use prev day data
        onh = pdh
        onl = pdl
        
        return {
            'pdh': pdh, 'pdl': pdl, 'pdc': pdc,
            'onh': onh, 'onl': onl
        }
    except:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Supersweep Analysis")
    parser.add_argument("--output", type=str, default="results/supersweep_results.parquet")
    parser.add_argument("--risk", type=float, default=300.0)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--model", type=str, default="models/sweep_CNN_Classic_v3_bidirectional.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    model = CNN_Classic(input_dim=4, seq_len=20).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    logger.info(f"Model loaded: {args.model}")
    
    # Load data
    data_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not data_path.exists():
        data_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    df = pd.read_parquet(data_path)
    if 'time' in df.columns:
        df['time_dt'] = pd.to_datetime(df['time'], utc=True)
    elif 'time_dt' not in df.columns:
        df['time_dt'] = df.index
    df = df.sort_values('time_dt').reset_index(drop=True)
    logger.info(f"Loaded {len(df)} bars")
    
    # Resample for different ATR timeframes
    df_5m = df.set_index('time_dt').resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_5m['tr'] = df_5m['high'] - df_5m['low']
    df_5m['atr'] = df_5m['tr'].rolling(14).mean()
    df_5m['ema200'] = calculate_ema(df_5m['close'], 200)
    
    df_15m = df.set_index('time_dt').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_15m['tr'] = df_15m['high'] - df_15m['low']
    df_15m['atr'] = df_15m['tr'].rolling(14).mean()
    df_15m['ema200'] = calculate_ema(df_15m['close'], 200)
    
    # Weekly VWAP on 1m data
    df['volume'] = df.get('volume', 1)  # Default volume if missing
    df['vwap'] = calculate_vwap(df, 'W')
    
    # Test portion (last 30%)
    n = len(df)
    test_start = int(n * 0.7)
    
    logger.info(f"Testing on {n - test_start} bars (last 30%)")
    
    all_trades = []
    last_i = 0
    trade_count = 0
    
    for i in range(test_start + 20, n - 200, 5):
        if i - last_i < 15:
            continue
        
        # CNN detection
        window = df.iloc[i-20:i][['open', 'high', 'low', 'close']].values
        mean, std = np.mean(window), np.std(window)
        if std == 0:
            std = 1.0
        feats = (window - mean) / std
        
        x = torch.FloatTensor(feats).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(x).item()
        
        if prob < args.threshold:
            continue
        
        last_i = i
        trigger_time = df.iloc[i]['time_dt']
        base_price = df.iloc[i]['close']
        
        # Get market context
        hour = trigger_time.hour
        day_of_week = trigger_time.dayofweek
        
        # VWAP
        vwap = df.iloc[i].get('vwap', base_price)
        above_vwap = base_price > vwap
        
        # EMA 200
        try:
            ema200_5m = df_5m.loc[:trigger_time]['ema200'].iloc[-1]
            above_ema200_5m = base_price > ema200_5m
        except:
            above_ema200_5m = None
        
        try:
            ema200_15m = df_15m.loc[:trigger_time]['ema200'].iloc[-1]
            above_ema200_15m = base_price > ema200_15m
        except:
            above_ema200_15m = None
        
        # Session levels
        levels = get_session_levels(df, trigger_time)
        if levels:
            above_pdh = base_price > levels['pdh']
            below_pdl = base_price < levels['pdl']
            above_pdc = base_price > levels['pdc']
            above_onh = base_price > levels['onh']
            below_onl = base_price < levels['onl']
        else:
            above_pdh = below_pdl = above_pdc = above_onh = below_onl = None
        
        # Get ATRs
        try:
            atr_5m = df_5m.loc[:trigger_time]['atr'].iloc[-1]
        except:
            continue
        try:
            atr_15m = df_15m.loc[:trigger_time]['atr'].iloc[-1]
        except:
            continue
        
        if pd.isna(atr_5m) or pd.isna(atr_15m):
            continue
        
        future = df.iloc[i+1:i+200]
        
        # Test each configuration
        for cfg in CONFIGS:
            atr = atr_5m if cfg['atr_tf'] == '5m' else atr_15m
            
            # Entry
            if cfg['entry_offset'] == 0:
                entry = base_price
                fill_bar = i
            else:
                limit = round_tick(base_price + cfg['entry_offset'] * atr, 'u')
                fills = future[future['high'] >= limit]
                if fills.empty:
                    continue
                entry = limit
                fill_bar = fills.index[0]
            
            # Stop and TP
            stop = round_tick(entry - atr, 'd')
            risk_dist = entry - stop
            if risk_dist <= 0:
                continue
            tp = round_tick(entry + risk_dist * cfg['tp_mult'], 'u')
            
            contracts = max(1, int(args.risk / (risk_dist * PV)))
            actual_risk = contracts * risk_dist * PV
            
            # Simulate
            tf = df.iloc[fill_bar+1:fill_bar+150]
            if len(tf) == 0:
                continue
            
            sl = tf[tf['low'] <= stop]
            tph = tf[tf['high'] >= tp]
            si = sl.index[0] if not sl.empty else 999999
            ti = tph.index[0] if not tph.empty else 999999
            
            if ti < si:
                outcome = 'WIN'
                pnl = contracts * risk_dist * cfg['tp_mult'] * PV
                exit_idx = ti
            elif si < 999999:
                outcome = 'LOSS'
                pnl = -actual_risk
                exit_idx = si
            else:
                outcome = 'TIMEOUT'
                pnl = 0
                exit_idx = tf.index[-1]
            
            duration = (df.iloc[exit_idx]['time_dt'] - df.iloc[fill_bar]['time_dt']).total_seconds() / 60
            mae = entry - tf['low'].min()
            
            trade = {
                'trigger_time': trigger_time,
                'config': cfg['name'],
                'entry_offset': cfg['entry_offset'],
                'atr_tf': cfg['atr_tf'],
                'tp_mult': cfg['tp_mult'],
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'atr': atr,
                'contracts': contracts,
                'outcome': outcome,
                'pnl': pnl,
                'duration_mins': duration,
                'mae': mae,
                'hour': hour,
                'day_of_week': day_of_week,
                'above_vwap': above_vwap,
                'above_ema200_5m': above_ema200_5m,
                'above_ema200_15m': above_ema200_15m,
                'above_pdh': above_pdh,
                'below_pdl': below_pdl,
                'above_pdc': above_pdc,
                'above_onh': above_onh,
                'below_onl': below_onl,
            }
            all_trades.append(trade)
        
        trade_count += 1
        if trade_count % 100 == 0:
            logger.info(f"Processed {trade_count} triggers, {len(all_trades)} trade records...")
    
    # Save results
    results_df = pd.DataFrame(all_trades)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path)
    
    logger.info(f"Saved {len(results_df)} trade records to {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUPERSWEEP SUMMARY")
    print("=" * 60)
    print(f"Total triggers: {trade_count}")
    print(f"Total trade records: {len(results_df)}")
    
    # Best configs
    print("\n=== TOP 10 CONFIGS BY WIN RATE ===")
    cfg_stats = results_df.groupby('config').agg({
        'outcome': lambda x: (x == 'WIN').sum(),
        'pnl': ['count', 'sum']
    })
    cfg_stats.columns = ['wins', 'total', 'pnl']
    cfg_stats['wr'] = cfg_stats['wins'] / cfg_stats['total']
    cfg_stats = cfg_stats[cfg_stats['total'] >= 50].sort_values('wr', ascending=False)
    
    for cfg in cfg_stats.head(10).itertuples():
        print(f"  {cfg.Index}: {cfg.wins}/{cfg.total} = {cfg.wr*100:.1f}% WR, ${cfg.pnl:+,.0f}")
    
    # Best filters
    print("\n=== FILTER ANALYSIS ===")
    for filter_col in ['above_vwap', 'above_ema200_5m', 'above_ema200_15m', 'above_pdc']:
        filtered = results_df[results_df[filter_col] == True]
        if len(filtered) > 50:
            wins = (filtered['outcome'] == 'WIN').sum()
            total = len(filtered[filtered['outcome'].isin(['WIN', 'LOSS'])])
            if total > 0:
                print(f"  {filter_col}=True: {wins}/{total} = {wins/total*100:.1f}% WR")


if __name__ == "__main__":
    main()
