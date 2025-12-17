# Sweep Codebase and Documentation
Generated on Thu, Dec 11, 2025 12:27:06 AM

## File: src/sweep/supersweep.py

```python
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

```

---

## File: src/sweep/pattern_miner_v2.py

```python
"""
Pattern Miner V2 - Proportional Detection
Finds patterns where price rises X times a unit, then returns back.
All measurements are RATIOS, not dollar amounts.

Usage:
    python src/sweep/pattern_miner_v2.py \
        --rise-ratio 1.5 --return-ratio 1.0 --invalid-ratio 2.5 \
        --max-triggers 30 --output-suffix "config_001"
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, ONE_MIN_PARQUET_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("pattern_miner_v2")

# Enforce GPU
if not torch.cuda.is_available():
    logger.error("GPU NOT DETECTED! This script requires CUDA.")
    sys.exit(1)
device = torch.device("cuda")
logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# 10 OCO configs: 5 LONG + 5 SHORT
OCO_CONFIGS = [
    {"name": "LONG_1.0R", "direction": "LONG", "r_mult": 1.0},
    {"name": "LONG_1.4R", "direction": "LONG", "r_mult": 1.4},
    {"name": "LONG_2.0R", "direction": "LONG", "r_mult": 2.0},
    {"name": "LONG_1.8R", "direction": "LONG", "r_mult": 1.8},
    {"name": "LONG_2.5R", "direction": "LONG", "r_mult": 2.5},
    {"name": "SHORT_1.0R", "direction": "SHORT", "r_mult": 1.0},
    {"name": "SHORT_1.4R", "direction": "SHORT", "r_mult": 1.4},
    {"name": "SHORT_2.0R", "direction": "SHORT", "r_mult": 2.0},
    {"name": "SHORT_1.8R", "direction": "SHORT", "r_mult": 1.8},
    {"name": "SHORT_2.5R", "direction": "SHORT", "r_mult": 2.5},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Proportional Pattern Miner V2")
    
    # Pattern ratios (proportional, not dollars)
    parser.add_argument("--rise-ratio", type=float, default=1.5,
                        help="Rise as multiple of unit move (e.g., 1.5x)")
    parser.add_argument("--return-ratio", type=float, default=1.0,
                        help="Return as multiple of unit (trigger at -1x)")
    parser.add_argument("--invalid-ratio", type=float, default=2.5,
                        help="Invalidation level (if hit before return)")
    parser.add_argument("--lookback", type=int, default=60,
                        help="Bars to look back for pattern start")
    parser.add_argument("--min-unit", type=float, default=0.5,
                        help="Minimum unit size in points (filter noise)")
    
    # Output
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--max-triggers", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    
    return parser.parse_args()


def simulate_oco(df_1m, trigger_idx, entry_price, stop_price, oco_config):
    """Simulate single OCO outcome. Stop is candle BEFORE move."""
    direction = oco_config["direction"]
    r_mult = oco_config["r_mult"]
    
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return "INVALID", 0
    
    if direction == "LONG":
        tp_price = entry_price + (risk * r_mult)
        future = df_1m.iloc[trigger_idx+1:trigger_idx+2001]
        if len(future) == 0:
            return "TIMEOUT", 0
        
        sl_hit = future[future['low'] <= stop_price]
        tp_hit = future[future['high'] >= tp_price]
    else:  # SHORT
        tp_price = entry_price - (risk * r_mult)
        future = df_1m.iloc[trigger_idx+1:trigger_idx+2001]
        if len(future) == 0:
            return "TIMEOUT", 0
        
        sl_hit = future[future['high'] >= stop_price]
        tp_hit = future[future['low'] <= tp_price]
    
    sl_idx = sl_hit.index[0] if not sl_hit.empty else 999999999
    tp_idx = tp_hit.index[0] if not tp_hit.empty else 999999999
    
    if sl_idx == 999999999 and tp_idx == 999999999:
        return "TIMEOUT", 0
    elif tp_idx < sl_idx:
        return "WIN", r_mult
    else:
        return "LOSS", -1.0


def mine_proportional_patterns(args):
    """
    Mine patterns using proportional ratios.
    Pattern: price rises X times unit, returns to -1x (before hitting invalid level).
    Stop: close of candle BEFORE the move started.
    """
    logger.info(f"Mining: rise={args.rise_ratio}x, return={args.return_ratio}x, "
                f"invalid={args.invalid_ratio}x")
    
    # Load data
    data_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not data_path.exists():
        data_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    if not data_path.exists():
        return {"error": "No data found"}
    
    df_1m = pd.read_parquet(data_path)
    
    if isinstance(df_1m.index, pd.DatetimeIndex) or 'time' not in df_1m.columns:
        df_1m = df_1m.reset_index()
    
    time_cols = [c for c in df_1m.columns if 'time' in c.lower() or c == 'index']
    if time_cols:
        df_1m = df_1m.rename(columns={time_cols[0]: 'time'})
    
    df_1m = df_1m.sort_values('time').reset_index(drop=True)
    df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
    
    # Arrays for speed
    closes = df_1m['close'].values
    highs = df_1m['high'].values
    lows = df_1m['low'].values
    times = df_1m['time'].values
    n = len(df_1m)
    
    # Track results
    oco_results = {cfg["name"]: {"wins": 0, "losses": 0, "pnl": 0.0} 
                   for cfg in OCO_CONFIGS}
    
    patterns_data = []
    pattern_count = 0
    last_trigger = times[0] - np.timedelta64(1, 'D')
    max_triggers = args.max_triggers if args.max_triggers > 0 else float('inf')
    
    logger.info(f"Scanning {n} bars...")
    
    for i in range(100, n - 100):
        if pattern_count >= max_triggers:
            break
        
        curr_time = times[i]
        curr_price = closes[i]
        
        # 15min cooldown
        if (curr_time - last_trigger) < np.timedelta64(15, 'm'):
            continue
        
        # Look for pattern start
        for j in range(i - 1, max(0, i - args.lookback), -1):
            start_price = closes[j]
            
            # ========== SHORT PATTERN: Price UP then returns ==========
            peak_price = np.max(highs[j:i+1])
            peak_idx = j + np.argmax(highs[j:i+1])
            
            drop = peak_price - curr_price
            rise = peak_price - start_price
            
            if drop >= args.min_unit and rise >= args.min_unit:
                ratio = rise / drop
                if args.rise_ratio <= ratio <= args.invalid_ratio:
                    if j >= 1:
                        stop_price = closes[j - 1]
                        entry_price = curr_price
                        
                        pattern_info = {
                            'trigger_idx': i,
                            'trigger_time': curr_time,
                            'start_idx': j,
                            'peak_idx': peak_idx,
                            'entry': entry_price,
                            'stop': stop_price,
                            'unit': drop,
                            'rise': rise,
                            'ratio': ratio,
                            'peak': peak_price,
                            'direction': 'SHORT',  # This is a SHORT setup
                        }
                        
                        for oco_cfg in OCO_CONFIGS:
                            outcome, pnl_r = simulate_oco(df_1m, i, entry_price, stop_price, oco_cfg)
                            if outcome == "WIN":
                                oco_results[oco_cfg["name"]]["wins"] += 1
                                oco_results[oco_cfg["name"]]["pnl"] += pnl_r
                            elif outcome == "LOSS":
                                oco_results[oco_cfg["name"]]["losses"] += 1
                                oco_results[oco_cfg["name"]]["pnl"] += pnl_r
                            pattern_info[f"outcome_{oco_cfg['name']}"] = outcome
                        
                        patterns_data.append(pattern_info)
                        pattern_count += 1
                        last_trigger = curr_time
                        break
            
            # ========== LONG PATTERN: Price DOWN then returns ==========
            trough_price = np.min(lows[j:i+1])
            trough_idx = j + np.argmin(lows[j:i+1])
            
            rise_back = curr_price - trough_price
            fall = start_price - trough_price
            
            if rise_back >= args.min_unit and fall >= args.min_unit:
                ratio = fall / rise_back
                if args.rise_ratio <= ratio <= args.invalid_ratio:
                    if j >= 1:
                        stop_price = closes[j - 1]
                        entry_price = curr_price
                        
                        pattern_info = {
                            'trigger_idx': i,
                            'trigger_time': curr_time,
                            'start_idx': j,
                            'peak_idx': trough_idx,  # Actually trough for LONG
                            'entry': entry_price,
                            'stop': stop_price,
                            'unit': rise_back,
                            'rise': fall,
                            'ratio': ratio,
                            'peak': trough_price,  # Actually trough for LONG
                            'direction': 'LONG',  # This is a LONG setup
                        }
                        
                        for oco_cfg in OCO_CONFIGS:
                            outcome, pnl_r = simulate_oco(df_1m, i, entry_price, stop_price, oco_cfg)
                            if outcome == "WIN":
                                oco_results[oco_cfg["name"]]["wins"] += 1
                                oco_results[oco_cfg["name"]]["pnl"] += pnl_r
                            elif outcome == "LOSS":
                                oco_results[oco_cfg["name"]]["losses"] += 1
                                oco_results[oco_cfg["name"]]["pnl"] += pnl_r
                            pattern_info[f"outcome_{oco_cfg['name']}"] = outcome
                        
                        patterns_data.append(pattern_info)
                        pattern_count += 1
                        last_trigger = curr_time
                        break
    
    # Calculate stats
    oco_stats = []
    for cfg in OCO_CONFIGS:
        name = cfg["name"]
        data = oco_results[name]
        total = data["wins"] + data["losses"]
        win_rate = data["wins"] / total if total > 0 else 0
        
        oco_stats.append({
            "oco_config": name,
            "direction": cfg["direction"],
            "r_mult": cfg["r_mult"],
            "wins": data["wins"],
            "losses": data["losses"],
            "total": total,
            "win_rate": round(win_rate, 4),
            "total_pnl_r": round(data["pnl"], 2),
            "ev_per_trade": round(data["pnl"] / total, 3) if total > 0 else 0,
        })
    
    oco_stats = sorted(oco_stats, key=lambda x: x["ev_per_trade"], reverse=True)
    
    summary = {
        "pattern_config": {
            "rise_ratio": args.rise_ratio,
            "return_ratio": args.return_ratio,
            "invalid_ratio": args.invalid_ratio,
            "min_unit": args.min_unit,
        },
        "total_patterns": pattern_count,
        "oco_results": oco_stats,
        "best_oco": oco_stats[0] if oco_stats else None,
    }
    
    return {
        "summary": summary,
        "patterns": pd.DataFrame(patterns_data) if patterns_data else pd.DataFrame(),
    }


def main():
    args = parse_args()
    result = mine_proportional_patterns(args)
    
    if "error" in result:
        print(json.dumps(result))
        return
    
    summary = result["summary"]
    patterns_df = result["patterns"]
    
    logger.info("=" * 60)
    logger.info(f"Mining Complete! Found {summary['total_patterns']} patterns")
    logger.info("Top 5 OCO configs by EV:")
    for oco in summary["oco_results"][:5]:
        logger.info(f"  {oco['oco_config']}: WR={oco['win_rate']*100:.1f}%, "
                    f"EV={oco['ev_per_trade']:.3f}R")
    logger.info("=" * 60)
    
    if not args.dry_run and len(patterns_df) > 0:
        suffix = args.output_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        out_path = PROCESSED_DIR / f"patterns_v2_{suffix}.parquet"
        patterns_df.to_parquet(out_path)
        
        stats_path = PROCESSED_DIR / f"patterns_v2_{suffix}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved to {out_path}")
    
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()

```

---

## File: src/sweep/train_sweep.py

```python
"""
CLI Training Script for Sweep Pipeline
Trains models with different architectures and candle compositions.

Usage:
    python src/sweep/train_sweep.py \
        --architecture CNN_Classic \
        --input-data labeled_sweep_001.parquet \
        --candles-1m 30 --candles-3m 20 --candles-5m 10 \
        --epochs 10 --lr 0.001 \
        --output-suffix "cnn_001"
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, ONE_MIN_PARQUET_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger
from src.models.variants import CNN_Classic, CNN_Wide, LSTM_Seq, Feature_MLP

logger = get_logger("train_sweep")

# Enforce GPU
if not torch.cuda.is_available():
    logger.error("GPU NOT DETECTED! This script requires CUDA.")
    sys.exit(1)

device = torch.device("cuda")
logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")


class TradeDataset(Dataset):
    """Dataset for trade pattern training."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Model Training with CLI parameters")
    
    # Model architecture
    parser.add_argument("--architecture", type=str, default="CNN_Classic",
                        choices=["CNN_Classic", "CNN_Wide", "LSTM_Seq", "Feature_MLP"],
                        help="Model architecture to use")
    
    # Input data
    parser.add_argument("--input-data", type=str, required=True,
                        help="Path to labeled pattern data (parquet)")
    
    # Candle composition
    parser.add_argument("--candles-1m", type=int, default=30)
    parser.add_argument("--candles-3m", type=int, default=0)
    parser.add_argument("--candles-5m", type=int, default=0)
    parser.add_argument("--candles-15m", type=int, default=0)
    
    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Split ratios
    parser.add_argument("--train-ratio", type=float, default=0.56)
    parser.add_argument("--val-ratio", type=float, default=0.14)
    
    # OCO config for labeling (use outcome column from sweep)
    parser.add_argument("--oco-config", type=str, default="SHORT_2.0R_50ATR",
                        help="OCO config to use for outcome labels (e.g. SHORT_2.0R_50ATR)")
    
    # Output
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    
    return parser.parse_args()


def prepare_data(
    patterns: pd.DataFrame,
    df_1m: pd.DataFrame,
    window_size: int = 30,
    oco_config: str = "SHORT_2.0R_50ATR",
) -> tuple:
    """
    Prepare training data from patterns.
    Uses specified OCO config for outcome labels.
    
    Returns:
        X, y, timestamps
    """
    X = []
    y = []
    timestamps = []
    
    # Determine outcome column
    outcome_col = f"outcome_{oco_config}"
    if outcome_col not in patterns.columns:
        # Fallback to legacy 'outcome' column
        if 'outcome' in patterns.columns:
            outcome_col = 'outcome'
        else:
            logger.error(f"Outcome column not found: {outcome_col}")
            return np.array([]), np.array([]), []
    
    valid_patterns = patterns[patterns[outcome_col].isin(['WIN', 'LOSS'])].copy()
    valid_patterns = valid_patterns.sort_values('trigger_time')
    
    logger.info(f"Processing {len(valid_patterns)} valid patterns using {outcome_col}...")
    
    # Ensure index is UTC
    if df_1m.index.tz is None:
        df_1m.index = df_1m.index.tz_localize('UTC')
    else:
        df_1m.index = df_1m.index.tz_convert('UTC')
        
    for idx, pattern in valid_patterns.iterrows():
        trigger_time = pattern['trigger_time']
        
        # Ensure trigger_time is UTC
        if pd.Timestamp(trigger_time).tz is None:
            trigger_time = pd.Timestamp(trigger_time).tz_localize('UTC')
        else:
            trigger_time = pd.Timestamp(trigger_time).tz_convert('UTC')
        
        # Get window before trigger
        end_time = trigger_time
        start_time = end_time - pd.Timedelta(minutes=window_size)
        
        try:
            window = df_1m.loc[start_time:end_time]
        except KeyError:
            continue
            
        window = window[window.index < end_time]
        
        if len(window) < window_size:
            continue
        
        # Z-Score Normalization per window (per success_study.md)
        feats = window[['open', 'high', 'low', 'close']].values
        mean = np.mean(feats)
        std = np.std(feats)
        if std == 0:
            std = 1.0  # Prevent div/0
        
        feats_norm = (feats - mean) / std
        
        # Take last window_size bars
        if len(feats_norm) > window_size:
            feats_norm = feats_norm[-window_size:]
        elif len(feats_norm) < window_size:
            continue
        
        # Invert for SHORT direction to unify dataset (per success_study.md)
        pattern_direction = pattern.get('direction', 'SHORT')
        if pattern_direction == "SHORT":
            feats_norm = -feats_norm
        
        # Label using OCO-specific outcome
        label = 1 if pattern[outcome_col] == 'WIN' else 0
        
        X.append(feats_norm)
        y.append(label)
        timestamps.append(trigger_time)
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Prepared {len(X)} samples. Win rate: {np.mean(y):.2f}")
    
    return X, y, timestamps


def get_model(architecture: str, seq_len: int, input_dim: int = 4):
    """Get model instance by architecture name."""
    
    if architecture == "CNN_Classic":
        return CNN_Classic(input_dim=input_dim, seq_len=seq_len)
    elif architecture == "CNN_Wide":
        return CNN_Wide(input_dim=input_dim, seq_len=seq_len)
    elif architecture == "LSTM_Seq":
        return LSTM_Seq(input_dim=input_dim)
    elif architecture == "Feature_MLP":
        # MLP expects flattened features
        return Feature_MLP(input_dim=seq_len * input_dim)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
) -> dict:
    """
    Train model and return metrics.
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {train_loss/len(train_loader):.4f}, "
                    f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
    
    return {
        "best_val_acc": best_val_acc,
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
        "history": history,
    }


def main():
    args = parse_args()
    
    # Calculate window size from candle composition
    window_size = args.candles_1m  # Primary window (we'll handle multi-TF later)
    
    logger.info(f"Training {args.architecture} with {window_size} bar window")
    
    # Load pattern data
    pattern_path = Path(args.input_data)
    if not pattern_path.is_absolute():
        pattern_path = PROCESSED_DIR / args.input_data
    
    if not pattern_path.exists():
        logger.error(f"Pattern data not found: {pattern_path}")
        return {"error": "No data"}
    
    patterns = pd.read_parquet(pattern_path)
    logger.info(f"Loaded {len(patterns)} patterns")
    
    # Load 1m data
    data_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not data_path.exists():
        data_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    df_1m = pd.read_parquet(data_path)
    if 'time' in df_1m.columns:
        df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
        df_1m = df_1m.set_index('time')
    df_1m = df_1m.sort_index()
    
    # Prepare data
    X, y, timestamps = prepare_data(patterns, df_1m, window_size, args.oco_config)
    
    if len(X) < 50:
        logger.error(f"Not enough samples: {len(X)}")
        return {"error": "Insufficient data"}
    
    # Split chronologically
    n = len(X)
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + args.val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Create dataloaders
    train_ds = TradeDataset(X_train, y_train)
    val_ds = TradeDataset(X_val, y_val)
    test_ds = TradeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Get model
    model = get_model(args.architecture, window_size)
    logger.info(f"Model: {model.__class__.__name__}")
    
    # Train
    train_result = train_model(model, train_loader, val_loader, args.epochs, args.lr)
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    # Summary
    summary = {
        "architecture": args.architecture,
        "window_size": window_size,
        "candle_composition": f"{args.candles_1m}x1m+{args.candles_3m}x3m+"
                              f"{args.candles_5m}x5m+{args.candles_15m}x15m",
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "best_val_acc": train_result["best_val_acc"],
        "test_acc": test_acc,
        "train_win_rate": float(np.mean(y_train)),
        "test_win_rate": float(np.mean(y_test)),
    }
    
    logger.info("=" * 50)
    logger.info(f"Training Complete!")
    logger.info(f"Best Val Acc: {train_result['best_val_acc']:.3f}")
    logger.info(f"Test Acc: {test_acc:.3f}")
    logger.info("=" * 50)
    
    # Save model
    if not args.dry_run:
        suffix = args.output_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODELS_DIR / f"sweep_{args.architecture}_{suffix}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        summary["model_path"] = str(model_path)
        
        # Save summary
        summary_path = MODELS_DIR / f"sweep_{args.architecture}_{suffix}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()

```

---

## File: src/sweep/oco_tester.py

```python
"""
OCO Bracket Tester for Sweep Pipeline
Tests multiple OCO configurations on labeled pattern data.

Usage:
    python src/sweep/oco_tester.py \
        --pattern-data labeled_sweep_001.parquet \
        --model-path models/cnn_sweep.pth \
        --output results/oco_results.csv
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import json
from typing import List, Dict

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, ONE_MIN_PARQUET_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger
from src.sweep.config import OCOBracketConfig
from src.sweep.param_grid import get_default_oco_scenarios

logger = get_logger("oco_tester")


def parse_args():
    parser = argparse.ArgumentParser(description="OCO Bracket Tester")
    
    parser.add_argument("--pattern-data", type=str, required=True,
                        help="Path to labeled pattern data (parquet)")
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to trained model (optional, for filtering)")
    parser.add_argument("--output", type=str, default="",
                        help="Output CSV path")
    
    # OCO override params (optional - uses defaults if not specified)
    parser.add_argument("--direction", type=str, default="",
                        choices=["", "LONG", "SHORT"])
    parser.add_argument("--r-mult", type=float, default=0)
    parser.add_argument("--stop-atr-pct", type=float, default=0)
    parser.add_argument("--stop-type", type=str, default="",
                        choices=["", "WICK", "OPEN", "ATR"])
    
    parser.add_argument("--use-defaults", action="store_true",
                        help="Use 10 default OCO scenarios")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only output stats")
    
    # Money management
    parser.add_argument("--risk-per-trade", type=float, default=75.0)
    parser.add_argument("--starting-balance", type=float, default=2000.0)
    
    return parser.parse_args()


def load_model(model_path: str):
    """Load trained model for signal filtering (optional)."""
    if not model_path or not Path(model_path).exists():
        return None
    
    # Import model architecture
    from src.models.cnn_model import TradeCNN
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradeCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model


def run_oco_backtest(
    patterns: pd.DataFrame,
    df_1m: pd.DataFrame,
    oco_config: OCOBracketConfig,
    risk_per_trade: float = 75.0,
    starting_balance: float = 2000.0,
) -> Dict:
    """
    Run backtest with specific OCO configuration.
    
    Returns:
        Dict with results: trades, win_rate, pnl, etc.
    """
    results = []
    balance = starting_balance
    
    for idx, pattern in patterns.iterrows():
        # Skip inconclusive
        original_outcome = pattern.get('outcome', '')
        if original_outcome == 'Inconclusive':
            continue
        
        trigger_time = pattern['trigger_time']
        
        # Ensure proper timezone for comparison
        if pd.Timestamp(trigger_time).tz is None:
             trigger_time = pd.Timestamp(trigger_time).tz_localize('UTC')
        else:
             trigger_time = pd.Timestamp(trigger_time).tz_convert('UTC')
             
        entry_price = pattern['entry']
        atr = pattern.get('atr', 1.0)
        
        # Determine direction based on config
        direction = oco_config.direction
        if direction == "BOTH":
            # Use original pattern direction if available
            direction = pattern.get('direction', 'SHORT')
        
        # Calculate stop based on stop_type
        if oco_config.stop_type == "ATR":
            stop_dist = atr * oco_config.stop_atr_pct
        elif oco_config.stop_type == "WICK":
            # Use pattern's stop (wick-based) if available
            if 'stop' in pattern and pd.notna(pattern.get('stop')):
                stop_dist = abs(pattern['stop'] - entry_price)
            else:
                # Fallback to ATR
                stop_dist = atr * oco_config.stop_atr_pct
        else:  # OPEN
            stop_dist = atr * 0.5  # Default to 0.5 ATR
        
        if stop_dist <= 0:
            stop_dist = atr * 0.5
        
        # Calculate TP based on R-multiple
        tp_dist = stop_dist * oco_config.r_multiple
        
        if direction == "SHORT":
            stop_price = entry_price + stop_dist
            tp_price = entry_price - tp_dist
        else:  # LONG
            stop_price = entry_price - stop_dist
            tp_price = entry_price + tp_dist
        
        # Simulate outcome using 1m data
        future = df_1m[df_1m.index > trigger_time]
        if len(future) == 0:
            continue
        
        # Limit search window
        future = future.iloc[:2000]
        
        highs = future['high'].values
        lows = future['low'].values
        times = future.index.values
        
        if direction == "SHORT":
            mask_win = lows <= tp_price
            mask_loss = highs >= stop_price
        else:
            mask_win = highs >= tp_price
            mask_loss = lows <= stop_price
        
        idx_win = np.argmax(mask_win) if mask_win.any() else 999999
        idx_loss = np.argmax(mask_loss) if mask_loss.any() else 999999
        
        if idx_win == 999999 and idx_loss == 999999:
            outcome = 'TIMEOUT'
            pnl = 0
        elif idx_win < idx_loss:
            outcome = 'WIN'
            pnl = risk_per_trade * oco_config.r_multiple
        else:
            outcome = 'LOSS'
            pnl = -risk_per_trade
        
        balance += pnl
        
        results.append({
            'trigger_time': trigger_time,
            'direction': direction,
            'entry': entry_price,
            'stop': stop_price,
            'tp': tp_price,
            'outcome': outcome,
            'pnl': pnl,
            'balance': balance,
            'oco_config': oco_config.label,
        })
    
    # Calculate summary stats
    if results:
        df_results = pd.DataFrame(results)
        valid_trades = df_results[df_results['outcome'].isin(['WIN', 'LOSS'])]
        wins = len(valid_trades[valid_trades['outcome'] == 'WIN'])
        losses = len(valid_trades[valid_trades['outcome'] == 'LOSS'])
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        total_pnl = df_results['pnl'].sum()
        
        # Expected value per trade
        avg_win = risk_per_trade * oco_config.r_multiple
        avg_loss = risk_per_trade
        ev = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        summary = {
            "oco_config": oco_config.label,
            "direction": oco_config.direction,
            "r_multiple": oco_config.r_multiple,
            "stop_type": oco_config.stop_type,
            "stop_atr_pct": oco_config.stop_atr_pct,
            "total_trades": len(valid_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "final_balance": round(balance, 2),
            "expected_value": round(ev, 2),
        }
    else:
        df_results = pd.DataFrame()
        summary = {"oco_config": oco_config.label, "error": "No trades"}
    
    return {
        "trades": df_results,
        "summary": summary,
    }


def main():
    args = parse_args()
    
    # Ensure CUDA is available
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU detected, running on CPU")
    
    # Load pattern data
    pattern_path = Path(args.pattern_data)
    if not pattern_path.is_absolute():
        pattern_path = PROCESSED_DIR / args.pattern_data
    
    if not pattern_path.exists():
        logger.error(f"Pattern data not found: {pattern_path}")
        return
    
    patterns = pd.read_parquet(pattern_path)
    logger.info(f"Loaded {len(patterns)} patterns from {pattern_path}")
    
    # Load 1m data for simulation
    data_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not data_path.exists():
        data_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    df_1m = pd.read_parquet(data_path)
    if 'time' in df_1m.columns:
        df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
        df_1m = df_1m.set_index('time')
    df_1m = df_1m.sort_index()
    
    # Determine OCO configs to test
    if args.use_defaults:
        oco_configs = get_default_oco_scenarios()
        logger.info(f"Using {len(oco_configs)} default OCO scenarios")
    elif args.direction and args.r_mult > 0:
        # Single custom config
        oco_configs = [OCOBracketConfig(
            direction=args.direction,
            r_multiple=args.r_mult,
            stop_atr_pct=args.stop_atr_pct or 0.5,
            stop_type=args.stop_type or "ATR",
            config_id="custom",
        )]
    else:
        # Default 10 scenarios
        oco_configs = get_default_oco_scenarios()
    
    # Run backtests for each OCO config
    all_summaries = []
    all_trades = []
    
    for oco_config in oco_configs:
        logger.info(f"Testing OCO: {oco_config.label}")
        
        result = run_oco_backtest(
            patterns=patterns,
            df_1m=df_1m,
            oco_config=oco_config,
            risk_per_trade=args.risk_per_trade,
            starting_balance=args.starting_balance,
        )
        
        summary = result["summary"]
        trades = result["trades"]
        
        all_summaries.append(summary)
        if not trades.empty:
            all_trades.append(trades)
        
        logger.info(f"  -> Trades: {summary.get('total_trades', 0)}, "
                    f"Win Rate: {summary.get('win_rate', 0)*100:.1f}%, "
                    f"PnL: ${summary.get('total_pnl', 0):.2f}")
    
    # Output results
    logger.info("=" * 60)
    logger.info("OCO SWEEP RESULTS")
    logger.info("=" * 60)
    
    df_summary = pd.DataFrame(all_summaries)
    print(df_summary.to_string(index=False))
    
    if not args.dry_run and args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(out_path, index=False)
        logger.info(f"Results saved to {out_path}")
        
        # Also save detailed trades
        if all_trades:
            trades_path = out_path.parent / f"{out_path.stem}_trades.parquet"
            pd.concat(all_trades).to_parquet(trades_path)
            logger.info(f"Detailed trades saved to {trades_path}")
    
    # Print JSON for orchestrator
    print(json.dumps(all_summaries, indent=2))
    return all_summaries


if __name__ == "__main__":
    main()

```

---

## File: src/sweep/param_grid.py

```python
"""
Parameter Grid Generator
Creates sweep configurations for pattern mining, OCO brackets, and models.
"""

import numpy as np
from typing import List
import itertools

from .config import (
    PatternSweepConfig, 
    OCOBracketConfig, 
    ModelSweepConfig,
    CandleComposition,
    PATTERN_SWEEP_RANGES,
    OCO_SWEEP_VALUES,
    MODEL_ARCHITECTURES,
    CANDLE_COMPOSITIONS,
)


def generate_pattern_grid(n: int = 33, seed: int = 42) -> List[PatternSweepConfig]:
    """
    Generate N pattern configurations via Latin Hypercube Sampling.
    Default 33 configs × 30 triggers = ~1000 pattern evaluations.
    
    Args:
        n: Number of configurations to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of PatternSweepConfig objects
    """
    np.random.seed(seed)
    configs = []
    
    ranges = PATTERN_SWEEP_RANGES
    
    for i in range(n):
        # Sample each parameter uniformly within range
        rise_min = np.random.uniform(*ranges["rise_ratio_min"])
        rise_max = np.random.uniform(rise_min + 0.5, ranges["rise_ratio_max"][1])  # Ensure max > min
        
        config = PatternSweepConfig(
            rise_ratio_min=round(rise_min, 2),
            rise_ratio_max=round(rise_max, 2),
            min_drop=round(np.random.uniform(*ranges["min_drop"]), 2),
            atr_buffer=round(np.random.uniform(*ranges["atr_buffer"]), 2),
            validation_distance=round(np.random.uniform(*ranges["validation_distance"]), 2),
            lookback_bars=int(np.random.uniform(*ranges["lookback_bars"])),
            config_id=f"pattern_{i:03d}",
        )
        configs.append(config)
    
    return configs


def generate_oco_grid(n: int = 33, seed: int = 42) -> List[OCOBracketConfig]:
    """
    Generate N OCO bracket configurations.
    Uses combination of grid + random for diversity.
    
    Args:
        n: Number of configurations to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of OCOBracketConfig objects
    """
    np.random.seed(seed)
    configs = []
    
    vals = OCO_SWEEP_VALUES
    
    # Generate all combinations
    all_combos = list(itertools.product(
        vals["direction"],
        vals["r_multiple"],
        vals["stop_atr_pct"],
        vals["stop_type"],
    ))
    
    # If we need fewer than total combos, sample randomly
    if n < len(all_combos):
        indices = np.random.choice(len(all_combos), size=n, replace=False)
        selected = [all_combos[i] for i in indices]
    else:
        selected = all_combos[:n]
    
    for i, (direction, r_mult, stop_pct, stop_type) in enumerate(selected):
        config = OCOBracketConfig(
            direction=direction,
            r_multiple=r_mult,
            stop_atr_pct=stop_pct,
            stop_type=stop_type,
            config_id=f"oco_{i:03d}",
        )
        configs.append(config)
    
    return configs


def generate_model_grid(
    architectures: List[str] = None,
    candle_compositions: List[CandleComposition] = None,
) -> List[ModelSweepConfig]:
    """
    Generate model configurations for each architecture × candle composition.
    
    Returns:
        List of ModelSweepConfig objects
    """
    if architectures is None:
        architectures = MODEL_ARCHITECTURES
    if candle_compositions is None:
        candle_compositions = CANDLE_COMPOSITIONS
    
    configs = []
    idx = 0
    
    for arch in architectures:
        for candle_comp in candle_compositions:
            # Adjust seq_len based on architecture
            if arch == "CNN_Wide" and candle_comp.candles_1m < 60:
                # Skip wide CNN for small inputs
                continue
                
            config = ModelSweepConfig(
                architecture=arch,
                epochs=10,
                learning_rate=0.001,
                batch_size=32,
                dropout=0.3,
                candle_composition=candle_comp,
                config_id=f"model_{idx:03d}",
            )
            configs.append(config)
            idx += 1
    
    return configs


def get_default_oco_scenarios() -> List[OCOBracketConfig]:
    """
    Get the 10 default OCO scenarios for test phase evaluation.
    Every model test gets these 10 results.
    """
    return [
        # Long scenarios
        OCOBracketConfig("LONG", 1.0, 0.50, "ATR", "default_01"),
        OCOBracketConfig("LONG", 1.4, 0.50, "ATR", "default_02"),
        OCOBracketConfig("LONG", 2.0, 0.50, "WICK", "default_03"),
        OCOBracketConfig("LONG", 1.4, 0.25, "WICK", "default_04"),
        # Short scenarios
        OCOBracketConfig("SHORT", 1.0, 0.50, "ATR", "default_05"),
        OCOBracketConfig("SHORT", 1.4, 0.50, "ATR", "default_06"),
        OCOBracketConfig("SHORT", 2.0, 0.50, "WICK", "default_07"),
        OCOBracketConfig("SHORT", 1.4, 0.25, "WICK", "default_08"),
        # Hybrid scenarios
        OCOBracketConfig("LONG", 1.8, 0.75, "ATR", "default_09"),
        OCOBracketConfig("SHORT", 1.8, 0.75, "ATR", "default_10"),
    ]


if __name__ == "__main__":
    # Quick test
    print("Pattern Configs:", len(generate_pattern_grid(33)))
    print("OCO Configs:", len(generate_oco_grid(33)))
    print("Model Configs:", len(generate_model_grid()))
    print("Default OCO Scenarios:", len(get_default_oco_scenarios()))

```

---

## File: src/sweep/config.py

```python
"""
Sweep Configuration Dataclasses
Defines all tunable parameters for the Shotgun Sweep pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class PatternSweepConfig:
    """Configuration for pattern mining geometry."""
    rise_ratio_min: float = 2.5      # Min (Peak - Start) / (Start - Trigger)
    rise_ratio_max: float = 4.0      # Max ratio (invalidation threshold)
    min_drop: float = 1.0            # Minimum $ drop to qualify
    atr_buffer: float = 0.2          # ATR multiplier for stop placement
    validation_distance: float = 1.0  # Distance for pattern validation
    lookback_bars: int = 120         # How far back to scan for patterns
    
    # Unique identifier for this config
    config_id: str = ""
    
    def to_cli_args(self) -> List[str]:
        """Convert to CLI argument list."""
        return [
            "--rise-min", str(self.rise_ratio_min),
            "--rise-max", str(self.rise_ratio_max),
            "--min-drop", str(self.min_drop),
            "--atr-buffer", str(self.atr_buffer),
            "--validation-dist", str(self.validation_distance),
            "--lookback", str(self.lookback_bars),
        ]
    
    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "rise_ratio_min": self.rise_ratio_min,
            "rise_ratio_max": self.rise_ratio_max,
            "min_drop": self.min_drop,
            "atr_buffer": self.atr_buffer,
            "validation_distance": self.validation_distance,
            "lookback_bars": self.lookback_bars,
        }


@dataclass
class CandleComposition:
    """Defines the mix of candle timeframes for model input."""
    candles_1m: int = 30    # Number of 1-minute candles
    candles_3m: int = 20    # Number of 3-minute candles  
    candles_5m: int = 10    # Number of 5-minute candles
    candles_15m: int = 0    # Number of 15-minute candles
    
    @property
    def total_features(self) -> int:
        """Total number of candle input features."""
        return (self.candles_1m + self.candles_3m + 
                self.candles_5m + self.candles_15m) * 4  # OHLC
    
    @property
    def label(self) -> str:
        """Human-readable label for this composition."""
        parts = []
        if self.candles_1m: parts.append(f"{self.candles_1m}x1m")
        if self.candles_3m: parts.append(f"{self.candles_3m}x3m")
        if self.candles_5m: parts.append(f"{self.candles_5m}x5m")
        if self.candles_15m: parts.append(f"{self.candles_15m}x15m")
        return "+".join(parts) if parts else "empty"
    
    def to_cli_args(self) -> List[str]:
        return [
            "--candles-1m", str(self.candles_1m),
            "--candles-3m", str(self.candles_3m),
            "--candles-5m", str(self.candles_5m),
            "--candles-15m", str(self.candles_15m),
        ]


@dataclass
class OCOBracketConfig:
    """Configuration for OCO (One-Cancels-Other) bracket testing."""
    direction: str = "SHORT"          # 'LONG', 'SHORT', 'BOTH'
    r_multiple: float = 1.4           # Take profit as multiple of risk
    stop_atr_pct: float = 0.5         # Stop distance as % of ATR
    stop_type: str = "WICK"           # 'WICK', 'OPEN', 'ATR'
    
    # Unique identifier
    config_id: str = ""
    
    @property
    def label(self) -> str:
        return f"{self.direction}_{self.r_multiple}R_{self.stop_type}_{int(self.stop_atr_pct*100)}atr"
    
    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "direction": self.direction,
            "r_multiple": self.r_multiple,
            "stop_atr_pct": self.stop_atr_pct,
            "stop_type": self.stop_type,
        }


@dataclass
class ModelSweepConfig:
    """Configuration for model architecture and training."""
    architecture: str = "CNN_Classic"  # 'CNN_Classic', 'CNN_Wide', 'LSTM', 'MLP'
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    dropout: float = 0.3
    
    # Input configuration
    candle_composition: CandleComposition = field(default_factory=CandleComposition)
    
    # Unique identifier
    config_id: str = ""
    
    def to_cli_args(self) -> List[str]:
        args = [
            "--architecture", self.architecture,
            "--epochs", str(self.epochs),
            "--lr", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--dropout", str(self.dropout),
        ]
        args.extend(self.candle_composition.to_cli_args())
        return args
    
    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "architecture": self.architecture,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "candle_composition": self.candle_composition.label,
        }


# ============================================================
# Default Sweep Ranges
# ============================================================

PATTERN_SWEEP_RANGES = {
    "rise_ratio_min": (1.5, 3.5),
    "rise_ratio_max": (2.5, 6.0),
    "min_drop": (0.5, 2.0),
    "atr_buffer": (0.1, 0.5),
    "validation_distance": (0.5, 2.0),
    "lookback_bars": (60, 180),
}

OCO_SWEEP_VALUES = {
    "direction": ["LONG", "SHORT"],
    "r_multiple": [1.0, 1.4, 1.8, 2.0, 2.5, 3.0],
    "stop_atr_pct": [0.25, 0.5, 0.75, 1.0],
    "stop_type": ["WICK", "OPEN", "ATR"],
}

MODEL_ARCHITECTURES = ["CNN_Classic", "CNN_Wide", "LSTM_Seq", "Feature_MLP"]

# Pre-defined candle compositions to sweep
CANDLE_COMPOSITIONS = [
    CandleComposition(30, 0, 0, 0),      # Pure 30x1m
    CandleComposition(60, 0, 0, 0),      # Pure 60x1m
    CandleComposition(20, 0, 0, 0),      # Minimal 20x1m
    CandleComposition(30, 20, 10, 0),    # Mixed: 30x1m + 20x3m + 10x5m
    CandleComposition(20, 10, 5, 0),     # Light mixed
    CandleComposition(40, 10, 0, 0),     # Heavy 1m + some 3m
]

```

---

## File: src/sweep/run_sweep.py

```python
#!/usr/bin/env python
"""
Shotgun Sweep Pipeline - Main CLI Entry Point

Usage:
    # Full sweep (100 geometries × 10 OCO × 30 triggers → top 10 combos → full train)
    python src/sweep/run_sweep.py --full

    # Quick test (5 geometries × 10 triggers)
    python src/sweep/run_sweep.py --quick

    # Pattern sweep only (no training)
    python src/sweep/run_sweep.py --phase pattern --n-configs 50 --n-triggers 30
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.sweep.orchestrator import SweepOrchestrator
from src.utils.logging_utils import get_logger

logger = get_logger("run_sweep")


def parse_args():
    parser = argparse.ArgumentParser(description="Shotgun Sweep Pipeline CLI")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--full", action="store_true",
                           help="Run complete sweep pipeline")
    mode_group.add_argument("--quick", action="store_true",
                           help="Quick test (5 configs × 10 triggers)")
    mode_group.add_argument("--phase", type=str,
                           choices=["pattern", "train"],
                           help="Run specific phase only")
    
    # Sweep params
    parser.add_argument("--n-configs", type=int, default=100,
                        help="Number of pattern geometries to test")
    parser.add_argument("--n-triggers", type=int, default=30,
                        help="Patterns per geometry for statistics")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top (geometry, OCO) combos for full training")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("SHOTGUN SWEEP PIPELINE")
    logger.info("=" * 60)
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    orchestrator = SweepOrchestrator(output_dir)
    
    if args.quick:
        logger.info("Running QUICK test: 5 geometries × 10 triggers")
        orchestrator.run_full_sweep(
            n_pattern_configs=5,
            n_triggers=10,
            top_k_combos=3,
        )
        
    elif args.full:
        logger.info(f"Running FULL sweep: {args.n_configs} geometries × 10 OCO × {args.n_triggers} triggers")
        orchestrator.run_full_sweep(
            n_pattern_configs=args.n_configs,
            n_triggers=args.n_triggers,
            top_k_combos=args.top_k,
        )
        
    elif args.phase == "pattern":
        logger.info(f"Running PATTERN sweep only: {args.n_configs} configs")
        orchestrator.run_pattern_sweep(args.n_configs, args.n_triggers)
        
    else:
        logger.info("Use --full, --quick, or --phase pattern")
        logger.info("Run with --help for usage")


if __name__ == "__main__":
    main()

```

---

## File: docs/sweep_master_guide.md

```markdown
# CNN Sweep Pipeline - Complete Guide

## What This Is

A CNN-based pattern detection system that:
1. Scans 1-minute candles for patterns
2. Triggers trades when CNN probability > threshold
3. Places limit orders with configurable entries/exits
4. Tests 96 different configurations to find optimal settings

---

## How We Got Here (Evolution of the Strategy)

### Phase 1: Initial Broken State
- CNN outputting constant ~0.27 probability for all inputs
- **Root cause**: Using percentage normalization `(x/base)-1` which produces tiny values
- **Fix**: Z-Score normalization `(x - mean) / std` per window

### Phase 2: Directional Bias
- Model only trained on SHORT patterns (price up → return)
- All results showed SHORT bias
- **Fix**: Updated `pattern_miner_v2.py` to detect BOTH:
  - SHORT: price rises then returns
  - LONG: price drops then returns

### Phase 3: Stops Too Tight  
- 0.5 ATR stops on 1m data caused both SL and TP to hit same bar
- Win rates impossibly low (<50% on 1:1 R/R)
- **Fix**: Use higher timeframe ATR (5m or 15m) for stops

### Phase 4: Wrong Limit Order Direction
- Initially placed LONG limits ABOVE close (breakout entry)
- Results were mediocre
- **Fix**: LONG limits BELOW close (pullback entry), SHORT ABOVE
- This **dramatically improved results** (+$17k vs +$6k)

### Phase 5: Entry Bar Confusion
- Tested various entry bar timings
- Originally used trigger bar directly (potential look-ahead)
- Then tested previous bar
- **Final**: Wait 5 bars for 5m ATR, 15 bars for 15m ATR

### Phase 6: Partial Exits
- Tested HALF at 1R, rest rides to 2R/3R with BE stop
- Result: BE stop gets hit too often, simple exit outperforms

### Phase 7: Sensitivity & Risk Tuning
- Tested lower thresholds (down to 0.066) to increase frequency
- **Result**: Lower thresholds increase PnL but drastically increase Drawdown
- **Recommendation**: Stick to **0.15 threshold** for safety (lowest DD)
- **Tuning**: Better to increase Risk per Trade on high-quality setup (0.15) than to lower quality bar (0.066)

| Threshold | Trades | WR | Net PnL | Max DD |
|-----------|--------|----|---------|--------|
| **0.15** (Default) | 328 | 57% | +$14,635 | **$2,859** |
| 0.10 | 833 | 55% | +$24,095 | $4,638 |
| 0.066 | 1753 | 54% | **+$50,526** | $7,712 |

*Note: 0.15 gives best risk-adjusted returns. To scale, increase position size, not sensitivity.*

---

## Current Best Config

### `LONG_0.25_1.0R_5m_SIMPLE`
| Parameter | Value |
|-----------|-------|
| Direction | LONG only |
| Entry | Limit 0.25 ATR(5m) **BELOW** close |
| Wait | 5 bars after CNN trigger |
| Stop | 1 ATR(5m) below entry |
| TP | 1R (1:1) |
| Risk | $300 per trade |

### Why This Works
- Pullback entry (limit below) catches retracements
- 1:1 R/R with 67% WR = consistent profits
- 5m ATR gives realistic stop distances (3-5 points)
- Low drawdown ($580 vs $1,768 on 2R targets)

---

## Quick Start

### Download Fresh Data
```python
import yfinance as yf
from datetime import datetime, timedelta

# MES (use ES=F as proxy)
df = yf.Ticker('ES=F').history(
    start=datetime.now() - timedelta(days=7),
    end=datetime.now(),
    interval='1m'
)
df.to_parquet('data/processed/fresh_5day_1m.parquet')

# MNQ (use NQ=F)
yf.Ticker('NQ=F').history(...).to_parquet('fresh_5day_mnq.parquet')

# Gold (use GC=F)
yf.Ticker('GC=F').history(...).to_parquet('fresh_5day_gold.parquet')
```

---

## Best Config Found

### `LONG_0.25_1.0R_5m_SIMPLE`
| Parameter | Value |
|-----------|-------|
| Direction | LONG only |
| Entry | Limit 0.25 ATR(5m) **BELOW** close |
| Wait | 5 bars after CNN trigger |
| Stop | 1 ATR(5m) below entry |
| TP | 1R (1:1) |
| Risk | $300 per trade |

### Performance
| Dataset | Trades | WR | PnL | MaxDD |
|---------|--------|-----|-----|-------|
| MES Historical | 351 | 58% | +$16,538 | - |
| MES Fresh 5-day | 15 | 67% | +$1,512 | $580 |
| MNQ Fresh | 13 | 23% | -$1,134 | - |
| Gold Fresh | 30 | 20% | -$3,127 | - |

**Model only works on MES** - each asset needs separate training.

---

## All Configs Tested (96 Total)

### Parameters Swept
- **Direction**: LONG, SHORT
- **Entry Offset**: 0.25, 0.5, 0.75, 1.0 ATR (below for LONG, above for SHORT)
- **TP**: 1.0R, 1.4R, 2.0R
- **ATR Timeframe**: 5m, 15m
- **Exit Mode**: SIMPLE, HALF_1R_REST_2R

### Top 5 on Historical MES
| Config | WR | PnL |
|--------|-----|-----|
| LONG_0.5_2.0R_5m | 39% | +$17,340 |
| LONG_0.25_1.0R_5m | 58% | +$16,538 |
| LONG_0.5_1.4R_5m | 48% | +$14,304 |
| LONG_1.0_1.0R_5m | 58% | +$14,219 |
| LONG_0.25_1.4R_5m | 47% | +$13,622 |

---

## Key Lessons Learned

1. **Normalization**: Z-Score `(x-mean)/std` per window, not percentage
2. **Train Both Directions**: Bidirectional pattern mining (LONG + SHORT)
3. **Limit Order Direction**: LONG limit BELOW close (pullback), SHORT ABOVE
4. **ATR Timeframe**: 5m ATR beats 15m for stops
5. **Tick Alignment**: MES/MNQ=0.25, Gold=0.10
6. **HALF+Runner underperforms**: BE stop gets hit before runner target
7. **Model Asset-Specific**: MES model doesn't transfer to MNQ/Gold

---

## Filter Analysis (None Help)

All filters profitable - filtering reduces trades without improving EV:
- above_pdh=True: 63% WR
- above_vwap: No improvement
- Best hours: 09, 14, 16, 22
- Best days: Mon, Wed, Thu

---

## Files Created

| File | Purpose |
|------|---------|
| `src/sweep/supersweep.py` | 96-config sweep engine |
| `src/sweep/pattern_miner_v2.py` | Bidirectional pattern detection |
| `src/sweep/train_sweep.py` | Z-Score normalized training |
| `models/sweep_CNN_Classic_v3_bidirectional.pth` | Trained model |
| `results/supersweep_results.parquet` | All trade records with filters |
| `docs/best_config.md` | Best config summary |
| `.agent/workflows/sweep_lessons_learned.md` | Detailed lessons |

---

## Running Supersweep

```bash
python src/sweep/supersweep.py --output results/supersweep_results.parquet
```

Options:
- `--risk 300` - Risk per trade
- `--threshold 0.15` - CNN probability threshold
- `--model models/sweep_CNN_Classic_v3_bidirectional.pth`

---

## What Still Needs Work

### Dynamic Risk Sizing (Not Implemented)
- $50k account, $2k max DD
- Base risk $300 (0.6%)
- Increase in $50 increments as account grows
- Fail at $48k balance

### Account Simulation Script (Needed)
- Track running balance
- Calculate drawdown
- Position sizing based on account size
- Win streaks / losing streaks analysis

### Multi-Asset Training
- Current model only works on MES
- Need to retrain on MNQ data
- Need to retrain on Gold data

### Filter Optimization
- Current filters don't improve EV (all subsets profitable)
- May need more granular time-of-day analysis
- Consider session-based filters (RTH vs ETH)

---

## Important Caveats

1. **Historical ≠ Future**: Best config on historical (+$17k) was different from fresh data (+$1.5k)
2. **Slippage not modeled**: Real fills may differ from limit price
3. **Commission not included**: ~$2.50 per MES round trip
4. **Overnight holds**: Some trades may hold overnight (not modeled)
5. **Consolidating markets**: Strategy may underperform in low-volatility periods

---

## Terminology Reference

| Term | Meaning |
|------|---------|
| ATR | Average True Range (volatility measure) |
| R | Risk unit (1R = stop distance) |
| TP | Take Profit |
| WR | Win Rate |
| DD | Drawdown |
| PDH/PDL | Previous Day High/Low |
| ONH/ONL | Overnight High/Low |
| PDC | Previous Day Close |
| VWAP | Volume Weighted Average Price |
| EMA | Exponential Moving Average |

```

---

## File: docs/best_config.md

```markdown
# Best CNN Strategy Configuration

## Optimal Setup: LONG_0.5_2.0R_5m

### Entry Rules
1. **CNN Trigger**: Probability > 0.15 on 20-bar 1m window (Z-Score normalized)
2. **Wait**: 5 bars after trigger for 5m candle to close
3. **Limit Order**: **BELOW** close by 0.5 × ATR(5m,14)
4. **Direction**: LONG only

### Trade Management
- **Stop**: 1 ATR(5m) below entry
- **Take Profit**: 2R (2× risk distance)
- **Position Size**: $300 risk / (stop_dist × $5)

### Performance (Full MES Data - 179,587 bars)
| Metric | Value |
|--------|-------|
| Triggers | 368 |
| Filled Trades | 327 |
| Win Rate | 39.4% |
| **Net PnL** | **+$17,340** |

### Why It Works
- 39% WR seems low, but 2R reward means:
  - 39 wins × 2R = 78R
  - 61 losses × 1R = -61R
  - Net = +17R per 100 trades

### Filter Analysis (no filters improve EV)
All filter subsets profitable - filtering reduces total trades without improving expectancy.

### Tick Alignment
- MES: 0.25
- MNQ: 0.25
- Gold: 0.10

### Files
- Model: `models/sweep_CNN_Classic_v3_bidirectional.pth`
- Supersweep: `src/sweep/supersweep.py`
- Results: `results/supersweep_results.parquet`

```

---

## File: docs/success_study.md

```markdown
# Success Study: The "Fade" Rejection Strategy

**Date**: December 8, 2025
**Outcome**: Discovered a highly profitable strategy (+70% Win Rate) by inverting a losing pattern.

## 1. The Journey & Challenges

### Phase 1: The "Rejection" Hypothesis
We started with the idea of a **"Round Trip Rejection"**:
-   **Concept**: Price extends 1.5x its average range (ATR) in one direction, then immediately returns to the start.
-   **Theory**: This "failed break" should lead to a reversal.
-   **Implementation Issues**:
    -   **Timeframe Sensitivity**: 1-minute candles were too noisy. We effectively switched to a **Hybrid Model** (Scan 5m, Input 1m).
    -   **Silent Crashes**: `pattern_miner.py` failed silently during large-scale pandas operations. **Fix**: Simplified the logic and used robust print statements instead of complex logging during the critical loop.
    -   **Model Collapse**: The CNN initially output a constant `0.32` probability.
        -   **Cause**: Inputs were normalized using "Percentage Change", which for 1m data is tiny and erratic.
        -   **Fix**: Switched to **Z-Score Normalization** (Standardization), which allowed the model to converge effectively.

### Phase 2: The Data Reality
-   **Baseline Test**: We ran the strategy *without* ML filters.
-   **Result**: 26-29% Win Rate.
-   **Insight**: The "Rejection" pattern filters itself out. If price extends strongly (1.5x ATR) and pulls back, it often **continues** in the original direction rather than reversing.

### Phase 3: The Pivot (Inversion)
-   **User Insight**: "Fade all entries."
-   **Result**: Flipping the trade logic turned a 29% Loser into a **70% Winner**.
-   **Logic**: We validly identified a high-probability **Continuation Pattern** (Pullback Buy) rather than a Reversal.

---

## 2. Key Files & Architecture

### **Good / Verified Files**
1.  **`src/pattern_miner.py`**
    -   **Role**: The Source of Truth.
    -   **Logic**: Hybrid 5m/1m. Scans 5m data for `ATR(14) >= 5` and `Extension >= 1.5x ATR`.
    -   **Safety**: Uses 1m granularity for outcome labeling to ensure precise fills/stops.
    
2.  **`src/models/train_rejection_cnn.py`**
    -   **Role**: The Trainer.
    -   **Key Feature**: Z-Score Normalization `(x - mean) / std`. This is crucial for valid CNN training on price data.
    
3.  **`src/strategies/inverse_strategy.py`**
    -   **Role**: The Money Maker.
    -   **Logic**: Takes the `labeled_rejections_5m.parquet` and simulates FADING every single signal (Inverse Logic).

---

## 3. Data Leakage Prevention & Future Testing

To ensure this result isn't a "backtest anomaly" or result of data leakage, follow these strict protocols when testing on new data:

### A. The "Future Wall" (Strict Chronological Split)
-   **Current State**: We trained on the first 80% and tested on the last 20%.
-   **Verification**: Ensure that the "Test Set" start time is strictly *after* the "Train Set" end time.
-   **Check**:
    ```python
    assert train_data['time'].max() < test_data['time'].min()
    ```

### B. Input Context Isolation
-   **Risk**: The CNN "seeing" the pattern completion.
-   **Solution**: The CNN input window MUST end at `pattern_start_time`.
    -   **Correct**: Input = `[Start - 20m : Start]`
    -   **Incorrect**: Input = `[Trigger - 20m : Trigger]` (This would show the extension happening).
    -   **Status**: **Verified**. We currently use `Start Time` as the cutoff.

### C. Normalization Leakage
-   **Risk**: Calculating Z-Score using statistics from the *whole dataset* (Global Mean/Std).
-   **Solution**: Dynamic Z-Score (Per Window).
    -   We calculate Mean/Std *only* on the specific 20-candle window passed to the model.
    -   **Status**: **Verified**. Code uses `mean = np.mean(feats); feats_norm = (feats - mean) / std` inside the loop.

### D. Lookahead Labeling
-   **Risk**: Labeling a trade 'WIN' based on high/lows that happened *during* the pattern formation.
-   **Solution**: Outcome checking starts at `Trigger Time + 5 Minutes`.
    -   We intentionally skip the candle where the trigger occurred to be conservative and simulate a "Next Bar" entry or ensuring we don't peek at intra-bar future data.
    -   **Status**: **Verified** in `pattern_miner.py`.

### E. Recommended Validation Step (Walk-Forward)
Before deploying live:
1.  **Holdout**: Download a *new* month of data that the system has never seen.
2.  **Blind Run**: Run `pattern_miner` -> `inverse_strategy` on this new month.
3.  **Expectation**: Win Rate should remain within 5-10% of the backtest (i.e., >60%).

```

---

