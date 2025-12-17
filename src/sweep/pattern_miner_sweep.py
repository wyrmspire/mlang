"""
CLI Pattern Miner for Sweep Pipeline
Tests pattern geometries AND 10 OCO configurations per pattern.
Outputs ranked (geometry + OCO) combinations for selection.

Usage:
    python src/sweep/pattern_miner_sweep.py \
        --rise-min 2.5 --rise-max 4.0 \
        --min-drop 1.0 --atr-buffer 0.2 \
        --max-triggers 30 \
        --output-suffix "sweep_001"
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, ONE_MIN_PARQUET_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("pattern_miner_sweep")

# 10 OCO configurations to test per pattern
OCO_CONFIGS = [
    {"name": "SHORT_1.0R_50ATR", "direction": "SHORT", "r_mult": 1.0, "stop_atr": 0.5},
    {"name": "SHORT_1.4R_50ATR", "direction": "SHORT", "r_mult": 1.4, "stop_atr": 0.5},
    {"name": "SHORT_2.0R_50ATR", "direction": "SHORT", "r_mult": 2.0, "stop_atr": 0.5},
    {"name": "SHORT_1.4R_25ATR", "direction": "SHORT", "r_mult": 1.4, "stop_atr": 0.25},
    {"name": "LONG_1.0R_50ATR", "direction": "LONG", "r_mult": 1.0, "stop_atr": 0.5},
    {"name": "LONG_1.4R_50ATR", "direction": "LONG", "r_mult": 1.4, "stop_atr": 0.5},
    {"name": "LONG_2.0R_50ATR", "direction": "LONG", "r_mult": 2.0, "stop_atr": 0.5},
    {"name": "LONG_1.4R_25ATR", "direction": "LONG", "r_mult": 1.4, "stop_atr": 0.25},
    {"name": "SHORT_1.8R_75ATR", "direction": "SHORT", "r_mult": 1.8, "stop_atr": 0.75},
    {"name": "LONG_1.8R_75ATR", "direction": "LONG", "r_mult": 1.8, "stop_atr": 0.75},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Pattern Miner with 10 OCO configs")
    
    # Pattern geometry
    parser.add_argument("--rise-min", type=float, default=2.5)
    parser.add_argument("--rise-max", type=float, default=4.0)
    parser.add_argument("--min-drop", type=float, default=1.0)
    parser.add_argument("--atr-buffer", type=float, default=0.2)
    parser.add_argument("--lookback", type=int, default=120)
    
    # Output
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-triggers", type=int, default=30)
    
    return parser.parse_args()


def simulate_oco(df_1m, trigger_idx, entry_price, atr, oco_config, 
                 highs_3m, lows_slice):
    """Simulate single OCO outcome for a pattern."""
    direction = oco_config["direction"]
    r_mult = oco_config["r_mult"]
    stop_atr = oco_config["stop_atr"]
    
    if direction == "SHORT":
        stop_dist = atr * stop_atr
        stop_price = entry_price + stop_dist
        tp_price = entry_price - (stop_dist * r_mult)
        
        future = df_1m.iloc[trigger_idx+1:trigger_idx+2001]
        if len(future) == 0:
            return "TIMEOUT", 0
        
        sl_hit = future[future['high'] >= stop_price]
        tp_hit = future[future['low'] <= tp_price]
    else:  # LONG
        stop_dist = atr * stop_atr
        stop_price = entry_price - stop_dist
        tp_price = entry_price + (stop_dist * r_mult)
        
        future = df_1m.iloc[trigger_idx+1:trigger_idx+2001]
        if len(future) == 0:
            return "TIMEOUT", 0
        
        sl_hit = future[future['low'] <= stop_price]
        tp_hit = future[future['high'] >= tp_price]
    
    sl_idx = sl_hit.index[0] if not sl_hit.empty else 999999999
    tp_idx = tp_hit.index[0] if not tp_hit.empty else 999999999
    
    if sl_idx == 999999999 and tp_idx == 999999999:
        return "TIMEOUT", 0
    elif tp_idx < sl_idx:
        return "WIN", r_mult  # Win pays R-multiple
    else:
        return "LOSS", -1.0   # Loss is 1R


def mine_patterns_multi_oco(args):
    """
    Mine patterns and test ALL 10 OCO configs for each pattern.
    Returns stats for each OCO configuration.
    """
    logger.info(f"Mining patterns: rise={args.rise_min}-{args.rise_max}")
    
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
    
    # ATR calculation
    df_3m = df_1m.set_index('time').resample('3T').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna().reset_index()
    
    df_3m['tr'] = np.maximum(
        df_3m['high'] - df_3m['low'],
        np.maximum(
            abs(df_3m['high'] - df_3m['close'].shift(1)),
            abs(df_3m['low'] - df_3m['close'].shift(1))
        )
    )
    df_3m['atr'] = df_3m['tr'].rolling(14).mean()
    df_atr = df_3m[['time', 'atr', 'high', 'low']].rename(
        columns={'high': '3m_high', 'low': '3m_low'})
    df_1m = pd.merge_asof(df_1m, df_atr, on='time', direction='backward')
    
    # Arrays for speed
    closes = df_1m['close'].values
    times = df_1m['time'].values
    highs = df_1m['high'].values
    lows = df_1m['low'].values
    atrs = df_1m['atr'].values
    n = len(df_1m)
    
    # Track results for each OCO config
    oco_results = {cfg["name"]: {"wins": 0, "losses": 0, "pnl": 0.0} 
                   for cfg in OCO_CONFIGS}
    
    pattern_count = 0
    patterns_data = []
    last_trade_time = times[0] - np.timedelta64(1, 'D')
    lookback = args.lookback
    max_triggers = args.max_triggers if args.max_triggers > 0 else float('inf')
    
    logger.info(f"Scanning {n} bars for up to {max_triggers} patterns...")
    
    for i in range(200, n):
        if pattern_count >= max_triggers:
            break
        
        curr_time = times[i]
        curr_price = closes[i]
        
        if (curr_time - last_trade_time) < np.timedelta64(15, 'm'):
            continue
        
        start_scan = max(0, i - lookback)
        
        for j in range(i - 1, start_scan, -1):
            start_price = closes[j]
            drop_size = start_price - curr_price
            
            if drop_size < args.min_drop:
                continue
            
            peak_price = np.max(highs[j:i])
            rise_size = peak_price - start_price
            
            if rise_size <= 0:
                continue
            
            ratio = rise_size / drop_size
            
            if args.rise_min <= ratio < args.rise_max:
                if pd.isna(atrs[j]):
                    continue
                
                atr = atrs[j]
                entry_price = curr_price
                
                # Store pattern data for later training
                pattern_info = {
                    'trigger_idx': i,
                    'trigger_time': curr_time,
                    'entry': entry_price,
                    'atr': atr,
                    'ratio': ratio,
                }
                
                # Test ALL 10 OCO configs for this pattern
                for oco_cfg in OCO_CONFIGS:
                    outcome, pnl_r = simulate_oco(
                        df_1m, i, entry_price, atr, oco_cfg, 
                        highs[j:i], lows[j:i]
                    )
                    
                    if outcome == "WIN":
                        oco_results[oco_cfg["name"]]["wins"] += 1
                        oco_results[oco_cfg["name"]]["pnl"] += pnl_r
                    elif outcome == "LOSS":
                        oco_results[oco_cfg["name"]]["losses"] += 1
                        oco_results[oco_cfg["name"]]["pnl"] += pnl_r
                    
                    # Store outcome for this OCO
                    pattern_info[f"outcome_{oco_cfg['name']}"] = outcome
                
                patterns_data.append(pattern_info)
                pattern_count += 1
                last_trade_time = curr_time
                break
    
    # Calculate stats for each OCO config
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
            "stop_atr": cfg["stop_atr"],
            "wins": data["wins"],
            "losses": data["losses"],
            "total": total,
            "win_rate": round(win_rate, 4),
            "total_pnl_r": round(data["pnl"], 2),
            "ev_per_trade": round(data["pnl"] / total, 3) if total > 0 else 0,
        })
    
    # Sort by EV per trade
    oco_stats = sorted(oco_stats, key=lambda x: x["ev_per_trade"], reverse=True)
    
    summary = {
        "pattern_geometry": {
            "rise_min": args.rise_min,
            "rise_max": args.rise_max,
            "min_drop": args.min_drop,
            "atr_buffer": args.atr_buffer,
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
    result = mine_patterns_multi_oco(args)
    
    if "error" in result:
        print(json.dumps(result))
        return
    
    summary = result["summary"]
    patterns_df = result["patterns"]
    
    logger.info("=" * 60)
    logger.info(f"Mining Complete! Found {summary['total_patterns']} patterns")
    logger.info("OCO Results (sorted by EV):")
    logger.info("-" * 60)
    
    for oco in summary["oco_results"][:5]:  # Top 5
        logger.info(f"  {oco['oco_config']}: "
                    f"WR={oco['win_rate']*100:.1f}%, "
                    f"EV={oco['ev_per_trade']:.3f}R, "
                    f"PnL={oco['total_pnl_r']:.1f}R")
    
    logger.info("=" * 60)
    
    if not args.dry_run and len(patterns_df) > 0:
        suffix = args.output_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save patterns
        out_path = PROCESSED_DIR / f"labeled_sweep_{suffix}.parquet"
        patterns_df.to_parquet(out_path)
        
        # Save summary
        stats_path = PROCESSED_DIR / f"sweep_stats_{suffix}.json"
        with open(stats_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved to {out_path}")
    
    # Output JSON for orchestrator
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
