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
