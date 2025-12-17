"""
Continuous 5m Scanner with CNN Filter (V2)
Based on smart_cnn.py working approach:
1. Scan every 5m candle close
2. Get 20 1m candles BEFORE
3. Test BOTH directions: normal features for LONG, inverted for SHORT
4. Trigger trade if prob > threshold
5. Test 10 OCO configs per trigger

Usage:
    python src/sweep/continuous_scanner.py \
        --model models/sweep_CNN_Classic_v2_unified.pth \
        --threshold 0.6 \
        --output results/scanner_results.csv
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, ONE_MIN_PARQUET_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger
from src.models.variants import CNN_Classic

logger = get_logger("scanner")

# GPU enforcement
if not torch.cuda.is_available():
    logger.error("GPU NOT DETECTED!")
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
    parser = argparse.ArgumentParser(description="5m Continuous Scanner V2")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained CNN model")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="CNN probability threshold for triggering")
    parser.add_argument("--window", type=int, default=20,
                        help="Number of 1m candles for CNN input")
    parser.add_argument("--test-ratio", type=float, default=0.3,
                        help="Last X% of data to use as test period")
    parser.add_argument("--output", type=str, default="",
                        help="Output CSV path")
    parser.add_argument("--risk-per-trade", type=float, default=75.0)
    
    return parser.parse_args()


def load_model(model_path: str, window_size: int):
    """Load trained CNN model."""
    model = CNN_Classic(input_dim=4, seq_len=window_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model


def get_predictions(model, window_data: np.ndarray):
    """
    Get CNN probabilities for BOTH directions.
    LONG: normal features
    SHORT: inverted features (per smart_cnn.py pattern)
    Uses Z-Score normalization (per success_study.md)
    """
    # Z-Score Normalization per window
    mean = np.mean(window_data)
    std = np.std(window_data)
    if std == 0:
        std = 1.0
    feats_norm = (window_data - mean) / std
    
    # Input for LONG (normal)
    input_long = torch.FloatTensor(feats_norm).unsqueeze(0).to(device)
    # Input for SHORT (inverted)
    input_short = torch.FloatTensor(-feats_norm).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prob_long = model(input_long).item()
        prob_short = model(input_short).item()
    
    return prob_long, prob_short


def simulate_oco(df_1m, trigger_idx, entry_price, atr, oco_config, risk_per_trade):
    """
    Simulate single OCO. Stop = 0.5 ATR.
    Position sized so that stop = $risk_per_trade.
    PnL = +risk*R for win, -risk for loss.
    """
    direction = oco_config["direction"]
    r_mult = oco_config["r_mult"]
    
    stop_dist = atr * 0.5  # Points
    
    if direction == "LONG":
        stop_price = entry_price - stop_dist
        tp_price = entry_price + (stop_dist * r_mult)
        
        future = df_1m.iloc[trigger_idx+1:trigger_idx+2001]
        if len(future) == 0:
            return "TIMEOUT", 0
        
        sl_hit = future[future['low'] <= stop_price]
        tp_hit = future[future['high'] >= tp_price]
    else:  # SHORT
        stop_price = entry_price + stop_dist
        tp_price = entry_price - (stop_dist * r_mult)
        
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
        return "WIN", risk_per_trade * r_mult  # Position sized so win = $risk × R
    else:
        return "LOSS", -risk_per_trade  # Position sized so loss = -$risk


def run_scanner(args):
    """Scan every 5m candle close in test period."""
    logger.info("Loading data...")
    
    # Load 1m data
    data_path = PROCESSED_DIR / "continuous_1m.parquet"
    if not data_path.exists():
        data_path = ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet"
    
    df_1m = pd.read_parquet(data_path)
    
    if 'time' in df_1m.columns:
        df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
        df_1m = df_1m.set_index('time')
    df_1m = df_1m.sort_index().reset_index(drop=True)
    
    # Calculate ATR
    df_1m['tr'] = np.maximum(
        df_1m['high'] - df_1m['low'],
        np.maximum(
            abs(df_1m['high'] - df_1m['close'].shift(1)),
            abs(df_1m['low'] - df_1m['close'].shift(1))
        )
    )
    df_1m['atr'] = df_1m['tr'].rolling(14).mean()
    
    n = len(df_1m)
    test_start = int(n * (1 - args.test_ratio))
    
    logger.info(f"Total bars: {n}, Test period starts at: {test_start}")
    
    # Load model
    model = load_model(args.model, args.window)
    
    # Track OCO results
    oco_results = {cfg["name"]: {"trades": 0, "wins": 0, "pnl": 0.0} 
                   for cfg in OCO_CONFIGS}
    
    triggers = []
    last_trigger_idx = 0
    trigger_count = 0
    
    # Scan every 5 bars (5m close)
    for i in range(test_start + args.window, n - 100, 5):
        # Cooldown (15 bars = 15 min)
        if i - last_trigger_idx < 15:
            continue
        
        # Get 20 1m candles before this point
        window = df_1m.iloc[i - args.window:i][['open', 'high', 'low', 'close']].values
        
        if len(window) < args.window:
            continue
        
        # Get CNN probabilities for BOTH directions
        prob_long, prob_short = get_predictions(model, window)
        
        # Check if either direction triggers
        direction = None
        prob = 0
        
        if prob_long >= args.threshold and prob_long > prob_short:
            direction = "LONG"
            prob = prob_long
        elif prob_short >= args.threshold and prob_short > prob_long:
            direction = "SHORT"
            prob = prob_short
        
        if direction is None:
            continue
        
        # Trigger!
        trigger_count += 1
        entry_price = df_1m.iloc[i]['close']
        atr = df_1m.iloc[i]['atr']
        
        if pd.isna(atr) or atr <= 0:
            continue
        
        if trigger_count <= 10:
            logger.info(f"Trigger {trigger_count}: {direction} prob={prob:.3f}")
        
        trigger_info = {
            'trigger_idx': i,
            'entry': entry_price,
            'direction': direction,
            'prob': prob,
            'prob_long': prob_long,
            'prob_short': prob_short,
            'atr': atr,
        }
        
        # Test only the relevant direction OCO configs
        for oco_cfg in OCO_CONFIGS:
            if oco_cfg["direction"] == direction:
                outcome, pnl = simulate_oco(
                    df_1m, i, entry_price, atr, oco_cfg, args.risk_per_trade
                )
                
                oco_results[oco_cfg["name"]]["trades"] += 1
                if outcome == "WIN":
                    oco_results[oco_cfg["name"]]["wins"] += 1
                oco_results[oco_cfg["name"]]["pnl"] += pnl
                
                trigger_info[f"outcome_{oco_cfg['name']}"] = outcome
                trigger_info[f"pnl_{oco_cfg['name']}"] = pnl
        
        triggers.append(trigger_info)
        last_trigger_idx = i
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"SCAN COMPLETE - {len(triggers)} triggers at threshold {args.threshold}")
    logger.info("=" * 60)
    
    results = []
    for cfg in OCO_CONFIGS:
        name = cfg["name"]
        data = oco_results[name]
        trades = data["trades"]
        wins = data["wins"]
        win_rate = wins / trades if trades > 0 else 0
        pnl = data["pnl"]
        
        results.append({
            "oco_config": name,
            "direction": cfg["direction"],
            "r_mult": cfg["r_mult"],
            "trades": trades,
            "wins": wins,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(pnl, 2),
            "ev_per_trade": round(pnl / trades, 2) if trades > 0 else 0,
        })
        
        if trades > 0:
            logger.info(f"  {name}: {trades} trades, "
                        f"WR={win_rate*100:.1f}%, PnL=${pnl:.2f}")
    
    # Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        pd.DataFrame(results).to_csv(out_path, index=False)
        logger.info(f"Results saved to {out_path}")
        
        if triggers:
            triggers_path = out_path.parent / f"{out_path.stem}_triggers.parquet"
            pd.DataFrame(triggers).to_parquet(triggers_path)
    
    print(json.dumps(results, indent=2))
    return results


def main():
    args = parse_args()
    run_scanner(args)


if __name__ == "__main__":
    main()
