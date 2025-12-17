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
