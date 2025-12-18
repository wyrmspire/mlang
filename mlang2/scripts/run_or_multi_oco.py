#!/usr/bin/env python
"""
Opening Range Multi-OCO Simulation with SMART STOPS
Uses actual OR levels as stops, not simple ATR offsets.

Usage:
    python scripts/run_or_multi_oco.py --start-date 2025-03-17 --weeks 3 --out results/or_multi_oco/
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features, FeatureConfig
from src.features.indicators import calculate_atr
from src.policy.library.opening_range import OpeningRangeScanner
from src.policy.oco_grid import get_or_oco_grid
from src.sim.stop_calculator import get_stop_for_or_retest, calculate_tp_from_risk
from src.labels.counterfactual import compute_smart_stop_counterfactual
from src.config import RESULTS_DIR


def make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Run Opening Range multi-OCO simulation with smart stops")
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--weeks", type=int, default=3, help="Number of weeks to simulate")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--lookback-1m", type=int, default=30, help="1m bars before entry for CNN")
    
    args = parser.parse_args()
    
    # Calculate end date
    start_date = pd.Timestamp(args.start_date)
    end_date = start_date + timedelta(weeks=args.weeks)
    
    # Setup output
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "or_multi_oco"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # R multiples to test for each stop
    TP_MULTIPLES = [1.0, 1.5, 2.0]
    
    print("=" * 60)
    print("Opening Range Multi-OCO Simulation (SMART STOPS)")
    print(f"Period: {start_date.date()} to {end_date.date()} ({args.weeks} weeks)")
    print(f"Stop: OR level + 0.25 ATR padding")
    print(f"R multiples: {TP_MULTIPLES}")
    print(f"Output: {out_dir}")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"Loaded {len(df)} 1m bars")
    
    # 2. Resample to higher timeframes
    print("\n[2] Resampling to higher timeframes...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    print(f"5m bars: {len(df_5m)}, 15m bars: {len(df_15m)}")
    
    # 3. Precompute ATR on 5m
    print("\n[3] Computing ATR...")
    df_5m_copy = df_5m.copy()
    df_5m_copy['atr'] = calculate_atr(df_5m_copy, 14)
    avg_atr = df_5m_copy['atr'].dropna().mean()
    print(f"Average 5m ATR: {avg_atr:.2f} points")
    
    # 4. Setup scanner and stepper
    print("\n[4] Running simulation...")
    scanner = OpeningRangeScanner(
        or_timeframe_minutes=15,
        retest_threshold_atr=0.25,
        cooldown_bars=30,
        direction="BOTH"
    )
    
    stepper = MarketStepper(df, start_idx=200, end_idx=len(df) - 200)
    feature_config = FeatureConfig(lookback_1m=args.lookback_1m)
    
    # 5. Collect decision points
    records = []
    decision_idx = 0
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        # Compute features
        features = compute_features(
            stepper,
            feature_config,
            df_5m=df_5m,
            df_15m=df_15m,
        )
        
        # Check scanner
        scan_result = scanner.scan(features.market_state, features)
        if not scan_result.triggered:
            continue
        
        # Get ATR at this point
        atr = features.atr if features.atr > 0 else avg_atr
        
        # Get direction and OR levels from scanner context
        direction = scan_result.context.get('direction', 'LONG')
        or_high = scan_result.context.get('or_high', 0)
        or_low = scan_result.context.get('or_low', 0)
        entry_price = features.current_price
        
        # === SMART STOP: Use OR level as stop ===
        stop_price, stop_reason = get_stop_for_or_retest(
            direction=direction,
            entry_price=entry_price,
            atr=atr,
            or_high=or_high,
            or_low=or_low,
            padding_atr=0.25
        )
        
        # Calculate risk for R sizing
        risk = abs(entry_price - stop_price)
        
        # Test multiple R multiples
        oco_results = {}
        for tp_mult in TP_MULTIPLES:
            oco_name = f"{direction}_OR_{tp_mult}R"
            
            cf = compute_smart_stop_counterfactual(
                df=df,
                entry_idx=step.bar_idx,
                direction=direction,
                stop_price=stop_price,
                tp_multiple=tp_mult,
                atr=atr,
                oco_name=oco_name
            )
            
            oco_results[oco_name] = {
                'outcome': cf.outcome,
                'pnl_dollars': cf.pnl_dollars,
                'pnl': cf.pnl,
                'mae': cf.mae,
                'mfe': cf.mfe,
                'bars_held': cf.bars_held,
                'stop_price': stop_price,
                'tp_price': cf.tp_price,
            }
        
        # Find best OCO
        best_oco = None
        best_pnl = float('-inf')
        for name, result in oco_results.items():
            if result['pnl_dollars'] > best_pnl:
                best_pnl = result['pnl_dollars']
                best_oco = name
        
        # Build CNN-ready record
        record = {
            'decision_id': f"or_{decision_idx:04d}",
            'timestamp': features.timestamp.isoformat() if features.timestamp else None,
            'bar_idx': step.bar_idx,
            'index': decision_idx,
            
            # Opening Range context
            'or_high': or_high,
            'or_low': or_low,
            'direction': direction,
            'retest_level': scan_result.context.get('retest_level', 0),
            
            # Stop info
            'stop_price': stop_price,
            'stop_reason': stop_reason,
            'risk_points': risk,
            
            # Market state
            'current_price': entry_price,
            'atr': atr,
            
            # CNN input: 30 1m bars before entry
            'x_price_1m': features.x_price_1m.tolist() if features.x_price_1m is not None else [],
            
            # Multi-OCO results
            'oco_results': oco_results,
            
            # Best OCO (CNN classification target)
            'best_oco': best_oco,
            'best_pnl': best_pnl,
        }
        
        records.append(record)
        decision_idx += 1
        
        # Progress
        if decision_idx % 5 == 0:
            print(f"  Triggers: {decision_idx}, Last: {direction} at {features.timestamp}, stop={stop_price:.2f}")
    
    print(f"\n[5] Completed! Total triggers: {len(records)}")
    
    # 6. Analyze results
    long_count = 0
    short_count = 0
    best_oco_counts = {}
    oco_pnl_totals = {}
    
    if records:
        print("\n[6] Analysis:")
        
        # Count by direction
        long_count = sum(1 for r in records if r['direction'] == 'LONG')
        short_count = sum(1 for r in records if r['direction'] == 'SHORT')
        print(f"  LONG triggers: {long_count}")
        print(f"  SHORT triggers: {short_count}")
        
        # Average risk
        avg_risk = sum(r['risk_points'] for r in records) / len(records)
        print(f"  Average risk (points): {avg_risk:.2f}")
        
        # Best OCO distribution
        for r in records:
            oco_name = r['best_oco']
            best_oco_counts[oco_name] = best_oco_counts.get(oco_name, 0) + 1
        
        print("\n  Best OCO distribution:")
        for name, count in sorted(best_oco_counts.items(), key=lambda x: -x[1]):
            print(f"    {name}: {count} ({count/len(records)*100:.1f}%)")
        
        # Total PnL by OCO
        print("\n  Total PnL by OCO:")
        for r in records:
            for oco_name, result in r['oco_results'].items():
                if oco_name not in oco_pnl_totals:
                    oco_pnl_totals[oco_name] = 0
                oco_pnl_totals[oco_name] += result['pnl_dollars']
        
        for name, total in sorted(oco_pnl_totals.items(), key=lambda x: -x[1]):
            print(f"    {name}: ${total:.2f}")
    
    # 7. Write output
    print(f"\n[7] Writing output to {out_dir}...")
    
    # Write JSONL
    output_path = out_dir / "or_multi_oco_records.jsonl"
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(make_serializable(r)) + '\n')
    print(f"  Wrote {len(records)} records to {output_path.name}")
    
    # Write summary
    summary = {
        'start_date': str(start_date.date()),
        'end_date': str(end_date.date()),
        'weeks': args.weeks,
        'stop_type': 'OR_LEVEL',
        'tp_multiples': TP_MULTIPLES,
        'total_triggers': len(records),
        'long_triggers': long_count,
        'short_triggers': short_count,
        'oco_pnl_totals': make_serializable(oco_pnl_totals),
        'best_oco_counts': best_oco_counts,
    }
    
    summary_path = out_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote summary to {summary_path.name}")
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
