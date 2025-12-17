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
