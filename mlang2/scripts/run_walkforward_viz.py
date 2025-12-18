#!/usr/bin/env python
"""
Walk-Forward Viz Export CLI
Run walk-forward experiments and export visualization artifacts.

Usage:
    python scripts/run_walkforward_viz.py --config experiment.json --out results/viz/my_run/
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment
from src.experiments.splits import generate_walk_forward_splits, WalkForwardConfig, Split
from src.viz.export import Exporter
from src.viz.config import VizConfig
from src.data.loader import load_continuous_contract
from src.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward experiment with viz export")
    parser.add_argument("--config", type=str, help="Path to experiment config JSON")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: auto-generate)")
    parser.add_argument("--include-full-series", action="store_true", help="Include full OHLCV series")
    parser.add_argument("--no-windows", action="store_true", help="Exclude price windows")
    
    # Quick-run params (if no config file)
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--end-date", type=str, default="2025-05-04", help="End date")
    parser.add_argument("--scanner", type=str, default="interval", help="Scanner ID")
    parser.add_argument("--train-weeks", type=int, default=3, help="Train weeks per split")
    parser.add_argument("--test-weeks", type=int, default=1, help="Test weeks per split")
    
    args = parser.parse_args()
    
    # Load or create experiment config
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = ExperimentConfig(**config_dict)
    else:
        # Use CLI params for quick run
        config = ExperimentConfig(
            name="walkforward_viz",
            start_date=args.start_date,
            end_date=args.end_date,
            scanner_id=args.scanner,
        )
    
    # Setup viz config
    viz_config = VizConfig(
        include_full_series=args.include_full_series,
        include_windows=not args.no_windows,
    )
    
    # Setup output directory
    run_id = args.run_id or config.name
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "viz" / run_id
    
    # Create exporter
    exporter = Exporter(
        config=viz_config,
        run_id=run_id,
        experiment_config=config.to_dict() if hasattr(config, 'to_dict') else {},
    )
    
    print("=" * 60)
    print(f"Walk-Forward Viz Export")
    print(f"Run ID: {run_id}")
    print(f"Output: {out_dir}")
    print("=" * 60)
    
    # Run experiment with exporter
    result = run_experiment(config, exporter=exporter)
    
    print(f"\nExperiment complete:")
    print(f"  Total records: {result.total_records}")
    print(f"  Win: {result.win_records}, Loss: {result.loss_records}")
    
    # Finalize export
    exporter.finalize(out_dir)
    
    print(f"\nViz artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
