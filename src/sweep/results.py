"""
Sweep Results Aggregation
Collects, ranks, and exports sweep results.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("sweep_results")


@dataclass
class SweepResults:
    """
    Aggregates and ranks results from all sweep phases.
    """
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir: Path = field(default_factory=lambda: PROCESSED_DIR / "sweep_results")
    
    pattern_results: List[Dict] = field(default_factory=list)
    model_results: List[Dict] = field(default_factory=list)
    oco_results: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_pattern_result(
        self,
        config_id: str,
        config: Dict,
        stats: Dict,
    ):
        """Add a pattern mining result."""
        self.pattern_results.append({
            "config_id": config_id,
            **config,
            "win_rate": stats.get("win_rate", 0),
            "total_patterns": stats.get("total_patterns", 0),
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
            "timestamp": datetime.now().isoformat(),
        })
    
    def add_model_result(
        self,
        config_id: str,
        pattern_id: str,
        architecture: str,
        metrics: Dict,
    ):
        """Add a model training result."""
        self.model_results.append({
            "config_id": config_id,
            "pattern_id": pattern_id,
            "architecture": architecture,
            "test_acc": metrics.get("test_acc", 0),
            "best_val_acc": metrics.get("best_val_acc", 0),
            "model_path": metrics.get("model_path", ""),
            "timestamp": datetime.now().isoformat(),
        })
    
    def add_oco_result(
        self,
        model_id: str,
        oco_config: Dict,
        metrics: Dict,
    ):
        """Add an OCO backtest result."""
        self.oco_results.append({
            "model_id": model_id,
            **oco_config,
            "win_rate": metrics.get("win_rate", 0),
            "total_pnl": metrics.get("total_pnl", 0),
            "expected_value": metrics.get("expected_value", 0),
            "total_trades": metrics.get("total_trades", 0),
            "timestamp": datetime.now().isoformat(),
        })
    
    def get_top_patterns(self, k: int = 10) -> pd.DataFrame:
        """Get top K patterns by win rate."""
        if not self.pattern_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.pattern_results)
        df = df[df['total_patterns'] > 5]  # Filter low-sample configs
        return df.sort_values("win_rate", ascending=False).head(k)
    
    def get_top_models(self, k: int = 5) -> pd.DataFrame:
        """Get top K models by test accuracy."""
        if not self.model_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.model_results)
        return df.sort_values("test_acc", ascending=False).head(k)
    
    def get_oco_summary(self) -> pd.DataFrame:
        """Get OCO results summary."""
        if not self.oco_results:
            return pd.DataFrame()
        
        return pd.DataFrame(self.oco_results)
    
    def export_summary(self, path: Path = None) -> Path:
        """Export full summary to JSON and CSV."""
        if path is None:
            path = self.output_dir / f"sweep_summary_{self.run_id}"
        
        # JSON export
        json_path = path.with_suffix(".json")
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "pattern_results": self.pattern_results,
            "model_results": self.model_results,
            "oco_results": self.oco_results,
            "summary": {
                "total_patterns_tested": len(self.pattern_results),
                "total_models_trained": len(self.model_results),
                "total_oco_backtests": len(self.oco_results),
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # CSV exports
        if self.pattern_results:
            pd.DataFrame(self.pattern_results).to_csv(
                path.parent / f"patterns_{self.run_id}.csv", index=False
            )
        
        if self.model_results:
            pd.DataFrame(self.model_results).to_csv(
                path.parent / f"models_{self.run_id}.csv", index=False
            )
        
        if self.oco_results:
            pd.DataFrame(self.oco_results).to_csv(
                path.parent / f"oco_{self.run_id}.csv", index=False
            )
        
        logger.info(f"Results exported to {path.parent}")
        return json_path
    
    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("SWEEP RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"\nRun ID: {self.run_id}")
        print(f"Patterns tested: {len(self.pattern_results)}")
        print(f"Models trained: {len(self.model_results)}")
        print(f"OCO backtests: {len(self.oco_results)}")
        
        # Top patterns
        top_patterns = self.get_top_patterns(5)
        if not top_patterns.empty:
            print("\nTOP 5 PATTERNS:")
            print("-" * 40)
            for _, row in top_patterns.iterrows():
                print(f"  {row['config_id']}: "
                      f"Win Rate={row['win_rate']*100:.1f}%, "
                      f"Samples={row['total_patterns']}")
        
        # Top models
        top_models = self.get_top_models(3)
        if not top_models.empty:
            print("\nTOP 3 MODELS:")
            print("-" * 40)
            for _, row in top_models.iterrows():
                print(f"  {row['architecture']} on {row['pattern_id']}: "
                      f"Test Acc={row['test_acc']:.3f}")
        
        # Best OCO
        oco_df = self.get_oco_summary()
        if not oco_df.empty:
            print("\nBEST OCO CONFIGURATIONS:")
            print("-" * 40)
            top_oco = oco_df.sort_values("expected_value", ascending=False).head(3)
            for _, row in top_oco.iterrows():
                print(f"  {row.get('oco_config', 'N/A')}: "
                      f"EV=${row.get('expected_value', 0):.2f}, "
                      f"Win Rate={row.get('win_rate', 0)*100:.1f}%")
        
        print("\n" + "=" * 60)


def load_results(run_id: str, output_dir: Path = None) -> SweepResults:
    """Load previous sweep results from disk."""
    if output_dir is None:
        output_dir = PROCESSED_DIR / "sweep_results"
    
    json_path = output_dir / f"sweep_summary_{run_id}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Results not found: {json_path}")
    
    with open(json_path) as f:
        data = json.load(f)
    
    results = SweepResults(run_id=run_id, output_dir=output_dir)
    results.pattern_results = data.get("pattern_results", [])
    results.model_results = data.get("model_results", [])
    results.oco_results = data.get("oco_results", [])
    
    return results


if __name__ == "__main__":
    # Quick test
    results = SweepResults()
    
    # Add dummy data
    results.add_pattern_result(
        "pattern_001",
        {"rise_min": 2.5, "rise_max": 4.0},
        {"win_rate": 0.42, "total_patterns": 100}
    )
    
    results.add_model_result(
        "model_001",
        "pattern_001",
        "CNN_Classic",
        {"test_acc": 0.55}
    )
    
    results.add_oco_result(
        "model_001",
        {"direction": "SHORT", "r_multiple": 1.4},
        {"win_rate": 0.42, "total_pnl": -500, "expected_value": -12.5}
    )
    
    results.print_summary()
