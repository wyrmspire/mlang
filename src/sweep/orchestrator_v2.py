"""
Sweep Orchestrator V2 - Proportional Patterns
1. Sweep 100 ratio configs × 30 triggers × 10 OCO
2. Select top 10 combos
3. Mine full data (70%) with top configs
4. Train unified CNN on all patterns
5. Test with 5m scan + 10 OCO per trigger
"""

import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
import sys
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("orchestrator_v2")

# GPU check
if not torch.cuda.is_available():
    logger.error("GPU NOT DETECTED!")
    sys.exit(1)
logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


def generate_ratio_grid(n_configs: int = 100):
    """Generate diverse ratio configurations using Latin Hypercube."""
    from scipy.stats.qmc import LatinHypercube
    
    sampler = LatinHypercube(d=3, seed=42)
    samples = sampler.random(n=n_configs)
    
    configs = []
    for i, s in enumerate(samples):
        rise_ratio = 1.2 + s[0] * 3.0    # 1.2 to 4.2
        invalid_ratio = rise_ratio + 1.0 + s[1] * 3.0  # rise+1 to rise+4
        min_unit = 0.25 + s[2] * 0.75    # 0.25 to 1.0
        
        configs.append({
            "config_id": f"ratio_{i:03d}",
            "rise_ratio": round(rise_ratio, 2),
            "invalid_ratio": round(invalid_ratio, 2),
            "min_unit": round(min_unit, 2),
        })
    
    return configs


class SweepOrchestratorV2:
    """V2 Orchestrator with proportional patterns."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or PROCESSED_DIR / "sweep_v2_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.all_combos = []
        
    def run_ratio_sweep(self, n_configs: int = 100, n_triggers: int = 30):
        """Phase 1: Sweep ratio configurations."""
        logger.info(f"PHASE 1: Sweeping {n_configs} ratio configs × 10 OCO × {n_triggers} triggers")
        
        configs = generate_ratio_grid(n_configs)
        
        for i, cfg in enumerate(configs):
            logger.info(f"[{i+1}/{n_configs}] {cfg['config_id']}: rise={cfg['rise_ratio']}")
            
            cmd = [
                sys.executable, "-m", "src.sweep.pattern_miner_v2",
                "--rise-ratio", str(cfg["rise_ratio"]),
                "--invalid-ratio", str(cfg["invalid_ratio"]),
                "--min-unit", str(cfg["min_unit"]),
                "--max-triggers", str(n_triggers),
                "--output-suffix", f"{self.run_id}_{cfg['config_id']}",
            ]
            
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=str(Path(__file__).parents[2]), timeout=300,
                )
                
                for line in reversed(result.stdout.strip().split('\n')):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
                
                # Extract all (config, OCO) combos
                for oco in data.get("oco_results", []):
                    self.all_combos.append({
                        "config_id": cfg["config_id"],
                        "rise_ratio": cfg["rise_ratio"],
                        "invalid_ratio": cfg["invalid_ratio"],
                        "min_unit": cfg["min_unit"],
                        "oco_config": oco.get("oco_config"),
                        "direction": oco.get("direction"),
                        "r_mult": oco.get("r_mult"),
                        "win_rate": oco.get("win_rate", 0),
                        "ev_per_trade": oco.get("ev_per_trade", 0),
                        "total_pnl_r": oco.get("total_pnl_r", 0),
                        "total_trades": oco.get("total", 0),
                    })
                    
            except Exception as e:
                logger.error(f"Error: {e}")
        
        # Save and rank
        df = pd.DataFrame(self.all_combos)
        if not df.empty:
            df = df.sort_values("ev_per_trade", ascending=False)
            out_path = self.output_dir / f"combo_sweep_{self.run_id}.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"Saved {len(df)} combos to {out_path}")
        
        return df
    
    def get_top_combos(self, k: int = 10) -> pd.DataFrame:
        """Get top K combos by EV."""
        df = pd.DataFrame(self.all_combos)
        if df.empty:
            return df
        return df.sort_values("ev_per_trade", ascending=False).head(k)
    
    def run_full_mining(self, top_combos: pd.DataFrame):
        """Phase 2: Full mine 70% data with top configs."""
        logger.info(f"PHASE 2: Full mining with top {len(top_combos)} configs")
        
        all_patterns = []
        
        # Get unique ratio configs from top combos
        unique_configs = top_combos.drop_duplicates(subset=['rise_ratio', 'invalid_ratio', 'min_unit'])
        
        for _, cfg in unique_configs.iterrows():
            logger.info(f"Mining full: rise={cfg['rise_ratio']}")
            
            cmd = [
                sys.executable, "-m", "src.sweep.pattern_miner_v2",
                "--rise-ratio", str(cfg["rise_ratio"]),
                "--invalid-ratio", str(cfg["invalid_ratio"]),
                "--min-unit", str(cfg["min_unit"]),
                "--max-triggers", "0",  # No limit
                "--output-suffix", f"{self.run_id}_full_{cfg['config_id']}",
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(Path(__file__).parents[2]), timeout=600)
                
                # Load mined patterns
                pattern_file = PROCESSED_DIR / f"patterns_v2_{self.run_id}_full_{cfg['config_id']}.parquet"
                if pattern_file.exists():
                    df = pd.read_parquet(pattern_file)
                    all_patterns.append(df)
                    
            except Exception as e:
                logger.error(f"Mining error: {e}")
        
        if all_patterns:
            combined = pd.concat(all_patterns, ignore_index=True)
            combined = combined.sort_values('trigger_time')
            
            # Save combined
            out_path = PROCESSED_DIR / f"patterns_v2_combined_{self.run_id}.parquet"
            combined.to_parquet(out_path)
            logger.info(f"Combined {len(combined)} patterns")
            return combined
        
        return pd.DataFrame()
    
    def run_full_sweep(
        self,
        n_configs: int = 100,
        n_triggers: int = 30,
        top_k: int = 10,
    ):
        """Run complete V2 pipeline."""
        logger.info("=" * 60)
        logger.info("SWEEP PIPELINE V2")
        logger.info("=" * 60)
        
        # Phase 1: Ratio sweep
        combo_df = self.run_ratio_sweep(n_configs, n_triggers)
        
        # Get top combos
        top_combos = self.get_top_combos(top_k)
        
        if top_combos.empty:
            logger.error("No valid combos found!")
            return
        
        logger.info("TOP COMBOS:")
        for _, c in top_combos.iterrows():
            logger.info(f"  {c['config_id']} + {c['oco_config']}: EV={c['ev_per_trade']:.3f}R")
        
        # Phase 2: Full mining
        patterns = self.run_full_mining(top_combos)
        
        # Save summary
        summary = {
            "run_id": self.run_id,
            "total_combos": len(combo_df),
            "top_combos": top_combos.to_dict('records'),
            "total_patterns_mined": len(patterns) if not patterns.empty else 0,
        }
        
        summary_path = self.output_dir / f"sweep_summary_{self.run_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("=" * 60)
        logger.info(f"SWEEP COMPLETE - {summary_path}")
        logger.info("=" * 60)
        
        return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-configs", type=int, default=100)
    parser.add_argument("--n-triggers", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    
    args = parser.parse_args()
    
    orchestrator = SweepOrchestratorV2()
    orchestrator.run_full_sweep(args.n_configs, args.n_triggers, args.top_k)


if __name__ == "__main__":
    main()
