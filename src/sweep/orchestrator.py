"""
Sweep Orchestrator - Coordinates full pipeline
1. Pattern Sweep: Test geometry configs × 10 OCO each
2. Rank all (geometry, OCO) combos by EV
3. Top 10 combos → full mining → training
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import PROCESSED_DIR, MODELS_DIR
from src.utils.logging_utils import get_logger
from src.sweep.param_grid import generate_pattern_grid

logger = get_logger("orchestrator")


@dataclass
class ComboResult:
    """Single (geometry, OCO) combination result."""
    pattern_config_id: str
    rise_min: float
    rise_max: float
    min_drop: float
    atr_buffer: float
    oco_config: str
    direction: str
    r_mult: float
    stop_atr: float
    win_rate: float
    ev_per_trade: float
    total_pnl_r: float
    total_patterns: int


class SweepOrchestrator:
    """Orchestrates the full shotgun sweep pipeline."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or PROCESSED_DIR / "sweep_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.all_combos: List[ComboResult] = []
        
        logger.info(f"Sweep run ID: {self.run_id}")
    
    def run_pattern_sweep(
        self,
        n_configs: int = 100,
        n_triggers: int = 30,
    ) -> pd.DataFrame:
        """
        Run pattern sweep: test N geometry configs × 10 OCO each.
        Returns DataFrame with all (geometry, OCO) combinations ranked.
        """
        logger.info(f"Starting Pattern Sweep: {n_configs} geometries × 10 OCO × {n_triggers} triggers")
        
        configs = generate_pattern_grid(n_configs)
        
        for i, config in enumerate(configs):
            logger.info(f"[{i+1}/{n_configs}] Testing {config.config_id}")
            
            cmd = [
                sys.executable, "-m", "src.sweep.pattern_miner_sweep",
                "--rise-min", str(config.rise_ratio_min),
                "--rise-max", str(config.rise_ratio_max),
                "--min-drop", str(config.min_drop),
                "--atr-buffer", str(config.atr_buffer),
                "--lookback", str(config.lookback_bars),
                "--max-triggers", str(n_triggers),
                "--output-suffix", f"{self.run_id}_{config.config_id}",
            ]
            
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=str(Path(__file__).parents[2]), timeout=300,
                )
                
                # Parse JSON output
                for line in reversed(result.stdout.strip().split('\n')):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
                
                # Extract OCO results and add to combo list
                geom = data.get("pattern_geometry", {})
                total_patterns = data.get("total_patterns", 0)
                
                for oco in data.get("oco_results", []):
                    combo = ComboResult(
                        pattern_config_id=config.config_id,
                        rise_min=geom.get("rise_min", 0),
                        rise_max=geom.get("rise_max", 0),
                        min_drop=geom.get("min_drop", 0),
                        atr_buffer=geom.get("atr_buffer", 0),
                        oco_config=oco.get("oco_config", ""),
                        direction=oco.get("direction", ""),
                        r_mult=oco.get("r_mult", 0),
                        stop_atr=oco.get("stop_atr", 0),
                        win_rate=oco.get("win_rate", 0),
                        ev_per_trade=oco.get("ev_per_trade", 0),
                        total_pnl_r=oco.get("total_pnl_r", 0),
                        total_patterns=total_patterns,
                    )
                    self.all_combos.append(combo)
                    
            except Exception as e:
                logger.error(f"Error for {config.config_id}: {e}")
        
        # Create ranked DataFrame
        df = pd.DataFrame([vars(c) for c in self.all_combos])
        if not df.empty:
            df = df.sort_values("ev_per_trade", ascending=False)
        
        # Save
        out_path = self.output_dir / f"combo_sweep_{self.run_id}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(df)} combos to {out_path}")
        
        return df
    
    def get_top_combos(self, k: int = 10) -> pd.DataFrame:
        """Get top K (geometry, OCO) combinations by EV."""
        df = pd.DataFrame([vars(c) for c in self.all_combos])
        if df.empty:
            return df
        return df.sort_values("ev_per_trade", ascending=False).head(k)
    
    def run_full_training(self, top_combos: pd.DataFrame):
        """
        For each top combo:
        1. Re-mine with NO trigger limit (full dataset)
        2. Train CNN on preceding price behavior
        """
        logger.info(f"Training on top {len(top_combos)} combos...")
        
        for idx, combo in top_combos.iterrows():
            logger.info(f"Full training: {combo['pattern_config_id']} + {combo['oco_config']}")
            
            # Step 1: Re-mine full dataset
            mine_cmd = [
                sys.executable, "-m", "src.sweep.pattern_miner_sweep",
                "--rise-min", str(combo["rise_min"]),
                "--rise-max", str(combo["rise_max"]),
                "--min-drop", str(combo["min_drop"]),
                "--atr-buffer", str(combo["atr_buffer"]),
                "--max-triggers", "0",  # No limit - mine all
                "--output-suffix", f"{self.run_id}_full_{combo['pattern_config_id']}",
            ]
            
            try:
                subprocess.run(mine_cmd, capture_output=True, text=True,
                              cwd=str(Path(__file__).parents[2]), timeout=600)
                
                # Step 2: Train on full data
                pattern_file = f"labeled_sweep_{self.run_id}_full_{combo['pattern_config_id']}.parquet"
                
                train_cmd = [
                    sys.executable, "-m", "src.sweep.train_sweep",
                    "--architecture", "CNN_Classic",
                    "--input-data", str(PROCESSED_DIR / pattern_file),
                    "--epochs", "10",
                    "--output-suffix", f"{self.run_id}_{combo['pattern_config_id']}_{combo['oco_config']}",
                ]
                
                subprocess.run(train_cmd, capture_output=True, text=True,
                              cwd=str(Path(__file__).parents[2]), timeout=600)
                
            except Exception as e:
                logger.error(f"Training error: {e}")
        
        logger.info("Full training complete!")
    
    def run_full_sweep(
        self,
        n_pattern_configs: int = 100,
        n_triggers: int = 30,
        top_k_combos: int = 10,
    ) -> Dict:
        """Complete shotgun sweep pipeline."""
        logger.info("=" * 60)
        logger.info("SHOTGUN SWEEP PIPELINE")
        logger.info("=" * 60)
        
        # Phase 1: Pattern × OCO Sweep
        logger.info("\n[PHASE 1] Pattern Geometry × OCO Sweep")
        combo_df = self.run_pattern_sweep(n_pattern_configs, n_triggers)
        
        # Phase 2: Select Top Combos
        logger.info("\n[PHASE 2] Selecting Top Combos")
        top_combos = self.get_top_combos(top_k_combos)
        
        if top_combos.empty:
            logger.error("No valid combos found!")
            return {"error": "No combos"}
        
        logger.info("TOP COMBOS:")
        for _, c in top_combos.iterrows():
            logger.info(f"  {c['pattern_config_id']} + {c['oco_config']}: "
                       f"EV={c['ev_per_trade']:.3f}R")
        
        # Phase 3: Full Training on Top Combos
        logger.info("\n[PHASE 3] Full Training on Top Combos")
        self.run_full_training(top_combos)
        
        # Save summary
        summary = {
            "run_id": self.run_id,
            "total_combos_tested": len(combo_df),
            "top_combos": top_combos.to_dict('records'),
        }
        
        summary_path = self.output_dir / f"sweep_summary_{self.run_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("=" * 60)
        logger.info(f"SWEEP COMPLETE - Results: {summary_path}")
        logger.info("=" * 60)
        
        return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--phase", type=str, choices=["pattern", "train"])
    parser.add_argument("--n-configs", type=int, default=100)
    parser.add_argument("--n-triggers", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    
    args = parser.parse_args()
    orchestrator = SweepOrchestrator()
    
    if args.full or not args.phase:
        orchestrator.run_full_sweep(args.n_configs, args.n_triggers, args.top_k)
    elif args.phase == "pattern":
        orchestrator.run_pattern_sweep(args.n_configs, args.n_triggers)


if __name__ == "__main__":
    main()
