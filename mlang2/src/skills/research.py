"""
Research Skills
Workflows for experiments, walk-forward tests, and sweeps.
"""

from pathlib import Path
from typing import Optional, Dict
import pandas as pd

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment, ExperimentResult
from src.experiments.splits import generate_walk_forward_splits, WalkForwardConfig

def run_research_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Skill: Run a standard experiment and return results.
    """
    return run_experiment(config)

def run_simple_walkforward(
    name: str,
    train_weeks: int = 6,
    test_weeks: int = 1,
    start_date: str = "2025-03-17",
    scanner_id: str = "level_proximity"
) -> Dict:
    """
    Skill: Run a walk-forward test over a specified period.
    """
    print(f"Starting walk-forward research: {name}")
    
    # Generate splits
    wf_config = WalkForwardConfig(
        train_days=train_weeks * 7,
        test_days=test_weeks * 7,
        num_splits=1,  # Start with 1 for now
    )
    
    # This is a simplification, we would ideally use splits.py
    # But for a "skill" we want it to be high level.
    
    # For now, let's just use the logic from test_walkforward.py
    # but encapsulated as a reusable skill.
    
    # (Implementation details would follow, calling into src.experiments)
    return {"status": "success", "message": "Walk-forward started (simulated)"}
