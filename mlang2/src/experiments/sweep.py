"""
Parameter Sweep
Run experiments across parameter grids.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import itertools
import copy
from pathlib import Path
import json

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment, ExperimentResult
from src.config import RESULTS_DIR


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""
    base_config: ExperimentConfig
    sweep_params: Dict[str, List[Any]] = field(default_factory=dict)
    
    # e.g. {'oco_config.tp_multiple': [1.2, 1.4, 1.6],
    #       'oco_config.stop_atr': [0.8, 1.0, 1.2]}


def generate_sweep_configs(sweep: SweepConfig) -> List[ExperimentConfig]:
    """
    Generate all config combinations from sweep parameters.
    """
    if not sweep.sweep_params:
        return [sweep.base_config]
    
    # Generate all combinations
    param_names = list(sweep.sweep_params.keys())
    param_values = list(sweep.sweep_params.values())
    
    configs = []
    for values in itertools.product(*param_values):
        # Deep copy base config
        config = copy.deepcopy(sweep.base_config)
        
        # Apply parameter values
        for name, value in zip(param_names, values):
            _set_nested_attr(config, name, value)
        
        # Update name
        param_str = '_'.join(f"{n.split('.')[-1]}={v}" for n, v in zip(param_names, values))
        config.name = f"{sweep.base_config.name}_{param_str}"
        
        configs.append(config)
    
    return configs


def _set_nested_attr(obj, path: str, value):
    """Set nested attribute using dot notation."""
    parts = path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def run_sweep(sweep: SweepConfig) -> List[ExperimentResult]:
    """
    Run all experiments in a sweep.
    """
    configs = generate_sweep_configs(sweep)
    print(f"Running sweep with {len(configs)} configurations")
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1}/{len(configs)}: {config.name} ---")
        
        try:
            result = run_experiment(config)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Save sweep results
    sweep_id = sweep.base_config.name
    save_sweep_results(results, sweep_id)
    
    return results


def save_sweep_results(results: List[ExperimentResult], sweep_id: str):
    """Save sweep results to JSON."""
    output_path = RESULTS_DIR / f"sweep_{sweep_id}.json"
    
    data = {
        'sweep_id': sweep_id,
        'num_experiments': len(results),
        'results': [r.to_dict() for r in results],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"Saved sweep results to {output_path}")
