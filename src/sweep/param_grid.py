"""
Parameter Grid Generator
Creates sweep configurations for pattern mining, OCO brackets, and models.
"""

import numpy as np
from typing import List
import itertools

from .config import (
    PatternSweepConfig, 
    OCOBracketConfig, 
    ModelSweepConfig,
    CandleComposition,
    PATTERN_SWEEP_RANGES,
    OCO_SWEEP_VALUES,
    MODEL_ARCHITECTURES,
    CANDLE_COMPOSITIONS,
)


def generate_pattern_grid(n: int = 33, seed: int = 42) -> List[PatternSweepConfig]:
    """
    Generate N pattern configurations via Latin Hypercube Sampling.
    Default 33 configs × 30 triggers = ~1000 pattern evaluations.
    
    Args:
        n: Number of configurations to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of PatternSweepConfig objects
    """
    np.random.seed(seed)
    configs = []
    
    ranges = PATTERN_SWEEP_RANGES
    
    for i in range(n):
        # Sample each parameter uniformly within range
        rise_min = np.random.uniform(*ranges["rise_ratio_min"])
        rise_max = np.random.uniform(rise_min + 0.5, ranges["rise_ratio_max"][1])  # Ensure max > min
        
        config = PatternSweepConfig(
            rise_ratio_min=round(rise_min, 2),
            rise_ratio_max=round(rise_max, 2),
            min_drop=round(np.random.uniform(*ranges["min_drop"]), 2),
            atr_buffer=round(np.random.uniform(*ranges["atr_buffer"]), 2),
            validation_distance=round(np.random.uniform(*ranges["validation_distance"]), 2),
            lookback_bars=int(np.random.uniform(*ranges["lookback_bars"])),
            config_id=f"pattern_{i:03d}",
        )
        configs.append(config)
    
    return configs


def generate_oco_grid(n: int = 33, seed: int = 42) -> List[OCOBracketConfig]:
    """
    Generate N OCO bracket configurations.
    Uses combination of grid + random for diversity.
    
    Args:
        n: Number of configurations to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of OCOBracketConfig objects
    """
    np.random.seed(seed)
    configs = []
    
    vals = OCO_SWEEP_VALUES
    
    # Generate all combinations
    all_combos = list(itertools.product(
        vals["direction"],
        vals["r_multiple"],
        vals["stop_atr_pct"],
        vals["stop_type"],
    ))
    
    # If we need fewer than total combos, sample randomly
    if n < len(all_combos):
        indices = np.random.choice(len(all_combos), size=n, replace=False)
        selected = [all_combos[i] for i in indices]
    else:
        selected = all_combos[:n]
    
    for i, (direction, r_mult, stop_pct, stop_type) in enumerate(selected):
        config = OCOBracketConfig(
            direction=direction,
            r_multiple=r_mult,
            stop_atr_pct=stop_pct,
            stop_type=stop_type,
            config_id=f"oco_{i:03d}",
        )
        configs.append(config)
    
    return configs


def generate_model_grid(
    architectures: List[str] = None,
    candle_compositions: List[CandleComposition] = None,
) -> List[ModelSweepConfig]:
    """
    Generate model configurations for each architecture × candle composition.
    
    Returns:
        List of ModelSweepConfig objects
    """
    if architectures is None:
        architectures = MODEL_ARCHITECTURES
    if candle_compositions is None:
        candle_compositions = CANDLE_COMPOSITIONS
    
    configs = []
    idx = 0
    
    for arch in architectures:
        for candle_comp in candle_compositions:
            # Adjust seq_len based on architecture
            if arch == "CNN_Wide" and candle_comp.candles_1m < 60:
                # Skip wide CNN for small inputs
                continue
                
            config = ModelSweepConfig(
                architecture=arch,
                epochs=10,
                learning_rate=0.001,
                batch_size=32,
                dropout=0.3,
                candle_composition=candle_comp,
                config_id=f"model_{idx:03d}",
            )
            configs.append(config)
            idx += 1
    
    return configs


def get_default_oco_scenarios() -> List[OCOBracketConfig]:
    """
    Get the 10 default OCO scenarios for test phase evaluation.
    Every model test gets these 10 results.
    """
    return [
        # Long scenarios
        OCOBracketConfig("LONG", 1.0, 0.50, "ATR", "default_01"),
        OCOBracketConfig("LONG", 1.4, 0.50, "ATR", "default_02"),
        OCOBracketConfig("LONG", 2.0, 0.50, "WICK", "default_03"),
        OCOBracketConfig("LONG", 1.4, 0.25, "WICK", "default_04"),
        # Short scenarios
        OCOBracketConfig("SHORT", 1.0, 0.50, "ATR", "default_05"),
        OCOBracketConfig("SHORT", 1.4, 0.50, "ATR", "default_06"),
        OCOBracketConfig("SHORT", 2.0, 0.50, "WICK", "default_07"),
        OCOBracketConfig("SHORT", 1.4, 0.25, "WICK", "default_08"),
        # Hybrid scenarios
        OCOBracketConfig("LONG", 1.8, 0.75, "ATR", "default_09"),
        OCOBracketConfig("SHORT", 1.8, 0.75, "ATR", "default_10"),
    ]


if __name__ == "__main__":
    # Quick test
    print("Pattern Configs:", len(generate_pattern_grid(33)))
    print("OCO Configs:", len(generate_oco_grid(33)))
    print("Model Configs:", len(generate_model_grid()))
    print("Default OCO Scenarios:", len(get_default_oco_scenarios()))
