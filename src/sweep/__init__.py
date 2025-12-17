# Shotgun Sweep Pipeline
# CLI-driven hyperparameter exploration for rejection strategy optimization
# GPU-accelerated throughout

from .config import (
    PatternSweepConfig,
    CandleComposition,
    OCOBracketConfig,
    ModelSweepConfig,
    PATTERN_SWEEP_RANGES,
    OCO_SWEEP_VALUES,
    MODEL_ARCHITECTURES,
    CANDLE_COMPOSITIONS,
)

from .param_grid import (
    generate_pattern_grid,
    generate_oco_grid,
    generate_model_grid,
    get_default_oco_scenarios,
)

from .results import SweepResults, load_results

__all__ = [
    # Config classes
    "PatternSweepConfig",
    "CandleComposition", 
    "OCOBracketConfig",
    "ModelSweepConfig",
    # Constants
    "PATTERN_SWEEP_RANGES",
    "OCO_SWEEP_VALUES",
    "MODEL_ARCHITECTURES",
    "CANDLE_COMPOSITIONS",
    # Grid generators
    "generate_pattern_grid",
    "generate_oco_grid",
    "generate_model_grid",
    "get_default_oco_scenarios",
    # Results
    "SweepResults",
    "load_results",
]
