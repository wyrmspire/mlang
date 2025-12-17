"""
MAE/MFE Analysis
Max Adverse/Favorable Excursion distributions.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from src.datasets.decision_record import DecisionRecord


@dataclass
class ExcursionMetrics:
    """MAE/MFE distribution metrics."""
    # MAE (Max Adverse Excursion - how much against you)
    mae_mean: float
    mae_std: float
    mae_median: float
    mae_max: float
    
    # MFE (Max Favorable Excursion - how much for you)
    mfe_mean: float
    mfe_std: float
    mfe_median: float
    mfe_max: float
    
    # Distributions
    mae_distribution: np.ndarray
    mfe_distribution: np.ndarray


def compute_excursions(records: List[DecisionRecord]) -> ExcursionMetrics:
    """Compute MAE/MFE metrics from decision records."""
    if not records:
        return ExcursionMetrics(
            mae_mean=0, mae_std=0, mae_median=0, mae_max=0,
            mfe_mean=0, mfe_std=0, mfe_median=0, mfe_max=0,
            mae_distribution=np.array([]),
            mfe_distribution=np.array([])
        )
    
    mae_values = np.array([r.cf_mae for r in records])
    mfe_values = np.array([r.cf_mfe for r in records])
    
    return ExcursionMetrics(
        mae_mean=np.mean(mae_values),
        mae_std=np.std(mae_values),
        mae_median=np.median(mae_values),
        mae_max=np.max(mae_values),
        mfe_mean=np.mean(mfe_values),
        mfe_std=np.std(mfe_values),
        mfe_median=np.median(mfe_values),
        mfe_max=np.max(mfe_values),
        mae_distribution=mae_values,
        mfe_distribution=mfe_values,
    )


def compute_excursions_by_outcome(
    records: List[DecisionRecord]
) -> dict:
    """Compute MAE/MFE separately for wins and losses."""
    wins = [r for r in records if r.cf_outcome == 'WIN']
    losses = [r for r in records if r.cf_outcome == 'LOSS']
    
    return {
        'wins': compute_excursions(wins),
        'losses': compute_excursions(losses),
        'all': compute_excursions(records),
    }
