"""
Labeler
Main labeling pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pandas as pd

from src.labels.counterfactual import (
    CounterfactualLabel, 
    compute_counterfactual,
    compute_multi_oco_counterfactuals
)
from src.sim.oco import OCOConfig
from src.sim.bar_fill_model import BarFillConfig
from src.sim.costs import CostModel, DEFAULT_COSTS


@dataclass
class LabelConfig:
    """Configuration for labeling."""
    # Primary OCO for counterfactual
    oco_config: OCOConfig = field(default_factory=OCOConfig)
    
    # Optional: additional OCO variants for multi-armed bandit
    oco_variants: List[OCOConfig] = field(default_factory=list)
    
    # Fill model
    fill_config: BarFillConfig = field(default_factory=BarFillConfig)
    
    # Cost model
    cost_model: CostModel = field(default_factory=lambda: DEFAULT_COSTS)
    
    # Simulation
    max_bars: int = 200
    
    def to_dict(self) -> dict:
        return {
            'oco_config': self.oco_config.to_dict(),
            'oco_variants': [o.to_dict() for o in self.oco_variants],
            'fill_config': self.fill_config.to_dict(),
            'max_bars': self.max_bars,
        }


class Labeler:
    """
    Main labeling class.
    
    Takes decision points and adds counterfactual labels.
    """
    
    def __init__(self, config: LabelConfig):
        self.config = config
    
    def label_decision_point(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        atr: float
    ) -> CounterfactualLabel:
        """
        Label a single decision point.
        
        Args:
            df: Full dataframe
            entry_idx: Index of decision point
            atr: ATR at decision point
            
        Returns:
            CounterfactualLabel
        """
        return compute_counterfactual(
            df=df,
            entry_idx=entry_idx,
            oco_config=self.config.oco_config,
            atr=atr,
            fill_config=self.config.fill_config,
            costs=self.config.cost_model,
            max_bars=self.config.max_bars
        )
    
    def label_decision_point_multi(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        atr: float
    ) -> Dict[str, CounterfactualLabel]:
        """
        Label with multiple OCO variants.
        """
        all_ocos = [self.config.oco_config] + self.config.oco_variants
        
        return compute_multi_oco_counterfactuals(
            df=df,
            entry_idx=entry_idx,
            oco_configs=all_ocos,
            atr=atr,
            fill_config=self.config.fill_config,
            costs=self.config.cost_model
        )
    
    def label_batch(
        self,
        df: pd.DataFrame,
        entry_indices: List[int],
        atrs: List[float]
    ) -> List[CounterfactualLabel]:
        """
        Label a batch of decision points.
        """
        results = []
        for idx, atr in zip(entry_indices, atrs):
            label = self.label_decision_point(df, idx, atr)
            results.append(label)
        return results


# Convenience function
def create_default_labeler(
    direction: str = "LONG",
    tp_multiple: float = 1.4,
    stop_atr: float = 1.0
) -> Labeler:
    """Create labeler with common defaults."""
    oco = OCOConfig(
        direction=direction,
        tp_multiple=tp_multiple,
        stop_atr=stop_atr,
        name=f"{direction}_{tp_multiple}R"
    )
    config = LabelConfig(oco_config=oco)
    return Labeler(config)
