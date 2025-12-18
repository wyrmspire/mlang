"""
Counterfactual Labeler
Label ALL decision points with "what would have happened".
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from src.labels.future_window import FutureWindowProvider
from src.labels.trade_outcome import TradeOutcome, compute_trade_outcome
from src.sim.oco import OCOConfig, create_oco_bracket
from src.sim.bar_fill_model import BarFillConfig, BarFillEngine
from src.sim.costs import CostModel, DEFAULT_COSTS


@dataclass
class CounterfactualLabel:
    """
    Counterfactual outcome label.
    
    "What WOULD have happened if we traded here?"
    """
    # Primary outcome
    outcome: str          # WIN, LOSS, TIMEOUT
    pnl: float           # Points
    pnl_dollars: float   # Actual dollars (with costs)
    
    # Excursions
    mae: float           # Max Adverse Excursion (points)
    mfe: float           # Max Favorable Excursion (points)
    mae_atr: float       # MAE normalized by ATR
    mfe_atr: float       # MFE normalized by ATR
    
    # Timing
    bars_held: int
    
    # Prices
    entry_price: float
    exit_price: float
    stop_price: float
    tp_price: float
    
    # OCO config used
    oco_config: OCOConfig = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cf_outcome': self.outcome,
            'cf_pnl': self.pnl,
            'cf_pnl_dollars': self.pnl_dollars,
            'cf_mae': self.mae,
            'cf_mfe': self.mfe,
            'cf_mae_atr': self.mae_atr,
            'cf_mfe_atr': self.mfe_atr,
            'cf_bars_held': self.bars_held,
            'cf_entry_price': self.entry_price,
            'cf_exit_price': self.exit_price,
        }


def compute_counterfactual(
    df: pd.DataFrame,
    entry_idx: int,
    oco_config: OCOConfig,
    atr: float,
    fill_config: BarFillConfig = None,
    costs: CostModel = None,
    max_bars: int = 200
) -> CounterfactualLabel:
    """
    Compute counterfactual outcome for a decision point.
    
    "If we entered here with this OCO, what would happen?"
    
    Args:
        df: Full dataframe
        entry_idx: Bar index of decision point
        oco_config: OCO configuration to simulate
        atr: ATR at decision point (for bracket calculation)
        fill_config: Bar fill configuration
        costs: Cost model
        max_bars: Max bars to simulate
        
    Returns:
        CounterfactualLabel with complete outcome info
    """
    costs = costs or DEFAULT_COSTS
    fill_config = fill_config or BarFillConfig()
    
    # Create OCO bracket
    entry_bar = df.iloc[entry_idx]
    base_price = entry_bar['close']
    
    bracket = create_oco_bracket(
        config=oco_config,
        base_price=base_price,
        atr=atr,
        costs=costs
    )
    
    # Create future provider
    future_provider = FutureWindowProvider(df, entry_idx)
    
    # Compute outcome
    outcome = compute_trade_outcome(
        future_provider=future_provider,
        entry_price=bracket.entry_price,
        direction=oco_config.direction,
        stop_loss=bracket.stop_price,
        take_profit=bracket.tp_price,
        max_bars=max_bars,
        fill_config=fill_config
    )
    
    # Calculate dollar PnL
    pnl_dollars = costs.calculate_pnl(
        bracket.entry_price,
        outcome.exit_price,
        oco_config.direction,
        contracts=1,
        include_commission=True
    )
    
    return CounterfactualLabel(
        outcome=outcome.outcome,
        pnl=outcome.pnl,
        pnl_dollars=pnl_dollars,
        mae=outcome.mae,
        mfe=outcome.mfe,
        mae_atr=outcome.mae / atr if atr > 0 else 0,
        mfe_atr=outcome.mfe / atr if atr > 0 else 0,
        bars_held=outcome.bars_held,
        entry_price=bracket.entry_price,
        exit_price=outcome.exit_price,
        stop_price=bracket.stop_price,
        tp_price=bracket.tp_price,
        oco_config=oco_config,
    )


def compute_multi_oco_counterfactuals(
    df: pd.DataFrame,
    entry_idx: int,
    oco_configs: List[OCOConfig],
    atr: float,
    fill_config: BarFillConfig = None,
    costs: CostModel = None
) -> Dict[str, CounterfactualLabel]:
    """
    Compute counterfactual outcomes for multiple OCO variants.
    
    Enables "which OCO would have worked best?" analysis.
    
    Returns:
        Dict mapping oco_config.name to CounterfactualLabel
    """
    results = {}
    
    for oco in oco_configs:
        label = compute_counterfactual(
            df=df,
            entry_idx=entry_idx,
            oco_config=oco,
            atr=atr,
            fill_config=fill_config,
            costs=costs
        )
        name = oco.name or f"{oco.direction}_{oco.tp_multiple}R"
        results[name] = label
    
    return results


def label_is_good_skip(cf_label: CounterfactualLabel) -> bool:
    """
    Determine if skipping this trade was a good decision.
    
    "Skipped good" = would have lost
    "Skipped bad" = would have won
    
    From the perspective of improving the model, we want to
    learn to skip the losses.
    """
    return cf_label.outcome == 'LOSS'


def label_is_bad_skip(cf_label: CounterfactualLabel) -> bool:
    """Trade we skipped but should have taken (would have won)."""
    return cf_label.outcome == 'WIN'


def compute_smart_stop_counterfactual(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    stop_price: float,
    tp_multiple: float,
    atr: float,
    fill_config: BarFillConfig = None,
    costs: CostModel = None,
    max_bars: int = 200,
    oco_name: str = ""
) -> CounterfactualLabel:
    """
    Compute counterfactual with pre-calculated stop price.
    
    Use this with stop_calculator for smart stops based on
    candle levels, ranges, swings, etc.
    
    Args:
        df: Full dataframe
        entry_idx: Bar index of decision point
        direction: 'LONG' or 'SHORT'
        stop_price: Pre-calculated stop price (from stop_calculator)
        tp_multiple: R multiple for take profit
        atr: ATR for reference (not used for stop)
        fill_config: Bar fill configuration
        costs: Cost model
        max_bars: Max bars to simulate
        oco_name: Name for this configuration
        
    Returns:
        CounterfactualLabel with complete outcome info
    """
    costs = costs or DEFAULT_COSTS
    fill_config = fill_config or BarFillConfig()
    
    entry_bar = df.iloc[entry_idx]
    entry_price = entry_bar['close']
    
    # Calculate risk and TP
    if direction == "LONG":
        risk = entry_price - stop_price
        tp_price = entry_price + (risk * tp_multiple)
    else:
        risk = stop_price - entry_price
        tp_price = entry_price - (risk * tp_multiple)
    
    # Round to tick
    stop_price = costs.round_to_tick(stop_price, 'down' if direction == 'LONG' else 'up')
    tp_price = costs.round_to_tick(tp_price, 'up' if direction == 'LONG' else 'down')
    
    # Create future provider
    future_provider = FutureWindowProvider(df, entry_idx)
    
    # Compute outcome
    outcome = compute_trade_outcome(
        future_provider=future_provider,
        entry_price=entry_price,
        direction=direction,
        stop_loss=stop_price,
        take_profit=tp_price,
        max_bars=max_bars,
        fill_config=fill_config
    )
    
    # Calculate dollar PnL
    pnl_dollars = costs.calculate_pnl(
        entry_price,
        outcome.exit_price,
        direction,
        contracts=1,
        include_commission=True
    )
    
    # Create a minimal OCOConfig for storage
    oco_config = OCOConfig(
        direction=direction,
        stop_atr=0,  # Not used
        tp_multiple=tp_multiple,
        name=oco_name
    )
    
    return CounterfactualLabel(
        outcome=outcome.outcome,
        pnl=outcome.pnl,
        pnl_dollars=pnl_dollars,
        mae=outcome.mae,
        mfe=outcome.mfe,
        mae_atr=outcome.mae / atr if atr > 0 else 0,
        mfe_atr=outcome.mfe / atr if atr > 0 else 0,
        bars_held=outcome.bars_held,
        entry_price=entry_price,
        exit_price=outcome.exit_price,
        stop_price=stop_price,
        tp_price=tp_price,
        oco_config=oco_config,
    )
