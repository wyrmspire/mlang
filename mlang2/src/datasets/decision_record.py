"""
Decision Record
Record logged at every decision point (including NO_TRADE).
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from src.policy.actions import Action, SkipReason
from src.sim.oco import OCOConfig


@dataclass
class DecisionRecord:
    """
    Complete record of a decision point.
    
    Logged at every scanner trigger, not just taken trades.
    This is the core training data structure.
    """
    
    # =========================================================================
    # Identifiers
    # =========================================================================
    timestamp: pd.Timestamp
    bar_idx: int
    decision_id: str = ""          # Unique ID for this decision
    
    # =========================================================================
    # Decision Point Context
    # =========================================================================
    scanner_id: str = ""           # Which scanner triggered
    scanner_context: Dict[str, Any] = field(default_factory=dict)
    
    # =========================================================================
    # Decision Made
    # =========================================================================
    action: Action = Action.NO_TRADE
    skip_reason: SkipReason = SkipReason.NOT_SKIPPED
    skip_reason_detail: str = ""
    
    # =========================================================================
    # Order Configuration (if PLACE_ORDER)
    # =========================================================================
    oco_config: Optional[OCOConfig] = None
    
    # =========================================================================
    # Features (CAUSAL - at decision time)
    # =========================================================================
    # Price windows for CNN
    x_price_1m: Optional[np.ndarray] = None     # (120, 5) or configured
    x_price_5m: Optional[np.ndarray] = None     # (24, 5)
    x_price_15m: Optional[np.ndarray] = None    # (8, 5)
    
    # Context vector for MLP
    x_context: Optional[np.ndarray] = None      # (20,) or configured
    
    # Current market state
    current_price: float = 0.0
    atr: float = 0.0
    
    # =========================================================================
    # Counterfactual Labels (FUTURE-AWARE)
    # =========================================================================
    # These answer: "What WOULD have happened if we traded here?"
    cf_outcome: str = ""           # WIN, LOSS, TIMEOUT
    cf_pnl: float = 0.0           # Points
    cf_pnl_dollars: float = 0.0   # With costs
    cf_mae: float = 0.0           # Max Adverse Excursion
    cf_mfe: float = 0.0           # Max Favorable Excursion
    cf_mae_atr: float = 0.0       # Normalized
    cf_mfe_atr: float = 0.0
    cf_bars_held: int = 0
    cf_entry_price: float = 0.0
    cf_exit_price: float = 0.0
    
    # Optional: outcomes for multiple OCO variants
    cf_multi_oco: Optional[Dict[str, Dict]] = None
    
    # =========================================================================
    # Methods
    # =========================================================================
    
    def is_trade(self) -> bool:
        """Was a trade actually placed?"""
        return self.action == Action.PLACE_ORDER
    
    def was_skipped(self) -> bool:
        """Was this opportunity skipped?"""
        return self.action == Action.NO_TRADE
    
    def is_good_skip(self) -> bool:
        """Skipped and would have lost."""
        return self.was_skipped() and self.cf_outcome == 'LOSS'
    
    def is_bad_skip(self) -> bool:
        """Skipped but would have won."""
        return self.was_skipped() and self.cf_outcome == 'WIN'
    
    def get_label_for_training(self) -> int:
        """Get classification label for training."""
        if self.cf_outcome == 'WIN':
            return 1
        elif self.cf_outcome == 'LOSS':
            return 0
        else:  # TIMEOUT
            return -1  # Could exclude or treat as separate class
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'bar_idx': self.bar_idx,
            'decision_id': self.decision_id,
            'scanner_id': self.scanner_id,
            'action': self.action.value,
            'skip_reason': self.skip_reason.value,
            'current_price': self.current_price,
            'atr': self.atr,
            'cf_outcome': self.cf_outcome,
            'cf_pnl': self.cf_pnl,
            'cf_pnl_dollars': self.cf_pnl_dollars,
            'cf_mae': self.cf_mae,
            'cf_mfe': self.cf_mfe,
            'cf_bars_held': self.cf_bars_held,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'DecisionRecord':
        """Create from dictionary."""
        record = DecisionRecord(
            timestamp=pd.Timestamp(d['timestamp']) if d.get('timestamp') else None,
            bar_idx=d.get('bar_idx', 0),
            decision_id=d.get('decision_id', ''),
            scanner_id=d.get('scanner_id', ''),
            action=Action(d.get('action', 'NO_TRADE')),
            skip_reason=SkipReason(d.get('skip_reason', 'NOT_SKIPPED')),
            current_price=d.get('current_price', 0.0),
            atr=d.get('atr', 0.0),
            cf_outcome=d.get('cf_outcome', ''),
            cf_pnl=d.get('cf_pnl', 0.0),
            cf_pnl_dollars=d.get('cf_pnl_dollars', 0.0),
            cf_mae=d.get('cf_mae', 0.0),
            cf_mfe=d.get('cf_mfe', 0.0),
            cf_bars_held=d.get('cf_bars_held', 0),
        )
        return record
