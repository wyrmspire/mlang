"""
Actions
Decision action types and policy decision structure.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.sim.oco import OCOConfig


class Action(Enum):
    """What action to take at a decision point."""
    NO_TRADE = "NO_TRADE"         # Skip this opportunity
    PLACE_ORDER = "PLACE_ORDER"   # Enter new position
    MANAGE = "MANAGE"             # Adjust existing position
    EXIT = "EXIT"                 # Close position


class SkipReason(Enum):
    """
    Why a decision point was skipped.
    
    This is crucial for understanding dataset composition:
    - FILTER_BLOCK: Filtered out before reaching policy
    - COOLDOWN: Too soon after last trade
    - IN_POSITION: Already have open position
    - POLICY_NO: Policy decided not to trade
    - OTHER: Other reason
    """
    NOT_SKIPPED = "NOT_SKIPPED"   # Trade was taken
    FILTER_BLOCK = "FILTER_BLOCK"
    COOLDOWN = "COOLDOWN"
    IN_POSITION = "IN_POSITION"
    POLICY_NO = "POLICY_NO"
    OTHER = "OTHER"


@dataclass
class PolicyDecision:
    """
    Complete decision at a decision point.
    """
    action: Action
    skip_reason: SkipReason = SkipReason.NOT_SKIPPED
    reason_detail: str = ""       # Human-readable explanation
    
    # If PLACE_ORDER
    order_config: Optional[OCOConfig] = None
    
    # Scanner context that led to this decision
    scanner_id: str = ""
    scanner_context: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence/score (for ML-based policy)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action.value,
            'skip_reason': self.skip_reason.value,
            'reason_detail': self.reason_detail,
            'scanner_id': self.scanner_id,
            'confidence': self.confidence,
            'order_config': self.order_config.to_dict() if self.order_config else None,
        }


def make_no_trade(
    reason: SkipReason,
    detail: str = "",
    scanner_id: str = ""
) -> PolicyDecision:
    """Helper to create NO_TRADE decision."""
    return PolicyDecision(
        action=Action.NO_TRADE,
        skip_reason=reason,
        reason_detail=detail,
        scanner_id=scanner_id,
    )


def make_trade(
    order_config: OCOConfig,
    scanner_id: str = "",
    confidence: float = 1.0,
    context: Dict[str, Any] = None
) -> PolicyDecision:
    """Helper to create PLACE_ORDER decision."""
    return PolicyDecision(
        action=Action.PLACE_ORDER,
        skip_reason=SkipReason.NOT_SKIPPED,
        order_config=order_config,
        scanner_id=scanner_id,
        confidence=confidence,
        scanner_context=context or {},
    )
