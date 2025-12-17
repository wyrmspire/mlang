"""
Trade Record
Record of a completed trade (after exit).
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TradeRecord:
    """
    Record of a completed trade.
    
    Only created when a trade exits (via SL, TP, or timeout).
    """
    
    # Identifiers
    trade_id: str = ""
    decision_id: str = ""          # Links to original DecisionRecord
    
    # Entry
    entry_time: Optional[pd.Timestamp] = None
    entry_bar: int = 0
    entry_price: float = 0.0
    direction: str = ""
    
    # Exit
    exit_time: Optional[pd.Timestamp] = None
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""          # 'SL', 'TP', 'TIMEOUT', 'MANUAL'
    
    # Outcome
    outcome: str = ""              # 'WIN', 'LOSS', 'TIMEOUT'
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    gross_pnl: float = 0.0
    commission: float = 0.0
    
    # Analytics
    bars_held: int = 0
    mae: float = 0.0               # Max Adverse Excursion
    mfe: float = 0.0               # Max Favorable Excursion
    r_multiple: float = 0.0        # PnL / initial risk
    
    # Context at entry
    scanner_id: str = ""
    entry_atr: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'decision_id': self.decision_id,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'entry_bar': self.entry_bar,
            'entry_price': self.entry_price,
            'direction': self.direction,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_bar': self.exit_bar,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'outcome': self.outcome,
            'pnl_points': self.pnl_points,
            'pnl_dollars': self.pnl_dollars,
            'bars_held': self.bars_held,
            'mae': self.mae,
            'mfe': self.mfe,
            'r_multiple': self.r_multiple,
            'scanner_id': self.scanner_id,
        }
