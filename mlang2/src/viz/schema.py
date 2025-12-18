"""
Viz Schema
Dataclasses for visualization export.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


@dataclass
class VizWindow:
    """
    OHLCV windows captured at decision time.
    Used by the UI to render "what the model saw".
    """
    x_price_1m: List[List[float]] = field(default_factory=list)  # (lookback, 5)
    x_price_5m: List[List[float]] = field(default_factory=list)
    x_price_15m: List[List[float]] = field(default_factory=list)
    x_context: List[float] = field(default_factory=list)
    
    # Normalization metadata for denormalization in UI
    norm_method: str = "zscore"
    norm_params: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'x_price_1m': self.x_price_1m,
            'x_price_5m': self.x_price_5m,
            'x_price_15m': self.x_price_15m,
            'x_context': self.x_context,
            'norm_method': self.norm_method,
            'norm_params': self.norm_params,
        }


@dataclass
class VizOCO:
    """
    OCO bracket snapshot for visualization.
    """
    entry_price: float = 0.0
    stop_price: float = 0.0
    tp_price: float = 0.0
    entry_type: str = "LIMIT"
    direction: str = "LONG"
    
    reference_type: str = "PRICE"
    reference_value: float = 0.0
    atr_at_creation: float = 0.0
    max_bars: int = 200
    
    # Config values for tooltip display
    stop_atr: float = 1.0
    tp_multiple: float = 1.4
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_price': self.entry_price,
            'stop_price': self.stop_price,
            'tp_price': self.tp_price,
            'entry_type': self.entry_type,
            'direction': self.direction,
            'reference_type': self.reference_type,
            'reference_value': self.reference_value,
            'atr_at_creation': self.atr_at_creation,
            'max_bars': self.max_bars,
            'stop_atr': self.stop_atr,
            'tp_multiple': self.tp_multiple,
        }


@dataclass
class VizFill:
    """
    Order fill event.
    """
    order_id: str = ""
    fill_type: str = ""  # 'ENTRY', 'SL', 'TP', 'TIMEOUT'
    price: float = 0.0
    bar_idx: int = 0
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'fill_type': self.fill_type,
            'price': self.price,
            'bar_idx': self.bar_idx,
            'timestamp': self.timestamp,
        }


@dataclass
class VizDecision:
    """
    Decision point for visualization.
    References DecisionRecord but adds viz-specific fields.
    """
    decision_id: str = ""
    timestamp: Optional[str] = None
    bar_idx: int = 0
    index: int = 0  # For paging (Next/Prev)
    
    scanner_id: str = ""
    scanner_context: Dict[str, Any] = field(default_factory=dict)
    
    action: str = "NO_TRADE"
    skip_reason: str = ""
    
    # Market state at decision
    current_price: float = 0.0
    atr: float = 0.0
    
    # Counterfactual label
    cf_outcome: str = ""
    cf_pnl_dollars: float = 0.0
    
    # Model outputs (if available)
    model_logits: Optional[List[float]] = None
    model_probs: Optional[List[float]] = None
    
    # Window (if include_windows=True)
    window: Optional[VizWindow] = None
    
    # OCO (if PLACE_ORDER)
    oco: Optional[VizOCO] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_id': self.decision_id,
            'timestamp': self.timestamp,
            'bar_idx': self.bar_idx,
            'index': self.index,
            'scanner_id': self.scanner_id,
            'scanner_context': self.scanner_context,
            'action': self.action,
            'skip_reason': self.skip_reason,
            'current_price': self.current_price,
            'atr': self.atr,
            'cf_outcome': self.cf_outcome,
            'cf_pnl_dollars': self.cf_pnl_dollars,
            'model_logits': self.model_logits,
            'model_probs': self.model_probs,
            'window': self.window.to_dict() if self.window else None,
            'oco': self.oco.to_dict() if self.oco else None,
        }


@dataclass
class VizTrade:
    """
    Completed trade for visualization.
    Reuses TradeRecord.to_dict() fields, adds lifecycle.
    """
    trade_id: str = ""
    decision_id: str = ""
    index: int = 0  # For paging
    
    direction: str = ""
    size: int = 1  # Contracts
    
    # Entry
    entry_time: Optional[str] = None
    entry_bar: int = 0
    entry_price: float = 0.0
    
    # Exit
    exit_time: Optional[str] = None
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    
    # Outcome
    outcome: str = ""
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    r_multiple: float = 0.0
    
    # Analytics
    bars_held: int = 0
    mae: float = 0.0
    mfe: float = 0.0
    
    # Lifecycle events
    fills: List[VizFill] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'decision_id': self.decision_id,
            'index': self.index,
            'direction': self.direction,
            'size': self.size,
            'entry_time': self.entry_time,
            'entry_bar': self.entry_bar,
            'entry_price': self.entry_price,
            'exit_time': self.exit_time,
            'exit_bar': self.exit_bar,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'outcome': self.outcome,
            'pnl_points': self.pnl_points,
            'pnl_dollars': self.pnl_dollars,
            'r_multiple': self.r_multiple,
            'bars_held': self.bars_held,
            'mae': self.mae,
            'mfe': self.mfe,
            'fills': [f.to_dict() for f in self.fills],
        }


@dataclass
class VizBarSeries:
    """
    Full OHLCV series for overview mode.
    """
    timeframe: str = "1m"
    bars: List[Dict[str, Any]] = field(default_factory=list)  # [{time, o, h, l, c, v}, ...]
    
    # Trade markers for overlay
    trade_markers: List[Dict[str, Any]] = field(default_factory=list)  # [{bar_idx, type, price}, ...]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timeframe': self.timeframe,
            'bars': self.bars,
            'trade_markers': self.trade_markers,
        }


@dataclass
class VizSplit:
    """
    Single walk-forward split summary.
    """
    split_id: str = ""
    split_idx: int = 0
    
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    
    # Counts
    num_decisions: int = 0
    num_trades: int = 0
    
    # Performance
    total_pnl: float = 0.0
    win_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'split_id': self.split_id,
            'split_idx': self.split_idx,
            'train_start': self.train_start,
            'train_end': self.train_end,
            'test_start': self.test_start,
            'test_end': self.test_end,
            'num_decisions': self.num_decisions,
            'num_trades': self.num_trades,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
        }


@dataclass
class VizRun:
    """
    Top-level run metadata.
    """
    run_id: str = ""
    fingerprint: str = ""
    created_at: Optional[str] = None
    
    # Config snapshot
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Splits summary
    splits: List[VizSplit] = field(default_factory=list)
    
    # Totals
    total_decisions: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'fingerprint': self.fingerprint,
            'created_at': self.created_at,
            'config': self.config,
            'splits': [s.to_dict() for s in self.splits],
            'total_decisions': self.total_decisions,
            'total_trades': self.total_trades,
            'total_pnl': self.total_pnl,
        }
