"""
Viz Export
Exporter class that collects events during simulation and writes artifacts.
"""

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from src.viz.schema import (
    VizRun, VizSplit, VizDecision, VizTrade, VizOCO, VizFill, VizWindow, VizBarSeries
)
from src.viz.config import VizConfig
from src.datasets.decision_record import DecisionRecord
from src.datasets.trade_record import TradeRecord
from src.sim.oco import OCOBracket
from src.features.pipeline import FeatureBundle


class Exporter:
    """
    Collects events during backtest/simulation for viz export.
    
    Usage:
        exporter = Exporter(config, run_id="my_run")
        # During simulation:
        exporter.on_decision(decision, features)
        exporter.on_bracket_created(decision_id, bracket)
        exporter.on_order_fill(decision_id, fill_type, price, bar_idx, timestamp)
        exporter.on_trade_closed(trade)
        # At end:
        exporter.finalize(out_dir)
    """
    
    def __init__(
        self,
        config: VizConfig,
        run_id: Optional[str] = None,
        experiment_config: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.experiment_config = experiment_config or {}
        
        # Storage
        self.decisions: List[VizDecision] = []
        self.trades: List[VizTrade] = []
        self.splits: List[VizSplit] = []
        
        # Tracking
        self._decision_idx = 0
        self._trade_idx = 0
        self._current_split_id: Optional[str] = None
        
        # Temp storage for linking
        self._pending_ocos: Dict[str, VizOCO] = {}  # decision_id -> oco
        self._pending_fills: Dict[str, List[VizFill]] = {}  # decision_id -> fills
    
    def set_split(self, split_id: str, split_idx: int, train_start: str, train_end: str, test_start: str, test_end: str):
        """Start a new split."""
        self._current_split_id = split_id
        self.splits.append(VizSplit(
            split_id=split_id,
            split_idx=split_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))
    
    def on_decision(
        self,
        decision: DecisionRecord,
        features: Optional[FeatureBundle] = None,
        model_logits: Optional[List[float]] = None,
        model_probs: Optional[List[float]] = None
    ):
        """Record a decision point."""
        viz_decision = VizDecision(
            decision_id=decision.decision_id,
            timestamp=decision.timestamp.isoformat() if decision.timestamp else None,
            bar_idx=decision.bar_idx,
            index=self._decision_idx,
            scanner_id=decision.scanner_id,
            scanner_context=decision.scanner_context,
            action=decision.action.value,
            skip_reason=decision.skip_reason.value if decision.skip_reason else "",
            current_price=decision.current_price,
            atr=decision.atr,
            cf_outcome=decision.cf_outcome,
            cf_pnl_dollars=decision.cf_pnl_dollars,
        )
        
        # Add model outputs
        if self.config.include_model_outputs:
            viz_decision.model_logits = model_logits
            viz_decision.model_probs = model_probs
        
        # Add window data
        if self.config.include_windows and features:
            viz_decision.window = VizWindow(
                x_price_1m=features.x_price_1m.tolist() if features.x_price_1m is not None else [],
                x_price_5m=features.x_price_5m.tolist() if features.x_price_5m is not None else [],
                x_price_15m=features.x_price_15m.tolist() if features.x_price_15m is not None else [],
                x_context=features.x_context.tolist() if features.x_context is not None else [],
            )
        
        self.decisions.append(viz_decision)
        self._decision_idx += 1
        
        # Update split stats
        if self.splits:
            self.splits[-1].num_decisions += 1
    
    def on_bracket_created(self, decision_id: str, bracket: OCOBracket):
        """Record OCO bracket creation."""
        viz_oco = VizOCO(
            entry_price=bracket.entry_price,
            stop_price=bracket.stop_price,
            tp_price=bracket.tp_price,
            entry_type=bracket.config.entry_type,
            direction=bracket.config.direction,
            reference_type=bracket.config.reference.value,
            reference_value=bracket.reference_value,
            atr_at_creation=bracket.atr_at_creation,
            max_bars=bracket.config.max_bars,
            stop_atr=bracket.config.stop_atr,
            tp_multiple=bracket.config.tp_multiple,
        )
        
        self._pending_ocos[decision_id] = viz_oco
        
        # Link back to decision
        for d in reversed(self.decisions):
            if d.decision_id == decision_id:
                d.oco = viz_oco
                break
    
    def on_order_fill(
        self,
        decision_id: str,
        fill_type: str,
        price: float,
        bar_idx: int,
        timestamp: Optional[str] = None
    ):
        """Record an order fill."""
        fill = VizFill(
            order_id=f"{decision_id}_{fill_type}",
            fill_type=fill_type,
            price=price,
            bar_idx=bar_idx,
            timestamp=timestamp,
        )
        
        if decision_id not in self._pending_fills:
            self._pending_fills[decision_id] = []
        self._pending_fills[decision_id].append(fill)
    
    def on_trade_closed(self, trade: TradeRecord):
        """Record a completed trade."""
        viz_trade = VizTrade(
            trade_id=trade.trade_id,
            decision_id=trade.decision_id,
            index=self._trade_idx,
            direction=trade.direction,
            size=1,  # Fixed for now
            entry_time=trade.entry_time.isoformat() if trade.entry_time else None,
            entry_bar=trade.entry_bar,
            entry_price=trade.entry_price,
            exit_time=trade.exit_time.isoformat() if trade.exit_time else None,
            exit_bar=trade.exit_bar,
            exit_price=trade.exit_price,
            exit_reason=trade.exit_reason,
            outcome=trade.outcome,
            pnl_points=trade.pnl_points,
            pnl_dollars=trade.pnl_dollars,
            r_multiple=trade.r_multiple,
            bars_held=trade.bars_held,
            mae=trade.mae,
            mfe=trade.mfe,
        )
        
        # Attach fills
        if trade.decision_id in self._pending_fills:
            viz_trade.fills = self._pending_fills.pop(trade.decision_id)
        
        self.trades.append(viz_trade)
        self._trade_idx += 1
        
        # Update split stats
        if self.splits:
            self.splits[-1].num_trades += 1
            self.splits[-1].total_pnl += trade.pnl_dollars
    
    def finalize(self, out_dir: Path) -> Path:
        """Write all artifacts to disk."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Update split win rates
        for split in self.splits:
            split_trades = [t for t in self.trades if any(
                d.decision_id == t.decision_id for d in self.decisions
            )]
            wins = sum(1 for t in split_trades if t.outcome == 'WIN')
            split.win_rate = wins / len(split_trades) if split_trades else 0.0
        
        # Build run summary
        run = VizRun(
            run_id=self.run_id,
            fingerprint=self._compute_fingerprint(),
            created_at=datetime.now().isoformat(),
            config=self.experiment_config,
            splits=self.splits,
            total_decisions=len(self.decisions),
            total_trades=len(self.trades),
            total_pnl=sum(t.pnl_dollars for t in self.trades),
        )
        
        # Write run.json
        run_path = out_dir / "run.json"
        with open(run_path, 'w') as f:
            json.dump(run.to_dict(), f, indent=2)
        
        # Write decisions.jsonl
        decisions_path = out_dir / "decisions.jsonl"
        with open(decisions_path, 'w') as f:
            for d in self.decisions:
                f.write(json.dumps(d.to_dict()) + '\n')
        
        # Write trades.jsonl
        trades_path = out_dir / "trades.jsonl"
        with open(trades_path, 'w') as f:
            for t in self.trades:
                f.write(json.dumps(t.to_dict()) + '\n')
        
        # Write manifest.json
        manifest = {
            'run_id': self.run_id,
            'created_at': run.created_at,
            'files': {
                'run': 'run.json',
                'decisions': 'decisions.jsonl',
                'trades': 'trades.jsonl',
            },
            'counts': {
                'decisions': len(self.decisions),
                'trades': len(self.trades),
                'splits': len(self.splits),
            },
            'checksums': {
                'run': self._file_checksum(run_path),
                'decisions': self._file_checksum(decisions_path),
                'trades': self._file_checksum(trades_path),
            }
        }
        
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Viz export complete: {out_dir}")
        print(f"  Decisions: {len(self.decisions)}")
        print(f"  Trades: {len(self.trades)}")
        
        return out_dir
    
    def _compute_fingerprint(self) -> str:
        """Compute a fingerprint for this run."""
        content = json.dumps(self.experiment_config, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _file_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
