# Git Diff Report

**Generated**: Wed, Dec 17, 2025  5:52:19 PM

**Local Branch**: pr-4

**Comparing Against**: origin/pr-4

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M gitrdif.sh
 M mlang2/src/experiments/runner.py
?? gitrdiff.md
?? mlang2/scripts/
?? mlang2/src/viz/
?? mlang2/tests/
```

### Uncommitted Diff

```diff
diff --git a/gitrdif.sh b/gitrdif.sh
index 41ff6c5..5f97ba9 100644
--- a/gitrdif.sh
+++ b/gitrdif.sh
@@ -57,6 +57,35 @@ echo "Generating diff: local $BRANCH vs $REMOTE_BRANCH..."
         echo ""
     fi
     
+    # NEW: Show contents of untracked files (new files not yet staged)
+    UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null)
+    if [ -n "$UNTRACKED" ]; then
+        echo "### New Untracked Files"
+        echo ""
+        for file in $UNTRACKED; do
+            # Skip binary files and very large files
+            if [ -f "$file" ] && file "$file" | grep -q text; then
+                LINES=$(wc -l < "$file" 2>/dev/null || echo "0")
+                if [ "$LINES" -lt 500 ]; then
+                    echo "#### \`$file\`"
+                    echo ""
+                    echo '```'
+                    cat "$file" 2>/dev/null
+                    echo '```'
+                    echo ""
+                else
+                    echo "#### \`$file\` ($LINES lines - truncated)"
+                    echo ""
+                    echo '```'
+                    head -100 "$file" 2>/dev/null
+                    echo "... ($LINES total lines)"
+                    echo '```'
+                    echo ""
+                fi
+            fi
+        done
+    fi
+    
     echo "---"
     echo ""
     
diff --git a/mlang2/src/experiments/runner.py b/mlang2/src/experiments/runner.py
index b474314..454ff61 100644
--- a/mlang2/src/experiments/runner.py
+++ b/mlang2/src/experiments/runner.py
@@ -7,9 +7,12 @@ import pandas as pd
 import numpy as np
 from pathlib import Path
 from dataclasses import dataclass
-from typing import List, Optional
+from typing import List, Optional, TYPE_CHECKING
 import uuid
 
+if TYPE_CHECKING:
+    from src.viz.export import Exporter
+
 from src.experiments.config import ExperimentConfig
 from src.experiments.fingerprint import compute_fingerprint
 from src.experiments.splits import generate_walk_forward_splits, WalkForwardConfig
@@ -65,7 +68,10 @@ class ExperimentResult:
         }
 
 
-def run_experiment(config: ExperimentConfig) -> ExperimentResult:
+def run_experiment(
+    config: ExperimentConfig,
+    exporter: Optional['Exporter'] = None
+) -> ExperimentResult:
     """
     Run a complete experiment:
     
@@ -75,6 +81,10 @@ def run_experiment(config: ExperimentConfig) -> ExperimentResult:
     4. Write to shards
     5. Train model
     6. Return results
+    
+    Args:
+        config: Experiment configuration
+        exporter: Optional Exporter for viz output
     """
     print(f"Running experiment: {config.name}")
     
@@ -167,6 +177,10 @@ def run_experiment(config: ExperimentConfig) -> ExperimentResult:
         
         records.append(record)
         
+        # === VIZ EXPORT HOOK ===
+        if exporter:
+            exporter.on_decision(record, features)
+        
         # Update cooldown if trade placed
         if record.action == Action.PLACE_ORDER:
             cooldown.record_trade(step.bar_idx, cf_label.outcome, features.timestamp)
```

### New Untracked Files

#### `gitrdiff.md`

```
```

#### `mlang2/scripts/run_walkforward_viz.py`

```
#!/usr/bin/env python
"""
Walk-Forward Viz Export CLI
Run walk-forward experiments and export visualization artifacts.

Usage:
    python scripts/run_walkforward_viz.py --config experiment.json --out results/viz/my_run/
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment
from src.experiments.splits import generate_walk_forward_splits, WalkForwardConfig, Split
from src.viz.export import Exporter
from src.viz.config import VizConfig
from src.data.loader import load_continuous_contract
from src.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward experiment with viz export")
    parser.add_argument("--config", type=str, help="Path to experiment config JSON")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: auto-generate)")
    parser.add_argument("--include-full-series", action="store_true", help="Include full OHLCV series")
    parser.add_argument("--no-windows", action="store_true", help="Exclude price windows")
    
    # Quick-run params (if no config file)
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--end-date", type=str, default="2025-05-04", help="End date")
    parser.add_argument("--scanner", type=str, default="interval", help="Scanner ID")
    parser.add_argument("--train-weeks", type=int, default=3, help="Train weeks per split")
    parser.add_argument("--test-weeks", type=int, default=1, help="Test weeks per split")
    
    args = parser.parse_args()
    
    # Load or create experiment config
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = ExperimentConfig(**config_dict)
    else:
        # Use CLI params for quick run
        config = ExperimentConfig(
            name="walkforward_viz",
            start_date=args.start_date,
            end_date=args.end_date,
            scanner_id=args.scanner,
        )
    
    # Setup viz config
    viz_config = VizConfig(
        include_full_series=args.include_full_series,
        include_windows=not args.no_windows,
    )
    
    # Setup output directory
    run_id = args.run_id or config.name
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "viz" / run_id
    
    # Create exporter
    exporter = Exporter(
        config=viz_config,
        run_id=run_id,
        experiment_config=config.to_dict() if hasattr(config, 'to_dict') else {},
    )
    
    print("=" * 60)
    print(f"Walk-Forward Viz Export")
    print(f"Run ID: {run_id}")
    print(f"Output: {out_dir}")
    print("=" * 60)
    
    # Run experiment with exporter
    result = run_experiment(config, exporter=exporter)
    
    print(f"\nExperiment complete:")
    print(f"  Total records: {result.total_records}")
    print(f"  Win: {result.win_records}, Loss: {result.loss_records}")
    
    # Finalize export
    exporter.finalize(out_dir)
    
    print(f"\nViz artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
```

#### `mlang2/src/viz/__init__.py`

```
"""
Viz Package
Export pipelines for React UI visualization.
"""
```

#### `mlang2/src/viz/config.py`

```
"""
Viz Config
Configuration for visualization export.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VizConfig:
    """Configuration for viz export."""
    
    # What to include
    include_full_series: bool = False  # Full OHLCV for overview mode
    include_windows: bool = True       # x_price windows at decision time
    include_model_outputs: bool = True # Logits/probabilities
    
    # Window settings
    window_lookback_1m: int = 120
    window_lookback_5m: int = 24
    window_lookback_15m: int = 8
    
    # Output format
    output_format: str = "jsonl"  # 'json' or 'jsonl'
    compress: bool = False
    
    def to_dict(self) -> dict:
        return {
            'include_full_series': self.include_full_series,
            'include_windows': self.include_windows,
            'include_model_outputs': self.include_model_outputs,
            'window_lookback_1m': self.window_lookback_1m,
            'window_lookback_5m': self.window_lookback_5m,
            'window_lookback_15m': self.window_lookback_15m,
            'output_format': self.output_format,
            'compress': self.compress,
        }
```

#### `mlang2/src/viz/export.py`

```
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
```

#### `mlang2/src/viz/schema.py`

```
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
```

#### `mlang2/tests/test_viz_export.py`

```
"""
Tests for Viz Export Pipeline
"""

import sys
import json
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz.schema import VizDecision, VizTrade, VizOCO, VizWindow, VizRun
from src.viz.export import Exporter
from src.viz.config import VizConfig


class TestVizSchema(unittest.TestCase):
    """Test viz schema dataclasses."""
    
    def test_viz_decision_to_dict(self):
        """VizDecision should produce valid JSON-serializable dict."""
        decision = VizDecision(
            decision_id="abc123",
            timestamp="2025-03-17T10:30:00",
            bar_idx=100,
            index=0,
            scanner_id="interval_60",
            action="PLACE_ORDER",
            current_price=5000.0,
            atr=10.0,
            cf_outcome="WIN",
            cf_pnl_dollars=250.0,
        )
        
        d = decision.to_dict()
        
        # Should be JSON-serializable
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)
        
        # Key fields should be present
        self.assertEqual(d['decision_id'], 'abc123')
        self.assertEqual(d['action'], 'PLACE_ORDER')

    def test_viz_trade_to_dict(self):
        """VizTrade should produce valid JSON-serializable dict."""
        trade = VizTrade(
            trade_id="trade_001",
            decision_id="abc123",
            direction="LONG",
            entry_price=5000.0,
            exit_price=5014.0,
            pnl_dollars=70.0,
            outcome="WIN",
        )
        
        d = trade.to_dict()
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)
        self.assertEqual(d['outcome'], 'WIN')


class TestExporter(unittest.TestCase):
    """Test Exporter class."""
    
    def test_exporter_finalize_creates_files(self):
        """Exporter.finalize() should create manifest, run, decisions, trades files."""
        config = VizConfig(include_windows=False)
        exporter = Exporter(config, run_id="test_run")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "viz_output"
            exporter.finalize(out_dir)
            
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertTrue((out_dir / "run.json").exists())
            self.assertTrue((out_dir / "decisions.jsonl").exists())
            self.assertTrue((out_dir / "trades.jsonl").exists())

    def test_exporter_decision_trade_link(self):
        """Decisions and trades should be linkable via decision_id."""
        from src.datasets.decision_record import DecisionRecord
        from src.datasets.trade_record import TradeRecord
        from src.policy.actions import Action, SkipReason
        import pandas as pd
        
        config = VizConfig(include_windows=False)
        exporter = Exporter(config, run_id="test_link")
        
        # Add a decision
        decision = DecisionRecord(
            timestamp=pd.Timestamp("2025-03-17 10:30:00"),
            bar_idx=100,
            decision_id="link_test_001",
            scanner_id="test",
            action=Action.PLACE_ORDER,
            skip_reason=SkipReason.NOT_SKIPPED,
            current_price=5000.0,
            atr=10.0,
            cf_outcome="WIN",
            cf_pnl_dollars=100.0,
        )
        exporter.on_decision(decision, None)
        
        # Add a trade
        trade = TradeRecord(
            trade_id="trade_link_001",
            decision_id="link_test_001",
            entry_price=5000.0,
            exit_price=5010.0,
            pnl_dollars=50.0,
            outcome="WIN",
        )
        exporter.on_trade_closed(trade)
        
        # Check linkage
        self.assertEqual(len(exporter.decisions), 1)
        self.assertEqual(len(exporter.trades), 1)
        self.assertEqual(exporter.decisions[0].decision_id, exporter.trades[0].decision_id)


class TestVizConfig(unittest.TestCase):
    """Test VizConfig."""
    
    def test_config_defaults(self):
        """VizConfig should have sensible defaults."""
        config = VizConfig()
        self.assertFalse(config.include_full_series)
        self.assertTrue(config.include_windows)
        self.assertEqual(config.output_format, "jsonl")


if __name__ == "__main__":
    unittest.main()
```

---

## Commits Ahead (local changes not on remote)

```
```

## Commits Behind (remote changes not pulled)

```
```

---

## File Changes (what you'd get from remote)

```
```

---

## Full Diff (green = new on remote, red = removed on remote)

```diff
```
