    def write(self, record: DecisionRecord):
        """Add a record to the buffer."""
        self._buffer.append(record)
        self._total_records += 1
        
        if len(self._buffer) >= self.records_per_shard:
            self._flush_shard()
    
    def write_batch(self, records: List[DecisionRecord]):
        """Write multiple records."""
        for record in records:
            self.write(record)
    
    def _flush_shard(self):
        """Write buffered records to a shard file."""
        if not self._buffer:
            return
        
        # Convert to DataFrame
        rows = []
        array_refs = []
        
        for i, record in enumerate(self._buffer):
            row = record.to_dict()
            
            # Store numpy arrays separately
            record_id = f"{self.experiment_id}_{self._shard_idx}_{i}"
            
            if record.x_price_1m is not None:
                arr_path = self._save_array(record.x_price_1m, f"{record_id}_x_price_1m")
                row['x_price_1m_path'] = str(arr_path)
            
            if record.x_price_5m is not None:
                arr_path = self._save_array(record.x_price_5m, f"{record_id}_x_price_5m")
                row['x_price_5m_path'] = str(arr_path)
            
            if record.x_price_15m is not None:
                arr_path = self._save_array(record.x_price_15m, f"{record_id}_x_price_15m")
                row['x_price_15m_path'] = str(arr_path)
            
            if record.x_context is not None:
                arr_path = self._save_array(record.x_context, f"{record_id}_x_context")
                row['x_context_path'] = str(arr_path)
            
            rows.append(row)
        
        # Write parquet
        df = pd.DataFrame(rows)
        shard_path = self.output_dir / f"shard_{self._shard_idx:04d}.parquet"
        df.to_parquet(shard_path)
        
        # Clear buffer
        self._buffer = []
        self._shard_idx += 1
    
    def _save_array(self, arr: np.ndarray, name: str) -> Path:
        """Save numpy array to file."""
        path = self._arrays_dir / f"{name}.npy"
        np.save(path, arr)
        return path
    
    def close(self):
        """Flush remaining records and write metadata."""
        self._flush_shard()
        
        # Write manifest
        manifest = {
            'experiment_id': self.experiment_id,
            'total_records': self._total_records,
            'num_shards': self._shard_idx,
            'records_per_shard': self.records_per_shard,
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

```

### src/eval/__init__.py

```python
# Eval module
"""Trade-quality metrics and breakdowns."""

```

### src/eval/breakdown.py

```python
"""
Breakdown Analysis
Metrics by setup, time of day, volatility regime.
"""

from typing import List, Dict
from collections import defaultdict

from src.datasets.decision_record import DecisionRecord
from src.eval.metrics import TradeMetrics, compute_from_records


def breakdown_by_scanner(
    records: List[DecisionRecord]
) -> Dict[str, TradeMetrics]:
    """Metrics grouped by scanner/setup type."""
    by_scanner = defaultdict(list)
    
    for r in records:
        by_scanner[r.scanner_id].append(r)
    
    return {k: compute_from_records(v) for k, v in by_scanner.items()}


def breakdown_by_hour(
    records: List[DecisionRecord]
) -> Dict[int, TradeMetrics]:
    """Metrics grouped by hour (NY time)."""
    by_hour = defaultdict(list)
    
    for r in records:
        if r.timestamp:
            hour = r.timestamp.hour
            by_hour[hour].append(r)
    
    return {k: compute_from_records(v) for k, v in sorted(by_hour.items())}


def breakdown_by_day(
    records: List[DecisionRecord]
) -> Dict[int, TradeMetrics]:
    """Metrics grouped by day of week (0=Mon, 4=Fri)."""
    by_day = defaultdict(list)
    
    for r in records:
        if r.timestamp:
            dow = r.timestamp.weekday()
            by_day[dow].append(r)
    
    return {k: compute_from_records(v) for k, v in sorted(by_day.items())}


def breakdown_by_action(
    records: List[DecisionRecord]
) -> Dict[str, dict]:
    """
    Analyze by action taken.
    
    Returns stats for:
    - Trades taken
    - Trades skipped (broken down by skip reason)
    - Skipped good (would have lost)
    - Skipped bad (would have won)
    """
    taken = [r for r in records if r.is_trade()]
    skipped = [r for r in records if r.was_skipped()]
    
    skipped_good = [r for r in skipped if r.is_good_skip()]
    skipped_bad = [r for r in skipped if r.is_bad_skip()]
    
    return {
        'taken': {
            'count': len(taken),
            'metrics': compute_from_records(taken),
        },
        'skipped': {
            'count': len(skipped),
            'good_skips': len(skipped_good),
            'bad_skips': len(skipped_bad),
            'good_skip_rate': len(skipped_good) / len(skipped) if skipped else 0,
        },
        'by_skip_reason': _count_skip_reasons(skipped),
    }


def _count_skip_reasons(records: List[DecisionRecord]) -> Dict[str, int]:
    """Count records by skip reason."""
    counts = defaultdict(int)
    for r in records:
        counts[r.skip_reason.value] += 1
    return dict(counts)


def print_breakdown_summary(
    records: List[DecisionRecord],
    title: str = "Breakdown Summary"
):
    """Print formatted breakdown summary."""
    print(f"\n{'='*50}")
    print(title)
    print('='*50)
    
    # Overall
    overall = compute_from_records(records)
    print(f"\nOverall: {overall.total_trades} records, "
          f"{overall.win_rate:.1%} WR, "
          f"${overall.total_pnl:.2f} PnL")
    
    # By hour
    print("\nBy Hour (NY):")
    by_hour = breakdown_by_hour(records)
    for hour, m in by_hour.items():
        if m.total_trades > 0:
            print(f"  {hour:02d}:00 - {m.total_trades:4d} trades, "
                  f"{m.win_rate:.1%} WR, ${m.total_pnl:7.2f}")
    
    # By action
    print("\nBy Action:")
    by_action = breakdown_by_action(records)
    print(f"  Taken: {by_action['taken']['count']}")
    print(f"  Skipped: {by_action['skipped']['count']} "
          f"({by_action['skipped']['good_skips']} good, "
          f"{by_action['skipped']['bad_skips']} bad)")

```

### src/eval/mae_mfe.py

```python
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

```

### src/eval/metrics.py

```python
"""
Trade Metrics
Expectancy, win rate, payoff ratio, drawdown.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from src.datasets.trade_record import TradeRecord
from src.datasets.decision_record import DecisionRecord


@dataclass
class TradeMetrics:
    """Comprehensive trade metrics."""
    total_trades: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    
    # PnL
    total_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    
    # Risk-adjusted
    payoff_ratio: float      # avg_win / avg_loss
    expectancy: float        # (WR * avg_win) - ((1-WR) * avg_loss)
    profit_factor: float     # gross_win / gross_loss
    
    # Drawdown
    max_drawdown: float
    max_drawdown_pct: float
    
    # Additional
    avg_bars_held: float
    avg_r_multiple: float


def compute_trade_metrics(trades: List[TradeRecord]) -> TradeMetrics:
    """Compute metrics from trade records."""
    if not trades:
        return TradeMetrics(
            total_trades=0, wins=0, losses=0, timeouts=0, win_rate=0,
            total_pnl=0, avg_pnl=0, avg_win=0, avg_loss=0,
            payoff_ratio=0, expectancy=0, profit_factor=0,
            max_drawdown=0, max_drawdown_pct=0,
            avg_bars_held=0, avg_r_multiple=0
        )
    
    # Basic counts
    wins = [t for t in trades if t.outcome == 'WIN']
    losses = [t for t in trades if t.outcome == 'LOSS']
    timeouts = [t for t in trades if t.outcome == 'TIMEOUT']
    
    win_count = len(wins)
    loss_count = len(losses)
    total = len(trades)
    
    win_rate = win_count / total if total > 0 else 0
    
    # PnL
    total_pnl = sum(t.pnl_dollars for t in trades)
    avg_pnl = total_pnl / total if total > 0 else 0
    
    avg_win = np.mean([t.pnl_dollars for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t.pnl_dollars) for t in losses]) if losses else 0
    
    # Risk-adjusted
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    gross_win = sum(t.pnl_dollars for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')
    
    # Drawdown
    equity_curve = np.cumsum([t.pnl_dollars for t in trades])
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = peak - equity_curve
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    max_dd_pct = max_dd / np.max(peak) if np.max(peak) > 0 else 0
    
    # Additional
    avg_bars = np.mean([t.bars_held for t in trades])
    avg_r = np.mean([t.r_multiple for t in trades if t.r_multiple != 0])
    
    return TradeMetrics(
        total_trades=total,
        wins=win_count,
        losses=loss_count,
        timeouts=len(timeouts),
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        expectancy=expectancy,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_bars_held=avg_bars,
        avg_r_multiple=avg_r if not np.isnan(avg_r) else 0,
    )


def compute_from_records(records: List[DecisionRecord]) -> TradeMetrics:
    """Compute metrics from decision records using counterfactual outcomes."""
    if not records:
        return compute_trade_metrics([])
    
    # Convert counterfactual outcomes to simple format
    class SimpleRecord:
        def __init__(self, r: DecisionRecord):
            self.outcome = r.cf_outcome
            self.pnl_dollars = r.cf_pnl_dollars
            self.bars_held = r.cf_bars_held
            self.r_multiple = 0  # Not tracked in decision records
    
    simple = [SimpleRecord(r) for r in records if r.cf_outcome in ['WIN', 'LOSS', 'TIMEOUT']]
    
    # Reuse computation
    return compute_trade_metrics(simple)

```

### src/experiments/__init__.py

```python
# Experiments module
"""Experiment framework - configs, sweeps, reports."""

```

### src/experiments/config.py

```python
"""
Experiment Configuration
Central config dataclass for experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from src.features.pipeline import FeatureConfig
from src.labels.labeler import LabelConfig
from src.sim.oco import OCOConfig
from src.sim.bar_fill_model import BarFillConfig
from src.sim.costs import CostModel, DEFAULT_COSTS
from src.models.train import TrainConfig
from src.datasets.schema import DatasetSchema


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.
    
    Single source of truth for all parameters.
    """
    # Identification
    name: str = "experiment"
    description: str = ""
    
    # Data range
    start_date: str = ""
    end_date: str = ""
    timeframe: str = "1m"
    
    # Scanner
    scanner_id: str = "always"
    scanner_params: Dict[str, Any] = field(default_factory=dict)
    
    # Features
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Labels
    label_config: LabelConfig = field(default_factory=LabelConfig)
    oco_config: OCOConfig = field(default_factory=OCOConfig)
    
    # Simulation
    fill_config: BarFillConfig = field(default_factory=BarFillConfig)
    cost_model: CostModel = field(default_factory=lambda: DEFAULT_COSTS)
    
    # Training
    train_config: TrainConfig = field(default_factory=TrainConfig)
    
    # Schema
    schema: DatasetSchema = field(default_factory=DatasetSchema)
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'timeframe': self.timeframe,
            'scanner_id': self.scanner_id,
            'scanner_params': self.scanner_params,
            'feature_config': self.feature_config.to_dict(),
            'oco_config': self.oco_config.to_dict(),
            'fill_config': self.fill_config.to_dict(),
            'train_config': self.train_config.to_dict(),
            'schema': self.schema.to_dict(),
            'seed': self.seed,
        }
    
    def to_cli_args(self) -> List[str]:
        """Generate CLI arguments."""
        args = [
            '--name', self.name,
            '--start-date', self.start_date,
            '--end-date', self.end_date,
            '--timeframe', self.timeframe,
            '--scanner', self.scanner_id,
            '--seed', str(self.seed),
        ]
        args.extend(self.oco_config.to_cli_args())
        return args
    
    def save(self, path: Path):
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        config = cls(
            name=data.get('name', 'experiment'),
            description=data.get('description', ''),
            start_date=data.get('start_date', ''),
            end_date=data.get('end_date', ''),
            timeframe=data.get('timeframe', '1m'),
            scanner_id=data.get('scanner_id', 'always'),
            seed=data.get('seed', 42),
        )
        
        # Load nested configs if present
        if 'oco_config' in data:
            oco = data['oco_config']
            config.oco_config = OCOConfig(
                direction=oco.get('direction', 'LONG'),
                tp_multiple=oco.get('tp_multiple', 1.4),
                stop_atr=oco.get('stop_atr', 1.0),
            )
        
        return config

```

### src/experiments/fingerprint.py

```python
"""
Experiment Fingerprint
SHA256 hash for reproducibility tracking.
"""

import hashlib
import json
from typing import Any

from src.experiments.config import ExperimentConfig


def compute_fingerprint(config: ExperimentConfig) -> str:
    """
    Compute SHA256 fingerprint of experiment configuration.
    
    Ensures reproducibility tracking - same config = same fingerprint.
    
    Returns:
        First 16 characters of SHA256 hash
    """
    # Serialize config to deterministic JSON
    config_dict = config.to_dict()
    
    # Sort keys for determinism
    json_str = json.dumps(config_dict, sort_keys=True, default=str)
    
    # Compute hash
    hash_obj = hashlib.sha256(json_str.encode())
    
    return hash_obj.hexdigest()[:16]


def verify_fingerprint(
    config: ExperimentConfig,
    expected: str
) -> bool:
    """
    Verify that config matches expected fingerprint.
    """
    actual = compute_fingerprint(config)
    return actual == expected

```

### src/experiments/report.py

```python
"""
Report Generation
Generate markdown reports from experiment results.
"""

from pathlib import Path
from typing import List
import pandas as pd

from src.experiments.runner import ExperimentResult
from src.config import RESULTS_DIR


def generate_report(
    results: List[ExperimentResult],
    output_path: Path = None
) -> Path:
    """
    Generate markdown report from experiment results.
    """
    output_path = output_path or RESULTS_DIR / "report.md"
    
    lines = [
        "# Experiment Report",
        "",
        f"Generated: {pd.Timestamp.now()}",
        "",
        f"Total experiments: {len(results)}",
        "",
        "## Summary",
        "",
    ]
    
    # Create summary table
    lines.extend([
        "| Name | Records | WIN | LOSS | Best Val Loss | Best Epoch |",
        "|------|---------|-----|------|---------------|------------|",
    ])
    
    for r in results:
        val_loss = f"{r.train_result.best_val_loss:.4f}" if r.train_result else "N/A"
        epoch = str(r.train_result.best_epoch) if r.train_result else "N/A"
        
        lines.append(
            f"| {r.config.name} | {r.total_records} | {r.win_records} | "
            f"{r.loss_records} | {val_loss} | {epoch} |"
        )
    
    lines.extend(["", "## Configuration Details", ""])
    
    for r in results:
        lines.extend([
            f"### {r.config.name}",
            "",
            f"- Fingerprint: `{r.fingerprint}`",
            f"- Scanner: {r.config.scanner_id}",
            f"- Direction: {r.config.oco_config.direction}",
            f"- TP Multiple: {r.config.oco_config.tp_multiple}",
            f"- Stop ATR: {r.config.oco_config.stop_atr}",
            f"- Records: {r.total_records} ({r.win_records}W / {r.loss_records}L)",
            "",
        ])
    
    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Report saved to {output_path}")
    return output_path

```

### src/experiments/runner.py

```python
"""
Experiment Runner
Run a single experiment end-to-end.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import uuid

from src.experiments.config import ExperimentConfig
from src.experiments.fingerprint import compute_fingerprint
from src.experiments.splits import generate_walk_forward_splits, WalkForwardConfig

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes

from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features
from src.policy.scanners import get_scanner
from src.policy.filters import DEFAULT_FILTERS
from src.policy.cooldown import CooldownManager
from src.policy.actions import Action, SkipReason

from src.labels.labeler import Labeler
from src.datasets.decision_record import DecisionRecord
from src.datasets.writer import ShardWriter
from src.datasets.reader import create_dataloader

from src.models.fusion import FusionModel
from src.models.train import train_model, TrainResult

from src.config import PROCESSED_DIR, SHARDS_DIR, RESULTS_DIR


@dataclass
class ExperimentResult:
    """Result of running an experiment."""
    config: ExperimentConfig
    fingerprint: str
    
    # Dataset stats
    total_records: int
    win_records: int
    loss_records: int
    timeout_records: int
    
    # Training results
    train_result: Optional[TrainResult] = None
    
    # Created at
    created_at: pd.Timestamp = None
    
    def to_dict(self):
        return {
            'fingerprint': self.fingerprint,
            'total_records': self.total_records,
            'win_records': self.win_records,
            'loss_records': self.loss_records,
            'timeout_records': self.timeout_records,
            'best_val_loss': self.train_result.best_val_loss if self.train_result else None,
            'best_epoch': self.train_result.best_epoch if self.train_result else None,
        }


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Run a complete experiment:
    
    1. Load data
    2. Generate decision records at scanner points
    3. Label all records with counterfactual outcomes
    4. Write to shards
    5. Train model
    6. Return results
    """
    print(f"Running experiment: {config.name}")
    
    # Compute fingerprint
    fingerprint = compute_fingerprint(config)
    print(f"Fingerprint: {fingerprint}")
    
    # 1. Load and prepare data
    print("Loading data...")
    df = load_continuous_contract()
    
    # Filter by date range
    if config.start_date:
        df = df[df['time'] >= config.start_date]
    if config.end_date:
        df = df[df['time'] <= config.end_date]
    
    df = df.reset_index(drop=True)
    print(f"Data range: {df['time'].min()} to {df['time'].max()}")
    print(f"Total bars: {len(df)}")
    
    # Resample to higher timeframes
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    
    # 2. Generate decision records
    print("Generating decision records...")
    stepper = MarketStepper(df, start_idx=200, end_idx=len(df) - 200)
    scanner = get_scanner(config.scanner_id, **config.scanner_params)
    labeler = Labeler(config.label_config)
    cooldown = CooldownManager()
    
    records: List[DecisionRecord] = []
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        # Compute features
        features = compute_features(
            stepper,
            config.feature_config,
            df_5m=df_5m,
            df_15m=df_15m,
        )
        
        # Check if scanner triggers
        scan_result = scanner.scan(features.market_state, features)
        if not scan_result.triggered:
            continue
        
        # Check filters
        filter_result = DEFAULT_FILTERS.check(features)
        if not filter_result.passed:
            skip_reason = SkipReason.FILTER_BLOCK
        # Check cooldown
        elif cooldown.is_on_cooldown(step.bar_idx, features.timestamp)[0]:
            skip_reason = SkipReason.COOLDOWN
        else:
            skip_reason = SkipReason.NOT_SKIPPED
        
        # Create record
        record = DecisionRecord(
            timestamp=features.timestamp,
            bar_idx=step.bar_idx,
            decision_id=str(uuid.uuid4())[:8],
            scanner_id=config.scanner_id,
            action=Action.NO_TRADE if skip_reason != SkipReason.NOT_SKIPPED else Action.PLACE_ORDER,
            skip_reason=skip_reason,
            x_price_1m=features.x_price_1m,
            x_price_5m=features.x_price_5m,
            x_price_15m=features.x_price_15m,
            x_context=features.x_context,
            current_price=features.current_price,
            atr=features.atr,
        )
        
        # 3. Label with counterfactual outcome
        cf_label = labeler.label_decision_point(df, step.bar_idx, features.atr)
        record.cf_outcome = cf_label.outcome
        record.cf_pnl = cf_label.pnl
        record.cf_pnl_dollars = cf_label.pnl_dollars
        record.cf_mae = cf_label.mae
        record.cf_mfe = cf_label.mfe
        record.cf_mae_atr = cf_label.mae_atr
        record.cf_mfe_atr = cf_label.mfe_atr
        record.cf_bars_held = cf_label.bars_held
        
        records.append(record)
        
        # Update cooldown if trade placed
        if record.action == Action.PLACE_ORDER:
            cooldown.record_trade(step.bar_idx, cf_label.outcome, features.timestamp)
    
    print(f"Generated {len(records)} decision records")
    
    # Count outcomes
    win_count = sum(1 for r in records if r.cf_outcome == 'WIN')
    loss_count = sum(1 for r in records if r.cf_outcome == 'LOSS')
    timeout_count = sum(1 for r in records if r.cf_outcome == 'TIMEOUT')
    
    print(f"Outcomes: {win_count} WIN, {loss_count} LOSS, {timeout_count} TIMEOUT")
    
    # 4. Write to shards
    shard_dir = SHARDS_DIR / fingerprint
    print(f"Writing shards to {shard_dir}")
    
    with ShardWriter(shard_dir, experiment_id=fingerprint) as writer:
        for record in records:
            writer.write(record)
    
    # 5. Train model (if enough data)
    train_result = None
    if win_count + loss_count >= 100:
        print("Training model...")
        
        # Create dataloaders (simple 80/20 split for now)
        loader = create_dataloader(shard_dir, batch_size=config.train_config.batch_size)
        
        # Split into train/val
        dataset = loader.dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        from torch.utils.data import random_split, DataLoader
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=config.train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.train_config.batch_size)
        
        # Create model
        model = FusionModel(
            context_dim=config.schema.x_context_dim,
            num_classes=2,  # WIN/LOSS
            dropout=config.train_config.dropout,
        )
        
        # Train
        train_result = train_model(model, train_loader, val_loader, config.train_config)
    
    # 6. Return results
    return ExperimentResult(
        config=config,
        fingerprint=fingerprint,
        total_records=len(records),
        win_records=win_count,
        loss_records=loss_count,
        timeout_records=timeout_count,
        train_result=train_result,
        created_at=pd.Timestamp.now(),
    )

```

### src/experiments/splits.py

```python
"""
Walk-Forward Splits
Time-series cross-validation with embargo.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WalkForwardConfig:
    """Walk-forward split configuration."""
    train_weeks: int = 3
    test_weeks: int = 1
    embargo_bars: int = 100  # Gap to prevent feature leakage
    min_train_records: int = 1000


@dataclass
class Split:
    """Single train/test split."""
    split_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    def __repr__(self):
        return (f"Split(train={self.train_start.date()}→{self.train_end.date()}, "
                f"test={self.test_start.date()}→{self.test_end.date()})")


def generate_walk_forward_splits(
    df: pd.DataFrame,
    config: WalkForwardConfig,
    time_col: str = 'time'
) -> List[Split]:
    """
    Generate walk-forward splits with embargo gaps.
    
    Layout:
    |---train---|--embargo--|---test---|---train---|--embargo--|---test---|
    
    Args:
        df: DataFrame with time column
        config: Split configuration
        time_col: Name of time column
        
    Returns:
        List of Split objects
    """
    times = pd.to_datetime(df[time_col]).sort_values()
    start_time = times.min()
    end_time = times.max()
    
    train_duration = pd.Timedelta(weeks=config.train_weeks)
    test_duration = pd.Timedelta(weeks=config.test_weeks)
    embargo_duration = pd.Timedelta(minutes=config.embargo_bars)  # Assuming 1m bars
    
    splits = []
    current_start = start_time
    split_idx = 0
    
    while True:
        train_end = current_start + train_duration
        embargo_start = train_end
        embargo_end = embargo_start + embargo_duration
        test_start = embargo_end
