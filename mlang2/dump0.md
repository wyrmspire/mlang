# MLang2 Project Code Dump
Generated: Wed, Dec 17, 2025  5:18:44 PM

## Project Structure
```
cnn_results.txt
printcode.sh
README.md
src/__init__.py
src/config.py
src/datasets/__init__.py
src/datasets/decision_record.py
src/datasets/reader.py
src/datasets/schema.py
src/datasets/trade_record.py
src/datasets/writer.py
src/eval/__init__.py
src/eval/breakdown.py
src/eval/mae_mfe.py
src/eval/metrics.py
src/experiments/__init__.py
src/experiments/config.py
src/experiments/fingerprint.py
src/experiments/report.py
src/experiments/runner.py
src/experiments/splits.py
src/experiments/sweep.py
src/features/__init__.py
src/features/context.py
src/features/indicators.py
src/features/levels.py
src/features/pipeline.py
src/features/state.py
src/features/time_features.py
src/labels/__init__.py
src/labels/counterfactual.py
src/labels/future_window.py
src/labels/labeler.py
src/labels/trade_outcome.py
src/models/__init__.py
src/models/context_mlp.py
src/models/encoders.py
src/models/fusion.py
src/models/heads.py
src/models/train.py
src/policy/__init__.py
src/policy/actions.py
src/policy/cooldown.py
src/policy/filters.py
src/policy/library/__init__.py
src/policy/library/mid_day_reversal.py
src/policy/scanners.py
src/sim/__init__.py
src/sim/account.py
src/sim/bar_fill_model.py
src/sim/costs.py
src/sim/execution.py
src/sim/oco.py
src/sim/stepper.py
src/skills/__init__.py
src/skills/data.py
src/skills/model.py
src/skills/registry.py
src/skills/research.py
test_cnn_filter.py
test_output.txt
test_walkforward.py
verify_modular_strategies.py
verify_skills.py
```

## Source Files

### printcode.sh

```bash
#!/bin/bash
# =============================================================================
# printcode.sh - Dump project code to markdown files
# =============================================================================
# 
# Outputs project structure and code to dump1.md, dump2.md, etc.
# Each file contains ~1000 lines.
#
# Excludes:
#   - __pycache__
#   - .git
#   - data/ (raw data files)
#   - cache/
#   - shards/
#   - models/ (trained weights)
#   - results/
#   - *.parquet, *.pth, *.json (data files)
#   - *.pyc
#
# Usage: ./printcode.sh
# =============================================================================

set -e

OUTPUT_PREFIX="dump"
LINES_PER_FILE=1000
TEMP_FILE=$(mktemp)

# Project root (where this script lives)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "# MLang2 Project Code Dump" > "$TEMP_FILE"
echo "Generated: $(date)" >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"

# Project structure
echo "## Project Structure" >> "$TEMP_FILE"
echo '```' >> "$TEMP_FILE"
find "$PROJECT_ROOT" -type f \
    ! -path "*/__pycache__/*" \
    ! -path "*/.git/*" \
    ! -path "*/data/*" \
    ! -path "*/cache/*" \
    ! -path "*/shards/*" \
    ! -path "*/models/*.pth" \
    ! -path "*/results/*" \
    ! -name "*.pyc" \
    ! -name "*.parquet" \
    ! -name "*.pth" \
    ! -name "continuous_contract.json" \
    ! -name "dump*.md" \
    | sed "s|$PROJECT_ROOT/||" \
    | sort >> "$TEMP_FILE"
echo '```' >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"

# Collect all code files
echo "## Source Files" >> "$TEMP_FILE"
echo "" >> "$TEMP_FILE"

find "$PROJECT_ROOT" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" \) \
    ! -path "*/__pycache__/*" \
    ! -path "*/.git/*" \
    ! -path "*/data/*" \
    ! -path "*/cache/*" \
    ! -path "*/shards/*" \
    ! -path "*/results/*" \
    ! -name "dump*.md" \
    | sort | while read -r file; do
    
    rel_path="${file#$PROJECT_ROOT/}"
    ext="${file##*.}"
    
    echo "### $rel_path" >> "$TEMP_FILE"
    echo "" >> "$TEMP_FILE"
    
    # Determine language for syntax highlighting
    case "$ext" in
        py) lang="python" ;;
        sh) lang="bash" ;;
        md) lang="markdown" ;;
        yaml|yml) lang="yaml" ;;
        *) lang="" ;;
    esac
    
    echo "\`\`\`$lang" >> "$TEMP_FILE"
    cat "$file" >> "$TEMP_FILE"
    echo "" >> "$TEMP_FILE"
    echo "\`\`\`" >> "$TEMP_FILE"
    echo "" >> "$TEMP_FILE"
done

# Split into chunks
total_lines=$(wc -l < "$TEMP_FILE")
num_files=$(( (total_lines + LINES_PER_FILE - 1) / LINES_PER_FILE ))

echo "Total lines: $total_lines"
echo "Splitting into $num_files files..."

# Remove old dump files
rm -f "$PROJECT_ROOT"/${OUTPUT_PREFIX}*.md

# Split
split -l $LINES_PER_FILE -d -a 1 "$TEMP_FILE" "$PROJECT_ROOT/${OUTPUT_PREFIX}"

# Rename to .md
for f in "$PROJECT_ROOT"/${OUTPUT_PREFIX}*; do
    if [[ ! "$f" =~ \.md$ ]]; then
        mv "$f" "${f}.md"
    fi
done

# Cleanup
rm -f "$TEMP_FILE"

echo "Done! Created:"
ls -la "$PROJECT_ROOT"/${OUTPUT_PREFIX}*.md

```

### README.md

```markdown
# MLang2 - Trade Simulation & Research Platform

A deterministic, causal-correct platform for simulating trades on continuous contract data and training models to predict trade outcomes.

## Project Structure

```
mlang2/
├── src/
│   ├── skills/             # High-level agent skills (registry)
│   ├── config.py           # Central configuration
│   ├── data/               # Data loading & resampling
│   ├── sim/                # Deterministic simulation engine
│   ├── features/           # Causal feature computation
│   ├── policy/             # Scanners, filters, actions
│   ├── labels/             # Future-aware labeling (quarantined)
│   ├── datasets/           # Record schemas & sharding
│   ├── models/             # Neural network architectures
│   ├── experiments/        # Experiment framework
│   └── eval/               # Trade metrics & analysis
├── data/
│   ├── raw/                # continuous_contract.json
│   └── processed/          # Parquet files
├── cache/                  # Cached indicators
├── shards/                 # Training data shards
├── models/                 # Trained model checkpoints
├── results/                # Experiment results
└── printcode.sh            # Code dump utility
```

## Core Concepts

### Causal vs Future-Aware

- **`features/`**: Only uses past data via `stepper.get_history()`
- **`labels/`**: Can access future via `FutureWindowProvider` (quarantined)

### Decision Records

Every scanner trigger creates a `DecisionRecord` with:
- Features (causal, at decision time)
- Action taken (`PLACE_ORDER` or `NO_TRADE`)
- Skip reason (if applicable)
- Counterfactual label ("what WOULD have happened")

### Counterfactual Labeling

We label ALL decision points with what would have happened if we traded:
- Enables training on both positive (took trade, won) and negative (skipped, would have lost) examples
- `NO_TRADE` is an action, NOT a label class

## Quick Start

### Using Agent Skills

MLang2 is designed to be agent-friendly. High-level workflows are encapsulated in the `skills` layer.

```python
from src import skills, list_available_skills

# See what's possible
print(list_available_skills())

# Ingest data
processed_paths = skills.get_skill("ingest_raw_data")()

# See data status
print(skills.get_skill("get_data_summary")())
```

```python
from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment
from src.sim.oco import OCOConfig

# Configure experiment
config = ExperimentConfig(
    name="my_experiment",
    start_date="2024-01-01",
    end_date="2024-01-31",
    scanner_id="level_proximity",
    oco_config=OCOConfig(
        direction="LONG",
        tp_multiple=1.4,
        stop_atr=1.0,
    ),
)

# Run
result = run_experiment(config)
print(f"Records: {result.total_records}")
print(f"Win rate: {result.win_records / result.total_records:.1%}")
```

## CLI Example

```bash
# Run experiment
python -m src.experiments.runner \
    --name test_run \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --scanner level_proximity \
    --direction LONG \
    --tp-mult 1.4

# Dump code
./printcode.sh
```

## Key Design Decisions

1. **Deterministic simulation**: Same inputs → same outputs
2. **NO_TRADE is action, not label**: Labels are counterfactual outcomes
3. **Scanner-driven decision points**: Not every bar is a decision
4. **OCO intrabar tie-break**: Explicit rules for same-bar SL/TP hits
5. **Walk-forward splits**: With embargo gaps to prevent leakage
6. **Experiment fingerprint**: SHA256 for reproducibility

```

### src/__init__.py

```python
# MLang2 - Trade Simulation & Research Platform
"""
A deterministic, causal-correct platform for simulating trades,
logging decisions, and training models to predict counterfactual outcomes.
"""

from src.skills.registry import list_available_skills, registry as skills

__version__ = "0.1.0"

```

### src/config.py

```python
"""
MLang2 Configuration
Central configuration for paths, constants, and defaults.
"""

from pathlib import Path
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import List

# =============================================================================
# BASE PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = BASE_DIR / "cache"
SHARDS_DIR = BASE_DIR / "shards"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DIR, CACHE_DIR, SHARDS_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TIMEZONE
# =============================================================================

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
DEFAULT_TZ = NY_TZ

# =============================================================================
# SESSION TIMES (New York)
# =============================================================================

SESSION_RTH_START = "09:30"   # Regular Trading Hours
SESSION_RTH_END = "16:00"
SESSION_GLOBEX_START = "18:00"
SESSION_GLOBEX_END = "09:30"

# =============================================================================
# INSTRUMENT CONSTANTS (MES)
# =============================================================================

TICK_SIZE = 0.25
POINT_VALUE = 5.0
COMMISSION_PER_SIDE = 1.25  # ~$2.50 round trip

# =============================================================================
# INDICATOR DEFAULTS
# =============================================================================

DEFAULT_EMA_PERIOD = 200
DEFAULT_RSI_PERIOD = 14
DEFAULT_ADR_PERIOD = 14
DEFAULT_ATR_PERIOD = 14

# =============================================================================
# FEATURE DEFAULTS
# =============================================================================

DEFAULT_LOOKBACK_MINUTES = 120  # 2 hours
DEFAULT_LOOKBACK_1M = 120       # 2 hours of 1m bars
DEFAULT_LOOKBACK_5M = 24        # 2 hours of 5m bars
DEFAULT_LOOKBACK_15M = 8        # 2 hours of 15m bars

# =============================================================================
# SIMULATION DEFAULTS
# =============================================================================

DEFAULT_MAX_BARS_IN_TRADE = 200
DEFAULT_SLIPPAGE_TICKS = 0.5

# =============================================================================
# DATA FILES
# =============================================================================

CONTINUOUS_CONTRACT_PATH = RAW_DATA_DIR / "continuous_contract.json"

```

### src/datasets/__init__.py

```python
# Datasets module
"""Record schemas, sharding, and data loading."""

```

### src/datasets/decision_record.py

```python
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

```

### src/datasets/reader.py

```python
"""
Shard Reader
Read sharded datasets for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Iterator, Optional
import json

import torch
from torch.utils.data import Dataset, DataLoader

from src.datasets.decision_record import DecisionRecord
from src.datasets.schema import DatasetSchema, DEFAULT_SCHEMA
from src.config import SHARDS_DIR


class ShardReader:
    """
    Read sharded DecisionRecords.
    """
    
    def __init__(self, shard_dir: Path = None):
        self.shard_dir = Path(shard_dir or SHARDS_DIR)
        
        # Load manifest
        manifest_path = self.shard_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}
        
        # Find shard files
        self.shard_paths = sorted(self.shard_dir.glob("shard_*.parquet"))
        self.arrays_dir = self.shard_dir / "arrays"
    
    def __len__(self) -> int:
        return self.manifest.get('total_records', 0)
    
    def __iter__(self) -> Iterator[DecisionRecord]:
        """Iterate over all records."""
        for shard_path in self.shard_paths:
            df = pd.read_parquet(shard_path)
            
            for _, row in df.iterrows():
                record = DecisionRecord.from_dict(row.to_dict())
                
                # Load arrays
                if 'x_price_1m_path' in row and pd.notna(row['x_price_1m_path']):
                    record.x_price_1m = np.load(row['x_price_1m_path'])
                
                if 'x_price_5m_path' in row and pd.notna(row['x_price_5m_path']):
                    record.x_price_5m = np.load(row['x_price_5m_path'])
                
                if 'x_price_15m_path' in row and pd.notna(row['x_price_15m_path']):
                    record.x_price_15m = np.load(row['x_price_15m_path'])
                
                if 'x_context_path' in row and pd.notna(row['x_context_path']):
                    record.x_context = np.load(row['x_context_path'])
                
                yield record
    
    def to_dataframe(self) -> pd.DataFrame:
        """Load all metadata (without arrays) as DataFrame."""
        dfs = [pd.read_parquet(p) for p in self.shard_paths]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)


class DecisionDataset(Dataset):
    """
    PyTorch Dataset for training.
    """
    
    def __init__(
        self,
        shard_dir: Path,
        schema: DatasetSchema = None,
        include_timeout: bool = False
    ):
        self.schema = schema or DEFAULT_SCHEMA
        self.include_timeout = include_timeout
        
        # Load all records (for simplicity - could be lazy)
        reader = ShardReader(shard_dir)
        self.records = []
        
        for record in reader:
            # Filter by label
            if record.cf_outcome == 'TIMEOUT' and not include_timeout:
                continue
            if record.cf_outcome not in self.schema.y_classification:
                continue
            
            self.records.append(record)
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int):
        record = self.records[idx]
        
        # Price windows - (C, L) format for Conv1d
        x_price_1m = torch.FloatTensor(record.x_price_1m.T) if record.x_price_1m is not None else torch.zeros(5, 120)
        x_price_5m = torch.FloatTensor(record.x_price_5m.T) if record.x_price_5m is not None else torch.zeros(5, 24)
        x_price_15m = torch.FloatTensor(record.x_price_15m.T) if record.x_price_15m is not None else torch.zeros(5, 8)
        
        # Context vector
        x_context = torch.FloatTensor(record.x_context) if record.x_context is not None else torch.zeros(self.schema.x_context_dim)
        
        # Label
        label_idx = self.schema.get_label_idx(record.cf_outcome) if record.cf_outcome in self.schema.y_classification else 0
        y = torch.LongTensor([label_idx])
        
        # Regression targets
        y_reg = torch.FloatTensor([
            record.cf_pnl,
            record.cf_mae,
            record.cf_mfe,
            float(record.cf_bars_held)
        ])
        
        return {
            'x_price_1m': x_price_1m,
            'x_price_5m': x_price_5m,
            'x_price_15m': x_price_15m,
            'x_context': x_context,
            'y': y,
            'y_reg': y_reg,
        }


def create_dataloader(
    shard_dir: Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """Create PyTorch DataLoader from shard directory."""
    dataset = DecisionDataset(shard_dir, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

```

### src/datasets/schema.py

```python
"""
Dataset Schema
Explicit separation of x_price (CNN) from x_context (MLP).
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DatasetSchema:
    """
    Defines the structure of training data.
    
    Separates:
    - x_price: Raw OHLCV windows for CNN
    - x_context: Derived features for MLP
    - y: Labels (classification and regression)
    """
    
    # =========================================================================
    # Price Windows (for CNN)
    # =========================================================================
    # Shape: (lookback, channels) where channels = OHLCV = 5
    x_price_2h_1m: Tuple[int, int] = (120, 5)   # 120 1m bars = 2 hours
    x_price_2h_5m: Tuple[int, int] = (24, 5)    # 24 5m bars = 2 hours
    x_price_2h_15m: Tuple[int, int] = (8, 5)    # 8 15m bars = 2 hours
    
    # =========================================================================
    # Context Vector (for MLP)
    # =========================================================================
    x_context_dim: int = 20
    
    x_context_features: List[str] = field(default_factory=lambda: [
        'dist_ema_5m_200_atr',
        'dist_ema_15m_200_atr',
        'dist_vwap_session_atr',
        'dist_vwap_weekly_atr',
        'dist_nearest_1h_level_atr',
        'dist_nearest_4h_level_atr',
        'dist_pdh_atr',
        'dist_pdl_atr',
        'adr_pct_used',
        'rsi_5m_14_norm',
        'rsi_15m_14_norm',
        'relative_volume',
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos',
        'is_rth',
        'is_first_hour',
        'is_last_hour',
        'mins_into_session_norm',
    ])
    
    # =========================================================================
    # Labels
    # =========================================================================
    # Classification: Counterfactual outcome
    # NOTE: NO_TRADE is NOT a label class - it's an action
    y_classification: List[str] = field(default_factory=lambda: [
        'WIN',
        'LOSS',
        'TIMEOUT'
    ])
    
    # Regression targets
    y_regression: List[str] = field(default_factory=lambda: [
        'cf_pnl',
        'cf_mae',
        'cf_mfe',
        'cf_bars_held'
    ])
    
    def to_dict(self) -> dict:
        return {
            'x_price_2h_1m': self.x_price_2h_1m,
            'x_price_2h_5m': self.x_price_2h_5m,
            'x_price_2h_15m': self.x_price_2h_15m,
            'x_context_dim': self.x_context_dim,
            'x_context_features': self.x_context_features,
            'y_classification': self.y_classification,
            'y_regression': self.y_regression,
        }
    
    def get_label_idx(self, label: str) -> int:
        """Get index for classification label."""
        return self.y_classification.index(label)
    
    def label_from_idx(self, idx: int) -> str:
        """Get label name from index."""
        return self.y_classification[idx]


# Default schema
DEFAULT_SCHEMA = DatasetSchema()


def validate_record_schema(record, schema: DatasetSchema = None) -> bool:
    """
    Validate that a DecisionRecord matches the schema.
    """
    schema = schema or DEFAULT_SCHEMA
    
    # Check price windows
    if record.x_price_1m is not None:
        expected = schema.x_price_2h_1m
        actual = record.x_price_1m.shape
        if actual != expected:
            return False
    
    # Check context vector
    if record.x_context is not None:
        if len(record.x_context) != schema.x_context_dim:
            return False
    
    # Check label is valid
    if record.cf_outcome and record.cf_outcome not in schema.y_classification:
        return False
    
    return True

```

### src/datasets/trade_record.py

```python
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

```

### src/datasets/writer.py

```python
"""
Shard Writer
Write decision records to sharded parquet files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import json
import uuid

from src.datasets.decision_record import DecisionRecord
from src.config import SHARDS_DIR


class ShardWriter:
    """
    Write DecisionRecords to sharded files.
    
    Features:
    - Fixed number of records per shard
    - Parquet format for efficient storage
    - Separate files for numpy arrays (optional)
    """
    
    def __init__(
        self,
        output_dir: Path = None,
        records_per_shard: int = 10000,
        experiment_id: str = None
    ):
        self.output_dir = Path(output_dir or SHARDS_DIR)
        self.records_per_shard = records_per_shard
        self.experiment_id = experiment_id or str(uuid.uuid4())[:8]
        
        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buffer
        self._buffer: List[DecisionRecord] = []
        self._shard_idx = 0
        self._total_records = 0
        
        # Arrays storage
        self._arrays_dir = self.output_dir / "arrays"
        self._arrays_dir.mkdir(exist_ok=True)
    
