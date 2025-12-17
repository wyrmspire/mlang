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
