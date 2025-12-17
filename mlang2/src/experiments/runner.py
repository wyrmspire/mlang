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
