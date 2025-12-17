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
