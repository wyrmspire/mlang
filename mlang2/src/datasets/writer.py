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
