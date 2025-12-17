"""
Feature Store
Cached indicator computation to avoid recomputing on every experiment.
"""

import pandas as pd
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from src.config import CACHE_DIR


class FeatureStore:
    """
    Cached indicator and feature computation.
    
    Avoids recomputing expensive indicators on every experiment run.
    Cache is keyed by: data hash + indicator name + params.
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, pd.Series] = {}
    
    def _compute_data_hash(self, df: pd.DataFrame, columns: list = None) -> str:
        """Compute hash of DataFrame for cache key."""
        if columns:
            data = df[columns].values.tobytes()
        else:
            data = df.values.tobytes()
        return hashlib.md5(data).hexdigest()[:12]
    
    def _get_cache_key(
        self,
        data_hash: str,
        indicator: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate cache key."""
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"{indicator}_{data_hash}_{params_hash}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cached file."""
        return self.cache_dir / f"{cache_key}.parquet"
    
    def get_or_compute(
        self,
        df: pd.DataFrame,
        indicator: str,
        params: Dict[str, Any],
        compute_fn: Callable[[pd.DataFrame, Dict[str, Any]], pd.Series],
        columns: list = None
    ) -> pd.Series:
        """
        Get indicator from cache or compute and cache.
        
        Args:
            df: Source DataFrame
            indicator: Indicator name (e.g., 'ema', 'rsi')
            params: Indicator parameters (e.g., {'period': 200})
            compute_fn: Function to compute indicator if not cached
            columns: Columns to use for data hash (default: all)
            
        Returns:
            Computed indicator as Series
        """
        data_hash = self._compute_data_hash(df, columns)
        cache_key = self._get_cache_key(data_hash, indicator, params)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            result = cached['value']
            self._memory_cache[cache_key] = result
            return result
        
        # Compute
        result = compute_fn(df, params)
        
        # Save to disk cache
        pd.DataFrame({'value': result}).to_parquet(cache_path)
        
        # Save to memory cache
        self._memory_cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """Clear all cached data."""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
    
    def cache_size(self) -> Dict[str, int]:
        """Get cache statistics."""
        files = list(self.cache_dir.glob("*.parquet"))
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            'files': len(files),
            'bytes': total_bytes,
            'memory_entries': len(self._memory_cache)
        }


# Global feature store instance
_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get or create global feature store."""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store
