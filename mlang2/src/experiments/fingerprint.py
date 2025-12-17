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
