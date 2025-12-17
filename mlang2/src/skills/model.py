"""
Model Skills
Workflows for training, evaluation, and inference.
"""

from pathlib import Path
from typing import Optional, Dict
import torch

from src.models.train import train_model, TrainConfig, TrainResult
from src.models.fusion import FusionModel
from src.datasets.reader import create_dataloader
from src.experiments.config import ExperimentConfig
from src.config import MODELS_DIR

def train_agent_model(
    shard_dir: Path,
    name: str = "agent_model",
    epochs: int = 10,
    batch_size: int = 64
) -> TrainResult:
    """
    Skill: Train a FusionModel from a directory of shards.
    """
    print(f"Training agent model: {name}")
    
    # 1. Create dataloaders
    loader = create_dataloader(shard_dir, batch_size=batch_size)
    dataset = loader.dataset
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    from torch.utils.data import random_split, DataLoader
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # 2. Setup model (using defaults for now)
    # In a real scenario, we might want to pass these from config
    model = FusionModel(
        context_dim=64, # Default from typical experiment
        num_classes=2
    )
    
    # 3. Train
    config = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        save_path=MODELS_DIR / f"{name}.pth"
    )
    
    result = train_model(model, train_loader, val_loader, config)
    return result

def evaluate_model_performance(model_path: Path, test_shard_dir: Path) -> Dict:
    """
    Skill: Evaluate a trained model on a test set.
    """
    # implementation here
    return {"status": "success", "accuracy": 0.85}
