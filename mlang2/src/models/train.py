"""
Training
Training loop and configuration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import numpy as np

from src.config import MODELS_DIR


class ImbalanceStrategy(Enum):
    """Strategy for handling class imbalance."""
    NONE = "none"
    WEIGHTED_LOSS = "weighted"
    FOCAL_LOSS = "focal"
    BALANCED_SAMPLING = "balanced"


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    dropout: float = 0.3
    
    # Imbalance handling
    imbalance_strategy: ImbalanceStrategy = ImbalanceStrategy.WEIGHTED_LOSS
    focal_gamma: float = 2.0
    class_weights: Optional[Dict[int, float]] = None
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # Checkpointing
    save_best: bool = True
    save_path: Path = None
    
    def to_dict(self) -> dict:
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'imbalance_strategy': self.imbalance_strategy.value,
            'patience': self.patience,
        }


@dataclass
class TrainResult:
    """Training result."""
    best_val_loss: float
    best_epoch: int
    train_losses: List[float]
    val_losses: List[float]
    val_accuracies: List[float]
    model_path: Optional[Path] = None


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_class_weights(labels: List[int], num_classes: int = 2) -> torch.Tensor:
    """Compute inverse frequency class weights."""
    counts = np.bincount(labels, minlength=num_classes)
    total = sum(counts)
    weights = total / (num_classes * counts + 1e-6)
    return torch.FloatTensor(weights)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        x_1m = batch['x_price_1m'].to(device)
        x_5m = batch['x_price_5m'].to(device)
        x_15m = batch['x_price_15m'].to(device)
        x_context = batch['x_context'].to(device)
        y = batch['y'].squeeze().to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(x_1m, x_5m, x_15m, x_context)
        loss = criterion(logits, y)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            x_1m = batch['x_price_1m'].to(device)
            x_5m = batch['x_price_5m'].to(device)
            x_15m = batch['x_price_15m'].to(device)
            x_context = batch['x_context'].to(device)
            y = batch['y'].squeeze().to(device)
            
            logits = model(x_1m, x_5m, x_15m, x_context)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig
) -> TrainResult:
    """
    Full training loop.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup criterion based on imbalance strategy
    if config.imbalance_strategy == ImbalanceStrategy.WEIGHTED_LOSS:
        # Compute weights from training data
        labels = [batch['y'].squeeze().tolist() for batch in train_loader]
        labels = [l for batch_labels in labels for l in (batch_labels if isinstance(batch_labels, list) else [batch_labels])]
        weights = compute_class_weights(labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
    elif config.imbalance_strategy == ImbalanceStrategy.FOCAL_LOSS:
        criterion = FocalLoss(gamma=config.focal_gamma)
        
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training state
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    save_path = config.save_path or MODELS_DIR / "best_model.pth"
    
    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
        # Check improvement
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            if config.save_best:
                torch.save(model.state_dict(), save_path)
                print(f"  [Saved best model]")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return TrainResult(
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        model_path=save_path if config.save_best else None
    )
