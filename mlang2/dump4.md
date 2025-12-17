    
    Returns:
        (reached: bool, bars_to_reach: int or -1 if not reached)
    """
    future = future_provider.get_future(within_bars)
    
    if len(future) == 0:
        return (False, -1)
    
    if direction == 'UP':
        hits = np.where(future['high'].values >= target_price)[0]
    else:
        hits = np.where(future['low'].values <= target_price)[0]
    
    if len(hits) > 0:
        return (True, int(hits[0]) + 1)
    
    return (False, -1)

```

### src/models/__init__.py

```python
# Models module
"""Neural network architectures and training."""

```

### src/models/context_mlp.py

```python
"""
Context MLP
MLP encoder for context feature vector.
"""

import torch
import torch.nn as nn


class ContextMLP(nn.Module):
    """
    MLP for encoding context features.
    
    Input: (batch, context_dim) e.g., (64, 20)
    Output: (batch, embedding_dim)
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        embedding_dim: int = 32,
        hidden_dims: list = None,
        dropout: float = 0.3
    ):
        super().__init__()
        
        hidden_dims = hidden_dims or [64, 64]
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, embedding_dim))
        
        self.net = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, input_dim)
            
        Returns:
            (batch, embedding_dim)
        """
        return self.net(x)

```

### src/models/encoders.py

```python
"""
CNN Encoders
Price window encoders for pattern recognition.
"""

import torch
import torch.nn as nn
from typing import Tuple


class CNNEncoder(nn.Module):
    """
    1D CNN for encoding price windows.
    
    Input: (batch, channels, length) e.g., (64, 5, 120)
    Output: (batch, embedding_dim)
    """
    
    def __init__(
        self,
        input_channels: int = 5,      # OHLCV
        seq_length: int = 120,        # 2 hours of 1m
        embedding_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # length / 2
            
            # Conv block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # length / 4
            
            # Conv block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # length / 8
            
            # Conv block 4
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (batch, 128, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, channels, length)
            
        Returns:
            (batch, embedding_dim)
        """
        x = self.features(x)
        x = self.fc(x)
        return x


class MultiTFEncoder(nn.Module):
    """
    Encode multiple timeframe price windows.
    
    Separate CNN for each timeframe, then concatenate.
    """
    
    def __init__(
        self,
        tf_configs: dict = None,
        embedding_dim_per_tf: int = 32,
        dropout: float = 0.3
    ):
        """
        Args:
            tf_configs: Dict of {name: (length, channels)}
                Default: {'1m': (120, 5), '5m': (24, 5), '15m': (8, 5)}
            embedding_dim_per_tf: Embedding size per timeframe
        """
        super().__init__()
        
        self.tf_configs = tf_configs or {
            '1m': (120, 5),
            '5m': (24, 5),
            '15m': (8, 5),
        }
        
        self.encoders = nn.ModuleDict()
        for name, (length, channels) in self.tf_configs.items():
            self.encoders[name] = CNNEncoder(
                input_channels=channels,
                seq_length=length,
                embedding_dim=embedding_dim_per_tf,
                dropout=dropout
            )
        
        self.total_dim = embedding_dim_per_tf * len(self.tf_configs)
    
    def forward(
        self,
        x_1m: torch.Tensor,
        x_5m: torch.Tensor,
        x_15m: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode all timeframes and concatenate.
        
        Returns:
            (batch, total_dim)
        """
        embeddings = []
        
        if '1m' in self.encoders:
            embeddings.append(self.encoders['1m'](x_1m))
        if '5m' in self.encoders:
            embeddings.append(self.encoders['5m'](x_5m))
        if '15m' in self.encoders:
            embeddings.append(self.encoders['15m'](x_15m))
        
        return torch.cat(embeddings, dim=-1)

```

### src/models/fusion.py

```python
"""
Fusion Model
Combine CNN price encoders with context MLP.
"""

import torch
import torch.nn as nn

from src.models.encoders import MultiTFEncoder
from src.models.context_mlp import ContextMLP


class FusionModel(nn.Module):
    """
    CNN + MLP fusion for decision classification.
    
    Architecture:
    - MultiTFEncoder processes price windows
    - ContextMLP processes context features
    - Concatenate and pass through classification head
    """
    
    def __init__(
        self,
        context_dim: int = 20,
        price_embedding_per_tf: int = 32,
        context_embedding: int = 32,
        num_classes: int = 2,  # WIN/LOSS
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Price encoder
        self.price_encoder = MultiTFEncoder(
            embedding_dim_per_tf=price_embedding_per_tf,
            dropout=dropout
        )
        
        # Context encoder
        self.context_encoder = ContextMLP(
            input_dim=context_dim,
            embedding_dim=context_embedding,
            dropout=dropout
        )
        
        # Combined dimension
        combined_dim = self.price_encoder.total_dim + context_embedding
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        
        self.num_classes = num_classes
    
    def forward(
        self,
        x_price_1m: torch.Tensor,
        x_price_5m: torch.Tensor,
        x_price_15m: torch.Tensor,
        x_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_price_*: Price windows (batch, channels, length)
            x_context: Context vector (batch, context_dim)
            
        Returns:
            Logits (batch, num_classes)
        """
        # Encode price
        price_emb = self.price_encoder(x_price_1m, x_price_5m, x_price_15m)
        
        # Encode context
        context_emb = self.context_encoder(x_context)
        
        # Fuse
        combined = torch.cat([price_emb, context_emb], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits
    
    def predict_proba(
        self,
        x_price_1m: torch.Tensor,
        x_price_5m: torch.Tensor,
        x_price_15m: torch.Tensor,
        x_context: torch.Tensor
    ) -> torch.Tensor:
        """Get probability of WIN class."""
        logits = self.forward(x_price_1m, x_price_5m, x_price_15m, x_context)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1] if self.num_classes == 2 else probs  # P(WIN)


class SimpleCNN(nn.Module):
    """
    Simple CNN model using only 1m price data.
    Good for baseline comparisons.
    """
    
    def __init__(
        self,
        input_channels: int = 5,
        seq_length: int = 120,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
            
        Returns:
            Logits (batch, num_classes)
        """
        x = self.features(x)
        return self.classifier(x)

```

### src/models/heads.py

```python
"""
Model Heads
Classification and regression heads.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head for binary or multi-class.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits."""
        return self.net(x)


class RegressionHead(nn.Module):
    """
    Regression head for continuous outputs (PnL, MAE, MFE).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskHead(nn.Module):
    """
    Multi-task head for joint classification + regression.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        num_regression: int = 4,  # pnl, mae, mfe, bars_held
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        self.classification_head = nn.Linear(hidden_dim, num_classes)
        self.regression_head = nn.Linear(hidden_dim, num_regression)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns:
            Dict with 'logits' and 'regression' tensors
        """
        shared = self.shared(x)
        
        return {
            'logits': self.classification_head(shared),
            'regression': self.regression_head(shared),
        }

```

### src/models/train.py

```python
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

```

### src/policy/__init__.py

```python
# Policy module
"""Decision logic - scanners, filters, and actions."""

```

### src/policy/actions.py

```python
"""
Actions
Decision action types and policy decision structure.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.sim.oco import OCOConfig


class Action(Enum):
    """What action to take at a decision point."""
    NO_TRADE = "NO_TRADE"         # Skip this opportunity
    PLACE_ORDER = "PLACE_ORDER"   # Enter new position
    MANAGE = "MANAGE"             # Adjust existing position
    EXIT = "EXIT"                 # Close position


class SkipReason(Enum):
    """
    Why a decision point was skipped.
    
    This is crucial for understanding dataset composition:
    - FILTER_BLOCK: Filtered out before reaching policy
    - COOLDOWN: Too soon after last trade
    - IN_POSITION: Already have open position
    - POLICY_NO: Policy decided not to trade
    - OTHER: Other reason
    """
    NOT_SKIPPED = "NOT_SKIPPED"   # Trade was taken
    FILTER_BLOCK = "FILTER_BLOCK"
    COOLDOWN = "COOLDOWN"
    IN_POSITION = "IN_POSITION"
    POLICY_NO = "POLICY_NO"
    OTHER = "OTHER"


@dataclass
class PolicyDecision:
    """
    Complete decision at a decision point.
    """
    action: Action
    skip_reason: SkipReason = SkipReason.NOT_SKIPPED
    reason_detail: str = ""       # Human-readable explanation
    
    # If PLACE_ORDER
    order_config: Optional[OCOConfig] = None
    
    # Scanner context that led to this decision
    scanner_id: str = ""
    scanner_context: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence/score (for ML-based policy)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action.value,
            'skip_reason': self.skip_reason.value,
            'reason_detail': self.reason_detail,
            'scanner_id': self.scanner_id,
            'confidence': self.confidence,
            'order_config': self.order_config.to_dict() if self.order_config else None,
        }


def make_no_trade(
    reason: SkipReason,
    detail: str = "",
    scanner_id: str = ""
) -> PolicyDecision:
    """Helper to create NO_TRADE decision."""
    return PolicyDecision(
        action=Action.NO_TRADE,
        skip_reason=reason,
        reason_detail=detail,
        scanner_id=scanner_id,
    )


def make_trade(
    order_config: OCOConfig,
    scanner_id: str = "",
    confidence: float = 1.0,
    context: Dict[str, Any] = None
) -> PolicyDecision:
    """Helper to create PLACE_ORDER decision."""
    return PolicyDecision(
        action=Action.PLACE_ORDER,
        skip_reason=SkipReason.NOT_SKIPPED,
        order_config=order_config,
        scanner_id=scanner_id,
        confidence=confidence,
        scanner_context=context or {},
    )

```

### src/policy/cooldown.py

```python
"""
Cooldown
Manage trade cooldown periods to prevent overtrading.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class CooldownConfig:
    """Cooldown configuration."""
    min_bars_between_trades: int = 15  # Minimum bars between entries
    min_bars_after_loss: int = 30      # Extra cooldown after a loss
    max_trades_per_day: int = 10       # Maximum trades per trading day


class CooldownManager:
    """
    Manage trade cooldown state.
    """
    
    def __init__(self, config: CooldownConfig = None):
        self.config = config or CooldownConfig()
        self._last_trade_bar: int = -999
        self._last_outcome: str = ""
        self._trades_today: int = 0
        self._current_date: Optional[pd.Timestamp] = None
    
    def reset(self):
        """Reset cooldown state."""
        self._last_trade_bar = -999
        self._last_outcome = ""
        self._trades_today = 0
        self._current_date = None
    
    def record_trade(
        self,
        bar_idx: int,
        outcome: str = "",
        timestamp: pd.Timestamp = None
    ):
        """Record that a trade was taken."""
        self._last_trade_bar = bar_idx
        self._last_outcome = outcome
        
        # Track trades per day
        if timestamp:
            trade_date = timestamp.date()
            if self._current_date != trade_date:
                self._current_date = trade_date
                self._trades_today = 0
            self._trades_today += 1
    
    def is_on_cooldown(
        self,
        current_bar: int,
        timestamp: pd.Timestamp = None
    ) -> tuple:
        """
        Check if currently on cooldown.
        
        Returns:
            (on_cooldown: bool, reason: str)
        """
        # Check max trades per day
        if timestamp:
            current_date = timestamp.date()
            if current_date == self._current_date:
                if self._trades_today >= self.config.max_trades_per_day:
                    return (True, f"Max trades per day ({self.config.max_trades_per_day}) reached")
        
        # Check bars since last trade
        bars_since = current_bar - self._last_trade_bar
        
        # Extra cooldown after loss
        if self._last_outcome == 'LOSS':
            min_bars = self.config.min_bars_after_loss
        else:
            min_bars = self.config.min_bars_between_trades
        
        if bars_since < min_bars:
            return (True, f"Cooldown: {bars_since}/{min_bars} bars since last trade")
        
        return (False, "")
    
    def bars_remaining(self, current_bar: int) -> int:
        """Get bars remaining in cooldown."""
        bars_since = current_bar - self._last_trade_bar
        
        if self._last_outcome == 'LOSS':
            min_bars = self.config.min_bars_after_loss
        else:
            min_bars = self.config.min_bars_between_trades
        
        return max(0, min_bars - bars_since)

```

### src/policy/filters.py

```python
"""
Filters
Pre-trade filters that block decisions before reaching policy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from src.features.pipeline import FeatureBundle
from src.features.time_features import Session


@dataclass
class FilterResult:
    """Result from a filter check."""
    passed: bool
    filter_id: str
    reason: str = ""


class Filter(ABC):
    """Base class for pre-trade filters."""
    
    @property
    @abstractmethod
    def filter_id(self) -> str:
        pass
    
    @abstractmethod
    def check(self, features: FeatureBundle) -> FilterResult:
        """Check if filter passes."""
        pass


class SessionFilter(Filter):
