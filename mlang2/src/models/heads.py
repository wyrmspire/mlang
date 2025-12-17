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
