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
