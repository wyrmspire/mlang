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
