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
