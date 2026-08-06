"""Feed-forward networks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = max(0, x*W1 + b1)*W2 + b2
    """
    
    def __init__(self, embedding_dim: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.
        
        Args:
            embedding_dim: Input/output dimension
            ff_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.linear1 = nn.Linear(embedding_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
        
        Returns:
            Output tensor [batch_size, seq_len, embedding_dim]
        """
        # First linear layer + activation
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        
        # Second linear layer
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x


class GELU(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) activation.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Activated tensor
        """
        return x * torch.sigmoid(1.702 * x)


class FeedForwardGELU(nn.Module):
    """
    Feed-Forward Network with GELU activation.
    """
    
    def __init__(self, embedding_dim: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize feed-forward network with GELU.
        
        Args:
            embedding_dim: Input/output dimension
            ff_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.linear1 = nn.Linear(embedding_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embedding_dim)
        self.gelu = GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
        
        Returns:
            Output tensor [batch_size, seq_len, embedding_dim]
        """
        x = self.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x
