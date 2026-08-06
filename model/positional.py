"""Positional encoding"""

import torch
import torch.nn as nn
import logging
import math

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    """
    
    def __init__(self, embedding_dim: int, max_seq_length: int):
        """
        Initialize positional encoding.
        
        Args:
            embedding_dim: Embedding dimension
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Dimension-dependent frequency
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                            -(math.log(10000.0) / embedding_dim))
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        if embedding_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
        
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
