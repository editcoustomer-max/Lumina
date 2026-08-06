"""Transformer encoder blocks and layers"""

import torch
import torch.nn as nn
import logging

from model.attention import CausalSelfAttention
from model.feedforward import FeedForwardGELU

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block.
    Consists of: Multi-head attention -> Feed-forward with residual connections and layer normalization.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, 
                 max_seq_length: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        # Self-attention
        self.attention = CausalSelfAttention(embedding_dim, num_heads, max_seq_length, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # Feed-forward
        self.ff = FeedForwardGELU(embedding_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
        
        Returns:
            Output tensor [batch_size, seq_len, embedding_dim]
        """
        # Multi-head self-attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of multiple transformer blocks.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int,
                 num_layers: int, max_seq_length: int, dropout: float = 0.1):
        """
        Initialize transformer encoder.
        
        Args:
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            num_layers: Number of transformer blocks
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ff_dim, max_seq_length, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(embedding_dim)
        
        logger.info(f"Built TransformerEncoder with {num_layers} blocks")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer encoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
        
        Returns:
            Output tensor [batch_size, seq_len, embedding_dim]
        """
        for block in self.blocks:
            x = block(x)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        return x
