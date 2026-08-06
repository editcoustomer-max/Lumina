"""Transformer model layers"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            embedding_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            q: Query tensor [batch_size, seq_len, embedding_dim]
            k: Key tensor [batch_size, seq_len, embedding_dim]
            v: Value tensor [batch_size, seq_len, embedding_dim]
            mask: Attention mask (optional)
        
        Returns:
            Attention output [batch_size, seq_len, embedding_dim]
        """
        batch_size = q.shape[0]
        
        # Linear projections and reshape for multi-head
        Q = self.query(q).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(k).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(v).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).reshape(batch_size, -1, self.embedding_dim)
        
        # Output projection
        output = self.fc_out(context)
        
        return output


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network (position-wise feed-forward network in Transformer).
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
        self.fc1 = nn.Linear(embedding_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
        
        Returns:
            Output tensor [batch_size, seq_len, embedding_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with attention, feed-forward, and layer normalization.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Attention
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # Feed-forward
        self.ffn = FeedForwardNetwork(embedding_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
            mask: Attention mask (optional)
        
        Returns:
            Output tensor [batch_size, seq_len, embedding_dim]
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x


class CausalMask:
    """
    Create causal masks for autoregressive generation.
    """
    
    @staticmethod
    def create_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensor on
        
        Returns:
            Causal mask [seq_len, seq_len]
        """
        # Lower triangular matrix (allows attending to current and past)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
