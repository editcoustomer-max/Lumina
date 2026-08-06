"""Attention mechanisms"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
    """
    
    def __init__(self, d_k: int, dropout: float = 0.1):
        """
        Initialize attention mechanism.
        
        Args:
            d_k: Dimension of key/query
            dropout: Dropout rate
        """
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_k]
            key: Key tensor [batch_size, seq_len, d_k]
            value: Value tensor [batch_size, seq_len, d_v]
            mask: Optional attention mask
        
        Returns:
            Attention output [batch_size, seq_len, d_v]
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            embedding_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.d_k = embedding_dim // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, embedding_dim]
            key: Key tensor [batch_size, seq_len, embedding_dim]
            value: Value tensor [batch_size, seq_len, embedding_dim]
            mask: Optional attention mask
        
        Returns:
            Attention output [batch_size, seq_len, embedding_dim]
        """
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embedding_dim)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output


class CausalSelfAttention(nn.Module):
    """
    Causal (autoregressive) self-attention for language modeling.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int, max_seq_length: int, dropout: float = 0.1):
        """
        Initialize causal self-attention.
        
        Args:
            embedding_dim: Total embedding dimension
            num_heads: Number of attention heads
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.d_k = embedding_dim // num_heads
        self.max_seq_length = max_seq_length
        
        # Linear projections
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer
        causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length))
        self.register_buffer('causal_mask', causal_mask.unsqueeze(0).unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of causal self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
        
        Returns:
            Attention output [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Get causal mask for current sequence length
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # Apply attention with causal mask
        attn_output = self.attention(Q, K, V, causal_mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embedding_dim)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output
