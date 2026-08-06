"""Embedding layers for tokenizer"""

import math
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize token embedding.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initialize with normal distribution
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim ** -0.5)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
        
        Returns:
            Embeddings [batch_size, seq_len, embedding_dim]
        """
        return self.embedding(token_ids) * math.sqrt(self.embedding_dim)


class PositionalEmbedding(nn.Module):
    """
    Positional encoding for transformer.
    Uses sinusoidal positional encoding (Transformer paper).
    """
    
    def __init__(self, embedding_dim: int, max_seq_length: int = 512):
        """
        Initialize positional embedding.
        
        Args:
            embedding_dim: Embedding dimension
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Create positional encoding
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Dimension indices
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * 
            (-math.log(10000.0) / embedding_dim)
        )
        
        # Sinusoidal encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        if embedding_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trainable)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        
        Args:
            embeddings: Token embeddings [batch_size, seq_len, embedding_dim]
        
        Returns:
            Embeddings with positional encoding added
        """
        seq_len = embeddings.size(1)
        return embeddings + self.pe[:, :seq_len, :]


class EmbeddingLayer(nn.Module):
    """
    Combined token and positional embedding layer.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_length: int = 512, dropout: float = 0.1):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(embedding_dim, max_seq_length)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
        
        Returns:
            Embeddings with positional encoding [batch_size, seq_len, embedding_dim]
        """
        # Token embeddings
        embeddings = self.token_embedding(token_ids)
        
        # Add positional encoding
        embeddings = self.positional_embedding(embeddings)
        
        # Dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
