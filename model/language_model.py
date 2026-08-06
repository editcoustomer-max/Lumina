"""Complete Transformer Language Model"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any

from model.transformer import TransformerEncoder
from model.positional import PositionalEncoding

logger = logging.getLogger(__name__)


class TransformerLM(nn.Module):
    """
    Complete Transformer-based Language Model.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int,
                 num_layers: int, ff_dim: int, max_seq_length: int, dropout: float = 0.1):
        """
        Initialize Transformer Language Model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward hidden dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized TransformerLM with {total_params:,} parameters")
    
    def _init_weights(self):
        """
        Initialize model weights.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of language model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
        
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'max_seq_length': self.max_seq_length,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
    
    def count_parameters(self) -> int:
        """
        Count total trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
