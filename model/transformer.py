"""Transformer-based language model"""

import torch
import torch.nn as nn
import logging
from model.layers import TransformerBlock, CausalMask
from tokenizer.embeddings import EmbeddingLayer

logger = logging.getLogger(__name__)


class TransformerLM(nn.Module):
    """
    Transformer-based language model for text generation.
    Decoder-only architecture (similar to GPT).
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, 
                 num_layers: int, ff_dim: int, max_seq_length: int = 512,
                 dropout: float = 0.1):
        """
        Initialize Transformer language model.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            ff_dim: Feed-forward network dimension
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
        
        # Embeddings (token + positional)
        self.embeddings = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output layer (vocabulary prediction)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, token_ids: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            token_ids: Input token IDs [batch_size, seq_len]
            return_hidden: Whether to return hidden states
        
        Returns:
            Logits [batch_size, seq_len, vocab_size]
            Or (logits, hidden_states) if return_hidden=True
        """
        device = token_ids.device
        seq_len = token_ids.size(1)
        
        # Embeddings
        x = self.embeddings(token_ids)
        
        # Create causal mask
        causal_mask = CausalMask.create_mask(seq_len, device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Transformer blocks
        hidden_states = []
        for block in self.transformer_blocks:
            x = block(x, causal_mask)
            if return_hidden:
                hidden_states.append(x.detach())
        
        # Final normalization and output projection
        x = self.final_norm(x)
        logits = self.output_layer(x)
        
        if return_hidden:
            return logits, hidden_states
        
        return logits
    
    def get_config(self) -> dict:
        """
        Get model configuration.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'max_seq_length': self.max_seq_length,
        }
    
    @staticmethod
    def from_config(config: dict) -> 'TransformerLM':
        """
        Create model from configuration dictionary.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Initialized TransformerLM model
        """
        return TransformerLM(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            ff_dim=config['ff_dim'],
            max_seq_length=config.get('max_seq_length', 512),
            dropout=config.get('dropout', 0.1),
        )
