"""Tokenizer factory and convenience functions"""

import logging
from typing import Optional

from tokenizer.base import BaseTokenizer
from tokenizer.character import CharacterTokenizer
from tokenizer.bpe import BPETokenizer

logger = logging.getLogger(__name__)


class TokenizerFactory:
    """
    Factory for creating tokenizers.
    """
    
    @staticmethod
    def create_tokenizer(tokenizer_type: str = 'character', 
                        vocab_size: int = 256,
                        **kwargs) -> BaseTokenizer:
        """
        Create a tokenizer of specified type.
        
        Args:
            tokenizer_type: Type of tokenizer (character, bpe)
            vocab_size: Vocabulary size (for BPE)
            **kwargs: Additional arguments for tokenizer
        
        Returns:
            Tokenizer instance
        """
        if tokenizer_type == 'character':
            logger.info("Creating character tokenizer")
            return CharacterTokenizer()
        
        elif tokenizer_type == 'bpe':
            logger.info(f"Creating BPE tokenizer with vocab_size={vocab_size}")
            return BPETokenizer(vocab_size=vocab_size, **kwargs)
        
        else:
            logger.error(f"Unknown tokenizer type: {tokenizer_type}")
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    @staticmethod
    def load_tokenizer(filepath: str) -> BaseTokenizer:
        """
        Load tokenizer from file.
        
        Args:
            filepath: Path to tokenizer file
        
        Returns:
            Loaded tokenizer
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tokenizer_type = data.get('type')
        
        if tokenizer_type == 'character':
            tokenizer = CharacterTokenizer()
        elif tokenizer_type == 'bpe':
            tokenizer = BPETokenizer(vocab_size=data.get('vocab_size', 256))
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        tokenizer.load(filepath)
        logger.info(f"Loaded {tokenizer_type} tokenizer from {filepath}")
        
        return tokenizer
