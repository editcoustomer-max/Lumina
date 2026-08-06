"""Character-level tokenizer"""

import logging
import json
from typing import List, Dict
from pathlib import Path

from tokenizer.base import BaseTokenizer

logger = logging.getLogger(__name__)


class CharacterTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.
    Each character is a token.
    """
    
    def __init__(self):
        """
        Initialize character tokenizer.
        """
        super().__init__()
        self.vocab = {}
        self.itos = {}
        self.stoi = {}
        self.vocab_size = 0
    
    def build_vocab(self, text: str):
        """
        Build character vocabulary from text.
        
        Args:
            text: Input text
        """
        # Get unique characters
        chars = sorted(set(text))
        
        # Create mappings
        self.stoi = {char: idx for idx, char in enumerate(chars)}
        self.itos = {idx: char for char, idx in self.stoi.items()}
        self.vocab = self.stoi.copy()
        self.vocab_size = len(chars)
        
        logger.info(f"Built character vocabulary with {self.vocab_size} characters")
        logger.debug(f"Vocabulary: {chars}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to character IDs.
        
        Args:
            text: Input text
        
        Returns:
            List of character IDs
        """
        if not self.stoi:
            logger.error("Vocabulary not built. Call build_vocab first.")
            return []
        
        ids = []
        for char in text:
            if char in self.stoi:
                ids.append(self.stoi[char])
            else:
                logger.warning(f"Character not in vocabulary: {repr(char)}")
                # Use a default token or skip
                continue
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode character IDs to text.
        
        Args:
            ids: List of character IDs
        
        Returns:
            Decoded text
        """
        if not self.itos:
            logger.error("Vocabulary not built. Call build_vocab first.")
            return ""
        
        text = ""
        for token_id in ids:
            if token_id in self.itos:
                text += self.itos[token_id]
            else:
                logger.warning(f"Token ID not in vocabulary: {token_id}")
                continue
        
        return text
    
    def save(self, filepath: str):
        """
        Save tokenizer to JSON file.
        
        Args:
            filepath: Path to save to
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'type': 'character',
            'vocab': self.vocab,
            'itos': self.itos,
            'vocab_size': self.vocab_size
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved tokenizer to {filepath}")
    
    def load(self, filepath: str):
        """
        Load tokenizer from JSON file.
        
        Args:
            filepath: Path to load from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if data['type'] != 'character':
            logger.warning(f"Tokenizer type mismatch: expected 'character', got '{data['type']}'")
        
        self.vocab = data['vocab']
        self.itos = {int(k): v for k, v in data['itos'].items()}
        self.stoi = {v: int(k) for k, v in self.itos.items()}
        self.vocab_size = data['vocab_size']
        
        logger.info(f"Loaded tokenizer from {filepath}")
