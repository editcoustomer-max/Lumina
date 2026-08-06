"""Vocabulary builder for tokenizer"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class VocabularyBuilder:
    """
    Build and manage vocabulary from text data.
    """
    
    def __init__(self, vocab_size: int = 10000, special_tokens: List[str] = None):
        """
        Initialize vocabulary builder.
        
        Args:
            vocab_size: Maximum vocabulary size
            special_tokens: List of special tokens to reserve (e.g., [PAD], [UNK], [CLS])
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        
        # Token to ID and ID to token mappings
        self.token2id = {}
        self.id2token = {}
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """
        Initialize special tokens in vocabulary.
        """
        for token in self.special_tokens:
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token
    
    def build_from_texts(self, texts: List[str], tokenize_func=None):
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of text strings
            tokenize_func: Optional custom tokenization function
        """
        if tokenize_func is None:
            tokenize_func = lambda x: x.split()
        
        # Count token frequencies
        token_counts = Counter()
        for text in texts:
            tokens = tokenize_func(text)
            token_counts.update(tokens)
        
        # Add most common tokens to vocabulary
        # Reserve space for special tokens already added
        available_slots = self.vocab_size - len(self.special_tokens)
        most_common = token_counts.most_common(available_slots)
        
        for token, count in most_common:
            if token not in self.token2id:
                token_id = len(self.token2id)
                self.token2id[token] = token_id
                self.id2token[token_id] = token
        
        logger.info(f"Built vocabulary with {len(self.token2id)} tokens")
    
    def build_from_file(self, filepath: str, tokenize_func=None):
        """
        Build vocabulary from text file.
        
        Args:
            filepath: Path to text file
            tokenize_func: Optional custom tokenization function
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.build_from_texts([text], tokenize_func)
    
    def add_token(self, token: str) -> int:
        """
        Add a token to vocabulary.
        
        Args:
            token: Token string
        
        Returns:
            Token ID
        """
        if token not in self.token2id:
            if len(self.token2id) >= self.vocab_size:
                logger.warning(f"Vocabulary size limit ({self.vocab_size}) reached")
                return self.token2id["[UNK]"]
            
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token
        
        return self.token2id[token]
    
    def get_id(self, token: str) -> int:
        """
        Get token ID.
        
        Args:
            token: Token string
        
        Returns:
            Token ID (or UNK token ID if not found)
        """
        return self.token2id.get(token, self.token2id["[UNK]"])
    
    def get_token(self, token_id: int) -> str:
        """
        Get token from ID.
        
        Args:
            token_id: Token ID
        
        Returns:
            Token string (or "[UNK]" if not found)
        """
        return self.id2token.get(token_id, "[UNK]")
    
    def encode(self, tokens: List[str]) -> List[int]:
        """
        Encode list of tokens to IDs.
        
        Args:
            tokens: List of token strings
        
        Returns:
            List of token IDs
        """
        return [self.get_id(token) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> List[str]:
        """
        Decode list of token IDs to tokens.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            List of token strings
        """
        return [self.get_token(tid) for tid in token_ids]
    
    def __len__(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Number of tokens in vocabulary
        """
        return len(self.token2id)
    
    def save(self, filepath: str):
        """
        Save vocabulary to JSON file.
        
        Args:
            filepath: Path to save vocabulary
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        vocab_data = {
            "token2id": self.token2id,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved vocabulary to {filepath}")
    
    def load(self, filepath: str):
        """
        Load vocabulary from JSON file.
        
        Args:
            filepath: Path to vocabulary file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token2id = vocab_data["token2id"]
        self.special_tokens = vocab_data["special_tokens"]
        self.vocab_size = vocab_data["vocab_size"]
        
        # Rebuild id2token mapping
        self.id2token = {v: k for k, v in self.token2id.items()}
        
        logger.info(f"Loaded vocabulary from {filepath} ({len(self.token2id)} tokens)")
