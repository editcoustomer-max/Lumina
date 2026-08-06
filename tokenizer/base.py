"""Base tokenizer interface"""

import logging
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.
    """
    
    def __init__(self):
        """
        Initialize tokenizer.
        """
        self.vocab = {}
        self.vocab_size = 0
        self.itos = {}  # index to string mapping
        self.stoi = {}  # string to index mapping
    
    @abstractmethod
    def build_vocab(self, text: str):
        """
        Build vocabulary from text.
        
        Args:
            text: Input text
        """
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
        
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text
        """
        pass
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Size of vocabulary
        """
        return self.vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Get vocabulary dictionary.
        
        Returns:
            Vocabulary mapping
        """
        return self.vocab.copy()
    
    @abstractmethod
    def save(self, filepath: str):
        """
        Save tokenizer to file.
        
        Args:
            filepath: Path to save to
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """
        Load tokenizer from file.
        
        Args:
            filepath: Path to load from
        """
        pass
