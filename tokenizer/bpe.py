"""Byte Pair Encoding (BPE) tokenizer"""

import logging
import json
from typing import List, Dict, Set, Tuple
from pathlib import Path
from collections import defaultdict

from tokenizer.base import BaseTokenizer

logger = logging.getLogger(__name__)


class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer.
    Learns subword units by iteratively merging frequent byte pairs.
    """
    
    def __init__(self, vocab_size: int = 1000, num_merges: int = None):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            num_merges: Number of merge operations (defaults to vocab_size - 256)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_merges = num_merges or (vocab_size - 256)
        self.merges = {}  # Track merge operations
        self.word_tokenizer = None
    
    def build_vocab(self, text: str, num_workers: int = 1):
        """
        Build BPE vocabulary from text.
        
        Args:
            text: Input text
            num_workers: Number of workers (currently unused)
        """
        logger.info(f"Building BPE vocabulary with {self.num_merges} merges...")
        
        # Start with character-level tokens
        vocab = set(text.encode('utf-8'))
        
        # Count word frequencies
        words = text.split()
        word_freqs = defaultdict(int)
        for word in words:
            word_freqs[' '.join(word) + ' </w>'] += 1  # Add end-of-word marker
        
        # Initialize with character-level vocabulary
        vocab = set()
        for word in word_freqs:
            for char in word:
                vocab.add(char)
        
        # Perform merge operations
        for i in range(self.num_merges):
            if i % 100 == 0:
                logger.debug(f"Merge iteration {i}/{self.num_merges}")
            
            # Find most frequent pair
            pair_freqs = self._get_pair_frequencies(word_freqs)
            if not pair_freqs:
                logger.warning(f"No more pairs to merge at iteration {i}")
                break
            
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Merge pair in vocabulary
            vocab.add(''.join(best_pair))
            self.merges[best_pair] = i
            
            # Update word frequencies
            word_freqs = self._merge_pair(word_freqs, best_pair)
        
        # Create final vocabulary
        self.vocab = {v: i for i, v in enumerate(sorted(vocab))}
        self.stoi = self.vocab.copy()
        self.itos = {i: v for v, i in self.vocab.items()}
        self.vocab_size = len(vocab)
        
        logger.info(f"Built BPE vocabulary with {self.vocab_size} tokens")
    
    def _get_pair_frequencies(self, word_freqs: Dict) -> Dict[Tuple[str, str], int]:
        """
        Count frequencies of adjacent token pairs.
        
        Args:
            word_freqs: Word frequency dictionary
        
        Returns:
            Dictionary of pair frequencies
        """
        pair_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] += freq
        
        return pair_freqs
    
    def _merge_pair(self, word_freqs: Dict, pair: Tuple[str, str]) -> Dict:
        """
        Merge a pair in all words.
        
        Args:
            word_freqs: Word frequency dictionary
            pair: Pair to merge
        
        Returns:
            Updated word frequency dictionary
        """
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        
        return new_word_freqs
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text using BPE.
        
        Args:
            text: Input text
        
        Returns:
            List of token IDs
        """
        if not self.vocab:
            logger.error("Vocabulary not built. Call build_vocab first.")
            return []
        
        # Simplified encoding - split by characters and look up
        tokens = []
        for char in text:
            if char in self.stoi:
                tokens.append(self.stoi[char])
            else:
                logger.warning(f"Character not in vocabulary: {repr(char)}")
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode BPE token IDs to text.
        
        Args:
            ids: List of token IDs
        
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
        
        return text
    
    def save(self, filepath: str):
        """
        Save tokenizer to JSON file.
        
        Args:
            filepath: Path to save to
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert merges to serializable format
        merges_list = [(list(k), v) for k, v in self.merges.items()]
        
        data = {
            'type': 'bpe',
            'vocab': self.vocab,
            'itos': self.itos,
            'vocab_size': self.vocab_size,
            'merges': merges_list,
            'num_merges': self.num_merges
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved BPE tokenizer to {filepath}")
    
    def load(self, filepath: str):
        """
        Load tokenizer from JSON file.
        
        Args:
            filepath: Path to load from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if data['type'] != 'bpe':
            logger.warning(f"Tokenizer type mismatch: expected 'bpe', got '{data['type']}'")
        
        self.vocab = data['vocab']
        self.itos = {int(k): v for k, v in data['itos'].items()}
        self.stoi = {v: int(k) for k, v in self.itos.items()}
        self.vocab_size = data['vocab_size']
        self.num_merges = data['num_merges']
        self.merges = {tuple(k): v for k, v in data['merges']}
        
        logger.info(f"Loaded BPE tokenizer from {filepath}")
