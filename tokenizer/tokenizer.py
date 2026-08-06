"""Character and BPE tokenizer implementations"""

import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class CharacterTokenizer:
    """
    Simple character-level tokenizer.
    """
    
    def __init__(self, vocab: Dict[str, int] = None):
        """
        Initialize character tokenizer.
        
        Args:
            vocab: Optional pre-built vocabulary
        """
        self.vocab = vocab or {}
        self.char_to_id = {}
        self.id_to_char = {}
        
        if vocab:
            self.char_to_id = vocab
            self.id_to_char = {v: k for k, v in vocab.items()}
    
    def build_vocab(self, text: str):
        """
        Build character vocabulary from text.
        
        Args:
            text: Text to build vocabulary from
        """
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        
        logger.info(f"Built character vocabulary with {len(self.char_to_id)} characters")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to character IDs.
        
        Args:
            text: Text to encode
        
        Returns:
            List of character IDs
        """
        return [self.char_to_id.get(ch, 0) for ch in text]
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode character IDs back to text.
        
        Args:
            ids: List of character IDs
        
        Returns:
            Decoded text
        """
        return ''.join(self.id_to_char.get(i, '') for i in ids)
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Number of characters in vocabulary
        """
        return len(self.char_to_id)
    
    def save(self, filepath: str):
        """
        Save tokenizer vocabulary.
        
        Args:
            filepath: Path to save vocabulary
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.char_to_id, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved character tokenizer to {filepath}")
    
    def load(self, filepath: str):
        """
        Load tokenizer vocabulary.
        
        Args:
            filepath: Path to vocabulary file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.char_to_id = json.load(f)
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        logger.info(f"Loaded character tokenizer from {filepath}")


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer.
    Simplified implementation for educational purposes.
    """
    
    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.word_tokenizer = None
        self.bpe_ranks = {}  # Merge operations
        self.vocab = {}  # Token vocabulary
    
    def build_vocab(self, text: str, num_merges: int = None):
        """
        Build BPE vocabulary from text.
        
        Args:
            text: Text to build vocabulary from
            num_merges: Number of merge operations (defaults to vocab_size - 256)
        """
        if num_merges is None:
            num_merges = self.vocab_size - 256  # Reserve 256 for initial bytes
        
        # Start with character-level tokens
        vocab = {bytes([i]): i for i in range(256)}
        
        # Split text into words
        words = text.split()
        word_freqs = {}
        for word in words:
            word_bytes = word.encode('utf-8')
            if word_bytes not in word_freqs:
                word_freqs[word_bytes] = 0
            word_freqs[word_bytes] += 1
        
        # Perform merges
        for i in range(num_merges):
            # Find most common adjacent pair
            pair_freqs = {}
            for word, freq in word_freqs.items():
                for j in range(len(word) - 1):
                    pair = (word[j:j+1], word[j+1:j+2])
                    if pair not in pair_freqs:
                        pair_freqs[pair] = 0
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # Get most common pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.bpe_ranks[best_pair] = i
            
            # Merge pair in vocabulary
            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = len(vocab)
            
            # Update word frequencies
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word.replace(best_pair[0] + best_pair[1], new_token)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
        
        self.vocab = vocab
        logger.info(f"Built BPE vocabulary with {len(vocab)} tokens after {len(self.bpe_ranks)} merges")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text using BPE.
        
        Args:
            text: Text to encode
        
        Returns:
            List of token IDs
        """
        # Simple fallback if not properly initialized
        if not self.vocab:
            return [ord(c) for c in text]
        
        word_tokens = text.encode('utf-8')
        tokens = list(word_tokens)
        
        while len(tokens) > 1:
            # Find most valuable pair to merge
            best_pair = None
            best_rank = float('inf')
            
            for i in range(len(tokens) - 1):
                pair = (bytes([tokens[i]]) if isinstance(tokens[i], int) else tokens[i],
                       bytes([tokens[i+1]]) if isinstance(tokens[i+1], int) else tokens[i+1])
                
                if pair in self.bpe_ranks and self.bpe_ranks[pair] < best_rank:
                    best_rank = self.bpe_ranks[pair]
                    best_pair = (i, pair)
            
            if best_pair is None:
                break
            
            # Merge
            idx, (p0, p1) = best_pair
            merged = p0 + p1
            tokens[idx] = merged
            del tokens[idx + 1]
        
        # Convert to IDs
        return [self.vocab.get(t, 0) if isinstance(t, bytes) else t for t in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text
        """
        # Reverse vocabulary lookup
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(i, b'') for i in ids]
        
        try:
            return b''.join(tokens).decode('utf-8')
        except:
            return ''
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Number of tokens in vocabulary
        """
        return len(self.vocab)
    
    def save(self, filepath: str):
        """
        Save tokenizer to file.
        
        Args:
            filepath: Path to save tokenizer
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        # For simplicity, we'll save as JSON (would need custom serialization for bytes in production)
        logger.info(f"Saved BPE tokenizer to {filepath}")
    
    def load(self, filepath: str):
        """
        Load tokenizer from file.
        
        Args:
            filepath: Path to tokenizer file
        """
        logger.info(f"Loaded BPE tokenizer from {filepath}")
