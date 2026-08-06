"""Text preprocessing utilities"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing utilities.
    """
    
    @staticmethod
    def clean_text(text: str, lowercase: bool = False, remove_special_chars: bool = False) -> str:
        """
        Clean text.
        
        Args:
            text: Text to clean
            lowercase: Convert to lowercase
            remove_special_chars: Remove special characters
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        if lowercase:
            text = text.lower()
        
        if remove_special_chars:
            # Keep only alphanumeric and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.!?,\'-]', '', text)
        
        return text
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace.
        
        Args:
            text: Text to normalize
        
        Returns:
            Normalized text
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved)
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty sentences and normalize
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    @staticmethod
    def get_text_stats(text: str) -> dict:
        """
        Get text statistics.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with text statistics
        """
        return {
            'characters': len(text),
            'words': len(text.split()),
            'lines': len(text.split('\n')),
            'sentences': len(TextPreprocessor.split_into_sentences(text)),
            'avg_word_length': sum(len(w) for w in text.split()) / len(text.split()) if text.split() else 0,
        }
