"""Data loader for training"""

import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    PyTorch Dataset for text data.
    """
    
    def __init__(self, text: str, tokenizer, seq_length: int = 512, stride: int = 1):
        """
        Initialize text dataset.
        
        Args:
            text: Full text corpus
            tokenizer: Tokenizer to use
            seq_length: Sequence length for training
            stride: Stride for creating sequences
        """
        self.seq_length = seq_length
        self.stride = stride
        self.tokenizer = tokenizer
        
        # Tokenize the entire text
        logger.info("Tokenizing dataset...")
        self.token_ids = tokenizer.encode(text)
        logger.info(f"Total tokens: {len(self.token_ids)}")
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.token_ids) - seq_length, stride):
            seq = self.token_ids[i:i + seq_length + 1]  # +1 for target
            self.sequences.append(seq)
        
        logger.info(f"Created {len(self.sequences)} sequences of length {seq_length}")
    
    def __len__(self) -> int:
        """
        Get dataset size.
        
        Returns:
            Number of sequences
        """
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target.
        
        Args:
            idx: Index of sequence
        
        Returns:
            Tuple of (input_ids, target_ids)
        """
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        
        return input_ids, target_ids


class DataLoaderFactory:
    """
    Factory for creating data loaders.
    """
    
    @staticmethod
    def create_loader(text: str, tokenizer, seq_length: int = 512, 
                     batch_size: int = 32, shuffle: bool = True,
                     num_workers: int = 0, pin_memory: bool = True,
                     train_split: float = 0.9) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders.
        
        Args:
            text: Full text corpus
            tokenizer: Tokenizer to use
            seq_length: Sequence length
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory
            train_split: Training/validation split ratio
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create dataset
        dataset = TextDataset(text, tokenizer, seq_length)
        
        # Split into train and validation
        train_size = int(len(dataset) * train_split)
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        logger.info(f"Train: {train_size}, Validation: {val_size}")
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        
        return train_loader, val_loader
    
    @staticmethod
    def create_inference_loader(text: str, tokenizer, seq_length: int = 512,
                               batch_size: int = 32, num_workers: int = 0) -> DataLoader:
        """
        Create data loader for inference (no shuffling, no target labels).
        
        Args:
            text: Text to load
            tokenizer: Tokenizer to use
            seq_length: Sequence length
            batch_size: Batch size
            num_workers: Number of workers
        
        Returns:
            Data loader
        """
        dataset = TextDataset(text, tokenizer, seq_length, stride=seq_length)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        return loader
