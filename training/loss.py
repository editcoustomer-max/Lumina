"""Loss functions for training"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class LanguageModelLoss(nn.Module):
    """
    Language model loss (cross-entropy).
    """
    
    def __init__(self, vocab_size: int, ignore_index: int = -100):
        """
        Initialize loss function.
        
        Args:
            vocab_size: Vocabulary size
            ignore_index: Token index to ignore in loss calculation
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
        
        Returns:
            Loss value (scalar)
        """
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Calculate loss
        loss = self.loss_fn(logits_flat, targets_flat)
        
        # Average loss (ignoring padded tokens)
        valid_mask = targets_flat != self.ignore_index
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = loss.mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling hard examples.
    Reduces weight of easy examples and focuses on hard ones.
    """
    
    def __init__(self, vocab_size: int, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -100):
        """
        Initialize focal loss.
        
        Args:
            vocab_size: Vocabulary size
            alpha: Weighting factor
            gamma: Focusing parameter
            ignore_index: Token index to ignore
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
        
        Returns:
            Loss value (scalar)
        """
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Cross-entropy loss
        ce_loss = self.ce_loss(logits_flat, targets_flat)
        
        # Probability of true class
        p = torch.exp(-ce_loss)
        
        # Focal loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        # Average
        valid_mask = targets_flat != self.ignore_index
        if valid_mask.sum() > 0:
            focal_loss = focal_loss[valid_mask].mean()
        else:
            focal_loss = focal_loss.mean()
        
        return focal_loss
