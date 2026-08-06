"""Validation utilities"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Validator:
    """
    Model validator.
    """
    
    def __init__(self, model: nn.Module, loss_fn: nn.Module, device: str = 'cpu'):
        """
        Initialize validator.
        
        Args:
            model: Model to validate
            loss_fn: Loss function
            device: Device to validate on
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = torch.device(device)
    
    @torch.no_grad()
    def validate(self, val_loader, num_batches: int = None) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
            num_batches: Optional limit on number of batches to evaluate
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(tqdm(val_loader, desc="Validating")):
            if num_batches and batch_idx >= num_batches:
                break
            
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            loss = self.loss_fn(logits, target_ids)
            
            total_loss += loss.item() * input_ids.size(0)
            num_samples += input_ids.size(0)
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        logger.info(f"Validation loss: {avg_loss:.4f}")
        
        return {
            'val_loss': avg_loss,
            'num_samples': num_samples,
        }
    
    @torch.no_grad()
    def compute_perplexity(self, val_loader, num_batches: int = None) -> float:
        """
        Compute perplexity on validation set.
        
        Args:
            val_loader: Validation data loader
            num_batches: Optional limit on number of batches
        
        Returns:
            Perplexity value
        """
        metrics = self.validate(val_loader, num_batches)
        perplexity = torch.exp(torch.tensor(metrics['val_loss'])).item()
        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity
