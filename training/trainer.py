"""Model trainer"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from model.transformer import TransformerLM
from model.checkpoint import CheckpointManager
from data.loader import DataLoaderFactory
from data.dataset import DatasetLoader
from training.loss import LanguageModelLoss
from training.scheduler import CosineAnnealingWarmup
from utils.metrics import MetricsTracker
from utils.helpers import load_yaml, format_time

logger = logging.getLogger(__name__)


class Trainer:
    """
    Model trainer.
    """
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            device: Device to train on
        """
        self.config = config
        self.device = torch.device(device)
        self.global_step = 0
        self.current_epoch = 0
        
        # Initialize model
        model_config = config['model']
        self.model = TransformerLM(
            vocab_size=model_config['vocab_size'],
            embedding_dim=model_config['embedding_dim'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            ff_dim=model_config['ff_dim'],
            max_seq_length=model_config['max_seq_length'],
            dropout=model_config.get('dropout', 0.1),
        ).to(self.device)
        
        # Initialize optimizer
        training_config = config['training']
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01),
        )
        
        # Initialize scheduler
        total_steps = training_config['epochs'] * 100  # Approximate
        self.scheduler = CosineAnnealingWarmup(
            self.optimizer,
            base_lr=training_config['learning_rate'],
            warmup_steps=training_config.get('warmup_steps', 0),
            total_steps=total_steps,
        )
        
        # Loss function
        self.loss_fn = LanguageModelLoss(model_config['vocab_size'])
        
        # Metrics
        self.metrics = MetricsTracker()
        
        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        epoch_metrics = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            loss = self.loss_fn(logits, target_ids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            self.metrics.update(loss=loss.item())
            self.global_step += 1
            
            # Log
            if (batch_idx + 1) % self.config['logging'].get('log_interval', 10) == 0:
                avg_loss = self.metrics.get_avg('loss')
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'lr': f"{current_lr:.2e}",
                })
                epoch_metrics['train_loss'] = avg_loss
                epoch_metrics['learning_rate'] = current_lr
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_metrics = MetricsTracker()
        
        pbar = tqdm(val_loader, desc="Validating")
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits = self.model(input_ids)
            loss = self.loss_fn(logits, target_ids)
            
            val_metrics.update(loss=loss.item())
            pbar.set_postfix({'loss': f"{val_metrics.get_avg('loss'):.4f}"})
        
        avg_val_loss = val_metrics.get_avg('loss')
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        return {'val_loss': avg_val_loss}
    
    def fit(self, train_loader, val_loader = None):
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        num_epochs = self.config['training']['epochs']
        checkpoint_config = self.config['checkpointing']
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_config.get('save_interval', 1) == 0:
                checkpoint_path = Path(checkpoint_config['checkpoint_dir']) / f"model_epoch_{epoch+1}.pt"
                CheckpointManager.save_checkpoint(
                    str(checkpoint_path),
                    self.model,
                    self.optimizer,
                    epoch=epoch + 1,
                    step=self.global_step,
                    metrics={**train_metrics, **val_metrics},
                )
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_metrics.get('train_loss', 0):.4f}")
