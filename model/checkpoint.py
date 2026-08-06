"""Model checkpoint management"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage model checkpoints.
    """
    
    @staticmethod
    def save_checkpoint(filepath: str, model: torch.nn.Module, 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: int = 0, step: int = 0, 
                       metrics: Optional[Dict[str, float]] = None):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            model: Model to save
            optimizer: Optional optimizer to save
            epoch: Current epoch
            step: Current training step
            metrics: Optional metrics dictionary
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model.get_config() if hasattr(model, 'get_config') else None,
            'epoch': epoch,
            'step': step,
            'metrics': metrics or {}
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    @staticmethod
    def load_checkpoint(filepath: str, device: str = 'cpu') -> Dict[str, Any]:
        """
        Load checkpoint from file.
        
        Args:
            filepath: Path to checkpoint
            device: Device to load onto
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=device)
        logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint
    
    @staticmethod
    def load_model_from_checkpoint(checkpoint_path: str, model_class = None, 
                                   device: str = 'cpu'):
        """
        Load model from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint
            model_class: Model class (optional, will infer from checkpoint)
            device: Device to load model on
        
        Returns:
            Loaded model
        """
        checkpoint = CheckpointManager.load_checkpoint(checkpoint_path, device)
        
        # If model class provided, create instance
        if model_class and 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = model_class(**config)
        else:
            logger.error("Model class required to load from checkpoint")
            return None
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        logger.info(f"Loaded model on device: {device}")
        return model
    
    @staticmethod
    def load_optimizer_from_checkpoint(checkpoint_path: str, optimizer) -> int:
        """
        Load optimizer state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            optimizer: Optimizer instance to load state into
        
        Returns:
            Starting epoch/step
        """
        checkpoint = CheckpointManager.load_checkpoint(checkpoint_path)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")
        
        return checkpoint.get('epoch', 0)
