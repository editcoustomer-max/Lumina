"""Model checkpointing utilities"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from model.transformer import TransformerLM

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage model checkpoints (saving and loading).
    """
    
    @staticmethod
    def save_checkpoint(filepath: str, model: TransformerLM, optimizer: torch.optim.Optimizer = None,
                       epoch: int = 0, step: int = 0, metrics: Dict[str, float] = None, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            model: Model to save
            optimizer: Optional optimizer state
            epoch: Current epoch
            step: Current step
            metrics: Optional metrics dictionary
            **kwargs: Additional data to save
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model.get_config(),
            'epoch': epoch,
            'step': step,
            'metrics': metrics or {},
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add any additional data
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath} (epoch={epoch}, step={step})")
    
    @staticmethod
    def load_checkpoint(filepath: str, model: TransformerLM = None, optimizer: torch.optim.Optimizer = None,
                       load_optimizer: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint
            model: Model to load weights into (optional)
            optimizer: Optimizer to load state into (optional)
            load_optimizer: Whether to load optimizer state
        
        Returns:
            Checkpoint dictionary
        """
        if not Path(filepath).exists():
            logger.error(f"Checkpoint not found: {filepath}")
            return {}
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model weights from {filepath}")
        
        if optimizer is not None and load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded optimizer state from {filepath}")
        
        logger.info(f"Loaded checkpoint: epoch={checkpoint.get('epoch', 0)}, step={checkpoint.get('step', 0)}")
        
        return checkpoint
    
    @staticmethod
    def load_model_from_checkpoint(filepath: str, device: str = 'cpu') -> TransformerLM:
        """
        Load a complete model from checkpoint.
        
        Args:
            filepath: Path to checkpoint
            device: Device to load model to
        
        Returns:
            Loaded model
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Recreate model from config
        config = checkpoint['model_config']
        model = TransformerLM.from_config(config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {filepath}")
        
        return model
    
    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """
        Find the latest checkpoint in a directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None
        
        # Find all .pt files
        checkpoints = list(checkpoint_path.glob('*.pt'))
        if not checkpoints:
            return None
        
        # Return most recently modified
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
