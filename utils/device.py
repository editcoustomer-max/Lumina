"""Device utilities"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(device: str = 'auto') -> torch.device:
    """
    Get torch device.
    
    Args:
        device: Device specification (cpu, cuda, auto)
    
    Returns:
        torch.device instance
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device_str = 'cuda'
            logger.info(f"CUDA available. Using GPU")
        else:
            device_str = 'cpu'
            logger.info(f"CUDA not available. Using CPU")
    else:
        device_str = device
    
    device_obj = torch.device(device_str)
    
    # Log device info
    if device_obj.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device_obj
