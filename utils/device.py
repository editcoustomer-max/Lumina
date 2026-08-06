"""Device detection and management"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(device: str = "auto") -> str:
    """
    Detect and return the best available device.
    
    Args:
        device: Device type - "cpu", "cuda", "mps", or "auto"
    
    Returns:
        Device string ("cpu" or "cuda" or "mps")
    """
    if device != "auto":
        return device
    
    if torch.cuda.is_available():
        logger.info(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        return "cuda"
    
    # Check for Apple Metal Performance Shaders
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Apple Metal Performance Shaders (MPS) detected")
        return "mps"
    
    logger.info("No GPU detected, using CPU")
    return "cpu"


def get_device_info() -> dict:
    """
    Get detailed device information.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "device": get_device(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads(),
    }
    
    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
    
    return info
