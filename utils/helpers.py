"""Helper functions"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import time

logger = logging.getLogger(__name__)


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        filepath: Path to YAML file
    
    Returns:
        Parsed YAML as dictionary
    """
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {filepath}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {filepath}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise


def save_yaml(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to YAML file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info(f"Saved configuration to {filepath}")


def ensure_dir(directory: str):
    """
    Ensure directory exists.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Format seconds into human readable time.
    
    Args:
        seconds: Number of seconds
    
    Returns:
        Formatted time string
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def count_parameters(model) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
