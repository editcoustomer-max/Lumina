"""General helper utilities"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        path: Path to YAML file
    
    Returns:
        Dictionary with loaded configuration
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str):
    """
    Save configuration to YAML file.
    
    Args:
        data: Configuration dictionary
        path: Path to save YAML file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: Path to save JSON file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_size(path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
    
    Returns:
        File size in bytes
    """
    return os.path.getsize(path)


def format_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get detailed model information.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "parameter_size_mb": (total_params * 4) / (1024 * 1024),  # 4 bytes per float32
    }


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    elif minutes > 0:
        return f"{minutes:02d}:{secs:02d}"
    else:
        return f"{secs}s"
