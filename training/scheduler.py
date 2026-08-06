"""Learning rate schedulers"""

import math
import logging
from typing import List

logger = logging.getLogger(__name__)


class LRScheduler:
    """
    Base learning rate scheduler.
    """
    
    def __init__(self, optimizer, base_lr: float):
        """
        Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_count = 0
    
    def step(self):
        """
        Update learning rate.
        """
        self.step_count += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """
        Get current learning rate.
        
        Returns:
            Learning rate
        """
        return self.base_lr


class ConstantLR(LRScheduler):
    """
    Constant learning rate (no scheduling).
    """
    
    def get_lr(self) -> float:
        return self.base_lr


class LinearWarmup(LRScheduler):
    """
    Linear warmup followed by constant learning rate.
    """
    
    def __init__(self, optimizer, base_lr: float, warmup_steps: int):
        """
        Initialize linear warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate
            warmup_steps: Number of warmup steps
        """
        super().__init__(optimizer, base_lr)
        self.warmup_steps = warmup_steps
    
    def get_lr(self) -> float:
        if self.step_count < self.warmup_steps:
            return self.base_lr * (self.step_count + 1) / self.warmup_steps
        return self.base_lr


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate scheduler.
    """
    
    def __init__(self, optimizer, base_lr: float, total_steps: int, min_lr: float = 0):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate
            total_steps: Total number of steps
            min_lr: Minimum learning rate
        """
        super().__init__(optimizer, base_lr)
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def get_lr(self) -> float:
        if self.step_count > self.total_steps:
            return self.min_lr
        
        progress = self.step_count / self.total_steps
        lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        return lr


class CosineAnnealingWarmup(LRScheduler):
    """
    Cosine annealing with linear warmup.
    """
    
    def __init__(self, optimizer, base_lr: float, warmup_steps: int, 
                 total_steps: int, min_lr: float = 0):
        """
        Initialize cosine annealing with warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate
            warmup_steps: Number of warmup steps
            total_steps: Total number of steps
            min_lr: Minimum learning rate
        """
        super().__init__(optimizer, base_lr)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def get_lr(self) -> float:
        if self.step_count < self.warmup_steps:
            return self.base_lr * (self.step_count + 1) / self.warmup_steps
        
        progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        return lr
