"""Training metrics and tracking"""

import time
from typing import Dict, List, Optional
from collections import deque
import numpy as np


class MetricsTracker:
    """
    Track and compute training metrics.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Window size for running average
        """
        self.window_size = window_size
        self.metrics = {}
        self.windows = {}
        self.history = {}
        self.start_time = time.time()
    
    def update(self, **kwargs):
        """
        Update metrics.
        
        Args:
            **kwargs: Metric name and value pairs
        """
        for key, value in kwargs.items():
            # Initialize if needed
            if key not in self.windows:
                self.windows[key] = deque(maxlen=self.window_size)
                self.history[key] = []
            
            # Add to window and history
            self.windows[key].append(value)
            self.history[key].append(value)
            self.metrics[key] = value
    
    def get_avg(self, key: str) -> float:
        """
        Get running average for a metric.
        
        Args:
            key: Metric name
        
        Returns:
            Running average
        """
        if key in self.windows and len(self.windows[key]) > 0:
            return float(np.mean(self.windows[key]))
        return 0.0
    
    def get_all_avg(self) -> Dict[str, float]:
        """
        Get running averages for all metrics.
        
        Returns:
            Dictionary of metric averages
        """
        return {key: self.get_avg(key) for key in self.windows.keys()}
    
    def get_history(self, key: str) -> List[float]:
        """
        Get full history for a metric.
        
        Args:
            key: Metric name
        
        Returns:
            List of historical values
        """
        return self.history.get(key, [])
    
    def get_elapsed_time(self) -> float:
        """
        Get elapsed time since creation.
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time
    
    def reset(self):
        """
        Reset all metrics.
        """
        self.metrics.clear()
        self.windows.clear()
        self.history.clear()
        self.start_time = time.time()


class ProgressTracker:
    """
    Track training progress.
    """
    
    def __init__(self, total_steps: int):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
    
    def step(self):
        """
        Increment step counter.
        """
        self.current_step += 1
    
    def get_progress(self) -> float:
        """
        Get progress as percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100
    
    def get_elapsed_time(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time
        """
        return time.time() - self.start_time
    
    def get_eta(self) -> float:
        """
        Get estimated time to completion.
        
        Returns:
            ETA in seconds
        """
        if self.current_step == 0:
            return 0.0
        
        elapsed = self.get_elapsed_time()
        avg_time_per_step = elapsed / self.current_step
        remaining_steps = self.total_steps - self.current_step
        
        return avg_time_per_step * remaining_steps
    
    def get_speed(self) -> float:
        """
        Get processing speed (steps per second).
        
        Returns:
            Steps per second
        """
        elapsed = self.get_elapsed_time()
        if elapsed == 0:
            return 0.0
        return self.current_step / elapsed
