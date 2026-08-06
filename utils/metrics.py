"""Metrics tracking utilities"""

import logging
from typing import Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track metrics during training.
    """
    
    def __init__(self):
        """
        Initialize metrics tracker.
        """
        self.metrics = defaultdict(list)
    
    def update(self, **kwargs):
        """
        Update metrics.
        
        Args:
            **kwargs: Metric name and value pairs
        """
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_avg(self, key: str) -> float:
        """
        Get average value of metric.
        
        Args:
            key: Metric name
        
        Returns:
            Average value
        """
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return sum(self.metrics[key]) / len(self.metrics[key])
    
    def get_latest(self, key: str) -> float:
        """
        Get latest value of metric.
        
        Args:
            key: Metric name
        
        Returns:
            Latest value
        """
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]
    
    def reset(self):
        """
        Reset all metrics.
        """
        self.metrics.clear()
    
    def get_dict(self) -> Dict[str, list]:
        """
        Get all metrics as dictionary.
        
        Returns:
            Dictionary of metrics
        """
        return dict(self.metrics)
