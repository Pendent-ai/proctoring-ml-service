"""
Proctor Engine - Base Validator Class

Handles model validation and metrics computation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from proctor.cfg import BaseConfig


class BaseValidator(ABC):
    """
    Base class for all validators.
    
    Validators handle the validation pipeline:
    1. setup() - Initialize validation components
    2. preprocess() - Prepare validation data
    3. evaluate() - Run evaluation
    4. compute_metrics() - Calculate metrics
    
    Attributes:
        cfg: Validation configuration
        model: Model to validate
        data: Validation data
    
    Example:
        >>> validator = ClassifierValidator(cfg)
        >>> metrics = validator(data_path)
    """
    
    def __init__(self, cfg: BaseConfig):
        """
        Initialize validator.
        
        Args:
            cfg: Validation configuration
        """
        self.cfg = cfg
        self.model = None
        self.data = None
        
        self.predictions = []
        self.targets = []
        
        self._callbacks = {}
    
    @abstractmethod
    def setup(self, data: str | Path | None = None):
        """
        Setup validation components.
        
        Args:
            data: Path to validation data
        """
        pass
    
    @abstractmethod
    def evaluate(self) -> dict:
        """
        Run evaluation on data.
        
        Returns:
            Raw evaluation results
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, results: dict) -> dict:
        """
        Compute metrics from evaluation results.
        
        Args:
            results: Raw evaluation results
            
        Returns:
            Computed metrics dictionary
        """
        pass
    
    def __call__(
        self,
        data: str | Path | None = None,
        **kwargs,
    ) -> dict:
        """
        Run full validation pipeline.
        
        Args:
            data: Path to validation data
            **kwargs: Additional validation arguments
            
        Returns:
            Validation metrics
        """
        self.run_callbacks("on_val_start")
        
        # Setup
        self.setup(data)
        
        # Evaluate
        results = self.evaluate()
        
        # Compute metrics
        metrics = self.compute_metrics(results)
        
        self.run_callbacks("on_val_end", metrics)
        
        return metrics
    
    def print_results(self, metrics: dict):
        """Print validation results."""
        print("\n" + "=" * 50)
        print("Validation Results")
        print("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 50 + "\n")
    
    def add_callback(self, event: str, callback):
        """Add callback for an event."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def run_callbacks(self, event: str, *args, **kwargs):
        """Run all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            callback(*args, **kwargs)
