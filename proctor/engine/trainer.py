from __future__ import annotations
"""
Proctor Engine - Base Trainer Class

Handles model training with data loading, optimization, and checkpointing.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from proctor.cfg import BaseConfig


class BaseTrainer(ABC):
    """
    Base class for all trainers.
    
    Trainers handle the training pipeline:
    1. setup() - Initialize training components
    2. train_epoch() - Train for one epoch
    3. validate() - Validate model
    4. save() - Checkpoint model
    
    Attributes:
        cfg: Training configuration
        model: Model to train
        optimizer: Optimizer instance
        epochs: Total training epochs
    
    Example:
        >>> trainer = ClassifierTrainer(cfg)
        >>> results = trainer.train(data_path, epochs=100)
    """
    
    def __init__(self, cfg: BaseConfig):
        """
        Initialize trainer.
        
        Args:
            cfg: Training configuration
        """
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        self.epochs = 0
        self.start_epoch = 0
        self.best_fitness = 0.0
        
        self.save_dir = Path("runs/train")
        self.weights_dir = self.save_dir / "weights"
        
        self._callbacks = {}
    
    @abstractmethod
    def setup(self, data: str | Path):
        """
        Setup training components.
        
        Args:
            data: Path to training data
        """
        pass
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Training metrics for this epoch
        """
        pass
    
    @abstractmethod
    def validate(self) -> dict:
        """
        Validate current model.
        
        Returns:
            Validation metrics
        """
        pass
    
    def __call__(
        self,
        data: str | Path,
        epochs: int = 100,
        patience: int = 10,
        resume: bool = False,
        **kwargs,
    ) -> dict:
        """
        Run full training.
        
        Args:
            data: Path to training data
            epochs: Number of training epochs
            patience: Early stopping patience
            resume: Resume from checkpoint
            **kwargs: Additional training arguments
            
        Returns:
            Final training results
        """
        self.epochs = epochs
        
        # Setup
        self._create_dirs()
        self.setup(data)
        
        if resume:
            self._resume_training()
        
        # Training loop
        no_improvement = 0
        
        for epoch in range(self.start_epoch, epochs):
            self.run_callbacks("on_epoch_start", epoch)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Check improvement
            fitness = self._get_fitness(val_metrics)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.save(self.weights_dir / "best.pt")
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Save last
            self.save(self.weights_dir / "last.pt")
            
            # Log
            self._log_epoch(epoch, train_metrics, val_metrics)
            
            self.run_callbacks("on_epoch_end", epoch, train_metrics, val_metrics)
            
            # Early stopping
            if no_improvement >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break
        
        # Final results
        return {
            "best_fitness": self.best_fitness,
            "epochs_trained": epoch + 1,
            "weights_path": str(self.weights_dir / "best.pt"),
        }
    
    def save(self, path: Path):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": self.epochs,
            "best_fitness": self.best_fitness,
            "model": self.model,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "date": datetime.now().isoformat(),
        }
        # Subclasses implement actual saving
        self._save_checkpoint(checkpoint, path)
    
    def _save_checkpoint(self, checkpoint: dict, path: Path):
        """Save checkpoint to file. Override in subclasses."""
        pass
    
    def _resume_training(self):
        """Resume training from checkpoint."""
        last_path = self.weights_dir / "last.pt"
        if last_path.exists():
            print(f"📥 Resuming from {last_path}")
            # Subclasses implement loading
    
    def _create_dirs(self):
        """Create training directories."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_fitness(self, metrics: dict) -> float:
        """Calculate fitness from metrics. Override for custom fitness."""
        return metrics.get("accuracy", 0.0)
    
    def _log_epoch(self, epoch: int, train: dict, val: dict):
        """Log epoch metrics."""
        print(f"Epoch {epoch + 1}/{self.epochs} - Train: {train}, Val: {val}")
    
    def add_callback(self, event: str, callback):
        """Add callback for an event."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def run_callbacks(self, event: str, *args, **kwargs):
        """Run all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            callback(*args, **kwargs)
