"""
Proctor Engine - Base Model Class

Following Ultralytics pattern for model architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from proctor.cfg import BaseConfig


class BaseModel(ABC):
    """
    Base class for all Proctor models.
    
    This class defines the common interface for video and audio proctoring models.
    Following Ultralytics pattern with task_map for modular architecture.
    
    Attributes:
        cfg: Model configuration
        predictor: Predictor instance for inference
        trainer: Trainer instance for training
        validator: Validator instance for evaluation
    
    Example:
        >>> from proctor import VideoProctor
        >>> model = VideoProctor()
        >>> results = model.predict(frame)
        >>> results = model.train(data_path)
    """
    
    def __init__(
        self,
        model: Optional[Union[str, Path]] = None,
        task: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize BaseModel.
        
        Args:
            model: Path to model weights or model name
            task: Model task (e.g., 'video', 'audio')
            verbose: Enable verbose output
        """
        self.model = model
        self.task = task
        self.verbose = verbose
        
        # Lazy-loaded components
        self._predictor = None
        self._trainer = None
        self._validator = None
        self._cfg = None
    
    @property
    @abstractmethod
    def task_map(self) -> dict:
        """
        Map task to model, trainer, validator, predictor classes.
        
        Returns:
            Dictionary mapping to component classes:
            {
                "model": ModelClass,
                "trainer": TrainerClass,
                "validator": ValidatorClass,
                "predictor": PredictorClass,
            }
        """
        pass
    
    @property
    def predictor(self):
        """Get or create predictor instance."""
        if self._predictor is None:
            predictor_cls = self.task_map.get("predictor")
            if predictor_cls:
                self._predictor = predictor_cls(self.cfg)
        return self._predictor
    
    @property
    def trainer(self):
        """Get or create trainer instance."""
        if self._trainer is None:
            trainer_cls = self.task_map.get("trainer")
            if trainer_cls:
                self._trainer = trainer_cls(self.cfg)
        return self._trainer
    
    @property
    def validator(self):
        """Get or create validator instance."""
        if self._validator is None:
            validator_cls = self.task_map.get("validator")
            if validator_cls:
                self._validator = validator_cls(self.cfg)
        return self._validator
    
    @property
    @abstractmethod
    def cfg(self) -> BaseConfig:
        """Get model configuration."""
        pass
    
    def predict(
        self,
        source: Any,
        **kwargs,
    ):
        """
        Run prediction on source.
        
        Args:
            source: Image, video, audio, or path
            **kwargs: Additional prediction arguments
            
        Returns:
            Results object with predictions
        """
        if self.predictor is None:
            raise RuntimeError("Predictor not available for this model")
        return self.predictor(source, **kwargs)
    
    def train(
        self,
        data: str | Path,
        **kwargs,
    ):
        """
        Train the model.
        
        Args:
            data: Path to training data
            **kwargs: Additional training arguments
            
        Returns:
            Training results
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not available for this model")
        return self.trainer(data, **kwargs)
    
    def val(
        self,
        data: str | Path | None = None,
        **kwargs,
    ):
        """
        Validate the model.
        
        Args:
            data: Path to validation data
            **kwargs: Additional validation arguments
            
        Returns:
            Validation metrics
        """
        if self.validator is None:
            raise RuntimeError("Validator not available for this model")
        return self.validator(data, **kwargs)
    
    def export(
        self,
        format: str = "onnx",
        **kwargs,
    ) -> Path:
        """
        Export model to different format.
        
        Args:
            format: Export format (onnx, torchscript, etc.)
            **kwargs: Export arguments
            
        Returns:
            Path to exported model
        """
        raise NotImplementedError("Export not implemented for this model")
    
    def info(self) -> dict:
        """Get model information."""
        return {
            "task": self.task,
            "model": str(self.model),
            "config": self.cfg.model_dump() if hasattr(self.cfg, "model_dump") else {},
        }
    
    def __call__(self, source: Any, **kwargs):
        """Run prediction when model is called."""
        return self.predict(source, **kwargs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, model={self.model})"
