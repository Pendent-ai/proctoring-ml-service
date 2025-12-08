"""
Proctor Engine - Base Predictor Class

Handles inference logic with preprocessing, inference, and postprocessing.
"""

from abc import ABC, abstractmethod
from typing import Any, Generator
import numpy as np

from proctor.cfg import BaseConfig
from proctor.engine.results import Results


class BasePredictor(ABC):
    """
    Base class for all predictors.
    
    Predictors handle the inference pipeline:
    1. preprocess() - Prepare input data
    2. inference() - Run model inference
    3. postprocess() - Process model outputs
    
    Attributes:
        cfg: Configuration for prediction
        model: Loaded model instance
        device: Computation device (cpu/cuda)
    
    Example:
        >>> predictor = VideoPredictor(cfg)
        >>> results = predictor(frame)
    """
    
    def __init__(self, cfg: BaseConfig):
        """
        Initialize predictor.
        
        Args:
            cfg: Predictor configuration
        """
        self.cfg = cfg
        self.model = None
        self.device = "cpu"
        self._callbacks = {}
        
        self.setup_model()
    
    @abstractmethod
    def setup_model(self):
        """Load and setup the model."""
        pass
    
    @abstractmethod
    def preprocess(self, source: Any) -> Any:
        """
        Preprocess input before inference.
        
        Args:
            source: Raw input (image, audio, etc.)
            
        Returns:
            Preprocessed input ready for model
        """
        pass
    
    @abstractmethod
    def inference(self, data: Any) -> Any:
        """
        Run model inference.
        
        Args:
            data: Preprocessed input
            
        Returns:
            Raw model outputs
        """
        pass
    
    @abstractmethod
    def postprocess(self, preds: Any, source: Any) -> Results:
        """
        Postprocess model outputs.
        
        Args:
            preds: Raw model predictions
            source: Original input for reference
            
        Returns:
            Results object with processed predictions
        """
        pass
    
    def __call__(self, source: Any, **kwargs) -> Results:
        """
        Run full prediction pipeline.
        
        Args:
            source: Input source
            **kwargs: Additional arguments
            
        Returns:
            Prediction results
        """
        # Preprocess
        preprocessed = self.preprocess(source)
        
        # Inference
        preds = self.inference(preprocessed)
        
        # Postprocess
        results = self.postprocess(preds, source)
        
        return results
    
    def stream(self, source: Any) -> Generator[Results, None, None]:
        """
        Stream predictions for continuous input.
        
        Args:
            source: Streaming source
            
        Yields:
            Results for each frame/chunk
        """
        raise NotImplementedError("Streaming not implemented")
    
    def warmup(self, imgsz: tuple = (640, 480)):
        """
        Warmup model with dummy data.
        
        Args:
            imgsz: Image size for warmup (width, height)
        """
        if self.model is not None:
            dummy = np.zeros((*imgsz[::-1], 3), dtype=np.uint8)
            self.preprocess(dummy)
            print(f"✅ {self.__class__.__name__} warmup complete")
    
    def add_callback(self, event: str, callback):
        """Add callback for an event."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def run_callbacks(self, event: str, *args, **kwargs):
        """Run all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            callback(*args, **kwargs)
