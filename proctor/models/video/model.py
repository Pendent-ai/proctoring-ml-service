"""
VideoProctor Model

Main video proctoring model combining YOLO and MediaPipe.
"""

from pathlib import Path
from typing import Any

from proctor.engine.model import BaseModel
from proctor.cfg import VideoConfig, get_settings


class VideoProctor(BaseModel):
    """
    Video proctoring model combining object detection and face analysis.
    
    Uses YOLO11 for object detection (phone, multiple persons) and
    MediaPipe for face/gaze analysis.
    
    Example:
        >>> from proctor import VideoProctor
        >>> model = VideoProctor()
        >>> results = model.predict(frame)
        >>> print(results.phone_detected, results.looking_away)
    """
    
    def __init__(
        self,
        model: str | Path | None = None,
        verbose: bool = True,
    ):
        """
        Initialize VideoProctor.
        
        Args:
            model: Path to model weights (optional)
            verbose: Enable verbose output
        """
        super().__init__(model=model, task="video", verbose=verbose)
        
        # Load settings if no model path provided
        if model is None:
            settings = get_settings()
            self.model = settings.yolo_model_path
    
    @property
    def task_map(self) -> dict:
        """Map task to component classes."""
        from proctor.models.video.predictor import VideoPredictor
        from proctor.models.video.trainer import ClassifierTrainer
        from proctor.models.video.validator import VideoValidator
        
        return {
            "predictor": VideoPredictor,
            "trainer": ClassifierTrainer,
            "validator": VideoValidator,
        }
    
    @property
    def cfg(self) -> VideoConfig:
        """Get model configuration."""
        if self._cfg is None:
            settings = get_settings()
            self._cfg = settings.to_video_config()
        return self._cfg
    
    def detect_objects(self, frame: Any) -> dict:
        """
        Run only object detection.
        
        Args:
            frame: Image frame
            
        Returns:
            Detection results dict
        """
        return self.predictor.detect_objects(frame)
    
    def analyze_face(self, frame: Any) -> dict:
        """
        Run only face analysis.
        
        Args:
            frame: Image frame
            
        Returns:
            Face analysis results dict
        """
        return self.predictor.analyze_face(frame)
    
    def export(self, format: str = "onnx", **kwargs) -> Path:
        """
        Export model to different format.
        
        Args:
            format: Export format (onnx, torchscript)
            **kwargs: Export arguments
            
        Returns:
            Path to exported model
        """
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized")
        
        return self.predictor.export(format, **kwargs)
    
    def close(self):
        """Release resources."""
        if self._predictor is not None:
            self._predictor.close()
