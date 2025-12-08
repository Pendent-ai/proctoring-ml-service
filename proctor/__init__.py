"""
Proctor - Real-time ML Proctoring System

A professional ML-based proctoring system following industry best practices.
Inspired by Ultralytics YOLO architecture patterns.

Usage:
    from proctor import VideoProctor, AudioProctor

    # Video proctoring
    video_proctor = VideoProctor()
    result = video_proctor.predict(frame)

    # Audio proctoring
    audio_proctor = AudioProctor()
    result = audio_proctor.predict(audio_samples)

    # Full pipeline with LiveKit
    from proctor import ProctoringService
    service = ProctoringService()
    await service.connect(room_name)

    # Training
    from proctor.training import VideoClipTrainer, TrainingConfig
    trainer = VideoClipTrainer(TrainingConfig(model_type="temporal"))
    result = trainer.train(annotations)
"""

__version__ = "0.1.0"
__author__ = "Pendent AI"

# Lazy imports for faster startup
from proctor.engine.model import BaseModel
from proctor.models.video import VideoProctor
from proctor.models.audio import AudioProctor
from proctor.cfg import VideoConfig, AudioConfig, get_settings

# Advanced components (lazy loaded when accessed)
def __getattr__(name: str):
    """Lazy load advanced components."""
    if name == "EnsembleClassifier":
        from proctor.engine.classifier import EnsembleClassifier
        return EnsembleClassifier
    elif name == "TemporalCheatingDetector":
        from proctor.engine.temporal import TemporalCheatingDetector
        return TemporalCheatingDetector
    elif name == "MultimodalDetector":
        from proctor.engine.fusion import MultimodalDetector
        return MultimodalDetector
    elif name == "FeatureExtractor":
        from proctor.engine.features import FeatureExtractor
        return FeatureExtractor
    elif name == "VideoClipTrainer":
        from proctor.training.clip_trainer import VideoClipTrainer
        return VideoClipTrainer
    elif name == "YOLOTrainer":
        from proctor.models.video.yolo_trainer import YOLOTrainer
        return YOLOTrainer
    elif name == "ProctoringService":
        from proctor.service.proctoring import ProctoringService
        return ProctoringService
    raise AttributeError(f"module 'proctor' has no attribute '{name}'")

# Public API
__all__ = [
    # Core models
    "BaseModel",
    "VideoProctor",
    "AudioProctor",
    # Configs
    "VideoConfig",
    "AudioConfig",
    "get_settings",
    # Advanced (lazy loaded)
    "EnsembleClassifier",
    "TemporalCheatingDetector",
    "MultimodalDetector",
    "FeatureExtractor",
    "VideoClipTrainer",
    "YOLOTrainer",
    "ProctoringService",
    # Version
    "__version__",
]
