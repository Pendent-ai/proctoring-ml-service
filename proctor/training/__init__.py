"""
Proctor training module.

Provides training pipelines for:
- Video clip-based cheating detection
- Custom YOLO training
- Ensemble and temporal model training
"""

from proctor.training.clip_trainer import (
    TrainingConfig,
    TrainingResult,
    VideoClipTrainer,
    train_from_config,
)

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "VideoClipTrainer",
    "train_from_config",
]
