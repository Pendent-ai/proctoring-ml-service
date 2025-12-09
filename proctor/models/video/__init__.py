from __future__ import annotations
"""
Video Proctoring Model

Combines YOLO object detection and MediaPipe face analysis.
"""

from proctor.models.video.model import VideoProctor
from proctor.models.video.predictor import VideoPredictor
from proctor.models.video.trainer import ClassifierTrainer
from proctor.models.video.validator import VideoValidator

__all__ = [
    "VideoProctor",
    "VideoPredictor",
    "ClassifierTrainer",
    "VideoValidator",
]
