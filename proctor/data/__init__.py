"""
Proctor Data Module

LiveKit integration for video/audio streams, alert publishing,
and dataset utilities for training.
"""

from proctor.data.video_receiver import VideoReceiver
from proctor.data.audio_receiver import AudioReceiver
from proctor.data.publisher import AlertPublisher
from proctor.data.dataset import (
    ClipAnnotation,
    FrameAnnotation,
    AnnotationLoader,
    VideoReader,
    FrameDataset,
    FeatureNormalizer,
    DataAugmentation,
)

__all__ = [
    # LiveKit integration
    "VideoReceiver",
    "AudioReceiver",
    "AlertPublisher",
    # Dataset utilities
    "ClipAnnotation",
    "FrameAnnotation",
    "AnnotationLoader",
    "VideoReader",
    "FrameDataset",
    "FeatureNormalizer",
    "DataAugmentation",
    # Data collectors
    "InterviewDataCollector",
    "SyntheticDataGenerator",
    "BehaviorLabel",
    "AudioLabel",
]

# Import collectors
from proctor.data.collectors import (
    InterviewDataCollector,
    SyntheticDataGenerator,
    BehaviorLabel,
    AudioLabel,
)
