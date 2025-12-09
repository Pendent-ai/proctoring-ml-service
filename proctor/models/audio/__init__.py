from __future__ import annotations
"""
Audio Proctoring Model

Voice activity detection, speaker analysis, and AI voice detection.
"""

from proctor.models.audio.model import AudioProctor
from proctor.models.audio.predictor import AudioPredictor
from proctor.models.audio.audio_models import (
    AudioConfig,
    AudioFeatureExtractor,
    AudioEncoder,
    VoiceClassifier,
    SpeakerDiarization,
    AIVoiceDetector,
    AudioProctoringModel,
    AudioTrainer
)

__all__ = [
    "AudioProctor",
    "AudioPredictor",
    # New models
    "AudioConfig",
    "AudioFeatureExtractor",
    "AudioEncoder",
    "VoiceClassifier",
    "SpeakerDiarization",
    "AIVoiceDetector",
    "AudioProctoringModel",
    "AudioTrainer",
]
