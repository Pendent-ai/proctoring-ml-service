"""
Audio Proctoring Model

Voice activity detection and speaker analysis.
"""

from proctor.models.audio.model import AudioProctor
from proctor.models.audio.predictor import AudioPredictor

__all__ = [
    "AudioProctor",
    "AudioPredictor",
]
