"""
AudioProctor Model

Main audio proctoring model with VAD and speaker analysis.
"""

from pathlib import Path
from typing import Any

from proctor.engine.model import BaseModel
from proctor.cfg import AudioConfig, get_settings


class AudioProctor(BaseModel):
    """
    Audio proctoring model for voice analysis.
    
    Uses Silero VAD for voice activity detection and
    custom speaker analysis for multiple voice detection.
    
    Example:
        >>> from proctor import AudioProctor
        >>> model = AudioProctor()
        >>> results = model.predict(audio_samples)
        >>> print(results.multiple_speakers, results.whispering_detected)
    """
    
    def __init__(
        self,
        verbose: bool = True,
    ):
        """
        Initialize AudioProctor.
        
        Args:
            verbose: Enable verbose output
        """
        super().__init__(model=None, task="audio", verbose=verbose)
    
    @property
    def task_map(self) -> dict:
        """Map task to component classes."""
        from proctor.models.audio.predictor import AudioPredictor
        
        return {
            "predictor": AudioPredictor,
            "trainer": None,  # Audio model doesn't need training
            "validator": None,
        }
    
    @property
    def cfg(self) -> AudioConfig:
        """Get model configuration."""
        if self._cfg is None:
            settings = get_settings()
            self._cfg = settings.to_audio_config()
        return self._cfg
    
    def calibrate(self, audio_samples: Any, sample_rate: int = 16000):
        """
        Calibrate baseline noise level.
        
        Args:
            audio_samples: Audio samples for calibration
            sample_rate: Audio sample rate
        """
        if self.predictor:
            self.predictor.calibrate(audio_samples, sample_rate)
    
    def reset(self):
        """Reset predictor state."""
        if self.predictor:
            self.predictor.reset()
    
    def get_statistics(self) -> dict:
        """Get audio analysis statistics."""
        if self.predictor:
            return self.predictor.get_statistics()
        return {}
