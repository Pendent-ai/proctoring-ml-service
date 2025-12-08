"""
Proctor Configuration Module

Pydantic-based configuration following best practices.
"""

from proctor.cfg.config import (
    BaseConfig,
    VideoConfig,
    AudioConfig,
    LiveKitConfig,
    Settings,
    get_settings,
    # Training configs
    EnsembleTrainingConfig,
    TemporalTrainingConfig,
    MultimodalTrainingConfig,
    YOLOTrainingConfig,
)

__all__ = [
    # Base configs
    "BaseConfig",
    "VideoConfig",
    "AudioConfig",
    "LiveKitConfig",
    "Settings",
    "get_settings",
    # Training configs
    "EnsembleTrainingConfig",
    "TemporalTrainingConfig",
    "MultimodalTrainingConfig",
    "YOLOTrainingConfig",
]
