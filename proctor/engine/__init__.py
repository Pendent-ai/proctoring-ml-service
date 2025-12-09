from __future__ import annotations
"""
Proctor Engine - Base classes and advanced models for proctoring.

Following Ultralytics pattern with:
- BaseModel: Core model class
- BasePredictor: Inference logic
- BaseTrainer: Training logic
- BaseValidator: Validation logic

Advanced components:
- FeatureExtractor: Enhanced 35+ feature extraction
- EnsembleClassifier: Stacking ensemble with calibration
- TemporalCheatingDetector: LSTM/Transformer for temporal patterns
- MultimodalDetector: Audio-visual fusion
"""

from proctor.engine.model import BaseModel
from proctor.engine.predictor import BasePredictor
from proctor.engine.trainer import BaseTrainer
from proctor.engine.validator import BaseValidator
from proctor.engine.results import Results, DetectionResults, AnalysisResults
from proctor.engine.features import EnhancedFeatureExtractor as FeatureExtractor
from proctor.engine.classifier import (
    EnsembleClassifier,
    EnsembleConfig,
    EnsembleMetrics,
    OnlineClassifier,
)
from proctor.engine.temporal import (
    TemporalCheatingDetector,
    TemporalConfig,
    TemporalLSTM,
    TemporalTransformer,
    SlidingWindowBuffer,
)
from proctor.engine.fusion import (
    MultimodalDetector,
    MultimodalFusion,
    FusionConfig,
    FusionStrategy,
    MultimodalBuffer,
)

__all__ = [
    # Base classes
    "BaseModel",
    "BasePredictor",
    "BaseTrainer",
    "BaseValidator",
    "Results",
    "DetectionResults",
    "AnalysisResults",
    # Feature extraction
    "FeatureExtractor",
    # Ensemble classifier
    "EnsembleClassifier",
    "EnsembleConfig",
    "EnsembleMetrics",
    "OnlineClassifier",
    # Temporal models
    "TemporalCheatingDetector",
    "TemporalConfig",
    "TemporalLSTM",
    "TemporalTransformer",
    "SlidingWindowBuffer",
    # Multimodal fusion
    "MultimodalDetector",
    "MultimodalFusion",
    "FusionConfig",
    "FusionStrategy",
    "MultimodalBuffer",
]
