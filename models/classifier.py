"""
Cheating Classifier Model
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from config import settings


@dataclass
class ClassifierResult:
    """Classification result."""
    cheating_probability: float = 0.0
    is_cheating: bool = False
    confidence: float = 0.0
    top_factors: list = None
    
    def __post_init__(self):
        if self.top_factors is None:
            self.top_factors = []


class CheatingClassifier:
    """XGBoost classifier for cheating detection."""
    
    # Feature names
    FEATURES = [
        "gaze_x_mean",
        "gaze_y_mean",
        "gaze_variance",
        "gaze_away_ratio",
        "head_yaw_mean",
        "head_yaw_variance",
        "head_pitch_mean",
        "head_movement_freq",
        "face_visible_ratio",
        "face_count_max",
        "phone_detected_ratio",
        "phone_duration",
        "multiple_faces_ratio",
        "eyes_closed_ratio",
        "looking_away_duration",
    ]
    
    def __init__(self, model_path: str | None = None):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to saved model. Uses default if None.
        """
        self.model = None
        self.threshold = settings.cheating_threshold
        
        path = model_path or settings.classifier_model_path
        
        if Path(path).exists():
            self.load(path)
        else:
            print("⚠️ No classifier model found. Using rule-based detection.")
    
    def load(self, path: str):
        """Load model from file."""
        if xgb is None:
            print("⚠️ XGBoost not installed. Using rule-based detection.")
            return
        
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        print(f"✅ Classifier loaded: {path}")
    
    def save(self, path: str):
        """Save model to file."""
        if self.model:
            self.model.save_model(path)
    
    def predict(self, features: dict) -> ClassifierResult:
        """
        Predict cheating probability from features.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            ClassifierResult with probability and factors.
        """
        result = ClassifierResult()
        
        # Use model if available
        if self.model is not None:
            X = self._prepare_features(features)
            proba = self.model.predict_proba(X)[0][1]
            result.cheating_probability = float(proba)
            result.is_cheating = proba >= self.threshold
            result.confidence = abs(proba - 0.5) * 2
            result.top_factors = self._get_top_factors(features)
        else:
            # Rule-based fallback
            result = self._rule_based_predict(features)
        
        return result
    
    def _prepare_features(self, features: dict) -> np.ndarray:
        """Prepare feature array for model."""
        X = np.zeros((1, len(self.FEATURES)))
        
        for i, name in enumerate(self.FEATURES):
            X[0, i] = features.get(name, 0.0)
        
        return X
    
    def _get_top_factors(self, features: dict) -> list:
        """Get top contributing factors."""
        factors = []
        
        if features.get("phone_detected_ratio", 0) > 0.1:
            factors.append("phone_use")
        if features.get("multiple_faces_ratio", 0) > 0.05:
            factors.append("multiple_faces")
        if features.get("gaze_away_ratio", 0) > 0.4:
            factors.append("looking_away")
        if features.get("face_visible_ratio", 1) < 0.8:
            factors.append("face_not_visible")
        
        return factors[:3]
    
    def _rule_based_predict(self, features: dict) -> ClassifierResult:
        """Rule-based prediction when no model is available."""
        result = ClassifierResult()
        
        score = 0.0
        factors = []
        
        # Phone detection (highest weight)
        phone_ratio = features.get("phone_detected_ratio", 0)
        if phone_ratio > 0:
            score += phone_ratio * 0.4
            factors.append("phone_use")
        
        # Multiple faces
        multi_face = features.get("multiple_faces_ratio", 0)
        if multi_face > 0.05:
            score += multi_face * 0.3
            factors.append("multiple_faces")
        
        # Looking away
        gaze_away = features.get("gaze_away_ratio", 0)
        if gaze_away > 0.3:
            score += gaze_away * 0.2
            factors.append("looking_away")
        
        # Face not visible
        face_visible = features.get("face_visible_ratio", 1)
        if face_visible < 0.8:
            score += (1 - face_visible) * 0.1
            factors.append("face_not_visible")
        
        result.cheating_probability = min(1.0, score)
        result.is_cheating = score >= self.threshold
        result.confidence = min(1.0, score / self.threshold) if score < self.threshold else 1.0
        result.top_factors = factors[:3]
        
        return result
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ):
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - 0 for normal, 1 for cheating
            **kwargs: Additional XGBoost parameters
        """
        if xgb is None:
            raise ImportError("XGBoost required for training")
        
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "use_label_encoder": False,
        }
        default_params.update(kwargs)
        
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(X, y)
        
        print("✅ Classifier trained")
        
        # Print feature importances
        importances = self.model.feature_importances_
        for name, imp in sorted(zip(self.FEATURES, importances), key=lambda x: -x[1])[:5]:
            print(f"  {name}: {imp:.3f}")
