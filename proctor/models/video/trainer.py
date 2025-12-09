from __future__ import annotations
"""
Classifier Trainer

Trains the cheating classifier on labeled data.
"""

from pathlib import Path
from typing import Any, Optional
import numpy as np
import json

from proctor.engine.trainer import BaseTrainer
from proctor.cfg import VideoConfig


class ClassifierTrainer(BaseTrainer):
    """
    Trainer for the cheating classifier.
    
    Uses XGBoost for binary classification.
    
    Example:
        >>> trainer = ClassifierTrainer(cfg)
        >>> results = trainer(data_path, epochs=100)
    """
    
    # Feature names for the classifier
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
    
    def __init__(self, cfg: VideoConfig):
        """Initialize classifier trainer."""
        super().__init__(cfg)
        
        self.xgb = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
    
    def setup(self, data: str | Path):
        """
        Setup training data.
        
        Args:
            data: Path to training data (JSON or CSV)
        """
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost required for training: pip install xgboost")
        
        data_path = Path(data)
        
        if data_path.suffix == ".json":
            self._load_json_data(data_path)
        elif data_path.suffix == ".csv":
            self._load_csv_data(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
        
        # Initialize model
        self.model = self.xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
        )
        
        print(f"✅ Training data loaded: {len(self.X_train)} samples")
    
    def _load_json_data(self, path: Path):
        """Load training data from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        X = []
        y = []
        
        for sample in data:
            features = [sample.get(f, 0.0) for f in self.FEATURES]
            X.append(features)
            y.append(sample.get("is_cheating", 0))
        
        X = np.array(X)
        y = np.array(y)
        
        # Split train/val
        split_idx = int(len(X) * 0.8)
        self.X_train, self.X_val = X[:split_idx], X[split_idx:]
        self.y_train, self.y_val = y[:split_idx], y[split_idx:]
    
    def _load_csv_data(self, path: Path):
        """Load training data from CSV."""
        import csv
        
        X = []
        y = []
        
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                features = [float(row.get(f, 0)) for f in self.FEATURES]
                X.append(features)
                y.append(int(row.get("is_cheating", 0)))
        
        X = np.array(X)
        y = np.array(y)
        
        split_idx = int(len(X) * 0.8)
        self.X_train, self.X_val = X[:split_idx], X[split_idx:]
        self.y_train, self.y_val = y[:split_idx], y[split_idx:]
    
    def train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch.
        
        Note: XGBoost trains all at once, so this is called once.
        """
        if epoch > 0:
            return {"loss": 0.0}
        
        # Train model
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False,
        )
        
        # Get training predictions
        train_preds = self.model.predict_proba(self.X_train)[:, 1]
        train_acc = np.mean((train_preds >= 0.5) == self.y_train)
        
        return {"accuracy": train_acc, "loss": 0.0}
    
    def validate(self) -> dict:
        """Validate current model."""
        if self.X_val is None:
            return {}
        
        preds = self.model.predict_proba(self.X_val)[:, 1]
        accuracy = np.mean((preds >= 0.5) == self.y_val)
        
        # Calculate AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(self.y_val, preds)
        except ImportError:
            auc = 0.0
        
        return {
            "accuracy": accuracy,
            "auc": auc,
        }
    
    def _save_checkpoint(self, checkpoint: dict, path: Path):
        """Save model checkpoint."""
        self.model.save_model(str(path))
        print(f"💾 Model saved: {path}")
    
    def _get_fitness(self, metrics: dict) -> float:
        """Calculate fitness from metrics."""
        return metrics.get("auc", metrics.get("accuracy", 0.0))
    
    def get_feature_importance(self) -> dict:
        """Get feature importances."""
        if self.model is None:
            return {}
        
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.FEATURES, importances),
            key=lambda x: -x[1],
        ))
