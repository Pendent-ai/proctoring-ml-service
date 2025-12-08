"""
Ensemble classifier with stacking and calibration for cheating detection.

This module implements a robust ensemble approach combining:
- XGBoost: Gradient boosting with tree structure
- LightGBM: Efficient gradient boosting
- CatBoost: Gradient boosting with categorical features
- MLP: Neural network for non-linear patterns

Features:
- Stacking ensemble with meta-learner
- Isotonic calibration for probability estimates
- SMOTE for handling class imbalance
- Cross-validation based training
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from proctor.utils.logger import get_logger

logger = get_logger(__name__)


# Optional imports for enhanced ensemble
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available, using GradientBoosting fallback")

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available, using GradientBoosting fallback")

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not available, skipping in ensemble")

try:
    from imblearn.over_sampling import SMOTE

    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    logger.warning("imbalanced-learn not available, SMOTE disabled")


class EnsembleMethod(str, Enum):
    """Ensemble method options."""

    STACKING = "stacking"
    VOTING = "voting"
    BAGGING = "bagging"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble classifier."""

    # Ensemble settings
    method: EnsembleMethod = EnsembleMethod.STACKING
    use_calibration: bool = True
    calibration_method: str = "isotonic"  # 'isotonic' or 'sigmoid'
    use_smote: bool = True
    smote_ratio: float = 0.8

    # Cross-validation
    cv_folds: int = 5
    random_state: int = 42

    # XGBoost parameters
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0

    # LightGBM parameters
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 8
    lgb_learning_rate: float = 0.1
    lgb_num_leaves: int = 31
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8

    # CatBoost parameters
    cat_iterations: int = 200
    cat_depth: int = 6
    cat_learning_rate: float = 0.1

    # MLP parameters
    mlp_hidden_layers: tuple = (128, 64, 32)
    mlp_activation: str = "relu"
    mlp_learning_rate: float = 0.001
    mlp_max_iter: int = 500
    mlp_early_stopping: bool = True

    # Meta-learner (for stacking)
    meta_learner: str = "xgboost"  # 'xgboost', 'logistic', 'mlp'


@dataclass
class EnsembleMetrics:
    """Metrics from ensemble evaluation."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    cv_scores: list[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    classification_report: str = ""
    feature_importances: dict[str, float] = field(default_factory=dict)


class EnsembleClassifier:
    """
    Ensemble classifier for cheating detection.

    Combines multiple classifiers using stacking with a meta-learner.
    Includes calibration for reliable probability estimates.
    """

    def __init__(self, config: EnsembleConfig | None = None):
        """Initialize ensemble classifier.

        Args:
            config: Configuration for the ensemble
        """
        self.config = config or EnsembleConfig()
        self.scaler = StandardScaler()
        self.ensemble: StackingClassifier | None = None
        self.calibrated_model: CalibratedClassifierCV | None = None
        self.feature_names: list[str] = []
        self._is_fitted = False

    def _create_base_estimators(self) -> list[tuple[str, Any]]:
        """Create list of base estimators for ensemble."""
        estimators = []

        # XGBoost or fallback
        if HAS_XGBOOST:
            xgb = XGBClassifier(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                reg_alpha=self.config.xgb_reg_alpha,
                reg_lambda=self.config.xgb_reg_lambda,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=-1,
            )
            estimators.append(("xgboost", xgb))
        else:
            gb = GradientBoostingClassifier(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                random_state=self.config.random_state,
            )
            estimators.append(("gradient_boosting", gb))

        # LightGBM or fallback
        if HAS_LIGHTGBM:
            lgb = LGBMClassifier(
                n_estimators=self.config.lgb_n_estimators,
                max_depth=self.config.lgb_max_depth,
                learning_rate=self.config.lgb_learning_rate,
                num_leaves=self.config.lgb_num_leaves,
                subsample=self.config.lgb_subsample,
                colsample_bytree=self.config.lgb_colsample_bytree,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            estimators.append(("lightgbm", lgb))
        else:
            gb2 = GradientBoostingClassifier(
                n_estimators=self.config.lgb_n_estimators,
                max_depth=5,
                learning_rate=self.config.lgb_learning_rate,
                random_state=self.config.random_state + 1,
            )
            estimators.append(("gradient_boosting_2", gb2))

        # CatBoost (optional)
        if HAS_CATBOOST:
            cat = CatBoostClassifier(
                iterations=self.config.cat_iterations,
                depth=self.config.cat_depth,
                learning_rate=self.config.cat_learning_rate,
                random_state=self.config.random_state,
                verbose=False,
            )
            estimators.append(("catboost", cat))

        # MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=self.config.mlp_hidden_layers,
            activation=self.config.mlp_activation,
            learning_rate_init=self.config.mlp_learning_rate,
            max_iter=self.config.mlp_max_iter,
            early_stopping=self.config.mlp_early_stopping,
            random_state=self.config.random_state,
            validation_fraction=0.15,
        )
        estimators.append(("mlp", mlp))

        return estimators

    def _create_meta_learner(self) -> Any:
        """Create meta-learner for stacking."""
        if self.config.meta_learner == "xgboost" and HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        elif self.config.meta_learner == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=300,
                random_state=self.config.random_state,
            )
        else:
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.config.random_state
            )

    def _apply_smote(
        self, X: NDArray[np.float32], y: NDArray[np.int32]
    ) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Apply SMOTE to handle class imbalance."""
        if not self.config.use_smote or not HAS_SMOTE:
            return X, y

        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        minority_count = counts.min()
        majority_count = counts.max()

        if minority_count / majority_count < self.config.smote_ratio:
            try:
                smote = SMOTE(
                    sampling_strategy=self.config.smote_ratio,
                    random_state=self.config.random_state,
                    k_neighbors=min(5, minority_count - 1),
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)
                logger.info(
                    f"SMOTE applied: {len(y)} -> {len(y_resampled)} samples"
                )
                return X_resampled, y_resampled
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}, using original data")

        return X, y

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        feature_names: list[str] | None = None,
    ) -> "EnsembleClassifier":
        """Train the ensemble classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels of shape (n_samples,)
            feature_names: Optional list of feature names

        Returns:
            Self for method chaining
        """
        logger.info(f"Training ensemble on {len(y)} samples with {X.shape[1]} features")

        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Apply SMOTE for class imbalance
        X_balanced, y_balanced = self._apply_smote(X, y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_balanced)

        # Create ensemble
        base_estimators = self._create_base_estimators()
        meta_learner = self._create_meta_learner()

        self.ensemble = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=self.config.cv_folds,
            stack_method="predict_proba",
            passthrough=True,  # Include original features
            n_jobs=-1,
        )

        # Fit ensemble
        logger.info("Fitting stacking ensemble...")
        self.ensemble.fit(X_scaled, y_balanced)

        # Apply calibration if enabled
        if self.config.use_calibration:
            logger.info(
                f"Applying {self.config.calibration_method} calibration..."
            )
            self.calibrated_model = CalibratedClassifierCV(
                self.ensemble,
                method=self.config.calibration_method,
                cv="prefit",  # Already fitted
            )
            self.calibrated_model.fit(X_scaled, y_balanced)

        self._is_fitted = True
        logger.info("Ensemble training complete")

        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        """Predict class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        X_scaled = self.scaler.transform(X)

        if self.calibrated_model is not None:
            return self.calibrated_model.predict(X_scaled)
        return self.ensemble.predict(X_scaled)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted before prediction")

        X_scaled = self.scaler.transform(X)

        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_scaled)
        return self.ensemble.predict_proba(X_scaled)

    def evaluate(
        self, X: NDArray[np.float32], y: NDArray[np.int32]
    ) -> EnsembleMetrics:
        """Evaluate classifier performance.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Evaluation metrics
        """
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted before evaluation")

        X_scaled = self.scaler.transform(X)
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        # Compute metrics
        metrics = EnsembleMetrics(
            accuracy=float(accuracy_score(y, y_pred)),
            precision=float(precision_score(y, y_pred, zero_division=0)),
            recall=float(recall_score(y, y_pred, zero_division=0)),
            f1=float(f1_score(y, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y, y_proba))
            if len(np.unique(y)) > 1
            else 0.0,
            classification_report=classification_report(y, y_pred),
        )

        # Cross-validation scores
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        model = self.calibrated_model or self.ensemble
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1")
        metrics.cv_scores = cv_scores.tolist()
        metrics.cv_mean = float(np.mean(cv_scores))
        metrics.cv_std = float(np.std(cv_scores))

        # Feature importances (from XGBoost if available)
        metrics.feature_importances = self._get_feature_importances()

        return metrics

    def _get_feature_importances(self) -> dict[str, float]:
        """Extract feature importances from base estimators."""
        importances: dict[str, float] = {}

        if self.ensemble is None:
            return importances

        for name, estimator in self.ensemble.named_estimators_.items():
            if hasattr(estimator, "feature_importances_"):
                for i, imp in enumerate(estimator.feature_importances_):
                    feat_name = (
                        self.feature_names[i]
                        if i < len(self.feature_names)
                        else f"feature_{i}"
                    )
                    if feat_name not in importances:
                        importances[feat_name] = 0.0
                    importances[feat_name] += float(imp)

        # Normalize
        if importances:
            total = sum(importances.values())
            importances = {k: v / total for k, v in importances.items()}

        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str | Path) -> None:
        """Save classifier to disk.

        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "config": self.config,
            "scaler": self.scaler,
            "ensemble": self.ensemble,
            "calibrated_model": self.calibrated_model,
            "feature_names": self.feature_names,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(save_dict, path)
        logger.info(f"Saved ensemble classifier to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleClassifier":
        """Load classifier from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded classifier
        """
        save_dict = joblib.load(path)

        classifier = cls(config=save_dict["config"])
        classifier.scaler = save_dict["scaler"]
        classifier.ensemble = save_dict["ensemble"]
        classifier.calibrated_model = save_dict["calibrated_model"]
        classifier.feature_names = save_dict["feature_names"]
        classifier._is_fitted = save_dict["is_fitted"]

        logger.info(f"Loaded ensemble classifier from {path}")
        return classifier


class OnlineClassifier:
    """
    Lightweight classifier for real-time inference.

    Uses the trained ensemble but optimized for single-sample prediction.
    Includes confidence smoothing and adaptive thresholds.
    """

    def __init__(
        self,
        ensemble: EnsembleClassifier,
        confidence_threshold: float = 0.7,
        smoothing_window: int = 5,
    ):
        """Initialize online classifier.

        Args:
            ensemble: Trained ensemble classifier
            confidence_threshold: Threshold for positive prediction
            smoothing_window: Window size for confidence smoothing
        """
        self.ensemble = ensemble
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self._confidence_history: list[float] = []

    def predict_single(
        self, features: NDArray[np.float32]
    ) -> tuple[bool, float, float]:
        """Predict on a single sample with smoothing.

        Args:
            features: Feature vector of shape (n_features,)

        Returns:
            Tuple of (is_cheating, raw_confidence, smoothed_confidence)
        """
        # Reshape for prediction
        X = features.reshape(1, -1)

        # Get probability
        proba = self.ensemble.predict_proba(X)[0, 1]

        # Update history
        self._confidence_history.append(float(proba))
        if len(self._confidence_history) > self.smoothing_window:
            self._confidence_history.pop(0)

        # Compute smoothed confidence
        smoothed = np.mean(self._confidence_history)

        # Make prediction
        is_cheating = smoothed >= self.confidence_threshold

        return is_cheating, float(proba), float(smoothed)

    def reset(self) -> None:
        """Reset confidence history."""
        self._confidence_history.clear()

    def update_threshold(self, new_threshold: float) -> None:
        """Update confidence threshold dynamically.

        Args:
            new_threshold: New threshold value
        """
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
