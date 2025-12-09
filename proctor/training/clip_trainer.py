from __future__ import annotations
"""
Video clip trainer for temporal cheating detection.

This module provides a complete training pipeline for models that
learn from labeled video clips to detect cheating patterns.

Features:
- End-to-end training from video clips
- Feature extraction during training
- Support for temporal and multimodal models
- Experiment tracking and checkpointing
- Cross-validation support
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from proctor.data.dataset import (
    AnnotationLoader,
    ClipAnnotation,
    FeatureNormalizer,
    VideoReader,
)
from proctor.engine.classifier import EnsembleClassifier, EnsembleConfig
from proctor.engine.features import FeatureExtractor
from proctor.engine.temporal import TemporalCheatingDetector, TemporalConfig
from proctor.engine.fusion import MultimodalDetector, FusionConfig
from proctor.utils.logger import get_logger

logger = get_logger(__name__)


# Check for optional imports
try:
    import torch
    from torch.utils.data import DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.model_selection import train_test_split, StratifiedKFold

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class TrainingConfig:
    """Configuration for video clip training."""

    # Data settings
    annotations_path: str = ""
    video_dir: str = ""
    audio_dir: str = ""
    output_dir: str = "runs/train"

    # Feature extraction
    clip_length: int = 30  # frames per clip
    feature_dim: int = 35  # number of features
    extract_audio: bool = True

    # Model selection
    model_type: str = "ensemble"  # 'ensemble', 'temporal', 'multimodal'

    # Training settings
    test_split: float = 0.2
    val_split: float = 0.1
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    patience: int = 10
    random_state: int = 42

    # Cross-validation
    use_cv: bool = False
    cv_folds: int = 5

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_interval: int = 10

    # Experiment tracking
    experiment_name: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass 
class TrainingResult:
    """Results from training run."""

    model_path: str
    metrics: dict[str, float]
    history: dict[str, list[float]]
    config: TrainingConfig
    duration_seconds: float
    timestamp: str
    cv_results: list[dict[str, float]] | None = None


class VideoClipTrainer:
    """
    Trainer for video clip-based cheating detection.

    Handles the complete pipeline from video clips to trained model.
    """

    def __init__(self, config: TrainingConfig | None = None):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.feature_extractor: FeatureExtractor | None = None
        self.normalizer: FeatureNormalizer | None = None
        self.model: Any = None
        self._video_predictor = None

    def _init_feature_extractor(self) -> None:
        """Initialize the feature extractor (lazy loading)."""
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()

    def _init_video_predictor(self) -> None:
        """Initialize video predictor for frame processing."""
        if self._video_predictor is None:
            from proctor.models.video import VideoPredictor

            self._video_predictor = VideoPredictor()

    def extract_features_from_clip(
        self,
        video_path: str | Path,
        start_frame: int,
        end_frame: int,
    ) -> NDArray[np.float32]:
        """
        Extract features from a video clip.

        Args:
            video_path: Path to video file
            start_frame: Start frame index
            end_frame: End frame index

        Returns:
            Feature matrix of shape (n_frames, n_features)
        """
        self._init_feature_extractor()
        self._init_video_predictor()

        with VideoReader(video_path) as reader:
            frames = reader.read_clip(start_frame, end_frame)

        features_list = []
        for frame in frames:
            # Get predictions from video predictor
            result = self._video_predictor.predict(frame)

            # Extract features from predictions
            frame_features = self.feature_extractor.extract_from_predictions(
                result.detections,
                result.gaze,
                result.head_pose,
            )
            features_list.append(frame_features)

        return np.array(features_list, dtype=np.float32)

    def prepare_dataset(
        self,
        annotations: list[ClipAnnotation],
    ) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        """
        Prepare dataset from annotations.

        Args:
            annotations: List of clip annotations

        Returns:
            Tuple of (features, labels)
        """
        logger.info(f"Preparing dataset from {len(annotations)} clips...")

        all_features = []
        all_labels = []

        for i, ann in enumerate(annotations):
            try:
                # Extract features from clip
                clip_features = self.extract_features_from_clip(
                    ann.video_path,
                    ann.start_frame,
                    ann.end_frame,
                )

                # Pad or truncate to clip_length
                if len(clip_features) < self.config.clip_length:
                    # Pad with last frame
                    padding = np.repeat(
                        clip_features[-1:],
                        self.config.clip_length - len(clip_features),
                        axis=0,
                    )
                    clip_features = np.concatenate([clip_features, padding])
                else:
                    clip_features = clip_features[: self.config.clip_length]

                all_features.append(clip_features)
                all_labels.append(ann.label)

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(annotations)} clips")

            except Exception as e:
                logger.warning(f"Error processing clip {ann.video_path}: {e}")
                continue

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)

        logger.info(f"Dataset prepared: {X.shape[0]} samples, shape {X.shape}")
        return X, y

    def prepare_frame_dataset(
        self,
        annotations: list[ClipAnnotation],
    ) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        """
        Prepare frame-level dataset for ensemble classifier.

        Flattens temporal dimension for traditional ML classifiers.

        Args:
            annotations: List of clip annotations

        Returns:
            Tuple of (features, labels) with aggregated clip features
        """
        X_sequences, y_sequences = self.prepare_dataset(annotations)

        # Aggregate features over time (mean and std)
        n_samples = X_sequences.shape[0]
        n_features = X_sequences.shape[2]

        # Create aggregated features
        X_agg = np.zeros((n_samples, n_features * 3), dtype=np.float32)

        for i in range(n_samples):
            seq = X_sequences[i]
            X_agg[i, :n_features] = np.mean(seq, axis=0)  # Mean
            X_agg[i, n_features : 2 * n_features] = np.std(seq, axis=0)  # Std
            X_agg[i, 2 * n_features :] = np.max(seq, axis=0) - np.min(seq, axis=0)  # Range

        return X_agg, y_sequences

    def train_ensemble(
        self,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
    ) -> tuple[EnsembleClassifier, dict[str, Any]]:
        """
        Train ensemble classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Trained classifier and metrics
        """
        logger.info("Training ensemble classifier...")

        config = EnsembleConfig(
            use_calibration=True,
            use_smote=True,
            cv_folds=self.config.cv_folds if self.config.use_cv else 5,
        )

        classifier = EnsembleClassifier(config)
        classifier.fit(X_train, y_train)

        # Evaluate
        metrics = {}
        if X_val is not None and y_val is not None:
            eval_result = classifier.evaluate(X_val, y_val)
            metrics = {
                "accuracy": eval_result.accuracy,
                "precision": eval_result.precision,
                "recall": eval_result.recall,
                "f1": eval_result.f1,
                "roc_auc": eval_result.roc_auc,
            }

        return classifier, metrics

    def train_temporal(
        self,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
    ) -> tuple[TemporalCheatingDetector, dict[str, Any]]:
        """
        Train temporal LSTM model.

        Args:
            X_train: Training sequences (n_samples, seq_len, n_features)
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels

        Returns:
            Trained detector and training history
        """
        logger.info("Training temporal model...")

        config = TemporalConfig(
            input_dim=X_train.shape[2],
            sequence_length=X_train.shape[1],
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            patience=self.config.patience,
        )

        detector = TemporalCheatingDetector(config)
        history = detector.fit(X_train, y_train, X_val, y_val)

        # Calculate final metrics
        metrics = {}
        if X_val is not None and y_val is not None:
            y_pred = detector.predict(X_val)
            y_proba = detector.predict_proba(X_val)[:, 1]

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
            )

            metrics = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "recall": float(recall_score(y_val, y_pred, zero_division=0)),
                "f1": float(f1_score(y_val, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_val, y_proba))
                if len(np.unique(y_val)) > 1
                else 0.0,
            }

        return detector, {"history": history, "metrics": metrics}

    def train(
        self,
        annotations: list[ClipAnnotation] | None = None,
    ) -> TrainingResult:
        """
        Run complete training pipeline.

        Args:
            annotations: Optional pre-loaded annotations

        Returns:
            Training results
        """
        start_time = datetime.now()

        # Load annotations if not provided
        if annotations is None:
            if self.config.annotations_path.endswith(".json"):
                annotations = AnnotationLoader.load_json(self.config.annotations_path)
            else:
                annotations = AnnotationLoader.load_csv(self.config.annotations_path)

        # Prepare output directory
        output_dir = Path(self.config.output_dir)
        if self.config.experiment_name:
            output_dir = output_dir / self.config.experiment_name
        else:
            output_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare dataset based on model type
        if self.config.model_type == "ensemble":
            X, y = self.prepare_frame_dataset(annotations)
        else:
            X, y = self.prepare_dataset(annotations)

        # Normalize features
        self.normalizer = FeatureNormalizer(method="standard")
        if self.config.model_type == "ensemble":
            X = self.normalizer.fit_transform(X)
        else:
            # Normalize each feature across all samples and timesteps
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_flat = self.normalizer.fit_transform(X_flat)
            X = X_flat.reshape(original_shape)

        # Save normalizer
        self.normalizer.save(output_dir / "normalizer.json")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_split,
            random_state=self.config.random_state,
            stratify=y,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.val_split / (1 - self.config.test_split),
            random_state=self.config.random_state,
            stratify=y_train,
        )

        logger.info(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        # Train model
        history = {}
        if self.config.model_type == "ensemble":
            self.model, metrics = self.train_ensemble(X_train, y_train, X_val, y_val)
            model_path = output_dir / "ensemble_model.joblib"
            self.model.save(model_path)

        elif self.config.model_type == "temporal":
            self.model, results = self.train_temporal(X_train, y_train, X_val, y_val)
            history = results.get("history", {})
            metrics = results.get("metrics", {})
            model_path = output_dir / "temporal_model.pt"
            self.model.save(model_path)

        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        # Evaluate on test set
        if self.config.model_type == "ensemble":
            test_metrics = self.model.evaluate(X_test, y_test)
            metrics["test_accuracy"] = test_metrics.accuracy
            metrics["test_f1"] = test_metrics.f1
        else:
            y_pred = self.model.predict(X_test)
            from sklearn.metrics import accuracy_score, f1_score

            metrics["test_accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["test_f1"] = float(f1_score(y_test, y_pred, zero_division=0))

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Save training info
        result = TrainingResult(
            model_path=str(model_path),
            metrics=metrics,
            history=history,
            config=self.config,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat(),
        )

        # Save result summary
        self._save_result(result, output_dir)

        logger.info(f"Training complete in {duration:.1f}s")
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Test metrics: {metrics}")

        return result

    def _save_result(self, result: TrainingResult, output_dir: Path) -> None:
        """Save training result to file."""
        import json

        summary = {
            "model_path": result.model_path,
            "metrics": result.metrics,
            "duration_seconds": result.duration_seconds,
            "timestamp": result.timestamp,
            "config": {
                "model_type": result.config.model_type,
                "epochs": result.config.epochs,
                "batch_size": result.config.batch_size,
                "learning_rate": result.config.learning_rate,
            },
        }

        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def cross_validate(
        self,
        annotations: list[ClipAnnotation] | None = None,
    ) -> dict[str, Any]:
        """
        Run k-fold cross-validation.

        Args:
            annotations: Optional pre-loaded annotations

        Returns:
            Cross-validation results
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for cross-validation")

        # Load annotations if not provided
        if annotations is None:
            if self.config.annotations_path.endswith(".json"):
                annotations = AnnotationLoader.load_json(self.config.annotations_path)
            else:
                annotations = AnnotationLoader.load_csv(self.config.annotations_path)

        # Prepare dataset
        if self.config.model_type == "ensemble":
            X, y = self.prepare_frame_dataset(annotations)
        else:
            X, y = self.prepare_dataset(annotations)

        # Cross-validation
        kfold = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{self.config.cv_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Normalize
            normalizer = FeatureNormalizer(method="standard")
            if self.config.model_type == "ensemble":
                X_train = normalizer.fit_transform(X_train)
                X_val = normalizer.transform(X_val)
            else:
                orig_shape_train = X_train.shape
                orig_shape_val = X_val.shape
                X_train = normalizer.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
                X_train = X_train.reshape(orig_shape_train)
                X_val = normalizer.transform(X_val.reshape(-1, X_val.shape[-1]))
                X_val = X_val.reshape(orig_shape_val)

            # Train
            if self.config.model_type == "ensemble":
                model, metrics = self.train_ensemble(X_train, y_train, X_val, y_val)
            else:
                model, results = self.train_temporal(X_train, y_train, X_val, y_val)
                metrics = results.get("metrics", {})

            fold_results.append(metrics)

        # Aggregate results
        aggregated = {}
        for key in fold_results[0].keys():
            values = [r[key] for r in fold_results]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

        logger.info(f"Cross-validation results: {aggregated}")

        return {
            "fold_results": fold_results,
            "aggregated": aggregated,
        }


def train_from_config(config_path: str | Path) -> TrainingResult:
    """
    Train model from configuration file.

    Args:
        config_path: Path to YAML/JSON config file

    Returns:
        Training results
    """
    import json
    import yaml

    config_path = Path(config_path)

    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    else:
        with open(config_path) as f:
            config_dict = json.load(f)

    config = TrainingConfig(**config_dict)
    trainer = VideoClipTrainer(config)

    return trainer.train()
