"""
Dataset classes for proctoring model training.

This module provides data loading utilities for:
- Frame-level feature datasets
- Video clip datasets with temporal labels
- Multimodal synchronized datasets
- Data augmentation pipelines
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from numpy.typing import NDArray

from proctor.utils.logger import get_logger

logger = get_logger(__name__)


# Optional imports
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LabelFormat(str, Enum):
    """Annotation label format."""

    BINARY = "binary"  # 0/1 for cheating/not cheating
    MULTICLASS = "multiclass"  # Multiple cheating types
    MULTILABEL = "multilabel"  # Multiple simultaneous behaviors
    TEMPORAL = "temporal"  # Frame-by-frame labels


@dataclass
class ClipAnnotation:
    """Annotation for a video clip."""

    video_path: str
    start_frame: int
    end_frame: int
    label: int  # 0 = no cheating, 1 = cheating
    label_name: str = ""
    cheating_type: str = ""  # e.g., "phone_use", "looking_away", "multiple_faces"
    confidence: float = 1.0  # Annotation confidence
    annotator: str = ""
    frame_labels: list[int] | None = None  # Per-frame labels if available


@dataclass
class FrameAnnotation:
    """Annotation for a single frame."""

    frame_index: int
    label: int
    features: NDArray[np.float32] | None = None
    detections: list[dict[str, Any]] | None = None


class AnnotationLoader:
    """Load annotations from various formats."""

    @staticmethod
    def load_csv(
        csv_path: str | Path,
        video_col: str = "video_path",
        start_col: str = "start_frame",
        end_col: str = "end_frame",
        label_col: str = "label",
    ) -> list[ClipAnnotation]:
        """
        Load annotations from CSV file.

        Args:
            csv_path: Path to CSV file
            video_col: Column name for video path
            start_col: Column name for start frame
            end_col: Column name for end frame
            label_col: Column name for label

        Returns:
            List of ClipAnnotation objects
        """
        import csv

        annotations = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ann = ClipAnnotation(
                    video_path=row[video_col],
                    start_frame=int(row[start_col]),
                    end_frame=int(row[end_col]),
                    label=int(row[label_col]),
                    label_name=row.get("label_name", ""),
                    cheating_type=row.get("cheating_type", ""),
                    confidence=float(row.get("confidence", 1.0)),
                    annotator=row.get("annotator", ""),
                )
                annotations.append(ann)

        logger.info(f"Loaded {len(annotations)} annotations from {csv_path}")
        return annotations

    @staticmethod
    def load_json(json_path: str | Path) -> list[ClipAnnotation]:
        """Load annotations from JSON file."""
        import json

        with open(json_path) as f:
            data = json.load(f)

        annotations = []
        for item in data.get("annotations", data):
            ann = ClipAnnotation(
                video_path=item["video_path"],
                start_frame=item["start_frame"],
                end_frame=item["end_frame"],
                label=item["label"],
                label_name=item.get("label_name", ""),
                cheating_type=item.get("cheating_type", ""),
                confidence=item.get("confidence", 1.0),
                annotator=item.get("annotator", ""),
                frame_labels=item.get("frame_labels"),
            )
            annotations.append(ann)

        logger.info(f"Loaded {len(annotations)} annotations from {json_path}")
        return annotations

    @staticmethod
    def save_json(
        annotations: list[ClipAnnotation],
        output_path: str | Path,
    ) -> None:
        """Save annotations to JSON file."""
        import json

        data = {
            "annotations": [
                {
                    "video_path": ann.video_path,
                    "start_frame": ann.start_frame,
                    "end_frame": ann.end_frame,
                    "label": ann.label,
                    "label_name": ann.label_name,
                    "cheating_type": ann.cheating_type,
                    "confidence": ann.confidence,
                    "annotator": ann.annotator,
                    "frame_labels": ann.frame_labels,
                }
                for ann in annotations
            ]
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(annotations)} annotations to {output_path}")


class VideoReader:
    """Read video frames efficiently."""

    def __init__(self, video_path: str | Path):
        """Initialize video reader.

        Args:
            video_path: Path to video file
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required. Install with: pip install opencv-python")

        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self, frame_idx: int) -> NDArray[np.uint8] | None:
        """Read a specific frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def read_clip(
        self,
        start_frame: int,
        end_frame: int,
        step: int = 1,
    ) -> list[NDArray[np.uint8]]:
        """Read a clip of frames."""
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(start_frame, end_frame, step):
            ret, frame = self.cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break

        return frames

    def __iter__(self) -> Iterator[NDArray[np.uint8]]:
        """Iterate over all frames."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        """Release video capture."""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FrameDataset:
    """Dataset for frame-level features."""

    def __init__(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int32],
        transform: Callable[[NDArray], NDArray] | None = None,
    ):
        """Initialize dataset.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            transform: Optional transform function
        """
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[NDArray[np.float32], int]:
        features = self.features[idx]
        label = self.labels[idx]

        if self.transform is not None:
            features = self.transform(features)

        return features, int(label)

    def get_class_weights(self) -> NDArray[np.float32]:
        """Calculate class weights for imbalanced data."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        weights = total / (len(unique) * counts)
        return weights.astype(np.float32)


if HAS_TORCH:

    class VideoClipDataset(Dataset):
        """PyTorch dataset for video clips."""

        def __init__(
            self,
            annotations: list[ClipAnnotation],
            feature_extractor: Callable[[NDArray], NDArray] | None = None,
            clip_length: int = 30,
            transform: Callable[[NDArray], torch.Tensor] | None = None,
            cache_features: bool = False,
        ):
            """Initialize dataset.

            Args:
                annotations: List of clip annotations
                feature_extractor: Function to extract features from frames
                clip_length: Number of frames per clip
                transform: Optional transform for features
                cache_features: Whether to cache extracted features
            """
            self.annotations = annotations
            self.feature_extractor = feature_extractor
            self.clip_length = clip_length
            self.transform = transform
            self.cache_features = cache_features
            self._cache: dict[int, torch.Tensor] = {}

        def __len__(self) -> int:
            return len(self.annotations)

        def __getitem__(
            self, idx: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Get a clip and its label."""
            if self.cache_features and idx in self._cache:
                features = self._cache[idx]
                return features, torch.tensor(self.annotations[idx].label)

            ann = self.annotations[idx]

            # Load video frames
            with VideoReader(ann.video_path) as reader:
                frames = reader.read_clip(
                    ann.start_frame,
                    min(ann.end_frame, ann.start_frame + self.clip_length),
                )

            # Pad if necessary
            while len(frames) < self.clip_length:
                frames.append(frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8))

            frames = np.array(frames[: self.clip_length])

            # Extract features if extractor provided
            if self.feature_extractor is not None:
                features = np.array([self.feature_extractor(f) for f in frames])
            else:
                # Use raw frames (normalized)
                features = frames.astype(np.float32) / 255.0

            # Apply transform
            if self.transform is not None:
                features = self.transform(features)
            else:
                features = torch.FloatTensor(features)

            # Cache if enabled
            if self.cache_features:
                self._cache[idx] = features

            label = torch.tensor(ann.label, dtype=torch.long)
            return features, label

        def clear_cache(self) -> None:
            """Clear feature cache."""
            self._cache.clear()

    class MultimodalClipDataset(Dataset):
        """Dataset for synchronized video and audio clips."""

        def __init__(
            self,
            annotations: list[ClipAnnotation],
            video_feature_extractor: Callable[[NDArray], NDArray],
            audio_feature_extractor: Callable[[NDArray], NDArray],
            audio_dir: str | Path,
            clip_length: int = 30,
            audio_sample_rate: int = 16000,
        ):
            """Initialize dataset.

            Args:
                annotations: List of clip annotations
                video_feature_extractor: Extract features from video frame
                audio_feature_extractor: Extract features from audio segment
                audio_dir: Directory containing audio files
                clip_length: Number of frames per clip
                audio_sample_rate: Audio sample rate
            """
            self.annotations = annotations
            self.video_extractor = video_feature_extractor
            self.audio_extractor = audio_feature_extractor
            self.audio_dir = Path(audio_dir)
            self.clip_length = clip_length
            self.audio_sample_rate = audio_sample_rate

        def __len__(self) -> int:
            return len(self.annotations)

        def __getitem__(
            self, idx: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Get video features, audio features, and label."""
            ann = self.annotations[idx]

            # Load video
            with VideoReader(ann.video_path) as reader:
                frames = reader.read_clip(
                    ann.start_frame,
                    min(ann.end_frame, ann.start_frame + self.clip_length),
                )
                fps = reader.fps

            # Pad frames if needed
            while len(frames) < self.clip_length:
                frames.append(frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8))
            frames = frames[: self.clip_length]

            # Extract video features
            video_features = np.array([self.video_extractor(f) for f in frames])

            # Load corresponding audio
            video_name = Path(ann.video_path).stem
            audio_path = self.audio_dir / f"{video_name}.wav"

            if audio_path.exists():
                audio_features = self._load_audio_features(
                    audio_path, ann.start_frame, ann.end_frame, fps
                )
            else:
                # Create zero audio features
                audio_features = np.zeros((self.clip_length, 5), dtype=np.float32)

            # Convert to tensors
            video_tensor = torch.FloatTensor(video_features)
            audio_tensor = torch.FloatTensor(audio_features)
            label = torch.tensor(ann.label, dtype=torch.long)

            return video_tensor, audio_tensor, label

        def _load_audio_features(
            self,
            audio_path: Path,
            start_frame: int,
            end_frame: int,
            fps: float,
        ) -> NDArray[np.float32]:
            """Load and extract audio features for time range."""
            try:
                import torchaudio

                waveform, sr = torchaudio.load(str(audio_path))

                # Calculate time range
                start_time = start_frame / fps
                end_time = end_frame / fps

                # Convert to samples
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)

                # Extract segment
                audio_segment = waveform[:, start_sample:end_sample]

                # Extract features for each frame
                samples_per_frame = int(sr / fps)
                features = []

                for i in range(self.clip_length):
                    frame_start = i * samples_per_frame
                    frame_end = (i + 1) * samples_per_frame

                    if frame_end <= audio_segment.shape[1]:
                        frame_audio = audio_segment[:, frame_start:frame_end]
                        feat = self.audio_extractor(frame_audio.numpy())
                        features.append(feat)
                    else:
                        features.append(np.zeros(5, dtype=np.float32))

                return np.array(features, dtype=np.float32)

            except Exception as e:
                logger.warning(f"Error loading audio {audio_path}: {e}")
                return np.zeros((self.clip_length, 5), dtype=np.float32)

    class BalancedSampler(Sampler):
        """Sampler for class-balanced batches."""

        def __init__(
            self,
            labels: list[int] | NDArray[np.int32],
            batch_size: int,
            oversample_minority: bool = True,
        ):
            """Initialize sampler.

            Args:
                labels: List of labels
                batch_size: Batch size
                oversample_minority: Whether to oversample minority class
            """
            self.labels = np.array(labels)
            self.batch_size = batch_size
            self.oversample_minority = oversample_minority

            # Get class indices
            self.class_indices: dict[int, list[int]] = {}
            for idx, label in enumerate(self.labels):
                if label not in self.class_indices:
                    self.class_indices[label] = []
                self.class_indices[label].append(idx)

            # Calculate samples per class
            if oversample_minority:
                max_class_count = max(len(v) for v in self.class_indices.values())
                self.samples_per_class = {
                    k: max_class_count for k in self.class_indices
                }
            else:
                self.samples_per_class = {
                    k: len(v) for k, v in self.class_indices.items()
                }

            self.total_samples = sum(self.samples_per_class.values())

        def __iter__(self) -> Iterator[int]:
            """Generate balanced indices."""
            indices = []

            for class_id, class_idx in self.class_indices.items():
                n_samples = self.samples_per_class[class_id]
                if n_samples > len(class_idx):
                    # Oversample
                    sampled = np.random.choice(class_idx, n_samples, replace=True)
                else:
                    sampled = np.random.choice(class_idx, n_samples, replace=False)
                indices.extend(sampled.tolist())

            # Shuffle
            np.random.shuffle(indices)
            return iter(indices)

        def __len__(self) -> int:
            return self.total_samples


class FeatureNormalizer:
    """Normalize features for training."""

    def __init__(self, method: str = "standard"):
        """Initialize normalizer.

        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.mean: NDArray[np.float32] | None = None
        self.std: NDArray[np.float32] | None = None
        self.min: NDArray[np.float32] | None = None
        self.max: NDArray[np.float32] | None = None
        self.median: NDArray[np.float32] | None = None
        self.iqr: NDArray[np.float32] | None = None

    def fit(self, features: NDArray[np.float32]) -> "FeatureNormalizer":
        """Fit normalizer to features."""
        if self.method == "standard":
            self.mean = np.mean(features, axis=0)
            self.std = np.std(features, axis=0) + 1e-8
        elif self.method == "minmax":
            self.min = np.min(features, axis=0)
            self.max = np.max(features, axis=0)
        elif self.method == "robust":
            self.median = np.median(features, axis=0)
            q75 = np.percentile(features, 75, axis=0)
            q25 = np.percentile(features, 25, axis=0)
            self.iqr = (q75 - q25) + 1e-8

        return self

    def transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Transform features."""
        if self.method == "standard":
            return (features - self.mean) / self.std
        elif self.method == "minmax":
            return (features - self.min) / (self.max - self.min + 1e-8)
        elif self.method == "robust":
            return (features - self.median) / self.iqr
        return features

    def fit_transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fit and transform."""
        return self.fit(features).transform(features)

    def save(self, path: str | Path) -> None:
        """Save normalizer parameters."""
        import json

        params = {
            "method": self.method,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "min": self.min.tolist() if self.min is not None else None,
            "max": self.max.tolist() if self.max is not None else None,
            "median": self.median.tolist() if self.median is not None else None,
            "iqr": self.iqr.tolist() if self.iqr is not None else None,
        }

        with open(path, "w") as f:
            json.dump(params, f)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        """Load normalizer from file."""
        import json

        with open(path) as f:
            params = json.load(f)

        normalizer = cls(method=params["method"])
        normalizer.mean = np.array(params["mean"]) if params["mean"] else None
        normalizer.std = np.array(params["std"]) if params["std"] else None
        normalizer.min = np.array(params["min"]) if params["min"] else None
        normalizer.max = np.array(params["max"]) if params["max"] else None
        normalizer.median = np.array(params["median"]) if params["median"] else None
        normalizer.iqr = np.array(params["iqr"]) if params["iqr"] else None

        return normalizer


class DataAugmentation:
    """Data augmentation for proctoring features."""

    @staticmethod
    def add_noise(
        features: NDArray[np.float32],
        noise_level: float = 0.05,
    ) -> NDArray[np.float32]:
        """Add Gaussian noise to features."""
        noise = np.random.randn(*features.shape) * noise_level
        return features + noise.astype(np.float32)

    @staticmethod
    def temporal_jitter(
        sequence: NDArray[np.float32],
        max_shift: int = 2,
    ) -> NDArray[np.float32]:
        """Apply temporal jittering to sequence."""
        seq_len = sequence.shape[0]
        shifts = np.random.randint(-max_shift, max_shift + 1, size=seq_len)

        jittered = np.zeros_like(sequence)
        for i in range(seq_len):
            new_idx = max(0, min(seq_len - 1, i + shifts[i]))
            jittered[i] = sequence[new_idx]

        return jittered

    @staticmethod
    def dropout_features(
        features: NDArray[np.float32],
        dropout_rate: float = 0.1,
    ) -> NDArray[np.float32]:
        """Randomly zero out features."""
        mask = np.random.random(features.shape) > dropout_rate
        return features * mask.astype(np.float32)

    @staticmethod
    def scale_features(
        features: NDArray[np.float32],
        scale_range: tuple[float, float] = (0.9, 1.1),
    ) -> NDArray[np.float32]:
        """Randomly scale features."""
        scale = np.random.uniform(scale_range[0], scale_range[1], features.shape[-1])
        return features * scale.astype(np.float32)

    @staticmethod
    def mixup(
        features1: NDArray[np.float32],
        features2: NDArray[np.float32],
        label1: int,
        label2: int,
        alpha: float = 0.2,
    ) -> tuple[NDArray[np.float32], float]:
        """Apply mixup augmentation."""
        lam = np.random.beta(alpha, alpha)
        mixed_features = lam * features1 + (1 - lam) * features2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_features, mixed_label
