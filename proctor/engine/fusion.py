from __future__ import annotations
"""
Multimodal fusion for audio-visual cheating detection.

This module combines video and audio features using attention-based
fusion mechanisms to improve detection accuracy.

Features:
- Cross-modal attention for feature alignment
- Late fusion with learned weights
- Joint embedding space for multimodal features
- Synchronized temporal processing
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from proctor.utils.logger import get_logger

logger = get_logger(__name__)


class FusionStrategy(str, Enum):
    """Multimodal fusion strategy."""

    EARLY = "early"  # Concatenate features before model
    LATE = "late"  # Combine predictions from separate models
    ATTENTION = "attention"  # Cross-modal attention
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion."""

    # Fusion settings
    strategy: FusionStrategy = FusionStrategy.ATTENTION
    
    # Input dimensions
    video_feature_dim: int = 30  # Video features (gaze, head, behavioral, detection)
    audio_feature_dim: int = 5  # Audio features (VAD, speaker, stress)
    sequence_length: int = 30  # Frames per sequence

    # Embedding dimensions
    video_embed_dim: int = 64
    audio_embed_dim: int = 32
    joint_embed_dim: int = 128

    # Cross-modal attention
    num_attention_heads: int = 4
    attention_dropout: float = 0.1

    # LSTM for temporal processing
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2

    # Output
    num_classes: int = 2

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    device: str = "auto"

    def get_device(self) -> torch.device:
        """Get torch device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for aligning video and audio features.
    
    Uses video features to attend to audio and vice versa.
    """

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim

        # Video attends to audio
        self.video_to_audio = nn.MultiheadAttention(
            embed_dim=video_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=audio_dim,
            vdim=audio_dim,
            batch_first=True,
        )

        # Audio attends to video
        self.audio_to_video = nn.MultiheadAttention(
            embed_dim=audio_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=video_dim,
            vdim=video_dim,
            batch_first=True,
        )

        # Layer norms
        self.video_norm = nn.LayerNorm(video_dim)
        self.audio_norm = nn.LayerNorm(audio_dim)

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        video_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            video: Video features of shape (batch, seq_len, video_dim)
            audio: Audio features of shape (batch, seq_len, audio_dim)
            video_mask: Optional mask for video
            audio_mask: Optional mask for audio

        Returns:
            Enhanced video features, enhanced audio features, attention weights
        """
        # Video attends to audio
        video_enhanced, v2a_weights = self.video_to_audio(
            query=video,
            key=audio,
            value=audio,
            key_padding_mask=audio_mask,
        )
        video_out = self.video_norm(video + video_enhanced)

        # Audio attends to video
        audio_enhanced, a2v_weights = self.audio_to_video(
            query=audio,
            key=video,
            value=video,
            key_padding_mask=video_mask,
        )
        audio_out = self.audio_norm(audio + audio_enhanced)

        attention_weights = {
            "video_to_audio": v2a_weights,
            "audio_to_video": a2v_weights,
        }

        return video_out, audio_out, attention_weights


class ModalityEncoder(nn.Module):
    """Encoder for a single modality."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        lstm_hidden: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        self.output_dim = lstm_hidden * 2  # Bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode modality.

        Args:
            x: Input of shape (batch, seq_len, input_dim)

        Returns:
            Encoded features of shape (batch, seq_len, output_dim)
        """
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        return x


class MultimodalFusion(nn.Module):
    """
    Multimodal fusion model for combined audio-video analysis.
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config

        # Modality encoders
        self.video_encoder = ModalityEncoder(
            input_dim=config.video_feature_dim,
            embed_dim=config.video_embed_dim,
            lstm_hidden=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
        )

        self.audio_encoder = ModalityEncoder(
            input_dim=config.audio_feature_dim,
            embed_dim=config.audio_embed_dim,
            lstm_hidden=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
        )

        # Cross-modal attention (if using attention strategy)
        video_lstm_out = config.lstm_hidden_size * 2
        audio_lstm_out = config.lstm_hidden_size * 2

        self.cross_attention: CrossModalAttention | None = None
        if config.strategy in [FusionStrategy.ATTENTION, FusionStrategy.HYBRID]:
            self.cross_attention = CrossModalAttention(
                video_dim=video_lstm_out,
                audio_dim=audio_lstm_out,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
            )

        # Fusion layer
        combined_dim = video_lstm_out + audio_lstm_out
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, config.joint_embed_dim),
            nn.LayerNorm(config.joint_embed_dim),
            nn.ReLU(),
            nn.Dropout(config.lstm_dropout),
        )

        # Self-attention for fused features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.joint_embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(config.joint_embed_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.joint_embed_dim, config.joint_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.lstm_dropout),
            nn.Linear(config.joint_embed_dim // 2, config.num_classes),
        )

        # Modality weights for late fusion (learnable)
        self.video_weight = nn.Parameter(torch.tensor(0.7))
        self.audio_weight = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        video_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Forward pass.

        Args:
            video: Video features of shape (batch, seq_len, video_dim)
            audio: Audio features of shape (batch, seq_len, audio_dim)
            video_mask: Optional video mask
            audio_mask: Optional audio mask

        Returns:
            Logits and auxiliary outputs (attention weights, etc.)
        """
        batch_size = video.size(0)

        # Encode modalities
        video_encoded = self.video_encoder(video)
        audio_encoded = self.audio_encoder(audio)

        # Cross-modal attention
        attention_weights = {}
        if self.cross_attention is not None:
            video_encoded, audio_encoded, attn = self.cross_attention(
                video_encoded, audio_encoded, video_mask, audio_mask
            )
            attention_weights.update(attn)

        # Concatenate modalities
        fused = torch.cat([video_encoded, audio_encoded], dim=-1)
        fused = self.fusion(fused)

        # Self-attention on fused features
        attn_out, self_attn_weights = self.self_attention(fused, fused, fused)
        fused = self.attn_norm(fused + attn_out)
        attention_weights["self_attention"] = self_attn_weights

        # Pool over sequence (use last hidden state)
        pooled = fused[:, -1, :]

        # Classification
        logits = self.classifier(pooled)

        aux_outputs = {
            "attention_weights": attention_weights,
            "video_weight": self.video_weight.item(),
            "audio_weight": self.audio_weight.item(),
        }

        return logits, aux_outputs


class MultimodalDataset(Dataset):
    """Dataset for multimodal sequences."""

    def __init__(
        self,
        video_sequences: NDArray[np.float32],
        audio_sequences: NDArray[np.float32],
        labels: NDArray[np.int32],
    ):
        """
        Initialize dataset.

        Args:
            video_sequences: Video features of shape (n_samples, seq_len, video_dim)
            audio_sequences: Audio features of shape (n_samples, seq_len, audio_dim)
            labels: Labels of shape (n_samples,)
        """
        self.video = torch.FloatTensor(video_sequences)
        self.audio = torch.FloatTensor(audio_sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.video[idx], self.audio[idx], self.labels[idx]


class MultimodalDetector:
    """
    High-level API for multimodal cheating detection.
    """

    def __init__(self, config: FusionConfig | None = None):
        """Initialize detector.

        Args:
            config: Model configuration
        """
        self.config = config or FusionConfig()
        self.device = self.config.get_device()
        self.model: MultimodalFusion | None = None
        self._is_fitted = False

        logger.info(f"Multimodal detector using device: {self.device}")

    def fit(
        self,
        video_train: NDArray[np.float32],
        audio_train: NDArray[np.float32],
        y_train: NDArray[np.int32],
        video_val: NDArray[np.float32] | None = None,
        audio_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
    ) -> dict[str, list[float]]:
        """
        Train the multimodal model.

        Args:
            video_train: Video training sequences
            audio_train: Audio training sequences
            y_train: Training labels
            video_val: Optional video validation sequences
            audio_val: Optional audio validation sequences
            y_val: Optional validation labels

        Returns:
            Training history
        """
        logger.info("Training multimodal fusion model...")

        # Create model
        self.model = MultimodalFusion(self.config).to(self.device)

        # Create datasets
        train_dataset = MultimodalDataset(video_train, audio_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if video_val is not None and audio_val is not None and y_val is not None:
            val_dataset = MultimodalDataset(video_val, audio_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.CrossEntropyLoss()

        # Training history
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        # Training loop
        for epoch in range(self.config.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for video_batch, audio_batch, batch_y in train_loader:
                video_batch = video_batch.to(self.device)
                audio_batch = audio_batch.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(video_batch, audio_batch)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * video_batch.size(0)
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validate
            val_loss = 0.0
            val_acc = 0.0
            if val_loader is not None:
                self.model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for video_batch, audio_batch, batch_y in val_loader:
                        video_batch = video_batch.to(self.device)
                        audio_batch = audio_batch.to(self.device)
                        batch_y = batch_y.to(self.device)

                        logits, _ = self.model(video_batch, audio_batch)
                        loss = criterion(logits, batch_y)

                        val_loss += loss.item() * video_batch.size(0)
                        _, predicted = torch.max(logits, 1)
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)

                val_loss /= val_total
                val_acc = val_correct / val_total
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._is_fitted = True
        logger.info("Multimodal training complete")

        return history

    def predict(
        self,
        video: NDArray[np.float32],
        audio: NDArray[np.float32],
    ) -> NDArray[np.int32]:
        """
        Predict class labels.

        Args:
            video: Video sequences of shape (n_samples, seq_len, video_dim)
            audio: Audio sequences of shape (n_samples, seq_len, audio_dim)

        Returns:
            Predicted labels
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        video_tensor = torch.FloatTensor(video).to(self.device)
        audio_tensor = torch.FloatTensor(audio).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(video_tensor, audio_tensor)
            _, predictions = torch.max(logits, 1)

        return predictions.cpu().numpy()

    def predict_proba(
        self,
        video: NDArray[np.float32],
        audio: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Predict class probabilities.

        Args:
            video: Video sequences
            audio: Audio sequences

        Returns:
            Class probabilities
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        video_tensor = torch.FloatTensor(video).to(self.device)
        audio_tensor = torch.FloatTensor(audio).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(video_tensor, audio_tensor)
            proba = F.softmax(logits, dim=1)

        return proba.cpu().numpy()

    def predict_single(
        self,
        video: NDArray[np.float32],
        audio: NDArray[np.float32],
    ) -> tuple[bool, float, dict[str, Any]]:
        """
        Predict on a single sample for real-time use.

        Args:
            video: Video sequence of shape (seq_len, video_dim)
            audio: Audio sequence of shape (seq_len, audio_dim)

        Returns:
            Tuple of (is_cheating, confidence, auxiliary_info)
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        video_tensor = torch.FloatTensor(video).unsqueeze(0).to(self.device)
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, aux = self.model(video_tensor, audio_tensor)
            proba = F.softmax(logits, dim=1)
            confidence = proba[0, 1].item()
            is_cheating = confidence >= 0.5

        aux_info = {
            "video_weight": aux["video_weight"],
            "audio_weight": aux["audio_weight"],
        }

        return is_cheating, confidence, aux_info

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "config": self.config,
            "model_state": self.model.state_dict(),
            "is_fitted": self._is_fitted,
        }
        torch.save(save_dict, path)
        logger.info(f"Saved multimodal model to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MultimodalDetector":
        """Load model from disk."""
        save_dict = torch.load(path, map_location="cpu")

        detector = cls(config=save_dict["config"])
        detector.model = MultimodalFusion(detector.config)
        detector.model.load_state_dict(save_dict["model_state"])
        detector.model = detector.model.to(detector.device)
        detector._is_fitted = save_dict["is_fitted"]

        logger.info(f"Loaded multimodal model from {path}")
        return detector


class MultimodalBuffer:
    """Buffer for synchronized multimodal real-time inference."""

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 5,
        video_dim: int = 30,
        audio_dim: int = 5,
    ):
        """
        Initialize buffer.

        Args:
            window_size: Number of frames in each window
            stride: Number of frames to slide
            video_dim: Video feature dimension
            audio_dim: Audio feature dimension
        """
        self.window_size = window_size
        self.stride = stride
        self.video_buffer: list[NDArray[np.float32]] = []
        self.audio_buffer: list[NDArray[np.float32]] = []
        self.video_dim = video_dim
        self.audio_dim = audio_dim

    def add_frame(
        self,
        video_features: NDArray[np.float32],
        audio_features: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
        """
        Add synchronized video and audio frame.

        Args:
            video_features: Video feature vector
            audio_features: Audio feature vector

        Returns:
            Tuple of (video_sequence, audio_sequence) if ready, else None
        """
        self.video_buffer.append(video_features)
        self.audio_buffer.append(audio_features)

        if len(self.video_buffer) >= self.window_size:
            # Extract windows
            video_seq = np.array(self.video_buffer[-self.window_size:])
            audio_seq = np.array(self.audio_buffer[-self.window_size:])

            # Slide buffers
            if len(self.video_buffer) >= self.window_size + self.stride:
                self.video_buffer = self.video_buffer[self.stride:]
                self.audio_buffer = self.audio_buffer[self.stride:]

            return video_seq, audio_seq

        return None

    def reset(self) -> None:
        """Clear buffers."""
        self.video_buffer.clear()
        self.audio_buffer.clear()

    def is_ready(self) -> bool:
        """Check if buffers have enough frames."""
        return len(self.video_buffer) >= self.window_size
