"""
Temporal sequence model for cheating detection.

This module implements LSTM/Transformer-based models that analyze
temporal patterns in proctoring features to detect suspicious behavior
over time windows.

Features:
- Bidirectional LSTM with self-attention
- Transformer encoder for long-range dependencies
- Sequence-level and frame-level predictions
- Sliding window inference for real-time use
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


class TemporalModelType(str, Enum):
    """Type of temporal model."""

    LSTM = "lstm"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"


@dataclass
class TemporalConfig:
    """Configuration for temporal models."""

    # Model type
    model_type: TemporalModelType = TemporalModelType.LSTM

    # Input dimensions
    input_dim: int = 35  # Number of features
    sequence_length: int = 30  # ~1 second at 30fps

    # LSTM settings
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True

    # Transformer settings
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_num_layers: int = 3
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1

    # Attention settings
    use_attention: bool = True
    attention_heads: int = 4

    # Output settings
    num_classes: int = 2
    output_mode: str = "sequence"  # 'sequence' or 'frame'

    # Training settings
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


class SelfAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            mask: Optional attention mask

        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim**0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)

        # Output projection with residual
        output = self.out(context)
        output = self.layer_norm(x + output)

        return output, attn_weights


class TemporalLSTM(nn.Module):
    """Bidirectional LSTM with self-attention for temporal cheating detection."""

    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.lstm_hidden_size),
            nn.LayerNorm(config.lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.lstm_dropout),
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.lstm_hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            bidirectional=config.lstm_bidirectional,
            batch_first=True,
        )

        # Effective hidden size after LSTM
        lstm_output_size = (
            config.lstm_hidden_size * 2
            if config.lstm_bidirectional
            else config.lstm_hidden_size
        )

        # Self-attention (optional)
        self.attention: SelfAttention | None = None
        if config.use_attention:
            self.attention = SelfAttention(
                lstm_output_size,
                num_heads=config.attention_heads,
                dropout=config.lstm_dropout,
            )

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.LayerNorm(lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(config.lstm_dropout),
            nn.Linear(lstm_output_size // 2, config.num_classes),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            lengths: Optional sequence lengths for masking

        Returns:
            Logits and attention weights (if attention is used)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Apply attention if enabled
        attn_weights = None
        if self.attention is not None:
            lstm_out, attn_weights = self.attention(lstm_out)

        # Output mode
        if self.config.output_mode == "sequence":
            # Use last hidden state for sequence classification
            output = lstm_out[:, -1, :]
        else:
            # Frame-level predictions
            output = lstm_out

        # Classification
        logits = self.classifier(output)

        return logits, attn_weights


class TemporalTransformer(nn.Module):
    """Transformer encoder for temporal pattern detection."""

    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(config.input_dim, config.transformer_d_model),
            nn.LayerNorm(config.transformer_d_model),
        )

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(
            config.sequence_length, config.transformer_d_model
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            dim_feedforward=config.transformer_dim_feedforward,
            dropout=config.transformer_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.transformer_num_layers
        )

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.transformer_d_model))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.transformer_d_model, config.transformer_d_model // 2),
            nn.LayerNorm(config.transformer_d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.transformer_dropout),
            nn.Linear(config.transformer_d_model // 2, config.num_classes),
        )

    def _create_positional_encoding(
        self, max_len: int, d_model: int
    ) -> nn.Parameter:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len + 1, d_model)  # +1 for CLS token
        position = torch.arange(0, max_len + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Logits and attention from last layer
        """
        batch_size, seq_len, _ = x.shape

        # Embed input
        x = self.input_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = x + self.pos_encoding[:, : seq_len + 1, :]

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=mask)

        # Use CLS token for classification
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)

        # Get attention from transformer (simplified)
        attention = torch.zeros(batch_size, seq_len + 1)

        return logits, attention


class HybridModel(nn.Module):
    """Hybrid LSTM + Transformer model."""

    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config

        # LSTM branch
        self.lstm_branch = TemporalLSTM(config)

        # Transformer branch (with smaller config)
        transformer_config = TemporalConfig(
            model_type=TemporalModelType.TRANSFORMER,
            input_dim=config.input_dim,
            sequence_length=config.sequence_length,
            transformer_d_model=config.lstm_hidden_size,
            transformer_nhead=4,
            transformer_num_layers=2,
            num_classes=config.num_classes,
        )
        self.transformer_branch = TemporalTransformer(transformer_config)

        # Fusion layer
        lstm_out_size = (
            config.lstm_hidden_size * 2
            if config.lstm_bidirectional
            else config.lstm_hidden_size
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.num_classes * 2, config.num_classes * 2),
            nn.ReLU(),
            nn.Linear(config.num_classes * 2, config.num_classes),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass combining LSTM and Transformer."""
        lstm_logits, lstm_attn = self.lstm_branch(x, lengths)
        transformer_logits, transformer_attn = self.transformer_branch(x)

        # Fuse predictions
        combined = torch.cat([lstm_logits, transformer_logits], dim=-1)
        logits = self.fusion(combined)

        attention = {
            "lstm": lstm_attn,
            "transformer": transformer_attn,
        }

        return logits, attention


class SequenceDataset(Dataset):
    """Dataset for temporal sequences."""

    def __init__(
        self,
        sequences: NDArray[np.float32],
        labels: NDArray[np.int32],
        sequence_length: int = 30,
    ):
        """
        Initialize dataset.

        Args:
            sequences: Feature sequences of shape (n_samples, seq_len, n_features)
            labels: Labels of shape (n_samples,)
            sequence_length: Expected sequence length
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class TemporalCheatingDetector:
    """
    High-level API for temporal cheating detection.

    Wraps the underlying models and provides train/predict interface.
    """

    def __init__(self, config: TemporalConfig | None = None):
        """Initialize detector.

        Args:
            config: Model configuration
        """
        self.config = config or TemporalConfig()
        self.device = self.config.get_device()
        self.model: nn.Module | None = None
        self._is_fitted = False

        logger.info(f"Using device: {self.device}")

    def _create_model(self) -> nn.Module:
        """Create the appropriate model based on config."""
        if self.config.model_type == TemporalModelType.LSTM:
            return TemporalLSTM(self.config)
        elif self.config.model_type == TemporalModelType.TRANSFORMER:
            return TemporalTransformer(self.config)
        else:
            return HybridModel(self.config)

    def fit(
        self,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
    ) -> dict[str, list[float]]:
        """
        Train the temporal model.

        Args:
            X_train: Training sequences of shape (n_samples, seq_len, n_features)
            y_train: Training labels
            X_val: Optional validation sequences
            y_val: Optional validation labels

        Returns:
            Training history with loss and accuracy
        """
        logger.info(f"Training {self.config.model_type.value} model...")

        # Create model
        self.model = self._create_model().to(self.device)

        # Create datasets
        train_dataset = SequenceDataset(
            X_train, y_train, self.config.sequence_length
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = SequenceDataset(
                X_val, y_val, self.config.sequence_length
            )
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

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)
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
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        logits, _ = self.model(batch_x)
                        loss = criterion(logits, batch_y)

                        val_loss += loss.item() * batch_x.size(0)
                        _, predicted = torch.max(logits, 1)
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)

                val_loss /= val_total
                val_acc = val_correct / val_total
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                scheduler.step(val_loss)

                # Early stopping check
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
        logger.info("Training complete")

        return history

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int32]:
        """
        Predict class labels.

        Args:
            X: Sequences of shape (n_samples, seq_len, n_features)

        Returns:
            Predicted labels
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(X_tensor)
            _, predictions = torch.max(logits, 1)

        return predictions.cpu().numpy()

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Predict class probabilities.

        Args:
            X: Sequences of shape (n_samples, seq_len, n_features)

        Returns:
            Class probabilities
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(X_tensor)
            proba = F.softmax(logits, dim=1)

        return proba.cpu().numpy()

    def predict_single_online(
        self, sequence: NDArray[np.float32]
    ) -> tuple[bool, float, NDArray[np.float32] | None]:
        """
        Predict on a single sequence for real-time use.

        Args:
            sequence: Single sequence of shape (seq_len, n_features)

        Returns:
            Tuple of (is_cheating, confidence, attention_weights)
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, attention = self.model(X)
            proba = F.softmax(logits, dim=1)
            confidence = proba[0, 1].item()
            is_cheating = confidence >= 0.5

        attn_weights = None
        if attention is not None:
            if isinstance(attention, torch.Tensor):
                attn_weights = attention.cpu().numpy()
            elif isinstance(attention, dict) and "lstm" in attention:
                if attention["lstm"] is not None:
                    attn_weights = attention["lstm"].cpu().numpy()

        return is_cheating, confidence, attn_weights

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
        logger.info(f"Saved temporal model to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TemporalCheatingDetector":
        """Load model from disk."""
        save_dict = torch.load(path, map_location="cpu")

        detector = cls(config=save_dict["config"])
        detector.model = detector._create_model()
        detector.model.load_state_dict(save_dict["model_state"])
        detector.model = detector.model.to(detector.device)
        detector._is_fitted = save_dict["is_fitted"]

        logger.info(f"Loaded temporal model from {path}")
        return detector


class SlidingWindowBuffer:
    """Buffer for real-time sliding window inference."""

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 5,
        n_features: int = 35,
    ):
        """
        Initialize buffer.

        Args:
            window_size: Number of frames in each window
            stride: Number of frames to slide
            n_features: Number of features per frame
        """
        self.window_size = window_size
        self.stride = stride
        self.n_features = n_features
        self.buffer: list[NDArray[np.float32]] = []
        self.frame_count = 0

    def add_frame(self, features: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """
        Add a frame and return sequence if window is ready.

        Args:
            features: Feature vector of shape (n_features,)

        Returns:
            Sequence of shape (window_size, n_features) if ready, else None
        """
        self.buffer.append(features)
        self.frame_count += 1

        if len(self.buffer) >= self.window_size:
            # Extract window
            sequence = np.array(self.buffer[-self.window_size :])

            # Slide buffer
            if len(self.buffer) >= self.window_size + self.stride:
                self.buffer = self.buffer[self.stride:]

            return sequence

        return None

    def reset(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
        self.frame_count = 0

    def is_ready(self) -> bool:
        """Check if buffer has enough frames."""
        return len(self.buffer) >= self.window_size
