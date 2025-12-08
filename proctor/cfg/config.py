"""
Proctor Configuration Classes

Pydantic-based configuration with validation and defaults.
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
from pathlib import Path


class BaseConfig(BaseModel):
    """Base configuration class for all proctoring configs."""
    
    verbose: bool = Field(default=True, description="Enable verbose output")
    device: str = Field(default="auto", description="Device to use (auto, cpu, cuda)")
    
    class Config:
        extra = "allow"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model_dump()})"


class VideoConfig(BaseConfig):
    """Configuration for video proctoring."""
    
    # Model paths
    yolo_model_path: str = Field(
        default="weights/yolo11n.pt",
        description="Path to YOLO model weights",
    )
    classifier_model_path: str = Field(
        default="weights/classifier.json",
        description="Path to classifier model",
    )
    
    # Processing
    frame_width: int = Field(default=640, description="Frame width for processing")
    frame_height: int = Field(default=480, description="Frame height for processing")
    process_fps: int = Field(default=10, description="Target FPS for processing")
    sliding_window: int = Field(default=60, description="Sliding window size in frames")
    
    # Detection thresholds
    yolo_confidence: float = Field(default=0.5, description="YOLO detection confidence threshold")
    face_confidence: float = Field(default=0.5, description="Face detection confidence threshold")
    cheating_threshold: float = Field(default=0.7, description="Cheating classifier threshold")
    
    # GPU settings
    use_gpu: bool = Field(default=True, description="Use GPU acceleration")
    gpu_memory_fraction: float = Field(default=0.8, description="GPU memory fraction to use")
    
    # Alert settings
    alert_cooldown: int = Field(default=5, description="Cooldown between same alerts (seconds)")
    
    # Gaze thresholds
    gaze_away_threshold_x: float = Field(default=0.3, description="Gaze X threshold for looking away")
    gaze_away_threshold_y: float = Field(default=0.25, description="Gaze Y threshold for looking away")


class AudioConfig(BaseConfig):
    """Configuration for audio proctoring."""
    
    # Processing
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    chunk_duration_ms: int = Field(default=500, description="Audio chunk duration in ms")
    window_size: int = Field(default=10, description="Sliding window size for temporal analysis")
    
    # VAD settings
    enable_vad: bool = Field(default=True, description="Enable voice activity detection")
    vad_threshold: float = Field(default=0.5, description="VAD confidence threshold")
    
    # Thresholds
    silence_threshold_db: float = Field(default=-45, description="Silence threshold in dB")
    whisper_threshold_db: float = Field(default=-35, description="Whisper threshold in dB")
    normal_speech_db: float = Field(default=-20, description="Normal speech level in dB")
    
    # Alert settings
    alert_cooldown_seconds: float = Field(default=15.0, description="Cooldown between same alerts")
    
    # Baseline calibration
    calibration_samples: int = Field(default=10, description="Samples for baseline calibration")


class LiveKitConfig(BaseConfig):
    """Configuration for LiveKit integration."""
    
    url: str = Field(default="ws://localhost:7880", description="LiveKit server URL")
    api_key: str = Field(default="devkey", description="LiveKit API key")
    api_secret: str = Field(default="secret", description="LiveKit API secret")
    
    # Publisher settings
    data_topic: str = Field(default="proctoring", description="Data channel topic for alerts")
    status_topic: str = Field(default="proctoring_status", description="Status update topic")


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    Uses pydantic-settings for automatic environment variable loading.
    """
    
    # LiveKit
    livekit_url: str = Field(default="ws://localhost:7880", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="devkey", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="secret", env="LIVEKIT_API_SECRET")
    
    # Model paths
    yolo_model_path: str = Field(default="weights/yolo11n.pt", env="YOLO_MODEL_PATH")
    classifier_model_path: str = Field(default="weights/classifier.json", env="CLASSIFIER_MODEL_PATH")
    
    # Processing
    process_fps: int = Field(default=10, env="PROCESS_FPS")
    frame_width: int = Field(default=640, env="FRAME_WIDTH")
    frame_height: int = Field(default=480, env="FRAME_HEIGHT")
    
    # GPU
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    use_gpu: bool = Field(default=True, env="USE_GPU")
    
    # Detection thresholds
    yolo_confidence: float = Field(default=0.5, env="YOLO_CONFIDENCE")
    face_confidence: float = Field(default=0.5, env="FACE_CONFIDENCE")
    cheating_threshold: float = Field(default=0.7, env="CHEATING_THRESHOLD")
    
    # Alert settings
    alert_cooldown: int = Field(default=5, env="ALERT_COOLDOWN")
    sliding_window: int = Field(default=60, env="SLIDING_WINDOW")
    
    # Database
    mongodb_uri: Optional[str] = Field(default=None, env="MONGODB_URI")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def to_video_config(self) -> VideoConfig:
        """Convert settings to VideoConfig."""
        return VideoConfig(
            yolo_model_path=self.yolo_model_path,
            classifier_model_path=self.classifier_model_path,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            process_fps=self.process_fps,
            sliding_window=self.sliding_window,
            yolo_confidence=self.yolo_confidence,
            face_confidence=self.face_confidence,
            cheating_threshold=self.cheating_threshold,
            use_gpu=self.use_gpu,
            gpu_memory_fraction=self.gpu_memory_fraction,
            alert_cooldown=self.alert_cooldown,
        )
    
    def to_audio_config(self) -> AudioConfig:
        """Convert settings to AudioConfig."""
        return AudioConfig()
    
    def to_livekit_config(self) -> LiveKitConfig:
        """Convert settings to LiveKitConfig."""
        return LiveKitConfig(
            url=self.livekit_url,
            api_key=self.livekit_api_key,
            api_secret=self.livekit_api_secret,
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Training configuration classes
class EnsembleTrainingConfig(BaseConfig):
    """Configuration for ensemble classifier training."""
    
    # Model architecture
    use_xgboost: bool = Field(default=True, description="Include XGBoost in ensemble")
    use_lightgbm: bool = Field(default=True, description="Include LightGBM in ensemble")
    use_catboost: bool = Field(default=True, description="Include CatBoost in ensemble")
    use_mlp: bool = Field(default=True, description="Include MLP in ensemble")
    
    # XGBoost params
    xgb_n_estimators: int = Field(default=200, description="XGBoost number of estimators")
    xgb_max_depth: int = Field(default=6, description="XGBoost max depth")
    xgb_learning_rate: float = Field(default=0.1, description="XGBoost learning rate")
    
    # Training
    use_smote: bool = Field(default=True, description="Use SMOTE for class imbalance")
    use_calibration: bool = Field(default=True, description="Use probability calibration")
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    random_state: int = Field(default=42, description="Random seed")


class TemporalTrainingConfig(BaseConfig):
    """Configuration for temporal LSTM/Transformer training."""
    
    # Model architecture
    model_type: str = Field(default="lstm", description="Model type: lstm, transformer, hybrid")
    input_dim: int = Field(default=35, description="Input feature dimension")
    sequence_length: int = Field(default=30, description="Sequence length (frames)")
    
    # LSTM settings
    lstm_hidden_size: int = Field(default=128, description="LSTM hidden size")
    lstm_num_layers: int = Field(default=2, description="Number of LSTM layers")
    lstm_dropout: float = Field(default=0.3, description="LSTM dropout rate")
    lstm_bidirectional: bool = Field(default=True, description="Use bidirectional LSTM")
    
    # Transformer settings
    transformer_d_model: int = Field(default=128, description="Transformer model dimension")
    transformer_nhead: int = Field(default=8, description="Number of attention heads")
    transformer_num_layers: int = Field(default=3, description="Number of transformer layers")
    
    # Attention
    use_attention: bool = Field(default=True, description="Use self-attention layer")
    attention_heads: int = Field(default=4, description="Number of attention heads")
    
    # Training
    epochs: int = Field(default=100, description="Maximum training epochs")
    batch_size: int = Field(default=32, description="Batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    weight_decay: float = Field(default=1e-5, description="Weight decay")
    patience: int = Field(default=10, description="Early stopping patience")


class MultimodalTrainingConfig(BaseConfig):
    """Configuration for multimodal fusion training."""
    
    # Fusion strategy
    fusion_strategy: str = Field(default="attention", description="Fusion: early, late, attention, hybrid")
    
    # Input dimensions
    video_feature_dim: int = Field(default=30, description="Video feature dimension")
    audio_feature_dim: int = Field(default=5, description="Audio feature dimension")
    
    # Embedding dimensions
    video_embed_dim: int = Field(default=64, description="Video embedding dimension")
    audio_embed_dim: int = Field(default=32, description="Audio embedding dimension")
    joint_embed_dim: int = Field(default=128, description="Joint embedding dimension")
    
    # Cross-modal attention
    num_attention_heads: int = Field(default=4, description="Number of attention heads")
    attention_dropout: float = Field(default=0.1, description="Attention dropout")
    
    # Training
    epochs: int = Field(default=100, description="Maximum training epochs")
    batch_size: int = Field(default=32, description="Batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    patience: int = Field(default=10, description="Early stopping patience")


class YOLOTrainingConfig(BaseConfig):
    """Configuration for custom YOLO training."""
    
    # Model
    base_model: str = Field(default="yolo11n.pt", description="Base YOLO model")
    model_size: str = Field(default="n", description="Model size: n, s, m, l, x")
    
    # Training
    epochs: int = Field(default=100, description="Training epochs")
    batch_size: int = Field(default=16, description="Batch size")
    image_size: int = Field(default=640, description="Input image size")
    learning_rate: float = Field(default=0.01, description="Initial learning rate")
    
    # Augmentation
    augmentation_level: str = Field(default="medium", description="Augmentation: none, light, medium, heavy")
    mosaic: float = Field(default=1.0, description="Mosaic augmentation probability")
    mixup: float = Field(default=0.0, description="Mixup augmentation probability")
    
    # Output
    project: str = Field(default="runs/train", description="Project directory")
    name: str = Field(default="proctoring_yolo", description="Experiment name")

