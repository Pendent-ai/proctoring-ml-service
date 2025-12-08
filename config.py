"""
Proctoring ML Service Configuration
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # LiveKit
    livekit_url: str = Field(default="ws://localhost:7880", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="devkey", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="secret", env="LIVEKIT_API_SECRET")
    
    # Model paths
    yolo_model_path: str = Field(default="weights/yolov8n.pt", env="YOLO_MODEL_PATH")
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
    alert_cooldown: int = Field(default=5, env="ALERT_COOLDOWN")  # seconds
    sliding_window: int = Field(default=60, env="SLIDING_WINDOW")  # frames
    
    # Database (optional)
    mongodb_uri: str | None = Field(default=None, env="MONGODB_URI")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
