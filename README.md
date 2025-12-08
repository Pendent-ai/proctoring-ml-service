# Proctoring ML Service

Real-time video and audio proctoring service using **YOLO11** (latest), MediaPipe, and custom ML models for AI interview platform.

## Features

### Video Proctoring
- **Object Detection**: YOLO11 (Ultralytics latest) for detecting phones, multiple persons, unauthorized objects
- **Face Analysis**: MediaPipe for face detection, gaze tracking, head pose estimation
- **Cheating Classifier**: XGBoost model trained on behavioral patterns
- **LiveKit Integration**: Subscribes to video tracks and publishes alerts via data channel

### Audio Proctoring
- **Voice Activity Detection (VAD)**: Silero VAD for accurate speech detection
- **Speaker Diarization**: Detect multiple speakers (potential cheating)
- **Background Speech Detection**: Identify off-screen assistance
- **Suspicious Audio Detection**: Detect unusual patterns (text-to-speech, recordings)
- **Real-time Analysis**: 500ms audio chunks processed with minimal latency

## YOLO11 Performance

| Model   | mAP50 | Speed (T4) | Size  | Notes |
|---------|-------|------------|-------|-------|
| yolo11n | 39.5% | 100+ FPS   | 2.6M  | ← Default (real-time) |
| yolo11s | 47.0% | 60 FPS     | 9.4M  | Balanced |
| yolo11m | 51.5% | 30 FPS     | 20.1M | Higher accuracy |
| yolo11l | 53.4% | 20 FPS     | 25.3M | |
| yolo11x | 54.7% | 11 FPS     | 56.9M | ← Best accuracy |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROCTORING ML SERVICE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  VIDEO TRACK → FRAME DECODER → [YOLO11 + MEDIAPIPE] → FEATURE EXTRACTION    │
│                                                              │               │
│                                                              ▼               │
│  AUDIO TRACK → AUDIO DECODER → [VAD + DIARIZATION] ──→ CLASSIFIER → ALERTS │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Create virtual environment (Python 3.11+ required)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the service (YOLO11 model downloads automatically if not present)
python main.py
```

## Documentation

- [Architecture](docs/architecture.md) - System design, data flow, scaling
- [API Reference](docs/api-reference.md) - Alert schemas, data channel topics

## Installation

```bash
# Create virtual environment (Python 3.11+ required)
python3.11 -m venv venv
source venv/bin/activate

# Install with pip (from pyproject.toml)
pip install -e .

# Or install with optional dependencies
pip install -e ".[audio,classifier]"  # Audio proctoring + XGBoost classifier
pip install -e ".[all]"               # All dependencies including dev
```

## Quick Start

```python
from proctor import VideoProctor, AudioProctor

# Video proctoring
video_model = VideoProctor()
result = video_model.predict(frame)
print(result.phone_detected, result.cheating_probability)

# Audio proctoring  
audio_model = AudioProctor()
result = audio_model.predict(audio_samples)
print(result.multiple_speakers, result.whispering_detected)

# Full service with LiveKit
from proctor.service import ProctoringService
service = ProctoringService()
await service.run("room-name")
```

## Project Structure

```
proctoring-ml-service/
├── main.py                     # Entry point
├── pyproject.toml              # Modern Python packaging
├── requirements.txt            # Legacy dependencies
├── Dockerfile                  # Container build
│
├── proctor/                    # Main Package (Ultralytics-style)
│   ├── __init__.py             # Package exports
│   │
│   ├── engine/                 # Base Classes
│   │   ├── model.py            # BaseModel with task_map pattern
│   │   ├── predictor.py        # BasePredictor (preprocess/inference/postprocess)
│   │   ├── trainer.py          # BaseTrainer (training loop)
│   │   ├── validator.py        # BaseValidator (evaluation)
│   │   └── results.py          # Result dataclasses
│   │
│   ├── models/                 # Proctor Models
│   │   ├── video/              # Video Proctoring
│   │   │   ├── model.py        # VideoProctor (YOLO11 + MediaPipe)
│   │   │   ├── predictor.py    # Frame analysis pipeline
│   │   │   ├── trainer.py      # Classifier training
│   │   │   └── validator.py    # Model validation
│   │   │
│   │   └── audio/              # Audio Proctoring
│   │       ├── model.py        # AudioProctor (Silero VAD)
│   │       └── predictor.py    # Audio analysis pipeline
│   │
│   ├── cfg/                    # Configuration
│   │   └── config.py           # Pydantic configs (VideoConfig, AudioConfig)
│   │
│   ├── data/                   # LiveKit Integration
│   │   ├── video_receiver.py   # Video frame receiving
│   │   ├── audio_receiver.py   # Audio chunk receiving
│   │   └── publisher.py        # Alert publishing
│   │
│   ├── service/                # Main Service
│   │   └── proctoring.py       # ProctoringService orchestration
│   │
│   └── utils/                  # Utilities
│       ├── logger.py           # Structured logging
│       └── alerts.py           # Alert types, severities, messages
│
├── scripts/                    # Training Scripts
│   ├── prepare_dataset.py      # Dataset preparation
│   ├── train_yolo.py           # YOLO11 fine-tuning
│   └── evaluate.py             # Model evaluation
│
├── weights/                    # Model Weights (Git LFS)
│   └── yolo11n.pt              # Base YOLO11 model
│
└── docs/                       # Documentation
    ├── architecture.md         # System architecture
    └── api-reference.md        # API documentation
```

## Environment Variables

```env
# LiveKit
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# Model paths
YOLO_MODEL_PATH=weights/yolo11n.pt
CLASSIFIER_MODEL_PATH=weights/classifier.json

# Processing
PROCESS_FPS=10
FRAME_WIDTH=640
FRAME_HEIGHT=480

# GPU
USE_GPU=true
GPU_MEMORY_FRACTION=0.8

# Thresholds
CHEATING_THRESHOLD=0.7
ALERT_COOLDOWN=5
```

## Docker

```bash
# Build
docker build -t proctoring-ml-service .

# Run with GPU
docker run --gpus all \
  -e LIVEKIT_URL=wss://your-server.livekit.cloud \
  -e LIVEKIT_API_KEY=key \
  -e LIVEKIT_API_SECRET=secret \
  proctoring-ml-service
```

## Training

```bash
# Train classifier on labeled data
from proctor import VideoProctor

model = VideoProctor()
results = model.train("data/training_data.json", epochs=100)
print(f"Best accuracy: {results['best_fitness']}")
```

## License

Proprietary - Pendent AI

