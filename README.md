# Proctoring ML Service

Real-time video proctoring service using **YOLO11** (latest), MediaPipe, and custom ML models for AI interview platform.

## Features

- **Object Detection**: YOLO11 (Ultralytics latest) for detecting phones, multiple persons, unauthorized objects
- **Face Analysis**: MediaPipe for face detection, gaze tracking, head pose estimation
- **Cheating Classifier**: XGBoost model trained on behavioral patterns
- **LiveKit Integration**: Subscribes to video tracks and publishes alerts via data channel

## YOLO11 Performance

| Model   | mAP50 | Speed (T4) | Size  |
|---------|-------|------------|-------|
| yolo11n | 39.5% | 100+ FPS   | 2.6M  | ← Default (real-time)
| yolo11s | 47.0% | 60 FPS     | 9.4M  |
| yolo11m | 51.5% | 30 FPS     | 20.1M |
| yolo11l | 53.4% | 20 FPS     | 25.3M |
| yolo11x | 54.7% | 11 FPS     | 56.9M | ← Best accuracy

## Architecture

```
VIDEO TRACK → FRAME DECODER → [YOLO11 + MEDIAPIPE] → FEATURE EXTRACTION → CLASSIFIER → ALERTS
```

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Download YOLO11 model
python scripts/download_models.py

# Run the service
python main.py
```

## Fine-tuning YOLO11

See [docs/fine_tuning.md](docs/fine_tuning.md) for instructions on fine-tuning YOLO11 for interview-specific objects.

```bash
# Prepare dataset
python scripts/prepare_dataset.py --input data/raw --output data/prepared

# Fine-tune model (uses yolo11n by default)
python scripts/train_yolo.py --data data/prepared/dataset.yaml --epochs 50

# Use larger model for better accuracy
python scripts/train_yolo.py --data data/prepared/dataset.yaml --model yolo11s.pt --epochs 100

# Evaluate model
python scripts/evaluate.py --model runs/detect/train/weights/best.pt
```

## Project Structure

```
proctoring-ml-service/
├── main.py                     # Entry point
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container build
│
├── livekit_client/             # LiveKit integration
│   ├── room.py                 # Room management
│   ├── track.py                # Track subscription
│   └── publisher.py            # Alert publishing
│
├── models/                     # ML models
│   ├── yolo.py                 # YOLOv8 wrapper
│   ├── mediapipe.py            # MediaPipe wrapper
│   └── classifier.py           # Cheating classifier
│
├── pipeline/                   # Processing pipeline
│   ├── processor.py            # Main frame processor
│   ├── features.py             # Feature extraction
│   └── alerts.py               # Alert generation
│
├── scripts/                    # Utility scripts
│   ├── download_models.py      # Download pre-trained models
│   ├── prepare_dataset.py      # Dataset preparation
│   ├── train_yolo.py           # YOLOv8 fine-tuning
│   └── evaluate.py             # Model evaluation
│
├── data/                       # Training data
│   ├── raw/                    # Raw images
│   ├── prepared/               # Processed dataset
│   └── annotations/            # YOLO format labels
│
├── weights/                    # Model weights
│   ├── yolo11n.pt              # Base YOLO11 nano
│   └── yolo11_interview.pt     # Fine-tuned model
│
└── docs/                       # Documentation
    └── fine_tuning.md          # Fine-tuning guide
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
GPU_MEMORY_FRACTION=0.8
```

## License

Proprietary - Pendent AI
