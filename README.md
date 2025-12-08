# Proctoring ML Service

Real-time video proctoring service using YOLOv8, MediaPipe, and custom ML models for AI interview platform.

## Features

- **Object Detection**: YOLOv8 for detecting phones, multiple persons, unauthorized objects
- **Face Analysis**: MediaPipe for face detection, gaze tracking, head pose estimation
- **Cheating Classifier**: XGBoost model trained on behavioral patterns
- **LiveKit Integration**: Subscribes to video tracks and publishes alerts via data channel

## Architecture

```
VIDEO TRACK → FRAME DECODER → [YOLO + MEDIAPIPE] → FEATURE EXTRACTION → CLASSIFIER → ALERTS
```

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model
python scripts/download_models.py

# Run the service
python main.py
```

## Fine-tuning YOLOv8

See [docs/fine_tuning.md](docs/fine_tuning.md) for instructions on fine-tuning YOLOv8 for interview-specific objects.

```bash
# Prepare dataset
python scripts/prepare_dataset.py --input data/raw --output data/prepared

# Fine-tune model
python scripts/train_yolo.py --data data/prepared/dataset.yaml --epochs 50

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
│   ├── yolov8n.pt              # Base YOLOv8 nano
│   └── yolov8_interview.pt     # Fine-tuned model
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
YOLO_MODEL_PATH=weights/yolov8n.pt
CLASSIFIER_MODEL_PATH=weights/classifier.json

# Processing
PROCESS_FPS=10
GPU_MEMORY_FRACTION=0.8
```

## License

Proprietary - Pendent AI
