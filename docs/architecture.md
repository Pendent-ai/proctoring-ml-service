# Proctoring ML Service Architecture

## Overview

The Proctoring ML Service is a real-time video and audio analysis service that detects cheating behaviors during AI-proctored interviews. It subscribes to LiveKit video and audio tracks, runs ML inference, and publishes alerts via LiveKit Data Channel.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROCTORING ML SERVICE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │   LiveKit   │    │               PROCESSING LAYER                   │   │
│  │   Server    │    │                                                  │   │
│  │             │    │  ┌─────────────────┐  ┌─────────────────────┐   │   │
│  │  ┌───────┐  │    │  │ Video Pipeline  │  │   Audio Pipeline    │   │   │
│  │  │ Video ├──┼────┼──► YOLO11         │  │   Silero VAD        │   │   │
│  │  │ Track │  │    │  │ MediaPipe      │  │   Speaker Analysis  │   │   │
│  │  └───────┘  │    │  │ Feature Extract│  │   Whisper Detect    │   │   │
│  │             │    │  └────────┬───────┘  └──────────┬──────────┘   │   │
│  │  ┌───────┐  │    │           │                     │              │   │
│  │  │ Audio ├──┼────┼───────────┼─────────────────────┘              │   │
│  │  │ Track │  │    │           │                                    │   │
│  │  └───────┘  │    │           ▼                                    │   │
│  │             │    │  ┌─────────────────────────────────────────┐   │   │
│  │  ┌───────┐  │    │  │           FUSION & CLASSIFICATION       │   │   │
│  │  │ Data  │◄─┼────┼──│  Feature Merger → XGBoost Classifier   │   │   │
│  │  │Channel│  │    │  │  Combined Score → Alert Generator       │   │   │
│  │  └───────┘  │    │  └─────────────────────────────────────────┘   │   │
│  └─────────────┘    └──────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. LiveKit Client Layer (`livekit_client/`)

| Component | File | Purpose |
|-----------|------|---------|
| Room Manager | `room_manager.py` | Token generation, room connection, event handling |
| Video Receiver | `video_receiver.py` | Receive video frames at target FPS |
| Audio Receiver | `audio_receiver.py` | Receive audio chunks, buffering |
| Alert Publisher | `publisher.py` | Publish alerts via LiveKit Data Channel |

### 2. Models Layer (`models/`)

| Component | File | Purpose |
|-----------|------|---------|
| YOLO11 Detector | `yolo.py` | Object detection (phone, person, laptop, book) |
| MediaPipe Analyzer | `mediapipe.py` | Face mesh, gaze tracking, head pose |
| Audio Analyzer | `audio.py` | VAD, speaker diarization, whispering |
| Cheating Classifier | `classifier.py` | XGBoost model for cheating probability |

### 3. Pipeline Layer (`pipeline/`)

| Component | File | Purpose |
|-----------|------|---------|
| Frame Processor | `processor.py` | Main video processing loop |
| Audio Pipeline | `audio_pipeline.py` | Audio chunk processing |
| Feature Extractor | `features.py` | Extract 15+ features from sliding window |
| Alert Generator | `alerts.py` | Rule-based alert generation with cooldowns |

### 4. Alert System (`alerts/`)

| Component | File | Purpose |
|-----------|------|---------|
| Alert Types | `types.py` | Enum of alert types, severities, messages |

## Data Flow

```
VIDEO FRAME (30 FPS)
    │
    ▼ (downsample to 10 FPS)
┌───────────────────┐
│  Frame Processor  │
├───────────────────┤
│ 1. Resize 640x480 │
│ 2. YOLO11 detect  │◄──── phone, person, laptop, book
│ 3. MediaPipe      │◄──── face, gaze, head pose
│ 4. Add to window  │
└─────────┬─────────┘
          │
          ▼ (every 60 frames / 2 sec)
┌───────────────────┐
│ Feature Extractor │
├───────────────────┤
│ gaze_x_mean       │
│ gaze_y_mean       │
│ gaze_variance     │
│ gaze_away_ratio   │
│ head_yaw_mean     │
│ phone_detected    │
│ face_visible_ratio│
│ ... (15 features) │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│    Classifier     │
├───────────────────┤
│ XGBoost model     │
│ OR rule-based     │
│                   │
│ cheating_prob:    │
│   0.0 - 1.0       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Alert Generator  │
├───────────────────┤
│ Check thresholds  │
│ Apply cooldowns   │
│ Determine severity│
└─────────┬─────────┘
          │
          ▼
    ALERT (if needed)
    → LiveKit Data Channel
```

## Alert Types

### Video Alerts
| Type | Trigger | Default Severity |
|------|---------|------------------|
| `phone_detected` | Phone visible in frame | HIGH |
| `multiple_faces` | More than 1 face detected | HIGH |
| `looking_away` | Gaze off-screen > 40% of window | MEDIUM |
| `face_not_visible` | No face detected > 20% of window | MEDIUM |
| `suspicious_behavior` | High cheating probability | VARIES |

### Audio Alerts
| Type | Trigger | Default Severity |
|------|---------|------------------|
| `multiple_voices` | > 1 speaker detected | HIGH |
| `background_speech` | Voice in background detected | MEDIUM |
| `whispering` | Low-energy speech detected | MEDIUM |
| `suspicious_audio` | Unusual audio patterns | MEDIUM |
| `microphone_issue` | No audio for 30+ seconds | LOW |

## Performance Characteristics

### Latency Budget (per frame)
| Stage | Time | Notes |
|-------|------|-------|
| Frame decode | 5ms | LiveKit → numpy |
| YOLO11 inference | 25ms | yolo11n on GPU |
| MediaPipe | 20ms | Face mesh + pose |
| Feature extraction | 1ms | Sliding window calc |
| Classifier | 1ms | XGBoost predict |
| Alert publish | 2ms | Data channel |
| **Total** | **~54ms** | Supports 18 FPS |

### Audio Latency Budget (per 500ms chunk)
| Stage | Time | Notes |
|-------|------|-------|
| Audio decode | 1ms | LiveKit → numpy |
| VAD | 5ms | Silero inference |
| Speaker analysis | 10ms | Embedding compare |
| **Total** | **~16ms** | Real-time capable |

### Resource Usage
| Resource | Usage | Notes |
|----------|-------|-------|
| GPU Memory | ~1.5GB | YOLO11n + MediaPipe |
| CPU | 20-30% | Audio processing |
| RAM | ~2GB | Model weights + buffers |
| Network | 50-100 KB/s | Alerts only |

## Scaling Considerations

### Single Instance Capacity
- **10 concurrent sessions** per GPU (T4)
- **25 concurrent sessions** per GPU (A100)

### Horizontal Scaling
```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ ML Service #1   │ │ ML Service #2   │ │ ML Service #3   │
│ GPU: T4         │ │ GPU: T4         │ │ GPU: T4         │
│ Sessions: 10    │ │ Sessions: 10    │ │ Sessions: 10    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Session Assignment
- Use consistent hashing based on `room_id`
- Each session only processed by one instance
- Sticky sessions for state continuity

## Configuration

### Environment Variables
```bash
# LiveKit
LIVEKIT_URL=wss://your-server.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# Model Paths
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
YOLO_CONFIDENCE=0.5
FACE_CONFIDENCE=0.5
CHEATING_THRESHOLD=0.7
ALERT_COOLDOWN=5
```

## Deployment

### Docker
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

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: proctoring-ml-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: ml-service
        image: proctoring-ml-service:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 4Gi
          requests:
            memory: 2Gi
```

## Monitoring

### Key Metrics
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `processing_latency_ms` | Per-frame latency | > 100ms |
| `frames_processed_total` | Total frames | N/A |
| `alerts_published_total` | Alerts by type | N/A |
| `active_sessions` | Current sessions | > 80% capacity |
| `gpu_utilization` | GPU usage | > 90% |

### Health Check Endpoint
```bash
GET /health
{
  "status": "healthy",
  "models_loaded": true,
  "active_sessions": 5,
  "uptime_seconds": 3600
}
```
