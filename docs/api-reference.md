# API Reference

## Overview

The Proctoring ML Service communicates via LiveKit Data Channel. It publishes alerts that can be consumed by the Node.js backend or any LiveKit client.

## Data Channel Topics

### `proctoring` (Outgoing)

All proctoring alerts are published to this topic.

```typescript
// Subscribe in Node.js backend
room.on('dataReceived', (payload, participant, topic) => {
  if (topic === 'proctoring') {
    const alert = JSON.parse(payload.toString());
    handleProctoringAlert(alert);
  }
});
```

## Alert Payload Schema

### Base Alert

```json
{
  "participant_id": "user_123",
  "timestamp": "2024-12-08T10:30:00.000Z",
  "type": "phone_detected",
  "severity": "high",
  "cheating_probability": 0.85,
  "source": "video",
  "details": {
    // Alert-specific details
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `participant_id` | string | LiveKit participant identity |
| `timestamp` | string | ISO 8601 timestamp |
| `type` | AlertType | Type of alert (see below) |
| `severity` | AlertSeverity | low, medium, high, critical |
| `cheating_probability` | float | 0.0 - 1.0 probability score |
| `source` | string | "video" or "audio" |
| `details` | object | Alert-specific data |

## Alert Types

### Video Alerts

#### `phone_detected`

Phone visible in the frame.

```json
{
  "type": "phone_detected",
  "severity": "high",
  "details": {
    "phone_confidence": 0.92,
    "phone_bbox": [100, 200, 150, 280],
    "phone_position": "hand"
  }
}
```

#### `multiple_faces`

More than one face detected in frame.

```json
{
  "type": "multiple_faces",
  "severity": "high",
  "details": {
    "face_count": 2,
    "primary_face_bbox": [120, 80, 320, 350],
    "other_faces": [
      {"bbox": [400, 100, 500, 250], "confidence": 0.88}
    ]
  }
}
```

#### `looking_away`

Candidate consistently looking away from screen.

```json
{
  "type": "looking_away",
  "severity": "medium",
  "details": {
    "gaze_away_ratio": 0.65,
    "gaze_direction": {"x": -0.8, "y": 0.2},
    "duration_seconds": 4.2
  }
}
```

#### `face_not_visible`

No face detected in frame.

```json
{
  "type": "face_not_visible",
  "severity": "medium",
  "details": {
    "face_visible_ratio": 0.15,
    "last_face_seen": "2024-12-08T10:29:55.000Z",
    "duration_seconds": 5.0
  }
}
```

#### `suspicious_behavior`

High cheating probability without specific trigger.

```json
{
  "type": "suspicious_behavior",
  "severity": "medium",
  "details": {
    "top_factors": ["gaze_variance", "head_movement_freq"],
    "feature_scores": {
      "gaze_variance": 0.85,
      "head_movement_freq": 0.72
    }
  }
}
```

### Audio Alerts

#### `multiple_voices`

Multiple distinct speakers detected.

```json
{
  "type": "multiple_voices",
  "severity": "high",
  "source": "audio",
  "details": {
    "speaker_count": 2,
    "primary_speaker_confidence": 0.95,
    "duration_seconds": 3.5
  }
}
```

#### `background_speech`

Speech detected from off-screen source.

```json
{
  "type": "background_speech",
  "severity": "medium",
  "source": "audio",
  "details": {
    "background_voice_confidence": 0.78,
    "primary_voice_active": true
  }
}
```

#### `whispering`

Low-volume speech detected (potential coaching).

```json
{
  "type": "whispering",
  "severity": "medium",
  "source": "audio",
  "details": {
    "audio_level_db": -32.5,
    "voice_confidence": 0.88,
    "duration_seconds": 2.1
  }
}
```

#### `suspicious_audio`

Unusual audio patterns detected.

```json
{
  "type": "suspicious_audio",
  "severity": "medium",
  "source": "audio",
  "details": {
    "pattern_type": "text_to_speech",
    "confidence": 0.72
  }
}
```

#### `microphone_issue`

No audio received for extended period.

```json
{
  "type": "microphone_issue",
  "severity": "low",
  "source": "audio",
  "details": {
    "silence_duration_seconds": 45,
    "last_audio_received": "2024-12-08T10:29:15.000Z"
  }
}
```

## Alert Severity Levels

| Level | Description | Response |
|-------|-------------|----------|
| `low` | Minor anomaly, may be false positive | Log only |
| `medium` | Noticeable violation, gentle warning | AI issues gentle reminder |
| `high` | Significant violation | AI issues serious warning |
| `critical` | Major violation, likely cheating | Consider termination |

## Alert Messages

The Node.js backend can use these pre-defined messages for the AI interviewer:

### Phone Detected

```javascript
const PHONE_MESSAGES = {
  gentle: "I noticed something in your hands. Please make sure your desk is clear and focus on the interview.",
  serious: "I need to remind you that using external devices is not permitted during this interview.",
  final: "This is a final warning. Any further use of phones or devices will terminate this interview."
};
```

### Multiple Faces

```javascript
const MULTIPLE_FACES_MESSAGES = {
  gentle: "I'm seeing multiple faces in the frame. Please ensure you're alone in the room.",
  serious: "This interview requires you to be alone. Please ensure no one else is visible.",
  final: "I've detected multiple people several times. This is your final warning."
};
```

### Looking Away

```javascript
const LOOKING_AWAY_MESSAGES = {
  gentle: "I've noticed you looking away from the screen quite a bit. Please try to maintain focus on the interview.",
  serious: "Please keep your attention on the screen. Looking away frequently affects the interview.",
  final: "Excessive looking away may indicate outside assistance. Please focus on the screen."
};
```

### Audio Alerts

```javascript
const AUDIO_MESSAGES = {
  multiple_voices: {
    gentle: "I'm picking up some background voices. Please ensure you're in a quiet environment.",
    serious: "I'm detecting another person speaking. Please confirm you're alone."
  },
  whispering: {
    gentle: "I noticed some quiet sounds. Please speak clearly so I can hear you well.",
    serious: "Please speak at a normal volume so we can continue the interview properly."
  }
};
```

## Session Summary

At the end of an interview, request a session summary:

### Request (via LiveKit Data Channel)

```json
{
  "action": "get_summary",
  "participant_id": "user_123"
}
```

### Response

```json
{
  "participant_id": "user_123",
  "session_duration_seconds": 1800,
  "frames_processed": 18000,
  "audio_chunks_processed": 3600,
  
  "integrity_score": 85,
  
  "alerts_summary": {
    "total": 5,
    "by_type": {
      "looking_away": 3,
      "phone_detected": 1,
      "background_speech": 1
    },
    "by_severity": {
      "low": 1,
      "medium": 3,
      "high": 1
    }
  },
  
  "feature_averages": {
    "gaze_away_ratio": 0.15,
    "face_visible_ratio": 0.95,
    "phone_detected_ratio": 0.02,
    "multiple_faces_ratio": 0.00
  },
  
  "timeline": [
    {
      "timestamp": "2024-12-08T10:05:00.000Z",
      "type": "looking_away",
      "severity": "medium"
    },
    {
      "timestamp": "2024-12-08T10:12:30.000Z",
      "type": "phone_detected",
      "severity": "high"
    }
  ]
}
```

## Error Handling

### Error Payload

```json
{
  "error": true,
  "code": "MODEL_LOAD_FAILED",
  "message": "Failed to load YOLO11 model",
  "timestamp": "2024-12-08T10:30:00.000Z"
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `MODEL_LOAD_FAILED` | Failed to load ML model |
| `FRAME_DECODE_ERROR` | Could not decode video frame |
| `AUDIO_DECODE_ERROR` | Could not decode audio chunk |
| `PROCESSING_TIMEOUT` | Processing took too long |
| `GPU_OOM` | GPU out of memory |

## Rate Limiting

| Alert Type | Cooldown (seconds) |
|------------|-------------------|
| `phone_detected` | 10 |
| `multiple_faces` | 15 |
| `looking_away` | 20 |
| `face_not_visible` | 10 |
| `multiple_voices` | 15 |
| `background_speech` | 20 |
| `whispering` | 20 |

Alerts of the same type for the same participant will not be published more frequently than the cooldown period.
