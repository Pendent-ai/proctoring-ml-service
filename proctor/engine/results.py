"""
Proctor Engine - Results Classes

Data classes for storing prediction results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List
from datetime import datetime
import numpy as np


@dataclass
class Results:
    """
    Base class for prediction results.
    
    Attributes:
        timestamp: When the prediction was made
        source: Original input reference
        speed: Inference timing information
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: Any = None
    speed: dict = field(default_factory=lambda: {
        "preprocess": 0.0,
        "inference": 0.0,
        "postprocess": 0.0,
    })
    
    @property
    def total_time_ms(self) -> float:
        """Total processing time in milliseconds."""
        return sum(self.speed.values())
    
    def to_dict(self) -> dict:
        """Convert results to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "speed_ms": self.speed,
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class DetectionResults(Results):
    """
    Results from object detection (YOLO).
    
    Attributes:
        boxes: Detected bounding boxes
        person_count: Number of persons detected
        phone_detected: Whether phone was detected
        laptop_detected: Whether laptop was detected
    """
    boxes: list = field(default_factory=list)
    person_count: int = 0
    phone_detected: bool = False
    phone_boxes: list = field(default_factory=list)
    laptop_detected: bool = False
    book_detected: bool = False
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "person_count": self.person_count,
            "phone_detected": self.phone_detected,
            "laptop_detected": self.laptop_detected,
            "book_detected": self.book_detected,
            "boxes": [
                {
                    "class": b.get("class"),
                    "confidence": b.get("confidence"),
                    "box": b.get("box"),
                }
                for b in self.boxes
            ],
        })
        return base


@dataclass
class FaceResults(Results):
    """
    Results from face analysis (MediaPipe).
    
    Attributes:
        face_detected: Whether a face was detected
        face_count: Number of faces detected
        gaze: Gaze direction (x, y)
        head_pose: Head pose (pitch, yaw, roll)
        looking_away: Whether person is looking away
        eyes_closed: Whether eyes are closed
    """
    face_detected: bool = False
    face_count: int = 0
    face_box: list | None = None
    landmarks: list | None = None
    
    # Gaze
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    looking_away: bool = False
    
    # Head pose
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    
    # Eyes
    left_eye_open: float = 1.0
    right_eye_open: float = 1.0
    eyes_closed: bool = False
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "face_detected": self.face_detected,
            "face_count": self.face_count,
            "gaze": {"x": self.gaze_x, "y": self.gaze_y},
            "head_pose": {
                "pitch": self.pitch,
                "yaw": self.yaw,
                "roll": self.roll,
            },
            "looking_away": self.looking_away,
            "eyes_closed": self.eyes_closed,
        })
        return base


@dataclass
class AnalysisResults(Results):
    """
    Combined video analysis results.
    
    Combines detection and face analysis with classifier prediction.
    """
    detection: DetectionResults | None = None
    face: FaceResults | None = None
    
    # Classifier
    cheating_probability: float = 0.0
    is_cheating: bool = False
    confidence: float = 0.0
    top_factors: list = field(default_factory=list)
    
    # Alert
    should_alert: bool = False
    alert_type: str | None = None
    alert_severity: str = "low"
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "detection": self.detection.to_dict() if self.detection else None,
            "face": self.face.to_dict() if self.face else None,
            "cheating_probability": self.cheating_probability,
            "is_cheating": self.is_cheating,
            "confidence": self.confidence,
            "top_factors": self.top_factors,
            "should_alert": self.should_alert,
            "alert_type": self.alert_type,
            "alert_severity": self.alert_severity,
        })
        return base


@dataclass
class AudioResults(Results):
    """
    Results from audio analysis.
    
    Attributes:
        voice_detected: Whether voice activity was detected
        speaker_count: Estimated number of speakers
        noise_level: Background noise level
    """
    # Voice activity
    voice_detected: bool = False
    voice_confidence: float = 0.0
    voice_duration_ms: int = 0
    
    # Speakers
    speaker_count: int = 1
    multiple_speakers: bool = False
    speaker_change_detected: bool = False
    
    # Anomalies
    whispering_detected: bool = False
    background_voice: bool = False
    suspicious_sounds: list = field(default_factory=list)
    
    # Audio quality
    audio_level_db: float = -60.0
    is_silent: bool = True
    noise_level: float = 0.0
    
    # Alert
    should_alert: bool = False
    alert_type: str | None = None
    alert_severity: str = "low"
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "voice_detected": self.voice_detected,
            "voice_confidence": self.voice_confidence,
            "speaker_count": self.speaker_count,
            "multiple_speakers": self.multiple_speakers,
            "whispering_detected": self.whispering_detected,
            "background_voice": self.background_voice,
            "audio_level_db": self.audio_level_db,
            "noise_level": self.noise_level,
            "should_alert": self.should_alert,
            "alert_type": self.alert_type,
            "alert_severity": self.alert_severity,
        })
        return base


@dataclass
class Alert:
    """
    Proctoring alert to be published.
    
    Attributes:
        type: Alert type (e.g., "phone_detected", "multiple_voices")
        severity: Alert severity (low, medium, high, critical)
        message: Human-readable alert message
        participant_id: ID of the participant
    """
    type: str
    severity: str
    message: str
    participant_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "video"  # "video" or "audio"
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "participant_id": self.participant_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "details": self.details,
        }
