"""
Frame Processing Pipeline
"""

import asyncio
import numpy as np
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field

from config import settings
from models import YOLODetector, MediaPipeAnalyzer, CheatingClassifier
from pipeline.features import FeatureExtractor
from pipeline.alerts import AlertGenerator


@dataclass
class ProcessingResult:
    """Result of processing a frame."""
    should_alert: bool = False
    alert_type: str | None = None
    severity: str = "low"
    cheating_probability: float = 0.0
    details: dict = field(default_factory=dict)


class FrameProcessor:
    """Main frame processing pipeline."""
    
    def __init__(self):
        """Initialize processing pipeline."""
        print("🔧 Initializing processing pipeline...")
        
        # Initialize models
        self.yolo = YOLODetector()
        self.mediapipe = MediaPipeAnalyzer()
        self.classifier = CheatingClassifier()
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(window_size=settings.sliding_window)
        
        # Alert generation
        self.alert_generator = AlertGenerator()
        
        # Per-participant data
        self.participant_data: dict[str, ParticipantState] = {}
        
        print("✅ Pipeline initialized")
    
    async def process_frame(
        self,
        frame: np.ndarray,
        participant_id: str,
    ) -> ProcessingResult | None:
        """
        Process a single frame.
        
        Args:
            frame: RGB image as numpy array
            participant_id: ID of the participant
            
        Returns:
            ProcessingResult if alert needed, None otherwise.
        """
        # Get or create participant state
        if participant_id not in self.participant_data:
            self.participant_data[participant_id] = ParticipantState()
        
        state = self.participant_data[participant_id]
        
        # Resize frame for processing
        frame = self._resize_frame(frame)
        
        # Run detection models in parallel
        yolo_result, face_result = await asyncio.gather(
            asyncio.to_thread(self.yolo.detect, frame),
            asyncio.to_thread(self.mediapipe.analyze, frame),
        )
        
        # Combine detections
        frame_data = {
            "timestamp": datetime.utcnow(),
            "yolo": yolo_result,
            "face": face_result,
        }
        
        # Add to history
        state.history.append(frame_data)
        
        # Extract features from sliding window
        features = self.feature_extractor.extract(list(state.history))
        
        # Run classifier
        classifier_result = self.classifier.predict(features)
        
        # Generate alert if needed
        alert = self.alert_generator.check(
            frame_data=frame_data,
            features=features,
            classifier_result=classifier_result,
            state=state,
        )
        
        if alert:
            return ProcessingResult(
                should_alert=True,
                alert_type=alert["type"],
                severity=alert["severity"],
                cheating_probability=classifier_result.cheating_probability,
                details={
                    "phone_detected": yolo_result["phone_detected"],
                    "face_count": face_result.face_count,
                    "looking_away": face_result.looking_away,
                    "face_visible": face_result.face_detected,
                    "factors": classifier_result.top_factors,
                },
            )
        
        return None
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to processing size."""
        import cv2
        
        h, w = frame.shape[:2]
        target_w, target_h = settings.frame_width, settings.frame_height
        
        if w != target_w or h != target_h:
            frame = cv2.resize(frame, (target_w, target_h))
        
        return frame
    
    def get_participant_summary(self, participant_id: str) -> dict:
        """Get summary for a participant."""
        if participant_id not in self.participant_data:
            return {}
        
        state = self.participant_data[participant_id]
        features = self.feature_extractor.extract(list(state.history))
        
        return {
            "participant_id": participant_id,
            "frames_processed": len(state.history),
            "alerts_sent": state.alert_count,
            "integrity_score": max(0, 100 - state.alert_count * 10),
            "features": features,
        }
    
    def close(self):
        """Cleanup resources."""
        self.mediapipe.close()


@dataclass
class ParticipantState:
    """State for a single participant."""
    history: deque = field(default_factory=lambda: deque(maxlen=settings.sliding_window))
    alert_count: int = 0
    last_alert_time: datetime | None = None
    last_alert_type: str | None = None
