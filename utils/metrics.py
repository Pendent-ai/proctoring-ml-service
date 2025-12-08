"""
Metrics Collection for Proctoring Service
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class ProcessingMetrics:
    """Metrics for a single participant."""
    frames_processed: int = 0
    alerts_sent: int = 0
    total_processing_time_ms: float = 0.0
    detections: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    alert_history: List[dict] = field(default_factory=list)
    
    @property
    def avg_processing_time_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.frames_processed


class MetricsCollector:
    """Collects and aggregates service metrics."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.participants: Dict[str, ProcessingMetrics] = {}
        
        # Global counters
        self.total_frames = 0
        self.total_alerts = 0
        self.total_connections = 0
    
    def get_participant_metrics(self, participant_id: str) -> ProcessingMetrics:
        """Get or create metrics for a participant."""
        if participant_id not in self.participants:
            self.participants[participant_id] = ProcessingMetrics()
        return self.participants[participant_id]
    
    def record_frame(
        self,
        participant_id: str,
        processing_time_ms: float,
        detections: dict,
    ):
        """Record frame processing metrics."""
        metrics = self.get_participant_metrics(participant_id)
        
        metrics.frames_processed += 1
        metrics.total_processing_time_ms += processing_time_ms
        
        # Track detection counts
        if detections.get("phone_detected"):
            metrics.detections["phone"] += 1
        if detections.get("face_count", 0) > 1:
            metrics.detections["multiple_faces"] += 1
        if detections.get("looking_away"):
            metrics.detections["looking_away"] += 1
        if not detections.get("face_detected", True):
            metrics.detections["face_not_visible"] += 1
        
        self.total_frames += 1
    
    def record_alert(
        self,
        participant_id: str,
        alert_type: str,
        severity: str,
        probability: float,
    ):
        """Record an alert."""
        metrics = self.get_participant_metrics(participant_id)
        
        metrics.alerts_sent += 1
        metrics.alert_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": alert_type,
            "severity": severity,
            "probability": probability,
        })
        
        self.total_alerts += 1
    
    def record_connection(self, participant_id: str):
        """Record a new participant connection."""
        self.total_connections += 1
        self.get_participant_metrics(participant_id)
    
    def get_summary(self) -> dict:
        """Get overall service summary."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "total_participants": len(self.participants),
            "total_connections": self.total_connections,
            "total_frames_processed": self.total_frames,
            "total_alerts_sent": self.total_alerts,
            "frames_per_second": self.total_frames / uptime if uptime > 0 else 0,
        }
    
    def get_participant_summary(self, participant_id: str) -> dict:
        """Get summary for a specific participant."""
        if participant_id not in self.participants:
            return {}
        
        metrics = self.participants[participant_id]
        
        return {
            "participant_id": participant_id,
            "frames_processed": metrics.frames_processed,
            "alerts_sent": metrics.alerts_sent,
            "avg_processing_time_ms": metrics.avg_processing_time_ms,
            "detection_counts": dict(metrics.detections),
            "integrity_score": self._calculate_integrity_score(metrics),
        }
    
    def _calculate_integrity_score(self, metrics: ProcessingMetrics) -> float:
        """
        Calculate an integrity score based on metrics.
        
        Returns:
            Score from 0-100, where 100 is perfect.
        """
        if metrics.frames_processed == 0:
            return 100.0
        
        score = 100.0
        
        # Deduct for alerts
        score -= metrics.alerts_sent * 5
        
        # Deduct for detection ratios
        phone_ratio = metrics.detections["phone"] / metrics.frames_processed
        score -= phone_ratio * 50
        
        multi_face_ratio = metrics.detections["multiple_faces"] / metrics.frames_processed
        score -= multi_face_ratio * 30
        
        looking_away_ratio = metrics.detections["looking_away"] / metrics.frames_processed
        score -= looking_away_ratio * 10
        
        return max(0.0, min(100.0, score))
