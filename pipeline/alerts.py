"""
Alert Generation Logic
"""

from datetime import datetime, timedelta
from dataclasses import dataclass

from config import settings
from models.classifier import ClassifierResult


@dataclass
class AlertConfig:
    """Configuration for a specific alert type."""
    threshold: float  # Feature threshold to trigger
    severity_medium: float  # Threshold for medium severity
    severity_high: float  # Threshold for high severity
    cooldown: int  # Seconds between alerts


class AlertGenerator:
    """Generate alerts based on detection results."""
    
    # Alert configurations
    ALERT_CONFIGS = {
        "phone_detected": AlertConfig(
            threshold=0.1,
            severity_medium=0.2,
            severity_high=0.5,
            cooldown=10,
        ),
        "multiple_faces": AlertConfig(
            threshold=0.05,
            severity_medium=0.1,
            severity_high=0.3,
            cooldown=15,
        ),
        "looking_away": AlertConfig(
            threshold=0.4,
            severity_medium=0.6,
            severity_high=0.8,
            cooldown=20,
        ),
        "face_not_visible": AlertConfig(
            threshold=0.2,  # 20% of frames without face
            severity_medium=0.4,
            severity_high=0.6,
            cooldown=10,
        ),
    }
    
    def __init__(self):
        """Initialize alert generator."""
        self.last_alerts: dict[str, dict[str, datetime]] = {}
    
    def check(
        self,
        frame_data: dict,
        features: dict,
        classifier_result: ClassifierResult,
        state: "ParticipantState",
    ) -> dict | None:
        """
        Check if an alert should be generated.
        
        Args:
            frame_data: Current frame detection results
            features: Extracted features from sliding window
            classifier_result: Classifier output
            state: Participant state
            
        Returns:
            Alert dictionary if alert needed, None otherwise.
        """
        participant_id = id(state)
        
        if participant_id not in self.last_alerts:
            self.last_alerts[participant_id] = {}
        
        last_alerts = self.last_alerts[participant_id]
        
        # Check each alert type
        alerts = []
        
        # Phone detected
        if frame_data["yolo"]["phone_detected"]:
            alert = self._check_alert(
                "phone_detected",
                features["phone_detected_ratio"],
                last_alerts,
            )
            if alert:
                alerts.append(alert)
        
        # Multiple faces
        if frame_data["face"].face_count > 1:
            alert = self._check_alert(
                "multiple_faces",
                features["multiple_faces_ratio"],
                last_alerts,
            )
            if alert:
                alerts.append(alert)
        
        # Looking away
        if frame_data["face"].looking_away:
            alert = self._check_alert(
                "looking_away",
                features["gaze_away_ratio"],
                last_alerts,
            )
            if alert:
                alerts.append(alert)
        
        # Face not visible
        if not frame_data["face"].face_detected:
            alert = self._check_alert(
                "face_not_visible",
                1.0 - features["face_visible_ratio"],
                last_alerts,
            )
            if alert:
                alerts.append(alert)
        
        # Return highest severity alert
        if alerts:
            alerts.sort(key=lambda a: {"low": 0, "medium": 1, "high": 2}[a["severity"]], reverse=True)
            best_alert = alerts[0]
            
            # Update state
            state.alert_count += 1
            state.last_alert_time = datetime.utcnow()
            state.last_alert_type = best_alert["type"]
            
            return best_alert
        
        return None
    
    def _check_alert(
        self,
        alert_type: str,
        value: float,
        last_alerts: dict[str, datetime],
    ) -> dict | None:
        """Check if specific alert should be triggered."""
        config = self.ALERT_CONFIGS.get(alert_type)
        if not config:
            return None
        
        # Check threshold
        if value < config.threshold:
            return None
        
        # Check cooldown
        last_time = last_alerts.get(alert_type)
        if last_time:
            elapsed = (datetime.utcnow() - last_time).total_seconds()
            if elapsed < config.cooldown:
                return None
        
        # Determine severity
        if value >= config.severity_high:
            severity = "high"
        elif value >= config.severity_medium:
            severity = "medium"
        else:
            severity = "low"
        
        # Update last alert time
        last_alerts[alert_type] = datetime.utcnow()
        
        return {
            "type": alert_type,
            "severity": severity,
            "value": value,
        }
