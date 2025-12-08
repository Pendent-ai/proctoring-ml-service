"""
Alert Types and Constants

Defines all proctoring alert types, severities, and response messages.
"""

from enum import Enum
from dataclasses import dataclass


class AlertType(str, Enum):
    """Types of proctoring alerts."""
    PHONE_DETECTED = "phone_detected"
    MULTIPLE_FACES = "multiple_faces"
    LOOKING_AWAY = "looking_away"
    FACE_NOT_VISIBLE = "face_not_visible"
    TAB_SWITCH = "tab_switch"
    AUDIO_ANOMALY = "audio_anomaly"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertMessage:
    """Alert message for different severities."""
    gentle: str
    serious: str
    final: str


# Alert messages sent to candidates via AI interviewer
ALERT_MESSAGES = {
    AlertType.PHONE_DETECTED: AlertMessage(
        gentle="I noticed something in your hands. Please make sure your desk is clear and focus on the interview.",
        serious="I need to remind you that using external devices is not permitted during this interview.",
        final="This is a final warning. Any further use of phones or devices will terminate this interview.",
    ),
    AlertType.MULTIPLE_FACES: AlertMessage(
        gentle="I'm seeing multiple faces in the frame. Please ensure you're alone in the room.",
        serious="This interview requires you to be alone. Please ensure no one else is visible.",
        final="I've detected multiple people several times. This is your final warning.",
    ),
    AlertType.LOOKING_AWAY: AlertMessage(
        gentle="I've noticed you looking away from the screen quite a bit. Please try to maintain focus on the interview.",
        serious="Please keep your attention on the screen. Looking away frequently affects the interview.",
        final="Consistent looking away has been noted. Please maintain focus going forward.",
    ),
    AlertType.FACE_NOT_VISIBLE: AlertMessage(
        gentle="I'm having trouble seeing you clearly. Could you adjust your camera?",
        serious="Your face needs to be visible throughout the interview. Please adjust your position.",
        final="Your face must be visible at all times. This is a final reminder.",
    ),
}


def get_alert_message(alert_type: AlertType, warning_level: int) -> str:
    """
    Get appropriate alert message based on type and warning level.
    
    Args:
        alert_type: Type of alert
        warning_level: 1=gentle, 2=serious, 3+=final
        
    Returns:
        Alert message string
    """
    messages = ALERT_MESSAGES.get(alert_type)
    if not messages:
        return "Please maintain proper interview conduct."
    
    if warning_level <= 1:
        return messages.gentle
    elif warning_level == 2:
        return messages.serious
    else:
        return messages.final


def get_severity_for_probability(
    probability: float,
    is_critical_type: bool = False,
) -> AlertSeverity:
    """
    Determine severity based on cheating probability.
    
    Args:
        probability: Cheating probability 0-1
        is_critical_type: Whether this is a critical alert type (phone, multiple faces)
        
    Returns:
        AlertSeverity
    """
    if is_critical_type and probability > 0.5:
        return AlertSeverity.CRITICAL
    
    if probability >= 0.8:
        return AlertSeverity.HIGH
    elif probability >= 0.5:
        return AlertSeverity.MEDIUM
    else:
        return AlertSeverity.LOW
