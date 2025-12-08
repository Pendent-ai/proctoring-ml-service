"""
Alert Publisher

Publishes proctoring alerts via LiveKit data channel.
"""

import json
from datetime import datetime
from typing import Optional

from proctor.engine.results import Alert

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False


class AlertPublisher:
    """
    Publishes proctoring alerts via LiveKit data channel.
    
    Handles rate limiting to prevent alert spam.
    
    Example:
        >>> publisher = AlertPublisher(room)
        >>> await publisher.publish(alert)
    """
    
    # Data channel topics
    ALERT_TOPIC = "proctoring"
    STATUS_TOPIC = "proctoring_status"
    
    def __init__(
        self,
        room: "rtc.Room",
        cooldown_seconds: float = 5.0,
    ):
        """
        Initialize alert publisher.
        
        Args:
            room: LiveKit room instance
            cooldown_seconds: Minimum seconds between same alert types
        """
        if not LIVEKIT_AVAILABLE:
            raise ImportError("livekit package required: pip install livekit")
        
        self.room = room
        self.cooldown_seconds = cooldown_seconds
        
        # Track last alert times for rate limiting
        self._last_alert_times: dict[str, datetime] = {}
    
    async def publish(self, alert: Alert) -> bool:
        """
        Publish a proctoring alert.
        
        Args:
            alert: Alert to publish
            
        Returns:
            True if published, False if rate-limited
        """
        # Create unique key for rate limiting
        alert_key = f"{alert.participant_id}:{alert.type}"
        
        # Check cooldown
        if not self._check_cooldown(alert_key):
            return False
        
        # Prepare payload
        payload = json.dumps(alert.to_dict()).encode("utf-8")
        
        # Publish to data channel
        await self.room.local_participant.publish_data(
            payload=payload,
            topic=self.ALERT_TOPIC,
            reliable=True,
        )
        
        # Update last alert time
        self._last_alert_times[alert_key] = datetime.utcnow()
        
        print(f"📤 Alert published: {alert.type} ({alert.severity})")
        
        return True
    
    async def publish_dict(self, alert_data: dict) -> bool:
        """
        Publish alert from dictionary.
        
        Args:
            alert_data: Alert data dictionary
            
        Returns:
            True if published, False if rate-limited
        """
        alert = Alert(
            type=alert_data.get("type", "unknown"),
            severity=alert_data.get("severity", "low"),
            message=alert_data.get("message", ""),
            participant_id=alert_data.get("participant_id", ""),
            source=alert_data.get("source", "video"),
            details=alert_data.get("details", {}),
        )
        
        return await self.publish(alert)
    
    async def publish_status(
        self,
        participant_id: str,
        status: dict,
    ):
        """
        Publish periodic status update.
        
        Args:
            participant_id: ID of the participant
            status: Status data dictionary
        """
        payload = json.dumps({
            "type": "status_update",
            "participant_id": participant_id,
            "timestamp": datetime.utcnow().isoformat(),
            **status,
        }).encode("utf-8")
        
        await self.room.local_participant.publish_data(
            payload=payload,
            topic=self.STATUS_TOPIC,
            reliable=False,
        )
    
    def _check_cooldown(self, alert_key: str) -> bool:
        """Check if enough time has passed since last alert."""
        last_time = self._last_alert_times.get(alert_key)
        
        if last_time is None:
            return True
        
        elapsed = (datetime.utcnow() - last_time).total_seconds()
        return elapsed >= self.cooldown_seconds
    
    def reset_cooldown(self, participant_id: Optional[str] = None):
        """Reset alert cooldowns."""
        if participant_id:
            # Reset only for specific participant
            keys_to_remove = [
                k for k in self._last_alert_times
                if k.startswith(f"{participant_id}:")
            ]
            for key in keys_to_remove:
                del self._last_alert_times[key]
        else:
            self._last_alert_times.clear()
