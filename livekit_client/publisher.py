"""
LiveKit Client - Alert Publisher
"""

import json
from datetime import datetime
from livekit import rtc


class AlertPublisher:
    """Publishes proctoring alerts via LiveKit data channel."""
    
    TOPIC = "proctoring"
    
    def __init__(self, room: rtc.Room):
        self.room = room
        self.last_alert_time: dict[str, datetime] = {}
        self.cooldown_seconds = 5  # Minimum seconds between alerts of same type
    
    async def publish_alert(self, alert: dict) -> bool:
        """
        Publish a proctoring alert to the data channel.
        
        Args:
            alert: Alert data containing type, severity, details, etc.
            
        Returns:
            True if alert was published, False if rate-limited.
        """
        alert_key = f"{alert.get('participant_id')}:{alert.get('type')}"
        
        # Check cooldown
        if not self._check_cooldown(alert_key):
            return False
        
        # Prepare payload
        payload = json.dumps(alert).encode("utf-8")
        
        # Publish to data channel
        await self.room.local_participant.publish_data(
            payload=payload,
            topic=self.TOPIC,
            reliable=True,
        )
        
        print(f"📤 Alert published: {alert.get('type')} ({alert.get('severity')})")
        
        # Update last alert time
        self.last_alert_time[alert_key] = datetime.utcnow()
        
        return True
    
    def _check_cooldown(self, alert_key: str) -> bool:
        """Check if enough time has passed since last alert of this type."""
        last_time = self.last_alert_time.get(alert_key)
        
        if last_time is None:
            return True
        
        elapsed = (datetime.utcnow() - last_time).total_seconds()
        return elapsed >= self.cooldown_seconds
    
    async def publish_status(self, participant_id: str, status: dict):
        """Publish periodic status update."""
        payload = json.dumps({
            "type": "status_update",
            "participant_id": participant_id,
            "timestamp": datetime.utcnow().isoformat(),
            **status,
        }).encode("utf-8")
        
        await self.room.local_participant.publish_data(
            payload=payload,
            topic="proctoring_status",
            reliable=False,
        )
