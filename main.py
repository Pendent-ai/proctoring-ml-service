"""
Proctoring ML Service - Main Entry Point

Subscribes to LiveKit video tracks and publishes proctoring alerts.
"""

import asyncio
import json
import signal
from datetime import datetime
from livekit import rtc
from livekit import api

from config import settings
from pipeline.processor import FrameProcessor
from livekit_client.publisher import AlertPublisher


class ProctoringService:
    """Main proctoring service that connects to LiveKit and processes video."""
    
    def __init__(self):
        self.room = rtc.Room()
        self.processor = FrameProcessor()
        self.publisher: AlertPublisher | None = None
        self.running = False
        self.tasks: list[asyncio.Task] = []
    
    async def connect(self, room_name: str, participant_identity: str = "ml-proctoring"):
        """Connect to a LiveKit room."""
        # Generate token
        token = api.AccessToken(
            settings.livekit_api_key,
            settings.livekit_api_secret,
        )
        token.with_identity(participant_identity)
        token.with_name("ML Proctoring Service")
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_subscribe=True,
            can_publish=True,
            can_publish_data=True,
        ))
        
        jwt = token.to_jwt()
        
        # Setup event handlers
        self._setup_handlers()
        
        # Connect to room
        await self.room.connect(settings.livekit_url, jwt)
        
        # Initialize publisher
        self.publisher = AlertPublisher(self.room)
        
        print(f"✅ Connected to room: {room_name}")
        self.running = True
    
    def _setup_handlers(self):
        """Setup LiveKit event handlers."""
        
        @self.room.on("track_subscribed")
        async def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                print(f"📹 Subscribed to video from: {participant.identity}")
                task = asyncio.create_task(
                    self._process_video_track(track, participant.identity)
                )
                self.tasks.append(task)
        
        @self.room.on("track_unsubscribed")
        async def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            print(f"📹 Unsubscribed from video: {participant.identity}")
        
        @self.room.on("participant_disconnected")
        async def on_participant_disconnected(participant: rtc.RemoteParticipant):
            print(f"👋 Participant left: {participant.identity}")
        
        @self.room.on("disconnected")
        async def on_disconnected():
            print("🔌 Disconnected from room")
            self.running = False
    
    async def _process_video_track(
        self,
        track: rtc.Track,
        participant_id: str,
    ):
        """Process video track frames."""
        video_stream = rtc.VideoStream(track)
        frame_count = 0
        skip_frames = 30 // settings.process_fps  # Process at target FPS
        
        async for frame_event in video_stream:
            if not self.running:
                break
            
            frame_count += 1
            
            # Skip frames to maintain target FPS
            if frame_count % skip_frames != 0:
                continue
            
            try:
                # Convert frame to numpy
                frame = self._frame_to_numpy(frame_event.frame)
                
                # Process frame
                result = await self.processor.process_frame(frame, participant_id)
                
                # Publish alert if needed
                if result and result.get("should_alert") and self.publisher:
                    await self.publisher.publish_alert({
                        "participant_id": participant_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": result["alert_type"],
                        "severity": result["severity"],
                        "cheating_probability": result["cheating_probability"],
                        "details": result.get("details", {}),
                    })
                    
            except Exception as e:
                print(f"❌ Frame processing error: {e}")
    
    def _frame_to_numpy(self, frame: rtc.VideoFrame):
        """Convert LiveKit frame to numpy array."""
        import numpy as np
        
        # Get frame buffer
        buffer = frame.convert(rtc.VideoBufferType.RGB24)
        
        # Create numpy array
        arr = np.frombuffer(buffer.data, dtype=np.uint8)
        arr = arr.reshape((buffer.height, buffer.width, 3))
        
        return arr
    
    async def disconnect(self):
        """Disconnect from room and cleanup."""
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        await self.room.disconnect()
        print("👋 Disconnected from LiveKit")


async def main():
    """Main entry point."""
    import sys
    
    # Get room name from args or env
    room_name = sys.argv[1] if len(sys.argv) > 1 else "test-room"
    
    service = ProctoringService()
    
    # Handle shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        print("\n⚠️ Shutting down...")
        asyncio.create_task(service.disconnect())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        await service.connect(room_name)
        
        # Keep running
        while service.running:
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await service.disconnect()


if __name__ == "__main__":
    print("🚀 Starting Proctoring ML Service...")
    asyncio.run(main())
