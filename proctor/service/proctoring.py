from __future__ import annotations
"""
Proctoring Service

Main service that connects to LiveKit and orchestrates video/audio proctoring.
"""

import asyncio
import signal
from datetime import datetime
from typing import Optional

from livekit import rtc, api

from proctor.models import VideoProctor, AudioProctor
from proctor.data import VideoReceiver, AudioReceiver, AlertPublisher
from proctor.engine.results import Alert
from proctor.cfg import get_settings
from proctor.utils import get_logger

logger = get_logger(__name__)


class ProctoringService:
    """
    Main proctoring service.
    
    Connects to LiveKit rooms and processes video/audio streams
    for proctoring analysis.
    
    Example:
        >>> from proctor.service import ProctoringService
        >>> service = ProctoringService()
        >>> await service.connect("interview-room-123")
        >>> # Service runs until disconnected
        >>> await service.disconnect()
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize proctoring service.
        
        Args:
            verbose: Enable verbose output
        """
        self.settings = get_settings()
        self.verbose = verbose
        
        # LiveKit
        self.room: Optional[rtc.Room] = None
        self.publisher: Optional[AlertPublisher] = None
        
        # Models
        self.video_proctor: Optional[VideoProctor] = None
        self.audio_proctor: Optional[AudioProctor] = None
        
        # Receivers
        self.video_receiver: Optional[VideoReceiver] = None
        self.audio_receiver: Optional[AudioReceiver] = None
        
        # State
        self.running = False
        self.tasks: list[asyncio.Task] = []
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize proctoring components."""
        logger.info("🔧 Initializing proctoring components...")
        
        try:
            # Initialize models
            self.video_proctor = VideoProctor(verbose=self.verbose)
            self.audio_proctor = AudioProctor(verbose=self.verbose)
            
            # Force predictor initialization to catch errors early
            logger.info("🔧 Initializing video predictor...")
            _ = self.video_proctor.predictor
            logger.info("✅ Video predictor initialized")
            
            logger.info("🔧 Initializing audio predictor...")
            _ = self.audio_proctor.predictor
            logger.info("✅ Audio predictor initialized")
            
            # Initialize receivers
            self.video_receiver = VideoReceiver(
                target_fps=self.settings.process_fps,
                target_width=self.settings.frame_width,
                target_height=self.settings.frame_height,
            )
            
            self.audio_receiver = AudioReceiver(
                target_sample_rate=16000,
                chunk_duration_ms=500,
            )
            
            logger.info("✅ Components initialized")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Model file not found: {e}")
            logger.error("⛔ Cannot start service without trained model. Shutting down gracefully.")
            raise SystemExit(1)
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            logger.error("⛔ Shutting down gracefully due to initialization error.")
            raise SystemExit(1)
    
    async def connect(
        self,
        room_name: str,
        participant_identity: str = "ml-proctoring",
    ):
        """
        Connect to a LiveKit room.
        
        Args:
            room_name: Name of the room to join
            participant_identity: Identity for this service
        """
        # Generate token
        token = api.AccessToken(
            self.settings.livekit_api_key,
            self.settings.livekit_api_secret,
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
        
        # Create room and setup handlers
        self.room = rtc.Room()
        self._setup_handlers()
        
        # Connect
        await self.room.connect(self.settings.livekit_url, jwt)
        
        # Initialize publisher
        self.publisher = AlertPublisher(self.room)
        
        logger.info(f"✅ Connected to room: {room_name}")
        self.running = True
    
    def _setup_handlers(self):
        """Setup LiveKit event handlers."""
        
        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"📹 Subscribed to video from: {participant.identity}")
                task = asyncio.create_task(
                    self._process_video(track, participant.identity)
                )
                self.tasks.append(task)
            
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"🎤 Subscribed to audio from: {participant.identity}")
                task = asyncio.create_task(
                    self._process_audio(track, participant.identity)
                )
                self.tasks.append(task)
        
        @self.room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"📹 Unsubscribed from video: {participant.identity}")
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"🎤 Unsubscribed from audio: {participant.identity}")
                self.audio_receiver.clear_buffer(participant.identity)
        
        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            logger.info(f"👋 Participant left: {participant.identity}")
        
        @self.room.on("disconnected")
        def on_disconnected():
            logger.info("🔌 Disconnected from room")
            self.running = False
    
    async def _process_video(self, track: rtc.Track, participant_id: str):
        """Process video track frames."""
        async for video_frame in self.video_receiver.receive(track, participant_id):
            if not self.running:
                break
            
            try:
                # Ensure proctor is initialized
                if self.video_proctor is None:
                    logger.error("❌ Video proctor not initialized")
                    continue
                
                # Run video proctoring
                result = self.video_proctor.predict(video_frame.frame)
                
                # Publish alert if needed
                if result.should_alert and self.publisher:
                    alert = Alert(
                        type=result.alert_type or "unknown",
                        severity=result.alert_severity,
                        message="",
                        participant_id=participant_id,
                        source="video",
                        details={
                            "cheating_probability": result.cheating_probability,
                            "phone_detected": result.detection.phone_detected if result.detection else False,
                            "face_count": result.face.face_count if result.face else 0,
                            "looking_away": result.face.looking_away if result.face else False,
                            "factors": result.top_factors,
                        },
                    )
                    await self.publisher.publish(alert)
                    
            except Exception as e:
                logger.error(f"❌ Video processing error: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    async def _process_audio(self, track: rtc.Track, participant_id: str):
        """Process audio track for proctoring analysis."""
        async for audio_chunk in self.audio_receiver.receive(track, participant_id):
            if not self.running:
                break
            
            try:
                # Run audio proctoring
                result = self.audio_proctor.predict(audio_chunk.samples)
                
                # Publish alert if needed
                if result.should_alert and self.publisher:
                    alert = Alert(
                        type=result.alert_type or "unknown",
                        severity=result.alert_severity,
                        message="",
                        participant_id=participant_id,
                        source="audio",
                        details={
                            "voice_detected": result.voice_detected,
                            "speaker_count": result.speaker_count,
                            "noise_level": result.noise_level,
                        },
                    )
                    await self.publisher.publish(alert)
                    
            except Exception as e:
                logger.error(f"❌ Audio processing error: {e}")
    
    async def disconnect(self):
        """Disconnect from room and cleanup."""
        self.running = False
        
        # Stop receivers
        if self.video_receiver:
            self.video_receiver.stop()
        if self.audio_receiver:
            self.audio_receiver.stop()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Disconnect from room
        if self.room:
            await self.room.disconnect()
        
        # Cleanup models
        if self.video_proctor:
            self.video_proctor.close()
        
        logger.info("👋 Disconnected from LiveKit")
    
    async def run(self, room_name: str):
        """
        Run the proctoring service.
        
        Args:
            room_name: Room to connect to
        """
        await self.connect(room_name)
        
        # Keep running until disconnected
        while self.running:
            await asyncio.sleep(1)


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
        await service.run(room_name)
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await service.disconnect()


if __name__ == "__main__":
    print("🚀 Starting Proctoring ML Service...")
    asyncio.run(main())
