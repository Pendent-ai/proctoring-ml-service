"""
LiveKit Frame Receiver

Handles receiving and decoding video frames from LiveKit tracks.
"""

import asyncio
import numpy as np
from livekit import rtc
from typing import AsyncGenerator, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ReceivedFrame:
    """A received video frame with metadata."""
    frame: np.ndarray
    timestamp: datetime
    participant_id: str
    frame_number: int
    width: int
    height: int


class FrameReceiver:
    """Receives and processes video frames from LiveKit tracks."""
    
    def __init__(
        self,
        target_fps: int = 10,
        source_fps: int = 30,
    ):
        """
        Initialize frame receiver.
        
        Args:
            target_fps: Target processing FPS
            source_fps: Expected source video FPS
        """
        self.target_fps = target_fps
        self.source_fps = source_fps
        self.skip_frames = max(1, source_fps // target_fps)
        
        self.frame_counts: dict[str, int] = {}
        self.running = True
    
    async def receive_frames(
        self,
        track: rtc.Track,
        participant_id: str,
    ) -> AsyncGenerator[ReceivedFrame, None]:
        """
        Receive frames from a video track.
        
        Args:
            track: LiveKit video track
            participant_id: ID of the participant
            
        Yields:
            ReceivedFrame objects at target FPS
        """
        video_stream = rtc.VideoStream(track)
        
        if participant_id not in self.frame_counts:
            self.frame_counts[participant_id] = 0
        
        async for frame_event in video_stream:
            if not self.running:
                break
            
            self.frame_counts[participant_id] += 1
            frame_num = self.frame_counts[participant_id]
            
            # Skip frames to achieve target FPS
            if frame_num % self.skip_frames != 0:
                continue
            
            # Convert frame to numpy
            frame_np = self._convert_frame(frame_event.frame)
            
            yield ReceivedFrame(
                frame=frame_np,
                timestamp=datetime.utcnow(),
                participant_id=participant_id,
                frame_number=frame_num,
                width=frame_event.frame.width,
                height=frame_event.frame.height,
            )
    
    def _convert_frame(self, frame: rtc.VideoFrame) -> np.ndarray:
        """Convert LiveKit VideoFrame to numpy array."""
        # Convert to RGB24 format
        buffer = frame.convert(rtc.VideoBufferType.RGB24)
        
        # Create numpy array
        arr = np.frombuffer(buffer.data, dtype=np.uint8)
        arr = arr.reshape((buffer.height, buffer.width, 3))
        
        return arr
    
    def stop(self):
        """Stop receiving frames."""
        self.running = False
    
    def get_stats(self, participant_id: str) -> dict:
        """Get statistics for a participant."""
        return {
            "participant_id": participant_id,
            "total_frames": self.frame_counts.get(participant_id, 0),
            "processed_frames": self.frame_counts.get(participant_id, 0) // self.skip_frames,
            "target_fps": self.target_fps,
        }
