from __future__ import annotations
"""
Video Receiver

Receives video frames from LiveKit tracks.
"""

import asyncio
from typing import AsyncGenerator, Optional
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False


@dataclass
class VideoFrame:
    """A received video frame with metadata."""
    frame: np.ndarray
    width: int
    height: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    participant_id: str = ""
    frame_number: int = 0


class VideoReceiver:
    """
    Receives video frames from LiveKit video tracks.
    
    Handles frame conversion, rate limiting, and buffering.
    
    Example:
        >>> receiver = VideoReceiver(target_fps=10)
        >>> async for frame in receiver.receive(track, participant_id):
        ...     process(frame)
    """
    
    def __init__(
        self,
        target_fps: int = 10,
        target_width: int = 640,
        target_height: int = 480,
    ):
        """
        Initialize video receiver.
        
        Args:
            target_fps: Target frames per second for processing
            target_width: Target frame width
            target_height: Target frame height
        """
        if not LIVEKIT_AVAILABLE:
            raise ImportError("livekit package required: pip install livekit")
        
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        
        self.running = True
        self._frame_counts: dict[str, int] = {}
    
    async def receive(
        self,
        track: "rtc.Track",
        participant_id: str,
        skip_frames: int = 3,
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        Receive video frames from a track.
        
        Args:
            track: LiveKit video track
            participant_id: ID of the participant
            skip_frames: Number of frames to skip between processing
            
        Yields:
            VideoFrame objects at target FPS
        """
        video_stream = rtc.VideoStream(track)
        
        # Initialize frame counter for this participant
        if participant_id not in self._frame_counts:
            self._frame_counts[participant_id] = 0
        
        frame_count = 0
        
        async for frame_event in video_stream:
            if not self.running:
                break
            
            frame_count += 1
            
            # Skip frames to maintain target FPS
            if frame_count % skip_frames != 0:
                continue
            
            # Convert frame
            frame_array = self._frame_to_numpy(frame_event.frame)
            
            # Resize if needed
            if frame_array.shape[1] != self.target_width or frame_array.shape[0] != self.target_height:
                frame_array = self._resize_frame(frame_array)
            
            self._frame_counts[participant_id] += 1
            
            yield VideoFrame(
                frame=frame_array,
                width=frame_array.shape[1],
                height=frame_array.shape[0],
                timestamp=datetime.utcnow(),
                participant_id=participant_id,
                frame_number=self._frame_counts[participant_id],
            )
    
    def _frame_to_numpy(self, frame: "rtc.VideoFrame") -> np.ndarray:
        """Convert LiveKit VideoFrame to numpy array."""
        # Convert to RGB24 format
        buffer = frame.convert(rtc.VideoBufferType.RGB24)
        
        # Create numpy array
        arr = np.frombuffer(buffer.data, dtype=np.uint8)
        arr = arr.reshape((buffer.height, buffer.width, 3))
        
        return arr
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target dimensions."""
        import cv2
        return cv2.resize(frame, (self.target_width, self.target_height))
    
    def stop(self):
        """Stop receiving frames."""
        self.running = False
    
    def get_frame_count(self, participant_id: str) -> int:
        """Get number of frames received for a participant."""
        return self._frame_counts.get(participant_id, 0)
    
    def reset(self, participant_id: Optional[str] = None):
        """Reset frame counters."""
        if participant_id:
            self._frame_counts[participant_id] = 0
        else:
            self._frame_counts.clear()
