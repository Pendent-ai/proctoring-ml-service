from __future__ import annotations
"""
Audio Receiver

Receives audio samples from LiveKit tracks.
"""

import asyncio
from typing import AsyncGenerator, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import numpy as np

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False


@dataclass
class AudioChunk:
    """A received audio chunk with metadata."""
    samples: np.ndarray
    sample_rate: int
    channels: int = 1
    duration_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    participant_id: str = ""


class AudioReceiver:
    """
    Receives audio chunks from LiveKit audio tracks.
    
    Buffers audio samples and yields chunks at specified duration.
    
    Example:
        >>> receiver = AudioReceiver(chunk_duration_ms=500)
        >>> async for chunk in receiver.receive(track, participant_id):
        ...     process(chunk)
    """
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        chunk_duration_ms: int = 500,
    ):
        """
        Initialize audio receiver.
        
        Args:
            target_sample_rate: Target sample rate for processing
            chunk_duration_ms: Duration of each audio chunk in ms
        """
        if not LIVEKIT_AVAILABLE:
            raise ImportError("livekit package required: pip install livekit")
        
        self.target_sample_rate = target_sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_samples = int(target_sample_rate * chunk_duration_ms / 1000)
        
        self.running = True
        self._buffers: dict[str, deque] = {}
    
    async def receive(
        self,
        track: "rtc.Track",
        participant_id: str,
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Receive audio chunks from a track.
        
        Args:
            track: LiveKit audio track
            participant_id: ID of the participant
            
        Yields:
            AudioChunk objects at specified duration
        """
        audio_stream = rtc.AudioStream(track)
        
        # Initialize buffer for this participant
        if participant_id not in self._buffers:
            self._buffers[participant_id] = deque()
        
        buffer = self._buffers[participant_id]
        
        async for frame_event in audio_stream:
            if not self.running:
                break
            
            # Convert frame to numpy
            samples = self._frame_to_numpy(frame_event.frame)
            
            # Add to buffer
            buffer.extend(samples)
            
            # Yield chunks when we have enough samples
            while len(buffer) >= self.chunk_samples:
                # Extract chunk
                chunk = np.array([buffer.popleft() for _ in range(self.chunk_samples)])
                
                yield AudioChunk(
                    samples=chunk,
                    sample_rate=self.target_sample_rate,
                    channels=1,
                    duration_ms=self.chunk_duration_ms,
                    timestamp=datetime.utcnow(),
                    participant_id=participant_id,
                )
    
    def _frame_to_numpy(self, frame: "rtc.AudioFrame") -> np.ndarray:
        """Convert LiveKit AudioFrame to numpy array."""
        # Get raw data as int16
        data = np.frombuffer(frame.data, dtype=np.int16)
        
        # Convert to float32 normalized [-1, 1]
        samples = data.astype(np.float32) / 32768.0
        
        # Convert to mono if stereo
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1)
        
        # Resample if needed
        if frame.sample_rate != self.target_sample_rate:
            samples = self._resample(samples, frame.sample_rate, self.target_sample_rate)
        
        return samples
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    def stop(self):
        """Stop receiving audio."""
        self.running = False
    
    def clear_buffer(self, participant_id: str):
        """Clear buffer for a participant."""
        if participant_id in self._buffers:
            self._buffers[participant_id].clear()
    
    def reset(self):
        """Reset all buffers."""
        for buffer in self._buffers.values():
            buffer.clear()
        self._buffers.clear()
