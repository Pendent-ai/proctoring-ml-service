"""
LiveKit Room Manager

Handles room connection, token generation, and lifecycle management.
"""

import asyncio
from datetime import datetime
from livekit import api, rtc

from config import settings


class RoomManager:
    """Manages LiveKit room connections."""
    
    def __init__(self):
        self.room = rtc.Room()
        self.connected = False
        self.room_name: str | None = None
        self.participant_identity: str | None = None
        
    def generate_token(
        self,
        room_name: str,
        identity: str = "ml-proctoring",
        name: str = "ML Proctoring Service",
    ) -> str:
        """
        Generate a LiveKit access token.
        
        Args:
            room_name: Name of the room to join
            identity: Participant identity
            name: Display name
            
        Returns:
            JWT token string
        """
        token = api.AccessToken(
            settings.livekit_api_key,
            settings.livekit_api_secret,
        )
        token.with_identity(identity)
        token.with_name(name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_subscribe=True,
            can_publish=False,  # We don't publish video
            can_publish_data=True,  # We publish alerts via data channel
        ))
        
        return token.to_jwt()
    
    async def connect(
        self,
        room_name: str,
        identity: str = "ml-proctoring",
    ) -> rtc.Room:
        """
        Connect to a LiveKit room.
        
        Args:
            room_name: Name of the room to join
            identity: Participant identity
            
        Returns:
            Connected Room instance
        """
        token = self.generate_token(room_name, identity)
        
        await self.room.connect(settings.livekit_url, token)
        
        self.connected = True
        self.room_name = room_name
        self.participant_identity = identity
        
        print(f"✅ Connected to room: {room_name}")
        
        return self.room
    
    async def disconnect(self):
        """Disconnect from the room."""
        if self.connected:
            await self.room.disconnect()
            self.connected = False
            print(f"👋 Disconnected from room: {self.room_name}")
    
    def get_remote_participants(self) -> list[rtc.RemoteParticipant]:
        """Get list of remote participants in the room."""
        return list(self.room.remote_participants.values())
    
    def get_video_tracks(self) -> list[tuple[rtc.RemoteParticipant, rtc.RemoteTrackPublication]]:
        """Get all subscribed video tracks."""
        tracks = []
        
        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.kind == rtc.TrackKind.KIND_VIDEO and publication.subscribed:
                    tracks.append((participant, publication))
        
        return tracks
