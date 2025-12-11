from __future__ import annotations
"""
FastAPI Server for Proctoring ML Service

Exposes HTTP API for backend to request room monitoring.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from proctor.service.proctoring import ProctoringService
from proctor.cfg import get_settings
from proctor.utils import get_logger

logger = get_logger(__name__)

# Track active proctoring sessions
active_sessions: dict[str, ProctoringService] = {}


class JoinRoomRequest(BaseModel):
    """Request to join a room for proctoring."""
    room_name: str
    participant_identity: str = "ml-proctoring"


class JoinRoomResponse(BaseModel):
    """Response after joining a room."""
    success: bool
    room_name: str
    message: str


class LeaveRoomRequest(BaseModel):
    """Request to leave a room."""
    room_name: str


class StatusResponse(BaseModel):
    """Status response."""
    active_rooms: list[str]
    total_sessions: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("🚀 Proctoring API starting...")
    yield
    # Cleanup on shutdown
    logger.info("👋 Shutting down, disconnecting from all rooms...")
    for room_name, service in list(active_sessions.items()):
        try:
            await service.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting from {room_name}: {e}")
    active_sessions.clear()


app = FastAPI(
    title="Proctoring ML Service",
    description="API for backend to request room monitoring",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_sessions": len(active_sessions)}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current service status."""
    return StatusResponse(
        active_rooms=list(active_sessions.keys()),
        total_sessions=len(active_sessions),
    )


@app.post("/join", response_model=JoinRoomResponse)
async def join_room(request: JoinRoomRequest, background_tasks: BackgroundTasks):
    """
    Join a room for proctoring.
    
    Called by backend when an interview starts.
    """
    room_name = request.room_name
    
    # Check if already monitoring this room
    if room_name in active_sessions:
        return JoinRoomResponse(
            success=True,
            room_name=room_name,
            message="Already monitoring this room",
        )
    
    try:
        # Create new proctoring service
        service = ProctoringService()
        
        # Connect to room (async)
        await service.connect(room_name, request.participant_identity)
        
        # Store session
        active_sessions[room_name] = service
        
        # Start processing in background
        background_tasks.add_task(run_proctoring, room_name, service)
        
        logger.info(f"✅ Started monitoring room: {room_name}")
        
        return JoinRoomResponse(
            success=True,
            room_name=room_name,
            message="Successfully joined room for proctoring",
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to join room {room_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_proctoring(room_name: str, service: ProctoringService):
    """Run proctoring in background until disconnected."""
    try:
        while service.running:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"❌ Proctoring error for {room_name}: {e}")
    finally:
        # Remove from active sessions
        if room_name in active_sessions:
            del active_sessions[room_name]
        logger.info(f"👋 Stopped monitoring room: {room_name}")


@app.post("/leave", response_model=JoinRoomResponse)
async def leave_room(request: LeaveRoomRequest):
    """
    Leave a room.
    
    Called by backend when an interview ends.
    """
    room_name = request.room_name
    
    if room_name not in active_sessions:
        return JoinRoomResponse(
            success=True,
            room_name=room_name,
            message="Not monitoring this room",
        )
    
    try:
        service = active_sessions[room_name]
        await service.disconnect()
        del active_sessions[room_name]
        
        logger.info(f"👋 Left room: {room_name}")
        
        return JoinRoomResponse(
            success=True,
            room_name=room_name,
            message="Successfully left room",
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to leave room {room_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def start_server(host: str = "0.0.0.0", port: int = 8001):
    """Start the FastAPI server."""
    import uvicorn
    
    settings = get_settings()
    port = settings.api_port if hasattr(settings, 'api_port') else port
    
    uvicorn.run(
        "proctor.api.server:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    start_server()
