from __future__ import annotations
"""
Proctor API

FastAPI server for receiving room join requests from backend.
"""

from proctor.api.server import app, start_server

__all__ = [
    "app",
    "start_server",
]
