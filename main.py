"""
Proctoring ML Service - Main Entry Point

Usage:
    python main.py [room_name]
    
Or using the proctor package directly:
    from proctor import ProctoringService
    service = ProctoringService()
    await service.run("room-name")
"""

import asyncio
from proctor.service import main


if __name__ == "__main__":
    print("🚀 Starting Proctoring ML Service...")
    asyncio.run(main())
