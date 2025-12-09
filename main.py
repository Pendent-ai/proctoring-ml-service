"""
Proctoring ML Service - Main Entry Point

Usage:
    python main.py              # Start API server (default)
    python main.py --port 8001  # Start on specific port
    
API Endpoints:
    POST /join   - Backend requests ML to join a room
    POST /leave  - Backend requests ML to leave a room
    GET  /status - Get active monitoring sessions
    GET  /health - Health check
"""
from __future__ import annotations

import argparse


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Proctoring ML Service")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    args = parser.parse_args()
    
    print("🚀 Starting Proctoring ML Service API...")
    print(f"📡 Listening on http://{args.host}:{args.port}")
    print()
    print("Endpoints:")
    print("  POST /join   - Request to monitor a room")
    print("  POST /leave  - Stop monitoring a room")
    print("  GET  /status - Active sessions")
    print("  GET  /health - Health check")
    print()
    
    from proctor.api.server import start_server
    start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()