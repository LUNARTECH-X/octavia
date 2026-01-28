import asyncio
import json
from typing import Dict, List, Set
from fastapi import WebSocket

class WebSocketManager:
    """Manages WebSocket connections for job progress updates"""
    
    def __init__(self):
        # job_id -> set of WebSockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept connection and add to job group"""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)
        print(f"WS: Client connected to job {job_id}. Total for job: {len(self.active_connections[job_id])}")

    def disconnect(self, websocket: WebSocket, job_id: str):
        """Remove connection from job group"""
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        print(f"WS: Client disconnected from job {job_id}")

    async def broadcast(self, job_id: str, message: dict):
        """Broadcast update to all clients watching a job"""
        if job_id not in self.active_connections:
            return

        dead_connections = set()
        
        # Create a list to iterate safely while potentially removing items
        for connection in list(self.active_connections[job_id]):
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"WS: Error sending to client for job {job_id}: {e}")
                dead_connections.add(connection)

        # Cleanup dead connections
        for dead in dead_connections:
            self.disconnect(dead, job_id)

ws_manager = WebSocketManager()
