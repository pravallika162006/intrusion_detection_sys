"""
WebSocket Connection Manager for streaming real-time live network detections.
"""

from typing import List
from fastapi import WebSocket
from backend.utils.logger import setup_logger

logger = setup_logger("WebSocketManager")

class WebSocketManager:
    """Manages active WebSocket connections from Live Dashboard clients."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected ({len(self.active_connections)} active)")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected ({len(self.active_connections)} active)")

    async def broadcast_event(self, event_data: dict):
        """Broadcasts a live flow detection event JSON to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(event_data)
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)
