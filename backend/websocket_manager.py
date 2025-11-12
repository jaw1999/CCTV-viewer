"""
WebSocket connection manager for real-time updates
"""
import asyncio
import json
from typing import Dict, Set, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        # All active connections
        self.active_connections: Set[WebSocket] = set()

        # Connections subscribed to specific feeds
        self.feed_subscribers: Dict[str, Set[WebSocket]] = {}

        # Connection metadata
        self.connection_metadata: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = {
            "client_id": client_id,
            "connected_at": datetime.utcnow(),
            "subscribed_feeds": set()
        }
        print(f"WebSocket connected: {client_id or 'anonymous'} (total: {len(self.active_connections)})")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

            # Remove from feed subscriptions
            metadata = self.connection_metadata.get(websocket, {})
            for feed_id in metadata.get("subscribed_feeds", set()):
                if feed_id in self.feed_subscribers:
                    self.feed_subscribers[feed_id].discard(websocket)
                    if not self.feed_subscribers[feed_id]:
                        del self.feed_subscribers[feed_id]

            # Remove metadata
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]

            print(f"WebSocket disconnected (total: {len(self.active_connections)})")

    def subscribe_to_feed(self, websocket: WebSocket, feed_id: str):
        """Subscribe a connection to updates for a specific feed"""
        if websocket not in self.active_connections:
            return

        if feed_id not in self.feed_subscribers:
            self.feed_subscribers[feed_id] = set()

        self.feed_subscribers[feed_id].add(websocket)

        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["subscribed_feeds"].add(feed_id)

        print(f"Subscribed to feed: {feed_id} (subscribers: {len(self.feed_subscribers[feed_id])})")

    def unsubscribe_from_feed(self, websocket: WebSocket, feed_id: str):
        """Unsubscribe a connection from a feed"""
        if feed_id in self.feed_subscribers:
            self.feed_subscribers[feed_id].discard(websocket)
            if not self.feed_subscribers[feed_id]:
                del self.feed_subscribers[feed_id]

        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["subscribed_feeds"].discard(feed_id)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific connection"""
        if websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"Error sending message: {e}")
                self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def send_to_feed_subscribers(self, feed_id: str, message: dict):
        """Send message to all subscribers of a specific feed"""
        if feed_id not in self.feed_subscribers:
            return

        disconnected = []
        for connection in self.feed_subscribers[feed_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to subscriber: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def send_detection_update(self, feed_id: str, detection_data: dict):
        """Send vehicle detection update to subscribers"""
        message = {
            "type": "detection",
            "feed_id": feed_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": detection_data
        }
        await self.send_to_feed_subscribers(feed_id, message)

    async def send_feed_status_update(self, feed_id: str, is_working: bool, has_vehicles: bool):
        """Send feed status update"""
        message = {
            "type": "feed_status",
            "feed_id": feed_id,
            "timestamp": datetime.utcnow().isoformat(),
            "is_working": is_working,
            "has_vehicles": has_vehicles
        }
        await self.send_to_feed_subscribers(feed_id, message)

    async def send_stats_update(self, stats: dict):
        """Broadcast system statistics update"""
        message = {
            "type": "stats",
            "timestamp": datetime.utcnow().isoformat(),
            "data": stats
        }
        await self.broadcast(message)

    async def send_heartbeat(self):
        """Send periodic heartbeat to all connections"""
        message = {
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat(),
            "connections": len(self.active_connections)
        }
        await self.broadcast(message)

    def get_stats(self) -> dict:
        """Get WebSocket connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "feed_subscriptions": {
                feed_id: len(subscribers)
                for feed_id, subscribers in self.feed_subscribers.items()
            }
        }


async def handle_websocket_messages(websocket: WebSocket, manager: ConnectionManager):
    """
    Handle incoming WebSocket messages from clients

    Message format:
    {
        "action": "subscribe" | "unsubscribe" | "ping",
        "feed_id": "CCTV-ID" (for subscribe/unsubscribe)
    }
    """
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                action = message.get("action")

                if action == "subscribe":
                    feed_id = message.get("feed_id")
                    if feed_id:
                        manager.subscribe_to_feed(websocket, feed_id)
                        await manager.send_personal_message({
                            "type": "subscribed",
                            "feed_id": feed_id
                        }, websocket)

                elif action == "unsubscribe":
                    feed_id = message.get("feed_id")
                    if feed_id:
                        manager.unsubscribe_from_feed(websocket, feed_id)
                        await manager.send_personal_message({
                            "type": "unsubscribed",
                            "feed_id": feed_id
                        }, websocket)

                elif action == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }, websocket)

            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON"
                }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)
