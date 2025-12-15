"""
WebSocket connection manager for real-time updates
"""
import asyncio
import json
import zlib
import base64
from typing import Dict, Set, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect


class MessageCompressor:
    """
    Compresses WebSocket messages for bandwidth reduction.

    Uses zlib compression with base64 encoding for text-based transport.
    Achieves 50-70% reduction for typical JSON messages.
    """

    def __init__(self, compression_level: int = 6, min_size: int = 100):
        """
        Initialize compressor.

        Args:
            compression_level: zlib compression level (1-9, default 6)
            min_size: Minimum message size to compress (bytes)
        """
        self.compression_level = compression_level
        self.min_size = min_size

    def compress(self, message: dict) -> dict:
        """
        Compress message if beneficial.

        Returns a message with 'compressed' flag and either original or compressed data.
        """
        json_str = json.dumps(message)
        original_size = len(json_str.encode('utf-8'))

        # Don't compress small messages
        if original_size < self.min_size:
            return {"compressed": False, "data": message}

        # Compress
        compressed = zlib.compress(json_str.encode('utf-8'), self.compression_level)
        compressed_b64 = base64.b64encode(compressed).decode('ascii')
        compressed_size = len(compressed_b64)

        # Only use compression if it actually saves space
        if compressed_size < original_size * 0.9:  # At least 10% reduction
            return {
                "compressed": True,
                "data": compressed_b64,
                "original_size": original_size,
                "compressed_size": compressed_size,
            }
        else:
            return {"compressed": False, "data": message}

    def decompress(self, message: dict) -> dict:
        """Decompress message if compressed."""
        if not message.get("compressed", False):
            return message.get("data", message)

        compressed_b64 = message["data"]
        compressed = base64.b64decode(compressed_b64)
        json_str = zlib.decompress(compressed).decode('utf-8')
        return json.loads(json_str)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self, enable_compression: bool = True):
        # All active connections
        self.active_connections: Set[WebSocket] = set()

        # Connections subscribed to specific feeds
        self.feed_subscribers: Dict[str, Set[WebSocket]] = {}

        # Connection metadata
        self.connection_metadata: Dict[WebSocket, dict] = {}

        # Message compression
        self.enable_compression = enable_compression
        self.compressor = MessageCompressor() if enable_compression else None

        # Statistics
        self._bytes_sent = 0
        self._bytes_saved = 0
        self._messages_sent = 0

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
                await self._send_message(websocket, message)
            except Exception as e:
                print(f"Error sending message: {e}")
                self.disconnect(websocket)

    async def _send_message(self, websocket: WebSocket, message: dict):
        """Internal method to send message with optional compression"""
        self._messages_sent += 1

        # Check if client supports compression
        metadata = self.connection_metadata.get(websocket, {})
        supports_compression = metadata.get("supports_compression", True)

        if self.enable_compression and supports_compression and self.compressor:
            compressed_msg = self.compressor.compress(message)

            if compressed_msg.get("compressed"):
                self._bytes_saved += (
                    compressed_msg["original_size"] - compressed_msg["compressed_size"]
                )
                self._bytes_sent += compressed_msg["compressed_size"]
            else:
                self._bytes_sent += len(json.dumps(message).encode('utf-8'))

            await websocket.send_json(compressed_msg)
        else:
            self._bytes_sent += len(json.dumps(message).encode('utf-8'))
            await websocket.send_json(message)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await self._send_message(connection, message)
            except Exception as e:
                print(f"Error broadcasting: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def send_to_feed_subscribers(self, feed_id: str, message: dict):
        """Send message to all subscribers of a specific feed"""
        if feed_id not in self.feed_subscribers or not self.feed_subscribers[feed_id]:
            return

        disconnected = []
        for connection in self.feed_subscribers[feed_id]:
            try:
                await self._send_message(connection, message)
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
        compression_ratio = 0
        if self._bytes_sent > 0:
            compression_ratio = round(self._bytes_saved / (self._bytes_sent + self._bytes_saved) * 100, 2)

        return {
            "total_connections": len(self.active_connections),
            "feed_subscriptions": {
                feed_id: len(subscribers)
                for feed_id, subscribers in self.feed_subscribers.items()
            },
            "compression": {
                "enabled": self.enable_compression,
                "messages_sent": self._messages_sent,
                "bytes_sent": self._bytes_sent,
                "bytes_saved": self._bytes_saved,
                "compression_ratio_percent": compression_ratio,
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
