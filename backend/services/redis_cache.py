"""
Redis caching layer for distributed CCTV Viewer deployments
"""
import json
import hashlib
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class RedisCacheConfig:
    """Redis cache configuration"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ssl: bool = False,
        prefix: str = "cctv:",
        default_ttl: int = 300,  # 5 minutes
        image_ttl: int = 60,  # 1 minute for images
        max_connections: int = 50,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ssl = ssl
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.image_ttl = image_ttl
        self.max_connections = max_connections


class RedisCache:
    """
    Async Redis cache for CCTV Viewer.

    Features:
    - Distributed caching for multi-instance deployments
    - Image caching with configurable TTL
    - Feed status and detection data caching
    - Pub/Sub for real-time updates across instances
    - Automatic reconnection handling
    """

    def __init__(self, config: Optional[RedisCacheConfig] = None):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Run: pip install redis")

        self.config = config or RedisCacheConfig()
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._connected = False

    async def connect(self) -> bool:
        """Establish connection to Redis"""
        try:
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                ssl=self.config.ssl,
                decode_responses=False,  # We handle bytes for images
                max_connections=self.config.max_connections,
            )

            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Close Redis connection"""
        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None

        if self._client:
            await self._client.close()
            self._client = None

        self._connected = False
        logger.info("Disconnected from Redis")

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis"""
        return self._connected and self._client is not None

    def _key(self, *parts: str) -> str:
        """Generate prefixed Redis key"""
        return self.config.prefix + ":".join(parts)

    # ========================================================================
    # Image Caching
    # ========================================================================

    async def get_image(self, feed_id: str) -> Optional[bytes]:
        """Get cached image for feed"""
        if not self.is_connected:
            return None

        try:
            key = self._key("image", feed_id)
            return await self._client.get(key)
        except Exception as e:
            logger.error(f"Redis get_image error: {e}")
            return None

    async def set_image(
        self,
        feed_id: str,
        image_bytes: bytes,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache image for feed"""
        if not self.is_connected:
            return False

        try:
            key = self._key("image", feed_id)
            ttl = ttl or self.config.image_ttl
            await self._client.setex(key, ttl, image_bytes)
            return True
        except Exception as e:
            logger.error(f"Redis set_image error: {e}")
            return False

    async def delete_image(self, feed_id: str) -> bool:
        """Delete cached image"""
        if not self.is_connected:
            return False

        try:
            key = self._key("image", feed_id)
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete_image error: {e}")
            return False

    # ========================================================================
    # Feed Status Caching
    # ========================================================================

    async def get_feed_status(self, feed_id: str) -> Optional[Dict[str, Any]]:
        """Get cached feed status"""
        if not self.is_connected:
            return None

        try:
            key = self._key("status", feed_id)
            data = await self._client.get(key)
            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Redis get_feed_status error: {e}")
            return None

    async def set_feed_status(
        self,
        feed_id: str,
        is_working: bool,
        has_vehicles: bool,
        vehicle_count: int = 0,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache feed status"""
        if not self.is_connected:
            return False

        try:
            key = self._key("status", feed_id)
            data = {
                "is_working": is_working,
                "has_vehicles": has_vehicles,
                "vehicle_count": vehicle_count,
                "updated_at": datetime.utcnow().isoformat(),
            }
            ttl = ttl or self.config.default_ttl
            await self._client.setex(key, ttl, json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Redis set_feed_status error: {e}")
            return False

    async def get_all_feed_status(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached feed statuses"""
        if not self.is_connected:
            return {}

        try:
            pattern = self._key("status", "*")
            statuses = {}

            async for key in self._client.scan_iter(match=pattern):
                feed_id = key.decode('utf-8').split(":")[-1]
                data = await self._client.get(key)
                if data:
                    statuses[feed_id] = json.loads(data.decode('utf-8'))

            return statuses
        except Exception as e:
            logger.error(f"Redis get_all_feed_status error: {e}")
            return {}

    # ========================================================================
    # Image Hash Tracking (for change detection)
    # ========================================================================

    async def get_image_hash(self, feed_id: str) -> Optional[str]:
        """Get stored image hash for change detection"""
        if not self.is_connected:
            return None

        try:
            key = self._key("hash", feed_id)
            data = await self._client.get(key)
            return data.decode('utf-8') if data else None
        except Exception as e:
            logger.error(f"Redis get_image_hash error: {e}")
            return None

    async def set_image_hash(self, feed_id: str, hash_value: str) -> bool:
        """Store image hash for change detection"""
        if not self.is_connected:
            return False

        try:
            key = self._key("hash", feed_id)
            await self._client.setex(key, self.config.default_ttl * 2, hash_value)
            return True
        except Exception as e:
            logger.error(f"Redis set_image_hash error: {e}")
            return False

    async def has_image_changed(self, feed_id: str, new_image_bytes: bytes) -> bool:
        """Check if image has changed since last cache"""
        new_hash = hashlib.md5(new_image_bytes).hexdigest()
        old_hash = await self.get_image_hash(feed_id)
        return old_hash != new_hash

    # ========================================================================
    # Detection Data Caching
    # ========================================================================

    async def cache_detection(
        self,
        feed_id: str,
        detection_data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache detection data"""
        if not self.is_connected:
            return False

        try:
            key = self._key("detection", feed_id)
            detection_data["cached_at"] = datetime.utcnow().isoformat()
            ttl = ttl or self.config.default_ttl
            await self._client.setex(key, ttl, json.dumps(detection_data))
            return True
        except Exception as e:
            logger.error(f"Redis cache_detection error: {e}")
            return False

    async def get_detection(self, feed_id: str) -> Optional[Dict[str, Any]]:
        """Get cached detection data"""
        if not self.is_connected:
            return None

        try:
            key = self._key("detection", feed_id)
            data = await self._client.get(key)
            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Redis get_detection error: {e}")
            return None

    # ========================================================================
    # Pub/Sub for Real-time Updates
    # ========================================================================

    async def publish_detection(self, feed_id: str, detection_data: Dict[str, Any]):
        """Publish detection update for other instances"""
        if not self.is_connected:
            return

        try:
            channel = self._key("updates", "detection")
            message = {
                "feed_id": feed_id,
                "data": detection_data,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self._client.publish(channel, json.dumps(message))
        except Exception as e:
            logger.error(f"Redis publish_detection error: {e}")

    async def subscribe_detections(self, callback):
        """Subscribe to detection updates from other instances"""
        if not self.is_connected:
            return

        try:
            self._pubsub = self._client.pubsub()
            channel = self._key("updates", "detection")
            await self._pubsub.subscribe(channel)

            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"].decode('utf-8'))
                    await callback(data)

        except Exception as e:
            logger.error(f"Redis subscribe_detections error: {e}")

    # ========================================================================
    # Statistics
    # ========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.is_connected:
            return {"connected": False}

        try:
            info = await self._client.info("memory")
            keys_info = await self._client.info("keyspace")

            # Count our keys
            image_count = 0
            status_count = 0
            async for _ in self._client.scan_iter(match=self._key("image", "*")):
                image_count += 1
            async for _ in self._client.scan_iter(match=self._key("status", "*")):
                status_count += 1

            return {
                "connected": True,
                "host": f"{self.config.host}:{self.config.port}",
                "memory_used_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                "cached_images": image_count,
                "cached_statuses": status_count,
                "keyspace": keys_info,
            }
        except Exception as e:
            logger.error(f"Redis get_stats error: {e}")
            return {"connected": True, "error": str(e)}


class HybridCache:
    """
    Hybrid cache that uses Redis when available, falls back to local LRU cache.

    This allows the application to work in both single-instance and
    distributed deployments without code changes.
    """

    def __init__(
        self,
        redis_config: Optional[RedisCacheConfig] = None,
        local_max_items: int = 5000,
        local_max_size_mb: int = 1024,
    ):
        self._redis: Optional[RedisCache] = None
        self._use_redis = False

        # Import local cache
        try:
            from ..core.cache import FeedCache
        except ImportError:
            from core.cache import FeedCache

        self._local = FeedCache(
            max_feeds=local_max_items,
            max_size_mb=local_max_size_mb,
        )

        if redis_config and REDIS_AVAILABLE:
            self._redis = RedisCache(redis_config)

    async def initialize(self) -> bool:
        """Initialize cache (connect to Redis if configured)"""
        if self._redis:
            connected = await self._redis.connect()
            self._use_redis = connected
            if connected:
                logger.info("Using Redis cache")
            else:
                logger.warning("Redis unavailable, using local cache")
        else:
            logger.info("Using local LRU cache (Redis not configured)")

        return True

    async def shutdown(self):
        """Shutdown cache connections"""
        if self._redis:
            await self._redis.disconnect()

    async def get_image(self, feed_id: str) -> Optional[bytes]:
        """Get cached image"""
        if self._use_redis:
            image = await self._redis.get_image(feed_id)
            if image:
                return image

        return self._local.get_image(feed_id)

    async def set_image(
        self,
        feed_id: str,
        image_bytes: bytes,
        is_working: bool = True,
        has_vehicles: bool = False,
    ):
        """Cache image in both local and Redis"""
        # Always update local cache (fast path)
        self._local.set_image(feed_id, image_bytes, is_working, has_vehicles)

        # Update Redis if available (async, non-blocking)
        if self._use_redis:
            await self._redis.set_image(feed_id, image_bytes)
            await self._redis.set_feed_status(
                feed_id, is_working, has_vehicles
            )

    def get_status(self, feed_id: str) -> Optional[bool]:
        """Get feed status (local only for speed)"""
        return self._local.get_status(feed_id)

    def get_all_status(self) -> Dict[str, bool]:
        """Get all feed statuses"""
        return self._local.get_all_status()

    def get_all_vehicle_detected(self) -> Dict[str, bool]:
        """Get all vehicle detection statuses"""
        return self._local.get_all_vehicle_detected()

    async def has_image_changed(self, feed_id: str, new_image_bytes: bytes) -> bool:
        """Check if image has changed"""
        return self._local.has_image_changed(feed_id, new_image_bytes)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "local": self._local.stats,
            "redis_enabled": self._use_redis,
        }

        return stats
