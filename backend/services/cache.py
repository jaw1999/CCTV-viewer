"""
In-memory LRU cache for CCTV feed images.
Thread-safe implementation with TTL support.
"""
import time
import threading
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: bytes
    timestamp: float
    size: int
    hash: str


class LRUCache:
    """Thread-safe LRU cache with size and count limits."""

    def __init__(self, max_mb: float = 500.0, max_n: int = 3000, ttl: float = 300.0):
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.max_n = max_n
        self.ttl = ttl
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._size = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Get value and hash from cache. Returns (None, None) on miss."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None, None

            entry = self._cache[key]
            if time.time() - entry.timestamp > self.ttl:
                self._cache.pop(key)
                self._size -= entry.size
                self._misses += 1
                return None, None

            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value, entry.hash

    def set(self, key: str, value: bytes) -> str:
        """Set value in cache, returns hash."""
        with self._lock:
            h = hashlib.md5(value).hexdigest()
            sz = len(value)

            if key in self._cache:
                old = self._cache.pop(key)
                self._size -= old.size

            # Evict entries if needed
            while (self._size + sz > self.max_bytes or len(self._cache) >= self.max_n) and self._cache:
                _, entry = self._cache.popitem(last=False)
                self._size -= entry.size
                self._evictions += 1

            self._cache[key] = CacheEntry(value, time.time(), sz, h)
            self._size += sz
            return h

    def set_if_changed(self, key: str, value: bytes) -> Tuple[bool, str]:
        """Set value only if content changed. Returns (changed, hash)."""
        with self._lock:
            h = hashlib.md5(value).hexdigest()
            if key in self._cache and self._cache[key].hash == h:
                # Same content, just refresh timestamp
                self._cache[key].timestamp = time.time()
                self._cache.move_to_end(key)
                return False, h
            self.set(key, value)
            return True, h

    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "size_mb": round(self._size / 1e6, 2),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }


class FeedCacheManager:
    """High-level cache manager for feed images and status."""

    def __init__(self, max_mb: float = 500.0, max_n: int = 3000, ttl: float = 300.0):
        self._cache = LRUCache(max_mb, max_n, ttl)
        self._status = {}
        self._vehicle_detected = {}

    def cache_image(self, feed_id: str, data: bytes, working: bool = True,
                    vehicles: bool = False) -> Tuple[bool, str]:
        """Cache feed image and update status. Returns (changed, hash)."""
        changed, h = self._cache.set_if_changed(feed_id, data)
        self._status[feed_id] = working
        self._vehicle_detected[feed_id] = vehicles
        return changed, h

    def get_image(self, feed_id: str) -> Optional[bytes]:
        """Get cached image for feed."""
        value, _ = self._cache.get(feed_id)
        return value

    def get_status(self, feed_id: str) -> bool:
        """Get working status for feed."""
        return self._status.get(feed_id, False)

    def get_vehicle_detected(self, feed_id: str) -> bool:
        """Get vehicle detection status for feed."""
        return self._vehicle_detected.get(feed_id, False)

    def get_all_status(self) -> dict:
        """Get all feed statuses."""
        return dict(self._status)

    def get_all_vehicle_detected(self) -> dict:
        """Get all vehicle detection statuses."""
        return dict(self._vehicle_detected)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = self._cache.stats()
        stats["working_feeds"] = sum(1 for v in self._status.values() if v)
        stats["feeds_with_vehicles"] = sum(1 for v in self._vehicle_detected.values() if v)
        return stats
