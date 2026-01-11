"""
Tests for LRUCache and FeedCache implementations
"""
import pytest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.cache import LRUCache, FeedCache, CacheEntry


class TestLRUCache:
    """Tests for LRUCache implementation"""

    def test_basic_get_set(self):
        """Test basic get and set operations"""
        cache = LRUCache[str](max_items=10)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_missing_key(self):
        """Test get returns None for missing key"""
        cache = LRUCache[str](max_items=10)
        assert cache.get("missing") is None

    def test_max_items_eviction(self):
        """Test LRU eviction when max items exceeded"""
        cache = LRUCache[int](max_items=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # All items should be present
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3

        # Add fourth item - should evict LRU (but we accessed 'a' above, so 'b' is LRU now... wait, we accessed all)
        # Actually after accessing all, the order is a, b, c
        # Adding new item should evict 'a' (first accessed after initial insertion)
        cache.set("d", 4)

        # 'a' should still be there because we accessed it
        # Actually, after the gets, order is: b, c, a (since we got a, b, c in that order)
        # So 'b' is LRU and should be evicted... let me reconsider

        # Actually the order after sets is: a, b, c
        # After get("a"), order is: b, c, a
        # After get("b"), order is: c, a, b
        # After get("c"), order is: a, b, c
        # So 'a' is LRU and should be evicted
        assert cache.get("a") is None
        assert cache.get("d") == 4

    def test_size_based_eviction(self):
        """Test eviction based on size limit"""
        cache = LRUCache[bytes](max_items=100, max_size_bytes=1000)

        cache.set("a", b"x" * 400)  # 400 bytes
        cache.set("b", b"y" * 400)  # 400 bytes

        assert cache.get("a") is not None
        assert cache.get("b") is not None

        # Add item that requires eviction
        cache.set("c", b"z" * 400)  # Would exceed 1000, so should evict

        # 'a' was LRU (we accessed b after a above), so it should be evicted
        # Actually after gets: order is a, b, so a is LRU
        assert cache.get("a") is None
        assert cache.get("c") is not None

    def test_ttl_expiration(self):
        """Test TTL-based expiration"""
        cache = LRUCache[str](max_items=10, ttl_seconds=0.1)

        cache.set("key", "value")
        assert cache.get("key") == "value"

        # Wait for TTL to expire
        time.sleep(0.15)
        assert cache.get("key") is None

    def test_delete(self):
        """Test delete operation"""
        cache = LRUCache[str](max_items=10)

        cache.set("key", "value")
        assert cache.get("key") == "value"

        result = cache.delete("key")
        assert result is True
        assert cache.get("key") is None

        # Delete non-existent key
        result = cache.delete("missing")
        assert result is False

    def test_clear(self):
        """Test clear operation"""
        cache = LRUCache[str](max_items=10)

        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")

        cache.clear()

        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") is None
        assert cache.size == 0

    def test_has_changed(self):
        """Test hash-based change detection"""
        cache = LRUCache[str](max_items=10)

        cache.set("key", "value", hash_value="hash1")

        assert cache.has_changed("key", "hash1") is False
        assert cache.has_changed("key", "hash2") is True
        assert cache.has_changed("missing", "hash1") is True

    def test_statistics(self):
        """Test cache statistics tracking"""
        cache = LRUCache[str](max_items=10)

        # Initial stats
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        cache.set("key", "value")

        # Hit
        cache.get("key")
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 0

        # Miss
        cache.get("missing")
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

        # Hit rate
        assert stats["hit_rate"] == 0.5

    def test_size_properties(self):
        """Test size and size_bytes properties"""
        cache = LRUCache[bytes](max_items=10)

        cache.set("a", b"x" * 100)
        cache.set("b", b"y" * 200)

        assert cache.size == 2
        assert cache.size_bytes == 300

    def test_oversized_item_rejected(self):
        """Test that items larger than max size are rejected"""
        cache = LRUCache[bytes](max_items=10, max_size_bytes=100)

        result = cache.set("big", b"x" * 200)
        assert result is False
        assert cache.get("big") is None

    def test_cleanup_expired(self):
        """Test cleanup_expired removes stale entries"""
        cache = LRUCache[str](max_items=10, ttl_seconds=0.1)

        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")

        time.sleep(0.15)

        removed = cache.cleanup_expired()
        assert removed == 3
        assert cache.size == 0


class TestFeedCache:
    """Tests for FeedCache implementation"""

    def test_basic_operations(self):
        """Test basic FeedCache operations"""
        cache = FeedCache(max_feeds=100, max_size_mb=10)

        image = b"fake_jpeg_data"
        cache.set_image("feed-1", image, is_working=True, has_vehicles=False)

        assert cache.get_image("feed-1") == image
        assert cache.get_status("feed-1") is True
        assert cache.get_vehicle_detected("feed-1") is False

    def test_vehicle_detected(self):
        """Test vehicle detection status"""
        cache = FeedCache(max_feeds=100)

        cache.set_image("feed-1", b"data", is_working=True, has_vehicles=True)
        assert cache.get_vehicle_detected("feed-1") is True

        cache.set_vehicle_detected("feed-1", False)
        assert cache.get_vehicle_detected("feed-1") is False

    def test_image_change_detection(self):
        """Test image change detection via hash"""
        cache = FeedCache(max_feeds=100)

        image1 = b"image_version_1"
        image2 = b"image_version_2"

        cache.set_image("feed-1", image1)

        # Same image should not be detected as changed
        assert cache.has_image_changed("feed-1", image1) is False

        # Different image should be detected as changed
        assert cache.has_image_changed("feed-1", image2) is True

    def test_get_all_status(self):
        """Test getting all feed statuses"""
        cache = FeedCache(max_feeds=100)

        cache.set_image("feed-1", b"data1", is_working=True)
        cache.set_image("feed-2", b"data2", is_working=False)
        cache.set_image("feed-3", b"data3", is_working=True)

        all_status = cache.get_all_status()
        assert all_status["feed-1"] is True
        assert all_status["feed-2"] is False
        assert all_status["feed-3"] is True

    def test_get_all_vehicle_detected(self):
        """Test getting all vehicle detection statuses"""
        cache = FeedCache(max_feeds=100)

        cache.set_image("feed-1", b"data1", has_vehicles=True)
        cache.set_image("feed-2", b"data2", has_vehicles=False)
        cache.set_image("feed-3", b"data3", has_vehicles=True)

        all_detected = cache.get_all_vehicle_detected()
        assert all_detected["feed-1"] is True
        assert all_detected["feed-2"] is False
        assert all_detected["feed-3"] is True

    def test_delete(self):
        """Test deleting a feed from cache"""
        cache = FeedCache(max_feeds=100)

        cache.set_image("feed-1", b"data", is_working=True, has_vehicles=True)

        cache.delete("feed-1")

        assert cache.get_image("feed-1") is None
        assert cache.get_status("feed-1") is None
        assert cache.get_vehicle_detected("feed-1") is False

    def test_clear(self):
        """Test clearing all cached data"""
        cache = FeedCache(max_feeds=100)

        cache.set_image("feed-1", b"data1")
        cache.set_image("feed-2", b"data2")
        cache.set_image("feed-3", b"data3")

        cache.clear()

        assert cache.get_image("feed-1") is None
        assert cache.get_image("feed-2") is None
        assert cache.get_image("feed-3") is None
        assert cache.get_all_status() == {}

    def test_stats(self):
        """Test cache statistics"""
        cache = FeedCache(max_feeds=100)

        cache.set_image("feed-1", b"x" * 1000, is_working=True, has_vehicles=True)
        cache.set_image("feed-2", b"y" * 2000, is_working=True, has_vehicles=False)
        cache.set_image("feed-3", b"z" * 1500, is_working=False, has_vehicles=False)

        stats = cache.stats

        assert stats["items"] == 3
        assert stats["size_bytes"] == 4500
        assert stats["feeds_working"] == 2
        assert stats["feeds_offline"] == 1
        assert stats["feeds_with_vehicles"] == 1
