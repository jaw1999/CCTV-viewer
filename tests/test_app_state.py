"""
Tests for AppState class - centralized application state management
"""
import pytest
import time
from unittest.mock import MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app_state import AppState, StreamConfig


class TestStreamConfig:
    """Tests for StreamConfig dataclass"""

    def test_default_values(self):
        """Test StreamConfig has correct defaults"""
        config = StreamConfig()
        assert config.enabled is False
        assert config.format == "cot"
        assert config.ip == "127.0.0.1"
        assert config.port == 8087
        assert config.latticeToken == ""
        assert config.latticeIntegration == "taiwan-cctv"

    def test_to_dict(self):
        """Test StreamConfig.to_dict() returns correct dict"""
        config = StreamConfig(enabled=True, format="lattice", ip="192.168.1.1")
        result = config.to_dict()

        assert result["enabled"] is True
        assert result["format"] == "lattice"
        assert result["ip"] == "192.168.1.1"
        assert result["port"] == 8087

    def test_from_dict(self):
        """Test StreamConfig.from_dict() creates correct object"""
        data = {
            "enabled": True,
            "format": "lattice",
            "ip": "10.0.0.1",
            "port": 9000,
            "latticeToken": "test-token"
        }
        config = StreamConfig.from_dict(data)

        assert config.enabled is True
        assert config.format == "lattice"
        assert config.ip == "10.0.0.1"
        assert config.port == 9000
        assert config.latticeToken == "test-token"


class TestAppState:
    """Tests for AppState class"""

    def test_initialization(self):
        """Test AppState initializes with correct defaults"""
        state = AppState()

        assert state.feeds_data == []
        assert state.yolo_model is None
        assert state.db_manager is None
        assert state.device == "cpu"
        assert state.cycle_counter == 0

    def test_feeds_data_thread_safety(self):
        """Test that feeds_data getter returns a copy"""
        state = AppState()
        original = [{"id": "test"}]
        state.feeds_data = original

        # Get feeds and modify
        retrieved = state.feeds_data
        retrieved.append({"id": "modified"})

        # Original should be unchanged
        assert len(state.feeds_data) == 1

    def test_get_feed_by_id(self):
        """Test get_feed_by_id returns correct feed"""
        state = AppState()
        state.feeds_data = [
            {"id": "feed-1", "name": "First"},
            {"id": "feed-2", "name": "Second"},
            {"id": "feed-3", "name": "Third"},
        ]

        feed = state.get_feed_by_id("feed-2")
        assert feed is not None
        assert feed["name"] == "Second"

        # Non-existent feed
        assert state.get_feed_by_id("non-existent") is None

    def test_feeds_count(self):
        """Test feeds_count property"""
        state = AppState()
        assert state.feeds_count == 0

        state.feeds_data = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        assert state.feeds_count == 3

    def test_cycle_counter(self):
        """Test cycle counter increment"""
        state = AppState()
        assert state.cycle_counter == 0

        result = state.increment_cycle()
        assert result == 1
        assert state.cycle_counter == 1

        state.increment_cycle()
        state.increment_cycle()
        assert state.cycle_counter == 3

    def test_last_update(self):
        """Test last_update property"""
        state = AppState()
        assert state.last_update == 0

        current_time = time.time()
        state.last_update = current_time
        assert state.last_update == current_time

    def test_retry_backoff(self):
        """Test feed retry backoff logic"""
        state = AppState()
        current_time = time.time()

        # Initially, all feeds should be retried
        assert state.should_retry_feed("feed-1", current_time) is True

        # Set backoff
        state.set_feed_backoff("feed-1", current_time, 60)

        # Should not retry immediately
        assert state.should_retry_feed("feed-1", current_time + 30) is False

        # Should retry after backoff period
        assert state.should_retry_feed("feed-1", current_time + 61) is True

    def test_clear_feed_backoff(self):
        """Test clearing feed backoff"""
        state = AppState()
        current_time = time.time()

        state.set_feed_backoff("feed-1", current_time, 60)
        assert state.should_retry_feed("feed-1", current_time + 30) is False

        state.clear_feed_backoff("feed-1")
        assert state.should_retry_feed("feed-1", current_time + 30) is True

    def test_exponential_backoff_interval(self):
        """Test get_backoff_interval returns exponential values"""
        state = AppState()

        # First call returns default
        interval1 = state.get_backoff_interval("feed-1", default=15.0)
        assert interval1 == 15.0

        # Second call returns doubled value
        interval2 = state.get_backoff_interval("feed-1", default=15.0)
        assert interval2 == 30.0

        # Third call returns doubled again
        interval3 = state.get_backoff_interval("feed-1", default=15.0)
        assert interval3 == 60.0

    def test_backoff_max_limit(self):
        """Test backoff interval respects max limit"""
        state = AppState()

        # Force high backoff
        for _ in range(10):
            state.get_backoff_interval("feed-1", default=100.0, max_backoff=300.0)

        # Should be capped at max
        interval = state.get_backoff_interval("feed-1", default=100.0, max_backoff=300.0)
        assert interval == 300.0

    def test_stream_config_update(self):
        """Test stream config update with audit logging"""
        state = AppState()

        # Update config
        updated = state.update_stream_config(
            enabled=True,
            format="lattice",
            ip="10.0.0.1"
        )

        assert updated.enabled is True
        assert updated.format == "lattice"
        assert updated.ip == "10.0.0.1"

        # Check audit log
        audit_log = state.get_audit_log(limit=10)
        assert len(audit_log) == 1
        assert audit_log[0]["action"] == "stream_config_update"

    def test_audit_log_limit(self):
        """Test audit log respects limit"""
        state = AppState()

        # Add many entries
        for i in range(20):
            state._log_audit(f"test_action_{i}", {"index": i})

        # Should only get requested limit
        log = state.get_audit_log(limit=5)
        assert len(log) == 5

    def test_get_stats(self):
        """Test get_stats returns comprehensive stats"""
        state = AppState()
        state.feeds_data = [{"id": "1"}, {"id": "2"}]
        state.device = "cuda"

        # Mock FeedCache
        from core.cache import FeedCache
        state.feed_cache = FeedCache(max_feeds=100, max_size_mb=10)

        stats = state.get_stats()

        assert "totalFeeds" in stats
        assert stats["totalFeeds"] == 2
        assert stats["device"] == "cuda"
        assert "cachedFeeds" in stats
        assert "cacheSize" in stats

    def test_initialize_feed_cache(self):
        """Test FeedCache initialization"""
        state = AppState()
        assert state.feed_cache is None

        state.initialize_feed_cache(max_feeds=1000, max_size_mb=500)

        assert state.feed_cache is not None
        assert state.feed_cache._image_cache.max_items == 1000

    def test_is_initialized(self):
        """Test is_initialized check"""
        state = AppState()
        assert state.is_initialized() is False

        # Add YOLO model mock
        state.yolo_model = MagicMock()
        assert state.is_initialized() is False

        # Add feed cache
        state.initialize_feed_cache()
        assert state.is_initialized() is False

        # Add feeds
        state.feeds_data = [{"id": "test"}]
        assert state.is_initialized() is True


class TestAppStateCacheIntegration:
    """Tests for AppState with FeedCache integration"""

    def test_cache_image(self):
        """Test caching and retrieving images"""
        state = AppState()
        state.initialize_feed_cache()

        test_image = b"fake_jpeg_data_here"
        state.cache_image("feed-1", test_image, is_working=True, has_vehicles=True)

        retrieved = state.get_cached_image("feed-1")
        assert retrieved == test_image

    def test_image_change_detection(self):
        """Test image change detection"""
        state = AppState()
        state.initialize_feed_cache()

        image1 = b"image_version_1"
        image2 = b"image_version_2"

        # First image - should be marked as changed
        assert state.has_image_changed("feed-1", image1) is True

        # Cache it
        state.cache_image("feed-1", image1)

        # Same image - should not be changed
        assert state.has_image_changed("feed-1", image1) is False

        # Different image - should be changed
        assert state.has_image_changed("feed-1", image2) is True

    def test_feed_status(self):
        """Test feed status get/set"""
        state = AppState()
        state.initialize_feed_cache()

        # Default is False
        assert state.get_feed_status("feed-1") is False

        state.set_feed_status("feed-1", True)
        assert state.get_feed_status("feed-1") is True

        state.set_feed_status("feed-1", False)
        assert state.get_feed_status("feed-1") is False

    def test_vehicle_detected(self):
        """Test vehicle detection status get/set"""
        state = AppState()
        state.initialize_feed_cache()

        # Default is False
        assert state.get_vehicle_detected("feed-1") is False

        state.set_vehicle_detected("feed-1", True)
        assert state.get_vehicle_detected("feed-1") is True

    def test_get_all_statuses(self):
        """Test getting all feed statuses"""
        state = AppState()
        state.initialize_feed_cache()

        state.set_feed_status("feed-1", True)
        state.set_feed_status("feed-2", False)
        state.set_feed_status("feed-3", True)

        all_status = state.get_all_feed_status()
        assert all_status["feed-1"] is True
        assert all_status["feed-2"] is False
        assert all_status["feed-3"] is True

    def test_cache_stats(self):
        """Test cache statistics"""
        state = AppState()
        state.initialize_feed_cache()

        # Cache some data
        state.cache_image("feed-1", b"a" * 1000)
        state.cache_image("feed-2", b"b" * 2000)

        stats = state.get_cache_stats()
        assert stats["items"] == 2
        assert stats["size_bytes"] >= 3000
