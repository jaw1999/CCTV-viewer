"""
Pytest fixtures and configuration for CCTV Viewer tests
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, AsyncMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ============================================================================
# Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Environment Configuration
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    os.environ["CCTV_ENV"] = "test"
    os.environ["CCTV_AUTH_ENABLED"] = "false"  # Disable auth for tests
    os.environ["CCTV_SSL_VERIFY"] = "false"  # Disable SSL for tests
    os.environ["CCTV_DEBUG"] = "true"
    yield
    # Cleanup
    for key in ["CCTV_ENV", "CCTV_AUTH_ENABLED", "CCTV_SSL_VERIFY", "CCTV_DEBUG"]:
        os.environ.pop(key, None)


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def test_db_manager():
    """Create a test database manager with in-memory SQLite"""
    from database import DatabaseManager

    db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await db_manager.init_db()
    yield db_manager
    await db_manager.close()


@pytest_asyncio.fixture
async def db_session(test_db_manager):
    """Provide a database session for tests"""
    async with test_db_manager.session() as session:
        yield session


# ============================================================================
# Tracker Fixtures
# ============================================================================

@pytest.fixture
def tracker_manager():
    """Create a tracker manager for tests"""
    from tracker import TrackerManager

    return TrackerManager(max_age=30, min_hits=3, iou_threshold=0.3)


@pytest.fixture
def vehicle_tracker():
    """Create a single vehicle tracker for tests"""
    from tracker import VehicleTracker

    return VehicleTracker(max_age=30, min_hits=3, iou_threshold=0.3)


# ============================================================================
# WebSocket Fixtures
# ============================================================================

@pytest.fixture
def ws_manager():
    """Create a WebSocket connection manager for tests"""
    from websocket_manager import ConnectionManager

    return ConnectionManager()


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket for tests"""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.close = AsyncMock()
    return ws


# ============================================================================
# HTTP Client Fixtures
# ============================================================================

@pytest.fixture
def test_client():
    """Create a synchronous test client for the FastAPI app"""
    # Import here to avoid circular imports
    from main import app

    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture
async def async_client():
    """Create an async test client for the FastAPI app"""
    from main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def sample_feed_data():
    """Sample feed data for tests"""
    return {
        "id": "TEST-001",
        "streamUrl": "http://example.com/stream",
        "imageUrl": "http://example.com/image.jpg",
        "description": "Test Camera",
        "roadName": "Test Road",
        "locationMile": "10K",
        "lat": "25.0330",
        "lon": "121.5654",
        "direction": "N",
    }


@pytest.fixture
def sample_feeds_list(sample_feed_data):
    """List of sample feeds for tests"""
    feeds = []
    for i in range(5):
        feed = sample_feed_data.copy()
        feed["id"] = f"TEST-{i:03d}"
        feed["lat"] = str(25.0 + i * 0.01)
        feed["lon"] = str(121.5 + i * 0.01)
        feeds.append(feed)
    return feeds


@pytest.fixture
def sample_detection_data():
    """Sample detection data for tests"""
    return {
        "vehicle_count": 3,
        "vehicle_types": ["car", "bus", "truck"],
        "confidence_avg": 0.85,
        "tracked_vehicles": [
            {"track_id": 1, "class": "car", "confidence": 0.9, "bbox": [100, 100, 200, 200]},
            {"track_id": 2, "class": "bus", "confidence": 0.85, "bbox": [300, 100, 500, 300]},
            {"track_id": 3, "class": "truck", "confidence": 0.8, "bbox": [100, 300, 300, 500]},
        ],
        "track_counts": {"car": 1, "bus": 1, "truck": 1},
    }


@pytest.fixture
def sample_yolo_results():
    """Mock YOLO results for tests"""
    import numpy as np

    class MockBox:
        def __init__(self, class_id, confidence, bbox):
            self.cls = [class_id]
            self.conf = [confidence]
            self._bbox = bbox

        @property
        def xyxy(self):
            class MockTensor:
                def __init__(self, data):
                    self.data = np.array(data)

                def cpu(self):
                    return self

                def numpy(self):
                    return self.data

            return [MockTensor(self._bbox)]

    class MockBoxes:
        def __init__(self, boxes):
            self._boxes = boxes

        def __len__(self):
            return len(self._boxes)

        def __iter__(self):
            return iter(self._boxes)

    class MockResults:
        def __init__(self):
            self.boxes = MockBoxes([
                MockBox(2, 0.9, [100, 100, 200, 200]),  # car
                MockBox(5, 0.85, [300, 100, 500, 300]),  # bus
                MockBox(7, 0.8, [100, 300, 300, 500]),  # truck
            ])

        def plot(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

    return MockResults()


# ============================================================================
# Stream Configuration Fixtures
# ============================================================================

@pytest.fixture
def sample_stream_config():
    """Sample stream configuration for tests"""
    return {
        "enabled": True,
        "format": "cot",
        "ip": "127.0.0.1",
        "port": 8087,
        "latticeToken": "",
        "latticeSandboxToken": "",
        "latticeIntegration": "test-integration",
        "latticeUrl": "",
    }


# ============================================================================
# Image Fixtures
# ============================================================================

@pytest.fixture
def sample_image_bytes():
    """Generate sample JPEG image bytes for tests"""
    from PIL import Image
    from io import BytesIO

    img = Image.new("RGB", (640, 480), color="blue")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


# ============================================================================
# Helper Functions
# ============================================================================

def assert_valid_error_response(response_json: dict):
    """Assert that a response is a valid error response"""
    assert "error" in response_json
    assert "message" in response_json
    assert "timestamp" in response_json


def assert_valid_feed(feed: dict):
    """Assert that a feed dictionary has required fields"""
    required_fields = ["id"]
    for field in required_fields:
        assert field in feed, f"Missing required field: {field}"
