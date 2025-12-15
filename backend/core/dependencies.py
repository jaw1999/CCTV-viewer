"""
FastAPI dependency injection for CCTV Viewer
"""
from typing import Optional, AsyncGenerator
from dataclasses import dataclass, field
from functools import lru_cache

from fastapi import Depends, Request

# Import components
try:
    from ..database import DatabaseManager
    from ..tracker import TrackerManager
    from ..websocket_manager import ConnectionManager
    from ..observability import MetricsCollector, HealthChecker, AlertManager, StructuredLogger
    from .config import get_settings, Settings
except ImportError:
    from database import DatabaseManager
    from tracker import TrackerManager
    from websocket_manager import ConnectionManager
    from observability import MetricsCollector, HealthChecker, AlertManager, StructuredLogger
    from core.config import get_settings, Settings


@dataclass
class AppState:
    """
    Application state container.
    Holds all shared state and managers.
    """
    # Data
    feeds_data: list = field(default_factory=list)
    feed_cache: dict = field(default_factory=dict)
    feed_status: dict = field(default_factory=dict)
    feed_vehicle_detected: dict = field(default_factory=dict)
    feed_image_hash: dict = field(default_factory=dict)
    feed_retry_after: dict = field(default_factory=dict)

    # Timing
    last_update: float = 0
    cycle_counter: int = 0

    # Components (initialized later)
    yolo_model: Optional[object] = None
    db_manager: Optional[DatabaseManager] = None
    tracker_manager: Optional[TrackerManager] = None
    ws_manager: Optional[ConnectionManager] = None
    http_client: Optional[object] = None

    # Stream config
    stream_config: dict = field(default_factory=lambda: {
        "enabled": False,
        "format": "cot",
        "ip": "127.0.0.1",
        "port": 8087,
        "latticeToken": "",
        "latticeSandboxToken": "",
        "latticeIntegration": "taiwan-cctv",
        "latticeUrl": ""
    })


# Global app state instance
_app_state: Optional[AppState] = None


def get_app_state() -> AppState:
    """Get the global application state"""
    global _app_state
    if _app_state is None:
        _app_state = AppState()
    return _app_state


def set_app_state(state: AppState):
    """Set the global application state"""
    global _app_state
    _app_state = state


# Dependency functions for FastAPI

async def get_db_manager(request: Request) -> Optional[DatabaseManager]:
    """
    Dependency to get database manager.
    Returns None if database is not enabled.
    """
    app_state = get_app_state()
    return app_state.db_manager


async def get_tracker_manager(request: Request) -> Optional[TrackerManager]:
    """
    Dependency to get tracker manager.
    Returns None if tracking is not enabled.
    """
    app_state = get_app_state()
    return app_state.tracker_manager


async def get_ws_manager(request: Request) -> Optional[ConnectionManager]:
    """
    Dependency to get WebSocket manager.
    """
    app_state = get_app_state()
    return app_state.ws_manager


@lru_cache()
def get_metrics() -> MetricsCollector:
    """Get metrics collector instance"""
    return MetricsCollector()


@lru_cache()
def get_logger() -> StructuredLogger:
    """Get structured logger instance"""
    settings = get_settings()
    return StructuredLogger("cctv.main", log_level=settings.logging.level)


def get_health_checker(request: Request) -> Optional[HealthChecker]:
    """
    Dependency to get health checker.
    """
    return getattr(request.app.state, 'health_checker', None)


@lru_cache()
def get_alert_manager() -> AlertManager:
    """Get alert manager instance"""
    from .config import CONFIG
    return AlertManager(CONFIG)


# Database session dependency

async def get_db_session(
    db_manager: Optional[DatabaseManager] = Depends(get_db_manager)
) -> AsyncGenerator:
    """
    Dependency to get database session.
    Yields a session within a transaction context.
    """
    if db_manager is None:
        yield None
        return

    async with db_manager.session() as session:
        yield session


# Feed data dependencies

def get_feeds_data() -> list:
    """Get current feeds data"""
    return get_app_state().feeds_data


def get_feed_cache() -> dict:
    """Get current feed cache"""
    return get_app_state().feed_cache


def get_feed_status() -> dict:
    """Get current feed status"""
    return get_app_state().feed_status


def get_feed_vehicle_detected() -> dict:
    """Get current vehicle detection status"""
    return get_app_state().feed_vehicle_detected


def get_stream_config() -> dict:
    """Get current stream configuration"""
    return get_app_state().stream_config


# Validation dependencies

def require_database(
    db_manager: Optional[DatabaseManager] = Depends(get_db_manager)
) -> DatabaseManager:
    """
    Dependency that requires database to be available.
    Raises HTTPException if database is not enabled.
    """
    from fastapi import HTTPException

    if db_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not enabled"
        )
    return db_manager


def require_tracker(
    tracker_manager: Optional[TrackerManager] = Depends(get_tracker_manager)
) -> TrackerManager:
    """
    Dependency that requires tracker to be available.
    Raises HTTPException if tracking is not enabled.
    """
    from fastapi import HTTPException

    if tracker_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Vehicle tracking is not enabled"
        )
    return tracker_manager
