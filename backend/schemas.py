"""
Pydantic schemas for request/response validation
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re


# ============================================================================
# Stream Configuration Schemas
# ============================================================================

class StreamConfigRequest(BaseModel):
    """Request schema for stream configuration"""
    model_config = ConfigDict(str_strip_whitespace=True)

    enabled: bool = Field(description="Enable or disable streaming")
    format: Literal["cot", "lattice", "chatsurfer"] = Field(
        description="Stream format: 'cot' for Cursor on Target, 'lattice' for Anduril Lattice, or 'chatsurfer' for ChatSurfer"
    )
    ip: str = Field(
        default="127.0.0.1",
        description="IP address for CoT UDP streaming"
    )
    port: int = Field(
        default=8087,
        ge=1,
        le=65535,
        description="Port for CoT UDP streaming"
    )
    latticeToken: Optional[str] = Field(
        default="",
        description="Lattice environment token"
    )
    latticeSandboxToken: Optional[str] = Field(
        default="",
        description="Lattice sandbox token"
    )
    latticeIntegration: Optional[str] = Field(
        default="taiwan-cctv",
        description="Lattice integration name"
    )
    latticeUrl: Optional[str] = Field(
        default="",
        description="Lattice API URL"
    )
    # ChatSurfer fields
    chatsurferSession: Optional[str] = Field(
        default="",
        description="ChatSurfer SESSION cookie value"
    )
    chatsurferRoom: Optional[str] = Field(
        default="",
        description="ChatSurfer room name"
    )
    chatsurferNickname: Optional[str] = Field(
        default="CCTV_Bot",
        description="ChatSurfer bot nickname"
    )
    chatsurferDomain: Optional[str] = Field(
        default="chatsurferxmppunclass",
        description="ChatSurfer domain ID"
    )

    @field_validator("ip")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        """Validate IP address format"""
        # Allow localhost
        if v in ("localhost", "127.0.0.1", "0.0.0.0"):
            return v
        # Validate IPv4 format
        ip_pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        if not re.match(ip_pattern, v):
            raise ValueError("Invalid IP address format")
        return v

    @field_validator("latticeUrl")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate Lattice URL format"""
        if not v:
            return v
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class StreamConfigResponse(BaseModel):
    """Response schema for stream configuration"""
    status: str
    config: Dict[str, Any]


# ============================================================================
# Search Schemas
# ============================================================================

class SearchRequest(BaseModel):
    """Request schema for feed search"""
    model_config = ConfigDict(str_strip_whitespace=True)

    road: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Filter by road name (partial match)"
    )
    has_vehicles: Optional[bool] = Field(
        default=None,
        description="Filter by vehicle detection status"
    )
    lat_min: Optional[float] = Field(
        default=None,
        ge=-90,
        le=90,
        description="Minimum latitude for bounding box"
    )
    lat_max: Optional[float] = Field(
        default=None,
        ge=-90,
        le=90,
        description="Maximum latitude for bounding box"
    )
    lon_min: Optional[float] = Field(
        default=None,
        ge=-180,
        le=180,
        description="Minimum longitude for bounding box"
    )
    lon_max: Optional[float] = Field(
        default=None,
        ge=-180,
        le=180,
        description="Maximum longitude for bounding box"
    )

    @field_validator("lat_max")
    @classmethod
    def validate_lat_range(cls, v: Optional[float], info) -> Optional[float]:
        """Ensure lat_max > lat_min"""
        if v is not None and info.data.get("lat_min") is not None:
            if v <= info.data["lat_min"]:
                raise ValueError("lat_max must be greater than lat_min")
        return v

    @field_validator("lon_max")
    @classmethod
    def validate_lon_range(cls, v: Optional[float], info) -> Optional[float]:
        """Ensure lon_max > lon_min"""
        if v is not None and info.data.get("lon_min") is not None:
            if v <= info.data["lon_min"]:
                raise ValueError("lon_max must be greater than lon_min")
        return v


class FeedSearchResult(BaseModel):
    """Single feed in search results"""
    id: str
    roadName: Optional[str] = None
    locationMile: Optional[str] = None
    description: Optional[str] = None
    lat: Optional[str] = None
    lon: Optional[str] = None
    direction: Optional[str] = None
    status: bool = False
    vehicleDetected: bool = False


class SearchResponse(BaseModel):
    """Response schema for feed search"""
    query: Dict[str, Any]
    count: int
    feeds: List[FeedSearchResult]


# ============================================================================
# Feed History Schemas
# ============================================================================

class FeedHistoryRequest(BaseModel):
    """Request schema for feed history"""
    hours: int = Field(
        default=24,
        ge=1,
        le=168,  # Max 1 week
        description="Number of hours of history to retrieve"
    )


class DetectionRecord(BaseModel):
    """Single detection record"""
    id: int
    timestamp: datetime
    vehicle_count: int
    vehicle_types: List[str]
    confidence_avg: float


class FeedHistoryResponse(BaseModel):
    """Response schema for feed history"""
    feed_id: str
    hours: int
    detections: List[DetectionRecord]


# ============================================================================
# Feed Stats Schemas
# ============================================================================

class FeedStatsRequest(BaseModel):
    """Request schema for feed statistics"""
    days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Number of days for statistics"
    )


class FeedStatsResponse(BaseModel):
    """Response schema for feed statistics"""
    feed_id: str
    total_detections: int
    total_vehicles: int
    avg_vehicles_per_detection: float
    period_days: int


# ============================================================================
# WebSocket Message Schemas
# ============================================================================

class WebSocketMessage(BaseModel):
    """Base WebSocket message schema"""
    action: Literal["subscribe", "unsubscribe", "ping"]
    feed_id: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Feed ID for subscribe/unsubscribe actions"
    )

    @field_validator("feed_id")
    @classmethod
    def validate_feed_id(cls, v: Optional[str], info) -> Optional[str]:
        """Validate feed_id is provided for subscribe/unsubscribe"""
        action = info.data.get("action")
        if action in ("subscribe", "unsubscribe") and not v:
            raise ValueError(f"feed_id is required for {action} action")
        return v


class WebSocketResponse(BaseModel):
    """WebSocket response message"""
    type: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    feed_id: Optional[str] = None
    message: Optional[str] = None


# ============================================================================
# Health Check Schemas
# ============================================================================

class ComponentHealth(BaseModel):
    """Health status of a single component"""
    component: str
    status: Literal["healthy", "degraded", "unhealthy"]
    message: str
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    last_check: Optional[datetime] = None


class SystemMetrics(BaseModel):
    """System resource metrics"""
    cpu_percent: float
    memory: Dict[str, float]
    disk: Dict[str, float]
    python_process: Dict[str, float]


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime
    duration_ms: float
    components: List[ComponentHealth]
    system: SystemMetrics


# ============================================================================
# API Stats Schemas
# ============================================================================

class WebSocketStats(BaseModel):
    """WebSocket connection statistics"""
    total_connections: int
    feed_subscriptions: Dict[str, int]


class StatsResponse(BaseModel):
    """Response schema for system statistics"""
    totalFeeds: int
    cachedFeeds: int
    workingFeeds: int
    offlineFeeds: int
    vehiclesDetectedFeeds: int
    lastUpdate: float
    cacheSize: float  # MB
    websocket: Optional[WebSocketStats] = None


# ============================================================================
# Map Data Schemas
# ============================================================================

class MapFeatureProperties(BaseModel):
    """Properties for GeoJSON feature"""
    id: str
    roadName: str
    locationMile: str
    description: str
    direction: str
    isWorking: bool
    hasVehicles: bool


class MapFeature(BaseModel):
    """GeoJSON feature for map display"""
    type: Literal["Feature"] = "Feature"
    geometry: Dict[str, Any]
    properties: MapFeatureProperties


class MapResponse(BaseModel):
    """GeoJSON FeatureCollection for map"""
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[MapFeature]


# ============================================================================
# Error Response Schemas
# ============================================================================

class ErrorDetail(BaseModel):
    """Validation error detail"""
    field: str
    message: str
    type: str


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    message: str
    timestamp: datetime
    details: Optional[List[ErrorDetail]] = None
    status_code: Optional[int] = None
