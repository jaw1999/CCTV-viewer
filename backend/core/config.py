"""
Configuration management for CCTV Viewer
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Optional, List

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    if not config_path.exists():
        print(f"Warning: config.yaml not found at {config_path}, using defaults")
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Configuration loaded from {config_path}")
    return config


# Load config at module level
CONFIG = load_config()


class DetectionSettings(BaseModel):
    """Detection configuration"""
    confidence_threshold: float = 0.6
    min_box_size: int = 20
    vehicle_classes: List[int] = [2, 5, 7]  # car, bus, truck


class TrackingSettings(BaseModel):
    """Tracking configuration"""
    enabled: bool = True
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


class PerformanceSettings(BaseModel):
    """Performance configuration"""
    batch_size: int = 50
    worker_threads: int = 8
    selective_skip_interval: int = 2


class HttpSettings(BaseModel):
    """HTTP client configuration"""
    max_connections: int = 300
    max_keepalive: int = 250
    keepalive_expiry: int = 60
    timeout: float = 15.0
    connect_timeout: float = 5.0


class DatabaseSettings(BaseModel):
    """Database configuration"""
    enabled: bool = True
    type: str = "sqlite"
    path: str = "./data/cctv_data.db"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None


class WebSocketSettings(BaseModel):
    """WebSocket configuration"""
    enabled: bool = True
    heartbeat_interval: int = 30
    max_connections: int = 100


class APISettings(BaseModel):
    """API configuration"""
    enable_cors: bool = True
    enable_gzip: bool = True
    rate_limit: int = 100
    burst_size: int = 20
    rate_limit_enabled: bool = True


class AlertSettings(BaseModel):
    """Alert configuration"""
    enabled: bool = False
    cooldown_seconds: int = 300


class LoggingSettings(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    file: Optional[str] = None
    max_size_mb: int = 100
    backup_count: int = 5


class StreamOutSettings(BaseModel):
    """Stream out configuration"""
    enabled: bool = False
    format: str = "cot"


class Settings(BaseSettings):
    """Application settings from environment and config"""

    # Environment
    env: str = Field(default="development", alias="CCTV_ENV")
    debug: bool = Field(default=False, alias="CCTV_DEBUG")

    # Security
    auth_enabled: bool = Field(default=True, alias="CCTV_AUTH_ENABLED")
    api_keys: str = Field(default="", alias="CCTV_API_KEYS")
    ssl_verify: bool = Field(default=True, alias="CCTV_SSL_VERIFY")

    # CORS
    cors_origins: str = Field(default="", alias="CCTV_CORS_ORIGINS")

    # Server
    host: str = Field(default="0.0.0.0", alias="CCTV_HOST")
    port: int = Field(default=8001, alias="CCTV_PORT")

    # Detection
    detection: DetectionSettings = Field(default_factory=DetectionSettings)

    # Tracking
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)

    # Performance
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)

    # HTTP
    http: HttpSettings = Field(default_factory=HttpSettings)

    # Database
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    # WebSocket
    websocket: WebSocketSettings = Field(default_factory=WebSocketSettings)

    # API
    api: APISettings = Field(default_factory=APISettings)

    # Alerts
    alerts: AlertSettings = Field(default_factory=AlertSettings)

    # Logging
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Stream Out
    stream_out: StreamOutSettings = Field(default_factory=StreamOutSettings)

    class Config:
        env_prefix = "CCTV_"
        env_file = ".env"
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_from_yaml()

    def _load_from_yaml(self):
        """Override settings from config.yaml"""
        if not CONFIG:
            return

        # Detection
        if "detection" in CONFIG:
            det_config = CONFIG["detection"]
            self.detection = DetectionSettings(
                confidence_threshold=det_config.get("confidence_threshold", 0.6),
                min_box_size=det_config.get("min_box_size", 20),
                vehicle_classes=det_config.get("vehicle_classes", [2, 5, 7]),
            )
            if "tracking" in det_config:
                track_config = det_config["tracking"]
                self.tracking = TrackingSettings(
                    enabled=track_config.get("enabled", True),
                    max_age=track_config.get("max_age", 30),
                    min_hits=track_config.get("min_hits", 3),
                    iou_threshold=track_config.get("iou_threshold", 0.3),
                )

        # Performance
        if "performance" in CONFIG:
            perf_config = CONFIG["performance"]
            self.performance = PerformanceSettings(
                batch_size=perf_config.get("batch_size", 50),
                worker_threads=perf_config.get("worker_threads", 8),
                selective_skip_interval=perf_config.get("selective_skip_interval", 2),
            )
            if "http" in perf_config:
                http_config = perf_config["http"]
                self.http = HttpSettings(
                    max_connections=http_config.get("max_connections", 300),
                    max_keepalive=http_config.get("max_keepalive", 250),
                    keepalive_expiry=http_config.get("keepalive_expiry", 60),
                    timeout=http_config.get("timeout", 15.0),
                    connect_timeout=http_config.get("connect_timeout", 5.0),
                )

        # Database
        if "database" in CONFIG:
            db_config = CONFIG["database"]
            self.database = DatabaseSettings(
                enabled=db_config.get("enabled", True),
                type=db_config.get("type", "sqlite"),
                path=db_config.get("path", "./data/cctv_data.db"),
                host=db_config.get("host"),
                port=db_config.get("port"),
                username=db_config.get("username"),
                password=db_config.get("password"),
                database=db_config.get("database"),
            )

        # WebSocket
        if "websocket" in CONFIG:
            ws_config = CONFIG["websocket"]
            self.websocket = WebSocketSettings(
                enabled=ws_config.get("enabled", True),
                heartbeat_interval=ws_config.get("heartbeat_interval", 30),
                max_connections=ws_config.get("max_connections", 100),
            )

        # API
        if "api" in CONFIG:
            api_config = CONFIG["api"]
            self.api = APISettings(
                enable_cors=api_config.get("enable_cors", True),
                enable_gzip=api_config.get("enable_gzip", True),
                rate_limit=api_config.get("rate_limit", 100),
                burst_size=api_config.get("burst_size", 20),
                rate_limit_enabled=api_config.get("rate_limit_enabled", True),
            )

        # Alerts
        if "alerts" in CONFIG:
            alert_config = CONFIG["alerts"]
            self.alerts = AlertSettings(
                enabled=alert_config.get("enabled", False),
                cooldown_seconds=alert_config.get("cooldown_seconds", 300),
            )

        # Logging
        if "logging" in CONFIG:
            log_config = CONFIG["logging"]
            self.logging = LoggingSettings(
                level=log_config.get("level", "INFO"),
                format=log_config.get("format", "json"),
                file=log_config.get("file"),
                max_size_mb=log_config.get("max_size_mb", 100),
                backup_count=log_config.get("backup_count", 5),
            )

        # Stream Out
        if "stream_out" in CONFIG:
            stream_config = CONFIG["stream_out"]
            self.stream_out = StreamOutSettings(
                enabled=stream_config.get("enabled", False),
                format=stream_config.get("format", "cot"),
            )

    @property
    def database_url(self) -> str:
        """Generate database URL from settings"""
        if self.database.type == "sqlite":
            return f"sqlite+aiosqlite:///{self.database.path}"
        elif self.database.type == "postgresql":
            return (
                f"postgresql+asyncpg://{self.database.username}:{self.database.password}"
                f"@{self.database.host}:{self.database.port}/{self.database.database}"
            )
        elif self.database.type == "mysql":
            return (
                f"mysql+aiomysql://{self.database.username}:{self.database.password}"
                f"@{self.database.host}:{self.database.port}/{self.database.database}"
            )
        else:
            raise ValueError(f"Unsupported database type: {self.database.type}")

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        if not self.cors_origins:
            return ["*"] if self.env != "production" else []
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def api_keys_list(self) -> List[str]:
        """Parse API keys from comma-separated string"""
        if not self.api_keys:
            return []
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
