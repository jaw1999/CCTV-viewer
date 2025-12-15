#!/usr/bin/env python3
"""
Observability Module for CCTV Viewer
Provides health checks, metrics, logging, and alerting capabilities
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
import psutil
import json

from prometheus_client import Counter, Gauge, Histogram, Summary, Info
from sqlalchemy import select, func

# Import database models
try:
    from .database import Feed as FeedModel, Detection as DetectionModel, VehicleTrack as VehicleTrackModel
except ImportError:
    from database import Feed as FeedModel, Detection as DetectionModel, VehicleTrack as VehicleTrackModel


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentType(str, Enum):
    """System component types"""
    DATABASE = "database"
    YOLO_MODEL = "yolo_model"
    HTTP_CLIENT = "http_client"
    WEBSOCKET = "websocket"
    CACHE = "cache"
    TRACKER = "tracker"
    FEED_SOURCE = "feed_source"


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component: str
    status: HealthStatus
    message: str
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    last_check: str = None

    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now(timezone.utc).isoformat()

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


class MetricsCollector:
    """Prometheus metrics collector for operational monitoring"""

    def __init__(self):
        # Detection metrics
        self.detections_total = Counter(
            'cctv_detections_total',
            'Total number of vehicle detections',
            ['feed_id', 'vehicle_class']
        )

        self.detection_confidence = Histogram(
            'cctv_detection_confidence',
            'Vehicle detection confidence scores',
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        )

        # Performance metrics
        self.yolo_inference_duration = Histogram(
            'cctv_yolo_inference_seconds',
            'YOLO model inference duration',
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        )

        self.feed_fetch_duration = Histogram(
            'cctv_feed_fetch_seconds',
            'Feed image fetch duration',
            ['status']
        )

        self.cycle_duration = Summary(
            'cctv_cycle_duration_seconds',
            'Detection cycle duration'
        )

        # System metrics
        self.feeds_total = Gauge(
            'cctv_feeds_total',
            'Total number of feeds'
        )

        self.feeds_online = Gauge(
            'cctv_feeds_online',
            'Number of online feeds'
        )

        self.feeds_with_vehicles = Gauge(
            'cctv_feeds_with_vehicles',
            'Number of feeds with detected vehicles'
        )

        self.cache_size_bytes = Gauge(
            'cctv_cache_size_bytes',
            'Size of feed cache in bytes'
        )

        self.active_websockets = Gauge(
            'cctv_active_websockets',
            'Number of active WebSocket connections'
        )

        # Error metrics
        self.errors_total = Counter(
            'cctv_errors_total',
            'Total number of errors',
            ['component', 'error_type']
        )

        self.feed_failures = Counter(
            'cctv_feed_failures_total',
            'Total number of feed fetch failures',
            ['feed_id', 'reason']
        )

        # Database metrics
        self.db_query_duration = Histogram(
            'cctv_db_query_seconds',
            'Database query duration',
            ['operation']
        )

        self.db_connections_active = Gauge(
            'cctv_db_connections_active',
            'Number of active database connections'
        )

        # Track metrics
        self.tracks_total = Counter(
            'cctv_tracks_total',
            'Total number of vehicle tracks created',
            ['vehicle_class']
        )

        self.tracks_active = Gauge(
            'cctv_tracks_active',
            'Number of active vehicle tracks',
            ['feed_id']
        )

        # System info
        self.system_info = Info(
            'cctv_system',
            'System information'
        )


class HealthChecker:
    """Comprehensive health checking for all system components"""

    def __init__(self, app_state):
        self.app_state = app_state
        self.logger = logging.getLogger(__name__)

    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status"""
        start_time = time.time()

        # Run all checks concurrently
        checks = await asyncio.gather(
            self.check_database(),
            self.check_yolo_model(),
            self.check_http_client(),
            self.check_websocket(),
            self.check_cache(),
            self.check_tracker(),
            self.check_feed_source(),
            return_exceptions=True
        )

        # Process results
        components = []
        overall_status = HealthStatus.HEALTHY

        for check in checks:
            if isinstance(check, Exception):
                self.logger.error(f"Health check failed: {check}")
                components.append(ComponentHealth(
                    component="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=str(check)
                ))
                overall_status = HealthStatus.UNHEALTHY
            else:
                components.append(check)
                # Determine overall status
                if check.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif check.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED

        # Get system metrics
        system_metrics = self.get_system_metrics()

        duration_ms = (time.time() - start_time) * 1000

        return {
            "status": overall_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration_ms, 2),
            "components": [c.to_dict() for c in components],
            "system": system_metrics
        }

    async def check_database(self) -> ComponentHealth:
        """Check database connectivity and performance"""
        start_time = time.time()

        try:
            if not hasattr(self.app_state, 'db_manager') or self.app_state.db_manager is None:
                return ComponentHealth(
                    component=ComponentType.DATABASE.value,
                    status=HealthStatus.UNHEALTHY,
                    message="Database manager not initialized"
                )

            # Test database connection with a simple query
            async with self.app_state.db_manager.session() as session:
                result = await session.execute(select(func.count()).select_from(FeedModel))
                feed_count = result.scalar()

            latency_ms = (time.time() - start_time) * 1000

            # Check if latency is concerning
            if latency_ms > 1000:
                status = HealthStatus.DEGRADED
                message = f"Database slow (query: {latency_ms:.0f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = "Database operational"

            return ComponentHealth(
                component=ComponentType.DATABASE.value,
                status=status,
                message=message,
                latency_ms=round(latency_ms, 2),
                metadata={"feed_count": feed_count}
            )

        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return ComponentHealth(
                component=ComponentType.DATABASE.value,
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {str(e)}"
            )

    async def check_yolo_model(self) -> ComponentHealth:
        """Check YOLO model availability"""
        try:
            if not hasattr(self.app_state, 'yolo_model') or self.app_state.yolo_model is None:
                return ComponentHealth(
                    component=ComponentType.YOLO_MODEL.value,
                    status=HealthStatus.UNHEALTHY,
                    message="YOLO model not loaded"
                )

            # Check if model is accessible
            model_info = {
                "type": type(self.app_state.yolo_model).__name__,
                "device": str(getattr(self.app_state.yolo_model, 'device', 'unknown'))
            }

            return ComponentHealth(
                component=ComponentType.YOLO_MODEL.value,
                status=HealthStatus.HEALTHY,
                message="YOLO model loaded",
                metadata=model_info
            )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.YOLO_MODEL.value,
                status=HealthStatus.UNHEALTHY,
                message=f"Model error: {str(e)}"
            )

    async def check_http_client(self) -> ComponentHealth:
        """Check HTTP client connectivity"""
        start_time = time.time()

        try:
            # HTTP client is created on-demand in main.py, so it's normal to not have it initially
            # Check if we have any cached feeds (which means HTTP client has worked)
            has_cache = hasattr(self.app_state, 'feed_cache') and len(self.app_state.feed_cache) > 0

            if has_cache:
                # HTTP client is working fine since we have fetched feeds
                latency_ms = (time.time() - start_time) * 1000
                return ComponentHealth(
                    component=ComponentType.HTTP_CLIENT.value,
                    status=HealthStatus.HEALTHY,
                    message="HTTP client operational",
                    latency_ms=round(latency_ms, 2),
                    metadata={"cached_feeds": len(self.app_state.feed_cache)}
                )
            else:
                # Not degraded, just initializing
                return ComponentHealth(
                    component=ComponentType.HTTP_CLIENT.value,
                    status=HealthStatus.HEALTHY,
                    message="HTTP client initializing"
                )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.HTTP_CLIENT.value,
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP client error: {str(e)}"
            )

    async def check_websocket(self) -> ComponentHealth:
        """Check WebSocket manager status"""
        try:
            if not hasattr(self.app_state, 'ws_manager'):
                return ComponentHealth(
                    component=ComponentType.WEBSOCKET.value,
                    status=HealthStatus.DEGRADED,
                    message="WebSocket manager not found"
                )

            active_connections = len(self.app_state.ws_manager.active_connections)

            return ComponentHealth(
                component=ComponentType.WEBSOCKET.value,
                status=HealthStatus.HEALTHY,
                message="WebSocket manager operational",
                metadata={"active_connections": active_connections}
            )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.WEBSOCKET.value,
                status=HealthStatus.DEGRADED,
                message=f"WebSocket error: {str(e)}"
            )

    async def check_cache(self) -> ComponentHealth:
        """Check feed cache status"""
        try:
            if not hasattr(self.app_state, 'feed_cache'):
                return ComponentHealth(
                    component=ComponentType.CACHE.value,
                    status=HealthStatus.DEGRADED,
                    message="Cache not initialized"
                )

            cache_size = sum(len(v) for v in self.app_state.feed_cache.values())
            cache_items = len(self.app_state.feed_cache)
            cache_size_mb = cache_size / (1024 * 1024)

            # Warn if cache is very large
            if cache_size_mb > 1000:  # 1 GB
                status = HealthStatus.DEGRADED
                message = f"Cache large ({cache_size_mb:.0f}MB)"
            else:
                status = HealthStatus.HEALTHY
                message = "Cache operational"

            return ComponentHealth(
                component=ComponentType.CACHE.value,
                status=status,
                message=message,
                metadata={
                    "size_mb": round(cache_size_mb, 2),
                    "items": cache_items
                }
            )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.CACHE.value,
                status=HealthStatus.DEGRADED,
                message=f"Cache error: {str(e)}"
            )

    async def check_tracker(self) -> ComponentHealth:
        """Check vehicle tracker status"""
        try:
            if not hasattr(self.app_state, 'tracker_manager'):
                return ComponentHealth(
                    component=ComponentType.TRACKER.value,
                    status=HealthStatus.DEGRADED,
                    message="Tracker not initialized"
                )

            total_trackers = len(self.app_state.tracker_manager.trackers)

            return ComponentHealth(
                component=ComponentType.TRACKER.value,
                status=HealthStatus.HEALTHY,
                message="Tracker operational",
                metadata={"active_trackers": total_trackers}
            )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.TRACKER.value,
                status=HealthStatus.DEGRADED,
                message=f"Tracker error: {str(e)}"
            )

    async def check_feed_source(self) -> ComponentHealth:
        """Check Taiwan Highway Bureau feed source"""
        try:
            feeds_total = len(getattr(self.app_state, 'feeds_data', []))
            feeds_online = sum(1 for status in getattr(self.app_state, 'feed_status', {}).values() if status)

            if feeds_total == 0:
                # During startup, feeds are still being loaded - this is normal
                return ComponentHealth(
                    component=ComponentType.FEED_SOURCE.value,
                    status=HealthStatus.HEALTHY,
                    message="Feeds loading..."
                )

            # Calculate online percentage
            online_pct = (feeds_online / feeds_total * 100) if feeds_total > 0 else 0

            if online_pct < 50:
                status = HealthStatus.UNHEALTHY
                message = f"Many feeds offline ({online_pct:.0f}% online)"
            elif online_pct < 80:
                status = HealthStatus.DEGRADED
                message = f"Some feeds offline ({online_pct:.0f}% online)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Feeds operational ({online_pct:.0f}% online)"

            return ComponentHealth(
                component=ComponentType.FEED_SOURCE.value,
                status=status,
                message=message,
                metadata={
                    "total": feeds_total,
                    "online": feeds_online,
                    "online_pct": round(online_pct, 1)
                }
            )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.FEED_SOURCE.value,
                status=HealthStatus.UNHEALTHY,
                message=f"Feed source error: {str(e)}"
            )

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics (CPU, memory, disk)"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu_percent": round(cpu_percent, 1),
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "percent": round(memory.percent, 1)
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "percent": round(disk.percent, 1)
                },
                "python_process": {
                    "memory_mb": round(psutil.Process().memory_info().rss / (1024**2), 2)
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}


class StructuredLogger:
    """Structured logging with context and JSON output"""

    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Create structured formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )

        # Console handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log(self, level: str, message: str, **context):
        """Log with structured context"""
        if context:
            message = f"{message} | context={json.dumps(context)}"
        getattr(self.logger, level)(message)

    def debug(self, message: str, **context):
        self._log('debug', message, **context)

    def info(self, message: str, **context):
        self._log('info', message, **context)

    def warning(self, message: str, **context):
        self._log('warning', message, **context)

    def error(self, message: str, **context):
        self._log('error', message, **context)

    def critical(self, message: str, **context):
        self._log('critical', message, **context)


class CircuitBreaker:
    """Circuit breaker pattern for graceful degradation"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

        self.logger = logging.getLogger(__name__)

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception(f"Circuit breaker open for {func.__name__}")
            else:
                self.state = "half_open"
                self.logger.info(f"Circuit breaker half-open for {func.__name__}")

        try:
            result = func(*args, **kwargs)

            # Success - reset if we were in half_open
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                self.logger.info(f"Circuit breaker closed for {func.__name__}")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.error(
                    f"Circuit breaker opened for {func.__name__}",
                    extra={"failure_count": self.failure_count}
                )

            raise e

    async def call_async(self, func, *args, **kwargs):
        """Execute async function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception(f"Circuit breaker open for {func.__name__}")
            else:
                self.state = "half_open"

        try:
            result = await func(*args, **kwargs)

            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.error(f"Circuit breaker opened for {func.__name__}")

            raise e

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time,
            "threshold": self.failure_threshold
        }


class AlertManager:
    """Alert management for critical system events"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alert_history: List[Dict] = []
        self.alert_cooldowns: Dict[str, float] = {}
        self.cooldown_period = 300  # 5 minutes

    async def send_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict] = None
    ):
        """Send an alert through configured channels"""

        # Check cooldown
        cooldown_key = f"{alert_type}:{severity}"
        if cooldown_key in self.alert_cooldowns:
            time_since_last = time.time() - self.alert_cooldowns[cooldown_key]
            if time_since_last < self.cooldown_period:
                self.logger.debug(f"Alert {cooldown_key} in cooldown")
                return

        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {}
        }

        self.alert_history.append(alert)
        self.alert_cooldowns[cooldown_key] = time.time()

        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history.pop(0)

        # Log alert
        log_method = getattr(self.logger, severity.lower(), self.logger.warning)
        log_method(f"ALERT: {alert_type} - {message}", extra=metadata or {})

        # Send through channels
        if self.config.get('alerts', {}).get('enabled'):
            await self._send_to_channels(alert)

    async def _send_to_channels(self, alert: Dict):
        """Send alert to configured channels (email, webhook, etc.)"""
        # Email alerts
        if self.config.get('alerts', {}).get('email', {}).get('enabled'):
            await self._send_email_alert(alert)

        # Webhook alerts
        if self.config.get('alerts', {}).get('webhook', {}).get('enabled'):
            await self._send_webhook_alert(alert)

    async def _send_email_alert(self, alert: Dict):
        """Send email alert via SMTP"""
        email_config = self.config.get('alerts', {}).get('email', {})

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            smtp_host = email_config.get('smtp_host', 'smtp.gmail.com')
            smtp_port = email_config.get('smtp_port', 587)
            from_address = email_config.get('from_address', '')
            to_addresses = email_config.get('to_addresses', [])
            username = email_config.get('username', from_address)
            password = email_config.get('password', '')

            if not all([from_address, to_addresses, password]):
                self.logger.warning("Email alert: Missing required configuration")
                return

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[CCTV Alert] {alert['severity'].upper()}: {alert['type']}"
            msg['From'] = from_address
            msg['To'] = ', '.join(to_addresses)

            # Plain text body
            text_body = f"""
CCTV Viewer Alert
=================

Type: {alert['type']}
Severity: {alert['severity']}
Time: {alert['timestamp']}

Message:
{alert['message']}

Metadata:
{json.dumps(alert.get('metadata', {}), indent=2)}
"""

            # HTML body
            html_body = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .alert {{ padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .critical {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
        .error {{ background-color: #fff3cd; border: 1px solid #ffc107; }}
        .warning {{ background-color: #fff3cd; border: 1px solid #ffeeba; }}
        .info {{ background-color: #d1ecf1; border: 1px solid #bee5eb; }}
        .metadata {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h2>CCTV Viewer Alert</h2>
    <div class="alert {alert['severity'].lower()}">
        <strong>Type:</strong> {alert['type']}<br>
        <strong>Severity:</strong> {alert['severity']}<br>
        <strong>Time:</strong> {alert['timestamp']}<br>
        <p><strong>Message:</strong> {alert['message']}</p>
    </div>
    <h3>Metadata</h3>
    <div class="metadata">
        <pre>{json.dumps(alert.get('metadata', {}), indent=2)}</pre>
    </div>
</body>
</html>
"""

            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            # Send email (run in thread to not block)
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_smtp_email,
                                       smtp_host, smtp_port, username, password,
                                       from_address, to_addresses, msg)

            self.logger.info(f"Email alert sent to {', '.join(to_addresses)}")

        except ImportError:
            self.logger.error("Email alert: smtplib not available")
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")

    def _send_smtp_email(self, host, port, username, password, from_addr, to_addrs, msg):
        """Send email via SMTP (synchronous, run in executor)"""
        import smtplib

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(from_addr, to_addrs, msg.as_string())

    async def _send_webhook_alert(self, alert: Dict):
        """Send webhook alert via HTTP POST"""
        webhook_config = self.config.get('alerts', {}).get('webhook', {})
        url = webhook_config.get('url', '')

        if not url:
            self.logger.warning("Webhook alert: No URL configured")
            return

        try:
            import httpx

            # Format payload (compatible with Slack, Discord, etc.)
            payload = {
                "text": f"*[{alert['severity'].upper()}]* {alert['type']}: {alert['message']}",
                "attachments": [
                    {
                        "color": self._severity_color(alert['severity']),
                        "fields": [
                            {"title": "Type", "value": alert['type'], "short": True},
                            {"title": "Severity", "value": alert['severity'], "short": True},
                            {"title": "Time", "value": alert['timestamp'], "short": False},
                            {"title": "Message", "value": alert['message'], "short": False},
                        ]
                    }
                ],
                # Also include raw data for custom webhooks
                "alert": alert,
            }

            # Add custom headers if configured
            headers = webhook_config.get('headers', {})
            headers['Content-Type'] = 'application/json'

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

            self.logger.info(f"Webhook alert sent to {url}")

        except ImportError:
            self.logger.error("Webhook alert: httpx not available")
        except Exception as e:
            self.logger.error(f"Webhook alert failed: {e}")

    def _severity_color(self, severity: str) -> str:
        """Get color code for severity (Slack/Discord format)"""
        colors = {
            "critical": "#dc3545",  # Red
            "error": "#fd7e14",     # Orange
            "warning": "#ffc107",   # Yellow
            "info": "#17a2b8",      # Blue
        }
        return colors.get(severity.lower(), "#6c757d")

    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts"""
        return self.alert_history[-limit:]

    def clear_cooldowns(self):
        """Clear all alert cooldowns (useful for testing)"""
        self.alert_cooldowns.clear()

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        from collections import Counter

        if not self.alert_history:
            return {"total": 0, "by_type": {}, "by_severity": {}}

        by_type = Counter(a["type"] for a in self.alert_history)
        by_severity = Counter(a["severity"] for a in self.alert_history)

        return {
            "total": len(self.alert_history),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "active_cooldowns": len(self.alert_cooldowns),
        }
