#!/usr/bin/env python3
"""
Observability module - health checks, metrics, logging, alerts
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

try:
    from .database import Feed as FeedModel, Detection as DetectionModel, VehicleTrack as VehicleTrackModel
except ImportError:
    from database import Feed as FeedModel, Detection as DetectionModel, VehicleTrack as VehicleTrackModel


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentType(str, Enum):
    DATABASE = "database"
    YOLO_MODEL = "yolo_model"
    HTTP_CLIENT = "http_client"
    WEBSOCKET = "websocket"
    CACHE = "cache"
    TRACKER = "tracker"
    FEED_SOURCE = "feed_source"


@dataclass
class ComponentHealth:
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
    """Prometheus metrics for the application."""

    def __init__(self):
        self.detections_total = Counter(
            'cctv_detections_total',
            'Total vehicle detections',
            ['feed_id', 'vehicle_class']
        )

        self.detection_confidence = Histogram(
            'cctv_detection_confidence',
            'Detection confidence scores',
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        )

        self.yolo_inference_duration = Histogram(
            'cctv_yolo_inference_seconds',
            'YOLO inference time',
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        )

        self.feed_fetch_duration = Histogram(
            'cctv_feed_fetch_seconds',
            'Feed fetch time',
            ['status']
        )

        self.cycle_duration = Summary(
            'cctv_cycle_duration_seconds',
            'Detection cycle time'
        )

        self.feeds_total = Gauge('cctv_feeds_total', 'Total feeds')
        self.feeds_online = Gauge('cctv_feeds_online', 'Online feeds')
        self.feeds_with_vehicles = Gauge('cctv_feeds_with_vehicles', 'Feeds with vehicles')
        self.cache_size_bytes = Gauge('cctv_cache_size_bytes', 'Cache size')
        self.active_websockets = Gauge('cctv_active_websockets', 'WebSocket connections')

        self.errors_total = Counter(
            'cctv_errors_total',
            'Total errors',
            ['component', 'error_type']
        )

        self.feed_failures = Counter(
            'cctv_feed_failures_total',
            'Feed failures',
            ['feed_id', 'reason']
        )

        self.db_query_duration = Histogram(
            'cctv_db_query_seconds',
            'DB query time',
            ['operation']
        )

        self.db_connections_active = Gauge('cctv_db_connections_active', 'Active DB connections')

        self.tracks_total = Counter(
            'cctv_tracks_total',
            'Vehicle tracks created',
            ['vehicle_class']
        )

        self.tracks_active = Gauge(
            'cctv_tracks_active',
            'Active tracks',
            ['feed_id']
        )

        self.system_info = Info('cctv_system', 'System info')


class HealthChecker:
    """Runs health checks on all system components."""

    def __init__(self, app_state):
        self.app_state = app_state
        self.logger = logging.getLogger(__name__)

    async def check_all(self) -> Dict[str, Any]:
        start_time = time.time()

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
                if check.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif check.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED

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
        start_time = time.time()

        try:
            if not hasattr(self.app_state, 'db_manager') or self.app_state.db_manager is None:
                return ComponentHealth(
                    component=ComponentType.DATABASE.value,
                    status=HealthStatus.UNHEALTHY,
                    message="Not initialized"
                )

            async with self.app_state.db_manager.session() as session:
                result = await session.execute(select(func.count()).select_from(FeedModel))
                feed_count = result.scalar()

            latency_ms = (time.time() - start_time) * 1000

            if latency_ms > 1000:
                status = HealthStatus.DEGRADED
                message = f"Slow ({latency_ms:.0f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = "OK"

            return ComponentHealth(
                component=ComponentType.DATABASE.value,
                status=status,
                message=message,
                latency_ms=round(latency_ms, 2),
                metadata={"feed_count": feed_count}
            )

        except Exception as e:
            self.logger.error(f"DB check failed: {e}")
            return ComponentHealth(
                component=ComponentType.DATABASE.value,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}"
            )

    async def check_yolo_model(self) -> ComponentHealth:
        try:
            if not hasattr(self.app_state, 'yolo_model') or self.app_state.yolo_model is None:
                return ComponentHealth(
                    component=ComponentType.YOLO_MODEL.value,
                    status=HealthStatus.UNHEALTHY,
                    message="Not loaded"
                )

            model_info = {
                "type": type(self.app_state.yolo_model).__name__,
                "device": str(getattr(self.app_state.yolo_model, 'device', 'unknown'))
            }

            return ComponentHealth(
                component=ComponentType.YOLO_MODEL.value,
                status=HealthStatus.HEALTHY,
                message="OK",
                metadata=model_info
            )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.YOLO_MODEL.value,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}"
            )

    async def check_http_client(self) -> ComponentHealth:
        start_time = time.time()

        try:
            feed_cache = getattr(self.app_state, 'feed_cache', None)
            cached_count = 0
            if feed_cache is not None:
                cache_stats = feed_cache.stats
                cached_count = cache_stats.get('items', 0)

            if cached_count > 0:
                latency_ms = (time.time() - start_time) * 1000
                return ComponentHealth(
                    component=ComponentType.HTTP_CLIENT.value,
                    status=HealthStatus.HEALTHY,
                    message="OK",
                    latency_ms=round(latency_ms, 2),
                    metadata={"cached_feeds": cached_count}
                )
            else:
                return ComponentHealth(
                    component=ComponentType.HTTP_CLIENT.value,
                    status=HealthStatus.HEALTHY,
                    message="Initializing"
                )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.HTTP_CLIENT.value,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}"
            )

    async def check_websocket(self) -> ComponentHealth:
        try:
            if not hasattr(self.app_state, 'ws_manager'):
                return ComponentHealth(
                    component=ComponentType.WEBSOCKET.value,
                    status=HealthStatus.DEGRADED,
                    message="Not initialized"
                )

            active = len(self.app_state.ws_manager.active_connections)

            return ComponentHealth(
                component=ComponentType.WEBSOCKET.value,
                status=HealthStatus.HEALTHY,
                message="OK",
                metadata={"active_connections": active}
            )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.WEBSOCKET.value,
                status=HealthStatus.DEGRADED,
                message=f"Error: {str(e)}"
            )

    async def check_cache(self) -> ComponentHealth:
        try:
            if not hasattr(self.app_state, 'feed_cache'):
                return ComponentHealth(
                    component=ComponentType.CACHE.value,
                    status=HealthStatus.DEGRADED,
                    message="Not initialized"
                )

            stats = self.app_state.feed_cache.stats
            size_mb = stats.get('size_bytes', 0) / (1024 * 1024)
            items = stats.get('items', 0)

            if size_mb > 1000:
                status = HealthStatus.DEGRADED
                message = f"Large ({size_mb:.0f}MB)"
            else:
                status = HealthStatus.HEALTHY
                message = "OK"

            return ComponentHealth(
                component=ComponentType.CACHE.value,
                status=status,
                message=message,
                metadata={"size_mb": round(size_mb, 2), "items": items}
            )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.CACHE.value,
                status=HealthStatus.DEGRADED,
                message=f"Error: {str(e)}"
            )

    async def check_tracker(self) -> ComponentHealth:
        try:
            if not hasattr(self.app_state, 'tracker_manager'):
                return ComponentHealth(
                    component=ComponentType.TRACKER.value,
                    status=HealthStatus.DEGRADED,
                    message="Not initialized"
                )

            total = len(self.app_state.tracker_manager.trackers)

            return ComponentHealth(
                component=ComponentType.TRACKER.value,
                status=HealthStatus.HEALTHY,
                message="OK",
                metadata={"active_trackers": total}
            )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.TRACKER.value,
                status=HealthStatus.DEGRADED,
                message=f"Error: {str(e)}"
            )

    async def check_feed_source(self) -> ComponentHealth:
        try:
            feeds_total = len(getattr(self.app_state, 'feeds_data', []))
            feeds_online = sum(1 for s in getattr(self.app_state, 'feed_status', {}).values() if s)

            if feeds_total == 0:
                return ComponentHealth(
                    component=ComponentType.FEED_SOURCE.value,
                    status=HealthStatus.HEALTHY,
                    message="Loading..."
                )

            pct = (feeds_online / feeds_total * 100) if feeds_total > 0 else 0

            if pct < 50:
                status = HealthStatus.UNHEALTHY
                message = f"{pct:.0f}% online"
            elif pct < 80:
                status = HealthStatus.DEGRADED
                message = f"{pct:.0f}% online"
            else:
                status = HealthStatus.HEALTHY
                message = f"{pct:.0f}% online"

            return ComponentHealth(
                component=ComponentType.FEED_SOURCE.value,
                status=status,
                message=message,
                metadata={"total": feeds_total, "online": feeds_online, "online_pct": round(pct, 1)}
            )

        except Exception as e:
            return ComponentHealth(
                component=ComponentType.FEED_SOURCE.value,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}"
            )

    def get_system_metrics(self) -> Dict[str, Any]:
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu_percent": round(cpu, 1),
                "memory": {
                    "total_gb": round(mem.total / (1024**3), 2),
                    "used_gb": round(mem.used / (1024**3), 2),
                    "percent": round(mem.percent, 1)
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
    """JSON-formatted logger with context support."""

    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log(self, level: str, message: str, **context):
        if context:
            message = f"{message} | {json.dumps(context)}"
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
    """
    Prevents cascading failures by tracking errors and temporarily
    blocking calls when a threshold is exceeded.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.logger = logging.getLogger(__name__)

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception(f"Circuit open: {func.__name__}")
            self.state = "half_open"
            self.logger.info(f"Circuit half-open: {func.__name__}")

        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                self.logger.info(f"Circuit closed: {func.__name__}")
            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.error(f"Circuit opened: {func.__name__}")
            raise e

    async def call_async(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception(f"Circuit open: {func.__name__}")
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
                self.logger.error(f"Circuit opened: {func.__name__}")
            raise e

    def get_state(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time,
            "threshold": self.failure_threshold
        }


class AlertManager:
    """Sends alerts via email/webhook when critical events occur."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alert_history: List[Dict] = []
        self.alert_cooldowns: Dict[str, float] = {}
        self.cooldown_period = 300  # 5 min

    async def send_alert(self, alert_type: str, severity: str, message: str,
                         metadata: Optional[Dict] = None):
        cooldown_key = f"{alert_type}:{severity}"
        if cooldown_key in self.alert_cooldowns:
            if time.time() - self.alert_cooldowns[cooldown_key] < self.cooldown_period:
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

        if len(self.alert_history) > 100:
            self.alert_history.pop(0)

        log_method = getattr(self.logger, severity.lower(), self.logger.warning)
        log_method(f"ALERT: {alert_type} - {message}", extra=metadata or {})

        if self.config.get('alerts', {}).get('enabled'):
            await self._send_to_channels(alert)

    async def _send_to_channels(self, alert: Dict):
        if self.config.get('alerts', {}).get('email', {}).get('enabled'):
            await self._send_email_alert(alert)
        if self.config.get('alerts', {}).get('webhook', {}).get('enabled'):
            await self._send_webhook_alert(alert)

    async def _send_email_alert(self, alert: Dict):
        email_cfg = self.config.get('alerts', {}).get('email', {})

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            smtp_host = email_cfg.get('smtp_host', 'smtp.gmail.com')
            smtp_port = email_cfg.get('smtp_port', 587)
            from_addr = email_cfg.get('from_address', '')
            to_addrs = email_cfg.get('to_addresses', [])
            username = email_cfg.get('username', from_addr)
            password = email_cfg.get('password', '')

            if not all([from_addr, to_addrs, password]):
                self.logger.warning("Email config incomplete")
                return

            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[CCTV Alert] {alert['severity'].upper()}: {alert['type']}"
            msg['From'] = from_addr
            msg['To'] = ', '.join(to_addrs)

            text_body = f"""
CCTV Alert
==========
Type: {alert['type']}
Severity: {alert['severity']}
Time: {alert['timestamp']}

{alert['message']}

{json.dumps(alert.get('metadata', {}), indent=2)}
"""

            html_body = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <h2>CCTV Alert</h2>
    <p><strong>Type:</strong> {alert['type']}</p>
    <p><strong>Severity:</strong> {alert['severity']}</p>
    <p><strong>Time:</strong> {alert['timestamp']}</p>
    <p>{alert['message']}</p>
    <pre style="background: #f5f5f5; padding: 10px;">{json.dumps(alert.get('metadata', {}), indent=2)}</pre>
</body>
</html>
"""

            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._send_smtp_email,
                smtp_host, smtp_port, username, password, from_addr, to_addrs, msg
            )

            self.logger.info(f"Email sent to {', '.join(to_addrs)}")

        except Exception as e:
            self.logger.error(f"Email failed: {e}")

    def _send_smtp_email(self, host, port, username, password, from_addr, to_addrs, msg):
        import smtplib
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(from_addr, to_addrs, msg.as_string())

    async def _send_webhook_alert(self, alert: Dict):
        webhook_cfg = self.config.get('alerts', {}).get('webhook', {})
        url = webhook_cfg.get('url', '')

        if not url:
            self.logger.warning("Webhook URL not configured")
            return

        try:
            import httpx

            payload = {
                "text": f"*[{alert['severity'].upper()}]* {alert['type']}: {alert['message']}",
                "attachments": [{
                    "color": self._severity_color(alert['severity']),
                    "fields": [
                        {"title": "Type", "value": alert['type'], "short": True},
                        {"title": "Severity", "value": alert['severity'], "short": True},
                        {"title": "Time", "value": alert['timestamp'], "short": False},
                        {"title": "Message", "value": alert['message'], "short": False},
                    ]
                }],
                "alert": alert,
            }

            headers = webhook_cfg.get('headers', {})
            headers['Content-Type'] = 'application/json'

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

            self.logger.info(f"Webhook sent to {url}")

        except Exception as e:
            self.logger.error(f"Webhook failed: {e}")

    def _severity_color(self, severity: str) -> str:
        return {
            "critical": "#dc3545",
            "error": "#fd7e14",
            "warning": "#ffc107",
            "info": "#17a2b8",
        }.get(severity.lower(), "#6c757d")

    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        return self.alert_history[-limit:]

    def clear_cooldowns(self):
        self.alert_cooldowns.clear()

    def get_alert_stats(self) -> Dict[str, Any]:
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
