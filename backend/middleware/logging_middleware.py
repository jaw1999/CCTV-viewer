"""
Request/Response logging middleware
"""
import time
import uuid
import logging
from typing import Callable, Optional
from datetime import datetime, timezone

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message


logger = logging.getLogger("cctv.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.

    Features:
    - Request/response timing
    - Unique request IDs for tracing
    - Configurable log levels
    - Path exclusions for noisy endpoints
    - Request body logging (optional)
    """

    def __init__(
        self,
        app,
        exclude_paths: Optional[list] = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
        slow_request_threshold_ms: float = 1000.0,
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/health/live",
            "/health/ready",
            "/metrics",
        ]
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.slow_request_threshold_ms = slow_request_threshold_ms

    def _should_log(self, path: str) -> bool:
        """Check if request should be logged"""
        for excluded in self.exclude_paths:
            if path.startswith(excluded):
                return False
        return True

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, considering proxies"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip logging for excluded paths
        if not self._should_log(request.url.path):
            return await call_next(request)

        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Record start time
        start_time = time.time()

        # Log request
        client_ip = self._get_client_ip(request)
        api_key_id = getattr(request.state, "api_key_id", None)

        request_log = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params) if request.query_params else None,
            "client_ip": client_ip,
            "api_key_id": api_key_id,
            "user_agent": request.headers.get("User-Agent", "")[:100],
        }

        logger.info(f"→ {request.method} {request.url.path}", extra=request_log)

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            response_log = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            }

            # Determine log level based on status code and duration
            if response.status_code >= 500:
                log_level = logging.ERROR
            elif response.status_code >= 400:
                log_level = logging.WARNING
            elif duration_ms > self.slow_request_threshold_ms:
                log_level = logging.WARNING
                response_log["slow_request"] = True
            else:
                log_level = logging.INFO

            logger.log(
                log_level,
                f"← {response.status_code} {request.method} {request.url.path} ({duration_ms:.0f}ms)",
                extra=response_log,
            )

            # Add request ID to response headers for tracing
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"✗ {request.method} {request.url.path} ({duration_ms:.0f}ms) - {type(e).__name__}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                    "exception": str(e),
                },
            )
            raise
