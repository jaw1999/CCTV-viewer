"""
Security middleware for API authentication and rate limiting
"""
import os
import time
import hashlib
import secrets
from collections import defaultdict
from typing import Optional, Dict, List
from datetime import datetime, timezone

from fastapi import Request, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader, APIKeyQuery
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


# API Key configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)


def get_api_keys_from_env() -> List[str]:
    """Load API keys from environment variable."""
    keys_str = os.environ.get("CCTV_API_KEYS", "")
    if not keys_str:
        return []
    return [k.strip() for k in keys_str.split(",") if k.strip()]


def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """Hash an API key for secure storage comparison"""
    return hashlib.sha256(key.encode()).hexdigest()


async def get_api_key(
    api_key_header: Optional[str] = Security(API_KEY_HEADER),
    api_key_query: Optional[str] = Security(API_KEY_QUERY),
) -> Optional[str]:
    """Extract API key from header or query parameter."""
    return api_key_header or api_key_query


async def verify_api_key(api_key: Optional[str] = Depends(get_api_key)) -> str:
    """Verify API key against configured keys."""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    valid_keys = get_api_keys_from_env()
    if not valid_keys:
        return api_key  # No keys configured = development mode

    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication middleware."""

    def __init__(self, app, exclude_paths: Optional[List[str]] = None,
                 exclude_prefixes: Optional[List[str]] = None, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.exclude_paths = set(exclude_paths or ["/", "/health", "/docs", "/redoc", "/openapi.json"])
        self.exclude_prefixes = exclude_prefixes or ["/metrics"]
        self.valid_keys = get_api_keys_from_env()

    def _is_excluded(self, path: str) -> bool:
        if path in self.exclude_paths:
            return True
        return any(path.startswith(prefix) for prefix in self.exclude_prefixes)

    async def dispatch(self, request: Request, call_next):
        # Skip if disabled or no keys configured
        if not self.enabled or not self.valid_keys:
            try:
                return await call_next(request)
            except RuntimeError as e:
                if "No response returned" in str(e):
                    return JSONResponse(status_code=500, content={"error": "internal_error"})
                raise

        if self._is_excluded(request.url.path):
            try:
                return await call_next(request)
            except RuntimeError as e:
                if "No response returned" in str(e):
                    return JSONResponse(status_code=500, content={"error": "internal_error"})
                raise

        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")

        if not api_key or api_key not in self.valid_keys:
            return JSONResponse(
                status_code=401 if not api_key else 403,
                content={"error": "unauthorized", "message": "Valid API key required"},
            )

        try:
            return await call_next(request)
        except RuntimeError as e:
            if "No response returned" in str(e):
                return JSONResponse(status_code=500, content={"error": "internal_error"})
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple token bucket rate limiter."""

    def __init__(self, app, requests_per_minute: int = 200, burst_size: int = 50,
                 exclude_paths: Optional[List[str]] = None, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.rate = requests_per_minute / 60.0
        self.burst_size = burst_size
        self.exclude_paths = set(exclude_paths or ["/health"])
        self.buckets: Dict[str, Dict] = defaultdict(
            lambda: {"tokens": burst_size, "last_update": time.time()}
        )

    def _is_excluded(self, path: str) -> bool:
        # Exclude health checks and snapshot endpoints (for fast live viewing)
        if path in self.exclude_paths:
            return True
        if "/snapshot" in path:
            return True
        return False

    def _get_client_id(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        if not self.enabled or self._is_excluded(request.url.path):
            try:
                return await call_next(request)
            except RuntimeError as e:
                if "No response returned" in str(e):
                    return JSONResponse(status_code=500, content={"error": "internal_error"})
                raise

        client_id = self._get_client_id(request)
        bucket = self.buckets[client_id]
        now = time.time()

        # Refill tokens
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(self.burst_size, bucket["tokens"] + elapsed * self.rate)
        bucket["last_update"] = now

        if bucket["tokens"] < 1:
            retry_after = int((1 - bucket["tokens"]) / self.rate) + 1
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limit", "message": f"Too many requests. Retry in {retry_after}s"},
                headers={"Retry-After": str(retry_after)},
            )

        bucket["tokens"] -= 1
        try:
            return await call_next(request)
        except RuntimeError as e:
            if "No response returned" in str(e):
                return JSONResponse(status_code=500, content={"error": "internal_error"})
            raise
