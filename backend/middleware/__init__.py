"""
Middleware package for CCTV Viewer
Contains security, logging, and rate limiting middleware
"""

from .security import (
    APIKeyMiddleware,
    get_api_key,
    verify_api_key,
    RateLimitMiddleware,
)
from .error_handler import (
    ErrorHandlerMiddleware,
    CCTVException,
    ValidationException,
    AuthenticationException,
    RateLimitException,
    NotFoundError
)
from .logging_middleware import RequestLoggingMiddleware

__all__ = [
    # Security
    "APIKeyMiddleware",
    "get_api_key",
    "verify_api_key",
    "RateLimitMiddleware",
    # Error handling
    "ErrorHandlerMiddleware",
    "CCTVException",
    "ValidationException",
    "AuthenticationException",
    "RateLimitException",
    "NotFoundError",
    # Logging
    "RequestLoggingMiddleware",
]
