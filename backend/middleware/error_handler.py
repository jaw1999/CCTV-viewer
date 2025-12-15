"""
Global error handling middleware for consistent error responses
"""
import traceback
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging


logger = logging.getLogger(__name__)


class CCTVException(Exception):
    """Base exception for CCTV application"""

    def __init__(
        self,
        message: str,
        error_code: str = "internal_error",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class ValidationException(CCTVException):
    """Validation error exception"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="validation_error",
            status_code=422,
            details=details,
        )


class AuthenticationException(CCTVException):
    """Authentication error exception"""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            error_code="authentication_required",
            status_code=401,
        )


class RateLimitException(CCTVException):
    """Rate limit exceeded exception"""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            message=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            error_code="rate_limit_exceeded",
            status_code=429,
            details={"retry_after": retry_after},
        )


class NotFoundError(CCTVException):
    """Resource not found exception"""

    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            message=f"{resource} with ID '{resource_id}' not found",
            error_code="not_found",
            status_code=404,
            details={"resource": resource, "id": resource_id},
        )


class ServiceUnavailableError(CCTVException):
    """Service unavailable exception"""

    def __init__(self, service: str, message: Optional[str] = None):
        super().__init__(
            message=message or f"{service} is currently unavailable",
            error_code="service_unavailable",
            status_code=503,
            details={"service": service},
        )


def format_validation_errors(errors: list) -> list:
    """Format Pydantic validation errors for API response"""
    formatted = []
    for error in errors:
        formatted.append({
            "field": ".".join(str(loc) for loc in error.get("loc", [])),
            "message": error.get("msg", "Validation error"),
            "type": error.get("type", "unknown"),
        })
    return formatted


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.

    Features:
    - Consistent error response format
    - Detailed error logging
    - Exception to HTTP status code mapping
    - Request context in error logs
    """

    def __init__(self, app, include_traceback: bool = False):
        super().__init__(app)
        self.include_traceback = include_traceback

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response

        except CCTVException as e:
            # Handle custom CCTV exceptions
            logger.warning(
                f"CCTV Exception: {e.error_code}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "error_code": e.error_code,
                    "message": e.message,
                },
            )
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict(),
            )

        except HTTPException as e:
            # Handle FastAPI HTTPExceptions
            logger.warning(
                f"HTTP Exception: {e.status_code}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": e.status_code,
                    "detail": e.detail,
                },
            )
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "http_error",
                    "message": e.detail,
                    "status_code": e.status_code,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except RequestValidationError as e:
            # Handle Pydantic validation errors
            logger.warning(
                f"Validation Error: {request.url.path}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "errors": e.errors(),
                },
            )
            return JSONResponse(
                status_code=422,
                content={
                    "error": "validation_error",
                    "message": "Request validation failed",
                    "details": format_validation_errors(e.errors()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except Exception as e:
            # Handle unexpected exceptions
            tb = traceback.format_exc()
            logger.error(
                f"Unhandled Exception: {type(e).__name__}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": tb,
                },
            )

            content = {
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if self.include_traceback:
                content["debug"] = {
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": tb.split("\n"),
                }

            return JSONResponse(
                status_code=500,
                content=content,
            )


async def custom_http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Custom handler for HTTPException"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


async def custom_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Custom handler for RequestValidationError"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": format_validation_errors(exc.errors()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
