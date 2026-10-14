from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class WordLensError(Exception):
    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = 400,
        details: Any = None,
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


def error_payload(
    request_id: str,
    error_code: str,
    message: str,
    details: Any = None,
) -> dict:
    return {
        "error_code": error_code,
        "message": message,
        "details": details,
        "request_id": request_id,
        "timestamp": utc_timestamp(),
    }


async def wordlens_error_handler(
    request: Request,
    exc: WordLensError,
):
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(
            request_id=request_id,
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
        ),
        headers={
            "X-Request-ID": request_id,
        },
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    return JSONResponse(
        status_code=422,
        content=error_payload(
            request_id=request_id,
            error_code="VALIDATION_ERROR",
            message="Request validation failed.",
            details=exc.errors(),
        ),
        headers={
            "X-Request-ID": request_id,
        },
    )


async def unexpected_error_handler(
    request: Request,
    exc: Exception,
):
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    return JSONResponse(
        status_code=500,
        content=error_payload(
            request_id=request_id,
            error_code="INTERNAL_ERROR",
            message="An unexpected server error occurred.",
            details=None,
        ),
        headers={
            "X-Request-ID": request_id,
        },
    )
