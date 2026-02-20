"""
Middleware stack for the LinkedIn Lead Scoring API.

- RequestIDMiddleware: adds X-Request-ID to every response
- RateLimitHeadersMiddleware: adds X-RateLimit-* info headers
- RequestLoggingMiddleware: appends one JSON line per request to logs/

Log paths are module-level constants so tests can redirect via monkeypatch.
"""
import json
import os
import time
import uuid
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Module-level constant — override in tests via monkeypatch.setattr
_REQUESTS_LOG = "logs/api_requests.jsonl"

# Rate limit info (informational only — not enforced server-side)
_RATE_LIMIT = 100  # requests per minute (informational)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID to every response for tracing.

    If the client sends an X-Request-ID header, the server echoes it back.
    Otherwise a new UUID4 is generated.
    """

    _MAX_ID_LENGTH = 128

    async def dispatch(self, request: Request, call_next) -> Response:
        raw_id = request.headers.get("x-request-id") or ""
        # Sanitize: ASCII-only, truncate to 128 chars
        sanitized = raw_id[:self._MAX_ID_LENGTH] if raw_id.isascii() and raw_id else ""
        request_id = sanitized or str(uuid.uuid4())
        # Store on request state so other middleware / handlers can access it
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """Add informational rate-limit headers to every response.

    These headers advertise the rate-limit policy to clients. Actual
    enforcement is delegated to an API gateway / reverse proxy in production.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(_RATE_LIMIT)
        # X-RateLimit-Remaining omitted — actual tracking delegated to API gateway
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request as a JSON line: timestamp, method, path, status, time."""

    async def dispatch(self, request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "response_time_ms": elapsed_ms,
        }

        try:
            log_path = _REQUESTS_LOG
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # log failures must never crash the request

        return response
