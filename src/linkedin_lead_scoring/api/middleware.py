"""
Request logging middleware.

Appends one JSON line per HTTP request to logs/api_requests.jsonl.
The log path is a module-level constant so tests can redirect it via
monkeypatch without env-var complications.
"""
import json
import os
import time
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Module-level constant — override in tests via monkeypatch.setattr
_REQUESTS_LOG = "logs/api_requests.jsonl"


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
