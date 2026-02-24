"""
FastAPI application for LinkedIn lead scoring.
v0.3.0 — adds /predict endpoint with model-at-startup loading pattern.
"""
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .middleware import (
    RateLimitHeadersMiddleware,
    RequestIDMiddleware,
    RequestLoggingMiddleware,
)
from .predict import is_model_loaded, lifespan, router as predict_router
from .schemas import HealthResponse

app = FastAPI(
    title="LinkedIn Lead Scoring API",
    description="MLOps project: Predict lead engagement probability",
    version="0.3.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Streamlit monitoring dashboard (Session C) to call the API
# ---------------------------------------------------------------------------

# Middleware stack (Starlette processes outermost-added first on response)
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RateLimitHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(predict_router)


# ---------------------------------------------------------------------------
# Custom exception handlers — structured ErrorResponse, no stack traces
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return Pydantic validation errors in ErrorResponse format."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Return HTTP exceptions in ErrorResponse format."""
    error_map = {
        400: "bad_request",
        404: "not_found",
        422: "validation_error",
        429: "rate_limited",
        500: "internal_error",
        503: "service_unavailable",
    }
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_map.get(exc.status_code, "http_error"),
            "message": str(exc.detail),
            "detail": None,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all: return generic 500 without leaking internals."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred. Please try again later.",
            "detail": None,
        },
    )


# ---------------------------------------------------------------------------
# Built-in endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> str:
    """Landing page"""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>LinkedIn Lead Scoring API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                }
                h1 { color: #0077B5; }
                a { color: #0077B5; text-decoration: none; }
                a:hover { text-decoration: underline; }
                ul { line-height: 2; }
            </style>
        </head>
        <body>
            <h1>🎯 LinkedIn Lead Scoring API</h1>
            <p>Welcome to the LinkedIn lead scoring API (OC6 MLOps project).</p>
            <p>This API predicts which LinkedIn contacts are most likely to engage
               with outreach campaigns.</p>
            <h2>Quick Links</h2>
            <ul>
                <li><a href="/docs">📚 API Documentation (Swagger UI)</a></li>
                <li><a href="/redoc">📖 API Documentation (ReDoc)</a></li>
                <li><a href="/health">❤️ Health Check</a></li>
            </ul>
            <h2>Endpoints</h2>
            <ul>
                <li><strong>POST /predict</strong> — Score a single lead</li>
                <li><strong>POST /predict/batch</strong> — Score up to 10 000 leads (coming soon)</li>
            </ul>
        </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check() -> HealthResponse:
    """Return service status including whether the ML model is loaded and ready."""
    return HealthResponse(
        status="healthy",
        service="linkedin-lead-scoring-api",
        version="0.3.0",
        model_loaded=is_model_loaded(),
    )
