"""
FastAPI application for LinkedIn lead scoring.
v0.3.0 — adds /predict endpoint with model-at-startup loading pattern.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened per environment in production via env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(predict_router)


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
