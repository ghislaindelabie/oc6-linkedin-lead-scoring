"""
FastAPI application for LinkedIn lead scoring.
Initial version with health check and placeholder endpoints.
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="LinkedIn Lead Scoring API",
    description="MLOps project: Predict lead engagement probability",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse)
async def root():
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
            <p>This API predicts which LinkedIn contacts are most likely to engage with outreach campaigns.</p>
            <h2>Quick Links</h2>
            <ul>
                <li><a href="/docs">📚 API Documentation (Swagger UI)</a></li>
                <li><a href="/redoc">📖 API Documentation (ReDoc)</a></li>
                <li><a href="/health">❤️ Health Check</a></li>
            </ul>
            <h2>Features</h2>
            <ul>
                <li>Lead engagement prediction (coming soon)</li>
                <li>Batch scoring for campaigns (coming soon)</li>
                <li>Model versioning support (coming soon)</li>
                <li>LemList integration (coming soon)</li>
            </ul>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "linkedin-lead-scoring-api",
        "version": "0.1.0",
    }
