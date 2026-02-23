# syntax=docker/dockerfile:1
# Dockerfile for HF Spaces deployment
# Optimized for FastAPI deployment with ML model

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and Python packages in a single layer
COPY requirements-prod.txt .
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-prod.txt

# Copy project files — use --chmod so no separate RUN step is needed
COPY --chmod=777 src/ ./src/
COPY --chmod=777 model/ ./model/
COPY --chmod=777 alembic/ ./alembic/
COPY --chmod=777 alembic.ini .

# Expose HF Spaces port
EXPOSE 7860

# Environment variables — PYTHONPATH makes the package importable
# without pip install (all runtime deps come from requirements-prod.txt)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    PYTHONPATH=/app/src

# Health check using stdlib urllib (no requests library needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run DB migrations then start the API server
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn linkedin_lead_scoring.api.main:app --host 0.0.0.0 --port 7860"]
