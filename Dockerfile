# Dockerfile for HF Spaces deployment
# Optimized for FastAPI deployment with ML model

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from pinned requirements
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy project files and install the package
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/
COPY model/ ./model/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Install the package (production mode — not editable, no deps since
# requirements-prod.txt already provides all runtime dependencies)
RUN pip install --no-cache-dir --no-deps .

# Create non-root user
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# Expose HF Spaces port
EXPOSE 7860

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production

# Health check using stdlib urllib (no requests library needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run DB migrations then start the API server
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn linkedin_lead_scoring.api.main:app --host 0.0.0.0 --port 7860"]
