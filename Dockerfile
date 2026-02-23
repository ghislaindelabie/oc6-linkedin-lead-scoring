# Dockerfile for HF Spaces deployment
# Follows HF Spaces Docker permissions pattern:
# https://huggingface.co/docs/hub/en/spaces-sdks-docker

FROM python:3.11-slim

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces expects UID 1000)
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python dependencies as user
COPY --chown=user requirements-prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy project files
COPY --chown=user src/ ./src/
COPY --chown=user model/ ./model/
COPY --chown=user alembic/ ./alembic/
COPY --chown=user alembic.ini .

# Expose HF Spaces port
EXPOSE 7860

# Environment variables — PYTHONPATH makes the package importable
# without pip install (all runtime deps come from requirements-prod.txt)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    PYTHONPATH=/home/user/app/src

# Health check using stdlib urllib (no requests library needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run DB migrations then start the API server
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn linkedin_lead_scoring.api.main:app --host 0.0.0.0 --port 7860"]
