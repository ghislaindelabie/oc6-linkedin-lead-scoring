"""Async SQLAlchemy engine for Supabase PostgreSQL (or SQLite for local dev).

In production: set DATABASE_URL to your Supabase connection string:
    postgresql://user:password@host:5432/dbname

For local development without a database, the module falls back to SQLite.
"""
import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from linkedin_lead_scoring.db.models import Base

# ---------------------------------------------------------------------------
# Resolve DATABASE_URL
# ---------------------------------------------------------------------------

_raw_url = os.environ.get("DATABASE_URL", "")

if _raw_url.startswith("postgresql://"):
    # Standard postgresql:// → rewrite to asyncpg driver
    DATABASE_URL = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _raw_url.startswith("postgresql+asyncpg://"):
    DATABASE_URL = _raw_url
else:
    # Fallback: local SQLite (no external service needed for dev/test)
    DATABASE_URL = "sqlite+aiosqlite:///./local_dev.db"

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args=_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session; close on exit.

    Usage in FastAPI endpoints::

        @app.post("/predict")
        async def predict(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        yield session
