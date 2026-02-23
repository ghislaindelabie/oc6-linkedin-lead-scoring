"""Async CRUD operations for prediction and API metric logging."""
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from linkedin_lead_scoring.db.models import ApiMetric, PredictionLog


async def log_prediction(
    session: AsyncSession,
    input_features: dict[str, Any],
    score: float,
    inference_ms: float,
    model_version: str,
) -> PredictionLog:
    """Insert a prediction record and return it.

    Args:
        session: Active async SQLAlchemy session.
        input_features: Dict of feature name → value sent to the model.
        score: Predicted engagement probability (0–1).
        inference_ms: Time taken for prediction in milliseconds.
        model_version: Identifier of the model used (e.g. "v1.0").

    Returns:
        The persisted PredictionLog instance.
    """
    row = PredictionLog(
        input_features=input_features,
        predicted_score=score,
        inference_time_ms=inference_ms,
        model_version=model_version,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def log_api_metric(
    session: AsyncSession,
    endpoint: str,
    status_code: int,
    response_ms: float,
) -> ApiMetric:
    """Insert an API request metric and return it.

    Args:
        session: Active async SQLAlchemy session.
        endpoint: Request path (e.g. "/predict").
        status_code: HTTP response status code.
        response_ms: Total response time in milliseconds.

    Returns:
        The persisted ApiMetric instance.
    """
    row = ApiMetric(
        endpoint=endpoint,
        status_code=status_code,
        response_time_ms=response_ms,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def get_recent_predictions(
    session: AsyncSession,
    limit: int = 100,
) -> list[PredictionLog]:
    """Return the most recent prediction logs, newest first.

    Args:
        session: Active async SQLAlchemy session.
        limit: Maximum number of rows to return.

    Returns:
        List of PredictionLog rows ordered by created_at descending.
    """
    result = await session.execute(
        select(PredictionLog)
        .order_by(desc(PredictionLog.created_at))
        .limit(limit)
    )
    return list(result.scalars().all())
