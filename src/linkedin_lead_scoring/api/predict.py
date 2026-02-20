"""
/predict endpoint — single lead scoring.

Model is loaded ONCE at application startup via the lifespan context manager.
In development mode (APP_ENV=development) with no model files, a deterministic
mock model is used so the API can be exercised without real artifacts.
"""
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException

from .schemas import BatchPredictionRequest, BatchPredictionResponse, LeadInput, LeadPrediction

# ---------------------------------------------------------------------------
# Model paths (relative to the working directory where uvicorn is launched)
# ---------------------------------------------------------------------------

_MODEL_PATH = "model/xgboost_model.joblib"
_PREPROCESSOR_PATH = "model/preprocessor.joblib"
_FEATURE_COLS_PATH = "model/feature_columns.json"
_MODEL_VERSION = "0.3.0"

# ---------------------------------------------------------------------------
# Module-level state — populated once at startup, never modified per-request
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "model": None,
    "preprocessor": None,
    "feature_cols": None,
    "model_version": "unknown",
    "model_loaded": False,
    "is_mock": False,
}


# ---------------------------------------------------------------------------
# Mock model for development / testing (no real artifacts required)
# ---------------------------------------------------------------------------


class _MockModel:
    """Deterministic stand-in for the real XGBoost model."""

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        # Fixed score of 0.65 — above threshold, medium-high confidence
        return np.array([[0.35, 0.65]] * n)


# ---------------------------------------------------------------------------
# Lifespan — imported and used by main.py
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once at startup; release on shutdown."""
    model_files_exist = all(
        os.path.exists(p)
        for p in (_MODEL_PATH, _PREPROCESSOR_PATH, _FEATURE_COLS_PATH)
    )

    if model_files_exist:
        try:
            import joblib

            _state["model"] = joblib.load(_MODEL_PATH)
            _state["preprocessor"] = joblib.load(_PREPROCESSOR_PATH)
            with open(_FEATURE_COLS_PATH) as f:
                _state["feature_cols"] = json.load(f)
            _state["model_version"] = _MODEL_VERSION
            _state["model_loaded"] = True
            _state["is_mock"] = False
        except Exception:
            # Startup failure — model_loaded stays False → /predict returns 503
            _state["model_loaded"] = False

    elif os.getenv("APP_ENV", "production") == "development":
        _state["model"] = _MockModel()
        _state["preprocessor"] = None
        _state["feature_cols"] = None
        _state["model_version"] = f"mock-{_MODEL_VERSION}"
        _state["model_loaded"] = True
        _state["is_mock"] = True

    yield

    # Cleanup on shutdown
    _state["model"] = None
    _state["preprocessor"] = None
    _state["feature_cols"] = None
    _state["model_loaded"] = False
    _state["is_mock"] = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_confidence(score: float) -> str:
    """Map a probability score to a human-readable confidence level."""
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _lead_to_dataframe(lead: LeadInput) -> pd.DataFrame:
    """Convert a LeadInput into a single-row DataFrame."""
    return pd.DataFrame([lead.model_dump()])


def is_model_loaded() -> bool:
    """Return True if the model is ready to serve predictions."""
    return _state["model_loaded"]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.post("/predict", response_model=LeadPrediction, summary="Score a single lead")
async def predict(lead: LeadInput) -> LeadPrediction:
    """
    Predict engagement probability for a single LinkedIn lead.

    Returns a score between 0 and 1, a binary label, a confidence level,
    the model version used, and the inference time in milliseconds.
    """
    if not _state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — service is not ready. Try again shortly.",
        )

    try:
        t0 = time.perf_counter()

        if _state["is_mock"]:
            proba = _state["model"].predict_proba(None)
            score = float(proba[0, 1])
        else:
            df = _lead_to_dataframe(lead)
            # Keep only the columns the model was trained on; fill missing with NaN
            feature_cols: list[str] = _state["feature_cols"]
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[feature_cols]
            X = _state["preprocessor"].transform(df)
            proba = _state["model"].predict_proba(X)
            score = float(proba[0, 1])

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return LeadPrediction(
            score=round(score, 4),
            label="engaged" if score >= 0.5 else "not_engaged",
            confidence=_get_confidence(score),
            model_version=_state["model_version"],
            inference_time_ms=round(elapsed_ms, 3),
        )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error.",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score a batch of leads (up to 10 000)",
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict engagement probability for a batch of LinkedIn leads.

    All leads are scored in a single vectorised pass through the model.
    Returns per-lead predictions plus summary statistics (total count,
    average score, number of high-engagement leads with score >= 0.5).
    """
    if not _state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — service is not ready. Try again shortly.",
        )

    try:
        leads = request.leads
        n = len(leads)
        t0 = time.perf_counter()

        if _state["is_mock"]:
            probas = _state["model"].predict_proba(leads)
            scores = probas[:, 1].tolist()
        else:
            df = pd.DataFrame([lead.model_dump() for lead in leads])
            feature_cols: list[str] = _state["feature_cols"]
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[feature_cols]
            X = _state["preprocessor"].transform(df)
            probas = _state["model"].predict_proba(X)
            scores = probas[:, 1].tolist()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_lead_ms = round(elapsed_ms / n, 3)

        predictions = [
            LeadPrediction(
                score=round(s, 4),
                label="engaged" if s >= 0.5 else "not_engaged",
                confidence=_get_confidence(s),
                model_version=_state["model_version"],
                inference_time_ms=per_lead_ms,
            )
            for s in scores
        ]

        avg_score = round(sum(scores) / n, 4)
        high_engagement_count = sum(1 for s in scores if s >= 0.5)

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=n,
            avg_score=avg_score,
            high_engagement_count=high_engagement_count,
        )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed due to an internal error.",
        )
