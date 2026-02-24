"""Shared pytest fixtures for API testing.

Imports are conditional so the E2E test suite (which only installs
pytest + httpx) can run without numpy, fastapi, or the application code.
"""
import json
from pathlib import Path

import pytest

try:
    import numpy as np
    from fastapi.testclient import TestClient
    from linkedin_lead_scoring.api.main import app
    _HAS_APP_DEPS = True
except ImportError:
    _HAS_APP_DEPS = False


@pytest.fixture
def client(monkeypatch):
    """Test client with lifespan — no model files, no dev mode → model_loaded=False."""
    if not _HAS_APP_DEPS:
        pytest.skip("Full app dependencies not installed")
    import linkedin_lead_scoring.api.predict as predict_module
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setattr(predict_module, "_MODEL_PATH", "/nonexistent/model.joblib")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def dev_client(monkeypatch):
    """Test client with mock model loaded (APP_ENV=development, no real model files needed)."""
    if not _HAS_APP_DEPS:
        pytest.skip("Full app dependencies not installed")
    monkeypatch.setenv("APP_ENV", "development")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_model(monkeypatch):
    """Patch predict module _state with a deterministic fake model.

    Use this when you need fine-grained control over predicted scores.
    For most endpoint tests, prefer dev_client which activates the built-in mock mode.
    """
    if not _HAS_APP_DEPS:
        pytest.skip("Full app dependencies not installed")
    import linkedin_lead_scoring.api.predict as predict_module

    class FakeModel:
        """Returns predict_proba = [[0.3, 0.7]] for every row."""

        def predict_proba(self, X):
            n = len(X) if hasattr(X, "__len__") and X is not None else 1
            return np.array([[0.3, 0.7]] * n)

    monkeypatch.setitem(predict_module._state, "model", FakeModel())
    monkeypatch.setitem(predict_module._state, "model_loaded", True)
    monkeypatch.setitem(predict_module._state, "is_mock", True)
    monkeypatch.setitem(predict_module._state, "model_version", "test-0.0.0")
    return FakeModel()


@pytest.fixture
def valid_lead():
    """Sample valid lead input dict."""
    return {
        "llm_quality": 75,
        "llm_engagement": 0.8,
        "llm_decision_maker": 0.6,
        "llm_company_fit": 1,
        "companyfoundedon": 2015.0,
        "llm_seniority": "Senior",
        "llm_industry": "Technology - SaaS",
        "llm_geography": "international_hub",
        "llm_business_type": "leaders",
        "industry": "Information Technology & Services",
        "companyindustry": "Software Development",
        "companysize": "51-200",
        "companytype": "Privately Held",
        "languages": "English, French",
        "location": "Paris, Île-de-France, France",
        "companylocation": "Paris, France",
        "summary": "Experienced SaaS executive with 10+ years in B2B sales.",
        "skills": "Leadership, SaaS, B2B Sales, CRM",
        "jobtitle": "VP of Sales",
    }


@pytest.fixture
def client_with_preprocessor(monkeypatch):
    """Test client with is_mock=False — exercises real preprocessing pipeline.

    Uses real feature_columns.json and numeric_medians.json from model/,
    but replaces the model with a FakeModel that accepts any DataFrame.
    """
    if not _HAS_APP_DEPS:
        pytest.skip("Full app dependencies not installed")
    import linkedin_lead_scoring.api.predict as predict_module

    class FakeModel:
        """Accepts any input and returns a fixed score."""

        def predict_proba(self, X):
            n = len(X) if hasattr(X, "__len__") else 1
            return np.array([[0.35, 0.65]] * n)

    # Load real feature columns
    feature_cols_path = Path("model/feature_columns.json")
    with open(feature_cols_path) as f:
        feature_cols = json.load(f)

    # Load real numeric medians
    medians_path = Path("model/numeric_medians.json")
    numeric_medians = None
    if medians_path.exists():
        with open(medians_path) as f:
            numeric_medians = json.load(f)

    monkeypatch.setenv("APP_ENV", "production")
    # Point to nonexistent files so lifespan doesn't try to joblib.load
    monkeypatch.setattr(predict_module, "_MODEL_PATH", "/nonexistent/model.joblib")

    with TestClient(app) as c:
        # Inject state after lifespan (overrides the failed load)
        monkeypatch.setitem(predict_module._state, "model", FakeModel())
        monkeypatch.setitem(predict_module._state, "preprocessor", {
            "target_encoder": None,
            "te_cols": [],
        })
        monkeypatch.setitem(predict_module._state, "feature_cols", feature_cols)
        monkeypatch.setitem(predict_module._state, "numeric_medians", numeric_medians)
        monkeypatch.setitem(predict_module._state, "model_loaded", True)
        monkeypatch.setitem(predict_module._state, "is_mock", False)
        monkeypatch.setitem(predict_module._state, "model_version", "test-real-0.0.0")
        yield c
