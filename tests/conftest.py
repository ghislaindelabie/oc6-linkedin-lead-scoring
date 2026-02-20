"""Shared pytest fixtures for API testing"""
import pytest
from fastapi.testclient import TestClient

from linkedin_lead_scoring.api.main import app


@pytest.fixture
def client():
    """Test client with lifespan — no model files, no dev mode → model_loaded=False."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def dev_client(monkeypatch):
    """Test client with mock model loaded (APP_ENV=development, no real model files needed)."""
    monkeypatch.setenv("APP_ENV", "development")
    with TestClient(app) as c:
        yield c


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
