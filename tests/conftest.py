"""Shared pytest fixtures for API testing"""
import pytest
from fastapi.testclient import TestClient
from src.linkedin_lead_scoring.api.main import app


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)
