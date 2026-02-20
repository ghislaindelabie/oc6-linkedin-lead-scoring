"""Integration tests for API endpoints"""
import pytest


@pytest.mark.integration
def test_api_docs_available(client):
    """Test that Swagger docs are accessible"""
    response = client.get("/docs")
    assert response.status_code == 200


@pytest.mark.integration
def test_redoc_available(client):
    """Test that ReDoc documentation is accessible"""
    response = client.get("/redoc")
    assert response.status_code == 200


@pytest.mark.integration
def test_openapi_schema(client):
    """Test OpenAPI schema is valid and contains expected metadata"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "LinkedIn Lead Scoring API"
    assert schema["info"]["version"] == "0.3.0"
    assert "/health" in schema["paths"]
