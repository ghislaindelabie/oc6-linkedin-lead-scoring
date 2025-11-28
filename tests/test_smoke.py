"""Smoke tests to verify basic API functionality"""


def test_health_endpoint(client):
    """Test health check endpoint returns correct status"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "linkedin-lead-scoring-api"
    assert "version" in data


def test_root_endpoint(client):
    """Test root endpoint returns HTML landing page"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "LinkedIn Lead Scoring API" in response.text
    assert "/docs" in response.text
