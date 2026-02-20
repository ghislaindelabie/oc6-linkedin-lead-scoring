"""Smoke tests to verify basic API functionality"""
import tomllib
from pathlib import Path


def test_joblib_declared_in_pyproject():
    """Verify joblib is declared as a project dependency"""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    joblib_deps = [d for d in deps if d.startswith("joblib")]
    assert joblib_deps, "joblib must be listed in [project].dependencies in pyproject.toml"


def test_joblib_importable():
    """Verify joblib can be imported (runtime check)"""
    import joblib
    assert joblib.__version__


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
