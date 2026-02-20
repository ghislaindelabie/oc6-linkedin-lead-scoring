"""Integration tests for API endpoints (Tasks B.1–B.6).

These tests exercise the full stack (request → middleware → endpoint →
model → logging → response) rather than individual components.
"""
import json

import pytest

VALID_LEAD = {
    "llm_quality": 75,
    "llm_engagement": 0.8,
    "llm_decision_maker": 0.6,
    "llm_company_fit": 1,
    "companyfoundedon": 2015.0,
    "llm_seniority": "Senior",
    "llm_industry": "Technology - SaaS",
    "llm_geography": "international_hub",
    "llm_business_type": "leaders",
    "jobtitle": "VP of Sales",
}


# ---------------------------------------------------------------------------
# Swagger / OpenAPI documentation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_api_docs_available(client):
    """Swagger docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


@pytest.mark.integration
def test_redoc_available(client):
    """ReDoc documentation is accessible."""
    response = client.get("/redoc")
    assert response.status_code == 200


@pytest.mark.integration
def test_openapi_schema(client):
    """OpenAPI schema is valid and contains expected metadata."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "LinkedIn Lead Scoring API"
    assert schema["info"]["version"] == "0.3.0"
    assert "/health" in schema["paths"]


@pytest.mark.integration
def test_openapi_includes_predict_endpoints(client):
    """Swagger docs expose both /predict and /predict/batch."""
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]
    # Both should be POST
    assert "post" in schema["paths"]["/predict"]
    assert "post" in schema["paths"]["/predict/batch"]


# ---------------------------------------------------------------------------
# Full prediction flow (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_predict_flow(dev_client):
    """End-to-end: POST /predict → score + label + confidence + version + timing."""
    response = dev_client.post("/predict", json=VALID_LEAD)
    assert response.status_code == 200
    data = response.json()
    # All response fields present
    assert 0.0 <= data["score"] <= 1.0
    assert data["label"] in ("engaged", "not_engaged")
    assert data["confidence"] in ("low", "medium", "high")
    assert data["model_version"].startswith("mock-")
    assert data["inference_time_ms"] >= 0.0
    # Label must agree with score
    expected_label = "engaged" if data["score"] >= 0.5 else "not_engaged"
    assert data["label"] == expected_label


@pytest.mark.integration
def test_full_batch_flow(dev_client):
    """End-to-end: POST /predict/batch → predictions list + summary stats."""
    payload = {"leads": [VALID_LEAD, {}, VALID_LEAD]}
    response = dev_client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 3
    assert len(data["predictions"]) == 3
    assert 0.0 <= data["avg_score"] <= 1.0
    assert data["high_engagement_count"] >= 0


@pytest.mark.integration
def test_predict_then_batch_same_session(dev_client):
    """Single and batch endpoints can be called in sequence on the same client."""
    r1 = dev_client.post("/predict", json=VALID_LEAD)
    assert r1.status_code == 200
    r2 = dev_client.post("/predict/batch", json={"leads": [VALID_LEAD] * 2})
    assert r2.status_code == 200
    # Scores should be identical (same mock model)
    assert r1.json()["score"] == r2.json()["predictions"][0]["score"]


# ---------------------------------------------------------------------------
# Health endpoint — model status
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_health_reports_model_loaded_in_dev(dev_client):
    """In development mode, /health reports model_loaded=True."""
    data = dev_client.get("/health").json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


@pytest.mark.integration
def test_health_reports_model_not_loaded_without_model(client):
    """Without model files and not in dev mode, model_loaded=False."""
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cors_on_predict(dev_client):
    """CORS preflight on /predict returns allow-origin header."""
    response = dev_client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" in response.headers


@pytest.mark.integration
def test_cors_on_batch(dev_client):
    """CORS preflight on /predict/batch returns allow-origin header."""
    response = dev_client.options(
        "/predict/batch",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" in response.headers


# ---------------------------------------------------------------------------
# Logging creates files (integration with middleware + predict logging)
# ---------------------------------------------------------------------------


@pytest.fixture
def _redirect_logs(tmp_path, monkeypatch):
    """Redirect both log files to tmp_path for isolated verification."""
    import linkedin_lead_scoring.api.middleware as mw_module
    import linkedin_lead_scoring.api.predict as predict_module

    monkeypatch.setattr(mw_module, "_REQUESTS_LOG", str(tmp_path / "api_requests.jsonl"))
    monkeypatch.setattr(predict_module, "_PREDICTIONS_LOG", str(tmp_path / "predictions.jsonl"))
    return tmp_path


@pytest.mark.integration
def test_predict_creates_both_log_files(dev_client, _redirect_logs):
    """A single /predict call produces entries in both log files."""
    tmp = _redirect_logs
    dev_client.post("/predict", json=VALID_LEAD)

    req_log = tmp / "api_requests.jsonl"
    pred_log = tmp / "predictions.jsonl"
    assert req_log.exists()
    assert pred_log.exists()

    # Request log entry is valid JSON with expected path
    req_entry = json.loads(req_log.read_text().strip().splitlines()[-1])
    assert req_entry["path"] == "/predict"
    assert req_entry["status_code"] == 200

    # Prediction log entry contains score matching the response
    pred_entry = json.loads(pred_log.read_text().strip())
    assert "score" in pred_entry
    assert "model_version" in pred_entry


@pytest.mark.integration
def test_batch_creates_prediction_log_entries(dev_client, _redirect_logs):
    """A batch of N leads produces N entries in predictions.jsonl."""
    tmp = _redirect_logs
    dev_client.post("/predict/batch", json={"leads": [VALID_LEAD] * 5})

    pred_log = tmp / "predictions.jsonl"
    lines = pred_log.read_text().strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        entry = json.loads(line)
        assert "timestamp" in entry
        assert "score" in entry
