"""Unit tests for /predict endpoint (Task B.2)"""
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
# Confidence helper — tested in isolation
# ---------------------------------------------------------------------------


class TestConfidenceMapping:
    def test_high_confidence(self):
        from linkedin_lead_scoring.api.predict import _get_confidence

        assert _get_confidence(0.7) == "high"
        assert _get_confidence(1.0) == "high"

    def test_medium_confidence(self):
        from linkedin_lead_scoring.api.predict import _get_confidence

        assert _get_confidence(0.4) == "medium"
        assert _get_confidence(0.5) == "medium"
        assert _get_confidence(0.699) == "medium"

    def test_low_confidence(self):
        from linkedin_lead_scoring.api.predict import _get_confidence

        assert _get_confidence(0.0) == "low"
        assert _get_confidence(0.399) == "low"


# ---------------------------------------------------------------------------
# /predict — happy path (mock/dev mode)
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_predict_returns_correct_schema(self, dev_client):
        response = dev_client.post("/predict", json=VALID_LEAD)
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0
        assert data["label"] in ("engaged", "not_engaged")
        assert data["confidence"] in ("low", "medium", "high")
        assert "model_version" in data
        assert data["inference_time_ms"] >= 0.0

    def test_predict_empty_lead_succeeds(self, dev_client):
        """All fields optional — empty body is valid."""
        response = dev_client.post("/predict", json={})
        assert response.status_code == 200

    def test_predict_label_consistent_with_score(self, dev_client):
        response = dev_client.post("/predict", json=VALID_LEAD)
        data = response.json()
        expected = "engaged" if data["score"] >= 0.5 else "not_engaged"
        assert data["label"] == expected

    def test_predict_confidence_consistent_with_score(self, dev_client):
        from linkedin_lead_scoring.api.predict import _get_confidence

        response = dev_client.post("/predict", json=VALID_LEAD)
        data = response.json()
        assert data["confidence"] == _get_confidence(data["score"])

    def test_predict_endpoint_in_openapi(self, dev_client):
        schema = dev_client.get("/openapi.json").json()
        assert "/predict" in schema["paths"]

    def test_predict_mock_model_version_set(self, dev_client):
        """Mock mode must report a model version string."""
        response = dev_client.post("/predict", json={})
        assert response.json()["model_version"] != ""


# ---------------------------------------------------------------------------
# /predict — validation errors
# ---------------------------------------------------------------------------


class TestPredictValidation:
    def test_llm_quality_over_100_returns_422(self, dev_client):
        response = dev_client.post("/predict", json={"llm_quality": 150})
        assert response.status_code == 422

    def test_negative_engagement_returns_422(self, dev_client):
        response = dev_client.post("/predict", json={"llm_engagement": -0.1})
        assert response.status_code == 422

    def test_wrong_type_returns_422(self, dev_client):
        response = dev_client.post("/predict", json={"llm_quality": "high"})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /predict — model not loaded → 503
# ---------------------------------------------------------------------------


class TestPredictModelNotLoaded:
    def test_predict_returns_503_when_model_not_loaded(self, client, monkeypatch):
        import linkedin_lead_scoring.api.predict as predict_module

        monkeypatch.setitem(predict_module._state, "model_loaded", False)
        response = client.post("/predict", json=VALID_LEAD)
        assert response.status_code == 503

    def test_503_body_is_structured(self, client, monkeypatch):
        import linkedin_lead_scoring.api.predict as predict_module

        monkeypatch.setitem(predict_module._state, "model_loaded", False)
        data = client.post("/predict", json={}).json()
        assert "detail" in data


# ---------------------------------------------------------------------------
# /health — model_loaded field
# ---------------------------------------------------------------------------


class TestHealthModelStatus:
    def test_health_model_loaded_true_in_dev_mode(self, dev_client):
        data = dev_client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_model_loaded_false_without_model(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    def test_cors_allow_origin_header_present(self, dev_client):
        response = dev_client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" in response.headers
