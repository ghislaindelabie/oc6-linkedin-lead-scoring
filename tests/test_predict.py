"""Unit tests for /predict and /predict/batch endpoints (Tasks B.2, B.3)"""
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

    def test_no_body_returns_422(self, dev_client):
        """POST with no JSON body at all should fail."""
        response = dev_client.post("/predict")
        assert response.status_code == 422

    def test_multiple_invalid_fields_returns_422(self, dev_client):
        """Several invalid fields at once still 422."""
        response = dev_client.post("/predict", json={
            "llm_quality": 200,
            "llm_engagement": -5.0,
            "llm_company_fit": 10,
        })
        assert response.status_code == 422
        # FastAPI reports multiple errors in the detail array
        errors = response.json()["detail"]
        assert len(errors) >= 2

    def test_boundary_quality_0_accepted(self, dev_client):
        response = dev_client.post("/predict", json={"llm_quality": 0})
        assert response.status_code == 200

    def test_boundary_quality_100_accepted(self, dev_client):
        response = dev_client.post("/predict", json={"llm_quality": 100})
        assert response.status_code == 200

    def test_boundary_engagement_0_accepted(self, dev_client):
        response = dev_client.post("/predict", json={"llm_engagement": 0.0})
        assert response.status_code == 200

    def test_boundary_engagement_1_accepted(self, dev_client):
        response = dev_client.post("/predict", json={"llm_engagement": 1.0})
        assert response.status_code == 200

    def test_boundary_decision_maker_0_accepted(self, dev_client):
        response = dev_client.post("/predict", json={"llm_decision_maker": 0.0})
        assert response.status_code == 200

    def test_boundary_decision_maker_1_accepted(self, dev_client):
        response = dev_client.post("/predict", json={"llm_decision_maker": 1.0})
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /predict — model not loaded → 503
# ---------------------------------------------------------------------------


class TestPredictModelNotLoaded:
    def test_predict_returns_503_when_model_not_loaded(self, client, monkeypatch):
        import linkedin_lead_scoring.api.predict as predict_module

        monkeypatch.setitem(predict_module._state, "model_loaded", False)
        response = client.post("/predict", json=VALID_LEAD)
        assert response.status_code == 503

    def test_503_body_is_structured_error_response(self, client, monkeypatch):
        """503 should use ErrorResponse schema: error + message fields."""
        import linkedin_lead_scoring.api.predict as predict_module

        monkeypatch.setitem(predict_module._state, "model_loaded", False)
        data = client.post("/predict", json={}).json()
        assert data["error"] == "service_unavailable"
        assert "message" in data

    def test_500_uses_structured_error_response(self, dev_client, monkeypatch):
        """Internal errors should use ErrorResponse schema, not raw detail."""
        import linkedin_lead_scoring.api.predict as predict_module

        # Force a crash by replacing model with something that raises
        class CrashModel:
            def predict_proba(self, X):
                raise RuntimeError("boom")

        monkeypatch.setitem(predict_module._state, "model", CrashModel())
        monkeypatch.setitem(predict_module._state, "is_mock", True)
        data = dev_client.post("/predict", json={}).json()
        assert data["error"] == "internal_error"
        assert "message" in data
        # Must NOT leak stack traces
        assert "boom" not in data.get("message", "")
        assert "Traceback" not in str(data)

    def test_422_uses_structured_error_response(self, dev_client):
        """Validation errors should use ErrorResponse schema."""
        data = dev_client.post("/predict", json={"llm_quality": 999}).json()
        assert data["error"] == "validation_error"
        assert "message" in data
        assert data["detail"] is not None


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


# ---------------------------------------------------------------------------
# /predict/batch (Task B.3)
# ---------------------------------------------------------------------------


class TestBatchPredictEndpoint:
    def test_batch_returns_correct_schema(self, dev_client):
        payload = {"leads": [VALID_LEAD, VALID_LEAD]}
        response = dev_client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total_count" in data
        assert "avg_score" in data
        assert "high_engagement_count" in data

    def test_batch_total_count_matches_input(self, dev_client):
        payload = {"leads": [VALID_LEAD] * 5}
        data = dev_client.post("/predict/batch", json=payload).json()
        assert data["total_count"] == 5
        assert len(data["predictions"]) == 5

    def test_batch_single_lead_works(self, dev_client):
        payload = {"leads": [VALID_LEAD]}
        response = dev_client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        assert response.json()["total_count"] == 1

    def test_batch_summary_stats_correct(self, dev_client):
        """Mock always returns 0.65 — all leads engaged, avg=0.65."""
        payload = {"leads": [VALID_LEAD] * 4}
        data = dev_client.post("/predict/batch", json=payload).json()
        assert data["total_count"] == 4
        assert data["avg_score"] == pytest.approx(0.65, abs=0.01)
        assert data["high_engagement_count"] == 4  # all scores >= 0.5

    def test_batch_high_engagement_count_accurate(self, dev_client):
        """high_engagement_count = leads with score >= 0.5."""
        payload = {"leads": [VALID_LEAD] * 3}
        data = dev_client.post("/predict/batch", json=payload).json()
        predictions = data["predictions"]
        expected = sum(1 for p in predictions if p["score"] >= 0.5)
        assert data["high_engagement_count"] == expected

    def test_batch_each_prediction_has_correct_schema(self, dev_client):
        payload = {"leads": [VALID_LEAD, {}]}  # one full, one empty (all optional)
        data = dev_client.post("/predict/batch", json=payload).json()
        for pred in data["predictions"]:
            assert 0.0 <= pred["score"] <= 1.0
            assert pred["label"] in ("engaged", "not_engaged")
            assert pred["confidence"] in ("low", "medium", "high")
            assert pred["inference_time_ms"] >= 0.0

    def test_batch_empty_list_returns_422(self, dev_client):
        response = dev_client.post("/predict/batch", json={"leads": []})
        assert response.status_code == 422

    def test_batch_503_when_model_not_loaded(self, client, monkeypatch):
        import linkedin_lead_scoring.api.predict as predict_module

        monkeypatch.setitem(predict_module._state, "model_loaded", False)
        response = client.post("/predict/batch", json={"leads": [VALID_LEAD]})
        assert response.status_code == 503

    def test_batch_over_10000_returns_422(self, dev_client):
        """Endpoint-level check: >10000 leads rejected by Pydantic validation."""
        payload = {"leads": [{}] * 10_001}
        response = dev_client.post("/predict/batch", json=payload)
        assert response.status_code == 422

    def test_batch_with_invalid_lead_returns_422(self, dev_client):
        """One bad apple spoils the batch — validation rejects the whole request."""
        payload = {"leads": [VALID_LEAD, {"llm_quality": 999}]}
        response = dev_client.post("/predict/batch", json=payload)
        assert response.status_code == 422

    def test_batch_endpoint_in_openapi(self, dev_client):
        schema = dev_client.get("/openapi.json").json()
        assert "/predict/batch" in schema["paths"]


# ---------------------------------------------------------------------------
# mock_model fixture usage (Task B.5)
# ---------------------------------------------------------------------------


class TestMockModelFixture:
    def test_mock_model_predict_returns_score(self, client, mock_model):
        """The mock_model conftest fixture injects a FakeModel into _state."""
        response = client.post("/predict", json=VALID_LEAD)
        assert response.status_code == 200
        data = response.json()
        assert data["score"] == pytest.approx(0.7, abs=0.01)
        assert data["model_version"] == "test-0.0.0"


# ---------------------------------------------------------------------------
# Real preprocessing path (is_mock=False) — exercises the full pipeline
# ---------------------------------------------------------------------------


class TestRealPreprocessingPath:
    def test_predict_with_real_preprocessing(self, client_with_preprocessor):
        response = client_with_preprocessor.post("/predict", json=VALID_LEAD)
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["score"] <= 1.0
        assert data["label"] in ("engaged", "not_engaged")
        assert data["model_version"] == "test-real-0.0.0"

    def test_predict_empty_lead_with_real_preprocessing(self, client_with_preprocessor):
        response = client_with_preprocessor.post("/predict", json={})
        assert response.status_code == 200

    def test_predict_partial_lead_with_real_preprocessing(self, client_with_preprocessor):
        partial = {"llm_quality": 80, "jobtitle": "CTO", "summary": "Tech leader"}
        response = client_with_preprocessor.post("/predict", json=partial)
        assert response.status_code == 200

    def test_batch_with_real_preprocessing(self, client_with_preprocessor):
        payload = {"leads": [VALID_LEAD, {}, {"jobtitle": "Manager"}]}
        response = client_with_preprocessor.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 3
        assert len(data["predictions"]) == 3


# ---------------------------------------------------------------------------
# Error path tests — uncovered branches in predict.py (T11)
# ---------------------------------------------------------------------------


class TestPredictErrorPaths:
    def test_predict_internal_error_500_with_real_preprocessing(
        self, client_with_preprocessor, monkeypatch
    ):
        """Model.predict_proba() raising in the non-mock path must return 500."""
        import linkedin_lead_scoring.api.predict as predict_module

        class CrashModel:
            def predict_proba(self, X):
                raise RuntimeError("deliberate crash in predict_proba")

        monkeypatch.setitem(predict_module._state, "model", CrashModel())
        monkeypatch.setitem(predict_module._state, "is_mock", False)

        response = client_with_preprocessor.post("/predict", json=VALID_LEAD)
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "internal_error"
        assert "message" in data
        # Stack trace must not leak
        assert "deliberate crash" not in data.get("message", "")
        assert "Traceback" not in str(data)

    def test_batch_internal_error_500(self, client_with_preprocessor, monkeypatch):
        """Batch endpoint: model.predict_proba() raising must return 500."""
        import linkedin_lead_scoring.api.predict as predict_module

        class CrashModel:
            def predict_proba(self, X):
                raise ValueError("batch crash")

        monkeypatch.setitem(predict_module._state, "model", CrashModel())
        monkeypatch.setitem(predict_module._state, "is_mock", False)

        response = client_with_preprocessor.post(
            "/predict/batch", json={"leads": [VALID_LEAD, {}]}
        )
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "internal_error"
        assert "batch crash" not in data.get("message", "")

    def test_predict_log_write_failure_non_blocking(
        self, client_with_preprocessor, monkeypatch, tmp_path
    ):
        """Prediction log write failure must not prevent a successful 200 response."""
        import linkedin_lead_scoring.api.predict as predict_module

        # Point log to a path where the parent is a file (unwritable as directory)
        unwritable_log = str(tmp_path / "not_a_dir" / "predictions.jsonl")
        # Create a file where the directory would be, so makedirs fails
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("I am a file, not a directory")
        monkeypatch.setattr(predict_module, "_PREDICTIONS_LOG", unwritable_log)

        response = client_with_preprocessor.post("/predict", json=VALID_LEAD)
        # Prediction must still succeed
        assert response.status_code == 200
        assert 0.0 <= response.json()["score"] <= 1.0

    def test_build_log_entry_has_expected_fields(self):
        """Unit-test _build_log_entry directly for required output fields."""
        from linkedin_lead_scoring.api.predict import _build_log_entry
        from linkedin_lead_scoring.api.schemas import LeadInput, LeadPrediction

        lead = LeadInput(llm_quality=80, jobtitle="CTO")
        prediction = LeadPrediction(
            score=0.72,
            label="engaged",
            confidence="high",
            model_version="test-0.3.0",
            inference_time_ms=12.5,
        )
        entry = _build_log_entry(lead, prediction)

        assert "timestamp" in entry
        assert "input" in entry
        assert "score" in entry
        assert "label" in entry
        assert "inference_ms" in entry
        assert "model_version" in entry
        assert entry["score"] == 0.72
        assert entry["label"] == "engaged"
        assert entry["model_version"] == "test-0.3.0"
        assert isinstance(entry["input"], dict)

    def test_lifespan_load_failure_model_not_loaded(self, monkeypatch, tmp_path):
        """When model file is a non-joblib file, lifespan catches the exception
        and model_loaded remains False."""
        import linkedin_lead_scoring.api.predict as predict_module
        from fastapi.testclient import TestClient
        from linkedin_lead_scoring.api.main import app

        # Create a file that exists but cannot be joblib.load()-ed
        bad_model_file = tmp_path / "bad_model.joblib"
        bad_model_file.write_text("not a joblib file")

        # Also need preprocessor and feature_cols to exist (mock with bad content)
        bad_preprocessor = tmp_path / "preprocessor.joblib"
        bad_preprocessor.write_text("not a preprocessor")
        feature_cols_file = tmp_path / "feature_columns.json"
        feature_cols_file.write_text('["col1", "col2"]')

        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setattr(predict_module, "_MODEL_PATH", str(bad_model_file))
        monkeypatch.setattr(predict_module, "_PREPROCESSOR_PATH", str(bad_preprocessor))
        monkeypatch.setattr(predict_module, "_FEATURE_COLS_PATH", str(feature_cols_file))

        with TestClient(app) as c:
            health = c.get("/health").json()
            assert health["model_loaded"] is False
