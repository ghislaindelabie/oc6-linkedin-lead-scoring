"""Tests for request and prediction logging (Task B.4)"""
import json
import pytest


VALID_LEAD = {
    "llm_quality": 75,
    "llm_engagement": 0.8,
    "llm_seniority": "Senior",
    "jobtitle": "VP of Sales",
}


# ---------------------------------------------------------------------------
# Fixtures — redirect log files to tmp_path so tests stay isolated
# ---------------------------------------------------------------------------


@pytest.fixture
def logged_dev_client(tmp_path, monkeypatch):
    """dev_client with log paths redirected to tmp_path."""
    import linkedin_lead_scoring.api.middleware as mw_module
    import linkedin_lead_scoring.api.predict as predict_module

    monkeypatch.setattr(mw_module, "_REQUESTS_LOG", str(tmp_path / "api_requests.jsonl"))
    monkeypatch.setattr(predict_module, "_PREDICTIONS_LOG", str(tmp_path / "predictions.jsonl"))
    monkeypatch.setenv("APP_ENV", "development")

    from fastapi.testclient import TestClient
    from linkedin_lead_scoring.api.main import app

    with TestClient(app) as c:
        yield c, tmp_path


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


class TestRequestLogging:
    def test_request_log_file_created(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        client.get("/health")
        assert (tmp_path / "api_requests.jsonl").exists()

    def test_request_log_entry_is_valid_json(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        client.get("/health")
        line = (tmp_path / "api_requests.jsonl").read_text().strip().splitlines()[0]
        entry = json.loads(line)  # raises if invalid
        assert isinstance(entry, dict)

    def test_request_log_has_required_fields(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        client.get("/health")
        entry = json.loads((tmp_path / "api_requests.jsonl").read_text().strip())
        assert "timestamp" in entry
        assert "method" in entry
        assert "path" in entry
        assert "status_code" in entry
        assert "response_time_ms" in entry

    def test_request_log_records_correct_method_and_path(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        client.get("/health")
        entry = json.loads((tmp_path / "api_requests.jsonl").read_text().strip())
        assert entry["method"] == "GET"
        assert entry["path"] == "/health"
        assert entry["status_code"] == 200

    def test_request_log_records_post_to_predict(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        client.post("/predict", json=VALID_LEAD)
        lines = (tmp_path / "api_requests.jsonl").read_text().strip().splitlines()
        predict_entries = [json.loads(l) for l in lines if "/predict" in l]
        assert len(predict_entries) >= 1
        assert predict_entries[0]["method"] == "POST"

    def test_request_log_appends_multiple_entries(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        client.get("/health")
        client.get("/health")
        lines = (tmp_path / "api_requests.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_request_log_response_time_is_non_negative(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        client.get("/health")
        entry = json.loads((tmp_path / "api_requests.jsonl").read_text().strip())
        assert entry["response_time_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------------------------


class TestPredictionLogging:
    def test_prediction_log_created_after_predict(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        client.post("/predict", json=VALID_LEAD)
        assert (tmp_path / "predictions.jsonl").exists()

    def test_prediction_log_entry_is_valid_json(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        client.post("/predict", json=VALID_LEAD)
        line = (tmp_path / "predictions.jsonl").read_text().strip()
        entry = json.loads(line)
        assert isinstance(entry, dict)

    def test_prediction_log_has_required_fields(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        client.post("/predict", json=VALID_LEAD)
        entry = json.loads((tmp_path / "predictions.jsonl").read_text().strip())
        assert "timestamp" in entry
        assert "input" in entry
        assert "score" in entry
        assert "label" in entry
        assert "inference_ms" in entry
        assert "model_version" in entry

    def test_prediction_log_score_matches_response(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        response = client.post("/predict", json=VALID_LEAD)
        response_score = response.json()["score"]
        log_score = json.loads((tmp_path / "predictions.jsonl").read_text().strip())["score"]
        assert log_score == pytest.approx(response_score, abs=0.001)

    def test_batch_logs_one_entry_per_lead(self, logged_dev_client):
        client, tmp_path = logged_dev_client
        client.post("/predict/batch", json={"leads": [VALID_LEAD] * 3})
        lines = (tmp_path / "predictions.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

    def test_log_write_failure_does_not_crash_endpoint(self, tmp_path, monkeypatch):
        """Endpoint must return 200 even if the log write raises."""
        import linkedin_lead_scoring.api.predict as predict_module

        # Redirect requests log too so middleware doesn't fail
        import linkedin_lead_scoring.api.middleware as mw_module
        monkeypatch.setattr(mw_module, "_REQUESTS_LOG", str(tmp_path / "req.jsonl"))

        # Make prediction log write fail
        monkeypatch.setattr(predict_module, "_PREDICTIONS_LOG", "/nonexistent/dir/predictions.jsonl")
        monkeypatch.setenv("APP_ENV", "development")

        from fastapi.testclient import TestClient
        from linkedin_lead_scoring.api.main import app

        with TestClient(app) as client:
            response = client.post("/predict", json=VALID_LEAD)
        assert response.status_code == 200
