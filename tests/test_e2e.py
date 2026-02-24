"""E2E tests for the staging API using httpx.

These tests are designed to run against the live staging API.
They're skipped in normal CI (no STAGING_URL env var) and only
run post-deploy in the staging workflow.
"""
import os
import pytest
import httpx

STAGING_URL = os.getenv("STAGING_URL", "")

pytestmark = pytest.mark.skipif(
    not STAGING_URL,
    reason="STAGING_URL not set — skipping E2E tests"
)


class TestE2EHealth:
    def test_health_endpoint_returns_200(self):
        resp = httpx.get(f"{STAGING_URL}/health", timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

    def test_health_model_loaded(self):
        resp = httpx.get(f"{STAGING_URL}/health", timeout=30)
        data = resp.json()
        assert data["model_loaded"] is True


class TestE2EPredict:
    def test_predict_single_lead(self):
        lead = {
            "llm_quality": 75,
            "llm_engagement": 0.8,
            "llm_decision_maker": 0.6,
            "llm_company_fit": 1,
            "llm_seniority": "Senior",
            "jobtitle": "VP of Sales",
        }
        resp = httpx.post(f"{STAGING_URL}/predict", json=lead, timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["score"] <= 1.0
        assert data["label"] in ("engaged", "not_engaged")

    def test_predict_empty_lead(self):
        resp = httpx.post(f"{STAGING_URL}/predict", json={}, timeout=30)
        assert resp.status_code == 200

    def test_predict_batch(self):
        leads = [
            {"llm_quality": 75, "jobtitle": "CEO"},
            {"llm_quality": 30},
            {},
        ]
        resp = httpx.post(f"{STAGING_URL}/predict/batch", json={"leads": leads}, timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 3
        assert len(data["predictions"]) == 3

    def test_predict_batch_vs_single_consistency(self):
        """Batch scores should match individual scores."""
        lead = {"llm_quality": 75, "llm_seniority": "Senior"}
        # Single
        single_resp = httpx.post(f"{STAGING_URL}/predict", json=lead, timeout=30)
        single_score = single_resp.json()["score"]
        # Batch of 1
        batch_resp = httpx.post(f"{STAGING_URL}/predict/batch", json={"leads": [lead]}, timeout=30)
        batch_score = batch_resp.json()["predictions"][0]["score"]
        assert abs(single_score - batch_score) < 0.0001


class TestE2EValidation:
    def test_invalid_lead_returns_422(self):
        resp = httpx.post(f"{STAGING_URL}/predict", json={"llm_quality": 999}, timeout=30)
        assert resp.status_code == 422

    def test_docs_endpoint_accessible(self):
        resp = httpx.get(f"{STAGING_URL}/docs", timeout=30, follow_redirects=True)
        assert resp.status_code == 200
