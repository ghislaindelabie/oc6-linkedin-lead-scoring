"""Tests for scripts/validate_pipeline.py"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# The validate_pipeline.py script does sys.path manipulation, so import carefully
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestLoadArtifacts:
    def test_load_artifacts_returns_expected_keys(self):
        from validate_pipeline import load_artifacts, MODEL_DIR
        artifacts = load_artifacts(MODEL_DIR)
        assert "model" in artifacts
        assert "preprocessor" in artifacts
        assert "feature_columns" in artifacts
        assert "numeric_medians" in artifacts

    def test_load_artifacts_model_is_not_none(self):
        from validate_pipeline import load_artifacts, MODEL_DIR
        artifacts = load_artifacts(MODEL_DIR)
        assert artifacts["model"] is not None

    def test_load_artifacts_feature_columns_is_list(self):
        from validate_pipeline import load_artifacts, MODEL_DIR
        artifacts = load_artifacts(MODEL_DIR)
        assert isinstance(artifacts["feature_columns"], list)
        assert len(artifacts["feature_columns"]) > 0

    def test_load_artifacts_preprocessor_is_dict(self):
        from validate_pipeline import load_artifacts, MODEL_DIR
        artifacts = load_artifacts(MODEL_DIR)
        # preprocessor should be a dict with target_encoder / te_cols
        assert isinstance(artifacts["preprocessor"], dict)

    def test_load_artifacts_numeric_medians_is_dict_or_none(self):
        from validate_pipeline import load_artifacts, MODEL_DIR
        artifacts = load_artifacts(MODEL_DIR)
        assert artifacts["numeric_medians"] is None or isinstance(
            artifacts["numeric_medians"], dict
        )

    def test_load_artifacts_missing_model_raises(self, tmp_path):
        """load_artifacts() must raise when the model file is missing."""
        from validate_pipeline import load_artifacts
        with pytest.raises(Exception):
            load_artifacts(tmp_path)  # no files exist here


class TestPredictLocal:
    def test_predict_local_full_lead(self):
        from validate_pipeline import predict_local, load_artifacts, MODEL_DIR, SAMPLE_LEADS
        artifacts = load_artifacts(MODEL_DIR)
        full_lead = next(s for s in SAMPLE_LEADS if s["name"] == "full_lead")
        score = predict_local(artifacts, full_lead["data"])
        assert 0.0 <= score <= 1.0

    def test_predict_local_empty_lead(self):
        from validate_pipeline import predict_local, load_artifacts, MODEL_DIR
        artifacts = load_artifacts(MODEL_DIR)
        score = predict_local(artifacts, {})
        assert 0.0 <= score <= 1.0

    def test_predict_local_all_sample_leads(self):
        from validate_pipeline import predict_local, load_artifacts, MODEL_DIR, SAMPLE_LEADS
        artifacts = load_artifacts(MODEL_DIR)
        for sample in SAMPLE_LEADS:
            score = predict_local(artifacts, sample["data"])
            assert 0.0 <= score <= 1.0, f"Invalid score for {sample['name']}: {score}"

    def test_predict_local_score_is_float(self):
        from validate_pipeline import predict_local, load_artifacts, MODEL_DIR
        artifacts = load_artifacts(MODEL_DIR)
        score = predict_local(artifacts, {"llm_quality": 75})
        assert isinstance(score, float)

    def test_predict_local_deterministic(self):
        """Two calls with the same input must return the same score."""
        from validate_pipeline import predict_local, load_artifacts, MODEL_DIR
        artifacts = load_artifacts(MODEL_DIR)
        lead = {"llm_quality": 60, "llm_seniority": "Senior", "jobtitle": "VP Sales"}
        score1 = predict_local(artifacts, lead)
        score2 = predict_local(artifacts, lead)
        assert score1 == score2


class TestPredictApi:
    def test_predict_api_returns_none_on_network_error(self):
        from validate_pipeline import predict_api
        # Port 99999 is invalid — connection refused immediately
        score = predict_api("http://localhost:99999", {"llm_quality": 50})
        assert score is None

    def test_predict_batch_api_returns_nones_on_error(self):
        from validate_pipeline import predict_batch_api
        results = predict_batch_api(
            "http://localhost:99999",
            [{"llm_quality": 50}, {}],
        )
        assert len(results) == 2
        assert all(r is None for r in results)

    def test_predict_api_returns_none_on_http_error(self):
        """HTTP 404 from the server should return None, not raise."""
        from validate_pipeline import predict_api
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("HTTP 404")
            mock_post.return_value = mock_resp
            score = predict_api("http://fakeserver.local", {"llm_quality": 50})
        assert score is None

    def test_predict_batch_api_returns_correct_count_on_success(self):
        """Successful batch API call should return one score per lead."""
        from validate_pipeline import predict_batch_api
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {
                "predictions": [{"score": 0.6}, {"score": 0.4}, {"score": 0.7}]
            }
            mock_post.return_value = mock_resp
            results = predict_batch_api(
                "http://fakeserver.local",
                [{"llm_quality": 50}, {}, {"llm_quality": 80}],
            )
        assert len(results) == 3
        assert results == [0.6, 0.4, 0.7]


class TestCompareScores:
    def test_compare_all_match(self, capsys):
        from validate_pipeline import compare_scores
        results = [
            {"name": "lead_a", "local": 0.5000, "api": 0.5000},
            {"name": "lead_b", "local": 0.7500, "api": 0.7500},
        ]
        assert compare_scores(results, tolerance=0.0001) is True

    def test_compare_mismatch_detected(self, capsys):
        from validate_pipeline import compare_scores
        results = [
            {"name": "lead_a", "local": 0.5000, "api": 0.9000},
        ]
        assert compare_scores(results, tolerance=0.0001) is False

    def test_compare_skips_missing_api(self, capsys):
        from validate_pipeline import compare_scores
        results = [
            {"name": "lead_a", "local": 0.5000},
        ]
        assert compare_scores(results, tolerance=0.0001) is True

    def test_compare_within_tolerance_passes(self, capsys):
        from validate_pipeline import compare_scores
        results = [
            {"name": "lead_a", "local": 0.5000, "api": 0.5001},
        ]
        # delta 0.0001 equals tolerance — should pass
        assert compare_scores(results, tolerance=0.0001) is True

    def test_compare_just_over_tolerance_fails(self, capsys):
        from validate_pipeline import compare_scores
        results = [
            {"name": "lead_a", "local": 0.5000, "api": 0.5002},
        ]
        assert compare_scores(results, tolerance=0.0001) is False

    def test_compare_mixed_some_mismatch(self, capsys):
        from validate_pipeline import compare_scores
        results = [
            {"name": "good", "local": 0.5000, "api": 0.5000},
            {"name": "bad",  "local": 0.5000, "api": 0.9999},
        ]
        assert compare_scores(results, tolerance=0.0001) is False

    def test_compare_empty_results(self, capsys):
        from validate_pipeline import compare_scores
        assert compare_scores([], tolerance=0.0001) is True


class TestSampleLeads:
    def test_sample_leads_not_empty(self):
        from validate_pipeline import SAMPLE_LEADS
        assert len(SAMPLE_LEADS) >= 5

    def test_each_sample_has_name_and_data(self):
        from validate_pipeline import SAMPLE_LEADS
        for sample in SAMPLE_LEADS:
            assert "name" in sample
            assert "data" in sample
            assert isinstance(sample["data"], dict)

    def test_full_lead_exists(self):
        from validate_pipeline import SAMPLE_LEADS
        names = [s["name"] for s in SAMPLE_LEADS]
        assert "full_lead" in names

    def test_empty_lead_exists(self):
        from validate_pipeline import SAMPLE_LEADS
        names = [s["name"] for s in SAMPLE_LEADS]
        assert "empty_lead" in names

    def test_sample_names_are_unique(self):
        from validate_pipeline import SAMPLE_LEADS
        names = [s["name"] for s in SAMPLE_LEADS]
        assert len(names) == len(set(names)), "Duplicate sample lead names detected"
