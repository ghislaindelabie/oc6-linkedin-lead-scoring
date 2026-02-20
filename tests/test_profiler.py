"""Unit tests for the inference profiler module.

Tests cover:
- profile_model_inference: timing statistics from repeated predict_proba calls
- run_cprofile: cProfile output captured correctly
- save_profile_results: JSON written to disk
- format_profile_summary: human-readable string output
"""
import json
import os
import tempfile

import numpy as np
import pytest
import xgboost as xgb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_xgb_model():
    """Train a tiny XGBoost classifier for fast testing (no I/O required)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def sample_X():
    """Feature matrix for benchmark calls."""
    rng = np.random.default_rng(1)
    return rng.standard_normal((50, 8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests for profile_model_inference
# ---------------------------------------------------------------------------

class TestProfileModelInference:
    def test_returns_dict(self, tiny_xgb_model, sample_X):
        from linkedin_lead_scoring.monitoring.profiler import profile_model_inference

        result = profile_model_inference(tiny_xgb_model, sample_X, n_calls=20)
        assert isinstance(result, dict)

    def test_required_keys(self, tiny_xgb_model, sample_X):
        from linkedin_lead_scoring.monitoring.profiler import profile_model_inference

        result = profile_model_inference(tiny_xgb_model, sample_X, n_calls=20)
        required = {"mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "n_calls"}
        assert required.issubset(result.keys())

    def test_n_calls_recorded(self, tiny_xgb_model, sample_X):
        from linkedin_lead_scoring.monitoring.profiler import profile_model_inference

        result = profile_model_inference(tiny_xgb_model, sample_X, n_calls=30)
        assert result["n_calls"] == 30

    def test_percentile_ordering(self, tiny_xgb_model, sample_X):
        from linkedin_lead_scoring.monitoring.profiler import profile_model_inference

        result = profile_model_inference(tiny_xgb_model, sample_X, n_calls=50)
        assert result["min_ms"] <= result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"] <= result["max_ms"]

    def test_times_are_positive(self, tiny_xgb_model, sample_X):
        from linkedin_lead_scoring.monitoring.profiler import profile_model_inference

        result = profile_model_inference(tiny_xgb_model, sample_X, n_calls=20)
        assert result["mean_ms"] > 0
        assert result["min_ms"] > 0

    def test_batch_size_one_works(self, tiny_xgb_model):
        """Single-row input should not raise."""
        from linkedin_lead_scoring.monitoring.profiler import profile_model_inference

        X_single = np.random.default_rng(5).standard_normal((1, 8)).astype(np.float32)
        result = profile_model_inference(tiny_xgb_model, X_single, n_calls=10)
        assert result["n_calls"] == 10


# ---------------------------------------------------------------------------
# Tests for save_profile_results
# ---------------------------------------------------------------------------

class TestSaveProfileResults:
    def test_creates_json_file(self, tiny_xgb_model, sample_X):
        from linkedin_lead_scoring.monitoring.profiler import (
            profile_model_inference,
            save_profile_results,
        )

        result = profile_model_inference(tiny_xgb_model, sample_X, n_calls=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "profile_results.json")
            returned = save_profile_results(result, path)

            assert os.path.exists(path)
            assert returned == path
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["n_calls"] == 10

    def test_creates_parent_dirs(self, tiny_xgb_model, sample_X):
        from linkedin_lead_scoring.monitoring.profiler import (
            profile_model_inference,
            save_profile_results,
        )

        result = profile_model_inference(tiny_xgb_model, sample_X, n_calls=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "results.json")
            save_profile_results(result, path)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Tests for format_profile_summary
# ---------------------------------------------------------------------------

class TestFormatProfileSummary:
    def test_returns_string(self, tiny_xgb_model, sample_X):
        from linkedin_lead_scoring.monitoring.profiler import (
            format_profile_summary,
            profile_model_inference,
        )

        result = profile_model_inference(tiny_xgb_model, sample_X, n_calls=10)
        summary = format_profile_summary(result)
        assert isinstance(summary, str)

    def test_contains_key_metrics(self, tiny_xgb_model, sample_X):
        from linkedin_lead_scoring.monitoring.profiler import (
            format_profile_summary,
            profile_model_inference,
        )

        result = profile_model_inference(tiny_xgb_model, sample_X, n_calls=10)
        summary = format_profile_summary(result)
        # Should mention p95 and mean somewhere
        assert "p95" in summary.lower() or "95" in summary
        assert "mean" in summary.lower()


# ---------------------------------------------------------------------------
# Tests for run_cprofile
# ---------------------------------------------------------------------------

class TestRunCProfile:
    def test_returns_stats_string(self, tiny_xgb_model, sample_X):
        from linkedin_lead_scoring.monitoring.profiler import run_cprofile

        stats_str = run_cprofile(tiny_xgb_model, sample_X, n_calls=10)
        assert isinstance(stats_str, str)
        assert len(stats_str) > 0

    def test_stats_contain_function_names(self, tiny_xgb_model, sample_X):
        from linkedin_lead_scoring.monitoring.profiler import run_cprofile

        stats_str = run_cprofile(tiny_xgb_model, sample_X, n_calls=10)
        # cProfile output always has "function calls" in header
        assert "function" in stats_str.lower() or "cumtime" in stats_str.lower()
