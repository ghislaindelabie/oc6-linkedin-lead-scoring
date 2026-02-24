"""Unit tests for the ONNX model optimization module.

Tests cover:
- XGBoost → ONNX conversion produces a valid ONNX model
- ONNX Runtime inference produces the same predictions as the original model
- Benchmark comparison returns required keys and valid values
- Results are saved to JSON correctly
- Memory measurement returns a positive value
"""
import json
import os
import tempfile

import numpy as np
import pytest
import xgboost as xgb

# Skip entire module if ONNX stack is not installed
pytest.importorskip("onnx", reason="onnx not installed")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
pytest.importorskip("onnxmltools", reason="onnxmltools not installed")


# ---------------------------------------------------------------------------
# Shared fixture: tiny XGBoost classifier (module-scoped for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 6)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = xgb.XGBClassifier(n_estimators=15, max_depth=3, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def sample_X():
    return np.random.default_rng(7).standard_normal((20, 6)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: convert_xgboost_to_onnx
# ---------------------------------------------------------------------------

class TestConvertXgboostToOnnx:
    def test_returns_onnx_model_object(self, tiny_model):
        from linkedin_lead_scoring.monitoring.onnx_optimizer import convert_xgboost_to_onnx
        import onnx

        onnx_model = convert_xgboost_to_onnx(tiny_model, n_features=6)
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_onnx_model_is_valid(self, tiny_model):
        from linkedin_lead_scoring.monitoring.onnx_optimizer import convert_xgboost_to_onnx
        import onnx

        onnx_model = convert_xgboost_to_onnx(tiny_model, n_features=6)
        # onnx.checker.check_model raises if invalid
        onnx.checker.check_model(onnx_model)

    def test_save_to_file(self, tiny_model):
        from linkedin_lead_scoring.monitoring.onnx_optimizer import convert_xgboost_to_onnx, save_onnx_model

        onnx_model = convert_xgboost_to_onnx(tiny_model, n_features=6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            returned = save_onnx_model(onnx_model, path)
            assert os.path.exists(path)
            assert returned == path
            assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# Tests: OnnxInferenceSession
# ---------------------------------------------------------------------------

class TestOnnxInferenceSession:
    def test_session_loads_from_file(self, tiny_model, sample_X):
        from linkedin_lead_scoring.monitoring.onnx_optimizer import (
            convert_xgboost_to_onnx, save_onnx_model, OnnxInferenceSession,
        )

        onnx_model = convert_xgboost_to_onnx(tiny_model, n_features=6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            save_onnx_model(onnx_model, path)
            session = OnnxInferenceSession(path)
            assert session is not None

    def test_predict_proba_shape(self, tiny_model, sample_X):
        from linkedin_lead_scoring.monitoring.onnx_optimizer import (
            convert_xgboost_to_onnx, save_onnx_model, OnnxInferenceSession,
        )

        onnx_model = convert_xgboost_to_onnx(tiny_model, n_features=6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            save_onnx_model(onnx_model, path)
            session = OnnxInferenceSession(path)
            proba = session.predict_proba(sample_X)
            assert proba.shape == (len(sample_X), 2)

    def test_predictions_match_xgboost(self, tiny_model, sample_X):
        """ONNX Runtime and XGBoost must produce the same class predictions."""
        from linkedin_lead_scoring.monitoring.onnx_optimizer import (
            convert_xgboost_to_onnx, save_onnx_model, OnnxInferenceSession,
        )

        xgb_proba = tiny_model.predict_proba(sample_X)

        onnx_model = convert_xgboost_to_onnx(tiny_model, n_features=6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            save_onnx_model(onnx_model, path)
            session = OnnxInferenceSession(path)
            ort_proba = session.predict_proba(sample_X)

        # Class predictions must match exactly
        xgb_labels = np.argmax(xgb_proba, axis=1)
        ort_labels = np.argmax(ort_proba, axis=1)
        assert np.array_equal(xgb_labels, ort_labels), (
            f"Label mismatch: XGB={xgb_labels[:5]}, ORT={ort_labels[:5]}"
        )


# ---------------------------------------------------------------------------
# Tests: benchmark_comparison
# ---------------------------------------------------------------------------

class TestBenchmarkComparison:
    def test_returns_dict_with_required_keys(self, tiny_model, sample_X):
        from linkedin_lead_scoring.monitoring.onnx_optimizer import (
            convert_xgboost_to_onnx, save_onnx_model, OnnxInferenceSession,
            benchmark_comparison,
        )

        onnx_model = convert_xgboost_to_onnx(tiny_model, n_features=6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            save_onnx_model(onnx_model, path)
            session = OnnxInferenceSession(path)
            result = benchmark_comparison(tiny_model, session, sample_X, n_calls=20)

        assert isinstance(result, dict)
        assert "joblib" in result
        assert "onnx" in result
        for backend in ("joblib", "onnx"):
            for key in ("mean_ms", "p95_ms", "p99_ms", "n_calls"):
                assert key in result[backend], f"Missing {key} in result[{backend!r}]"

    def test_speedup_key_present(self, tiny_model, sample_X):
        from linkedin_lead_scoring.monitoring.onnx_optimizer import (
            convert_xgboost_to_onnx, save_onnx_model, OnnxInferenceSession,
            benchmark_comparison,
        )

        onnx_model = convert_xgboost_to_onnx(tiny_model, n_features=6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            save_onnx_model(onnx_model, path)
            session = OnnxInferenceSession(path)
            result = benchmark_comparison(tiny_model, session, sample_X, n_calls=20)

        assert "speedup_mean" in result
        assert isinstance(result["speedup_mean"], float)

    def test_times_are_positive(self, tiny_model, sample_X):
        from linkedin_lead_scoring.monitoring.onnx_optimizer import (
            convert_xgboost_to_onnx, save_onnx_model, OnnxInferenceSession,
            benchmark_comparison,
        )

        onnx_model = convert_xgboost_to_onnx(tiny_model, n_features=6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            save_onnx_model(onnx_model, path)
            session = OnnxInferenceSession(path)
            result = benchmark_comparison(tiny_model, session, sample_X, n_calls=20)

        assert result["joblib"]["mean_ms"] > 0
        assert result["onnx"]["mean_ms"] > 0


# ---------------------------------------------------------------------------
# Tests: measure_memory_mb
# ---------------------------------------------------------------------------

class TestMeasureMemoryMb:
    def test_returns_positive_float(self, tiny_model, sample_X):
        from linkedin_lead_scoring.monitoring.onnx_optimizer import measure_memory_mb

        mem = measure_memory_mb(tiny_model, sample_X)
        assert isinstance(mem, float)
        assert mem > 0


# ---------------------------------------------------------------------------
# Tests: save_benchmark_results
# ---------------------------------------------------------------------------

class TestSaveBenchmarkResults:
    def test_creates_json_file(self):
        from linkedin_lead_scoring.monitoring.onnx_optimizer import save_benchmark_results

        data = {"joblib": {"mean_ms": 1.2}, "onnx": {"mean_ms": 0.8}, "speedup_mean": 1.5}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "benchmark.json")
            returned = save_benchmark_results(data, path)
            assert os.path.exists(path)
            assert returned == path
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["speedup_mean"] == 1.5
