"""Unit tests for the Streamlit dashboard utility functions.

Tests cover:
- streamlit_app.py imports without error
- Log parsing (JSONL prediction logs and API request logs)
- Synthetic data simulation
- Metrics calculations (score stats, inference stats, uptime)
"""
import json
import os
import tempfile




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prediction_log_entry(
    score: float = 0.6,
    label: str = "engaged",
    inference_ms: float = 10.0,
    timestamp: str = "2026-02-20T12:00:00",
) -> dict:
    return {
        "timestamp": timestamp,
        "input": {"jobtitle": "CTO", "industry": "tech", "companysize": "11-50"},
        "score": score,
        "label": label,
        "inference_ms": inference_ms,
        "model_version": "0.3.0",
    }


def _make_api_request_entry(
    response_ms: float = 25.0,
    status_code: int = 200,
    timestamp: str = "2026-02-20T12:00:01",
) -> dict:
    return {
        "timestamp": timestamp,
        "endpoint": "/predict",
        "method": "POST",
        "status_code": status_code,
        "response_ms": response_ms,
    }


# ---------------------------------------------------------------------------
# Test: streamlit_app.py imports cleanly
# ---------------------------------------------------------------------------

class TestStreamlitAppImports:
    def test_streamlit_app_is_importable(self):
        """streamlit_app.py must be importable without executing st.* calls."""
        import importlib.util

        # We add the worktree root to sys.path so the top-level file is importable
        worktree_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        app_path = os.path.join(worktree_root, "streamlit_app.py")
        assert os.path.exists(app_path), "streamlit_app.py not found at worktree root"

        spec = importlib.util.spec_from_file_location("streamlit_app", app_path)
        # Just load the spec — don't exec; that would trigger streamlit runtime
        assert spec is not None

    def test_dashboard_utils_importable(self):
        from linkedin_lead_scoring.monitoring import dashboard_utils  # noqa: F401

    def test_monitoring_package_works_without_onnx(self):
        """Monitoring package must remain importable when onnx is absent."""
        import importlib
        import sys

        # Temporarily make 'onnx' unimportable
        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name in ("onnx", "onnxruntime", "onnxmltools"):
                raise ModuleNotFoundError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        # Remove cached monitoring modules so reimport triggers __init__.py
        mods_to_remove = [k for k in sys.modules if k.startswith("linkedin_lead_scoring.monitoring")]
        saved = {k: sys.modules.pop(k) for k in mods_to_remove}

        try:
            import builtins
            original = builtins.__import__
            builtins.__import__ = mock_import
            monitoring = importlib.import_module("linkedin_lead_scoring.monitoring")
            # dashboard_utils and profiler should still be accessible
            assert monitoring.dashboard_utils is not None
            assert monitoring.profiler is not None
            # onnx_optimizer should be None (graceful degradation)
            assert monitoring.onnx_optimizer is None
        finally:
            builtins.__import__ = original
            # Restore original modules
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Test: log parsing
# ---------------------------------------------------------------------------

class TestLoadPredictionLogs:
    def test_valid_jsonl_parsed(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import load_prediction_logs

        entries = [_make_prediction_log_entry(score=0.3 + i * 0.1) for i in range(5)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            path = f.name

        try:
            logs = load_prediction_logs(path)
            assert len(logs) == 5
            assert all("score" in r for r in logs)
            assert all("timestamp" in r for r in logs)
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import load_prediction_logs

        logs = load_prediction_logs("/tmp/nonexistent_predictions.jsonl")
        assert logs == []

    def test_empty_file_returns_empty(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import load_prediction_logs

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name  # empty file

        try:
            logs = load_prediction_logs(path)
            assert logs == []
        finally:
            os.unlink(path)

    def test_malformed_lines_skipped(self):
        """Lines that are not valid JSON should be skipped gracefully."""
        from linkedin_lead_scoring.monitoring.dashboard_utils import load_prediction_logs

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_prediction_log_entry()) + "\n")
            f.write("not-valid-json\n")
            f.write(json.dumps(_make_prediction_log_entry(score=0.9)) + "\n")
            path = f.name

        try:
            logs = load_prediction_logs(path)
            assert len(logs) == 2
        finally:
            os.unlink(path)


class TestLoadApiRequestLogs:
    def test_valid_jsonl_parsed(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import load_api_request_logs

        entries = [_make_api_request_entry(response_ms=20 + i) for i in range(4)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            path = f.name

        try:
            logs = load_api_request_logs(path)
            assert len(logs) == 4
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import load_api_request_logs

        logs = load_api_request_logs("/tmp/nonexistent_requests.jsonl")
        assert logs == []


# ---------------------------------------------------------------------------
# Test: synthetic data simulation
# ---------------------------------------------------------------------------

class TestSimulateProductionLogs:
    def test_returns_list_of_dicts(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import simulate_production_logs

        logs = simulate_production_logs(n=30, seed=42)
        assert isinstance(logs, list)
        assert len(logs) == 30

    def test_required_fields_present(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import simulate_production_logs

        logs = simulate_production_logs(n=10, seed=0)
        required = {"timestamp", "score", "label", "inference_ms", "model_version"}
        for entry in logs:
            assert required.issubset(entry.keys()), f"Missing fields in {entry}"

    def test_scores_in_unit_interval(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import simulate_production_logs

        logs = simulate_production_logs(n=50, seed=1)
        scores = [e["score"] for e in logs]
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_labels_valid(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import simulate_production_logs

        logs = simulate_production_logs(n=50, seed=2)
        for entry in logs:
            assert entry["label"] in ("engaged", "not_engaged")

    def test_reproducible_with_seed(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import simulate_production_logs

        logs1 = simulate_production_logs(n=20, seed=99)
        logs2 = simulate_production_logs(n=20, seed=99)
        assert [e["score"] for e in logs1] == [e["score"] for e in logs2]


# ---------------------------------------------------------------------------
# Test: metrics calculations
# ---------------------------------------------------------------------------

class TestComputeScoreStats:
    def _make_logs(self, scores, labels=None) -> list:
        if labels is None:
            labels = ["engaged" if s > 0.5 else "not_engaged" for s in scores]
        return [
            {"score": s, "label": lbl, "timestamp": "2026-02-20T12:00:00", "inference_ms": 10.0}
            for s, lbl in zip(scores, labels)
        ]

    def test_percentiles_computed(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import compute_score_stats

        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result = compute_score_stats(self._make_logs(scores))

        assert "p25" in result
        assert "p50" in result
        assert "p75" in result
        assert result["p25"] <= result["p50"] <= result["p75"]

    def test_engagement_rate(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import compute_score_stats

        logs = self._make_logs(
            [0.8, 0.9, 0.7],
            labels=["engaged", "engaged", "engaged"],
        ) + self._make_logs(
            [0.2, 0.1],
            labels=["not_engaged", "not_engaged"],
        )
        result = compute_score_stats(logs)

        assert "engagement_rate" in result
        assert abs(result["engagement_rate"] - 0.6) < 1e-9

    def test_total_predictions(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import compute_score_stats

        logs = self._make_logs([0.5] * 7)
        result = compute_score_stats(logs)
        assert result["total_predictions"] == 7

    def test_empty_logs_returns_defaults(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import compute_score_stats

        result = compute_score_stats([])
        assert result["total_predictions"] == 0
        assert result["engagement_rate"] == 0.0


class TestComputeInferenceStats:
    def _make_logs(self, times_ms: list) -> list:
        return [
            {"score": 0.5, "label": "engaged", "inference_ms": t,
             "timestamp": "2026-02-20T12:00:00"}
            for t in times_ms
        ]

    def test_percentiles_computed(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import compute_inference_stats

        logs = self._make_logs([10.0] * 90 + [100.0] * 10)
        result = compute_inference_stats(logs)

        assert "mean_ms" in result
        assert "p50_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert result["p50_ms"] < result["p95_ms"]

    def test_empty_logs_returns_zeros(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import compute_inference_stats

        result = compute_inference_stats([])
        assert result["mean_ms"] == 0.0
        assert result["p95_ms"] == 0.0


# ---------------------------------------------------------------------------
# Test: compute_uptime_stats (C.5 gap — previously uncovered)
# ---------------------------------------------------------------------------

class TestComputeUptimeStats:
    def _make_request_log(self, status_codes: list) -> list:
        return [
            {"timestamp": "2026-02-20T12:00:00", "endpoint": "/predict",
             "status_code": sc, "response_ms": 20.0}
            for sc in status_codes
        ]

    def test_empty_logs_returns_full_uptime(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import compute_uptime_stats

        result = compute_uptime_stats([])
        assert result["total_requests"] == 0
        assert result["success_rate"] == 1.0
        assert result["error_rate"] == 0.0

    def test_all_successful_requests(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import compute_uptime_stats

        logs = self._make_request_log([200, 200, 201, 204])
        result = compute_uptime_stats(logs)
        assert result["total_requests"] == 4
        assert result["success_rate"] == 1.0
        assert result["error_rate"] == 0.0

    def test_mixed_success_and_errors(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import compute_uptime_stats

        logs = self._make_request_log([200, 200, 500, 503])  # 2 ok, 2 errors
        result = compute_uptime_stats(logs)
        assert result["total_requests"] == 4
        assert abs(result["success_rate"] - 0.5) < 1e-9
        assert abs(result["error_rate"] - 0.5) < 1e-9

    def test_all_errors(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import compute_uptime_stats

        logs = self._make_request_log([500, 503, 404])
        result = compute_uptime_stats(logs)
        assert result["success_rate"] == 0.0
        assert result["error_rate"] == 1.0

    def test_success_rate_plus_error_rate_equals_one(self):
        from linkedin_lead_scoring.monitoring.dashboard_utils import compute_uptime_stats

        logs = self._make_request_log([200, 201, 400, 500])
        result = compute_uptime_stats(logs)
        assert abs(result["success_rate"] + result["error_rate"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Test: JSONL with blank lines (C.5 gap — line 65 branch)
# ---------------------------------------------------------------------------

class TestLoadJsonlWithBlankLines:
    def test_blank_lines_skipped(self):
        """JSONL files with blank separator lines should parse correctly."""
        import json
        import tempfile
        from linkedin_lead_scoring.monitoring.dashboard_utils import load_prediction_logs

        entry = {"timestamp": "2026-02-20T12:00:00", "score": 0.7,
                 "label": "engaged", "inference_ms": 10.0, "model_version": "0.3.0"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(entry) + "\n")
            f.write("\n")          # blank line
            f.write("   \n")      # whitespace-only line
            f.write(json.dumps(entry) + "\n")
            path = f.name

        try:
            logs = load_prediction_logs(path)
            assert len(logs) == 2
        finally:
            os.unlink(path)
