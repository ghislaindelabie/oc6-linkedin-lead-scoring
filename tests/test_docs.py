"""Structural tests for required documentation files.

Verifies that docs/PERFORMANCE_REPORT.md and docs/MONITORING_GUIDE.md
exist and contain the required sections defined in SESSION_C_TASKS.md.
"""
import os
import pathlib

import pytest

WORKTREE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS_DIR = WORKTREE_ROOT / "docs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_doc(filename: str) -> str:
    path = DOCS_DIR / filename
    assert path.exists(), f"Missing documentation file: docs/{filename}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests: docs/PERFORMANCE_REPORT.md
# ---------------------------------------------------------------------------

class TestPerformanceReport:
    FILENAME = "PERFORMANCE_REPORT.md"

    def test_file_exists(self):
        assert (DOCS_DIR / self.FILENAME).exists()

    def test_has_baseline_section(self):
        content = _read_doc(self.FILENAME)
        assert "Baseline" in content or "baseline" in content

    def test_has_onnx_section(self):
        content = _read_doc(self.FILENAME)
        assert "ONNX" in content

    def test_has_speedup_or_improvement(self):
        content = _read_doc(self.FILENAME)
        assert any(kw in content for kw in ("speedup", "Speedup", "improvement", "Improvement", "faster"))

    def test_has_cprofile_or_bottleneck_section(self):
        content = _read_doc(self.FILENAME)
        assert any(kw in content for kw in ("cProfile", "cprofile", "bottleneck", "Bottleneck", "profile"))

    def test_has_recommendations_section(self):
        content = _read_doc(self.FILENAME)
        assert any(kw in content for kw in ("Recommendation", "recommendation", "Production"))

    def test_has_numeric_latency_values(self):
        """Report must include at least one numeric latency measurement (e.g. '0.21 ms')."""
        import re
        content = _read_doc(self.FILENAME)
        # Match patterns like "0.21 ms", "0.008ms", "26.5x"
        assert re.search(r"\d+\.\d+\s*(ms|x)", content), (
            "PERFORMANCE_REPORT.md should contain numeric latency or speedup values"
        )


# ---------------------------------------------------------------------------
# Tests: docs/MONITORING_GUIDE.md
# ---------------------------------------------------------------------------

class TestMonitoringGuide:
    FILENAME = "MONITORING_GUIDE.md"

    def test_file_exists(self):
        assert (DOCS_DIR / self.FILENAME).exists()

    def test_has_dashboard_access_section(self):
        content = _read_doc(self.FILENAME)
        assert any(kw in content for kw in ("dashboard", "Dashboard", "Streamlit", "streamlit"))

    def test_has_metrics_interpretation_section(self):
        content = _read_doc(self.FILENAME)
        assert any(kw in content for kw in ("metric", "Metric", "interpret", "Interpret"))

    def test_has_drift_section(self):
        content = _read_doc(self.FILENAME)
        assert any(kw in content for kw in ("drift", "Drift"))

    def test_has_retrain_section(self):
        content = _read_doc(self.FILENAME)
        assert any(kw in content for kw in ("retrain", "Retrain", "retraining", "model refresh"))

    def test_has_alert_thresholds(self):
        content = _read_doc(self.FILENAME)
        assert any(kw in content for kw in ("threshold", "Threshold", "alert", "Alert"))

    def test_minimum_length(self):
        """Guide should be substantive (at least 300 chars)."""
        content = _read_doc(self.FILENAME)
        assert len(content) >= 300, "MONITORING_GUIDE.md seems too short"
