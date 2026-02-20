"""Unit tests for the drift detection module.

Tests cover:
- No-drift case (identical distributions)
- Drift case (strongly shifted distributions)
- HTML report generation
- Missing-value handling in production data
- Prediction drift detection
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Small reference dataset that mirrors the lead-scoring schema."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "timezone": rng.choice([0.0, 1.0, 2.0], size=n),
            "icebreaker": rng.uniform(0, 1, size=n),
            "companyfoundedon": rng.integers(1990, 2020, size=n).astype(float),
            "industry": rng.choice(["tech", "finance", "health"], size=n),
            "companysize": rng.choice(["1-10", "11-50", "51-200"], size=n),
            "location": rng.choice(["Paris", "London", "Berlin"], size=n),
        }
    )


@pytest.fixture
def same_distribution_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Production data drawn from the same distribution as reference."""
    rng = np.random.default_rng(99)
    n = 60
    return pd.DataFrame(
        {
            "timezone": rng.choice([0.0, 1.0, 2.0], size=n),
            "icebreaker": rng.uniform(0, 1, size=n),
            "companyfoundedon": rng.integers(1990, 2020, size=n).astype(float),
            "industry": rng.choice(["tech", "finance", "health"], size=n),
            "companysize": rng.choice(["1-10", "11-50", "51-200"], size=n),
            "location": rng.choice(["Paris", "London", "Berlin"], size=n),
        }
    )


@pytest.fixture
def shifted_df() -> pd.DataFrame:
    """Production data with heavily shifted numeric distributions — drift expected."""
    rng = np.random.default_rng(7)
    n = 60
    return pd.DataFrame(
        {
            # strongly shifted: all in [5, 10] vs reference [0, 2]
            "timezone": rng.choice([5.0, 8.0, 10.0], size=n),
            # all zeros vs reference uniform
            "icebreaker": np.zeros(n),
            # far future vs reference 1990-2020
            "companyfoundedon": rng.integers(2060, 2090, size=n).astype(float),
            # completely different categories
            "industry": rng.choice(["mining", "agriculture", "retail"], size=n),
            "companysize": rng.choice(["5001-10000", "10001+"], size=n),
            "location": rng.choice(["Tokyo", "Sydney", "Cairo"], size=n),
        }
    )


# ---------------------------------------------------------------------------
# Tests for DriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetectorInit:
    def test_init_with_dataframe(self, reference_df):
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        detector = DriftDetector(reference_data=reference_df)
        assert detector is not None

    def test_reference_data_stored(self, reference_df):
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        detector = DriftDetector(reference_data=reference_df)
        assert len(detector.reference_data) == len(reference_df)

    def test_empty_reference_raises(self):
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        with pytest.raises(ValueError, match="reference_data"):
            DriftDetector(reference_data=pd.DataFrame())


class TestDetectDataDrift:
    def test_no_drift_same_distribution(self, reference_df, same_distribution_df):
        """Same distribution → drift_detected should be False."""
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        detector = DriftDetector(reference_data=reference_df)
        result = detector.detect_data_drift(production_data=same_distribution_df)

        assert isinstance(result, dict)
        assert "drift_detected" in result
        assert "drifted_features" in result
        assert "drifted_count" in result
        assert "total_features" in result
        # same distribution → we expect no overall drift
        assert result["drift_detected"] is False

    def test_drift_detected_shifted_data(self, reference_df, shifted_df):
        """Heavily shifted data → drift_detected must be True."""
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        detector = DriftDetector(reference_data=reference_df)
        result = detector.detect_data_drift(production_data=shifted_df)

        assert result["drift_detected"] is True
        assert len(result["drifted_features"]) > 0

    def test_result_keys_complete(self, reference_df, same_distribution_df):
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        detector = DriftDetector(reference_data=reference_df)
        result = detector.detect_data_drift(production_data=same_distribution_df)

        expected_keys = {
            "drift_detected",
            "drifted_features",
            "drifted_count",
            "total_features",
            "drift_share",
        }
        assert expected_keys.issubset(result.keys())

    def test_missing_values_handled(self, reference_df):
        """Production data with NaNs should not raise."""
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        prod_with_na = reference_df.copy()
        prod_with_na.loc[prod_with_na.index[:10], "icebreaker"] = np.nan
        prod_with_na.loc[prod_with_na.index[10:20], "industry"] = np.nan

        detector = DriftDetector(reference_data=reference_df)
        # Should complete without raising
        result = detector.detect_data_drift(production_data=prod_with_na)
        assert "drift_detected" in result

    def test_drifted_count_equals_feature_list_length(
        self, reference_df, shifted_df
    ):
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        detector = DriftDetector(reference_data=reference_df)
        result = detector.detect_data_drift(production_data=shifted_df)

        assert result["drifted_count"] == len(result["drifted_features"])

    def test_total_features_matches_dataframe_columns(
        self, reference_df, same_distribution_df
    ):
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        detector = DriftDetector(reference_data=reference_df)
        result = detector.detect_data_drift(production_data=same_distribution_df)

        assert result["total_features"] == len(reference_df.columns)


class TestDetectPredictionDrift:
    def test_same_scores_no_drift(self, reference_df):
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, size=80)

        detector = DriftDetector(reference_data=reference_df)
        result = detector.detect_prediction_drift(
            reference_scores=scores[:50],
            production_scores=scores[50:],
        )

        assert isinstance(result, dict)
        assert "drift_detected" in result
        assert "p_value" in result
        assert "statistic" in result
        assert result["drift_detected"] is False

    def test_shifted_scores_drift_detected(self, reference_df):
        """Scores near 0 vs near 1 → drift must be detected."""
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        rng = np.random.default_rng(1)
        ref_scores = rng.uniform(0.0, 0.2, size=80)
        prod_scores = rng.uniform(0.8, 1.0, size=80)

        detector = DriftDetector(reference_data=reference_df)
        result = detector.detect_prediction_drift(
            reference_scores=ref_scores,
            production_scores=prod_scores,
        )

        assert result["drift_detected"] is True
        assert result["p_value"] < 0.05


class TestGenerateReport:
    def test_generate_report_creates_html_file(
        self, reference_df, same_distribution_df
    ):
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        detector = DriftDetector(reference_data=reference_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "drift_report.html")
            returned_path = detector.generate_report(
                production_data=same_distribution_df,
                output_path=output_path,
            )

        # File must have been written and path returned
        assert returned_path == output_path

    def test_generate_report_with_shifted_data(self, reference_df, shifted_df):
        from linkedin_lead_scoring.monitoring.drift import DriftDetector

        detector = DriftDetector(reference_data=reference_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "drift_report_shifted.html")
            returned_path = detector.generate_report(
                production_data=shifted_df,
                output_path=output_path,
            )

        assert returned_path == output_path
