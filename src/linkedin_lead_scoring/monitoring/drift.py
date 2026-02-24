"""Drift detection using Evidently AI (v0.7+).

The DriftDetector class wraps Evidently's Report API to provide:
  - Data drift analysis (per-feature, KS/chi-squared tests)
  - Prediction score drift (KS test via scipy)
  - Full HTML report generation

Evidently 0.7 API notes:
  - `Report.run()` returns a `Snapshot` (not modifying the report in place).
  - `Snapshot.save_html(path)` writes the interactive HTML report.
  - `Snapshot.metric_results` is a dict of metric_id → result objects.
  - `DriftedColumnsCount` result: `.count.value` (int) and `.share.value` (float).
  - `ValueDrift` result per column: `.value` is the drift score (p-value for KS).
"""
import os
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import DataDriftPreset
from scipy import stats


class DriftDetector:
    """Detect distribution drift between reference and production data.

    Parameters
    ----------
    reference_data:
        Training/baseline DataFrame. Used as the reference distribution for
        all subsequent drift comparisons.

    Raises
    ------
    ValueError
        If `reference_data` is empty.
    """

    def __init__(self, reference_data: pd.DataFrame) -> None:
        if reference_data.empty:
            raise ValueError(
                "reference_data must not be empty; provide the training DataFrame."
            )
        self.reference_data: pd.DataFrame = reference_data.copy()
        # Feature columns (exclude target 'engaged' if present)
        self._feature_columns: List[str] = [
            c for c in reference_data.columns if c != "engaged"
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_data_drift(self, production_data: pd.DataFrame) -> dict:
        """Run Evidently data drift analysis on feature columns.

        Uses `DataDriftPreset` which applies KS test for numeric features and
        chi-squared / Cramer's V for categorical features.  Overall drift is
        flagged when ≥ 50 % of features drift (Evidently default threshold).

        Parameters
        ----------
        production_data:
            New data to compare against the reference distribution.

        Returns
        -------
        dict with keys:
            drift_detected (bool)  — True if overall drift triggered
            drifted_features (list[str]) — names of individually drifted columns
            drifted_count (int)    — number of drifted columns
            total_features (int)   — total number of feature columns analysed
            drift_share (float)    — fraction of drifted columns
        """
        ref = self.reference_data[self._feature_columns]
        prod = self._align_columns(production_data)

        # Build per-column ValueDrift metrics so we can list drifted columns
        per_col_metrics = [ValueDrift(column=col) for col in self._feature_columns]
        count_metric = DriftedColumnsCount()

        report = Report(metrics=[count_metric, *per_col_metrics])
        snapshot = report.run(current_data=prod, reference_data=ref)

        results = snapshot.metric_results
        drifted_features = self._extract_drifted_columns(results, per_col_metrics)

        # DriftedColumnsCount result carries .count.value and .share.value
        count_result = self._get_metric_result(results, count_metric)
        drifted_count = int(count_result.count.value) if count_result else len(drifted_features)
        drift_share = float(count_result.share.value) if count_result else (
            drifted_count / len(self._feature_columns) if self._feature_columns else 0.0
        )
        drift_detected = drift_share >= 0.5

        return {
            "drift_detected": drift_detected,
            "drifted_features": drifted_features,
            "drifted_count": drifted_count,
            "total_features": len(self._feature_columns),
            "drift_share": drift_share,
        }

    def detect_prediction_drift(
        self,
        reference_scores: Union[Sequence[float], np.ndarray],
        production_scores: Union[Sequence[float], np.ndarray],
        threshold: float = 0.05,
    ) -> dict:
        """Compare prediction score distributions with a two-sample KS test.

        Parameters
        ----------
        reference_scores:
            Array of predicted probabilities from the reference period.
        production_scores:
            Array of predicted probabilities from the production period.
        threshold:
            P-value threshold; scores are considered drifted when p < threshold.

        Returns
        -------
        dict with keys:
            drift_detected (bool)
            statistic (float)  — KS test statistic
            p_value (float)
            method (str)       — always "ks_2samp"
        """
        ref_arr = np.asarray(reference_scores, dtype=float)
        prod_arr = np.asarray(production_scores, dtype=float)

        ks_result = stats.ks_2samp(ref_arr, prod_arr)

        return {
            "drift_detected": bool(ks_result.pvalue < threshold),
            "statistic": float(ks_result.statistic),
            "p_value": float(ks_result.pvalue),
            "method": "ks_2samp",
        }

    def generate_report(
        self,
        production_data: pd.DataFrame,
        output_path: Union[str, Path],
    ) -> str:
        """Generate a full HTML Evidently drift report.

        Uses `DataDriftPreset` for a comprehensive view with distribution plots
        and statistical test results for every feature.

        Parameters
        ----------
        production_data:
            Current/production data to compare against the reference.
        output_path:
            File path where the HTML report will be written.

        Returns
        -------
        str
            Absolute path to the generated HTML file.
        """
        output_path = str(output_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        ref = self.reference_data[self._feature_columns]
        prod = self._align_columns(production_data)

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(current_data=prod, reference_data=ref)
        snapshot.save_html(output_path)

        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _align_columns(self, production_data: pd.DataFrame) -> pd.DataFrame:
        """Return production_data restricted to feature columns, in the same order."""
        available = [c for c in self._feature_columns if c in production_data.columns]
        return production_data[available]

    def _get_metric_result(self, results: dict, metric):
        """Return the result object for a given metric instance."""
        metric_id = metric.get_metric_id()
        return results.get(metric_id)

    def _extract_drifted_columns(self, results: dict, per_col_metrics: list) -> List[str]:
        """Return list of column names where drift was detected."""
        drifted = []
        for metric in per_col_metrics:
            result = self._get_metric_result(results, metric)
            if result is None:
                continue
            # ValueDrift result: .value is the drift score (KS p-value).
            # Drift is detected when the score is BELOW the threshold (0.05 default).
            # Evidently stores tests on the result; we check the widget counter text
            # to avoid coupling to internal implementation details.
            # Simpler approach: check if the widget label contains "Drift detected"
            # or use the test results when present.
            drifted_col = self._is_column_drifted(result, metric)
            if drifted_col:
                drifted.append(metric.column)
        return drifted

    def _is_column_drifted(self, result, metric) -> bool:
        """Determine if a single column's drift result indicates drift.

        Uses Evidently's structured result API: ``result.value`` is the
        statistical test score (KS p-value for numeric, chi-squared score for
        categorical).  Drift is detected when the score is below the threshold
        stored in the result's metric location (default 0.05).

        This replaces the earlier widget-text approach, which was coupled to
        Evidently's internal UI rendering and would silently break on upgrades.
        """
        if result is None:
            return False
        if not hasattr(result, "value"):
            return False
        try:
            threshold = result.metric_value_location.metric.params.get(
                "threshold", 0.05
            )
        except Exception:
            threshold = 0.05
        return float(result.value) < threshold
