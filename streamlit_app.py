"""Streamlit monitoring dashboard for the LinkedIn Lead Scoring API.

Sections
--------
1. API Health Overview    — live /health call + uptime from logs
2. Score Distribution     — reference vs production histograms + percentiles
3. Performance Metrics    — inference time distribution and trend
4. Data Drift Analysis    — Evidently drift status, per-feature indicators
5. Recent Predictions     — last 20 entries, filterable by score range

Data sources (in priority order)
---------------------------------
- logs/predictions.jsonl  — production prediction logs (Session B)
- logs/api_requests.jsonl — API request logs
- data/reference/training_reference.csv — reference data for drift
- Simulation               — generated automatically when files are absent

Run
---
    streamlit run streamlit_app.py
"""
import os
import tempfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from linkedin_lead_scoring.monitoring.dashboard_utils import (
    compute_inference_stats,
    compute_score_stats,
    compute_uptime_stats,
    load_api_request_logs,
    load_prediction_logs,
    simulate_production_logs,
)
from linkedin_lead_scoring.monitoring.drift import DriftDetector

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Lead Scoring Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREDICTIONS_LOG = os.getenv("PREDICTIONS_LOG", "logs/predictions.jsonl")
API_REQUESTS_LOG = os.getenv("API_REQUESTS_LOG", "logs/api_requests.jsonl")
REFERENCE_DATA_PATH = os.getenv(
    "REFERENCE_DATA_PATH", "data/reference/training_reference.csv"
)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
REPORTS_DIR = Path("reports")

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def get_prediction_logs() -> list[dict]:
    logs = load_prediction_logs(PREDICTIONS_LOG)
    if not logs:
        logs = simulate_production_logs(n=150, seed=42)
    return logs


@st.cache_data(ttl=60)
def get_api_request_logs() -> list[dict]:
    logs = load_api_request_logs(API_REQUESTS_LOG)
    if not logs:
        # Simulate simple request log from prediction logs
        pred_logs = get_prediction_logs()
        logs = [
            {
                "timestamp": e["timestamp"],
                "endpoint": "/predict",
                "method": "POST",
                "status_code": 200,
                "response_ms": e["inference_ms"] + np.random.default_rng(42).uniform(2, 8),
            }
            for e in pred_logs
        ]
    return logs


@st.cache_data(ttl=300)
def get_reference_data() -> pd.DataFrame:
    if os.path.exists(REFERENCE_DATA_PATH):
        return pd.read_csv(REFERENCE_DATA_PATH)
    # Fall back to the mlruns artifact (development environment)
    import glob

    candidates = glob.glob(
        "mlruns/**/processed_data/linkedin_leads_clean.csv", recursive=True
    )
    if candidates:
        return pd.read_csv(candidates[0])
    # Last resort: synthesise a minimal reference from simulation
    from linkedin_lead_scoring.monitoring.dashboard_utils import simulate_production_logs

    ref_logs = simulate_production_logs(n=300, seed=0)
    return pd.DataFrame([e["input"] for e in ref_logs])


# ---------------------------------------------------------------------------
# Drift report generation (cached — Evidently reports are expensive)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _generate_drift_report(
    detector: DriftDetector, prod_df: "pd.DataFrame", output_path: str
) -> None:
    """Generate and write the Evidently HTML drift report (cached 5 min)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    detector.generate_report(prod_df, output_path=output_path)


# ---------------------------------------------------------------------------
# API health check
# ---------------------------------------------------------------------------


def check_api_health() -> dict:
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        return {"status": resp.json().get("status", "unknown"), "reachable": True}
    except Exception:
        return {"status": "unreachable", "reachable": False}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 Lead Scoring Monitor")
    st.caption("OC6/OC8 MLOps project — v0.3.0")
    st.divider()
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    score_threshold = st.slider(
        "Engagement threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )

# ---------------------------------------------------------------------------
# Section 1: API Health Overview
# ---------------------------------------------------------------------------

st.header("1. API Health Overview")

pred_logs = get_prediction_logs()
req_logs = get_api_request_logs()
health = check_api_health()
score_stats = compute_score_stats(pred_logs)
uptime = compute_uptime_stats(req_logs)

col1, col2, col3, col4 = st.columns(4)
with col1:
    status_colour = "🟢" if health["reachable"] else "🔴"
    st.metric("API Status", f"{status_colour} {health['status']}")
with col2:
    st.metric(
        "Uptime",
        f"{uptime['success_rate'] * 100:.1f}%",
        delta=None,
    )
with col3:
    st.metric("Total Predictions", f"{score_stats['total_predictions']:,}")
with col4:
    st.metric(
        "Engagement Rate",
        f"{score_stats['engagement_rate'] * 100:.1f}%",
    )

# ---------------------------------------------------------------------------
# Section 2: Score Distribution
# ---------------------------------------------------------------------------

st.divider()
st.header("2. Score Distribution")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Production scores")
    scores = [e["score"] for e in pred_logs]
    df_scores = pd.DataFrame({"score": scores})
    st.bar_chart(df_scores["score"].value_counts(bins=20, sort=False))

    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("p25", f"{score_stats['p25']:.3f}")
    p_col2.metric("p50", f"{score_stats['p50']:.3f}")
    p_col3.metric("p75", f"{score_stats['p75']:.3f}")

with col_b:
    st.subheader("Label breakdown")
    label_counts = pd.Series([e["label"] for e in pred_logs]).value_counts()
    st.bar_chart(label_counts)

# ---------------------------------------------------------------------------
# Section 3: Performance Metrics
# ---------------------------------------------------------------------------

st.divider()
st.header("3. Performance Metrics")

infer_stats = compute_inference_stats(pred_logs)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean latency", f"{infer_stats['mean_ms']:.1f} ms")
m2.metric("p50 latency", f"{infer_stats['p50_ms']:.1f} ms")
m3.metric("p95 latency", f"{infer_stats['p95_ms']:.1f} ms")
m4.metric("p99 latency", f"{infer_stats['p99_ms']:.1f} ms")

# Inference time trend (indexed by position)
df_inf = pd.DataFrame({"inference_ms": [e["inference_ms"] for e in pred_logs]})
st.line_chart(df_inf["inference_ms"], use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Data Drift Analysis
# ---------------------------------------------------------------------------

st.divider()
st.header("4. Data Drift Analysis")

ref_df = get_reference_data()
feature_cols = [c for c in ref_df.columns if c != "engaged"]

prod_df = pd.DataFrame(
    [e.get("input", {}) for e in pred_logs if "input" in e]
)

# Only run drift if prod_df has at least some matching columns
common_cols = [c for c in feature_cols if c in prod_df.columns]

if len(common_cols) < 2:
    st.info(
        "Production data does not contain enough matching feature columns for "
        "drift analysis. Run the API with full feature input to enable drift tracking."
    )
else:
    with st.spinner("Running drift analysis…"):
        detector = DriftDetector(reference_data=ref_df[common_cols])
        drift_result = detector.detect_data_drift(prod_df[common_cols])

    overall = drift_result["drift_detected"]
    colour = "🔴 Drift detected" if overall else "🟢 No drift"
    st.subheader(f"Overall: {colour}")

    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric(
        "Drifted features",
        f"{drift_result['drifted_count']} / {drift_result['total_features']}",
    )
    d_col2.metric("Drift share", f"{drift_result['drift_share'] * 100:.0f}%")
    d_col3.metric(
        "Drifted columns",
        ", ".join(drift_result["drifted_features"]) or "none",
    )

    # Embedded Evidently HTML report — generated on demand, cached for 5 min
    with st.expander("Full Evidently drift report", expanded=False):
        if st.button("Generate / refresh drift report"):
            st.cache_data.clear()
        report_path = str(REPORTS_DIR / "drift_report.html")
        _generate_drift_report(detector, prod_df[common_cols], report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600, scrolling=True)

# ---------------------------------------------------------------------------
# Section 5: Recent Predictions
# ---------------------------------------------------------------------------

st.divider()
st.header("5. Recent Predictions")

min_score, max_score = st.slider(
    "Filter by score range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.05,
)

recent = [
    e
    for e in pred_logs[-20:]
    if min_score <= e.get("score", 0) <= max_score
]

if recent:
    df_recent = pd.DataFrame(recent)[
        ["timestamp", "score", "label", "inference_ms", "model_version"]
    ]
    df_recent = df_recent.sort_values("timestamp", ascending=False)
    st.dataframe(df_recent, use_container_width=True)
else:
    st.info("No predictions match the selected score range.")
