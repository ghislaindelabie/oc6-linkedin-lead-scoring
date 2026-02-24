"""Utility functions for the Streamlit monitoring dashboard.

All data-loading, simulation, and metrics-calculation logic lives here so
it can be unit-tested independently of the Streamlit runtime.

Log file formats
----------------
predictions (logs/predictions.jsonl):
    {"timestamp": "...", "input": {...}, "score": 0.73,
     "label": "engaged", "inference_ms": 12.5, "model_version": "0.3.0"}

api_requests (logs/api_requests.jsonl):
    {"timestamp": "...", "endpoint": "/predict", "method": "POST",
     "status_code": 200, "response_ms": 25.0}
"""
import json
import os
from datetime import datetime, timedelta
import numpy as np


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_prediction_logs(path: str) -> list[dict]:
    """Load prediction logs from a JSONL file.

    Skips lines that are not valid JSON.  Returns an empty list if the file
    does not exist or is empty.

    Parameters
    ----------
    path:
        Path to the JSONL predictions log file.
    """
    return _load_jsonl(path)


def load_api_request_logs(path: str) -> list[dict]:
    """Load API request logs from a JSONL file.

    Skips malformed lines.  Returns an empty list if the file does not exist.

    Parameters
    ----------
    path:
        Path to the JSONL API-request log file.
    """
    return _load_jsonl(path)


def _load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file, skipping malformed lines.  Returns [] on missing file."""
    if not os.path.exists(path):
        return []
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip invalid lines silently
    return records


# ---------------------------------------------------------------------------
# Synthetic data simulation
# ---------------------------------------------------------------------------

def simulate_production_logs(n: int = 100, seed: int = 42) -> list[dict]:
    """Generate synthetic prediction logs for demo / testing purposes.

    The simulated data introduces a slight distribution shift vs the reference
    data so the drift dashboard has something interesting to show.

    Parameters
    ----------
    n:
        Number of log entries to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list of dicts following the predictions JSONL schema.
    """
    rng = np.random.default_rng(seed)
    base_time = datetime(2026, 2, 20, 8, 0, 0)

    scores = rng.beta(a=2.5, b=3.5, size=n)  # slight shift vs training
    inference_times = rng.gamma(shape=2.0, scale=5.0, size=n)  # ms, right-skewed

    logs = []
    for i in range(n):
        score = float(np.clip(scores[i], 0.0, 1.0))
        label = "engaged" if score >= 0.5 else "not_engaged"
        timestamp = (base_time + timedelta(minutes=int(i * 3))).isoformat()
        logs.append(
            {
                "timestamp": timestamp,
                "input": {
                    "jobtitle": rng.choice(["CTO", "VP Engineering", "Director", "CEO"]),
                    "industry": rng.choice(["tech", "finance", "health"]),
                    "companysize": rng.choice(["1-10", "11-50", "51-200"]),
                },
                "score": round(score, 4),
                "label": label,
                "inference_ms": round(float(inference_times[i]), 2),
                "model_version": "0.3.0",
            }
        )
    return logs


# ---------------------------------------------------------------------------
# Metrics calculations
# ---------------------------------------------------------------------------

def compute_score_stats(logs: list[dict]) -> dict:
    """Compute summary statistics on predicted scores from prediction logs.

    Parameters
    ----------
    logs:
        List of prediction log entries (each must have 'score' and 'label').

    Returns
    -------
    dict with keys:
        total_predictions (int)
        engagement_rate (float)  — fraction labelled "engaged"
        p25, p50, p75 (float)   — score percentiles
        mean_score (float)
    """
    if not logs:
        return {
            "total_predictions": 0,
            "engagement_rate": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "mean_score": 0.0,
        }

    scores = np.array([e["score"] for e in logs], dtype=float)
    labels = [e.get("label", "") for e in logs]
    n_engaged = sum(1 for lbl in labels if lbl == "engaged")

    return {
        "total_predictions": len(logs),
        "engagement_rate": n_engaged / len(logs),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "mean_score": float(np.mean(scores)),
    }


def compute_inference_stats(logs: list[dict]) -> dict:
    """Compute inference-time distribution statistics.

    Parameters
    ----------
    logs:
        List of prediction log entries (each must have 'inference_ms').

    Returns
    -------
    dict with keys:
        mean_ms, p50_ms, p95_ms, p99_ms (float)
        min_ms, max_ms (float)
    """
    if not logs:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    times = np.array([e["inference_ms"] for e in logs], dtype=float)
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def compute_uptime_stats(request_logs: list[dict]) -> dict:
    """Compute uptime percentage from API request logs.

    Parameters
    ----------
    request_logs:
        List of API request log entries with 'status_code' field.

    Returns
    -------
    dict with keys:
        total_requests (int)
        success_rate (float)  — fraction of 2xx responses
        error_rate (float)
    """
    if not request_logs:
        return {"total_requests": 0, "success_rate": 1.0, "error_rate": 0.0}

    total = len(request_logs)
    n_ok = sum(1 for r in request_logs if 200 <= r.get("status_code", 0) < 300)
    success_rate = n_ok / total
    return {
        "total_requests": total,
        "success_rate": success_rate,
        "error_rate": 1.0 - success_rate,
    }
