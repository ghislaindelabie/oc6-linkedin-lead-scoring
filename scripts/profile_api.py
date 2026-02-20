#!/usr/bin/env python
"""Inference and API profiling script.

Usage
-----
# Profile the model directly (no API needed):
python scripts/profile_api.py --mode model --model-path model/xgboost_model.joblib

# Profile the live API (API must be running):
python scripts/profile_api.py --mode api --api-url http://localhost:7860 --n-requests 100

# Both modes:
python scripts/profile_api.py --mode both

Results are saved to reports/profile_results.json and reports/cprofile_stats.txt.

Implementation notes
--------------------
- Inference timing uses time.perf_counter() for high-resolution nanosecond accuracy.
- API profiling uses httpx with asyncio for concurrent requests.
- cProfile covers the full predict_proba() pipeline to identify bottlenecks.
- If no real model file is found, a synthetic XGBoost model is used as fallback
  so the script is always runnable for demonstration purposes.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np

# Ensure the src package is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from linkedin_lead_scoring.monitoring.profiler import (
    format_profile_summary,
    profile_model_inference,
    run_cprofile,
    save_profile_results,
)

REPORTS_DIR = Path("reports")
MODEL_DEFAULT = "model/xgboost_model.joblib"


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_or_create_model(model_path: str):
    """Load a joblib model or fall back to a synthetic XGBoost model."""
    import joblib
    import xgboost as xgb

    if os.path.exists(model_path):
        print(f"[model] Loading from {model_path}")
        return joblib.load(model_path)

    # Fallback: train a tiny synthetic model so the script always runs
    print(f"[model] {model_path} not found — using synthetic fallback model")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 10)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X, y)
    return model


def make_sample_input(model, n_rows: int = 1) -> np.ndarray:
    """Generate random float32 input matching the model's expected feature count."""
    # Try to infer feature count from the model
    if hasattr(model, "n_features_in_"):
        n_features = model.n_features_in_
    elif hasattr(model, "feature_names_in_"):
        n_features = len(model.feature_names_in_)
    else:
        n_features = 10  # safe fallback

    return np.random.default_rng(0).standard_normal((n_rows, n_features)).astype(np.float32)


# ---------------------------------------------------------------------------
# Model inference profiling
# ---------------------------------------------------------------------------

def run_model_profiling(model_path: str, n_calls: int = 1000) -> dict:
    """Profile model inference time and cProfile bottlenecks."""
    model = load_or_create_model(model_path)
    X = make_sample_input(model, n_rows=1)

    print(f"\n[profiler] Running {n_calls} inference calls (single row)…")
    results = profile_model_inference(model, X, n_calls=n_calls)
    print(format_profile_summary(results, label="Single-row XGBoost"))

    # Also profile a batch of 50 rows
    X_batch = make_sample_input(model, n_rows=50)
    batch_results = profile_model_inference(model, X_batch, n_calls=n_calls)
    print(format_profile_summary(batch_results, label="Batch-50 XGBoost"))

    # cProfile report
    print("\n[cProfile] Running profiling session…")
    stats_str = run_cprofile(model, X, n_calls=100)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cprofile_path = REPORTS_DIR / "cprofile_stats.txt"
    cprofile_path.write_text(stats_str, encoding="utf-8")
    print(f"[cProfile] Stats saved to {cprofile_path}")

    combined = {
        "single_row": results,
        "batch_50": batch_results,
        "model_path": model_path,
    }
    out_path = save_profile_results(combined, str(REPORTS_DIR / "profile_model.json"))
    print(f"[profiler] Results saved to {out_path}")
    return combined


# ---------------------------------------------------------------------------
# API latency profiling
# ---------------------------------------------------------------------------

SAMPLE_PAYLOAD = {
    "jobtitle": "CTO",
    "industry": "tech",
    "companysize": "11-50",
    "location": "Paris",
    "timezone": 1.0,
    "icebreaker": 0.5,
    "companyfoundedon": 2015.0,
}


async def _send_requests(api_url: str, n_requests: int) -> tuple[list[float], int, float]:
    """Send n_requests POST /predict calls concurrently and collect latencies."""
    endpoint = f"{api_url.rstrip('/')}/predict"
    latencies: list[float] = []
    errors = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [client.post(endpoint, json=SAMPLE_PAYLOAD) for _ in range(n_requests)]
        t0 = time.perf_counter()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_elapsed = time.perf_counter() - t0

    for resp in responses:
        if isinstance(resp, Exception):
            errors += 1
        else:
            latencies.append(resp.elapsed.total_seconds() * 1000)

    throughput = n_requests / total_elapsed
    return latencies, errors, throughput


def run_api_profiling(api_url: str, n_requests: int = 100) -> dict:
    """Profile the live API with concurrent requests."""
    print(f"\n[api] Sending {n_requests} concurrent requests to {api_url}…")

    latencies, errors, throughput = asyncio.run(
        _send_requests(api_url, n_requests)
    )

    if not latencies:
        print("[api] No successful responses — is the API running?")
        return {"error": "no_successful_responses", "api_url": api_url}

    arr = np.array(latencies)
    results = {
        "n_requests": n_requests,
        "n_errors": errors,
        "throughput_rps": round(throughput, 2),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "api_url": api_url,
    }
    print(format_profile_summary(results, label="API End-to-End"))
    print(f"  throughput: {results['throughput_rps']:.1f} req/s")
    print(f"  errors    : {errors}/{n_requests}")

    out_path = save_profile_results(results, str(REPORTS_DIR / "profile_api.json"))
    print(f"[api] Results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Profile model inference and API latency")
    parser.add_argument(
        "--mode",
        choices=["model", "api", "both"],
        default="model",
        help="What to profile (default: model)",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_DEFAULT,
        help=f"Path to joblib model file (default: {MODEL_DEFAULT})",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=1000,
        help="Number of inference calls for model profiling (default: 1000)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:7860",
        help="API base URL for API profiling",
    )
    parser.add_argument(
        "--n-requests",
        type=int,
        default=100,
        help="Number of concurrent API requests (default: 100)",
    )

    args = parser.parse_args()

    if args.mode in ("model", "both"):
        run_model_profiling(args.model_path, n_calls=args.n_calls)

    if args.mode in ("api", "both"):
        run_api_profiling(args.api_url, n_requests=args.n_requests)


if __name__ == "__main__":
    main()
