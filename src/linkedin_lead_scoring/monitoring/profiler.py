"""Model and API inference profiler.

Provides importable functions for benchmarking XGBoost model inference and
API endpoint latency.  Used by scripts/profile_api.py and for unit tests.

Key functions
-------------
profile_model_inference(model, X, n_calls) → dict
    Measure predict_proba() latency over n_calls repetitions using
    time.perf_counter() for high-resolution timing.

run_cprofile(model, X, n_calls) → str
    Run cProfile over a predict_proba() loop and return the stats as a string.

save_profile_results(results, path) → str
    Persist a profiling results dict to JSON.

format_profile_summary(results) → str
    Human-readable table of key latency percentiles.
"""
import cProfile
import io
import json
import os
import pstats
import time
from typing import Any, Union

import numpy as np


def profile_model_inference(
    model: Any,
    X: np.ndarray,
    n_calls: int = 1000,
) -> dict:
    """Benchmark predict_proba() latency over repeated calls.

    Each call is timed individually using ``time.perf_counter()`` so the
    distribution (not just the mean) can be reported.

    Parameters
    ----------
    model:
        A fitted scikit-learn-compatible classifier with ``predict_proba()``.
    X:
        Feature matrix passed to ``predict_proba()`` on every call.
    n_calls:
        Number of repetitions.

    Returns
    -------
    dict with keys:
        n_calls (int), mean_ms, p50_ms, p95_ms, p99_ms, min_ms, max_ms (float)
    """
    times_ms: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        model.predict_proba(X)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "n_calls": n_calls,
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def run_cprofile(
    model: Any,
    X: np.ndarray,
    n_calls: int = 100,
    sort_key: str = "cumulative",
    n_lines: int = 30,
) -> str:
    """Profile a predict_proba() loop with cProfile and return stats as a string.

    Parameters
    ----------
    model:
        Fitted model with predict_proba().
    X:
        Feature matrix.
    n_calls:
        Number of predict_proba() calls inside the profiled block.
    sort_key:
        cProfile sort key (e.g. 'cumulative', 'tottime').
    n_lines:
        Number of lines to include in the stats output.

    Returns
    -------
    str
        Formatted cProfile stats output.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(n_calls):
        model.predict_proba(X)
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats(sort_key)
    stats.print_stats(n_lines)
    return buf.getvalue()


def save_profile_results(results: dict, path: Union[str, os.PathLike]) -> str:
    """Write profiling results to a JSON file.

    Creates parent directories as needed.

    Parameters
    ----------
    results:
        Profiling results dict (e.g. from profile_model_inference).
    path:
        Destination file path.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path


def format_profile_summary(results: dict, label: str = "Inference") -> str:
    """Format profiling results as a human-readable table.

    Parameters
    ----------
    results:
        Dict from profile_model_inference (must contain mean_ms, p50_ms, etc.).
    label:
        Section header label.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines = [
        f"=== {label} Profiling Results ===",
        f"  n_calls : {results.get('n_calls', 'N/A')}",
        f"  mean    : {results.get('mean_ms', 0):.3f} ms",
        f"  p50     : {results.get('p50_ms', 0):.3f} ms",
        f"  p95     : {results.get('p95_ms', 0):.3f} ms",
        f"  p99     : {results.get('p99_ms', 0):.3f} ms",
        f"  min     : {results.get('min_ms', 0):.3f} ms",
        f"  max     : {results.get('max_ms', 0):.3f} ms",
    ]
    return "\n".join(lines)
