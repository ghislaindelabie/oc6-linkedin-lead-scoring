"""ONNX model conversion and benchmarking utilities.

Converts a trained XGBoost classifier to ONNX format using onnxmltools and
provides an OnnxInferenceSession wrapper for ONNX Runtime inference.  Also
includes benchmark_comparison() for a head-to-head joblib vs ONNX latency
comparison.

Important: onnxmltools requires its OWN FloatTensorType —
  ``onnxmltools.convert.common.data_types.FloatTensorType``
  NOT ``skl2onnx.common.data_types.FloatTensorType``.

Usage example
-------------
    from linkedin_lead_scoring.monitoring.onnx_optimizer import (
        convert_xgboost_to_onnx, save_onnx_model,
        OnnxInferenceSession, benchmark_comparison, save_benchmark_results,
    )

    onnx_model = convert_xgboost_to_onnx(xgb_clf, n_features=10)
    save_onnx_model(onnx_model, "model/model_optimized.onnx")
    session = OnnxInferenceSession("model/model_optimized.onnx")
    results = benchmark_comparison(xgb_clf, session, X_sample, n_calls=1000)
    save_benchmark_results(results, "reports/onnx_benchmark.json")
"""
import json
import os
import time
import tracemalloc
from typing import Union

import numpy as np
import onnx
import onnxruntime as rt
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_xgboost_to_onnx(model, n_features: int) -> onnx.ModelProto:
    """Convert a fitted XGBoost classifier to an ONNX ModelProto.

    Parameters
    ----------
    model:
        A fitted ``xgboost.XGBClassifier``.
    n_features:
        Number of input features (columns in the training feature matrix).

    Returns
    -------
    onnx.ModelProto
        The converted ONNX model ready for serialisation or inference.
    """
    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    return convert_xgboost(model, initial_types=initial_types)


def save_onnx_model(onnx_model: onnx.ModelProto, path: Union[str, os.PathLike]) -> str:
    """Serialise an ONNX model to disk.

    Creates parent directories as needed.

    Parameters
    ----------
    onnx_model:
        ONNX ModelProto to save.
    path:
        Destination file path (typically ``model/model_optimized.onnx``).

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    onnx.save_model(onnx_model, path)
    return path


# ---------------------------------------------------------------------------
# ONNX Runtime wrapper
# ---------------------------------------------------------------------------

class OnnxInferenceSession:
    """Thin wrapper around ``onnxruntime.InferenceSession``.

    Provides a ``predict_proba()`` interface matching scikit-learn/XGBoost so
    the session can be passed directly to ``benchmark_comparison()``.

    Parameters
    ----------
    model_path:
        Path to a serialised ``.onnx`` model file.
    """

    def __init__(self, model_path: Union[str, os.PathLike]) -> None:
        self._session = rt.InferenceSession(str(model_path))
        self._input_name: str = self._session.get_inputs()[0].name

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Run inference and return class probability matrix.

        Parameters
        ----------
        X:
            Float32 feature matrix with shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_classes)`` probability array.
        """
        X = np.asarray(X, dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: X})
        # ORT returns [labels, probabilities_dict_or_array]
        # For binary classifiers: outputs[1] is a list of dicts or 2-D array
        proba = outputs[1]
        if isinstance(proba, list):
            # List of dicts → convert to array
            proba = np.array([[d[k] for k in sorted(d)] for d in proba], dtype=np.float32)
        return np.asarray(proba, dtype=np.float32)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_comparison(
    joblib_model,
    onnx_session: OnnxInferenceSession,
    X: np.ndarray,
    n_calls: int = 1000,
) -> dict:
    """Benchmark joblib model vs ONNX Runtime on repeated predict_proba calls.

    Uses ``time.perf_counter()`` for high-resolution timing. Each backend
    runs ``n_calls`` predictions on the same ``X`` matrix, one call at a time
    (simulating real API single-request latency).

    Parameters
    ----------
    joblib_model:
        Fitted scikit-learn-compatible model with ``predict_proba()``.
    onnx_session:
        An ``OnnxInferenceSession`` wrapping the converted ONNX model.
    X:
        Feature matrix for the benchmark (typically 1 or a small batch).
    n_calls:
        Number of timed repetitions per backend.

    Returns
    -------
    dict with keys:
        joblib (dict)     — timing stats for the joblib model
        onnx (dict)       — timing stats for ONNX Runtime
        speedup_mean (float) — joblib_mean / onnx_mean (>1 means ONNX faster)
        n_calls (int)
    """
    def _time_calls(fn, n: int) -> dict:
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn(X)
            times.append((time.perf_counter() - t0) * 1000.0)
        arr = np.array(times)
        return {
            "n_calls": n,
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
        }

    joblib_stats = _time_calls(joblib_model.predict_proba, n_calls)
    onnx_stats = _time_calls(onnx_session.predict_proba, n_calls)

    speedup = joblib_stats["mean_ms"] / onnx_stats["mean_ms"] if onnx_stats["mean_ms"] > 0 else 0.0

    return {
        "joblib": joblib_stats,
        "onnx": onnx_stats,
        "speedup_mean": round(speedup, 3),
        "n_calls": n_calls,
    }


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def measure_memory_mb(model, X: np.ndarray, n_calls: int = 10) -> float:
    """Measure peak memory usage (in MB) of ``predict_proba()`` via tracemalloc.

    Parameters
    ----------
    model:
        Model with ``predict_proba()`` method.
    X:
        Feature matrix.
    n_calls:
        Number of calls inside the measured block.

    Returns
    -------
    float
        Peak memory usage in megabytes during the timed block.
    """
    tracemalloc.start()
    for _ in range(n_calls):
        model.predict_proba(X)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_benchmark_results(results: dict, path: Union[str, os.PathLike]) -> str:
    """Write benchmark comparison results to a JSON file.

    Parameters
    ----------
    results:
        Dict from ``benchmark_comparison()``.
    path:
        Destination path (e.g. ``reports/onnx_benchmark.json``).

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
