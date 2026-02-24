#!/usr/bin/env python
"""Convert XGBoost model to ONNX and benchmark inference performance.

Usage
-----
    python scripts/optimize_model.py
    python scripts/optimize_model.py --model-path model/xgboost_model.joblib \\
        --output model/model_optimized.onnx --n-calls 1000

Outputs
-------
    model/model_optimized.onnx       — converted ONNX model
    reports/onnx_benchmark.json      — benchmark results (mean/p95/p99/speedup)
    reports/cprofile_onnx_stats.txt  — cProfile of the ONNX inference path

Implementation notes
--------------------
- Uses onnxmltools for XGBoost → ONNX conversion (requires its own FloatTensorType).
- Benchmarks single-row predict_proba() to reflect real API usage.
- Falls back to a synthetic XGBoost model when the real model file is absent.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from linkedin_lead_scoring.monitoring.onnx_optimizer import (
    OnnxInferenceSession,
    benchmark_comparison,
    convert_xgboost_to_onnx,
    measure_memory_mb,
    save_benchmark_results,
    save_onnx_model,
)
from linkedin_lead_scoring.monitoring.profiler import run_cprofile

REPORTS_DIR = Path("reports")
MODEL_DEFAULT = "model/xgboost_model.joblib"
ONNX_DEFAULT = "model/model_optimized.onnx"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_or_create_model(model_path: str):
    import joblib, xgboost as xgb

    if os.path.exists(model_path):
        print(f"[model] Loading {model_path}")
        return joblib.load(model_path)

    print(f"[model] {model_path} not found — creating synthetic fallback (50 trees, 10 features)")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 10)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    m = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42)
    m.fit(X, y)
    return m


def get_n_features(model) -> int:
    if hasattr(model, "n_features_in_"):
        return model.n_features_in_
    if hasattr(model, "feature_names_in_"):
        return len(model.feature_names_in_)
    return 10


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XGBoost to ONNX and benchmark")
    parser.add_argument("--model-path", default=MODEL_DEFAULT)
    parser.add_argument("--output", default=ONNX_DEFAULT)
    parser.add_argument("--n-calls", type=int, default=1000)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ──────────────────────────────────────────────────────
    model = load_or_create_model(args.model_path)
    n_features = get_n_features(model)
    print(f"[model] n_features={n_features}")

    # ── 2. Convert to ONNX ─────────────────────────────────────────────────
    print("\n[onnx] Converting to ONNX…")
    onnx_model = convert_xgboost_to_onnx(model, n_features=n_features)
    onnx_path = save_onnx_model(onnx_model, args.output)
    onnx_size_kb = os.path.getsize(onnx_path) / 1024
    print(f"[onnx] Saved to {onnx_path} ({onnx_size_kb:.1f} KB)")

    # ── 3. Benchmark ───────────────────────────────────────────────────────
    session = OnnxInferenceSession(onnx_path)
    X_bench = np.random.default_rng(0).standard_normal((1, n_features)).astype(np.float32)

    print(f"\n[benchmark] Running {args.n_calls} calls each (single-row)…")
    results = benchmark_comparison(model, session, X_bench, n_calls=args.n_calls)

    jb = results["joblib"]
    rt_ = results["onnx"]
    print(f"\n  {'Metric':<10} {'joblib':>10} {'ONNX RT':>10}")
    print(f"  {'-'*30}")
    for k, label in [("mean_ms", "mean"), ("p50_ms", "p50"),
                     ("p95_ms", "p95"), ("p99_ms", "p99")]:
        print(f"  {label:<10} {jb[k]:>9.3f}ms {rt_[k]:>9.3f}ms")
    print(f"\n  Speedup (mean): {results['speedup_mean']:.2f}x "
          f"({'ONNX faster' if results['speedup_mean'] > 1 else 'joblib faster'})")

    # ── 4. Memory usage ────────────────────────────────────────────────────
    print("\n[memory] Measuring peak memory…")
    joblib_mem = measure_memory_mb(model, X_bench)
    onnx_mem = measure_memory_mb(session, X_bench)
    results["joblib"]["memory_mb"] = round(joblib_mem, 4)
    results["onnx"]["memory_mb"] = round(onnx_mem, 4)
    results["onnx_model_size_kb"] = round(onnx_size_kb, 2)
    print(f"  joblib peak: {joblib_mem:.4f} MB")
    print(f"  ONNX   peak: {onnx_mem:.4f} MB")

    # ── 5. cProfile (ONNX path) ────────────────────────────────────────────
    print("\n[cProfile] Profiling ONNX Runtime inference…")
    stats_str = run_cprofile(session, X_bench, n_calls=100)
    cprofile_path = REPORTS_DIR / "cprofile_onnx_stats.txt"
    cprofile_path.write_text(stats_str, encoding="utf-8")
    print(f"[cProfile] Saved to {cprofile_path}")

    # ── 6. Save results ────────────────────────────────────────────────────
    out = save_benchmark_results(results, str(REPORTS_DIR / "onnx_benchmark.json"))
    print(f"\n[results] Benchmark saved to {out}")


if __name__ == "__main__":
    main()
