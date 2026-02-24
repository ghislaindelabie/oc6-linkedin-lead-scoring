# Session Coordination — OC8 Parallel Development

**Version branch**: `v0.3.0`
**Last updated**: 2026-02-23 (Merging PRs progressively into v0.3.0)

---

## Session Status

| Session | Branch | Worktree | Status | Current Task | Last Commit |
|---------|--------|----------|--------|--------------|-------------|
| **Opus** (Coordinator) | `v0.3.0` | main repo | Active | Merging PRs | Progressive merge |
| **A** (Infra/CI/CD) | `feature/infra-cicd` | `worktrees/session-a` | **Merged** (PR #5) | — | All tasks done |
| **B** (API/Tests) | `feature/api-scoring` | `worktrees/session-b` | **Merged** (PR #6) | All B.1–B.7 done | 109 tests |
| **C** (Monitoring/Drift) | `feature/monitoring` | `worktrees/session-c` | **Merged** (PR #7) | All C.1–C.6 done | 86 tests |

## Merge Queue

| PR | Source Branch | Target | Status | Reviewer |
|----|--------------|--------|--------|----------|
| #5 | `feature/infra-cicd` | `v0.3.0` | **Ready** | Opus |
| B  | `feature/api-scoring` | `v0.3.0` | **Ready** | Opus |
| C  | `feature/monitoring` | `v0.3.0` | **Ready** | Opus |

## Dependency Tracker

| Dependency | Provider | Consumer(s) | Status |
|------------|----------|-------------|--------|
| Model artifact (`model/*.joblib`) | Session A | B, C | **Merged** (A.2) |
| Feature columns (`model/feature_columns.json`) | Session A | B, C | **Merged** (A.2) — 47 features |
| Reference data (`data/reference/`) | Session A | C | **Merged** (A.2) — 100 rows |
| DB module (`src/.../db/`) | Session A | B (logging) | **Merged** (A.3) |
| Alembic migrations (`alembic/`) | Session A | Deploy | **Merged** (A.3) |
| CI/CD workflows | Session A | All | **Merged** (A.5) — fixed for Python API deploy |
| API schemas finalized | Session B | C (for monitoring) | **Done** (B.1) |
| Production logging format | Session B | C (for drift analysis) | **Done** (B.4) |

## Shared Dependencies (pyproject.toml additions)

When a session needs a new dependency, record it here. Session A will integrate.

| Package | Needed By | Version | Added? |
|---------|-----------|---------|--------|
| `evidently` | C | `>=0.7.0` | ✅ In `requirements-streamlit.txt` |
| `streamlit` | C | `>=1.30.0` | ✅ In `requirements-streamlit.txt` |
| `joblib` | A, B | `>=1.3.0` | ✅ In `requirements-prod.txt` |
| `category-encoders` | A | `>=2.8.1` | ✅ In `requirements-prod.txt` (pinned) |
| `onnx` | C | `>=1.15.0` | In conda env — add to pyproject.toml |
| `onnxruntime` | C | `>=1.17.0` | In conda env — add to pyproject.toml |
| `onnxmltools` | C | `>=1.16.0` | In conda env — add to pyproject.toml |
| `asyncpg` | A | `>=0.29.0` | ✅ In `requirements-prod.txt` |
| `aiosqlite` | A | `>=0.19.0` | ✅ In `requirements-prod.txt` |
| `greenlet` | A | `>=3.0` | ✅ In `requirements-prod.txt` |

## Session C Progress Log

### C.1 — Drift Detection Module (complete)
- **Created**: `src/linkedin_lead_scoring/monitoring/__init__.py`, `drift.py`
- **Class**: `DriftDetector(reference_data)` — wraps Evidently 0.7 `Report`/`Snapshot` API
- **Methods**: `detect_data_drift(prod_df) → dict`, `detect_prediction_drift(ref_scores, prod_scores) → dict`, `generate_report(prod_df, path) → str`
- **Tests**: `tests/test_drift.py` — 13 tests, all passing
- **Key API note**: Evidently 0.7 — `Report.run()` returns `Snapshot`; use `snapshot.save_html()` / `snapshot.metric_results`
- **Reference data**: using mlruns artifact path (303 rows, 20 cols); `data/reference/` not yet created by Session A

### C.2 — Streamlit Monitoring Dashboard (complete)
- **Created**: `streamlit_app.py` (root, for HF Spaces), `src/linkedin_lead_scoring/monitoring/dashboard_utils.py`
- **Created**: `Dockerfile.streamlit`, `requirements-streamlit.txt`
- **Dashboard sections**: API Health, Score Distribution, Performance Metrics, Drift Analysis (Evidently HTML embed), Recent Predictions
- **Data strategy**: falls back to `simulate_production_logs()` when `logs/predictions.jsonl` absent
- **Reference data fallback**: checks `data/reference/training_reference.csv` → mlruns artifact → synthetic
- **Env vars**: `PREDICTIONS_LOG`, `API_REQUESTS_LOG`, `REFERENCE_DATA_PATH`, `API_BASE_URL`
- **Tests**: `tests/test_monitoring.py` — 19 tests covering log parsing, simulation, metrics
- **Note**: `httpx` already in `pyproject.toml`; `plotly` added to `requirements-streamlit.txt`

### C.3 — Performance Profiling (complete)
- **Created**: `src/linkedin_lead_scoring/monitoring/profiler.py`
  - `profile_model_inference(model, X, n_calls)` — perf_counter timing, returns mean/p50/p95/p99
  - `run_cprofile(model, X, n_calls)` — cProfile over predict_proba loop, returns stats string
  - `save_profile_results(results, path)` — JSON output
  - `format_profile_summary(results)` — human-readable table
- **Created**: `scripts/profile_api.py` — CLI with `--mode model|api|both`; synthetic fallback model; async httpx for concurrent API load test; saves to `reports/`
- **Created**: `notebooks/03_performance_analysis.ipynb` — baseline timing, distribution plots, cProfile section, ONNX comparison (reads C.4 output), findings/recommendations
- **Tests**: `tests/test_profiler.py` — 12 tests (49 total)
- **Known issue**: shared `oc6` conda env picks up whichever session's package was last installed; run `uv pip install -e .` from session-c root before testing

### C.4 — ONNX Model Optimization (complete)
- **Created**: `src/linkedin_lead_scoring/monitoring/onnx_optimizer.py`
  - `convert_xgboost_to_onnx(model, n_features) → onnx.ModelProto` — uses onnxmltools (NOT skl2onnx) FloatTensorType
  - `OnnxInferenceSession(path)` — wraps ORT, exposes `predict_proba()` matching sklearn API
  - `benchmark_comparison(joblib_model, onnx_session, X, n_calls) → dict` — perf_counter timing, `speedup_mean`
  - `measure_memory_mb(model, X, n_calls) → float` — tracemalloc peak
  - `save_benchmark_results(results, path) → str` — JSON to `reports/onnx_benchmark.json`
- **Created**: `scripts/optimize_model.py` — CLI: convert → benchmark 1000 calls → measure memory → cProfile ONNX path
- **Tests**: `tests/test_onnx_optimizer.py` — 11 tests (60 total)
- **Critical**: `onnxmltools.convert.common.data_types.FloatTensorType` must be used — NOT `skl2onnx.common.data_types.FloatTensorType` (causes RuntimeError)
- **ORT output**: `[labels, proba]` — proba may be numpy array or list-of-dicts; both handled in `predict_proba()`

### C.5 — Consolidate and Complete Monitoring Tests (complete)
- **Added to `tests/test_monitoring.py`** (19 → 25 tests):
  - `TestComputeUptimeStats` — 5 tests: empty logs, all-200, mixed success/errors, all-errors, success+error=1
  - `TestLoadJsonlWithBlankLines` — 1 test: blank and whitespace-only lines skipped in JSONL parser
- **Added to `tests/test_drift.py`** (13 → 19 tests):
  - `TestDriftDetectorEdgeCases` — 6 tests:
    - `test_target_column_excluded`: 'engaged' removed from `_feature_columns` at init
    - `test_single_numeric_column`: drift detection works with one feature
    - `test_production_has_extra_columns`: extra cols silently ignored via `_align_columns`
    - `test_extract_drifted_columns_skips_none_results`: covers `result is None: continue` branch (line 198)
    - `test_is_column_drifted_returns_false_for_none_result`: covers None guard (line 219)
    - `test_is_column_drifted_returns_false_for_result_without_widget`: covers no-widget fallback (line 229)
- **Total**: 72 tests, 99.55% coverage on monitoring package (was 95.07% after C.4)
- **Remaining uncovered**: `onnx_optimizer.py` line 124 (dict-based ORT proba output — defensive branch, never triggered by current ORT version)

### C.6 — Performance Report and Monitoring Guide (complete)
- **Created**: `docs/PERFORMANCE_REPORT.md`
  - Real benchmark data from `scripts/optimize_model.py` (500-call run)
  - Baseline: joblib mean 0.212 ms, p95 0.266 ms, memory 0.044 MB
  - ONNX: mean 0.008 ms, p95 0.005 ms, memory 0.001 MB — 26.5× speedup
  - cProfile analysis showing C++ engine dominates; Python overhead negligible
  - 5 production recommendations (deploy ONNX, keep joblib fallback, CI/CD regeneration, batch scoring, dashboard alerting)
- **Created**: `docs/MONITORING_GUIDE.md`
  - Dashboard access (local Streamlit + Docker/HF Spaces)
  - Metric interpretation tables with green/yellow/red thresholds
  - Drift indicator guide (covariate shift vs concept drift)
  - Retrain triggers (5 criteria with measurable thresholds)
  - Full alert threshold table with severity and owner
  - Log format reference and useful CLI commands
- **Created**: `tests/test_docs.py` — 14 structural tests verifying both docs exist and contain required sections
- **Generated**: `reports/onnx_benchmark.json`, `reports/cprofile_onnx_stats.txt` (supporting data)
- **Updated**: `README.md` monitoring section from "(Planned)" to implemented with capabilities list
- **Total tests**: 86 (was 72), coverage 99.55%

### Session C — All tasks complete. PR ready for v0.3.0.

## Notes

- Each session works ONLY in its worktree directory
- PRs target `v0.3.0`, never `main`
- Opus reviews all PRs before merge
- If you need a file owned by another session, create an interface/stub and document it here
- **Session C note**: production log format (`logs/predictions.jsonl`) defined in SESSION_C_TASKS.md; will simulate if not yet created by Session B
- **Worktree env collision**: the shared `oc6` conda env only holds one editable install at a time. Before running tests in any worktree, run `uv pip install -e .` from that worktree root.

## Manual Actions Required (user)

| Action | When | Notes |
|--------|------|-------|
| ~~Create Supabase staging project~~ | ~~Before merging PR #5~~ | ✅ Done |
| ~~Add `STAGING_DATABASE_URL` to GitHub repo secrets~~ | ~~Before first push to `v0.3.0`~~ | ✅ Done |
| ~~Merge PR #5 → #6 → #7 into `v0.3.0`~~ | ~~Now~~ | ✅ All merged |
| Set `DATABASE_URL` on production HF Space | Before production go-live | Needed for persistent prediction logging |
| Add `DATABASE_URL` GitHub secret | Before production deploy workflow | For CI/CD production deploy |
