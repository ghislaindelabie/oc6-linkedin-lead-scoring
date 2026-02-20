# Session Coordination — OC8 Parallel Development

**Version branch**: `v0.3.0`
**Last updated**: 2026-02-20

---

## Session Status

| Session | Branch | Worktree | Status | Current Task | Last Commit |
|---------|--------|----------|--------|--------------|-------------|
| **Opus** (Coordinator) | `v0.3.0` | main repo | Active | Setup complete | `0be90e2` |
| **A** (Infra/CI/CD) | `feature/infra-cicd` | `worktrees/session-a` | Not started | — | — |
| **B** (API/Tests) | `feature/api-scoring` | `worktrees/session-b` | Not started | — | — |
| **C** (Monitoring/Drift) | `feature/monitoring` | `worktrees/session-c` | In progress | C.4 ONNX optimization | C.3 done |

## Merge Queue

| PR | Source Branch | Target | Status | Reviewer |
|----|--------------|--------|--------|----------|
| — | — | — | — | — |

## Dependency Tracker

| Dependency | Provider | Consumer(s) | Status |
|------------|----------|-------------|--------|
| Model artifact (`model/*.joblib`) | Session A | B, C | Pending |
| Feature columns (`model/feature_columns.json`) | Session A | B, C | Pending |
| Reference data (`data/reference/`) | Session A | C | Pending |
| API schemas finalized | Session B | C (for monitoring) | Pending |
| Production logging format | Session B | C (for drift analysis) | Pending |

## Shared Dependencies (pyproject.toml additions)

When a session needs a new dependency, record it here. Session A will integrate.

| Package | Needed By | Version | Added? |
|---------|-----------|---------|--------|
| `evidently` | C | `>=0.7.0` | Installed in oc6 env (0.7.20) — add to pyproject.toml/requirements-prod.txt |
| `streamlit` | C | `>=1.30.0` | ✅ Added to `requirements-streamlit.txt` (dashboard only, not API) |
| `joblib` | A, B | `>=1.3.0` | No |
| `onnx` | C | `>=1.15.0` | Installed in oc6 env (1.20.1) — add to pyproject.toml |
| `onnxruntime` | C | `>=1.17.0` | Installed in oc6 env (1.24.2) — add to pyproject.toml |
| `onnxmltools` | C | `>=1.16.0` | Installed in oc6 env (1.16.0) — add to pyproject.toml |
| `scipy` | C | `>=1.11.0` | Check if already in env |
| `psycopg2-binary` | A | `>=2.9.0` | No |
| `supabase` | A | `>=2.0.0` | No |

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

### C.4 → C.6 — Pending
Tasks being worked on in sequence per SESSION_C_TASKS.md.

## Notes

- Each session works ONLY in its worktree directory
- PRs target `v0.3.0`, never `main`
- Opus reviews all PRs before merge
- If you need a file owned by another session, create an interface/stub and document it here
- **Session C note**: production log format (`logs/predictions.jsonl`) defined in SESSION_C_TASKS.md; will simulate if not yet created by Session B
- **Worktree env collision**: the shared `oc6` conda env only holds one editable install at a time. Before running tests in any worktree, run `conda run -n oc6 uv pip install -e .` from that worktree root to re-point the env at the correct `src/`.
