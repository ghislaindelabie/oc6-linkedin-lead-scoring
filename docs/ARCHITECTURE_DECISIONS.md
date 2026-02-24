# Architecture Decisions — LeadGen Scorer v0.3.0

This document records key architecture and infrastructure decisions made during the project, their rationale, and alternatives considered.

---

## ADR-1: Model Serving — joblib (not ONNX)

**Status**: Current

**Decision**: The production API loads the XGBoost model via `joblib.load()` at startup. ONNX Runtime is available as a benchmarking tool but is **not** used in the inference path.

**Context**: We benchmarked both approaches. ONNX Runtime is 26.5x faster on single-row inference (0.008 ms vs 0.21 ms mean). However, joblib latency at 0.21 ms/prediction is already sub-millisecond — well within API SLA requirements.

**Rationale**:
- Joblib is simpler: one artifact (`xgboost_model.joblib`), standard `predict_proba()` call
- ONNX adds 3 dependencies (`onnx`, `onnxruntime`, `onnxmltools`) and a conversion step
- 0.21 ms is already negligible compared to network latency (~50-200 ms on HF Spaces)
- ONNX conversion has a gotcha: must use `onnxmltools.convert.common.data_types.FloatTensorType`, NOT `skl2onnx`'s version (causes RuntimeError)

**Trade-off**: We accept slightly higher inference latency for simpler deployment and fewer dependencies in the production container.

### How to switch to ONNX in production

Everything is already implemented in `monitoring/onnx_optimizer.py`. The switch requires:

**Step 1: Generate the ONNX artifact**

```bash
python scripts/optimize_model.py \
  --model-path model/xgboost_model.joblib \
  --output model/model_optimized.onnx
```

This produces `model/model_optimized.onnx` (5.8 KB vs 800 KB joblib).

**Step 2: Add ONNX deps to production requirements**

In `requirements-prod.txt`, add:
```
onnxruntime>=1.17,<2.0
```

Note: only `onnxruntime` is needed at inference time. `onnx` and `onnxmltools` are only needed for conversion (build-time).

**Step 3: Modify model loading in `api/predict.py`**

Replace the lifespan model loading block (lines 82-95):

```python
# Current (joblib):
_state["model"] = joblib.load(_MODEL_PATH)

# New (ONNX with joblib fallback):
onnx_path = "model/model_optimized.onnx"
if os.path.exists(onnx_path):
    from linkedin_lead_scoring.monitoring.onnx_optimizer import OnnxInferenceSession
    _state["model"] = OnnxInferenceSession(onnx_path)
    logger.info("ONNX model loaded")
else:
    _state["model"] = joblib.load(_MODEL_PATH)
    logger.info("Joblib model loaded (ONNX not found)")
```

No other code changes needed — `OnnxInferenceSession.predict_proba()` has the same interface as XGBoost's.

**Step 4: Add ONNX conversion to CI/CD**

In `scripts/export_model.py`, add after the joblib export:
```python
from linkedin_lead_scoring.monitoring.onnx_optimizer import convert_xgboost_to_onnx, save_onnx_model
onnx_model = convert_xgboost_to_onnx(model, n_features=len(feature_columns))
save_onnx_model(onnx_model, "model/model_optimized.onnx")
```

**Step 5: Commit `model/model_optimized.onnx`** alongside the joblib artifact.

**Expected impact**: p95 latency drops from 0.27 ms to 0.005 ms. Model file shrinks from 800 KB to 5.8 KB.

---

## ADR-2: One-Hot Encoding — manual encoding (not pd.get_dummies)

**Status**: Current — critical for correctness

**Decision**: At inference time, OHE columns are created manually from a fixed `feature_columns.json` list, not via `pd.get_dummies()`.

**Context**: `pd.get_dummies()` produces different columns depending on which categories are present in the input. A single lead with `companysize="11-50"` generates only `companysize_11-50`, while a batch of 5 leads might generate `companysize_11-50`, `companysize_51-200`, `companysize_201-500`. The model expects all columns to be present.

**Implementation** (`features.py`, lines 155-169):
1. Initialize all OHE columns from `feature_columns.json` with value 0
2. For each category present in the input, set the corresponding column to 1
3. Drop extra columns; add missing columns as 0
4. Reorder to match the training column order

**Rationale**: Guarantees that single-lead and batch predictions produce identical scores for the same lead. Verified by 5 determinism tests in `test_features.py::TestOneHotEncodeDeterminism`.

---

## ADR-3: Model loaded once at startup (not per-request)

**Status**: Current — critical for performance

**Decision**: The ML model and all preprocessing artifacts are loaded once during the FastAPI `lifespan` context manager and stored in a module-level `_state` dict.

**Rationale**:
- `joblib.load()` reads from disk (~2-5 ms): acceptable once, unacceptable per-request
- Module-level state is shared across all async handlers (no copies)
- Cleanup happens automatically on shutdown via the lifespan exit

**Alternative rejected**: Per-request loading via dependency injection. Would add disk I/O to every prediction call and risk memory leaks from repeated deserialization.

---

## ADR-4: Two HF Spaces — API + Dashboard (not a monolith)

**Status**: Current

**Decision**: Deploy the FastAPI API and Streamlit dashboard as two separate Hugging Face Spaces.

| Space | SDK | Purpose |
|-------|-----|---------|
| `oc6-bizdev-ml-api` | Docker | FastAPI scoring API |
| `oc6-bizdev-monitoring` | Streamlit | Monitoring dashboard |

**Rationale**:
- Different runtimes: API needs Docker (custom Dockerfile), dashboard uses Streamlit SDK
- Independent scaling: API can restart without affecting monitoring
- Different dependency sets: API is lean (`requirements-prod.txt`), dashboard needs evidently, plotly, etc. (`requirements-streamlit.txt`)
- Different update cadence: dashboard can be redeployed without touching the model

**Trade-off**: Two spaces to manage instead of one. CORS must be configured to allow the dashboard to call the API.

---

## ADR-5: Two-stage deployment — staging then production

**Status**: Current

**Decision**: Pushes to `v0.3.0` auto-deploy to staging. Pushes to `main` auto-deploy to production.

```
feature/* --PR--> v0.3.0 --staging deploy--> main --production deploy-->
```

**Staging URLs**: `*-staging.hf.space`
**Production URLs**: `*.hf.space`

**Rationale**: Catch deployment issues (Docker builds, env vars, migrations) before they hit production. Staging uses a separate Supabase database.

---

## ADR-6: Database optional — API works offline with JSONL logs

**Status**: Current

**Decision**: The API can operate fully without a database. Prediction logs and API request logs are written to JSONL files (`logs/predictions.jsonl`, `logs/api_requests.jsonl`). Supabase PostgreSQL is available but not required.

**Implementation details**:
- Log writes wrapped in `try/except: pass` — failures never crash the endpoint
- Alembic migrations run at Docker startup but are non-fatal (`|| echo "WARNING"`)
- Local development automatically falls back to SQLite via `aiosqlite`
- `connection.py` auto-detects and rewrites the driver: `postgresql://` → `postgresql+asyncpg://`

**Rationale**: HF Spaces can restart at any time. The API must serve predictions even if the database is temporarily unreachable. JSONL files provide a minimal logging baseline that the monitoring dashboard can read.

---

## ADR-7: Target encoding for high-cardinality categoricals

**Status**: Current

**Decision**: Use `category_encoders.TargetEncoder` for 6 high-cardinality columns: `llm_industry`, `industry`, `companyindustry`, `languages`, `location`, `companylocation`.

**Rationale**: These columns have dozens of unique values. OHE would create 100+ sparse columns. Target encoding maps each category to the mean of the target variable (engagement rate), producing a single numeric column per feature.

**Serialization**: The fitted encoder is saved inside `model/preprocessor.joblib` as a dict:
```python
{"target_encoder": fitted_TargetEncoder, "te_cols": ["llm_industry", ...]}
```

---

## ADR-8: Evidently AI for drift detection (not custom statistics)

**Status**: Current

**Decision**: Use Evidently v0.7+ for data drift and prediction drift detection.

**What it provides**:
- `DataDriftPreset`: KS test (numeric) + chi-squared (categorical) per feature
- Overall drift flag: true when >= 50% of features have drifted (p-value < 0.05)
- Prediction drift: KS test on score distributions (reference vs production)
- Full HTML report with interactive distribution plots

**Rationale**: Evidently handles statistical test selection, multiple comparison issues, and visualization. Writing custom drift detection would require implementing the same statistical tests with less rigor.

**Important v0.7 API note**: `Report.run()` returns a `Snapshot` object. Use `snapshot.save_html()` for the report and `snapshot.metric_results` for programmatic access.

---

## ADR-9: Conditional ONNX import — graceful degradation

**Status**: Current

**Decision**: The `monitoring/__init__.py` imports `onnx_optimizer` inside a `try/except`. If onnx is not installed, `onnx_optimizer` is set to `None`.

```python
try:
    from linkedin_lead_scoring.monitoring import onnx_optimizer
except ImportError:
    onnx_optimizer = None
```

**Rationale**: The API container only installs `requirements-prod.txt` (no onnx). If `__init__.py` unconditionally imports `onnx_optimizer`, then importing `from linkedin_lead_scoring.monitoring import dashboard_utils` fails — breaking the monitoring dashboard import chain.

**Tested**: `test_monitoring.py::test_monitoring_package_works_without_onnx` verifies this by mocking onnx as absent.

---

## ADR-10: Mock model for development and testing

**Status**: Current

**Decision**: When `APP_ENV=development` or `APP_ENV=test`, the API loads a `_MockModel` that returns a fixed score of 0.65 for all inputs.

**Rationale**:
- CI/CD runs without real model artifacts (smaller Docker image, faster builds)
- Local development works without running `export_model.py` first
- Test fixtures can override `_state` directly via `monkeypatch.setattr()`
- Production (`APP_ENV=production`) requires real artifacts — returns 503 if missing

---

## ADR-11: Rate limiting — informational headers only

**Status**: Current

**Decision**: The `RateLimitHeadersMiddleware` adds `X-RateLimit-Limit` headers but does **not** enforce rate limits.

**Rationale**: Actual rate limiting should be handled by the reverse proxy or API gateway (HF Spaces infrastructure). The middleware signals intent to clients without adding state management complexity to the application layer.

---

## ADR-12: Shared feature engineering module

**Status**: Current — critical for consistency

**Decision**: A single `features.py` module is imported by both `scripts/export_model.py` (training) and `api/predict.py` (inference).

**Functions**: `extract_text_features()`, `fill_missing_values()`, `target_encode_columns()`, `one_hot_encode_columns()`, `preprocess_for_inference()`.

**Rationale**: Eliminates training/serving skew. Any change to feature engineering is automatically reflected in both training and inference. The module has 100% test coverage.

---

## ADR-13: Version pinning strategy

**Status**: Current

**Decision**: `requirements-prod.txt` pins minor versions (`xgboost>=3.2.0,<3.3`). `requirements-streamlit.txt` pins major versions (`evidently>=0.7.0,<1.0`).

**Rationale**: Production pins are tight because the XGBoost model binary format can change between minor versions (a model trained with 3.2.x may not load with 3.3.x). Monitoring tool pins are looser because they don't affect model artifacts.

**Pinned production deps**: xgboost 3.2.x, scikit-learn 1.8.x, pandas 2.3.x, numpy 2.4.x, category-encoders 2.8.x.

---

## ADR-14: Five CI/CD workflows

**Status**: Current

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | All pushes/PRs | Ruff lint, pytest + coverage, Docker build check |
| `staging.yml` | Push to `v*.*.*` | Tests + deploy API + dashboard to staging HF Spaces |
| `production.yml` | Push to `main` | Deploy API + dashboard to production HF Spaces |
| `dashboard.yml` | Push to `main` | Deploy Streamlit dashboard to production |
| `security.yml` | Weekly + manual | pip-audit (CVEs) + bandit (static analysis) |

**Rationale**: Separation of concerns. CI runs on every push (fast feedback), staging deploys only on version branches, production only on main. Security scanning runs weekly to catch new CVEs without blocking development.

---

## ADR-15: JSONL log format (not CSV, not database-only)

**Status**: Current

**Decision**: Prediction and API request logs use JSON Lines format (one JSON object per line).

**Rationale**:
- Structured: input features are nested dicts — CSV would require flattening or escaping
- Streaming-friendly: append-only, one line at a time, no header issues
- Human-readable: `cat logs/predictions.jsonl | jq .` for debugging
- Dashboard-compatible: `dashboard_utils.load_prediction_logs()` reads them directly
- No external dependency: works without database, Supabase, or any service

---

## ADR-16: Synthetic data fallback in dashboard

**Status**: Current

**Decision**: When real log files don't exist, the Streamlit dashboard generates synthetic data via `simulate_production_logs(n=100, seed=42)`.

**Implementation**: Beta distribution (a=2.5, b=3.5) for scores, Gamma (2.0, 5.0) for inference times — introduces slight drift vs training distribution so the drift panel has something to show.

**Rationale**: The dashboard must be functional during demos and development even without a running API generating real logs.
