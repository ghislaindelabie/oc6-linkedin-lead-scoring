# Session B — API & Tests

**Branch**: `feature/api-scoring`
**Worktree**: `worktrees/session-b`
**Role**: FastAPI prediction endpoint, request/response schemas, unit tests, integration tests

---

## IMPORTANT RULES

1. **Work ONLY in this worktree directory** — never `cd` outside it
2. **Push ONLY to `feature/api-scoring`** — never push to main or v0.3.0
3. **Own ONLY these files**: `src/linkedin_lead_scoring/api/`, `tests/test_api*.py`, `tests/test_predict*.py`, `tests/test_schemas*.py`, `tests/conftest.py`
4. **Do NOT touch**: `.github/`, `Dockerfile`, `src/linkedin_lead_scoring/monitoring/`, `src/linkedin_lead_scoring/db/`
5. Run `python -m pytest tests/ -v --tb=short` before every commit
6. Read `CLAUDE.md` and `SESSION_COORDINATION.md` for full context

---

## Context: Feature Columns

The model uses these input features (from `data/processed/linkedin_leads_clean.csv`):

**Numeric** (use directly):
- `llm_quality` (int, 0-100): Profile quality score
- `llm_engagement` (float, 0-1): Engagement likelihood
- `llm_decision_maker` (float, 0-1): Decision maker probability
- `llm_company_fit` (int, 0-2): Company fit score
- `companyfoundedon` (float, year): Company founding year

**Categorical** (need encoding by preprocessor):
- `llm_seniority`: Entry/Mid/Senior/Executive/C-Level
- `llm_industry`: Technology - SaaS, Consulting, etc.
- `llm_geography`: international_hub/regional_hub/other
- `llm_business_type`: leaders/experts/salespeople/workers/others
- `industry`: Information Technology & Services, etc.
- `companyindustry`: Software Development, etc.
- `companysize`: 1-10, 11-50, 51-200, 201-500, 501-1000, 1001-5000, 5001-10000, 10001+
- `companytype`: Public Company, Privately Held, etc.
- `languages`: Arabic, English, French (comma-separated)
- `location`: City, Region, Country
- `companylocation`: City, Country

**Text** (need preprocessing):
- `summary`: Professional summary
- `skills`: Comma-separated skills
- `jobtitle`: Job title

**Target**: `engaged` (int, 0 or 1)

---

## Task B.1: Design API Schemas

Rewrite `src/linkedin_lead_scoring/api/schemas.py`:

1. `LeadInput` — single lead prediction request:
   - All 19 feature fields with proper types, Optional where appropriate
   - Validation: llm_quality 0-100, llm_engagement 0-1, etc.
   - Example values in Field descriptions

2. `LeadPrediction` — single prediction response:
   - `score` (float, 0-1): engagement probability
   - `label` (str): "engaged" or "not_engaged"
   - `confidence` (str): "low"/"medium"/"high"
   - `model_version` (str)
   - `inference_time_ms` (float)

3. `BatchPredictionRequest` — list of LeadInput
4. `BatchPredictionResponse` — list of LeadPrediction + summary stats
5. `HealthResponse` — existing, add model_loaded boolean
6. `ErrorResponse` — structured error responses

Commit: `feat: design comprehensive API schemas with validation`

## Task B.2: Implement `/predict` Endpoint (CRITICAL)

Create `src/linkedin_lead_scoring/api/predict.py`:

1. **Model loading at startup** (CRITICAL — assignment explicitly requires this):
   ```python
   # Load ONCE at module level or via app lifespan
   model = None
   preprocessor = None

   @asynccontextmanager
   async def lifespan(app):
       global model, preprocessor
       model = joblib.load("model/xgboost_model.joblib")
       preprocessor = joblib.load("model/preprocessor.joblib")
       feature_cols = json.load(open("model/feature_columns.json"))
       yield
   ```

2. **POST /predict** endpoint:
   - Accept LeadInput
   - Preprocess features using loaded preprocessor
   - Run model.predict_proba()
   - Measure inference time
   - Return LeadPrediction

3. **Error handling**:
   - Missing required fields → 422 with clear message
   - Invalid value ranges → 422 with field-specific errors
   - Model not loaded → 503 Service Unavailable
   - Unexpected errors → 500 with safe error message (no stack traces in prod)

4. **For development without real model**: create a mock mode
   - If model files don't exist, use a dummy model that returns random scores
   - Controlled by `APP_ENV=development` env var
   - This allows testing the API flow before Session A provides the model

Update `src/linkedin_lead_scoring/api/main.py`:
- Import predict router
- Add lifespan for model loading
- Update health check to report model_loaded status
- Add CORS middleware (needed for Streamlit dashboard to call API)

Commit: `feat: implement /predict endpoint with model loading at startup`

## Task B.3: Implement `/predict/batch` Endpoint

Add to predict.py:
1. **POST /predict/batch**: Accept BatchPredictionRequest (list of leads)
2. Process all leads, return BatchPredictionResponse
3. Include summary: total_count, avg_score, high_engagement_count
4. Limit batch size to 100 leads max

Commit: `feat: add batch prediction endpoint`

## Task B.4: Add Production Logging Middleware

Create `src/linkedin_lead_scoring/api/middleware.py`:

1. **Request logging middleware**:
   - Log: timestamp, method, path, status_code, response_time_ms
   - JSON structured format
   - Write to `logs/api_requests.jsonl` (append mode)

2. **Prediction logging** (in predict.py):
   - After each prediction, log: timestamp, input_features, predicted_score, inference_time_ms, model_version
   - Write to `logs/predictions.jsonl`
   - This data will be used by Session C for drift detection

3. Log format example:
   ```json
   {"timestamp": "2026-02-20T10:30:00Z", "input": {...}, "score": 0.73, "label": "engaged", "inference_ms": 12.5, "model_version": "0.3.0"}
   ```

Commit: `feat: add structured JSON logging for requests and predictions`

## Task B.5: Write Unit Tests

Create `tests/test_predict.py`:
1. Test valid prediction returns correct schema
2. Test missing required fields returns 422
3. Test out-of-range values (llm_quality > 100, negative engagement)
4. Test wrong types (string where number expected)
5. Test model not loaded returns 503
6. Test batch endpoint with multiple leads
7. Test batch size limit (>100 rejected)

Create `tests/test_schemas.py`:
1. Test LeadInput validation
2. Test LeadPrediction serialization
3. Test optional vs required fields

Update `tests/conftest.py`:
1. Add fixture for sample valid lead input
2. Add fixture for mock model (returns deterministic scores)
3. Keep existing client fixture

**Use mocks for the model** — don't require real model file in tests:
```python
@pytest.fixture
def mock_model(monkeypatch):
    """Mock model that returns deterministic predictions"""
    class FakeModel:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]] * len(X))
    # Patch the model loading
    ...
```

Commit: `test: add comprehensive unit tests for prediction API`

## Task B.6: Write Integration Tests

Update `tests/test_api_integration.py`:
1. Test full prediction flow (request → preprocess → predict → response)
2. Test health endpoint reports model status
3. Test Swagger docs include new endpoints
4. Test CORS headers present
5. Test logging creates files

Commit: `test: add integration tests for API endpoints`

## Task B.7: Error Handling & Input Validation

1. Add custom exception handlers in main.py
2. Ensure no stack traces leak in production (APP_ENV=production)
3. Add request ID to all responses for tracing
4. Add rate limiting info in headers

Commit: `feat: add error handling and request tracing`

---

## Final Step

Create PR from `feature/api-scoring` → `v0.3.0` with:
- Summary of all endpoints added
- Test coverage report
- List of dependencies needed (for Session A to add to requirements-prod.txt)
