# Session A — Infrastructure, CI/CD & Model Export

**Branch**: `feature/infra-cicd`
**Worktree**: `worktrees/session-a`
**Role**: Infrastructure, deployment pipeline, model artifact management, database setup

---

## IMPORTANT RULES

1. **Work ONLY in this worktree directory** — never `cd` outside it
2. **Push ONLY to `feature/infra-cicd`** — never push to main or v0.3.0
3. **Own ONLY these files**: `.github/`, `Dockerfile`, `requirements-prod.txt`, `model/`, `scripts/`, `setup_env.sh`, `src/linkedin_lead_scoring/db/`, `alembic/`, `.env.example`
4. **Do NOT touch**: `src/linkedin_lead_scoring/api/` (Session B), `src/linkedin_lead_scoring/monitoring/` (Session C)
5. Run `python -m pytest tests/ -v --tb=short` before every commit
6. Read `CLAUDE.md` and `SESSION_COORDINATION.md` for full context

---

## Task A.1: Fix Test Infrastructure (FIRST — unblocks everything)

The tests are broken due to missing editable install.

1. Ensure `pyproject.toml` has correct build config
2. Add `joblib` to dependencies in `pyproject.toml` (needed for model export)
3. Update `requirements-prod.txt` to add `joblib>=1.3.0`
4. Run `pip install -e ".[dev]"` then `python -m pytest tests/ -v --tb=short`
5. Fix any remaining import issues
6. Commit: `fix: resolve test infrastructure and add joblib dependency`

## Task A.2: Export Model to `model/` Directory (CRITICAL — unblocks B and C)

The best trained model (XGBoost, F1=0.556) exists in MLflow filesystem tracking but needs to be exported as a standalone artifact.

Create `scripts/export_model.py` that:
1. Loads the best model from MLflow runs at `mlruns/796258469850849262/`
2. The best run is `XGBoost_Optuna_Tuning` with run_id `03c1b76a8dfc4d2399bdf4715996079d`
3. However this run doesn't have model.pkl in its artifacts — the model pkls are in `mlruns/796258469850849262/models/m-*/artifacts/model.pkl`
4. Alternative approach: create a script that re-trains the best model with the known best hyperparameters and exports it:
   - Parameters: n_estimators=255, max_depth=3, learning_rate=0.121, min_child_weight=7, subsample=0.784, colsample_bytree=0.988, gamma=3.513, scale_pos_weight=2.501
   - Save as `model/xgboost_model.joblib`
   - Save feature columns as `model/feature_columns.json`
   - Save a reference data sample as `data/reference/training_reference.csv` (first 100 rows of training data, for drift detection)
5. The cleaned data is at `data/processed/linkedin_leads_clean.csv` (303 rows, 20 columns)
6. Feature columns (19 features + 1 target `engaged`):
   - Numeric: llm_quality (int), llm_engagement (float), llm_decision_maker (float), llm_company_fit (int), companyfoundedon (float)
   - Categorical: llm_seniority, llm_industry, llm_geography, llm_business_type, industry, companyindustry, companysize, companytype, languages, location, companylocation
   - Text (need encoding): summary, skills, jobtitle
   - Target: engaged (int, binary 0/1)
7. The script must handle encoding (the notebook used target encoding + one-hot)
8. Also save the preprocessing pipeline (encoder) as `model/preprocessor.joblib`
9. Register the model in MLflow Model Registry

Commit: `feat: add model export script with preprocessing pipeline`

## Task A.3: Set Up Supabase PostgreSQL

1. Create `src/linkedin_lead_scoring/db/__init__.py`
2. Create `src/linkedin_lead_scoring/db/connection.py` — async SQLAlchemy engine with Supabase
3. Create `src/linkedin_lead_scoring/db/models.py` — SQLAlchemy models for:
   - `prediction_logs` table: id, timestamp, input_features (JSON), predicted_score, inference_time_ms, model_version
   - `api_metrics` table: id, timestamp, endpoint, status_code, response_time_ms
4. Create `src/linkedin_lead_scoring/db/repository.py` — CRUD operations
5. Update `.env.example` with Supabase connection string placeholder
6. Use `alembic/` for migrations
7. Connection URL read from `DATABASE_URL` env var, with fallback to SQLite for local dev

Commit: `feat: add Supabase PostgreSQL integration for production logging`

## Task A.4: Update Dockerfile for Real Model

1. Update `Dockerfile` to copy `model/` directory with actual model files
2. Ensure `requirements-prod.txt` includes all needed production deps
3. Add healthcheck that verifies model is loaded
4. Test Docker build locally: `docker build -t oc6-api .`
5. Test Docker run: `docker run -p 7860:7860 oc6-api`

Commit: `feat: update Dockerfile for production model serving`

## Task A.5: Enhance CI/CD Pipeline

Update `.github/workflows/ci.yml` to:
1. **Test job**: enforce coverage minimum (70%), add lint step
2. **Docker build job**: build and verify Docker image (on PRs)
3. **Deploy job**: push to HF Spaces (on main only) — keep existing logic
4. **Security scan job** (new): run basic security checks (pip-audit, bandit)
5. Add separate workflow for Streamlit dashboard deployment to second HF Space

Create `.github/workflows/security.yml`:
- Scheduled weekly Nuclei scan of deployed API (conservative mode)
- Reference patterns from `ghislaindelabie/datastreaming-monitoring` repo

Commit: `feat: enhance CI/CD with Docker build, security scan, and coverage enforcement`

## Task A.6: Test End-to-End Deployment

1. Verify CI/CD passes on the feature branch
2. Test Docker image locally
3. Verify HF Spaces deployment config is correct
4. Create PR from `feature/infra-cicd` → `v0.3.0`

---

## Dependency Notes

- After A.2 completes, notify Session B (they need model/feature_columns.json for API schemas)
- After A.3 completes, Session B can integrate DB logging in the API
- Session C needs `data/reference/training_reference.csv` from A.2 for drift detection
