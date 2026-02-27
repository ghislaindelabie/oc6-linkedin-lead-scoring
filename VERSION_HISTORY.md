# Version History

## v0.3.2 (2026-02-27) — Drift Analysis & Retraining Pipeline

### Added
- Notebook 04: drift monitoring analysis with real production data
- Retraining pipeline (v1 vs v2 comparison) with stratified K-fold cross-validation
- Holdout evaluation methodology (70/30 split) preventing data leakage
- Synthetic drift scenario generator (`scripts/generate_drift_scenarios.py`)
- New contact preparation script (`scripts/prepare_new_contacts.py`)
- Retraining script (`scripts/retrain_model.py`)
- KNOWN_ISSUES.md documenting target encoding and companysize limitations
- Performance profiling reports from notebook 03
- Hardened Dockerfile.streamlit and CORS configuration

### Fixed
- Data leakage in v2 evaluation (was training and evaluating on same 151 contacts)
- Ruff E741 lint error in test_dockerfile.py

### Model Results (v0.3.2)
- v1 (original, 303 contacts): CV F1=0.556, holdout F1=0.216
- v2 (retrained, +105 contacts): CV F1=0.530, holdout F1=0.571
- Drift reduced from 62% to 31% of features after retraining

---

## v0.3.1 (2026-02-25) — Stability & Documentation

### Added
- Business context and ROI scenario in README
- Testing examples (high/medium/low engagement profiles)
- Comprehensive documentation (architecture decisions, monitoring guide)

### Fixed
- Dashboard UnhashableParamError (Streamlit caching issue)
- API unreachable status on deployed dashboard
- Conditional conftest imports for lightweight CI

---

## v0.3.0 (2026-02-24) — Full MLOps Pipeline

### Added
- **API**: FastAPI with `/predict`, `/predict/batch`, `/health` endpoints
- **Monitoring**: Evidently AI drift detection + Streamlit dashboard
- **CI/CD**: 5 GitHub Actions workflows (ci, staging, production, dashboard, security)
- **Database**: Async SQLAlchemy + Supabase PostgreSQL (prediction logging)
- **ONNX**: Optional ONNX Runtime optimization (26.5x inference speedup)
- **Security**: Nuclei scanner, Playwright E2E tests, pip-audit, bandit
- **Tests**: 299 tests across 15 files (API, features, schemas, drift, monitoring)
- Production deployment workflow to Hugging Face Spaces
- Test hardening: OHE determinism, pipeline validation

### Infrastructure
- Docker containers for API and dashboard (separate HF Spaces)
- Staging + production environments with auto-deploy
- Pre-push hooks for branch protection

---

## v0.2.0 (2026-01-15) — MLflow Notebooks & Data Pipeline

### Added
- Notebook 01: data preparation with MLflow tracking
- Notebook 02: model training with Optuna hyperparameter tuning
- LLM enrichment pipeline (OpenAI GPT-4o-mini)
- Feature engineering: target encoding, one-hot encoding, text features
- Model export script with reproducible artifacts
- MLflow experiment tracking (filesystem mode)

### Model
- XGBoost classifier: F1=0.556, 47 features, 303 training contacts
- Optuna tuning: 100 trials, 5-fold stratified CV
- Artifacts: joblib model, preprocessor, feature columns, numeric medians

---

## v0.1.0 (2025-11-28) — Initial Project Setup

### Added
- Project scaffolding with pyproject.toml
- Conda + uv hybrid environment (environment.yml)
- Basic project structure (src layout)
- Initial data exploration
- Git repository with branching strategy
