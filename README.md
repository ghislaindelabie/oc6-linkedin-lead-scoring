---
title: OC6 Bizdev ML API - LinkedIn Lead Scoring
emoji: 🎯
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# OC6 — LinkedIn Lead Scoring with MLOps

ML pipeline for predicting LinkedIn contact engagement (reply/interest) with complete MLflow tracking.

## Project Overview

This project implements a complete MLOps pipeline for predicting LinkedIn lead engagement:
- **MLflow experiment tracking** from data preparation through model training
- **Jupyter notebooks** for data exploration and model development
- **FastAPI REST API** for lead scoring (skeleton deployed)
- **Hybrid conda + uv** environment for package management
- **CI/CD pipeline** with GitHub Actions
- **Deployment** to Hugging Face Spaces

## Current Status (v0.3.0-dev)

✅ **Completed**:
- Data preparation notebook with MLflow tracking
- Model training notebook (baseline + tree models + Optuna tuning)
- Hybrid environment setup (conda for scientific packages, uv for ML packages)
- FastAPI skeleton deployed to HF Spaces
- Production model export pipeline (`scripts/export_model.py`)
- Model artifacts committed: XGBoost model, preprocessor, feature columns (47)
- Drift detection reference dataset (`data/reference/training_reference.csv`, 100 rows)
- Async PostgreSQL DB layer (SQLAlchemy 2.0 + asyncpg + Alembic migrations)
- Docker updated for production: model loading, alembic, health check, correct module path
- CI/CD: three GitHub Actions workflows (lint+test+deploy, security scan, dashboard deploy)

🚧 **In Progress** (v0.3.0 parallel sessions):
- Session A: PR creation (A.1–A.5 done: deps, model export, DB, Docker, CI/CD)
- Session B: Prediction API endpoint (`/predict`, `/batch-predict`)
- Session C: Monitoring dashboard, drift detection

📋 **Planned**:
- LemList API integration for data collection
- Automated retraining pipeline

## Quickstart

### Setup Environment (Conda + uv Hybrid)

```bash
# Clone repository
git clone https://github.com/ghislaindelabie/oc6-linkedin-lead-scoring.git
cd oc6-linkedin-lead-scoring

# Option 1: Automated setup
bash setup_env.sh

# Option 2: Manual setup
conda env create -f environment.yml
conda activate oc6
uv pip install -e ".[dev]"

# Verify installation
python -c "import mlflow, xgboost, sklearn; print('✓ All packages ready!')"
```

See `SETUP_ENVIRONMENT.md` for detailed setup instructions.

### Run Notebooks

```bash
# Start MLflow UI (in terminal 1)
conda activate oc6
mlflow ui --port 5000

# Start Jupyter Lab (in terminal 2)
conda activate oc6
jupyter lab

# Open notebooks in notebooks/ directory
# 01_linkedin_data_prep.ipynb - Data preparation
# 02_linkedin_model_training.ipynb - Model training
```

### Run API Locally

```bash
conda activate oc6
uvicorn linkedin_lead_scoring.api.main:app --reload

# View at http://localhost:8000/docs
```

### Run Tests

```bash
conda activate oc6
pytest
pytest --cov=src/linkedin_lead_scoring --cov-report=term-missing
```

## API Endpoints

- `GET /` - Landing page
- `GET /health` - Health check
- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation

See `/docs` for detailed API schema.

## Project Structure

```
oc6-linkedin-lead-scoring/
├── src/linkedin_lead_scoring/        # Main package
│   ├── api/                          # FastAPI application
│   │   ├── main.py                   # API entry point
│   │   └── schemas.py                # Pydantic request/response models
│   ├── data/                         # Data processing utilities
│   ├── db/                           # Database layer (Supabase/SQLAlchemy)
│   ├── models/                       # Training & evaluation
│   └── utils/                        # MLflow helpers
├── scripts/
│   └── export_model.py               # Re-train & export production artifacts
├── model/                            # Committed production artifacts
│   ├── xgboost_model.joblib          # Trained XGBoost classifier (47 features)
│   ├── preprocessor.joblib           # Fitted TargetEncoder pipeline
│   └── feature_columns.json          # Ordered feature column names
├── data/
│   └── reference/
│       └── training_reference.csv    # 100-row baseline for drift detection
├── notebooks/                        # Jupyter notebooks with MLflow tracking
│   ├── 01_linkedin_data_prep.ipynb   # Data preparation & feature engineering
│   └── 02_linkedin_model_training.ipynb  # Model training & optimization
├── tests/                            # Test suite (pytest)
├── .github/workflows/                # CI/CD pipelines
├── environment.yml                   # Conda environment (scientific packages)
├── pyproject.toml                    # Project dependencies
├── requirements-prod.txt             # Pinned production dependencies
├── Dockerfile                        # Container for HF Spaces deployment
├── setup_env.sh                      # Automated environment setup script
└── README.md                         # This file
```

## Development Workflow

This project follows **Git Flow** with semantic versioning:
- `main` - Production-ready code (auto-deploys to HF Spaces)
- `release/X.Y.0` - Release preparation
- `feature/*` - Feature development
- `hotfix/*` - Emergency fixes

See `BRANCHING_STRATEGY.md` for detailed workflow.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/linkedin_lead_scoring --cov-report=term-missing

# Run specific test type
pytest -m integration
```

**Current test coverage:** Target 75%+

## Deployment

Automatic deployment to HF Spaces on push to `main` branch (after tests pass).

**Live API:** [https://ghislaindelabie-oc6-bizdev-ml-api.hf.space](https://ghislaindelabie-oc6-bizdev-ml-api.hf.space)

## MLOps Features

- **Experiment Tracking:**
  - MLflow tracking integrated from data preparation through model training
  - Automatic project root detection for centralized tracking
  - All data operations, model training, and hyperparameter tuning logged
  - Model registry ready for production deployment

- **Environment Management:**
  - Hybrid conda + uv approach for optimal package management
  - Conda: Scientific packages (numpy, pandas, scikit-learn, jupyter)
  - uv: Specialized ML packages (mlflow, xgboost, fastapi, optuna)
  - Automated setup script for reproducibility

- **Automated Testing:**
  - pytest with 75%+ coverage requirement
  - Integration and unit tests
  - CI/CD pipeline validates before deployment

- **CI/CD Pipeline:**
  - **`ci.yml`**: runs on every push/PR — ruff lint, pytest with 70% coverage gate, Docker build check (PRs only), deploy to HF Spaces API on main
  - **`security.yml`**: weekly pip-audit (dependency CVEs) + bandit (static analysis), results uploaded as artifacts
  - **`dashboard.yml`**: deploys Streamlit monitoring dashboard to `oc6-bizdev-monitoring` HF Space on push to main

- **Production Logging:**
  - Async SQLAlchemy + Supabase PostgreSQL for prediction and API metric logging
  - Tables: `prediction_logs` (score, features, inference time) + `api_metrics` (endpoint, status, latency)
  - Local dev falls back to SQLite automatically (no setup needed)
  - Alembic migrations in `alembic/` — run `alembic upgrade head` before first deploy

- **Model Monitoring:** (Session C — in progress)
  - Drift detection with Evidently AI
  - Streamlit dashboard for monitoring

## License

MIT License - see LICENSE file

## Contact

**Author:** Ghislain Delabie
**Email:** ghislain@delabie.tech
**Project:** OpenClassrooms OC6 - MLOps
