---
title: OC6 Bizdev ML API - LinkedIn Lead Scoring
emoji: 🎯
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# OC6 — LinkedIn Lead Scoring with MLOps

**Smart lead qualification tool for business development professionals** — predict which LinkedIn contacts will likely respond to your invite requests, enabling data-driven outreach prioritization.

## Business Context

This tool helps business development and sales teams **score LinkedIn prospects** before reaching out:
- **Use case**: You maintain a list of potential leads on LinkedIn. Before sending connection requests or outreach messages, you want to know which contacts are most likely to respond (engage, view your profile, accept invites)
- **Input**: LinkedIn profile data (job title, company, seniority, industry, location, profile summary, skills)
- **Output**: Engagement probability score (0-1) with confidence level → label as "engaged" or "not_engaged"
- **Business impact**: Focus outreach effort on high-probability prospects, reducing wasted messages, improving conversion rates

**Example scenario**: You have 500 potential leads in your target market. Without scoring, you send 500 outreach messages with a 5% response rate (25 replies). By using this model to rank prospects, you send only to the top-200 highest-scoring contacts and achieve a 15% response rate (30 replies) using 40% fewer messages.

## Project Overview

This is a complete MLOps pipeline for predicting LinkedIn lead engagement:
- **MLflow experiment tracking** from data preparation through model training
- **Jupyter notebooks** for data exploration and model development
- **FastAPI REST API** with `/predict` and `/predict/batch` endpoints (live on staging)
- **Streamlit monitoring dashboard** with drift detection and performance metrics
- **Hybrid conda + uv** environment for package management
- **CI/CD pipeline** with GitHub Actions (lint, test, Docker, deploy)
- **Deployment** to Hugging Face Spaces (staging + production)

## Current Status (v0.3.1)

**299 tests passing | 50% coverage threshold | 93% production code coverage**

Latest release includes hotfixes for dashboard and API stability (merged 2026-02-25). All core features deployed and tested.

### Architecture

```
                     GitHub Actions CI/CD
                            |
              push to v0.3.1 (staging) / main (production)
                            |
                +-----------+-----------+
                |                       |
        HF Space (Docker)       HF Space (Streamlit)
        FastAPI Scoring API     Monitoring Dashboard
          port 7860               port 8501
                |                       |
        Supabase PostgreSQL      Evidently AI
        (prediction logging)     (drift detection)
```

### What's done

- **Data pipeline**: Preparation + LLM enrichment notebooks with full MLflow tracking
- **Model**: XGBoost classifier (Optuna-tuned, F1=0.556) — 1,909 contacts, 47 features
- **Export**: `scripts/export_model.py` produces joblib model, preprocessor, feature columns, numeric medians
- **API**: FastAPI with `/predict` (single lead), `/predict/batch` (up to 10,000), `/health`
- **Preprocessing**: Target encoding, one-hot encoding (deterministic), text feature extraction, median imputation
- **Database**: Async SQLAlchemy + Supabase PostgreSQL (prediction + API metric logging)
- **Monitoring**: Evidently AI drift detection, Streamlit dashboard, ONNX optimization (26.5x speedup)
- **CI/CD**: 5 GitHub Actions workflows (ci, staging, dashboard, security, production)
- **Tests**: 15 test files, 299 tests — covers API, features, schemas, drift, monitoring, ONNX, profiler, docs, pipelines
- **Validation**: `scripts/validate_pipeline.py` — confirms local model predictions match staging API

### Recent Merges (v0.3.1 → v0.3.1)

| PR | Feature | Status |
|----|---------|--------|
| #12 | Hotfixes: dashboard + API stability | ✅ Merged 2026-02-25 |
| #11 | v0.3.1 release: full MLOps pipeline | ✅ Merged 2026-02-24 |
| #10 | Test hardening: OHE determinism + validation | ✅ Merged 2026-02-24 |
| #9 | Security: Nuclei scan + E2E tests | ✅ Merged 2026-02-24 |
| #8 | Production deployment workflow | ✅ Merged 2026-02-24 |

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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/health` | GET | Health check (model status, version) |
| `/predict` | POST | Score a single lead — returns `{score, label, inference_ms}` |
| `/predict/batch` | POST | Score up to 10,000 leads in one call |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

**Example request:**
```bash
curl -X POST https://ghislaindelabie-oc6-bizdev-ml-api-staging.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"jobtitle": "CTO", "industry": "tech", "companysize": "11-50", "llm_quality": 80}'
```

**Example response:**
```json
{"score": 0.73, "label": "engaged", "inference_ms": 0.21, "model_version": "0.3.1"}
```

## Testing Examples

Try these fake LinkedIn profiles to test the API and see engagement scores. The model has learned patterns from business development outreach data.

### ✅ High Engagement Profile (Likely to respond)

**Scenario**: Senior executive at growing tech company, actively engaged, decision-maker
```bash
curl -X POST https://ghislaindelabie-oc6-bizdev-ml-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "jobtitle": "VP of Sales",
    "industry": "Computer Software",
    "companyindustry": "Software Development",
    "companysize": "201-500",
    "companytype": "Privately Held",
    "companyfoundedon": 2015,
    "location": "San Francisco, California, United States",
    "llm_seniority": "Senior",
    "llm_quality": 85,
    "llm_engagement": 0.85,
    "llm_decision_maker": 0.9,
    "llm_company_fit": 1,
    "llm_geography": "international_hub",
    "llm_business_type": "leaders",
    "languages": "English, French",
    "summary": "15+ years B2B SaaS sales leadership. Scaled revenue from $2M to $50M. Growth-focused, always open to strategic partnerships.",
    "skills": "Sales Leadership, SaaS, Revenue Growth, Team Building, Negotiation"
  }'
```

**Expected response:**
```json
{
  "score": 0.78,
  "label": "engaged",
  "confidence": "high",
  "model_version": "0.3.1",
  "inference_time_ms": 0.21
}
```

### ❌ Low Engagement Profile (Unlikely to respond)

**Scenario**: Entry-level individual contributor at large corporation, generic profile, limited fit
```bash
curl -X POST https://ghislaindelabie-oc6-bizdev-ml-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "jobtitle": "Junior Analyst",
    "industry": "Banking",
    "companyindustry": "Financial Services",
    "companysize": "10001+",
    "companytype": "Public Company",
    "companyfoundedon": 1975,
    "location": "Mumbai, Maharashtra, India",
    "llm_seniority": "Entry",
    "llm_quality": 35,
    "llm_engagement": 0.2,
    "llm_decision_maker": 0.1,
    "llm_company_fit": 2,
    "llm_geography": "other",
    "llm_business_type": "workers",
    "languages": "English",
    "summary": "Working in finance.",
    "skills": "Excel, Data Entry"
  }'
```

**Expected response:**
```json
{
  "score": 0.22,
  "label": "not_engaged",
  "confidence": "high",
  "model_version": "0.3.1",
  "inference_time_ms": 0.21
}
```

### 🔶 Medium Engagement Profile (Borderline case)

**Scenario**: Mid-level manager at growing company, some alignment but not perfect fit
```bash
curl -X POST https://ghislaindelabie-oc6-bizdev-ml-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "jobtitle": "Product Manager",
    "industry": "Information Technology & Services",
    "companyindustry": "Software Development",
    "companysize": "51-200",
    "companytype": "Privately Held",
    "companyfoundedon": 2018,
    "location": "Paris, Île-de-France, France",
    "llm_seniority": "Mid",
    "llm_quality": 65,
    "llm_engagement": 0.55,
    "llm_decision_maker": 0.45,
    "llm_company_fit": 1,
    "llm_geography": "international_hub",
    "llm_business_type": "experts",
    "languages": "French, English, Spanish",
    "summary": "Product Manager with 8 years experience in SaaS. Interested in process optimization and team collaboration.",
    "skills": "Product Management, SaaS, User Research, Analytics, Agile"
  }'
```

**Expected response:**
```json
{
  "score": 0.52,
  "label": "engaged",
  "confidence": "medium",
  "model_version": "0.3.1",
  "inference_time_ms": 0.21
}
```

### Batch Testing (Multiple Leads)

Test multiple leads in a single API call:
```bash
curl -X POST https://ghislaindelabie-oc6-bizdev-ml-api.hf.space/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "leads": [
      {
        "jobtitle": "CTO",
        "industry": "Technology - SaaS",
        "companysize": "11-50",
        "llm_quality": 90,
        "llm_engagement": 0.9
      },
      {
        "jobtitle": "HR Administrator",
        "industry": "Retail",
        "companysize": "5001-10000",
        "llm_quality": 25,
        "llm_engagement": 0.15
      },
      {
        "jobtitle": "Sales Manager",
        "industry": "Real Estate",
        "companysize": "201-500",
        "llm_quality": 70,
        "llm_engagement": 0.65
      }
    ]
  }'
```

**Expected response:**
```json
{
  "predictions": [
    {"score": 0.85, "label": "engaged", "confidence": "high", "model_version": "0.3.1", "inference_time_ms": 0.21},
    {"score": 0.18, "label": "not_engaged", "confidence": "high", "model_version": "0.3.1", "inference_time_ms": 0.19},
    {"score": 0.68, "label": "engaged", "confidence": "high", "model_version": "0.3.1", "inference_time_ms": 0.20}
  ],
  "total_count": 3,
  "avg_score": 0.57,
  "high_engagement_count": 2
}
```

## Project Structure

```
oc6-linkedin-lead-scoring/
├── src/linkedin_lead_scoring/        # Main package
│   ├── api/                          # FastAPI application
│   │   ├── main.py                   # App entry point, CORS, exception handlers
│   │   ├── predict.py                # /predict and /predict/batch endpoints
│   │   ├── schemas.py                # Pydantic request/response models
│   │   └── middleware.py             # Request logging, rate limits, tracing
│   ├── monitoring/                   # Monitoring & optimization
│   │   ├── drift.py                  # Evidently AI drift detection (DriftDetector)
│   │   ├── dashboard_utils.py        # Log parsing, metrics, simulation
│   │   ├── onnx_optimizer.py         # XGBoost-to-ONNX conversion + benchmarks
│   │   └── profiler.py               # Inference timing (cProfile, perf_counter)
│   ├── features.py                   # Feature engineering & preprocessing
│   ├── data/                         # Data processing (LLM enrichment, utils)
│   └── db/                           # Async DB layer (SQLAlchemy + Supabase)
├── scripts/
│   ├── export_model.py               # Re-train & export production artifacts
│   ├── validate_pipeline.py          # Local-vs-API prediction comparison
│   ├── optimize_model.py             # ONNX conversion + benchmark
│   └── profile_api.py               # API load testing
├── model/                            # Committed production artifacts
│   ├── xgboost_model.joblib          # Trained XGBoost classifier (47 features)
│   ├── preprocessor.joblib           # Fitted TargetEncoder pipeline
│   ├── feature_columns.json          # Ordered feature column names
│   └── numeric_medians.json          # Median values for imputation
├── data/
│   ├── processed/linkedin_leads_clean.csv  # Training data (1,909 rows)
│   └── reference/training_reference.csv    # Drift detection baseline
├── streamlit_app.py                  # Monitoring dashboard entry point
├── notebooks/                        # Jupyter notebooks with MLflow tracking
├── tests/                            # Test suite (15 files, 299 tests)
├── .github/workflows/                # CI/CD (ci, staging, production, dashboard, security)
├── Dockerfile                        # API container for HF Spaces
├── requirements-prod.txt             # Production API dependencies
├── requirements-streamlit.txt        # Monitoring dashboard dependencies
├── environment.yml                   # Conda environment (scientific packages)
└── pyproject.toml                    # Project metadata & dev dependencies
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
python -m pytest tests/ -v --tb=short

# Run with coverage
python -m pytest tests/ --cov=src/linkedin_lead_scoring --cov-report=term-missing

# Validate model pipeline (local predictions match API)
python scripts/validate_pipeline.py --local-only
```

**Current status**: 299 tests, 55% overall coverage (93% on production code — two training-only data modules account for the gap).

## Deployment

Two-stage deployment via GitHub Actions:

| Environment | Trigger | API URL | Dashboard URL |
|-------------|---------|---------|---------------|
| **Staging** | Push to `v0.3.1` | [staging API](https://ghislaindelabie-oc6-bizdev-ml-api-staging.hf.space) | [staging dashboard](https://ghislaindelabie-oc6-bizdev-monitoring-staging.hf.space) |
| **Production** | Push to `main` | [production API](https://ghislaindelabie-oc6-bizdev-ml-api.hf.space) | [production dashboard](https://ghislaindelabie-oc6-bizdev-monitoring.hf.space) |

**Promotion flow:** `feature/* --> v0.3.1 (staging) --> main (production)`

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
  - 299 tests across 15 test files (unit, integration, structural)
  - Ruff linting on all source and test files
  - CI/CD pipeline validates before every deployment

- **CI/CD Pipeline (5 workflows):**
  - **`ci.yml`**: every push/PR — ruff lint, pytest with coverage gate, Docker build check
  - **`staging.yml`**: push to `v*.*.*` — deploy API + dashboard to staging HF Spaces
  - **`production.yml`**: push to `main` — deploy API + dashboard to production HF Spaces
  - **`dashboard.yml`**: deploy Streamlit monitoring dashboard on push to main
  - **`security.yml`**: weekly pip-audit (dependency CVEs) + bandit (static analysis)

- **Production Logging:**
  - Async SQLAlchemy + Supabase PostgreSQL for prediction and API metric logging
  - Tables: `prediction_logs` (score, features, inference time) + `api_metrics` (endpoint, status, latency)
  - Local dev falls back to SQLite automatically (no setup needed)
  - Alembic migrations in `alembic/` — run `alembic upgrade head` before first deploy

- **Model Monitoring:**
  - Evidently AI drift detection (data drift + prediction score drift)
  - Streamlit monitoring dashboard (`streamlit_app.py`) with live metrics
  - ONNX Runtime optimization: 26.5x inference speedup over joblib
  - Performance profiling with cProfile and tracemalloc
  - See `docs/MONITORING_GUIDE.md` and `docs/PERFORMANCE_REPORT.md`

## License

MIT License - see LICENSE file

## About

**Project:** OpenClassrooms OC6 - MLOps
**Purpose:** Business development tool for LinkedIn lead scoring and engagement prediction
