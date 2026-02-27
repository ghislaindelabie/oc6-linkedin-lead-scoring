---
title: OC6 Bizdev ML API - LinkedIn Lead Scoring
emoji: 🎯
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# LinkedIn Lead Scoring API

**Predict which LinkedIn contacts will engage with your outreach** — a REST API that scores prospects based on profile data, enabling data-driven lead prioritization for business development.

## What it does

You send LinkedIn profile data (job title, industry, seniority, company info...) and get back:
- **Engagement score** (0.0 to 1.0)
- **Label**: `engaged` or `not_engaged` (threshold: 0.5)
- **Confidence**: `low` / `medium` / `high`
- **Inference time** in milliseconds

The model is an XGBoost classifier trained on 303 real LemList campaign contacts enriched with LLM-generated features, achieving F1=0.556 on cross-validation.

## API Reference

**Base URLs:**
- Production: `https://ghislaindelabie-oc6-bizdev-ml-api.hf.space`
- Staging: `https://ghislaindelabie-oc6-bizdev-ml-api-staging.hf.space`
- Local: `http://localhost:7860`

### `POST /predict` — Score a single lead

**Request body** (all fields optional — missing fields use median/default values):

| Field | Type | Description |
|-------|------|-------------|
| `jobtitle` | string | Current job title |
| `industry` | string | LinkedIn industry label |
| `companyindustry` | string | Company industry |
| `companysize` | string | Size range: `1-10`, `11-50`, `51-200`, `201-500`, `501-1000`, `1001-5000`, `5001-10000`, `10001+` |
| `companytype` | string | `Privately Held`, `Public Company`, `Nonprofit`, etc. |
| `companyfoundedon` | number | Founding year (e.g. 2015) |
| `location` | string | Profile location (City, Region, Country) |
| `companylocation` | string | Company location |
| `languages` | string | Comma-separated languages |
| `summary` | string | Professional summary text |
| `skills` | string | Comma-separated skills |
| `llm_quality` | integer | Profile quality score (0-100) |
| `llm_engagement` | float | Engagement likelihood (0.0-1.0) |
| `llm_decision_maker` | float | Decision maker probability (0.0-1.0) |
| `llm_company_fit` | integer | Company fit (0, 1, or 2) |
| `llm_seniority` | string | `Entry`, `Mid`, `Senior`, `Executive`, `C-Level` |
| `llm_industry` | string | LLM-inferred industry |
| `llm_geography` | string | `international_hub`, `regional_hub`, `other` |
| `llm_business_type` | string | `leaders`, `experts`, `salespeople`, `workers`, `others` |

**Example:**
```bash
curl -X POST https://ghislaindelabie-oc6-bizdev-ml-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "jobtitle": "VP of Sales",
    "industry": "Computer Software",
    "companysize": "201-500",
    "llm_quality": 85,
    "llm_engagement": 0.85,
    "llm_decision_maker": 0.9,
    "llm_seniority": "Senior"
  }'
```

**Response:**
```json
{
  "score": 0.78,
  "label": "engaged",
  "confidence": "high",
  "model_version": "0.3.0",
  "inference_time_ms": 0.21
}
```

### `POST /predict/batch` — Score multiple leads (up to 10,000)

**Request body:**
```json
{
  "leads": [
    {"jobtitle": "CTO", "llm_quality": 90, "llm_engagement": 0.9},
    {"jobtitle": "Junior Analyst", "llm_quality": 25, "llm_engagement": 0.15}
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {"score": 0.85, "label": "engaged", "confidence": "high", "model_version": "0.3.0", "inference_time_ms": 0.21},
    {"score": 0.18, "label": "not_engaged", "confidence": "high", "model_version": "0.3.0", "inference_time_ms": 0.19}
  ],
  "total_count": 2,
  "avg_score": 0.52,
  "high_engagement_count": 1
}
```

### `GET /health` — Health check

```json
{"status": "healthy", "service": "linkedin-lead-scoring-api", "version": "0.3.0", "model_loaded": true}
```

### `GET /docs` — Swagger UI (interactive API documentation)

## n8n / Automation Integration

This API is designed for integration with n8n, Make, or any HTTP-capable automation tool.

### n8n HTTP Request node setup

1. **Method**: POST
2. **URL**: `https://ghislaindelabie-oc6-bizdev-ml-api.hf.space/predict`
3. **Body Content Type**: JSON
4. **Body**: Map your LinkedIn data fields to the API schema above

### Minimal payload (works with just a few fields)

The API handles missing fields gracefully — you can send as little or as much data as you have:

```json
{"jobtitle": "CTO", "industry": "Technology", "companysize": "51-200"}
```

### Batch scoring workflow

For bulk scoring (e.g., from a CSV or CRM export), use `/predict/batch`:
1. Collect up to 10,000 leads in an array
2. POST to `/predict/batch` with `{"leads": [...]}`
3. Parse the `predictions` array — each entry has `score`, `label`, `confidence`
4. Filter or sort by score to prioritize outreach

### Confidence levels

| Score range | Confidence | Label | Action |
|-------------|------------|-------|--------|
| 0.70 - 1.00 | high | engaged | Prioritize outreach |
| 0.40 - 0.69 | medium | engaged/not_engaged | Review manually |
| 0.00 - 0.39 | low | not_engaged | Skip or deprioritize |

## Local Development

### Setup

```bash
# Clone and setup
git clone https://github.com/ghislaindelabie/oc6-linkedin-lead-scoring.git
cd oc6-linkedin-lead-scoring

# Option 1: Automated
bash setup_env.sh

# Option 2: Manual
conda env create -f environment.yml
conda activate oc6
uv pip install -e ".[dev]"
```

### Run locally

```bash
# API (Terminal 1)
conda activate oc6
PYTHONPATH=src uvicorn linkedin_lead_scoring.api.main:app --port 7860

# Monitoring dashboard (Terminal 2)
conda activate oc6
streamlit run streamlit_app.py

# Open:
#   API Swagger UI:  http://localhost:7860/docs
#   Dashboard:       http://localhost:8501
```

### Run tests

```bash
conda activate oc6
python -m pytest tests/ -v --tb=short        # 377 tests
python -m pytest tests/ --cov=src/linkedin_lead_scoring --cov-report=term-missing
```

## Architecture

```
                     GitHub Actions CI/CD
                            |
              push to v0.3.x (staging) / main (production)
                            |
                +-----------+-----------+
                |                       |
        HF Space (Docker)       HF Space (Streamlit)
        FastAPI Scoring API     Monitoring Dashboard
          port 7860               port 8501
                |                       |
        logs/predictions.jsonl   Evidently AI
        (prediction logging)     (drift detection)
```

### Deployment

| Environment | Trigger | API | Dashboard |
|-------------|---------|-----|-----------|
| **Staging** | Push to `v0.3.x` | [API](https://ghislaindelabie-oc6-bizdev-ml-api-staging.hf.space) | [Dashboard](https://ghislaindelabie-oc6-bizdev-monitoring-staging.hf.space) |
| **Production** | Push to `main` | [API](https://ghislaindelabie-oc6-bizdev-ml-api.hf.space) | [Dashboard](https://ghislaindelabie-oc6-bizdev-monitoring.hf.space) |

**Promotion flow:** `feature/* --> v0.3.x (staging) --> main (production)`

## Project Structure

```
oc6-linkedin-lead-scoring/
├── src/linkedin_lead_scoring/        # Main package
│   ├── api/                          # FastAPI (main.py, predict.py, schemas.py, middleware.py)
│   ├── monitoring/                   # Drift detection, dashboard utils, ONNX optimizer, profiler
│   ├── db/                           # Async DB layer (SQLAlchemy + Supabase)
│   ├── data/                         # Data processing (LLM enrichment)
│   └── features.py                   # Feature engineering & preprocessing
├── model/                            # Production model artifacts (v1)
│   ├── xgboost_model.joblib          # Trained XGBoost classifier (47 features)
│   ├── preprocessor.joblib           # Fitted TargetEncoder pipeline
│   ├── feature_columns.json          # Ordered feature column names
│   └── numeric_medians.json          # Median values for imputation
├── model_v2/                         # Retrained model artifacts (experimental)
├── notebooks/                        # Jupyter notebooks with MLflow tracking
│   ├── 01_linkedin_data_prep.ipynb   # Data preparation
│   ├── 02_linkedin_model_training.ipynb  # Model training (Optuna)
│   ├── 03_performance_analysis.ipynb # Inference profiling (joblib vs ONNX)
│   └── 04_drift_monitoring_analysis.ipynb  # Drift analysis & retraining
├── scripts/                          # CLI tools (export, validate, optimize, profile, retrain)
├── tests/                            # 377 tests (API, features, schemas, drift, monitoring, pipelines)
├── docs/                             # Technical documentation
├── streamlit_app.py                  # Monitoring dashboard entry point
├── .github/workflows/                # CI/CD (5 workflows)
├── Dockerfile                        # API container
├── Dockerfile.streamlit              # Dashboard container
└── .old/                             # Archived planning documents
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Decisions](docs/ARCHITECTURE_DECISIONS.md) | 16 ADRs explaining key design choices |
| [Monitoring Guide](docs/MONITORING_GUIDE.md) | Dashboard usage, alert thresholds, commands |
| [Data Drift Guide](docs/DATA_DRIFT_GUIDE.md) | Drift theory, detection methods, Evidently integration |
| [Performance Report](docs/PERFORMANCE_REPORT.md) | joblib vs ONNX benchmarks (26.5x speedup) |
| [Top-K Precision Guide](docs/TOP_K_PRECISION_PRODUCTION_GUIDE.md) | Production lead prioritization strategy |
| [Known Issues](KNOWN_ISSUES.md) | Current limitations and recommended fixes |
| [Version History](VERSION_HISTORY.md) | Full changelog from v0.1.0 to v0.3.2 |

## MLOps Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | XGBoost + scikit-learn |
| Experiment Tracking | MLflow (filesystem) |
| API | FastAPI + uvicorn |
| Monitoring | Evidently AI + Streamlit |
| Database | Supabase PostgreSQL (async SQLAlchemy) |
| CI/CD | GitHub Actions (lint, test, Docker, deploy) |
| Deployment | Hugging Face Spaces (Docker) |
| Performance | ONNX Runtime (optional, 26.5x speedup) |

## License

MIT License — see LICENSE file.
