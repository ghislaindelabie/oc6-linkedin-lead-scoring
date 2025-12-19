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

## Current Status (v0.2.0-dev)

✅ **Completed**:
- Data preparation notebook with MLflow tracking
- Model training notebook (baseline + tree models + Optuna tuning)
- Hybrid environment setup (conda for scientific packages, uv for ML packages)
- FastAPI skeleton (v0.1.0 deployed to HF Spaces)

🚧 **In Progress**:
- Model validation and performance testing
- Feature engineering enhancements
- Production model deployment

📋 **Planned**:
- LemList API integration for data collection
- Automated retraining pipeline
- Model monitoring and drift detection

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
│   │   └── static/                   # Static files for landing page
│   ├── data/                         # Data processing
│   │   └── utils_data.py             # MLflow-integrated data utilities
│   ├── models/                       # Training & evaluation (planned)
│   └── utils/                        # MLflow helpers (planned)
├── notebooks/                        # Jupyter notebooks with MLflow tracking
│   ├── 01_linkedin_data_prep.ipynb   # Data preparation & feature engineering
│   └── 02_linkedin_model_training.ipynb  # Model training & optimization
├── tests/                            # Test suite (pytest)
├── data/                             # Raw data (not tracked in git)
├── mlruns/                           # MLflow tracking data (not tracked in git)
├── docs/                             # Documentation
│   ├── PROJECT_SUMMARY.md            # Complete implementation guide
│   ├── SETUP_ENVIRONMENT.md          # Environment setup instructions
│   └── BRANCHING_STRATEGY.md         # Git workflow
├── environment.yml                   # Conda environment (scientific packages)
├── pyproject.toml                    # uv dependencies (ML packages)
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
  - GitHub Actions for automated testing and deployment
  - Auto-deploy to Hugging Face Spaces on merge to main
  - Git Flow branching strategy for organized releases

- **Model Monitoring:** (Planned)
  - Database logging for predictions (GDPR-compliant)
  - Drift detection and performance monitoring

## License

MIT License - see LICENSE file

## Contact

**Author:** Ghislain Delabie
**Email:** ghislain@delabie.tech
**Project:** OpenClassrooms OC6 - MLOps
