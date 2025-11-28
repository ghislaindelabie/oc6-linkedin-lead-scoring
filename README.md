---
title: OC6 Bizdev ML API - LinkedIn Lead Scoring
emoji: 🎯
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# OC6 — LinkedIn Lead Scoring with MLOps

Production-ready ML deployment for LinkedIn lead engagement prediction.

## Project Overview

This project implements a complete MLOps pipeline for predicting LinkedIn lead engagement:
- MLflow experiment tracking and model versioning
- FastAPI REST API for lead scoring
- CI/CD pipeline with GitHub Actions
- Deployment to Hugging Face Spaces

## Quickstart

```bash
# Clone repository
git clone https://github.com/ghislaindelabie/oc6-linkedin-lead-scoring.git
cd oc6-linkedin-lead-scoring

# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-prod.txt

# Run API locally
uvicorn src.linkedin_lead_scoring.api.main:app --reload

# Run tests
pytest
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
├── src/linkedin_lead_scoring/  # Main package
│   ├── api/                    # FastAPI application
│   ├── data/                   # Data collection & features
│   ├── models/                 # Training & evaluation
│   └── utils/                  # MLflow helpers
├── tests/                      # Test suite
├── scripts/                    # Training & data scripts
├── notebooks/                  # Jupyter exploration
├── model/                      # Trained models
├── data/                       # Raw data (not tracked)
└── docs/                       # Documentation
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

- **Experiment Tracking:** MLflow for model versioning
- **Automated Testing:** pytest with 75%+ coverage requirement
- **CI/CD:** GitHub Actions for automated deployment
- **Model Monitoring:** Database logging for predictions (GDPR-compliant)
- **Drift Detection:** (Coming soon)

## License

MIT License - see LICENSE file

## Contact

**Author:** Ghislain Delabie
**Email:** ghislain@delabie.tech
**Project:** OpenClassrooms OC6 - MLOps
