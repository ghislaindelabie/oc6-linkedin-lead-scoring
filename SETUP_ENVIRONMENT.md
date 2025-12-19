# Environment Setup: Hybrid Conda + uv

**Strategy**: Use **conda** for big scientific packages (NumPy, pandas, scikit-learn) and **uv** for specialized ML/API packages (MLflow, FastAPI, XGBoost).

---

## Why This Approach?

- **Conda**: Excellent for managing binary dependencies (NumPy, pandas, matplotlib)
- **uv**: Fast package management for pure Python packages (10-100x faster than pip)
- **Hybrid**: Best of both worlds - stable scientific stack + fast specialized packages

---

## Step-by-Step Setup

### 1. Create Conda Environment

```bash
# Navigate to project
cd "/Users/ghislaindelabie/Projets dév/Formation OC/OC6 - Initiez-vous au MLOps/oc6-linkedin-lead-scoring"

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate it
conda activate oc6
```

**What this installs (via conda):**
- Python 3.11
- NumPy 2.x
- pandas 2.2+
- scikit-learn 1.5+
- imbalanced-learn 0.12+
- matplotlib, seaborn
- JupyterLab, ipykernel
- psycopg2 (PostgreSQL driver)
- uv (package manager)

---

### 2. Install Specialized Packages with uv

```bash
# Make sure oc6 environment is active
conda activate oc6

# Install project in editable mode with uv (fast!)
uv pip install -e .

# Or install dev dependencies too
uv pip install -e ".[dev]"
```

**What uv installs (from pyproject.toml):**
- MLflow 2.18+
- XGBoost 3.0+
- FastAPI + Uvicorn
- SHAP, Optuna, optuna-integration
- httpx (LemList API client)
- pytest, pytest-cov, pytest-asyncio (dev tools)

---

### 3. Verify Installation

```bash
# Check all critical packages
python -c "
import numpy as np
import pandas as pd
import sklearn
import mlflow
import xgboost as xgb
import fastapi
print('✓ NumPy:', np.__version__)
print('✓ Pandas:', pd.__version__)
print('✓ scikit-learn:', sklearn.__version__)
print('✓ MLflow:', mlflow.__version__)
print('✓ XGBoost:', xgb.__version__)
print('✓ FastAPI:', fastapi.__version__)
print('\\n✅ All packages installed successfully!')
"
```

---

## Quick Commands

### Daily Usage

```bash
# Activate environment
conda activate oc6

# Start Jupyter Lab
jupyter lab

# Start MLflow UI
mlflow ui --port 5000

# Run FastAPI server
uvicorn linkedin_lead_scoring.api.main:app --reload
```

### Update Packages

```bash
# Update conda packages
conda update --all

# Update uv packages
uv pip install --upgrade mlflow xgboost fastapi

# Or update all from pyproject.toml
uv pip install --upgrade -e .
```

### Add New Package

```bash
# If it's a big scientific package → add to environment.yml
# Then update environment:
conda env update -f environment.yml

# If it's a specialized package → add to pyproject.toml
# Then install with uv:
uv pip install -e .
```

---

## Package Split Strategy

### Install via CONDA (environment.yml)
✅ numpy, pandas, scikit-learn
✅ imbalanced-learn (matches scikit-learn version)
✅ matplotlib, seaborn
✅ jupyterlab, ipykernel
✅ psycopg2

**Why?** Better binary compatibility, handles complex C/Fortran dependencies

### Install via UV (pyproject.toml)
✅ mlflow, xgboost, optuna, optuna-integration
✅ fastapi, uvicorn, pydantic
✅ httpx, python-dotenv
✅ shap
✅ pytest, pytest-cov, pytest-asyncio

**Why?** Pure Python or simpler dependencies, uv is 10-100x faster

---

## Troubleshooting

### "ModuleNotFoundError" after setup
```bash
# Reinstall in editable mode
conda activate oc6
uv pip install -e .
```

### "NumPy ABI incompatibility"
```bash
# This shouldn't happen with conda managing NumPy
# But if it does, recreate environment:
conda deactivate
conda env remove -n oc6
conda env create -f environment.yml
conda activate oc6
uv pip install -e .
```

### "uv command not found"
```bash
# Install uv in base or oc6 environment
conda install -c conda-forge uv

# Or install via curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Environment Management

### List all conda environments
```bash
conda env list
```

### Export current environment (for reproducibility)
```bash
# Export conda packages
conda env export --no-builds > environment-lock.yml

# Export all packages (conda + pip/uv)
conda env export > environment-full.yml
```

### Delete environment (if needed)
```bash
conda deactivate
conda env remove -n oc6
```

### Clone environment (for experiments)
```bash
conda create --name oc6-experiment --clone oc6
```

---

## Summary

**One-time setup:**
```bash
conda env create -f environment.yml
conda activate oc6
uv pip install -e ".[dev]"
```

**Daily use:**
```bash
conda activate oc6
jupyter lab  # or mlflow ui, or uvicorn...
```

**That's it!** 🚀
