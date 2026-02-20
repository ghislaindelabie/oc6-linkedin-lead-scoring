# OC6/OC8 LinkedIn Lead Scoring — Project Instructions

## Project Context
- **Course**: OpenClassrooms OC6 (MLOps) + OC8 (Deploy & Monitor)
- **Company**: "Pret a Depenser" (fictional)
- **Goal**: Predict LinkedIn contact engagement (reply/interest) from profile data
- **Model**: XGBoost classifier trained on LemList campaign data (1,910 contacts)
- **Current version**: v0.3.0 (deployment phase)

## Tech Stack
- **API**: FastAPI (uvicorn, port 7860 on HF Spaces)
- **ML**: XGBoost, scikit-learn, MLflow (experiment tracking + model registry)
- **Monitoring**: Evidently AI (drift), Streamlit (dashboard)
- **Database**: Supabase (PostgreSQL) for production logging
- **Deployment**: Docker, Hugging Face Spaces (2 spaces: API + dashboard)
- **CI/CD**: GitHub Actions
- **Security/Availability**: Nuclei scanner, Uptime Kuma, Playwright E2E tests
- **Environment**: conda `oc6` + `uv pip install -e ".[dev]"`

## Environment Setup
```bash
conda activate oc6
uv pip install -e ".[dev]"
python -m pytest tests/ -v --tb=short
```

## Model Loading Pattern (CRITICAL)
The model MUST be loaded ONCE at API startup, never per-request:
```python
# CORRECT — load at module level or app startup event
model = joblib.load("model/xgboost_model.joblib")

# WRONG — never do this
@app.post("/predict")
async def predict(data):
    model = joblib.load(...)  # NO! Loads every request
```

## MLflow Integration (MANDATORY)
- All experiments tracked in MLflow
- Model registered in MLflow Model Registry
- Production model loaded from registry or exported artifact
- Track: parameters, metrics, artifacts, model versions

## Parallel Session Rules

### Git Worktree Setup
```
oc6-linkedin-lead-scoring/              (coordinator — Opus)
  worktrees/
    session-a/                          (feature/infra-cicd)
    session-b/                          (feature/api-scoring)
    session-c/                          (feature/monitoring)
```

### File Ownership (STRICT — do not cross-edit)
| Session | Owns | Must NOT Touch |
|---------|------|----------------|
| A (Infra) | `.github/`, `Dockerfile`, `requirements-prod.txt`, `model/`, `scripts/`, `setup_env.sh`, `alembic/`, `src/.../db/` | `src/.../api/`, `src/.../monitoring/` |
| B (API) | `src/.../api/`, `tests/test_api*`, `tests/test_predict*`, `tests/test_schemas*` | `.github/`, `src/.../monitoring/`, `src/.../db/` |
| C (Monitoring) | `src/.../monitoring/`, `streamlit_app.py`, `tests/test_monitoring*`, `tests/test_drift*`, `notebooks/03_*` | `.github/`, `src/.../api/`, `src/.../db/` |

### Shared Files Protocol
- `pyproject.toml`, `requirements-prod.txt`: Session A owns; B and C list needed deps in PR description
- `README.md`: each session writes ONLY its own section
- `conftest.py`: Session B owns; A and C add fixtures via their own conftest files in subdirs

### PR Workflow
1. Work on your feature branch in your worktree
2. Run tests: `python -m pytest tests/ -v --tb=short`
3. Commit and push to your feature branch
4. Create PR targeting `v0.3.0` (not main)
5. Opus coordinator reviews and merges with user approval

## Development Workflow (Skills)

Use these slash commands for consistent workflow:
- **`/dev`** — Use for each implementation task. Follows TDD: write tests first, then implement, then verify.
- **`/commit`** — Use after completing each task. Handles staging, diff review, and commit message formatting.
- **`/test`** — Run the test suite and analyze results.
- **`/document`** — Update documentation after significant changes.

**Typical task flow**:
1. Read the task description in your `SESSION_X_TASKS.md`
2. Use `/dev` to implement (test-first approach)
3. Verify tests pass: `python -m pytest tests/ -v --tb=short`
4. Use `/commit` to commit with proper conventional format
5. Push to your feature branch: `git push origin <your-branch>`
6. Move to the next task

## Testing
- Run `python -m pytest tests/ -v --tb=short` before every commit
- Use mocks for external services (Supabase, MLflow server, HF Spaces)
- Mock the model in API tests (don't require real model file)
- Minimum 70% coverage target on new code

## Compact Instructions
<!-- Preserved during context compaction -->
- Model loaded ONCE at startup, never per-request
- MLflow is mandatory for all experiment tracking
- Tests must pass before any commit
- Each session works ONLY in its worktree, pushes ONLY its branch
- PRs target v0.3.0, never main directly
- Never expose secrets or read .env files
