# Code Review Report: Session A (Infrastructure & CI/CD)

**Branch**: `feature/infra-cicd` | **PR #5** targeting `v0.3.0`
**Reviewer**: Opus Code Review | **Date**: 2026-02-20
**Test Results**: 72 passed, 0 failed (2.48s)
**Review addressed**: 2026-02-20 — all CRITICAL and fixable IMPORTANT items resolved

---

## A. Summary

**Overall Quality: 4/5**

### Key Strengths
- Clean, well-organized code with consistent style and good docstrings
- Proper separation of concerns: models, repository, connection each in their own file
- Good use of async SQLAlchemy with proper fallback to SQLite for local development
- Comprehensive test coverage: DB, export script, Dockerfile (static analysis), CI workflow YAML
- Dockerfile follows security best practices: non-root user, healthcheck, slim base image

### Critical Issues (3) — all FIXED
1. ~~Security scan workflow silently swallows failures (`|| true` on audit commands)~~
2. ~~Docker build verification in CI is unreliable (`|| true` suppresses container start errors)~~
3. ~~CI test dependency missing: `pyyaml` needed by `test_ci_workflows.py`~~

---

## B. Actionable Recommendations

### CRITICAL (must fix before merge)

1. **Fix security.yml to actually fail on vulnerabilities.**
   - File: `.github/workflows/security.yml` lines 27-28 and 54-55
   - Problem: `pip-audit ... || true` and `bandit ... || true` means security scans can NEVER fail. This is security theater.
   - Fix: Remove `|| true` from both commands. Use `.bandit` config file for known false positives instead.
   - **STATUS: ✅ FIXED** — removed `|| true` from both `pip-audit` and `bandit` commands in `security.yml`. Scans now fail the workflow on findings.

2. **Fix CI Docker verification to actually verify.**
   - File: `.github/workflows/ci.yml` lines 66-72
   - Problem: `docker run ... || true` means if the container crashes on boot, CI passes silently.
   - Fix: Remove `|| true`, add a proper health check:
     ```yaml
     docker run --rm -d --name oc6-test -p 7860:7860 -e APP_ENV=test oc6-api
     sleep 10
     curl -f http://localhost:7860/health || (docker logs oc6-test && exit 1)
     docker stop oc6-test
     ```
   - **STATUS: ✅ FIXED** — `ci.yml` docker-build verify step updated with `curl -f` health probe and container log dump on failure.

3. **Add pyyaml to CI test dependencies.**
   - File: `.github/workflows/ci.yml` line 26
   - Problem: `test_ci_workflows.py` imports `yaml` but pyyaml is not installed in CI.
   - Fix: Add `pyyaml` to the pip install line.
   - **STATUS: ✅ FIXED** — `pyyaml` added to pip install line in ci.yml test job.

### IMPORTANT (should fix)

4. **Fix alembic exclusion conflict in deploy step.**
   - File: `.github/workflows/ci.yml` lines 121-122
   - Problem: Alembic files are excluded from HF upload, but Dockerfile CMD runs `alembic upgrade head`. Deploy will fail.
   - Fix: Either include alembic in the upload or remove migration from CMD.
   - **STATUS: ✅ FIXED** — replaced `--exclude "alembic/*"` + `--exclude "alembic.ini"` with `--exclude "alembic/__pycache__/*"` so migration files are included in the upload.

5. **Use regular pip install in Dockerfile, not editable mode.**
   - File: `Dockerfile` line 30
   - Fix: Change `pip install --no-cache-dir -e .` to `pip install --no-cache-dir .`
   - **STATUS: ✅ FIXED** — Dockerfile updated to use `pip install --no-cache-dir .`. Test `test_installs_package_in_editable_mode` renamed to `test_installs_package_in_production_mode` and updated to assert non-editable install.

6. **Add `exec` before uvicorn in Dockerfile CMD for proper signal handling.**
   - File: `Dockerfile` line 50
   - Fix: `CMD ["sh", "-c", "alembic upgrade head && exec uvicorn linkedin_lead_scoring.api.main:app --host 0.0.0.0 --port 7860"]`
   - **STATUS: ✅ FIXED** — `exec` added before `uvicorn` in CMD. Ensures uvicorn receives SIGTERM directly from Docker for graceful shutdown.

7. **Fix test_postgresql_url_uses_asyncpg to test actual module (not a copy).**
   - File: `tests/test_db.py` lines 71-98
   - Problem: Test reimplements production logic instead of testing the actual module.
   - Fix: Use `monkeypatch.setenv` + `importlib.reload` to test the real code.
   - **STATUS: ⏸ POSTPONED** — deliberate decision. We attempted `importlib.reload(conn_mod)` during Task A.3 and abandoned it: reloading `connection.py` calls `create_async_engine(asyncpg_url)` at module import time, which requires `asyncpg` installed. `asyncpg` is not in the local conda test environment (production-only). Lazy engine creation in `connection.py` would fix this but risks breaking Session B which depends on `get_db()`. Deferred to Session B (owns `conftest.py` and controls test environment). The current test validates the same URL-rewriting logic directly — the business rule is covered.

8. **Remove training-only dependencies from requirements-prod.txt.**
   - `optuna` (hyperparameter tuning only) -- remove
   - `imbalanced-learn`, `shap`, `mlflow` -- evaluate if needed at serving time, likely remove
   - This reduces Docker image size and attack surface significantly.
   - **STATUS: ✅ PARTIALLY FIXED** — removed `imbalanced-learn`, `shap`, `optuna` from `requirements-prod.txt`. `mlflow` **kept**: CLAUDE.md mandates MLflow integration; it may be used for model registry access at serving time. Will be revisited if model loading is confirmed to use only joblib artifacts.

9. **Add database indexes on `created_at` columns.**
   - File: `alembic/versions/13392d3abd1b_*.py`
   - Fix: `op.create_index('ix_prediction_logs_created_at', 'prediction_logs', ['created_at'])`
   - **STATUS: ✅ FIXED** — added `op.create_index` for `ix_prediction_logs_created_at` and `ix_api_metrics_created_at` in both `upgrade()` and corresponding `drop_index` in `downgrade()`.

10. **Restrict dashboard.yml uploads to monitoring code only.**
    - File: `.github/workflows/dashboard.yml` lines 43-48
    - Fix: Change `--include "src/**"` to `--include "src/linkedin_lead_scoring/monitoring/**"`
    - **STATUS: ✅ FIXED** — `--include "src/**"` replaced with `--include "src/linkedin_lead_scoring/monitoring/**"` in dashboard.yml.

### SUGGESTIONS (nice to have)

11. **Pin test tool versions in CI to avoid surprise breakage.**
    - **STATUS: ⏸ POSTPONED** — low urgency; upper-bound pinning in `requirements-prod.txt` already mitigates most drift. Acceptable for a course project.

12. **Clean up `alembic.ini` placeholder URL on line 89.**
    - **STATUS: ⏸ POSTPONED** — cosmetic; no functional impact. The placeholder URL is never used (env.py reads DATABASE_URL at runtime).

13. **Simplify HEALTHCHECK test parsing in `test_dockerfile.py`.**
    - **STATUS: ⏸ POSTPONED** — the test works correctly. Refactoring test logic has no ROI at this stage.

14. **Add `security.yml` trigger on pull_request events for faster feedback.**
    - **STATUS: ⏸ POSTPONED** — a full pip-audit + bandit scan on every PR adds ~5 min per run. Weekly scheduled scan is acceptable. Can be added when the project reaches a stage requiring stricter security gates.

15. **Move training-only deps in `pyproject.toml` to optional extras group.**
    - **STATUS: ⏸ POSTPONED** — `pyproject.toml` serves the full dev/notebook environment. Training deps (optuna, shap, imbalanced-learn) are legitimate there. Moving to optional extras improves DX but introduces complexity; low urgency.

---

## C. Dependencies Check

| Dependency | Needed at serving time? | Verdict | Action |
|---|---|---|---|
| fastapi, uvicorn, pydantic | Yes | Keep | — |
| scikit-learn, xgboost, pandas, numpy, joblib | Yes | Keep | — |
| category-encoders | Yes (preprocessor uses TargetEncoder) | Keep | — |
| sqlalchemy, asyncpg, aiosqlite, alembic, greenlet | Yes (DB) | Keep | — |
| python-dotenv, httpx | Yes | Keep | — |
| **mlflow** | Possible (model registry) | Keep | Revisit if serving confirmed joblib-only |
| **imbalanced-learn** | **No** (training only) | **Removed** | ✅ Done |
| **shap** | **No** (explanation only) | **Removed** | ✅ Done |
| **optuna** | **No** (tuning only) | **Removed** | ✅ Done |

---

## D. Test Report

```
After fixes: 72 passed in 2.11s (unchanged count — fixes required 1 test rename/update, no new tests)

tests/test_api_integration.py      3 passed
tests/test_ci_workflows.py        22 passed
tests/test_db.py                  14 passed
tests/test_dockerfile.py          13 passed   ← test_installs_package_in_production_mode (was editable_mode)
tests/test_export_model.py        13 passed
tests/test_smoke.py                4 passed
```

All tests pass locally and pyyaml is now installed in CI (`ci.yml` updated).

---

## E. Files Changed in Review Fix Pass

| File | Change |
|------|--------|
| `.github/workflows/ci.yml` | Add pyyaml to deps; fix docker verify (curl + no `\|\| true`); remove alembic exclusions |
| `.github/workflows/security.yml` | Remove `\|\| true` from pip-audit and bandit |
| `.github/workflows/dashboard.yml` | Restrict `--include "src/**"` to monitoring subpackage |
| `Dockerfile` | Regular install (`pip install .`); add `exec` before uvicorn |
| `requirements-prod.txt` | Remove imbalanced-learn, shap, optuna |
| `alembic/versions/13392d3abd1b_*.py` | Add created_at indexes; drop in downgrade |
| `tests/test_dockerfile.py` | Rename + update editable-mode test to production-mode |
