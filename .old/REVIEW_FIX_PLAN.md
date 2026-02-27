# Review Fix Plan — Session A (feature/infra-cicd)

**Source**: REVIEW_REPORT_SESSION_A.md
**Branch**: `feature/infra-cicd`
**Baseline**: 72 tests passing
**Date**: 2026-02-20

---

## Goal

Address all CRITICAL and fixable IMPORTANT findings from the Opus code review before PR #5 is merged; document postponed items with clear rationale.

---

## Task Breakdown

### Track 1 — CI/CD Workflow Fixes
*(Files: `.github/workflows/ci.yml`, `.github/workflows/security.yml`, `.github/workflows/dashboard.yml`, `tests/test_ci_workflows.py`)*

| # | Finding | Severity | Size | Test change? |
|---|---------|----------|------|-------------|
| C1 | Add `pyyaml` to CI pip install line | CRITICAL | S | No (new dep, existing tests stay green) |
| C2 | Remove `|| true` from pip-audit & bandit in security.yml | CRITICAL | S | No |
| C3 | Fix docker verify step — remove `|| true`, add `curl` health probe | CRITICAL | S | No |
| I4 | Remove alembic exclusions from ci.yml deploy step | IMPORTANT | S | No |
| I10 | Restrict dashboard.yml `--include "src/**"` to monitoring subpackage | IMPORTANT | S | No |

**Files touched**: `ci.yml`, `security.yml`, `dashboard.yml`
**Tests**: `test_ci_workflows.py` — no changes needed (existing tests validate presence of commands, not `|| true` patterns)

---

### Track 2 — Dockerfile Fixes
*(Files: `Dockerfile`, `tests/test_dockerfile.py`)*

| # | Finding | Severity | Size | Test change? |
|---|---------|----------|------|-------------|
| I5 | Regular install (`pip install .`) instead of editable (`-e .`) | IMPORTANT | S | Yes — update `test_installs_package_in_editable_mode` |
| I6 | Add `exec` before uvicorn in CMD for proper signal handling | IMPORTANT | S | No (existing tests check port/host/path, not exec) |

**Files touched**: `Dockerfile`, `tests/test_dockerfile.py`

---

### Track 3 — Production Dependencies
*(Files: `requirements-prod.txt`)*

| # | Finding | Severity | Size | Test change? |
|---|---------|----------|------|-------------|
| I8 | Remove training-only deps: `imbalanced-learn`, `shap`, `optuna` | IMPORTANT | S | No |

**Note**: `mlflow` is kept — CLAUDE.md mandates MLflow integration; model registry access may be required at serving time.
**Note**: `pyproject.toml` is NOT changed here — it serves the dev/notebook environment where these packages are needed. Moving them to optional extras is a SUGGESTION (#15) and deferred.

---

### Track 4 — Database Migration
*(Files: `alembic/versions/13392d3abd1b_*.py`)*

| # | Finding | Severity | Size | Test change? |
|---|---------|----------|------|-------------|
| I9 | Add `created_at` indexes to both tables | IMPORTANT | S | No |

---

## Execution Strategy

**Linear execution** — all tasks are small (S), fast, and touch distinct files with minimal overlap. Parallel execution would add coordination overhead for no gain.

**Recommended order**:
1. Track 2 first (Dockerfile + test update — only track requiring a test change)
2. Track 1 (workflow fixes — CRITICAL items resolved)
3. Track 3 (remove deps)
4. Track 4 (alembic index)
5. Document postponed items in REVIEW_REPORT_SESSION_A.md
6. Run full test suite, commit, push

---

## Postponed Items — Documented Rationale

### IMPORTANT #7 — `test_postgresql_url_uses_asyncpg` tests module copy, not real code

**Review recommendation**: Use `monkeypatch.setenv + importlib.reload` to test the real module.

**Rationale for postpone**:
We attempted `importlib.reload(conn_mod)` during Task A.3 and abandoned it because reloading `connection.py` calls `create_async_engine(asyncpg_url)` at module import time. `asyncpg` is not installed in the local conda test environment (only in production Docker). This would break the test suite for all contributors running locally.

The fix requires either:
- Lazy engine creation in `connection.py` (refactoring the module's interface — risk of breaking Session B which depends on `get_db()`), or
- Installing `asyncpg` into the local conda env (changes dev setup requirements)

**Decision**: Defer to Session B, which owns `conftest.py` and controls the test environment. Document the constraint in the test file with a comment. The current test validates the same URL rewriting logic directly — the business rule is covered even if not via module reload.

### SUGGESTIONS #11–#15 — All deferred

| # | Finding | Rationale |
|---|---------|-----------|
| 11 | Pin test tool versions in CI | Low urgency; GH Actions caching + upper-bounds in pyproject.toml mitigate drift |
| 12 | Clean up `alembic.ini` placeholder URL | Cosmetic; no functional impact |
| 13 | Simplify HEALTHCHECK test parsing | Test works correctly; refactoring test logic has no ROI at this stage |
| 14 | Add `security.yml` trigger on PR events | Full security scan on every PR adds ~5 min; acceptable as weekly scan for now |
| 15 | Move training-only deps in pyproject.toml to optional extras | Nice-to-have DX improvement; low urgency, some refactoring risk |

---

## Testing Strategy

- Only **Track 2** requires a test change: rename and update `test_installs_package_in_editable_mode` to assert regular install.
- All other fixes are additive or removal — existing tests remain green.
- Run full suite (`python -m pytest tests/ -v --tb=short`) after each track, and once more at the end.
- Target: **72 tests passing** throughout (no new tests needed — this is a fix pass, not a feature).

---

## Branch Strategy

Stay on `feature/infra-cicd`. All fixes go into a single commit:
```
fix: address code review findings (CRITICAL + IMPORTANT)
```
Optionally split into two commits:
- `fix: address CRITICAL CI/CD and Dockerfile review findings`
- `fix: remove training-only production dependencies and add DB indexes`

---

## Documentation Impact

After fixes:
- Update `REVIEW_REPORT_SESSION_A.md` with status column for each item (FIXED / POSTPONED + rationale)
- No README change needed (internal infra changes only)
