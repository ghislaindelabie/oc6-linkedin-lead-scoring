# Session Coordination — OC8 Parallel Development

**Version branch**: `v0.3.0`
**Last updated**: 2026-02-20 (Session A — Tasks A.1–A.5 complete)

---

## Session Status

| Session | Branch | Worktree | Status | Current Task | Last Commit |
|---------|--------|----------|--------|--------------|-------------|
| **Opus** (Coordinator) | `v0.3.0` | main repo | Active | Setup complete | `0be90e2` |
| **A** (Infra/CI/CD) | `feature/infra-cicd` | `worktrees/session-a` | In Progress | A.6: PR | A.1–A.5 done |
| **B** (API/Tests) | `feature/api-scoring` | `worktrees/session-b` | Not started | — | — |
| **C** (Monitoring/Drift) | `feature/monitoring` | `worktrees/session-c` | Not started | — | — |

## Merge Queue

| PR | Source Branch | Target | Status | Reviewer |
|----|--------------|--------|--------|----------|
| — | — | — | — | — |

## Dependency Tracker

| Dependency | Provider | Consumer(s) | Status |
|------------|----------|-------------|--------|
| Model artifact (`model/*.joblib`) | Session A | B, C | **Ready** (A.2) |
| Feature columns (`model/feature_columns.json`) | Session A | B, C | **Ready** (A.2) — 47 features |
| Reference data (`data/reference/`) | Session A | C | **Ready** (A.2) — 100 rows |
| DB module (`src/.../db/`) | Session A | B (logging) | **Ready** (A.3) — `get_db()` dep, `log_prediction()`, `log_api_metric()` |
| Alembic migrations (`alembic/`) | Session A | Deploy | **Ready** (A.3) — run `alembic upgrade head` before first deploy |
| CI/CD: lint + test + coverage (`.github/workflows/ci.yml`) | Session A | All | **Ready** (A.5) — ruff, pytest, 70% coverage gate, docker-build on PRs |
| CI/CD: security scan (`.github/workflows/security.yml`) | Session A | All | **Ready** (A.5) — pip-audit + bandit weekly, artifacts uploaded |
| CI/CD: dashboard deploy (`.github/workflows/dashboard.yml`) | Session A | C | **Ready** (A.5) — deploys streamlit_app.py to oc6-bizdev-monitoring on push to main |
| API schemas finalized | Session B | C (for monitoring) | Pending |
| Production logging format | Session B | C (for drift analysis) | Pending |

## Shared Dependencies (pyproject.toml additions)

When a session needs a new dependency, record it here. Session A will integrate.

| Package | Needed By | Version | Added? |
|---------|-----------|---------|--------|
| `evidently` | C | `>=0.4.0` | No |
| `streamlit` | C | `>=1.30.0` | No |
| `joblib` | A, B | `>=1.3.0` | **Yes** (A.1) |
| `category-encoders` | A | `>=2.6.0` | **Yes** (A.2) |
| `onnx` | C | `>=1.15.0` | No |
| `onnxruntime` | C | `>=1.17.0` | No |
| `asyncpg` | A | `>=0.29.0` | **Yes** (A.3) — async PostgreSQL driver |
| `aiosqlite` | A | `>=0.19.0` | **Yes** (A.3) — local dev + test SQLite |
| `greenlet` | A | `>=3.0` | **Yes** (A.3) — required by SQLAlchemy async |
| `psycopg2-binary` | — | — | Not needed (asyncpg used instead) |
| `supabase` | — | — | Not needed (direct asyncpg connection) |

## Notes

- Each session works ONLY in its worktree directory
- PRs target `v0.3.0`, never `main`
- Opus reviews all PRs before merge
- If you need a file owned by another session, create an interface/stub and document it here
