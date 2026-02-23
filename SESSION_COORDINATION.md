# Session Coordination — OC8 Parallel Development

**Version branch**: `v0.3.0`
**Last updated**: 2026-02-23 (Merging PRs progressively into v0.3.0)

---

## Session Status

| Session | Branch | Worktree | Status | Current Task | Last Commit |
|---------|--------|----------|--------|--------------|-------------|
| **Opus** (Coordinator) | `v0.3.0` | main repo | Active | Merging PRs | Progressive merge |
| **A** (Infra/CI/CD) | `feature/infra-cicd` | `worktrees/session-a` | **Merged** (PR #5) | — | All tasks done |
| **B** (API/Tests) | `feature/api-scoring` | `worktrees/session-b` | **Merging** (PR #6) | All B.1–B.7 done | 109 tests |
| **C** (Monitoring/Drift) | `feature/monitoring` | `worktrees/session-c` | **Pending** (PR #7) | — | — |

## Merge Queue

| PR | Source Branch | Target | Status | Reviewer |
|----|--------------|--------|--------|----------|
| #5 | `feature/infra-cicd` | `v0.3.0` | **Ready** | Opus |
| B  | `feature/api-scoring` | `v0.3.0` | **Ready** | Opus |
| C  | `feature/monitoring` | `v0.3.0` | **Ready** | Opus |

## Dependency Tracker

| Dependency | Provider | Consumer(s) | Status |
|------------|----------|-------------|--------|
| Model artifact (`model/*.joblib`) | Session A | B, C | **Merged** (A.2) |
| Feature columns (`model/feature_columns.json`) | Session A | B, C | **Merged** (A.2) — 47 features |
| Reference data (`data/reference/`) | Session A | C | **Merged** (A.2) — 100 rows |
| DB module (`src/.../db/`) | Session A | B (logging) | **Merged** (A.3) |
| Alembic migrations (`alembic/`) | Session A | Deploy | **Merged** (A.3) |
| CI/CD workflows | Session A | All | **Merged** (A.5) — fixed for Python API deploy |
| API schemas finalized | Session B | C (for monitoring) | **Done** (B.1) |
| Production logging format | Session B | C (for drift analysis) | **Done** (B.4) |

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

## Manual Actions Required (user)

| Action | When | Notes |
|--------|------|-------|
| ~~Create Supabase staging project~~ | ~~Before merging PR #5~~ | ✅ Done — using resumed project |
| ~~Add `STAGING_DATABASE_URL` to GitHub repo secrets~~ | ~~Before first push to `v0.3.0`~~ | ✅ Done |
| HF Spaces creation | Automatic on first push to `v0.3.0` | `staging.yml` creates them with `\|\| true` |
| Set `DATABASE_URL` on production HF Space | Before Session B endpoint goes live | Needed for persistent prediction logging in production |
| Merge PR #5 → B → C into `v0.3.0` | Now | Triggers staging deploy automatically |
