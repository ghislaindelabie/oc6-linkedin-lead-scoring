# Session Coordination — OC8 Parallel Development

**Version branch**: `v0.3.0`
**Last updated**: 2026-02-20 (Session A — Task A.1 complete)

---

## Session Status

| Session | Branch | Worktree | Status | Current Task | Last Commit |
|---------|--------|----------|--------|--------------|-------------|
| **Opus** (Coordinator) | `v0.3.0` | main repo | Active | Setup complete | `0be90e2` |
| **A** (Infra/CI/CD) | `feature/infra-cicd` | `worktrees/session-a` | In Progress | A.2: Export model | A.1 done |
| **B** (API/Tests) | `feature/api-scoring` | `worktrees/session-b` | Not started | — | — |
| **C** (Monitoring/Drift) | `feature/monitoring` | `worktrees/session-c` | Not started | — | — |

## Merge Queue

| PR | Source Branch | Target | Status | Reviewer |
|----|--------------|--------|--------|----------|
| — | — | — | — | — |

## Dependency Tracker

| Dependency | Provider | Consumer(s) | Status |
|------------|----------|-------------|--------|
| Model artifact (`model/*.joblib`) | Session A | B, C | Pending |
| Feature columns (`model/feature_columns.json`) | Session A | B, C | Pending |
| Reference data (`data/reference/`) | Session A | C | Pending |
| API schemas finalized | Session B | C (for monitoring) | Pending |
| Production logging format | Session B | C (for drift analysis) | Pending |

## Shared Dependencies (pyproject.toml additions)

When a session needs a new dependency, record it here. Session A will integrate.

| Package | Needed By | Version | Added? |
|---------|-----------|---------|--------|
| `evidently` | C | `>=0.4.0` | No |
| `streamlit` | C | `>=1.30.0` | No |
| `joblib` | A, B | `>=1.3.0` | **Yes** (A.1) |
| `onnx` | C | `>=1.15.0` | No |
| `onnxruntime` | C | `>=1.17.0` | No |
| `psycopg2-binary` | A | `>=2.9.0` | No |
| `supabase` | A | `>=2.0.0` | No |

## Notes

- Each session works ONLY in its worktree directory
- PRs target `v0.3.0`, never `main`
- Opus reviews all PRs before merge
- If you need a file owned by another session, create an interface/stub and document it here
