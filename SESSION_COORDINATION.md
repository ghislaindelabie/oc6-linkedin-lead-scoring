# Session Coordination — OC8 Parallel Development

**Version branch**: `v0.3.0`
**Last updated**: 2026-02-20 (Session B — B.4 complete)

---

## Session Status

| Session | Branch | Worktree | Status | Current Task | Last Commit |
|---------|--------|----------|--------|--------------|-------------|
| **Opus** (Coordinator) | `v0.3.0` | main repo | Active | Setup complete | `0be90e2` |
| **A** (Infra/CI/CD) | `feature/infra-cicd` | `worktrees/session-a` | Not started | — | — |
| **B** (API/Tests) | `feature/api-scoring` | `worktrees/session-b` | In progress | B.5 — unit tests | B.4 done |
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
| API schemas finalized | Session B | C (for monitoring) | **Done** (B.1) |
| Production logging format | Session B | C (for drift analysis) | **Done** (B.4) — `logs/predictions.jsonl`, one JSON line per prediction: `{timestamp, input, score, label, inference_ms, model_version}` |

## Shared Dependencies (pyproject.toml additions)

When a session needs a new dependency, record it here. Session A will integrate.

| Package | Needed By | Version | Added? |
|---------|-----------|---------|--------|
| `evidently` | C | `>=0.4.0` | No |
| `streamlit` | C | `>=1.30.0` | No |
| `joblib` | A, B | `>=1.3.0` | No — **needed by B.2 real model path; add to requirements-prod.txt** |
| `onnx` | C | `>=1.15.0` | No |
| `onnxruntime` | C | `>=1.17.0` | No |
| `psycopg2-binary` | A | `>=2.9.0` | No |
| `supabase` | A | `>=2.0.0` | No |

## Notes

- Each session works ONLY in its worktree directory
- PRs target `v0.3.0`, never `main`
- Opus reviews all PRs before merge
- If you need a file owned by another session, create an interface/stub and document it here
