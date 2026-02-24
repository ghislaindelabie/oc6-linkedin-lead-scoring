# Blueprint: OC8 Finalization — Deploy & Monitor

**Goal**: Bring the OC8 project to production-ready state with all deployment, monitoring, and validation deliverables complete.

**Date**: 2026-02-24
**Branch**: `v0.3.0` (current: `04ce686`)
**Baseline**: 217 tests passing, 43.6% coverage

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| API (`/predict`, `/predict/batch`, `/health`) | **DONE** | Live on staging, OHE fix deployed |
| Model artifacts | **DONE** | Re-exported with correct medians |
| Staging CI/CD (`staging.yml`) | **DONE** | Auto-deploys on `v0.3.0` push |
| Prediction logging (Supabase) | **PARTIAL** | Code done, prod `DATABASE_URL` not set |
| Monitoring dashboard (Streamlit) | **PR #7 OPEN** | Has merge conflicts with v0.3.0 |
| Drift detection (Evidently) | **PR #7 OPEN** | Ready, needs merge |
| Security scanning (Nuclei) | **NOT STARTED** | Referenced in project scope |
| Uptime monitoring | **NOT STARTED** | Referenced in project scope |
| E2E tests (Playwright) | **NOT STARTED** | |
| Production deployment | **PARTIAL** | API live, dashboard needs API_ENDPOINT config |
| Local-vs-API validation | **DONE** | `scripts/validate_pipeline.py` — all scores match |

---

## Tasks

### Track 1: Merge & Stabilize (Opus coordinator)

| # | Task | Size | Deps | Files |
|---|------|------|------|-------|
| T1 | Merge PR #7 (monitoring) into v0.3.0 — resolve conflicts (README.md, pyproject.toml) | M | — | README.md, pyproject.toml, src/...monitoring/*, tests/test_monitoring* |
| T2 | Run full test suite post-merge, fix any breakage | S | T1 | tests/* |
| T3 | Raise coverage threshold from 10% to 40% in ci.yml + staging.yml | S | T2 | .github/workflows/ci.yml, .github/workflows/staging.yml |

### Track 2: Production Configuration (Sonnet subagent A — Infra)

| # | Task | Size | Deps | Files |
|---|------|------|------|-------|
| T4 | Fix `dashboard.yml` — add `API_ENDPOINT` env var on Streamlit Space | S | T1 | .github/workflows/dashboard.yml |
| T5 | Add production deployment workflow (`production.yml`) triggered on `main` push — deploy API + dashboard to prod Spaces | M | T1 | .github/workflows/production.yml (NEW) |
| T6 | Add `DATABASE_URL` secret config to production deploy step (uses existing GitHub secret) | S | T5 | .github/workflows/production.yml |

### Track 3: Security & Availability (Sonnet subagent B — Security)

| # | Task | Size | Deps | Files |
|---|------|------|------|-------|
| T7 | Add Nuclei security scan to `security.yml` (weekly scan of staging + prod API URLs) | M | — | .github/workflows/security.yml |
| T8 | Add Uptime Kuma check endpoint + documentation for external monitoring setup | S | — | docs/UPTIME_MONITORING.md (NEW) |
| T9 | Add Playwright E2E test: API health + predict round-trip against staging | M | T1 | tests/test_e2e.py (NEW), .github/workflows/staging.yml |

### Track 4: Test Hardening (Sonnet subagent C — Tests)

| # | Task | Size | Deps | Files |
|---|------|------|------|-------|
| T10 | Add integration tests for OHE determinism: single vs batch prediction consistency | S | T1 | tests/test_features.py |
| T11 | Add tests for drift detection module (on v0.3.0 after merge) | S | T1 | tests/test_drift.py |
| T12 | Add test for validate_pipeline.py (--local-only, mock API) | S | T1 | tests/test_validate_pipeline.py (NEW) |

### Track 5: Documentation & Cleanup

| # | Task | Size | Deps | Files |
|---|------|------|------|-------|
| T13 | Update README.md: production URLs, architecture diagram, quickstart for OC8 | M | T5 | README.md |
| T14 | Clean up planning docs: archive SESSION_*_TASKS.md, update SESSION_COORDINATION.md with final state | S | T13 | SESSION_COORDINATION.md, SESSION_*_TASKS.md |

---

## Execution Strategy: Parallel with Sonnet Subagents

After T1-T2 (merge + stabilize, done by Opus coordinator), the remaining tasks split into 3 independent tracks that can run in parallel as Sonnet `/dev` subagents:

```
T1-T2 (Opus, sequential)
  │
  ├── Subagent A (Sonnet): T4, T5, T6     — Infra/CI workflows
  │     Branch: feature/prod-deploy
  │     Owns: .github/workflows/*, docs/
  │
  ├── Subagent B (Sonnet): T7, T8, T9     — Security/E2E
  │     Branch: feature/security-e2e
  │     Owns: security.yml, tests/test_e2e.py, docs/UPTIME*
  │
  └── Subagent C (Sonnet): T10, T11, T12  — Test hardening
        Branch: feature/test-hardening
        Owns: tests/test_features.py, tests/test_drift.py, tests/test_validate*
```

Then T3 + T13 + T14 (cleanup) after all subagents merge back.

### Synchronization Points

1. **T1-T2 must complete first** — all subagents need the merged monitoring code
2. **Subagents A, B, C are independent** — no file overlap
3. **T3, T13, T14 run after all subagents merge** — final cleanup pass

### Subagent Instructions

Each subagent:
- Works on its own feature branch (created from `v0.3.0` post-T2)
- Uses **TDD** via `/dev`: write tests first → implement → verify
- Runs `python -m pytest tests/ -v --tb=short` before every commit
- Creates a PR targeting `v0.3.0` when done
- Uses `model: sonnet` for cost efficiency

### File Ownership (no cross-editing)

| Subagent | Owns | Must NOT Touch |
|----------|------|----------------|
| A (Infra) | `.github/workflows/dashboard.yml`, `.github/workflows/production.yml` | tests/test_e2e.py, security.yml |
| B (Security) | `.github/workflows/security.yml`, `tests/test_e2e.py`, `docs/UPTIME_MONITORING.md` | .github/workflows/production.yml, tests/test_drift.py |
| C (Tests) | `tests/test_features.py` (additions only), `tests/test_drift.py`, `tests/test_validate_pipeline.py` | .github/workflows/*, src/* |

---

## Branch Strategy

```
v0.3.0 (after T1-T2 merge)
  ├── feature/prod-deploy        (Subagent A: T4, T5, T6)
  ├── feature/security-e2e       (Subagent B: T7, T8, T9)
  └── feature/test-hardening     (Subagent C: T10, T11, T12)
```

All PRs target `v0.3.0`. User reviews and merges manually.
After all 3 merge → T3, T13, T14 on v0.3.0 directly → final PR to `main`.

---

## Testing Strategy

- **TDD for all subagents**: write failing test → implement → green → commit
- **Post-merge validation**: `python -m pytest tests/ -v --tb=short` must pass at every step
- **E2E validation (T9)**: Playwright test hits staging API (runs in CI after staging deploy)
- **Pipeline validation**: `python scripts/validate_pipeline.py --local-only` stays green throughout
- **Coverage target**: raise to 40% after monitoring tests merge (T3)

---

## Risks & Open Questions

| Risk | Mitigation |
|------|-----------|
| PR #7 merge conflicts may be complex | Preview shows README.md + possibly pyproject.toml — resolvable |
| Nuclei scanner needs target URLs | Use staging URL for CI; prod URL post-deploy |
| Playwright requires browser install in CI | Use `playwright install --with-deps chromium` in workflow |
| Production `DATABASE_URL` secret not yet created | **USER ACTION**: create Supabase prod project + add secret before T6 |
| Coverage may not reach 40% even after monitoring tests | Adjust threshold based on actual measurement post-T2 |

### User Decisions Needed

1. **Supabase production project**: Has it been created? If not, needed before T6.
2. **Coverage target**: 40% acceptable or aim higher?
3. **Nuclei scanner**: Run against staging only, or also prod?
4. **Uptime Kuma**: External service setup (user-managed) or just document how?

---

## Verification

After all tasks complete:
```bash
# Full test suite
python -m pytest tests/ -v --tb=short          # 250+ tests pass

# Pipeline validation
python scripts/validate_pipeline.py --local-only  # All scores match

# Staging E2E
python scripts/validate_pipeline.py --staging-url https://ghislaindelabie-oc6-bizdev-ml-api-staging.hf.space

# Coverage
python -m pytest tests/ --cov=src/linkedin_lead_scoring --cov-report=term-missing -q  # ≥40%
```
