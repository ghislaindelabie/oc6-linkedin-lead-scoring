# Blueprint: OC8 Finalization — Deploy & Monitor

**Goal**: Bring the OC8 project to production-ready state with all deployment, monitoring, and validation deliverables complete.

**Date**: 2026-02-24 (updated)
**Branch**: `v0.3.0` (current: `4819b27` — all CI fixes applied)
**Baseline**: 299 tests passing, 55.39% coverage (93.4% production code)

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| API (`/predict`, `/predict/batch`, `/health`) | **DONE** | Live on staging, OHE fix deployed |
| Model artifacts | **DONE** | Re-exported with correct medians |
| Staging CI/CD (`staging.yml`) | **DONE** | Auto-deploys on `v0.3.0` push |
| Prediction logging (Supabase) | **PARTIAL** | Code done, prod `DATABASE_URL` not set |
| Monitoring dashboard (Streamlit) | **DONE** | PR #7 merged into v0.3.0 |
| Drift detection (Evidently) | **DONE** | 299 tests, 55.39% coverage |
| Security scanning (Nuclei) | **PR #9** | Nuclei scan in security.yml (full on staging, safe on prod) |
| Uptime monitoring | **USER-MANAGED** | Uptime Kuma — user adds monitor manually |
| E2E tests | **PR #9** | httpx-based E2E tests against staging API |
| Production deployment | **PR #8** | production.yml + dashboard API_ENDPOINT config |
| Local-vs-API validation | **DONE** | `scripts/validate_pipeline.py` — all scores match |
| Test hardening | **PR #10** | OHE determinism, predict error paths, pipeline validation |
| CI fixes | **DONE** | ruff cleanup, monitoring deps, onnx conditional import |

---

## Coverage Analysis

**Post-merge status**: 298 tests, 55.35% coverage (1037 statements, 463 missed)

### Why coverage is only 55%

Two **dead-code modules** (not imported anywhere) account for 432 of the 463 missed lines:

| Module | Lines | Covered | Why uncovered |
|--------|-------|---------|---------------|
| `data/llm_enrichment.py` | 237 | 0% | Training notebook utility, requires OpenAI API |
| `data/utils_data.py` | 195 | 0% | Training notebook utility, MLflow data loading |
| **All other modules** | **605** | **93.4%** | Production code — well tested |

### Excluding dead code, production coverage is 93.4%

The remaining gaps in production code:
- `api/predict.py` (84.7%): error-handling branches (load failure, empty list guard) — **easy to test**
- `db/connection.py` (73.3%): PostgreSQL URL rewrite + get_db yield — **needs mock**
- `api/middleware.py` (94.9%): log write exception — **trivial to test**

### Recommended target: **55%** overall (raises to **65%** after T10-T12)

This corresponds to **95%+ of production code**. The 2 data modules (432 lines, 0%) are training-only notebook utilities — testing them requires OpenAI API mocking with no production benefit.

---

## Tasks

### Track 1: Merge & Stabilize (Opus coordinator) — DONE

| # | Task | Size | Deps | Files | Status |
|---|------|------|------|-------|--------|
| T1 | Merge PR #7 (monitoring) into v0.3.0 | M | — | README.md, pyproject.toml, SESSION_COORDINATION.md | **DONE** |
| T2 | Run full test suite post-merge | S | T1 | tests/* | **DONE** (299 pass) |
| T3 | Raise coverage threshold from 10% to 50% in ci.yml + staging.yml | S | T10-T12 | .github/workflows/ci.yml, .github/workflows/staging.yml | After PR merges |

### Track 2: Production Configuration (Subagent A) — PR #8

| # | Task | Size | Status |
|---|------|------|--------|
| T4 | Fix `dashboard.yml` — add `API_ENDPOINT` env var on Streamlit Space | S | **DONE** |
| T5 | Add production deployment workflow (`production.yml`) | M | **DONE** |
| T6 | Add `DATABASE_URL` secret config to production deploy step | S | **DONE** |

### Track 3: Security & Availability (Subagent B) — PR #9

| # | Task | Size | Status |
|---|------|------|--------|
| T7 | Add Nuclei security scan to `security.yml` (full on staging, safe on prod) | M | **DONE** |
| T8 | ~~Uptime Kuma~~ — **USER-MANAGED** | — | N/A |
| T9 | Add E2E tests: API health + predict round-trip against staging | M | **DONE** |

### Track 4: Test Hardening (Subagent C) — PR #10

| # | Task | Size | Status |
|---|------|------|--------|
| T10 | OHE determinism tests: single vs batch prediction consistency | S | **DONE** |
| T11 | Predict endpoint error path tests | S | **DONE** |
| T12 | validate_pipeline.py tests (--local-only, mock API) | S | **DONE** |

### Track 5: Documentation & Cleanup

| # | Task | Size | Deps | Status |
|---|------|------|------|--------|
| T13 | Update README.md, SESSION_COORDINATION.md, BLUEPRINT | M | T5 | **DONE** |
| T14 | Clean up planning docs: archive SESSION_*_TASKS.md | S | T13 | After final merge |

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
