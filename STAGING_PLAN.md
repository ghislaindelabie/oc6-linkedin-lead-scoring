# Staging Environment Plan

**Goal**: Automatically deploy to a staging environment when PRs merge to the version branch (`v0.3.0`), giving a pre-production validation step before anything reaches `main`.

**Assigned to**: Session A (owns `.github/workflows/`, `Dockerfile`, infrastructure)

---

## Architecture Decision: Same vs. Separate Infrastructure

### Option 1: Same containers, env var switch — NOT RECOMMENDED

Sharing HF Spaces or Supabase between staging and production defeats the purpose. A bad migration or corrupted data in staging would affect production. The evaluator would see through this.

### Option 2: Separate HF Spaces + Separate Supabase — RECOMMENDED

| Resource | Production | Staging |
|----------|-----------|---------|
| API Space | `ghislaindelabie/oc6-bizdev-ml-api` | `ghislaindelabie/oc6-bizdev-ml-api-staging` |
| Dashboard Space | `ghislaindelabie/oc6-bizdev-monitoring` | `ghislaindelabie/oc6-bizdev-monitoring-staging` |
| Database | Supabase project (prod) | Supabase project (staging) OR SQLite on Space |
| APP_ENV | `production` | `staging` |

**Why this works well**:
- HF Spaces free tier allows unlimited public spaces — zero extra cost
- Supabase free tier allows **2 active projects** — one prod, one staging fits perfectly
- The codebase already has SQLite fallback in `alembic/env.py` — so staging can work even without Supabase if needed
- True isolation: bad deploy to staging never touches production
- Evaluator sees a real multi-environment deployment pipeline

**Supabase decision**:

| Sub-option | Pros | Cons |
|-----------|------|------|
| **A. Separate Supabase project** | True isolation, realistic, impressive | Uses 2nd free project slot |
| **B. Same project, `staging_` table prefix** | Saves free slot, simpler secrets | Risk of cross-contamination, fragile naming convention |
| **C. SQLite on HF Space (no Supabase)** | Zero setup, already supported | No persistence across Space restarts, less realistic |

**Recommendation**: **Option A** (separate Supabase project) for evaluator credibility. If the free tier limit is a concern, **Option C** (SQLite) is a perfectly valid staging compromise — staging doesn't need persistent data.

---

## Deployment Flow (Current → Proposed)

### Current

```
feature/* → PR → v0.3.0 → PR → main
                                 ↓
                           Deploy to prod HF Spaces
```

### Proposed

```
feature/* → PR → v0.3.0 ──────────→ PR → main
                   ↓                        ↓
             Deploy to STAGING         Deploy to PROD
             HF Spaces (auto)         HF Spaces (auto)
```

Merges to `v0.3.0` trigger staging deploy. Merges to `main` trigger production deploy (unchanged).

---

## Task Breakdown

### Task 1: Create `staging.yml` workflow (Size: M)

**File**: `.github/workflows/staging.yml`

**Trigger**: Push to `v0.3.0` branch (i.e., when feature PRs are merged into it)

**Jobs**:
1. `test` — Run full test suite (same as ci.yml test job)
2. `deploy-staging-api` — Deploy API to staging HF Space
3. `deploy-staging-dashboard` — Deploy dashboard to staging HF Space
4. `smoke-test` — Hit staging health endpoint after deploy (optional but impressive)

**Logic**: Nearly identical to existing `ci.yml` deploy + `dashboard.yml`, but:
- Different Space names (`*-staging`)
- `APP_ENV=staging`
- Triggered on `v0.3.0` push instead of `main`
- Uses staging-specific secrets for DATABASE_URL

**Dependencies**: None
**Tests needed**: Yes — update `test_ci_workflows.py` to validate staging.yml structure
**Security**: New GitHub secrets needed (see Task 3)

---

### Task 2: Add staging environment support to application code (Size: S)

**Files**: `src/linkedin_lead_scoring/api/main.py` (minor)

**Changes**:
- Ensure `APP_ENV=staging` is recognized (currently only `development` and `production` are checked)
- Add staging banner/indicator to API `/health` response so it's visually distinguishable
- The Dockerfile already accepts `APP_ENV` as env var — no Docker changes needed

**Dependencies**: None
**Tests needed**: Yes — add test for staging env recognition
**Security**: No new concerns — staging uses same CORS and auth as production

---

### Task 3: Configure GitHub Secrets for staging (Size: S — manual step)

**New secrets needed**:

| Secret | Value | Purpose |
|--------|-------|---------|
| `HF_TOKEN` | (existing) | Same token works for both spaces — it's tied to the HF account, not a specific space |
| `STAGING_DATABASE_URL` | Supabase staging connection string | Staging DB (if using separate Supabase) |

**Note**: `HF_TOKEN` doesn't need duplication — one token can push to any space under the same account. The staging workflow just targets different space names.

If using SQLite for staging (Option C), no `STAGING_DATABASE_URL` is needed — the fallback in `alembic/env.py` already handles this.

**Dependencies**: Supabase staging project must be created first (manual)
**Tests needed**: No
**Security**: Never commit these values. Use GitHub Secrets UI only.

---

### Task 4: Create staging Supabase project (Size: S — manual step)

**Steps** (user performs manually):
1. Go to https://supabase.com/dashboard
2. Create new project: `oc6-linkedin-lead-scoring-staging`
3. Copy connection string
4. Add as `STAGING_DATABASE_URL` in GitHub Secrets
5. Alembic migrations will run automatically on first staging deploy

**Dependencies**: None
**Tests needed**: No
**Security**: Connection string stored only in GitHub Secrets

---

### Task 5: Create staging HF Spaces (Size: S — automated in workflow)

The `staging.yml` workflow creates the spaces automatically on first run using:
```bash
huggingface-cli repo create ghislaindelabie/oc6-bizdev-ml-api-staging --type space --space-sdk docker --yes
huggingface-cli repo create ghislaindelabie/oc6-bizdev-monitoring-staging --type space --space-sdk streamlit --yes
```

No manual setup needed — the workflow handles creation idempotently.

**Dependencies**: Task 1 (workflow must exist)
**Tests needed**: No

---

### Task 6: Update documentation (Size: S)

**Files**: `README.md` (Session A section), `docs/MONITORING_GUIDE.md`

**Changes**:
- Document staging URLs
- Add staging section to deployment architecture diagram
- Document the staging → production promotion flow

**Dependencies**: Tasks 1-5
**Tests needed**: No

---

## Execution Strategy

**Linear execution** — all tasks belong to Session A and are small. Parallel execution adds no value.

**Recommended order**:
1. Task 2 — Add `staging` env recognition (tiny code change)
2. Task 1 — Create `staging.yml` workflow (core deliverable)
3. Task 5 — Spaces auto-created on first workflow run
4. Task 3 — Configure secrets (manual, user does this)
5. Task 4 — Create Supabase staging project (manual, user does this)
6. Task 6 — Update docs

---

## Branch Strategy

Session A works on `feature/infra-cicd` (existing branch). This is a natural extension of the infra work already in progress.

---

## Testing Strategy

| Test | File | What it validates |
|------|------|-------------------|
| `test_staging_workflow_exists` | `tests/test_ci_workflows.py` | staging.yml is valid YAML with expected jobs |
| `test_staging_triggers_on_version_branch` | `tests/test_ci_workflows.py` | Trigger is push to v0.3.0 |
| `test_staging_deploys_to_staging_spaces` | `tests/test_ci_workflows.py` | Space names contain `-staging` |
| `test_staging_env_recognized` | `tests/test_api_integration.py` | APP_ENV=staging returns appropriate health response |

---

## What the Evaluator Sees

After implementation, the project demonstrates:

1. **Multi-environment deployment**: staging + production, each with its own infrastructure
2. **Automated promotion pipeline**: feature → version branch (staging) → main (production)
3. **Environment isolation**: separate databases, separate spaces, separate URLs
4. **CI/CD best practices**: tests gate every deployment, health checks validate each deploy
5. **Infrastructure as Code**: everything defined in GitHub Actions workflows, nothing manual except secrets

**Staging URLs** (will be live):
- API: `https://ghislaindelabie-oc6-bizdev-ml-api-staging.hf.space`
- Dashboard: `https://ghislaindelabie-oc6-bizdev-monitoring-staging.hf.space`

**Production URLs** (existing):
- API: `https://ghislaindelabie-oc6-bizdev-ml-api.hf.space`
- Dashboard: `https://ghislaindelabie-oc6-bizdev-monitoring.hf.space`

---

## Risks and Open Questions

| Risk/Question | Impact | Mitigation |
|--------------|--------|------------|
| HF Spaces free tier rate limits on deploys? | Staging deploys on every version branch push | Low risk — deploys are infrequent (only when feature PRs merge) |
| Supabase free tier: only 2 active projects | Can't add a 3rd environment later | Use SQLite for staging if this becomes an issue |
| Staging data accumulation | Disk/DB fills up | Add periodic cleanup or use SQLite (ephemeral on Space restart) |
| **USER DECISION**: Separate Supabase or SQLite for staging? | Affects Task 3 and 4 | Recommendation: separate Supabase (more impressive), but SQLite is fine too |

---

## Summary

| # | Task | Size | Owner | Dependencies |
|---|------|------|-------|-------------|
| 1 | Create `staging.yml` workflow | M | Session A | — |
| 2 | Add `staging` env recognition | S | Session A | — |
| 3 | Configure GitHub Secrets | S | User (manual) | — |
| 4 | Create Supabase staging project | S | User (manual) | — |
| 5 | Staging HF Spaces (auto-created) | S | Workflow | Task 1 |
| 6 | Update documentation | S | Session A | Tasks 1-5 |
