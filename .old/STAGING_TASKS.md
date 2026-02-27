# Staging Environment Tasks — Session A

**Source**: `STAGING_PLAN.md` (project root)
**Branch**: `feature/infra-cicd`
**Priority**: After review fixes (REVIEW_FIX_PLAN.md)

---

## Context

We need a staging environment that deploys automatically when feature PRs merge into the version branch (`v0.3.0`). Production deploys on `main` remain unchanged.

**Architecture**:
- Staging API: `ghislaindelabie/oc6-bizdev-ml-api-staging` (Docker Space)
- Staging Dashboard: `ghislaindelabie/oc6-bizdev-monitoring-staging` (Streamlit Space)
- Staging DB: Separate Supabase project (connection string in `STAGING_DATABASE_URL` secret)
- Spaces are auto-created by the workflow on first run

**Flow**: `feature/* → v0.3.0 (staging deploy) → main (production deploy)`

---

## Task S1: Create `staging.yml` workflow (Size: M)

**File**: `.github/workflows/staging.yml`

**Requirements**:
1. Trigger on push to version branches (`v[0-9]+.[0-9]+.[0-9]+` pattern)
2. Three jobs:
   - `test`: Same lint & test as ci.yml (reuse the same steps)
   - `deploy-staging-api`: Deploy API to staging HF Space (needs: test)
   - `deploy-staging-dashboard`: Deploy dashboard to staging HF Space (needs: test)
3. API deploy job:
   - Same structure as ci.yml `deploy` job but targeting `oc6-bizdev-ml-api-staging`
   - Set `APP_ENV=staging` as a Space variable via HF API
   - Set `DATABASE_URL` from `${{ secrets.STAGING_DATABASE_URL }}` as a Space secret
4. Dashboard deploy job:
   - Same structure as dashboard.yml but targeting `oc6-bizdev-monitoring-staging`
5. Use `huggingface-cli` (not `hf`) for all HF CLI commands — consistent with updated CLI
6. Space creation uses `|| true` (idempotent — space may already exist)

**Secrets used**: `HF_TOKEN` (existing), `STAGING_DATABASE_URL` (new — user will add manually)

**HF Space variables/secrets**: Use the `huggingface_hub` Python API to set environment:
```python
from huggingface_hub import HfApi
api = HfApi()
api.add_space_variable("ghislaindelabie/oc6-bizdev-ml-api-staging", "APP_ENV", "staging")
api.add_space_secret("ghislaindelabie/oc6-bizdev-ml-api-staging", "DATABASE_URL", "<from secret>")
```

---

## Task S2: Add tests for staging.yml (Size: S)

**File**: `tests/test_ci_workflows.py`

Add a `TestStagingWorkflow` class with these tests:
1. `test_staging_yml_exists` — file exists
2. `test_staging_triggers_on_version_branch` — push trigger uses version pattern
3. `test_staging_has_test_job` — test job exists
4. `test_staging_has_deploy_api_job` — deploy-staging-api job exists
5. `test_staging_has_deploy_dashboard_job` — deploy-staging-dashboard job exists
6. `test_staging_deploys_to_staging_spaces` — space names contain `-staging`
7. `test_staging_api_needs_test` — deploy-staging-api depends on test job
8. `test_staging_uses_hf_token` — HF_TOKEN secret is referenced

Follow the existing test patterns in the file (load_yaml helper, same assertion style).

---

## Task S3: Update documentation (Size: S)

After S1 and S2 are done:
1. Add a "Staging" section to the README (Session A section only)
2. Document staging URLs and the promotion flow

---

## Execution Order

1. Task S2 first (TDD — write tests)
2. Task S1 (create the workflow, make tests pass)
3. Task S3 (documentation)
4. Run full test suite, commit, push

---

## Manual Steps (User — NOT Session A)

These are documented for the user to do after Session A commits:
- Create Supabase staging project at https://supabase.com/dashboard
- Add `STAGING_DATABASE_URL` to GitHub repository secrets
- HF Spaces are created automatically by the workflow on first run
