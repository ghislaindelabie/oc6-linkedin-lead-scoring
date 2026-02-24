"""Static validation tests for GitHub Actions workflow YAML files."""
from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(__file__).parent.parent / ".github" / "workflows"
CI_YML = WORKFLOWS_DIR / "ci.yml"
SECURITY_YML = WORKFLOWS_DIR / "security.yml"
DASHBOARD_YML = WORKFLOWS_DIR / "dashboard.yml"
STAGING_YML = WORKFLOWS_DIR / "staging.yml"
PRODUCTION_YML = WORKFLOWS_DIR / "production.yml"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# ci.yml tests
# ---------------------------------------------------------------------------

class TestCiWorkflow:
    def test_ci_yml_exists(self):
        assert CI_YML.exists(), "ci.yml must exist"

    def test_ci_has_test_job(self):
        data = load_yaml(CI_YML)
        assert "test" in data["jobs"], "ci.yml must have a 'test' job"

    def test_ci_test_job_runs_pytest(self):
        data = load_yaml(CI_YML)
        steps = data["jobs"]["test"]["steps"]
        run_commands = " ".join(s.get("run", "") for s in steps)
        assert "pytest" in run_commands, "test job must run pytest"

    def test_ci_test_job_enforces_coverage_threshold(self):
        data = load_yaml(CI_YML)
        steps = data["jobs"]["test"]["steps"]
        run_commands = " ".join(s.get("run", "") for s in steps)
        assert "--cov-fail-under" in run_commands, \
            "test job must enforce coverage with --cov-fail-under"
        assert "cov-fail-under=10" in run_commands, "coverage threshold must be 10%"

    def test_ci_test_job_has_lint_step(self):
        data = load_yaml(CI_YML)
        steps = data["jobs"]["test"]["steps"]
        run_commands = " ".join(s.get("run", "") for s in steps)
        assert "ruff" in run_commands, "test job must include a ruff lint step"

    def test_ci_has_docker_build_job(self):
        data = load_yaml(CI_YML)
        assert "docker-build" in data["jobs"], \
            "ci.yml must have a 'docker-build' job"

    def test_ci_docker_build_only_on_prs(self):
        data = load_yaml(CI_YML)
        job = data["jobs"]["docker-build"]
        condition = job.get("if", "")
        assert "pull_request" in str(condition) or "github.event_name" in str(condition), \
            "docker-build job must only run on pull_request events"

    def test_ci_docker_build_runs_docker_build(self):
        data = load_yaml(CI_YML)
        steps = data["jobs"]["docker-build"]["steps"]
        run_commands = " ".join(s.get("run", "") for s in steps)
        assert "docker build" in run_commands, \
            "docker-build job must run 'docker build'"

    def test_ci_uses_python_311(self):
        data = load_yaml(CI_YML)
        steps = data["jobs"]["test"]["steps"]
        python_versions = []
        for s in steps:
            with_data = s.get("with", {}) or {}
            if "python-version" in with_data:
                python_versions.append(str(with_data["python-version"]))
        assert any("3.11" in v for v in python_versions), \
            "CI must use Python 3.11"

    def test_ci_deploy_job_exists(self):
        data = load_yaml(CI_YML)
        assert "deploy" in data["jobs"], "deploy job must be preserved"

    def test_ci_deploy_only_on_main(self):
        data = load_yaml(CI_YML)
        condition = str(data["jobs"]["deploy"].get("if", ""))
        assert "main" in condition, "deploy job must only run on main branch"


# ---------------------------------------------------------------------------
# security.yml tests
# ---------------------------------------------------------------------------

class TestSecurityWorkflow:
    def test_security_yml_exists(self):
        assert SECURITY_YML.exists(), "security.yml must exist"

    def test_security_has_schedule_trigger(self):
        data = load_yaml(SECURITY_YML)
        triggers = data.get("on", data.get(True, {}))
        assert "schedule" in triggers, \
            "security.yml must have a scheduled trigger"

    def test_security_has_manual_trigger(self):
        data = load_yaml(SECURITY_YML)
        triggers = data.get("on", data.get(True, {}))
        assert "workflow_dispatch" in triggers, \
            "security.yml must support manual trigger (workflow_dispatch)"

    def test_security_has_pip_audit_step(self):
        data = load_yaml(SECURITY_YML)
        all_steps = []
        for job in data["jobs"].values():
            all_steps.extend(job.get("steps", []))
        run_commands = " ".join(s.get("run", "") for s in all_steps)
        assert "pip-audit" in run_commands, \
            "security.yml must run pip-audit"

    def test_security_pip_audit_scans_requirements(self):
        data = load_yaml(SECURITY_YML)
        all_steps = []
        for job in data["jobs"].values():
            all_steps.extend(job.get("steps", []))
        run_commands = " ".join(s.get("run", "") for s in all_steps)
        assert "requirements-prod.txt" in run_commands, \
            "pip-audit must scan requirements-prod.txt"

    def test_security_has_bandit_step(self):
        data = load_yaml(SECURITY_YML)
        all_steps = []
        for job in data["jobs"].values():
            all_steps.extend(job.get("steps", []))
        run_commands = " ".join(s.get("run", "") for s in all_steps)
        assert "bandit" in run_commands, \
            "security.yml must run bandit"

    def test_security_uses_python_311(self):
        data = load_yaml(SECURITY_YML)
        all_steps = []
        for job in data["jobs"].values():
            all_steps.extend(job.get("steps", []))
        python_versions = []
        for s in all_steps:
            with_data = s.get("with", {}) or {}
            if "python-version" in with_data:
                python_versions.append(str(with_data["python-version"]))
        assert any("3.11" in v for v in python_versions), \
            "security.yml must use Python 3.11"

    def test_security_has_nuclei_job(self):
        data = load_yaml(SECURITY_YML)
        assert "nuclei" in data["jobs"], \
            "security.yml must have a 'nuclei' job"

    def test_security_nuclei_scans_staging_and_production(self):
        data = load_yaml(SECURITY_YML)
        nuclei_job = data["jobs"]["nuclei"]
        strategy = nuclei_job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        include = matrix.get("include", [])
        targets = [entry.get("target", "") for entry in include]
        names = [entry.get("name", "") for entry in include]
        assert any("staging" in t for t in targets), \
            "nuclei job must scan the staging target"
        assert any("staging" in n for n in names), \
            "nuclei job matrix must include a 'staging' entry"
        assert any(
            "staging" not in t and "hf.space" in t for t in targets
        ), "nuclei job must also scan the production target"

    def test_security_nuclei_uploads_results(self):
        data = load_yaml(SECURITY_YML)
        nuclei_steps = data["jobs"]["nuclei"].get("steps", [])
        uses_values = " ".join(s.get("uses", "") for s in nuclei_steps)
        assert "upload-artifact" in uses_values, \
            "nuclei job must upload scan results as artifacts"


# ---------------------------------------------------------------------------
# dashboard.yml tests
# ---------------------------------------------------------------------------

class TestDashboardWorkflow:
    def test_dashboard_yml_exists(self):
        assert DASHBOARD_YML.exists(), "dashboard.yml must exist"

    def test_dashboard_triggers_on_main_push(self):
        data = load_yaml(DASHBOARD_YML)
        triggers = data.get("on", data.get(True, {}))
        push = triggers.get("push", {})
        branches = push.get("branches", [])
        assert "main" in branches, \
            "dashboard.yml must trigger on push to main"

    def test_dashboard_targets_monitoring_hf_space(self):
        data = load_yaml(DASHBOARD_YML)
        all_steps = []
        for job in data["jobs"].values():
            all_steps.extend(job.get("steps", []))
        run_commands = " ".join(s.get("run", "") for s in all_steps)
        env_values = " ".join(
            str(v) for s in all_steps
            for v in (s.get("env", {}) or {}).values()
        )
        combined = run_commands + " " + env_values
        assert "oc6-bizdev-monitoring" in combined, \
            "dashboard.yml must target the oc6-bizdev-monitoring HF Space"

    def test_dashboard_uses_hf_token(self):
        data = load_yaml(DASHBOARD_YML)
        all_steps = []
        for job in data["jobs"].values():
            all_steps.extend(job.get("steps", []))
        env_keys = " ".join(
            k for s in all_steps
            for k in (s.get("env", {}) or {}).keys()
        )
        assert "HF_TOKEN" in env_keys, \
            "dashboard.yml must use HF_TOKEN secret"


# ---------------------------------------------------------------------------
# staging.yml tests
# ---------------------------------------------------------------------------

class TestStagingWorkflow:
    def test_staging_yml_exists(self):
        assert STAGING_YML.exists(), "staging.yml must exist"

    def test_staging_triggers_on_version_branch(self):
        data = load_yaml(STAGING_YML)
        triggers = data.get("on", data.get(True, {}))
        push = triggers.get("push", {})
        branches = push.get("branches", [])
        assert any("v" in str(b) and "[0-9]" in str(b) for b in branches), \
            "staging.yml must trigger on version branch pattern (e.g. v[0-9]+.*)"

    def test_staging_has_test_job(self):
        data = load_yaml(STAGING_YML)
        assert "test" in data["jobs"], "staging.yml must have a 'test' job"

    def test_staging_has_deploy_api_job(self):
        data = load_yaml(STAGING_YML)
        assert "deploy-staging-api" in data["jobs"], \
            "staging.yml must have a 'deploy-staging-api' job"

    def test_staging_has_deploy_dashboard_job(self):
        data = load_yaml(STAGING_YML)
        assert "deploy-staging-dashboard" in data["jobs"], \
            "staging.yml must have a 'deploy-staging-dashboard' job"

    def test_staging_deploys_to_staging_spaces(self):
        data = load_yaml(STAGING_YML)
        all_steps = []
        for job in data["jobs"].values():
            all_steps.extend(job.get("steps", []))
        env_values = " ".join(
            str(v) for s in all_steps
            for v in (s.get("env", {}) or {}).values()
        )
        run_commands = " ".join(s.get("run", "") for s in all_steps)
        combined = env_values + " " + run_commands
        assert "ml-api-staging" in combined or "monitoring-staging" in combined, \
            "staging.yml must target HF Spaces with '-staging' suffix"

    def test_staging_api_needs_test(self):
        data = load_yaml(STAGING_YML)
        needs = data["jobs"]["deploy-staging-api"].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "test" in needs, \
            "deploy-staging-api must depend on the test job"

    def test_staging_uses_hf_token(self):
        data = load_yaml(STAGING_YML)
        all_steps = []
        for job in data["jobs"].values():
            all_steps.extend(job.get("steps", []))
        env_keys = " ".join(
            k for s in all_steps
            for k in (s.get("env", {}) or {}).keys()
        )
        assert "HF_TOKEN" in env_keys, \
            "staging.yml must use HF_TOKEN secret"

    def test_staging_has_e2e_tests_job(self):
        data = load_yaml(STAGING_YML)
        assert "e2e-tests" in data["jobs"], \
            "staging.yml must have an 'e2e-tests' job"

    def test_staging_e2e_needs_deploy_jobs(self):
        data = load_yaml(STAGING_YML)
        e2e_job = data["jobs"]["e2e-tests"]
        needs = e2e_job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "deploy-staging-api" in needs, \
            "e2e-tests must depend on deploy-staging-api"

    def test_staging_e2e_runs_pytest(self):
        data = load_yaml(STAGING_YML)
        e2e_steps = data["jobs"]["e2e-tests"].get("steps", [])
        run_commands = " ".join(s.get("run", "") for s in e2e_steps)
        assert "pytest" in run_commands, \
            "e2e-tests job must run pytest"

    def test_staging_e2e_sets_staging_url(self):
        data = load_yaml(STAGING_YML)
        e2e_steps = data["jobs"]["e2e-tests"].get("steps", [])
        env_keys = " ".join(
            k for s in e2e_steps
            for k in (s.get("env", {}) or {}).keys()
        )
        assert "STAGING_URL" in env_keys, \
            "e2e-tests job must set STAGING_URL environment variable"

    def test_staging_e2e_sets_dashboard_url(self):
        data = load_yaml(STAGING_YML)
        e2e_steps = data["jobs"]["e2e-tests"].get("steps", [])
        env_keys = " ".join(
            k for s in e2e_steps
            for k in (s.get("env", {}) or {}).keys()
        )
        assert "DASHBOARD_URL" in env_keys, \
            "e2e-tests job must set DASHBOARD_URL environment variable"

    def test_staging_dashboard_sets_api_endpoint(self):
        content = STAGING_YML.read_text()
        assert "API_ENDPOINT" in content, \
            "staging.yml deploy-staging-dashboard must set API_ENDPOINT variable"


# ---------------------------------------------------------------------------
# production.yml tests
# ---------------------------------------------------------------------------

class TestProductionWorkflow:
    def test_production_yml_exists(self):
        assert PRODUCTION_YML.exists(), "production.yml must exist"

    def test_production_yml_is_valid_yaml(self):
        data = load_yaml(PRODUCTION_YML)
        assert isinstance(data, dict), "production.yml must be valid YAML"

    def test_production_triggers_on_main_push(self):
        data = load_yaml(PRODUCTION_YML)
        triggers = data.get("on", data.get(True, {}))
        push = triggers.get("push", {})
        branches = push.get("branches", [])
        assert "main" in branches, \
            "production.yml must trigger on push to main"

    def test_production_has_deploy_api_job(self):
        data = load_yaml(PRODUCTION_YML)
        assert "deploy-production-api" in data["jobs"], \
            "production.yml must have a 'deploy-production-api' job"

    def test_production_has_deploy_dashboard_job(self):
        data = load_yaml(PRODUCTION_YML)
        assert "deploy-production-dashboard" in data["jobs"], \
            "production.yml must have a 'deploy-production-dashboard' job"

    def test_production_api_targets_production_space(self):
        data = load_yaml(PRODUCTION_YML)
        all_steps = []
        for job in data["jobs"].values():
            all_steps.extend(job.get("steps", []))
        run_commands = " ".join(s.get("run", "") for s in all_steps)
        assert "oc6-bizdev-ml-api" in run_commands, \
            "production.yml must target the oc6-bizdev-ml-api HF Space"

    def test_production_dashboard_targets_monitoring_space(self):
        data = load_yaml(PRODUCTION_YML)
        all_steps = []
        for job in data["jobs"].values():
            all_steps.extend(job.get("steps", []))
        run_commands = " ".join(s.get("run", "") for s in all_steps)
        assert "oc6-bizdev-monitoring" in run_commands, \
            "production.yml must target the oc6-bizdev-monitoring HF Space"

    def test_production_api_sets_app_env_production(self):
        content = PRODUCTION_YML.read_text()
        assert "APP_ENV" in content and "production" in content, \
            "production.yml API job must set APP_ENV=production"

    def test_production_api_sets_cors_origins(self):
        content = PRODUCTION_YML.read_text()
        assert "CORS_ORIGINS" in content, \
            "production.yml API job must configure CORS_ORIGINS"

    def test_production_api_sets_database_url_secret(self):
        content = PRODUCTION_YML.read_text()
        assert "DATABASE_URL" in content, \
            "production.yml API job must configure DATABASE_URL secret"

    def test_production_dashboard_sets_api_endpoint(self):
        content = PRODUCTION_YML.read_text()
        assert "API_ENDPOINT" in content, \
            "production.yml dashboard job must set API_ENDPOINT variable"

    def test_production_uses_hf_token(self):
        data = load_yaml(PRODUCTION_YML)
        all_steps = []
        for job in data["jobs"].values():
            all_steps.extend(job.get("steps", []))
        env_keys = " ".join(
            k for s in all_steps
            for k in (s.get("env", {}) or {}).keys()
        )
        assert "HF_TOKEN" in env_keys, \
            "production.yml must use HF_TOKEN secret"

    def test_production_has_smoke_tests_job(self):
        data = load_yaml(PRODUCTION_YML)
        assert "smoke-tests" in data["jobs"], \
            "production.yml must have a 'smoke-tests' job"

    def test_production_smoke_tests_needs_deploy_jobs(self):
        data = load_yaml(PRODUCTION_YML)
        needs = data["jobs"]["smoke-tests"].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "deploy-production-api" in needs, \
            "smoke-tests must depend on deploy-production-api"
        assert "deploy-production-dashboard" in needs, \
            "smoke-tests must depend on deploy-production-dashboard"

    def test_production_smoke_checks_both_services(self):
        data = load_yaml(PRODUCTION_YML)
        smoke_steps = data["jobs"]["smoke-tests"].get("steps", [])
        run_commands = " ".join(s.get("run", "") for s in smoke_steps)
        assert "oc6-bizdev-ml-api.hf.space/health" in run_commands, \
            "smoke-tests must check API health"
        assert "oc6-bizdev-monitoring.hf.space" in run_commands, \
            "smoke-tests must check dashboard health"


# ---------------------------------------------------------------------------
# dashboard.yml API_ENDPOINT tests
# ---------------------------------------------------------------------------

class TestDashboardApiEndpoint:
    def test_dashboard_sets_api_endpoint(self):
        content = DASHBOARD_YML.read_text()
        assert "API_ENDPOINT" in content, \
            "dashboard.yml must set API_ENDPOINT variable for the monitoring space"
