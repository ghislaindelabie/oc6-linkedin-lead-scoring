"""Static validation tests for GitHub Actions workflow YAML files."""
from pathlib import Path

import pytest
import yaml

WORKFLOWS_DIR = Path(__file__).parent.parent / ".github" / "workflows"
CI_YML = WORKFLOWS_DIR / "ci.yml"
SECURITY_YML = WORKFLOWS_DIR / "security.yml"
DASHBOARD_YML = WORKFLOWS_DIR / "dashboard.yml"
STAGING_YML = WORKFLOWS_DIR / "staging.yml"


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
        assert "70" in run_commands, "coverage threshold must be 70%"

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
