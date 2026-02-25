"""Tests for Dockerfile correctness (static analysis — no docker build required)."""
from pathlib import Path

import pytest

DOCKERFILE_PATH = Path(__file__).parent.parent / "Dockerfile"
DOCKERFILE_STREAMLIT_PATH = Path(__file__).parent.parent / "Dockerfile.streamlit"


@pytest.fixture(scope="module")
def dockerfile_content() -> str:
    """Read the Dockerfile once for all tests in this module."""
    return DOCKERFILE_PATH.read_text()


@pytest.fixture(scope="module")
def dockerfile_lines(dockerfile_content) -> list[str]:
    return dockerfile_content.splitlines()


class TestDockerfileStructure:
    def test_dockerfile_exists(self):
        assert DOCKERFILE_PATH.exists(), "Dockerfile must exist at project root"

    def test_copies_src_directory(self, dockerfile_content):
        assert "src/" in dockerfile_content and "COPY" in dockerfile_content, \
            "Dockerfile must COPY src/ into the image"

    def test_copies_model_directory(self, dockerfile_content):
        assert "model/" in dockerfile_content and "COPY" in dockerfile_content, \
            "Dockerfile must COPY model/ into the image"

    def test_package_importable_via_pythonpath(self, dockerfile_content):
        assert "PYTHONPATH=" in dockerfile_content and "/src" in dockerfile_content, \
            "Dockerfile must set PYTHONPATH to include the src directory"


class TestDockerfileCmd:
    def test_cmd_uses_installed_package_path(self, dockerfile_content):
        """CMD must reference linkedin_lead_scoring.api.main, not src.linkedin_lead_scoring."""
        assert "linkedin_lead_scoring.api.main:app" in dockerfile_content, \
            "CMD must use the installed module path 'linkedin_lead_scoring.api.main:app'"

    def test_cmd_does_not_use_src_prefix(self, dockerfile_content):
        """src. prefix fails when package is installed — must be removed."""
        assert "src.linkedin_lead_scoring.api.main" not in dockerfile_content, \
            "CMD must NOT use 'src.linkedin_lead_scoring.api.main' (package is installed)"

    def test_cmd_uses_correct_port(self, dockerfile_content):
        assert "--port" in dockerfile_content and "7860" in dockerfile_content, \
            "CMD must bind to port 7860 (HF Spaces requirement)"

    def test_cmd_binds_to_all_interfaces(self, dockerfile_content):
        assert "0.0.0.0" in dockerfile_content, \
            "CMD must bind to 0.0.0.0 for container networking"


class TestDockerfileHealthcheck:
    def test_healthcheck_is_present(self, dockerfile_content):
        assert "HEALTHCHECK" in dockerfile_content, "Dockerfile must define a HEALTHCHECK"

    def test_healthcheck_does_not_use_requests_library(self, dockerfile_content):
        """requests is not in requirements-prod.txt — healthcheck must use stdlib."""
        # Simple check: 'import requests' must not appear in HEALTHCHECK context
        hc_block = dockerfile_content[dockerfile_content.find("HEALTHCHECK"):]
        # Only look at text until the next Dockerfile instruction
        next_instr = -1
        for keyword in ["CMD", "RUN", "COPY", "ENV", "EXPOSE", "USER", "FROM", "ARG"]:
            idx = hc_block.find(f"\n{keyword}", 1)
            if idx != -1 and (next_instr == -1 or idx < next_instr):
                next_instr = idx
        hc_section = hc_block[:next_instr] if next_instr != -1 else hc_block
        assert "import requests" not in hc_section, \
            "HEALTHCHECK must not use 'import requests' (not in requirements-prod.txt)"

    def test_healthcheck_checks_health_endpoint(self, dockerfile_content):
        assert "/health" in dockerfile_content, \
            "HEALTHCHECK must probe the /health endpoint"


class TestDockerfileAlembic:
    def test_runs_alembic_migrations_before_startup(self, dockerfile_content):
        """alembic upgrade head must run before uvicorn starts."""
        alembic_pos = dockerfile_content.find("alembic upgrade head")
        uvicorn_pos = dockerfile_content.find("uvicorn")
        assert alembic_pos != -1, \
            "Dockerfile must run 'alembic upgrade head' before starting the server"
        assert alembic_pos < uvicorn_pos, \
            "'alembic upgrade head' must appear before 'uvicorn' in the Dockerfile"


# ---------------------------------------------------------------------------
# Dockerfile.streamlit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def streamlit_dockerfile_content() -> str:
    """Read Dockerfile.streamlit once for all tests in this module."""
    return DOCKERFILE_STREAMLIT_PATH.read_text()


class TestDockerfileStreamlitSecurity:
    def test_streamlit_dockerfile_exists(self):
        assert DOCKERFILE_STREAMLIT_PATH.exists(), \
            "Dockerfile.streamlit must exist at project root"

    def test_creates_non_root_user(self, streamlit_dockerfile_content):
        """Dockerfile.streamlit must create a non-root user (UID 1000)."""
        assert "useradd" in streamlit_dockerfile_content, \
            "Dockerfile.streamlit must create a non-root user with useradd"

    def test_switches_to_non_root_user(self, streamlit_dockerfile_content):
        """Dockerfile.streamlit must switch to non-root user via USER directive."""
        lines = streamlit_dockerfile_content.splitlines()
        user_lines = [l.strip() for l in lines if l.strip().startswith("USER")]
        assert any("user" in l.lower() and "root" not in l.lower()
                    for l in user_lines), \
            "Dockerfile.streamlit must have a USER directive for a non-root user"

    def test_copy_uses_chown(self, streamlit_dockerfile_content):
        """COPY directives after USER switch should use --chown."""
        lines = streamlit_dockerfile_content.splitlines()
        found_user = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("USER") and "root" not in stripped.lower():
                found_user = True
            if found_user and stripped.startswith("COPY"):
                assert "--chown=" in stripped, \
                    f"COPY after USER must use --chown: {stripped}"
