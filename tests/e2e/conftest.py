"""E2E test configuration.

This conftest is intentionally minimal — E2E tests only need httpx
to make HTTP calls against live staging/production URLs. The root
tests/conftest.py imports numpy, fastapi, and the application code,
which are not available in the lightweight E2E CI environment.
"""
