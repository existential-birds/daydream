"""Contract tests for the enforced coverage gate (#932).

These pin the *configuration* — the enforcement surface itself is the
make-check / pre-push / CI pipeline, which cannot be asserted in-process.
"""

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def test_pytest_cov_is_exact_pinned_in_dev_group(pyproject: dict) -> None:
    pins = [d for d in pyproject["dependency-groups"]["dev"] if d.startswith("pytest-cov")]
    assert pins == ["pytest-cov==7.1.0"], f"expected exact pin 7.1.0, got {pins}"


def test_coverage_config_targets_daydream_branch_and_omits_atif(pyproject: dict) -> None:
    cov = pyproject["tool"]["coverage"]
    run = cov["run"]
    assert run["branch"] is True
    assert run["source"] == ["daydream"]
    assert run["omit"] == ["daydream/atif/*"]
    # XML report lands where CI's upload step expects it.
    assert cov["report"]["xml"] is True or "xml" in cov


def test_pytest_addopts_keeps_strict_markers_and_adds_cov(pyproject: dict) -> None:
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--strict-markers" in addopts, "existing strictness must be preserved"
    assert "--cov" in addopts
    assert "--cov-branch" in addopts
    assert "--cov-report=term-missing" in addopts
    assert "--cov-report=xml" in addopts


def test_coverage_artifacts_are_gitignored() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert ".coverage" in gitignore
    assert "coverage.xml" in gitignore


def test_fail_under_is_enforced_in_single_config_location(pyproject: dict) -> None:
    report = pyproject["tool"]["coverage"]["report"]
    value = report["fail_under"]
    assert isinstance(value, int) and 0 < value < 100
    # fail_under must NOT also be passed on any CLI surface — one config location.
    makefile = (REPO_ROOT / "Makefile").read_text()
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
    assert "fail_under" not in makefile and "--cov-fail-under" not in makefile
    assert "fail_under" not in ci and "--cov-fail-under" not in ci
