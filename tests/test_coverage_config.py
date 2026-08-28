"""Contract tests for the enforced coverage gate (#932).

These pin the *configuration* — the enforcement surface itself is the
make-check / pre-push / CI pipeline, which cannot be asserted in-process.
"""

import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def test_pytest_cov_is_exact_pinned_in_dev_group(pyproject: dict[str, Any]) -> None:
    pins = [d for d in pyproject["dependency-groups"]["dev"] if d.startswith("pytest-cov")]
    assert pins == ["pytest-cov==7.1.0"], f"expected exact pin 7.1.0, got {pins}"


def test_coverage_config_targets_daydream_branch_and_omits_atif(pyproject: dict[str, Any]) -> None:
    cov = pyproject["tool"]["coverage"]
    run = cov["run"]
    assert run["branch"] is True
    assert run["source"] == ["daydream"]
    assert run["omit"] == ["daydream/atif/*"]
    # XML report lands where CI's upload step expects it.
    assert "xml" in cov and cov["xml"].get("output") == "coverage.xml"


def test_coverage_flags_live_on_gated_invocations_not_global_addopts(
    pyproject: dict[str, Any],
) -> None:
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--strict-markers" in addopts, "existing strictness must be preserved"
    # Issue #336: coverage flags cannot live in global addopts — a targeted
    # subset run (`pytest tests/foo.py`) would inherit --cov and trip fail_under
    # even when all its tests pass. They belong on the full-suite invocations.
    for flag in ("--cov", "--cov-branch", "--cov-report=term-missing", "--cov-report=xml"):
        assert flag not in addopts, f"coverage flag {flag!r} must not be in global addopts"


def test_coverage_artifacts_are_gitignored() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert ".coverage" in gitignore
    assert "coverage.xml" in gitignore


def test_fail_under_is_enforced_in_single_config_location(pyproject: dict[str, Any]) -> None:
    report = pyproject["tool"]["coverage"]["report"]
    value = report["fail_under"]
    assert isinstance(value, int) and 0 < value < 100
    # fail_under must NOT also be passed on any CLI surface — one config location.
    makefile = (REPO_ROOT / "Makefile").read_text()
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
    assert "fail_under" not in makefile and "--cov-fail-under" not in makefile
    assert "fail_under" not in ci and "--cov-fail-under" not in ci


def test_make_test_target_runs_coverage_enabled_pytest() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text()
    test_recipe = makefile.split("test:\n", 1)[1].split("\n\n", 1)[0]
    assert "uv run pytest -n auto" in test_recipe, "test target must keep xdist parallelism"
    # Coverage flags moved here (from global addopts) so `make test` still gates
    # the full suite while bare/targeted pytest stays plain (#336).
    for flag in ("--cov", "--cov-branch", "--cov-report=term-missing", "--cov-report=xml"):
        assert flag in test_recipe, f"test target must carry coverage flag {flag!r}"


def test_check_target_includes_test_and_coverage_report() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text()
    check_line = next(line for line in makefile.splitlines() if line.startswith("check:"))
    assert "test" in check_line
    assert "coverage-report" in check_line
    assert makefile.count("coverage-report:") == 1


def test_ci_uploads_coverage_xml_artifact() -> None:
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
    assert "coverage.xml" in ci, "check job must upload coverage.xml"
    assert "actions/upload-artifact@" in ci
    # All actions in this repo are SHA-pinned; the artifact action must be too.
    for line in ci.splitlines():
        if "actions/upload-artifact@" in line:
            assert "#" in line and len(line.split("#", 1)[1].strip().split(" ")[0]) == 40, (
                f"upload-artifact must be SHA-pinned with version comment: {line!r}"
            )


def test_fail_under_is_a_measured_whole_percent(pyproject: dict[str, Any]) -> None:
    value = pyproject["tool"]["coverage"]["report"]["fail_under"]
    # Provisional 0 from Task 2 must never ship; the floor is a measured value.
    assert isinstance(value, int) and 1 <= value <= 99, (
        f"fail_under={value!r} is not a measured whole percent — run the Task 6 baseline"
    )


def test_ci_test_step_carries_same_coverage_invocation_as_local() -> None:
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
    # The check job's Run tests step carries the identical coverage flags as the
    # local Makefile test target (local == CI), NOT via global addopts — so
    # targeted runs stay plain and never trip the floor (#336).
    check_block = ci.split("Run tests", 1)[1].split("- name:", 1)[0]
    assert "uv run pytest -n auto --cov --cov-branch --cov-report=term-missing --cov-report=xml" in check_block


def test_coverage_docs_present_with_ratchet_and_local_command() -> None:
    docs = (REPO_ROOT / "docs" / "coverage.md").read_text()
    assert "uv run pytest -n auto" in docs, "local coverage command must be documented"
    assert "fail_under" in docs and "ratchet" in docs.lower()
    assert "daydream/atif" in docs, "rationale for the vendored exclusion must be restated"
    readme = (REPO_ROOT / "README.md").read_text()
    assert "docs/coverage.md" in readme
