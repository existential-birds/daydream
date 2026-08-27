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
