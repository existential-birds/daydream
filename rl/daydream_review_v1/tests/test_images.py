"""Container image contracts: immutable base inputs and the green-baseline gate.

Two kinds of contract live here. The static ``base.Dockerfile`` checks run with
no Docker at all — they assert the build pins every remote input to an immutable
version and verifies it for integrity before use. The two ``slow`` tests execute
real Docker builds for the red and green baseline paths: a repo whose suite is
red at the head commit must produce no image, while a green one builds and bakes
the checkout.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest
from conftest import PROJECT_ROOT

from daydream_review_v1.fixture import FIXTURE_SLUG

DOCKER_REQUIRED = pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")

FIXTURE_IMAGE = "daydream-rl/fixture"


def _build(*args: str) -> subprocess.CompletedProcess[str]:
    """Build the fixture repo image only; the ``base_image`` fixture owns the base."""
    return subprocess.run(
        ["uv", "run", "python", "images/build_images.py", "--only", FIXTURE_SLUG, "--no-base", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _tags() -> set[str]:
    listed = subprocess.run(
        ["docker", "images", f"{FIXTURE_IMAGE}", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {line.strip() for line in listed.stdout.splitlines() if line.strip()}


@pytest.mark.slow
@DOCKER_REQUIRED
def test_green_baseline_gate_fails_the_build_on_a_red_suite(base_image: str) -> None:
    """A repository whose suite is red at the head commit must produce NO image."""
    before = _tags()
    result = _build("--red")
    combined = result.stdout + result.stderr

    assert result.returncode != 0, "a red baseline built successfully — the gate is not enforcing"
    # The failure has to come from the test layer, not from an earlier step: a
    # build that died at `git checkout` would fail for the wrong reason and prove
    # nothing about the gate.
    assert "${TEST_COMMAND}" in combined, combined[-3000:]
    assert "FAILED (failures=1)" in combined, combined[-3000:]

    # `--red` rewrites the head commit, so its snapshot tag is one that did not
    # exist before; the gate must not have left it behind.
    assert _tags() <= before, f"a tag was published for a red baseline: {_tags() - before}"


@pytest.mark.slow
@DOCKER_REQUIRED
def test_green_baseline_builds_and_bakes_the_checkout(base_image: str) -> None:
    """The happy path, end to end: image builds, suite green, origin is local."""
    result = _build()
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]

    tag = f"{FIXTURE_IMAGE}:{'9b92381663058612621b186545f91bfb3a54079c'[:12]}"
    assert tag in _tags(), f"{tag} not built; have {_tags()}"

    probe = subprocess.run(
        [
            "docker", "run", "--rm", tag, "sh", "-c",
            "cd /work/repo && git rev-parse HEAD && git remote get-url origin && python -m unittest discover -q",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "9b92381663058612621b186545f91bfb3a54079c" in probe.stdout
    # origin is the in-container mirror, so daydream's terminal push stays inside
    # the container and no rollout needs a credential.
    assert "/srv/mirror.git" in probe.stdout


def test_docker_skip_is_per_test_not_module_wide() -> None:
    """M8 regression: the Docker skip is per-test, not module-wide, so the static
    build-contract tests (added in the next task) collect in a Docker-less CI."""
    module = sys.modules[__name__]
    assert "pytestmark" not in vars(module), "module-wide Docker skip would gate the static tests"
    assert "DOCKER_REQUIRED" in vars(module), "per-test DOCKER_REQUIRED marker missing"
    # Both slow integration tests must carry the skip; neither may rely on a
    # module-level marker that would also skip the static tests.
    assert getattr(test_green_baseline_gate_fails_the_build_on_a_red_suite, "pytestmark", None)
    assert getattr(test_green_baseline_builds_and_bakes_the_checkout, "pytestmark", None)
