"""Phase 5: the green-baseline gate, exercised against real docker builds.

The gate is the only thing standing between `fix_tests_pass` and noise: it pays a
rollout for a suite that passes after its fix, which means nothing at all if the
suite was already failing before the agent touched anything. So the test that
matters is not "a good repo builds" but "a red one does NOT" — and it has to be a
real `docker build`, because the enforcement IS the build failing.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from conftest import PROJECT_ROOT

from daydream_review_v1.fixture import FIXTURE_SLUG

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")

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
