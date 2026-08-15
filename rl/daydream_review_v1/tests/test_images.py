"""Container image contracts: immutable base inputs and the green-baseline gate.

Two kinds of contract live here. The static ``base.Dockerfile`` checks run with
no Docker at all — they assert the build pins every remote input to an immutable
version and verifies it for integrity before use. The two ``slow`` tests execute
real Docker builds for the red and green baseline paths: a repo whose suite is
red at the head commit must produce no image, while a green one builds and bakes
the checkout.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from conftest import PROJECT_ROOT, docker_daemon_is_available

from daydream_review_v1.fixture import FIXTURE_SLUG

DOCKER_REQUIRED = pytest.mark.skipif(
    not docker_daemon_is_available(),
    reason="docker is not installed or the daemon is unavailable",
)

FIXTURE_IMAGE = "daydream-rl/fixture"

BASE_DOCKERFILE = PROJECT_ROOT / "images" / "base.Dockerfile"


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


def test_docker_required_gates_on_daemon_reachability() -> None:
    """The per-test Docker skip must gate on daemon reachability, not client presence."""
    module = sys.modules[__name__]
    # The skip condition keys on the reachability predicate imported from conftest...
    assert "docker_daemon_is_available" in vars(module), "reachability predicate import missing"
    # ...and the stale client-presence probe is gone.
    assert "shutil" not in vars(module), "stale shutil.which probe remains"
    assert (
        DOCKER_REQUIRED.mark.kwargs["reason"]
        == "docker is not installed or the daemon is unavailable"
    )


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


@pytest.mark.parametrize(
    "required_literal",
    [
        pytest.param(
            "FROM python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36",
            id="python-base-index-digest",
        ),
        pytest.param("ARG UV_VERSION=0.11.29", id="uv-pin"),
        pytest.param('pip install --no-cache-dir "uv==${UV_VERSION}"', id="uv-exact-install"),
        pytest.param("ARG CLAUDE_CODE_VERSION=2.1.214", id="claude-version"),
        pytest.param(
            "release_fingerprint=31DDDE24DDFAB679F42D7BD2BAA929FF1A7ECACE",
            id="claude-release-fingerprint",
        ),
        pytest.param(
            (
                "ARG CODEX_VERSION=0.145.0",
                (
                    "amd64) target=x86_64-unknown-linux-musl; "
                    "checksum=bfaf13c9ba34f2ad764e4a916c49cf7177aeba329cf0f719e2227566fc8d662a ;;"
                ),
                (
                    "arm64) target=aarch64-unknown-linux-musl; "
                    "checksum=d384f90bc842450b42bd675feef06a12a46a3b1ca97efcb22566b270e4a11227 ;;"
                ),
            ),
            id="codex-version-and-checksums",
        ),
        pytest.param(
            (
                "ARG NODE_VERSION=22.17.1",
                "amd64) node_arch=x64; checksum=cfb6ac0cf339825fe36efd1f18a79016b02aca19fbfa6c9547c57e27dc09f6ea ;;",
                "arm64) node_arch=arm64; checksum=f53510706998cf044f634190416f0588e7e1937aecea938768952e0f0ac1f41b ;;",
            ),
            id="node-version-and-checksums",
        ),
        pytest.param("ARG PI_VERSION=0.82.1", id="pi-version"),
    ],
)
def test_base_dockerfile_pins_immutable_versions_and_checksums(
    required_literal: str | tuple[str, ...],
) -> None:
    """M1/M2/M3/M4 pin contract: every immutable identifier is present verbatim.

    Each version ARG that has inline checksums is bundled into a single
    parameter tuple so that bumping a version without updating its matching
    checksum(s) fails the test.
    """
    text = BASE_DOCKERFILE.read_text(encoding="utf-8")
    if isinstance(required_literal, str):
        assert required_literal in text, f"missing pinned literal {required_literal!r}"
    else:
        # Tuple: first element is the version ARG, remaining are checksums.
        # Assert ordering: version ARG appears before each checksum, so a
        # version bump cannot land without its matching checksum update.
        assert required_literal[0] in text, f"missing pinned literal {required_literal[0]!r}"
        pos = text.find(required_literal[0])
        for literal in required_literal[1:]:
            nxt = text.find(literal, pos + 1)
            assert nxt > pos, (
                f"checksum {literal!r} must appear after {required_literal[0]!r}"
            )
            pos = nxt


@pytest.mark.parametrize(
    "marker_chain",
    [
        pytest.param(
            (
                "claude-code.asc",        # release-key download
                '!= "${release_fingerprint}"',  # fingerprint-mismatch guard (real `!=` on the pin)
                "--import",               # gpg signing-key import
                "manifest.json",          # manifest download
                "manifest.json.sig",      # detached-signature download
                "--verify",               # gpg verification
                '"checksum"',             # manifest checksum extraction
                "sha256sum -c -",         # binary checksum verification
                "install -D -m 0755",     # install onto PATH
            ),
            id="claude",
        ),
        pytest.param(
            (
                "codex-${target}.tar.gz",  # archive download
                "sha256sum -c -",          # verify
                "tar -xzf",                # extract
            ),
            id="codex",
        ),
        pytest.param(
            (
                "node-v${NODE_VERSION}-linux-${node_arch}.tar.gz",  # archive download
                "sha256sum -c -",                                   # verify
                "tar -xzf",                                         # extract
            ),
            id="node",
        ),
        pytest.param(
            (
                "pi-coding-agent-${PI_VERSION}.tgz",  # pi package tarball download
                "sha256sum -c -",                     # verify
                "npm install -g",                     # install from verified tarball
            ),
            id="pi",
        ),
    ],
)
def test_base_dockerfile_verifies_downloads_before_use(marker_chain: tuple[str, ...]) -> None:
    """M3/M4/S2: each download-verify-extract chain is strictly ordered; verify precedes use."""
    text = BASE_DOCKERFILE.read_text(encoding="utf-8")
    pos = text.find(marker_chain[0])
    assert pos != -1, f"chain start {marker_chain[0]!r} not found"
    for marker in marker_chain[1:]:
        nxt = text.find(marker, pos + 1)
        assert nxt > pos, f"{marker!r} must appear after {text[pos : pos + 40]!r}"
        pos = nxt


@pytest.mark.parametrize(
    "forbidden_literal",
    [
        pytest.param("https://claude.ai/install.sh", id="remote-installer"),
        pytest.param("| bash", id="bash-pipe"),
        pytest.param("| tar", id="tar-pipe"),
    ],
)
def test_base_dockerfile_does_not_pipe_downloads_into_shell_or_tar(forbidden_literal: str) -> None:
    """M3/M4: no remote execution pipeline remains in the build contract."""
    text = BASE_DOCKERFILE.read_text(encoding="utf-8")
    assert forbidden_literal not in text, f"Dockerfile must not contain {forbidden_literal!r}"
