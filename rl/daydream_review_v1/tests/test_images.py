"""Container image contracts: immutable base inputs and the green-baseline gate.

Four kinds of contract live here. The static ``base.Dockerfile`` and
``repo.Dockerfile`` checks run with no Docker at all — they assert the build pins
every remote input to an immutable version and verifies it for integrity before
use. A **fast** tier — no Docker required — rejects ``--red`` invocations that
cannot select the fixture repo, ensuring the guard fires before any build
side-effect. A manifest/README tier asserts the locked-dependency policy: the
itsdangerous manifest entry installs strictly from its committed uv.lock and the
README documents the four mandatory setup rules. The five ``slow`` tests execute
real Docker builds: the red baseline path plants a failing assertion and the
build must die (enforcement IS the build failing), the green baseline path builds
and bakes the checkout, and the reference-image build proves the itsdangerous
image builds only when the locked setup plus the green baseline succeed. The
warm-host base build re-executes the gpg/checksum hardening on an already-warm
host via ``docker build --no-cache`` instead of serving the present base's
cached layers.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest
from conftest import PROJECT_ROOT, assert_docstring_guards, docker_daemon_is_available

from daydream_review_v1.fixture import FIXTURE_PR2_HEAD_SHA, FIXTURE_SLUG, build_fixture_repo
from images import build_images

DOCKER_REQUIRED = pytest.mark.skipif(
    not docker_daemon_is_available(),
    reason="docker is not installed or the daemon is unavailable",
)

FIXTURE_IMAGE = "daydream-rl/fixture"

BASE_DOCKERFILE = PROJECT_ROOT / "images" / "base.Dockerfile"


def _build(base_image: str, *args: str, slug: str = FIXTURE_SLUG) -> subprocess.CompletedProcess[str]:
    """Build the repo image for ``slug`` (the fixture by default); ``base_image`` owns the base."""
    return subprocess.run(
        ["uv", "run", "python", "images/build_images.py", "--only", slug, "--no-base", base_image, *args],
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


MANIFEST = PROJECT_ROOT / "images" / "manifest.toml"
README = PROJECT_ROOT / "README.md"
REFERENCE_IMAGE = "daydream-rl/itsdangerous"
REFERENCE_TAG = f"{REFERENCE_IMAGE}:4bb03cd68192"
REFERENCE_CORPUS = PROJECT_ROOT / "tests" / "fixtures" / "corpus-reference"
REFERENCE_SLUG = "pallets/itsdangerous"


@pytest.mark.parametrize(
    "argv,expected_stderr",
    [
        pytest.param(
            ["--red", "--base-only"],
            "--red cannot be combined with --base-only",
            id="base-only",
        ),
        pytest.param(
            [
                "--red",
                "--corpus",
                str(REFERENCE_CORPUS),
                "--only",
                REFERENCE_SLUG,
            ],
            "--red requires at least one selected fixture PR backed by fixture://daydream-rl-fixture",
            id="non-fixture-only",
        ),
    ],
)
def test_red_rejects_invocations_without_fixture(
    argv: list[str], expected_stderr: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--red must fail fast (status 2) when no fixture PR is selected, before any build."""
    monkeypatch.setattr(build_images, "_build_base", lambda: (0, None))
    monkeypatch.setattr(build_images, "_stream", lambda *args, **kwargs: None)

    status = build_images.main(argv)
    captured = capsys.readouterr()

    assert status == 2
    assert captured.out == ""
    assert captured.err == f"{expected_stderr}\n"


def test_no_base_requires_immutable_base_identity() -> None:
    """A --no-base snapshot build needs an explicit immutable base identity (status 2)."""
    # Missing value: argparse refuses the invocation before any build.
    with pytest.raises(SystemExit) as exc:
        build_images.main(["--no-base"])
    assert exc.value.code == 2

    # The mutable latest alias is not an accepted immutable identity.
    status = build_images.main(["--no-base", build_images.BASE_LATEST])
    assert status == 2


def test_no_base_requires_the_base_image_to_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A well-formed but locally absent --no-base identity fails fast (status 2).

    It must never reach a per-repo build: the failure is a configuration error
    (exit 2), not a per-PR build failure (exit 1).
    """
    built: list[str] = []
    monkeypatch.setattr(build_images, "_base_image_present", lambda _: False)
    monkeypatch.setattr(build_images, "build_repo_image", lambda *a, **k: built.append("built"))

    status = build_images.main(["--only", FIXTURE_SLUG, "--no-base", "daydream-rl/base:v1.2.3"])
    assert status == 2
    assert not built, "an absent base image must be refused before any repo build"


def test_immutable_base_image_accepts_only_versioned_identities() -> None:
    """The validator accepts versioned tags/digests; aliases and junk are refused."""
    digest_hex = "a" * 64
    accepted = [
        "daydream-rl/base:v1.2.3",
        "daydream-rl/base:1.2.3",
        "daydream-rl/base:v0.1.2-3-g5ce4c0e-dirty",  # git describe output
        f"daydream-rl/base@sha256:{digest_hex}",
        "daydream-rl/base:r2d2",  # Docker-grammar tag containing a digit
        "daydream-rl/base:deadbeef",  # digit-free git describe --always fallback (untagged clone)
        "daydream-rl/base:deadbeef-dirty",  # ...with a dirty tree
    ]
    rejected = [
        build_images.BASE_LATEST,           # the mutable alias
        "daydream-rl/base:stable",          # unversioned aliases
        "daydream-rl/base:dev",
        "daydream-rl/base:nightly",
        "daydream-rl/base:main",
        "daydream-rl/base:-foo",            # docker grammar: leading dash
        "daydream-rl/base:foo/bar",         # docker grammar: slash
        "daydream-rl/base:has space",       # docker grammar: whitespace
        "daydream-rl/base:",                # empty tag
        "daydream-rl/base:v1.2.3\n",        # trailing newline is not part of the identity
        f"daydream-rl/base@sha256:{'x' * 64}",  # not hex
        f"daydream-rl/base@sha256:{digest_hex[:-1]}",  # 63 hex, not 64
        f"daydream-rl/base@sha256:{digest_hex}\n",  # trailing newline
    ]
    for value in accepted:
        assert build_images._immutable_base_image(value) == value, value
    for value in rejected:
        assert build_images._immutable_base_image(value) is None, value


def test_build_base_selects_the_versioned_tag_not_the_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_base returns the immutable versioned tag by shape, never by position."""
    versioned = "daydream-rl/base:v1.2.3-3-g5ce4c0e"

    # Versioned tag first, as build_base_image emits it today.
    monkeypatch.setattr(build_images, "build_base_image", lambda: [versioned, build_images.BASE_LATEST])
    assert build_images._build_base() == (0, versioned)

    # Alias first: the selection must not depend on the tag-list order.
    monkeypatch.setattr(build_images, "build_base_image", lambda: [build_images.BASE_LATEST, versioned])
    assert build_images._build_base() == (0, versioned)

    # No immutable identity in the tags at all: a failure, never a bare alias.
    monkeypatch.setattr(build_images, "build_base_image", lambda: [build_images.BASE_LATEST])
    assert build_images._build_base() == (1, None)


def test_main_uses_immutable_base_for_repository_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh and --no-base paths both pass exactly one immutable base to every repo build."""
    versioned = "daydream-rl/base:v1.2.3"
    digest = "daydream-rl/base@sha256:" + "a" * 64

    received: list[str] = []

    def _record(entry, *, head_sha, base_sha, base_image, red):
        received.append(base_image)
        return f"{entry.image}:{head_sha[:12]}"

    monkeypatch.setattr(build_images, "_build_base", lambda: (0, versioned))
    monkeypatch.setattr(build_images, "build_repo_image", _record)
    monkeypatch.setattr(build_images, "_base_image_present", lambda _: True)

    # Fresh path (no --no-base): every build uses the versioned tag from the base build.
    assert build_images.main(["--only", FIXTURE_SLUG]) == 0
    assert received, "fresh path built no repo image"
    assert received == [versioned] * len(received)

    # --no-base path: every build uses the explicit immutable identity.
    received.clear()
    assert build_images.main(["--only", FIXTURE_SLUG, "--no-base", digest]) == 0
    assert received == [digest] * len(received)

    # The mutable alias is never selected for a snapshot build.
    assert build_images.BASE_LATEST not in received


def test_repo_dockerfile_requires_an_immutable_base_image_arg() -> None:
    """repo.Dockerfile must declare ARG BASE_IMAGE with no mutable default."""
    text = (PROJECT_ROOT / "images" / "repo.Dockerfile").read_text(encoding="utf-8")
    arg_line = next(line for line in text.splitlines() if line.startswith("ARG BASE_IMAGE"))
    assert "=" not in arg_line, f"ARG BASE_IMAGE must have no default: {arg_line!r}"
    assert "latest" not in arg_line
    # The base is still consumed via FROM, declared after the ARG.
    assert text.find("FROM ${BASE_IMAGE}") > text.find("ARG BASE_IMAGE")
    assert "daydream-rl/base:latest" not in text
    # The comment spelling the accepted identity grammar must stay in sync with
    # the single source of truth in build_images.py, so a grammar change cannot
    # leave this Dockerfile silently out of date.
    assert build_images.IMMUTABLE_BASE_FORMAT in text


@pytest.mark.slow
@DOCKER_REQUIRED
def test_green_baseline_gate_fails_the_build_on_a_red_suite(base_image: str) -> None:
    """A repository whose suite is red at the head commit must produce NO image."""
    result = _build(base_image, "--red")
    combined = result.stdout + result.stderr

    assert result.returncode != 0, "a red baseline built successfully — the gate is not enforcing"
    # The failure has to come from the test layer, not from an earlier step: a
    # build that died at `git checkout` would fail for the wrong reason and prove
    # nothing about the gate.
    assert "${TEST_COMMAND}" in combined, combined[-3000:]
    assert "FAILED (failures=1)" in combined, combined[-3000:]

    # `--red` rewrites the HEAD commit (PR #2) red; the earlier PR (#1) in the
    # corpus stays green and legitimately produces an image. So the invariant is
    # that the red head's own snapshot tag must not exist — not that no new tag
    # appears at all (PR #1's image is expected). Compute the red head SHA the
    # same way materialize_mirror does (build_fixture_repo(red=True)) and assert
    # its image is absent.
    with tempfile.TemporaryDirectory(prefix="daydream-rl-redhead-") as tmp:
        red_head = build_fixture_repo(Path(tmp) / "repo", red=True).pr2_head_sha
    red_tag = f"{FIXTURE_IMAGE}:{red_head[:12]}"
    assert red_tag not in _tags(), (
        f"a tag was published for the red baseline: {red_tag} (have {_tags()})"
    )


@pytest.mark.slow
@DOCKER_REQUIRED
def test_green_baseline_builds_and_bakes_the_checkout(base_image: str) -> None:
    """The happy path, end to end: image builds, suite green, origin is local."""
    result = _build(base_image)
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    assert f"skipping base build; reusing {base_image}" in result.stdout

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


def test_reference_manifest_entry_consumes_its_committed_lock() -> None:
    """The itsdangerous entry installs strictly from its committed uv.lock."""
    text = MANIFEST.read_text(encoding="utf-8")
    entry = text[text.index('[repos."pallets/itsdangerous"]') :]
    assert "uv sync --locked --no-default-groups --group tests" in entry
    assert "UV_PROJECT_ENVIRONMENT=/opt/repo-venv" in entry
    assert "UV_PYTHON_DOWNLOADS=never" in entry
    assert 'test_command = "/opt/repo-venv/bin/python -m pytest -q"' in entry
    assert "pip install" not in entry


def test_fixture_manifest_entry_stays_dependency_free() -> None:
    """The deterministic fixture entry is untouched: no setup, its unittest command."""
    text = MANIFEST.read_text(encoding="utf-8")
    head = text[: text.index('[repos."pallets/itsdangerous"]')]
    assert 'setup_cmds = []' in head
    assert 'test_command = "python -m unittest discover -q"' in head


def test_readme_documents_the_locked_dependency_policy() -> None:
    """The 'Adding a repository' section states the four mandatory setup rules."""
    text = README.read_text(encoding="utf-8")
    section = text[text.index("## Adding a repository") :]
    for marker in (
        "committed at the head SHA",            # rule 1: lockfile committed at baked head
        "rejects lock drift",                   # rule 2: no silent drift
        "uv sync --locked",                     # rule 2: the concrete mode
        "outside `/work/repo`",                 # rule 3: external environment
        "UV_PROJECT_ENVIRONMENT=/opt/repo-venv",  # rule 3: the concrete env
        "/opt/repo-venv/bin/python -m pytest -q",  # rule 4: locked test command
        "image-build failure",                  # no-fallback statement
        "fall back to unconstrained pip",       # no-fallback statement
    ):
        assert marker in section, f"README '## Adding a repository' must state {marker!r}"


@pytest.mark.slow
@DOCKER_REQUIRED
def test_reference_image_builds_with_locked_dependencies(base_image: str) -> None:
    """The reference image builds only when the locked setup + green baseline succeed."""
    result = _build(base_image, "--corpus", str(REFERENCE_CORPUS), slug=REFERENCE_SLUG)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined[-3000:]
    assert f"built {REFERENCE_TAG}" in result.stdout, combined[-3000:]
    # Discriminating probe: only the locked setup creates /opt/repo-venv with the
    # test deps; a pip-based setup would leave it absent, so this fails a regression.
    # It also pins the editable-install invariant: the package under test must
    # resolve from the baked /work/repo checkout, not a venv copy, so the rollout
    # re-run in suite_non_regression exercises the agent's edits rather than stale code.
    probe = subprocess.run(
        ["docker", "run", "--rm", REFERENCE_TAG, "sh", "-c",
         "test -x /opt/repo-venv/bin/python && /opt/repo-venv/bin/python -c '"
         "import pytest, freezegun; "
         "import itsdangerous; "
         "assert \"/work/repo\" in itsdangerous.__file__, itsdangerous.__file__'"],
        capture_output=True, text=True, check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


@pytest.mark.slow
@DOCKER_REQUIRED
def test_base_layer_hardening_executes_on_warm_host() -> None:
    """Warm-host live coverage of the base-image hardening.

    Builds a fresh base image with ``--no-cache`` and verifies the resulting
    image actually contains the hardening layers (gpg-verify and checksum
    verification) via ``docker image history``, and that the run-as-agent
    privilege-drop seam really drops root in the built image. The session
    ``base_image``
    fixture short-circuits when the image exists (conftest.py:40-55) and every
    repo build passes ``--no-base``, so on a warm host the hardening is
    otherwise never re-run; this test forces the build to run and then probes
    the produced image. The asserted observables (image-history markers and a
    clean build return code) confirm the hardening layers are present and the
    build succeeded, but they do not by themselves prove the layers were
    re-executed rather than served from cache -- a log-text ``CACHED`` heuristic
    is deliberately avoided because BuildKit logs ``CACHED`` for FROM steps
    whose base images are already in the local store.
    """
    wheel = build_images.build_wheel(build_images.DIST_DIR)
    tag = f"{build_images.BASE_REPOSITORY}:warmhost-{uuid.uuid4().hex[:8]}"
    try:
        result = subprocess.run(
            build_images._base_build_cmd(wheel, [tag], no_cache=True),
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, combined[-4000:]
        # Functional probe of the freshly built image, not the log: ``docker
        # image history`` records the RUN steps docker actually executed, so the
        # gpg-verify and checksum layers being present is the hardening having
        # re-run (a failed gpg verify or checksum mismatch would have died the
        # build).
        probe = subprocess.run(
            ["docker", "image", "history", tag, "--no-trunc", "--format", "{{.CreatedBy}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert probe.returncode == 0, f"throwaway image {tag} not produced"
        assert "gpg --batch --verify" in probe.stdout, (
            "built image lacks the gpg-verify hardening layer"
        )
        assert "sha256sum -c" in probe.stdout, (
            "built image lacks the checksum-verification hardening layer"
        )
        # The privilege-drop seam must actually work in the built image, not
        # just be declared: the image's default user stays root, and the
        # root-owned run-as-agent wrapper drops to a non-root uid (the `agent`
        # user). A missing or non-executable wrapper is exactly the
        # rollout-time failure this probe catches at build time.
        default_uid = subprocess.run(
            ["docker", "run", "--rm", tag, "id", "-u"],
            capture_output=True, text=True, check=False,
        )
        dropped_uid = subprocess.run(
            ["docker", "run", "--rm", tag, "run-as-agent", "id", "-u"],
            capture_output=True, text=True, check=False,
        )
        assert default_uid.returncode == 0, default_uid.stdout + default_uid.stderr
        assert dropped_uid.returncode == 0, dropped_uid.stdout + dropped_uid.stderr
        assert default_uid.stdout.strip() == "0", "image default user must stay root"
        assert dropped_uid.stdout.strip() != "0", "run-as-agent must drop off root"
    finally:
        subprocess.run(["docker", "rmi", tag], capture_output=True, text=True, check=False)


@pytest.mark.slow
@DOCKER_REQUIRED
def test_real_docker_deep_flow_fix_pipeline_write_as_agent(base_image: str) -> None:
    """A real (non-fake) docker invocation completes a fix-pipeline write as the
    agent uid after the harness handoff, and the write reaches the in-container
    origin mirror.

    repo.Dockerfile chowns /work/repo at build time (idempotent defense-in-depth
    against the launch-time handoff), so the baked checkout is already
    agent-owned. The in-container origin mirror /srv/mirror.git is baked into
    the image at build time (COPY mirror.git /srv/mirror.git), but no build
    layer chowns it; the harness re-chowns the checkout plus the mirror at
    launch (harness.py:148-162), covering the mirror. This drives that exact
    handoff then the deep flow's
    terminal write sequence (.daydream/ mkdir, git apply a fix patch,
    git add/commit, git push HEAD:main) as the agent uid inside the real image,
    and asserts the push reached /srv/mirror.git. When a docker daemon is
    reachable but the write fails, this FAILS (never skips).
    """
    result = _build(base_image)
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    tag = f"{FIXTURE_IMAGE}:{FIXTURE_PR2_HEAD_SHA[:12]}"
    assert tag in _tags(), f"{tag} not built; have {_tags()}"

    # A real one-line fix patch that applies cleanly to the baked calc.py tree
    # (deterministic fixture content; the hunk matches the baked pr2 tree).
    fix_patch = (
        "diff --git a/calc.py b/calc.py\n"
        "index 319252d..f5de56a 100644\n"
        "--- a/calc.py\n"
        "+++ b/calc.py\n"
        "@@ -3,7 +3,7 @@\n"
        " \n"
        " def add(a: int, b: int) -> int:\n"
        "     \"\"\"Return the sum of *a* and *b*.\"\"\"\n"
        "-    return a + b\n"
        "+    return a + b  # fixed\n"
        " \n"
        " \n"
        " def divide(a: int, b: int) -> float:\n"
    )

    script = (
        # The harness handoff (harness.py:148-162): hand checkout + mirror to agent.
        "chown -R agent:agent /work/repo /srv/mirror.git && "
        # The deep flow's fix-pipeline write, run as the agent uid. The fix patch
        # arrives on stdin (docker run -i) and is applied via `git apply -`, so
        # no base64/coreutils dependency is introduced into the image contract.
        "run-as-agent sh -c '"
        "cd /work/repo && "
        "mkdir -p .daydream && "
        "cat > .daydream/recommended.patch && "
        "git apply .daydream/recommended.patch && "
        "git add -A && "
        # The deep flow injects a fallback identity when none is configured
        # (fresh-CI env, git_ops.py:1902-1912); the baked image carries no
        # user.name/user.email, so the commit must do the same or it dies
        # "Author identity unknown".
        "git -c user.email=daydream@localhost -c user.name=daydream commit -q -m fix && "
        "git push -q origin HEAD:main"
        "' && "
        # Same container: prove the write reached the in-container mirror.
        "git --git-dir=/srv/mirror.git show main:calc.py"
    )
    probe = subprocess.run(
        ["docker", "run", "--rm", "-i", tag, "sh", "-c", script],
        input=fix_patch, capture_output=True, text=True, check=False,
    )
    # The write reached the in-container origin mirror: the same container that
    # pushed now reads main:calc.py back out of /srv/mirror.git and it carries
    # the agent's fix (each docker run --rm starts from the baked image, so the
    # mirror state must be verified inside the one container that wrote it).
    assert probe.returncode == 0, (
        "real docker fix-pipeline write failed as agent uid: "
        f"{probe.stdout}{probe.stderr}"
    )
    assert "# fixed" in probe.stdout, (
        "the agent's fix did not reach the in-container origin mirror: "
        f"{probe.stdout}{probe.stderr}"
    )


def test_real_docker_write_docstring_describes_build_chown_and_rechown() -> None:
    """The real-docker-write docstring must describe the CURRENT design: the image
    chowns the checkout at build time AND the harness re-chowns the checkout plus
    the mirror at launch. The stale 'no chown (this issue forbids re-adding one)'
    and 'runtime-created mirror' claims are gone."""
    assert_docstring_guards(
        test_real_docker_deep_flow_fix_pipeline_write_as_agent,
        gone=("no chown", "forbids re-adding one", "runtime-created"),
        present=("chowns /work/repo at build time", "/srv/mirror.git", "baked"),
    )


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


def _slow_test_names() -> list[str]:
    """Names of this module's @pytest.mark.slow tests, derived from the marker
    itself so a newly added slow test is covered without editing a name list."""
    return sorted(
        name
        for name, obj in vars(sys.modules[__name__]).items()
        if any(getattr(m, "name", None) == "slow" for m in getattr(obj, "pytestmark", ()))
    )


def test_docker_skip_is_per_test_not_module_wide() -> None:
    """M8 regression: the Docker skip is per-test, not module-wide, so the static
    build-contract tests (added in the next task) collect in a Docker-less CI."""
    module = sys.modules[__name__]
    assert "pytestmark" not in vars(module), "module-wide Docker skip would gate the static tests"
    assert "DOCKER_REQUIRED" in vars(module), "per-test DOCKER_REQUIRED marker missing"
    # Every slow test must carry the skip; none may rely on a module-level
    # marker that would also skip the static tests.  Enumerate by the slow
    # marker (not a hardcoded name list) so the reference-image test and any
    # future slow test are covered.  Check for actual skipif marker content,
    # not just pytestmark attribute existence.
    slow_tests = _slow_test_names()
    assert slow_tests, "no @pytest.mark.slow tests found"
    for name in slow_tests:
        test_func = getattr(module, name)
        marks = getattr(test_func, "pytestmark", [])
        assert marks, f"{name} has no pytestmark (would not skip)"
        assert any(getattr(m, "name", None) == "skipif" for m in marks), (
            f"{name} missing a skipif marker"
        )


@pytest.mark.parametrize(
    "required_literal",
    [
        pytest.param(
            "FROM python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36",
            id="python-base-index-digest",
        ),
        pytest.param(
            (
                "FROM ghcr.io/astral-sh/uv:0.11.29@"
                "sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc AS uv"
            ),
            id="uv-image-digest-pin",
        ),
        pytest.param("COPY --from=uv /uv /uvx /bin/", id="uv-copy-from-image"),
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


def test_build_emits_canonical_argv_for_all_call_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F1: _build is the single helper; slug defaults to the fixture, and the
    reference build passes the reference slug + corpus through the same argv."""
    captured: list[list[str]] = []

    def _capture(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _capture)
    base = "daydream-rl/base:v1.2.3"

    # One shared 5-element prefix; each case is (extra _build args, slug,
    # expected argv tail) so the delta between call sites — slug, --red,
    # --corpus — is the visible invariant, not three verbatim argv copies.
    prefix = ["uv", "run", "python", "images/build_images.py", "--only"]
    cases: list[tuple[tuple[str, ...], str | None, list[str]]] = [
        ((), None, [FIXTURE_SLUG, "--no-base", base]),
        (("--red",), None, [FIXTURE_SLUG, "--no-base", base, "--red"]),
        (
            ("--corpus", str(REFERENCE_CORPUS)),
            REFERENCE_SLUG,
            [REFERENCE_SLUG, "--no-base", base, "--corpus", str(REFERENCE_CORPUS)],
        ),
    ]
    for extra_args, slug, expected_tail in cases:
        kwargs: dict[str, str] = {} if slug is None else {"slug": slug}
        _build(base, *extra_args, **kwargs)
        assert captured[-1] == prefix + expected_tail, captured[-1]

    assert "_build_reference" not in vars(sys.modules[__name__])


def test_corpus_and_slug_literals_single_source() -> None:
    """F2: corpus path and reference slug each resolve through one named constant."""
    src = Path(__file__).read_text(encoding="utf-8")
    corpus = "corpus-" + "reference"          # built to avoid self-matching
    slug = "pallets/" + "itsdangerous"        # built to avoid self-matching
    marker = '[repos."' + slug + '"]'
    # Scan only the module-top constants block and the manifest-entry tests,
    # mirroring the F3 guard's window-narrowing: a docstring or comment
    # elsewhere quoting the path cannot trip the guard.
    constants_end = src.index("REFERENCE_SLUG = ") + len('REFERENCE_SLUG = "' + slug + '"')
    constants = src[src.index("MANIFEST = ") : constants_end]
    assert constants.count(corpus) == 1, "corpus path must appear only in REFERENCE_CORPUS"
    # The slug literal is REFERENCE_SLUG once plus one occurrence per
    # manifest-entry marker; counting relative to the markers keeps the guard
    # valid when a third manifest test is added or the marker is extracted
    # into a shared constant.
    assert constants.count(slug) == 1, "slug literal must appear in the constants block only in REFERENCE_SLUG"
    manifest = src[
        src.index(marker) : src.index("def test_readme_documents_the_locked_dependency_policy")
    ]
    assert manifest.count(slug) == manifest.count(marker), (
        "slug literal must appear once per manifest marker"
    )
    assert "REFERENCE_CORPUS" in src and "REFERENCE_SLUG" in src


def test_module_docstring_describes_actual_contracts() -> None:
    """F4: the docstring names the four contract kinds and counts the slow tests
    that actually carry the marker, so the count cannot drift from the file."""
    doc = sys.modules[__name__].__doc__ or ""
    for kind in ("Dockerfile", "--red", "manifest", "slow", "reference"):
        assert kind in doc, f"docstring must name contract kind/area {kind!r}"
    slow_tests = _slow_test_names()
    number = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(
        len(slow_tests), str(len(slow_tests))
    )
    assert f"the {number} ``slow`` tests" in doc.lower(), (
        f"docstring must count the {len(slow_tests)} slow tests: {slow_tests}"
    )
    assert "two slow" not in doc.lower(), "stale 'two slow tests' count must be gone"


def test_reference_probe_quotes_the_work_repo_path() -> None:
    """F3 guard (verify-only): the sh -c probe's /work/repo is escaped double-quoted,
    so the single-quoted sh -c region is never terminated by a bare path."""
    src = Path(__file__).read_text(encoding="utf-8")
    quoted = '\\"' + "/work/repo" + '\\"'   # built to avoid self-matching
    bare = "'" + "/work/repo" + "'"         # built to avoid self-matching
    # Scan only the reference probe's own python -c payload, anchored on its
    # distinctive assert target rather than on source formatting: reordered
    # kwargs, split payload literals, or a reformatted sh -c list cannot move
    # the window, and prose elsewhere quoting the path cannot trip it either.
    probe_end = src.index("itsdangerous.__file__")
    probe_start = src.rindex("python -c '", 0, probe_end)
    probe_src = src[probe_start : probe_end]
    assert quoted in probe_src, "python -c body must receive a quoted path literal"
    assert bare not in probe_src, "a single-quoted path would break the sh -c region"


def test_run_as_agent_wrapper_drops_privilege() -> None:
    """The wrapper setprivs down to the agent identity (image contract). Root
    ownership is established only inside the built image (base.Dockerfile chowns
    run-as-agent root:root), so no st_uid check is made on the checkout copy."""
    wrapper = PROJECT_ROOT / "images" / "run-as-agent"
    assert wrapper.is_file(), "run-as-agent wrapper missing"
    mode = wrapper.stat().st_mode & 0o777
    assert mode & 0o400 and mode & 0o100, "wrapper must be owner-executable (0755)"
    text = wrapper.read_text(encoding="utf-8")
    assert "setpriv" in text, "wrapper must drop privileges via setpriv"
    assert "agent" in text, "wrapper must target the non-root agent user"


def test_base_image_has_distinct_agent_identity() -> None:
    """base.Dockerfile declares non-root agent and verifier users, an
    agent-owned archive, and the explicit setpriv provider (util-linux)."""
    dockerfile = (PROJECT_ROOT / "images" / "base.Dockerfile").read_text(encoding="utf-8")
    assert "util-linux" in dockerfile  # setpriv provider pinned explicitly, not implicit
    assert "useradd" in dockerfile or "adduser" in dockerfile
    assert "agent" in dockerfile
    assert "verifier" in dockerfile  # read-only verifier identity provisioned
    assert "chown" in dockerfile and "agent:agent" in dockerfile  # /rollout (incl. archive) is agent-owned


def test_repo_image_chowns_checkout_to_agent() -> None:
    """repo.Dockerfile must hand the cloned /work/repo tree to the agent uid.

    The image clones /work/repo as root (no USER directive), so without a chown
    layer an agent-uid process hits EACCES on its first write. The harness
    re-chowns at launch (idempotent against this), but the image should be
    self-sufficient defense-in-depth — root-owned by default is the failure the
    issue describes.
    """
    dockerfile = (PROJECT_ROOT / "images" / "repo.Dockerfile").read_text(encoding="utf-8")
    # Pin the anchored layer exactly (recursive, full /work/repo target, agent
    # uid), and require it to sit after the root-run setup.sh/TEST_COMMAND
    # layers so their outputs (e.g. .venv from uv sync) are agent-owned too.
    # Bare substring checks let a dropped -R, a subpath target, a retargeted
    # uid, or a chown moved before the setup layers pass green.
    chown_layer = "RUN chown -R agent:agent /work/repo"
    assert chown_layer in dockerfile
    assert dockerfile.index(chown_layer) > dockerfile.index("RUN cd /work/repo && sh /tmp/setup.sh")
    assert dockerfile.index(chown_layer) > dockerfile.index("RUN cd /work/repo && ${TEST_COMMAND}")


def test_readme_documents_single_reward_axis_and_metric() -> None:
    from conftest import PROJECT_ROOT

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "two axes" not in readme.lower()
    assert "suite_non_regression" in readme
    assert "intrinsic_composite" in readme


def test_configs_and_pyproject_reflect_the_new_contract() -> None:
    from conftest import PROJECT_ROOT

    docker = (PROJECT_ROOT / "configs" / "eval-docker.toml").read_text(encoding="utf-8")
    stub = (PROJECT_ROOT / "configs" / "eval-stub.toml").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "verdict" not in docker.lower()  # stale "it's a verdict" wording
    assert "suite_non_regression" in docker or "non-regression" in docker
    assert "reward axes" not in stub.lower()  # stale "both reward axes" wording
    assert "0.2.0" in pyproject
