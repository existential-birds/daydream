"""Container image contracts: immutable base inputs and the green-baseline gate.

Four kinds of contract live here. The static ``base.Dockerfile`` and
``repo.Dockerfile`` checks run with no Docker at all — they assert the build pins
every remote input to an immutable version and verifies it for integrity before
use. A **fast** tier — no Docker required — rejects ``--red`` invocations that
cannot select the fixture repo, ensuring the guard fires before any build
side-effect. A manifest/README tier asserts the locked-dependency policy: the
itsdangerous manifest entry installs strictly from its committed uv.lock and the
README documents the four mandatory setup rules. The three ``slow`` tests execute
real Docker builds: the red baseline path plants a failing assertion and the
build must die (enforcement IS the build failing), the green baseline path builds
and bakes the checkout, and the reference-image build proves the itsdangerous
image builds only when the locked setup plus the green baseline succeed.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from conftest import PROJECT_ROOT, docker_daemon_is_available

from daydream_review_v1.fixture import FIXTURE_SLUG, build_fixture_repo
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
    # re-run in fix_tests_pass exercises the agent's edits rather than stale code.
    probe = subprocess.run(
        ["docker", "run", "--rm", REFERENCE_TAG, "sh", "-c",
         "test -x /opt/repo-venv/bin/python && /opt/repo-venv/bin/python -c '"
         "import pytest, freezegun; "
         "import itsdangerous; "
         "assert \"/work/repo\" in itsdangerous.__file__, itsdangerous.__file__'"],
        capture_output=True, text=True, check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


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
    # module-level marker that would also skip the static tests.  Check for
    # actual skipif marker content, not just pytestmark attribute existence.
    for name in (
        "test_green_baseline_gate_fails_the_build_on_a_red_suite",
        "test_green_baseline_builds_and_bakes_the_checkout",
    ):
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

    _build(base)
    _build(base, "--red")
    _build(base, "--corpus", str(REFERENCE_CORPUS), slug=REFERENCE_SLUG)

    assert captured[0] == [
        "uv", "run", "python", "images/build_images.py",
        "--only", FIXTURE_SLUG, "--no-base", base,
    ]
    assert captured[1] == [
        "uv", "run", "python", "images/build_images.py",
        "--only", FIXTURE_SLUG, "--no-base", base, "--red",
    ]
    assert captured[2] == [
        "uv", "run", "python", "images/build_images.py",
        "--only", REFERENCE_SLUG, "--no-base", base,
        "--corpus", str(REFERENCE_CORPUS),
    ]
    assert "_build_reference" not in vars(sys.modules[__name__])


def test_corpus_and_slug_literals_single_source() -> None:
    """F2: corpus path and reference slug each resolve through one named constant."""
    src = Path(__file__).read_text(encoding="utf-8")
    corpus = "corpus-" + "reference"          # built to avoid self-matching
    slug = "pallets/" + "itsdangerous"        # built to avoid self-matching
    assert src.count(corpus) == 1, "corpus path must appear only in REFERENCE_CORPUS"
    assert src.count(slug) == 3, "slug literal must be REFERENCE_SLUG + the 2 manifest markers"
    assert "REFERENCE_CORPUS" in src and "REFERENCE_SLUG" in src


def test_module_docstring_describes_actual_contracts() -> None:
    """F4: the docstring names the four contract kinds and three slow tests."""
    doc = sys.modules[__name__].__doc__ or ""
    for kind in ("Dockerfile", "--red", "manifest", "slow", "reference"):
        assert kind in doc, f"docstring must name contract kind/area {kind!r}"
    assert "three" in doc.lower(), "docstring must count the three slow tests"
    assert "two" not in doc.lower(), "stale 'two slow tests' count must be gone"
