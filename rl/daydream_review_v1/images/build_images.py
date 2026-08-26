"""Build the rollout container images for the ``daydream-review-v1`` environment.

Two rules govern everything in this file.

**One image == one PR snapshot.** Each image is tagged with the 12-character head
SHA of the pull request it bakes (``daydream-rl/fixture:9b92381663058``), which is
exactly the tag :meth:`DaydreamReviewTaskset.load` stamps onto the task. Nothing
clones at rollout time and no rollout carries credentials: the repository, at that
one commit, is already inside the image with ``origin`` pointing at an in-container
mirror.

**The baseline must be green.** The last layer of ``repo.Dockerfile`` runs the
repository's own test suite at the head commit, so a red suite fails the build and
produces no image. That is not a nicety — the ``suite_non_regression`` metric
records a rollout for a suite that passes after its fix, so a baseline that was
already red makes that signal pure noise. This script never catches, retries or
downgrades that failure; a failed build exits non-zero and the image simply does
not exist.

Usage::

    uv run python images/build_images.py
    uv run python images/build_images.py --only existential-birds/daydream-rl-fixture
    uv run python images/build_images.py --no-base daydream-rl/base:v1.2.3 --corpus ../corpora/train

``--red`` is the gate's own test: it plants a failing assertion in the fixture
repository's head commit and expects the build to die at the final layer.

Exit codes:
    0 — all requested images built successfully.
    1 — one or more images failed to build (Docker, test suite, or setup error).
    2 — invalid arguments or configuration (refused to run).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

from daydream_review_v1.corpus import harvested_corpus
from daydream_review_v1.fixture import (
    FIXTURE_BASE_SHA,
    FIXTURE_PR1_HEAD_SHA,
    FIXTURE_PR2_HEAD_SHA,
    FIXTURE_SLUG,
    build_fixture_repo,
)

# Both are private to the environment package, and both are imported rather than
# reimplemented on purpose: an image keyed differently from the way the taskset
# keys it is an image no task can ever find.
from daydream_review_v1.taskset import _ManifestEntry, _repo_slug, load_manifest

IMAGES_DIR = Path(__file__).resolve().parent
PROJECT_DIR = IMAGES_DIR.parent
#: The daydream repository root — three levels up from ``images/``.
REPO_ROOT = IMAGES_DIR.parents[2]

DEFAULT_MANIFEST = IMAGES_DIR / "manifest.toml"
DEFAULT_CORPUS = PROJECT_DIR / "tests" / "fixtures" / "corpus-mini"
DIST_DIR = IMAGES_DIR / "dist"

BASE_REPOSITORY = "daydream-rl/base"
BASE_LATEST = f"{BASE_REPOSITORY}:latest"

#: The accepted immutable-base identity grammar, spelled exactly once. Every
#: user-facing prose site — argparse help, the exit-2 message, the validator
#: docstring, and (via a test-pinned comment) ``repo.Dockerfile`` — derives
#: from this literal, and the patterns below implement exactly it. Changing the
#: identity shape means changing this one literal (and the patterns).
IMMUTABLE_BASE_FORMAT = "daydream-rl/base:<tag> or daydream-rl/base@sha256:<64 hex>"

#: A versioned base tag, e.g. ``daydream-rl/base:v0.1.2-3-g5ce4c0e`` (git describe).
#: The tag portion follows Docker's reference grammar — ASCII alphanumerics plus
#: ``_ . -``, never starting with ``.`` or ``-``, at most 128 characters. The
#: separate digit check below is what makes a tag *versioned* rather than an
#: unversioned alias such as ``stable`` or ``dev``.
_BASE_TAG_RE = re.compile(r"^daydream-rl/base:[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}\Z")
#: Versioned tags must contain a digit — ``git describe`` output from a real
#: tag always does, and a digit is what excludes unversioned aliases like
#: ``stable``. On a clone with no applicable tag (untagged or shallow),
#: ``--always`` falls back to a bare hex SHA that can be all ``a-f`` and so
#: digit-free; that is the same immutable identity, so a pure-hex ``--always``
#: fallback is accepted too, while the mutable aliases (``stable``/``dev``) are
#: never pure hex.
_BASE_TAG_CONTAINS_DIGIT = re.compile(r"[0-9]")
_BASE_TAG_ALWAYS_FALLBACK_RE = re.compile(r"[0-9a-f]{7,40}(-dirty)?\Z")
#: A canonical content digest, e.g. ``daydream-rl/base@sha256:<64 hex>``.
_BASE_DIGEST_RE = re.compile(r"^daydream-rl/base@sha256:[0-9a-f]{64}\Z")

#: Manifest ``clone_url`` sentinel meaning "materialize the deterministic fixture
#: repository" instead of cloning anything (``daydream_review_v1.fixture``).
FIXTURE_CLONE_URL = "fixture://daydream-rl-fixture"


def _capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    """Run *cmd* and return its stdout, raising on a non-zero exit."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _stream(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run *cmd* with its output inherited, raising on a non-zero exit.

    Docker build logs go straight to the terminal rather than into a buffer: when
    the green-baseline gate trips, the failing test output IS the message, and
    capturing it would bury it behind a ``CalledProcessError`` repr.
    """
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def build_wheel(dist_dir: Path) -> str:
    """Build the daydream wheel from the parent repo into *dist_dir*.

    The directory is wiped first so exactly one wheel can be present. Wheel names
    carry the ``project.version``, not the commit, so a stale wheel from an older
    checkout would sit there under the same filename and be silently baked in.

    Returns:
        The wheel's filename, which the base image takes as ``DAYDREAM_WHEEL``.
    """
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True)
    _stream(["uv", "build", "--wheel", "--out-dir", str(dist_dir), str(REPO_ROOT)])

    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in {dist_dir}, found {[w.name for w in wheels]}")
    return wheels[0].name


def base_tag() -> str:
    """Version tag for the base image: the repo root's ``git describe``.

    Includes ``-dirty``. A base image built from an uncommitted working tree is a
    base image nobody else can reproduce, and the tag is the only place that fact
    survives into the registry.
    """
    describe = _capture(["git", "describe", "--tags", "--always", "--dirty"], cwd=REPO_ROOT)
    return f"{BASE_REPOSITORY}:{describe}"


def _immutable_base_image(value: str) -> str | None:
    """Return *value* when it is an explicit immutable base identity, else ``None``.

    The accepted grammar is the single literal ``IMMUTABLE_BASE_FORMAT``: a
    versioned tag whose tag portion fits Docker's reference grammar and either
    contains a digit or is the bare hex ``git describe --always`` fallback
    emitted on an untagged/shallow clone (excluding unversioned aliases like
    ``stable``/``dev``, which are never pure hex), or a canonical SHA-256
    digest. The mutable ``latest`` alias is rejected explicitly, before the
    patterns are consulted, so a snapshot build never silently rides the alias
    as it moves.
    """
    if value == BASE_LATEST:
        return None
    if _BASE_DIGEST_RE.match(value):
        return value
    if _BASE_TAG_RE.match(value):
        tag = value.split(":", 1)[1]
        if _BASE_TAG_CONTAINS_DIGIT.search(tag) or _BASE_TAG_ALWAYS_FALLBACK_RE.match(tag):
            return value
    return None


def _base_build_cmd(wheel: Path | str, tags: list[str], *, no_cache: bool = False) -> list[str]:
    """Assemble the ``docker build`` argv for the base image.

    Centralising the argv (rather than inlining it per call site) keeps the
    throwaway warm-host build in ``test_images.py`` honest with the production
    build path. ``no_cache`` defeats the layer cache (the warm-host test's whole
    point); without it the layer cache applies as normal.
    """
    cmd = [
        "docker",
        "build",
        "-f",
        str(IMAGES_DIR / "base.Dockerfile"),
        "--build-arg",
        f"DAYDREAM_WHEEL={wheel}",
    ]
    if no_cache:
        cmd.append("--no-cache")
    for tag in tags:
        cmd += ["-t", tag]
    cmd.append(str(IMAGES_DIR))
    return cmd


def build_base_image() -> list[str]:
    """Build and tag the shared base image. Returns the tags applied."""
    wheel = build_wheel(DIST_DIR)
    tags = [base_tag(), BASE_LATEST]
    cmd = _base_build_cmd(wheel, tags)
    _stream(cmd)
    return tags


def _build_base() -> tuple[int, str | None]:
    """Build the base image and report its tags. Returns ``(exit_code, versioned_tag)``.

    On success the immutable versioned tag produced by ``base_tag()`` is
    returned as the identity a snapshot build must consume — never the mutable
    alias. The tag is selected by shape (the same ``_immutable_base_image``
    predicate ``--no-base`` validates against), not by position:
    ``build_base_image()`` tags the same build twice, and whichever order it
    applies them in a snapshot must not inherit the alias.
    """
    try:
        tags = build_base_image()
    except subprocess.CalledProcessError as exc:
        # The docker log above is the message; a traceback would only bury it.
        print(f"FAILED base image: exit {exc.returncode}", file=sys.stderr)
        return (1, None)
    for tag in tags:
        print(f"built {tag}")
    versioned = next((t for t in tags if _immutable_base_image(t) is not None), None)
    if versioned is None:
        print(f"FAILED base image: no immutable versioned tag among {tags}", file=sys.stderr)
        return (1, None)
    return (0, versioned)


def _base_image_present(base_image: str) -> bool:
    """Whether *base_image* exists in the local docker daemon.

    A well-formed but locally absent ``--no-base`` identity would otherwise
    fail late, per repo build, misattributed as a per-PR build failure; this is
    the same existence probe the session fixture uses before ``--base-only``.
    """
    probe = subprocess.run(["docker", "image", "inspect", base_image], capture_output=True, check=False)
    return probe.returncode == 0


def write_setup_script(ctx: Path, setup_cmds: list[str]) -> None:
    """Render the manifest entry's ``setup_cmds`` into ``<ctx>/setup.sh``.

    An empty list yields a script that does nothing, so every repo image keeps the
    same layer shape whether or not its repository needs a dependency install.
    """
    lines = [
        "#!/bin/sh",
        "# Generated by images/build_images.py from the manifest entry's setup_cmds.",
        "set -eu",
        *setup_cmds,
    ]
    (ctx / "setup.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_mirror(entry: _ManifestEntry, ctx: Path, *, red: bool) -> dict[str, str]:
    """Put a bare ``mirror.git`` in the build context *ctx*.

    Returns:
        A SHA translation map, empty for a real clone. It matters only under
        ``--red``: planting a failing assertion rewrites the fixture's head commit,
        so that commit's SHA changes and the SHA pinned in the corpus no longer
        exists. Without the remap the build would die at ``git checkout`` and prove
        nothing about the green-baseline gate.
    """
    mirror = ctx / "mirror.git"
    if entry.clone_url != FIXTURE_CLONE_URL:
        _stream(["git", "clone", "--mirror", entry.clone_url, str(mirror)])
        return {}

    with tempfile.TemporaryDirectory(prefix="daydream-rl-fixture-") as tmp:
        repo = build_fixture_repo(Path(tmp) / "repo", red=red)
        _stream(["git", "clone", "--mirror", str(repo.path), str(mirror)])
    return {
        FIXTURE_BASE_SHA: repo.base_sha,
        FIXTURE_PR1_HEAD_SHA: repo.pr1_head_sha,
        FIXTURE_PR2_HEAD_SHA: repo.pr2_head_sha,
    }


def _validate_red_flags(*, red: bool, base_only: bool, manifest: dict[str, _ManifestEntry], prs: list) -> int | None:
    """Validate ``--red`` constraints before any build starts.

    Returns ``None`` when the flags are valid, or an exit code (2) when
    a constraint is violated.
    """
    if red and base_only:
        print("--red cannot be combined with --base-only", file=sys.stderr)
        return 2

    if red:
        fixture_selected = (
            FIXTURE_SLUG in manifest
            and manifest[FIXTURE_SLUG].clone_url == FIXTURE_CLONE_URL
            and any(_repo_slug(pr.clone_url) == FIXTURE_SLUG for pr in prs)
        )
        if not fixture_selected:
            print(
                "--red requires at least one selected fixture PR backed by " + FIXTURE_CLONE_URL,
                file=sys.stderr,
            )
            return 2
    return None


def build_repo_image(entry: _ManifestEntry, *, head_sha: str, base_sha: str, base_image: str, red: bool) -> str:
    """Build one PR-snapshot image and return the tag it was given.

    Raises:
        subprocess.CalledProcessError: If any layer fails — including the final
            test layer, which is the green-baseline gate.
    """
    with tempfile.TemporaryDirectory(prefix="daydream-rl-ctx-") as tmp:
        ctx = Path(tmp)
        remap = materialize_mirror(entry, ctx, red=red)
        write_setup_script(ctx, entry.setup_cmds)

        head = remap.get(head_sha, head_sha)
        base = remap.get(base_sha, base_sha)
        tag = f"{entry.image}:{head[:12]}"
        try:
            _stream(
                [
                    "docker",
                    "build",
                    "-f",
                    str(IMAGES_DIR / "repo.Dockerfile"),
                    "--build-arg",
                    f"BASE_IMAGE={base_image}",
                    "--build-arg",
                    f"HEAD_SHA={head}",
                    "--build-arg",
                    f"BASE_SHA={base}",
                    "--build-arg",
                    f"TEST_COMMAND={entry.test_command}",
                    "-t",
                    tag,
                    str(ctx),
                ]
            )
        except subprocess.CalledProcessError:
            # The green-baseline gate tripped: the suite at the head commit was
            # red, so the build died at the final test layer. Some docker
            # daemons still leave the partially-built image tagged on failure;
            # that dangling tag is exactly what test_images.py asserts must NOT
            # exist for a red baseline. Remove it so the invariant holds
            # regardless of the host's docker behavior, then re-raise.
            subprocess.run(
                ["docker", "rmi", tag],
                capture_output=True,
                text=True,
                check=False,
            )
            raise
    return tag


def _resolve_base_image(args: argparse.Namespace) -> tuple[int, str | None]:
    """Resolve the immutable base image every snapshot build pins.

    Returns ``(exit_code, base_image)``; a non-zero exit means the caller must
    stop before any repo build. The ``--no-base`` path reuses an explicit
    immutable identity and refuses (status 2) an invalid or locally absent one;
    the fresh path builds the base and returns its versioned tag. A
    ``(0, None)`` success from the fresh path is an invariant violation.
    """
    if args.no_base is not None:
        base_image = _immutable_base_image(args.no_base)
        if base_image is None:
            print(
                f"invalid immutable base image {args.no_base!r}: expected {IMMUTABLE_BASE_FORMAT}; "
                "'latest' is not allowed",
                file=sys.stderr,
            )
            return (2, None)
        if not _base_image_present(base_image):
            print(f"base image {base_image!r} does not exist locally; build it first with --base-only", file=sys.stderr)
            return (2, None)
        print(f"skipping base build; reusing {base_image}")
        return (0, base_image)

    status, base_image = _build_base()
    if status:
        return (status, None)
    # A successful base build must produce the immutable versioned tag; a
    # None tag on the success path is an invariant violation, never a cue to
    # fall back. Deliberately a runtime check rather than an ``assert``:
    # ``python -O`` strips asserts, and a stripped check would let a
    # ``(0, None)`` base build feed BASE_IMAGE=None into every snapshot.
    if base_image is None:
        raise RuntimeError("_build_base() reported success without an immutable versioned tag")
    return (0, base_image)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the daydream-review-v1 rollout images.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="images/manifest.toml")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="harvested-corpus directory")
    parser.add_argument("--only", metavar="SLUG", help="build only this repo slug (owner/name)")
    base = parser.add_mutually_exclusive_group()
    base.add_argument(
        "--no-base",
        metavar="BASE_IMAGE",
        help=f"reuse the given immutable base image ({IMMUTABLE_BASE_FORMAT})",
    )
    base.add_argument("--base-only", action="store_true", help=f"build {BASE_LATEST} and no repo image")
    parser.add_argument(
        "--red",
        action="store_true",
        help="plant a failing test in the fixture repo's head commit; the build MUST fail at the test layer",
    )
    args = parser.parse_args(argv)

    if args.red and args.base_only:
        print("--red cannot be combined with --base-only", file=sys.stderr)
        return 2

    if args.base_only:
        status, _ = _build_base()
        return status

    manifest = load_manifest(args.manifest)
    prs = sorted(harvested_corpus(args.corpus).prs, key=lambda pr: (_repo_slug(pr.clone_url), pr.pr_number))
    if args.only:
        prs = [pr for pr in prs if _repo_slug(pr.clone_url) == args.only]
        if not prs:
            print(f"no PR in {args.corpus} belongs to {args.only}", file=sys.stderr)
            return 2

    red_status = _validate_red_flags(red=args.red, base_only=args.base_only, manifest=manifest, prs=prs)
    if red_status is not None:
        return red_status

    status, base_image = _resolve_base_image(args)
    if status:
        return status
    # ``_resolve_base_image`` raises on ``(0, None)``, so a success here always
    # carries a base image; the cast narrows the tuple for the type checker
    # without duplicating the runtime guard in main.
    base_image = cast(str, base_image)

    built: list[str] = []
    failed: list[str] = []
    for pr in prs:
        slug = _repo_slug(pr.clone_url)
        entry = manifest.get(slug)
        if entry is None:
            # Same rule as the taskset: a corpus PR with no manifest entry is an
            # error, never a silent skip.
            print(f"FAILED {slug}#{pr.pr_number}: no entry in {args.manifest}", file=sys.stderr)
            failed.append(f"{slug}#{pr.pr_number}")
            continue
        if not pr.base_sha:
            print(f"FAILED {slug}#{pr.pr_number}: corpus record has no base_sha", file=sys.stderr)
            failed.append(f"{slug}#{pr.pr_number}")
            continue
        try:
            built.append(
                build_repo_image(
                    entry,
                    head_sha=pr.head_sha,
                    base_sha=pr.base_sha,
                    base_image=base_image,
                    red=args.red,
                )
            )
        except subprocess.CalledProcessError as exc:
            # Not swallowed: the log above is the real message and the exit code
            # below is non-zero. The remaining PRs are still attempted so one red
            # baseline does not hide the state of the rest of the corpus.
            print(f"FAILED {slug}#{pr.pr_number}: exit {exc.returncode}", file=sys.stderr)
            failed.append(f"{slug}#{pr.pr_number}")

    for tag in built:
        print(f"built {tag}")
    if failed:
        print(f"{len(failed)} image(s) failed to build: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
