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
produces no image. That is not a nicety — the ``fix_tests_pass`` reward pays a
rollout for a suite that passes after its fix, so a baseline that was already red
makes that reward pure noise. This script never catches, retries or downgrades that
failure; a failed build exits non-zero and the image simply does not exist.

Usage::

    uv run python images/build_images.py
    uv run python images/build_images.py --only existential-birds/daydream-rl-fixture
    uv run python images/build_images.py --no-base --corpus ../corpora/train

``--red`` is the gate's own test: it plants a failing assertion in the fixture
repository's head commit and expects the build to die at the final layer.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from daydream.benchmark.corpus import harvested_corpus

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


def build_base_image() -> list[str]:
    """Build and tag the shared base image. Returns the tags applied."""
    wheel = build_wheel(DIST_DIR)
    tags = [base_tag(), BASE_LATEST]
    cmd = [
        "docker",
        "build",
        "-f",
        str(IMAGES_DIR / "base.Dockerfile"),
        "--build-arg",
        f"DAYDREAM_WHEEL={wheel}",
    ]
    for tag in tags:
        cmd += ["-t", tag]
    cmd.append(str(IMAGES_DIR))
    _stream(cmd)
    return tags


def _build_base() -> int:
    """Build the base image and report its tags. Returns a process exit code."""
    try:
        tags = build_base_image()
    except subprocess.CalledProcessError as exc:
        # The docker log above is the message; a traceback would only bury it.
        print(f"FAILED base image: exit {exc.returncode}", file=sys.stderr)
        return 1
    for tag in tags:
        print(f"built {tag}")
    return 0


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
    return tag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the daydream-review-v1 rollout images.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="images/manifest.toml")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="`daydream bench harvest` corpus dir")
    parser.add_argument("--only", metavar="SLUG", help="build only this repo slug (owner/name)")
    base = parser.add_mutually_exclusive_group()
    base.add_argument("--no-base", action="store_true", help=f"reuse the existing {BASE_LATEST} image")
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
        return _build_base()

    manifest = load_manifest(args.manifest)
    prs = sorted(harvested_corpus(args.corpus).prs, key=lambda pr: (_repo_slug(pr.clone_url), pr.pr_number))
    if args.only:
        prs = [pr for pr in prs if _repo_slug(pr.clone_url) == args.only]
        if not prs:
            print(f"no PR in {args.corpus} belongs to {args.only}", file=sys.stderr)
            return 2

    if args.red:
        fixture_selected = (
            FIXTURE_SLUG in manifest
            and manifest[FIXTURE_SLUG].clone_url == FIXTURE_CLONE_URL
            and any(_repo_slug(pr.clone_url) == FIXTURE_SLUG for pr in prs)
        )
        if not fixture_selected:
            print(
                "--red requires at least one selected fixture PR backed by fixture://daydream-rl-fixture",
                file=sys.stderr,
            )
            return 2

    if args.no_base:
        print(f"skipping base build; reusing {BASE_LATEST}")
    else:
        status = _build_base()
        if status:
            return status

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
                    base_image=BASE_LATEST,
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
