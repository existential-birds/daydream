"""Real-Git/Fake-GitHub harness for the remote-CI evidence paths.

This module owns every fake-GitHub identity/evidence response the remote-CI tests
need, plus the one pre-push-hook seeder that publishes them for the *pushed* SHA.
``NoCIRemote`` is the fixture-facing convenience wrapper for "a push that lands
and whose PR has no configured CI"; the remote-CI outcome tests drive
:func:`start_remote_ci_fake` directly.
"""

from __future__ import annotations

import re
import shlex
import threading
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest

from daydream import remote_ci
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git

_FULL_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")

# The single fixture PR identity every remote-CI test serves.
PR_NUMBER = 7
BASE_REPOSITORY = "base-user/project"
HEAD_REPOSITORY = "fork-user/project"


def _wait_for_pushed_sha(sha_path: Path, stop: threading.Event, *, poll_seconds: float = 0.01,) -> str | None:
    """Wait until the pre-push hook publishes one complete commit SHA."""
    while True:
        try:
            candidate = sha_path.read_text().strip()
        except FileNotFoundError:
            candidate = ""
        if _FULL_SHA_RE.fullmatch(candidate) is not None:
            return candidate
        if stop.wait(poll_seconds):
            return None


def write_pre_push_sha_hook(project: Path, *, sha_path: Path, ready_path: Path, marker: Path | None = None,) -> None:
    """Install a pre-push hook that publishes the pushed SHA and waits for ``ready_path``.

    The hook atomically writes the pushed ``local_sha`` to *sha_path*, then blocks
    until the test writes *ready_path* (bounded by a 3000 * 0.01s retry loop). When
    *marker* is given, the hook also writes ``ran`` to it as evidence it executed.
    """
    hook = project / ".git" / "hooks" / "pre-push"
    if hook.exists():
        raise AssertionError(f"refusing to replace existing pre-push hook: {hook}")
    sha_temp_prefix = f"{sha_path}.tmp"
    marker_line = f"printf '%s\\n' ran > {shlex.quote(str(marker))}\n" if marker else ""
    hook.write_text(
        "#!/bin/sh\n"
        "read local_ref local_sha remote_ref remote_sha\n"
        f"{marker_line}"
        f"sha_tmp={shlex.quote(sha_temp_prefix)}.$$\n"
        "cleanup_sha_tmp() { rm -f \"$sha_tmp\"; }\n"
        "trap cleanup_sha_tmp EXIT HUP INT TERM\n"
        "printf '%s\\n' \"$local_sha\" > \"$sha_tmp\"\n"
        f"mv \"$sha_tmp\" {shlex.quote(str(sha_path))}\n"
        "trap - EXIT HUP INT TERM\n"
        "i=0\n"
        f"while [ ! -f {shlex.quote(str(ready_path))} ]; do\n"
        "  i=$((i + 1))\n"
        "  [ \"$i\" -lt 3000 ] || exit 91\n"
        "  sleep 0.01\n"
        "done\n"
    )
    hook.chmod(0o755)


def seed_pr_identity(fake_gh: FakeGh, *, head_sha: str, branch: str = "feature", title: str = "Fix") -> None:
    """Publish P04 identity for *head_sha*: the repo view plus the PR view."""
    fake_gh.set_response("repo-view", value=BASE_REPOSITORY)
    fake_gh.serve_pr_view(
        {
            "number": PR_NUMBER, "title": title, "body": "", "state": "OPEN", "headRefName": branch,
            "baseRefName": "main", "headRefOid": head_sha,
            "url": f"https://github.com/{BASE_REPOSITORY}/pull/{PR_NUMBER}",
            "headRepository": {"nameWithOwner": HEAD_REPOSITORY}, "headRepositoryOwner": {"login": "fork-user"},
        }
    )


def serve_no_ci_evidence(fake_gh: FakeGh, *, branch: str, head_sha: str) -> None:
    """Publish the exact-SHA REST catalog proving PR 7 has no configured CI."""
    fake_gh.set_response("GET", f"repos/{BASE_REPOSITORY}/pulls/{PR_NUMBER}", {
            "number": PR_NUMBER, "html_url": f"https://github.com/{BASE_REPOSITORY}/pull/{PR_NUMBER}", "state": "open",
            "base": {"ref": "main", "repo": {"full_name": BASE_REPOSITORY}},
            "head": {"ref": branch, "sha": head_sha, "repo": {"full_name": HEAD_REPOSITORY}},
            "merge_commit_sha": None,
        }
    )
    fake_gh.set_response("GET", f"repos/{BASE_REPOSITORY}/rules/branches/main?per_page=100&page=1", [])
    fake_gh.set_response(
        "GET", f"repos/{BASE_REPOSITORY}/branches/main/protection/required_status_checks",
        {"strict": False, "contexts": [], "checks": []},
    )
    fake_gh.set_response("GET", f"repos/{BASE_REPOSITORY}/actions/workflows?per_page=100&page=1",
        {"total_count": 0, "workflows": []},
    )
    fake_gh.set_response(
        "GET", f"repos/{BASE_REPOSITORY}/commits/{head_sha}/check-runs?filter=latest&per_page=100&page=1",
        {"total_count": 0, "check_runs": []},
    )
    fake_gh.set_response("GET", f"repos/{BASE_REPOSITORY}/commits/{head_sha}/statuses?per_page=100&page=1", [])


class RemoteCISeeder(NamedTuple):
    """A running seeder thread plus the state its owning test asserts on."""

    thread: threading.Thread
    errors: list[BaseException]
    stop: threading.Event
    pushed_shas: list[str]


def start_remote_ci_fake(
    project: Path, fake_gh: FakeGh, hook_marker: Path, *, outcome: str, branch: str = "feature",
    on_pushed: Callable[[str], None] | None = None,
) -> RemoteCISeeder:
    """Let the real pre-push hook publish the new SHA to the external fake.

    *outcome* selects the evidence the seeder publishes for that SHA: ``"no_ci"``
    (the shared :func:`serve_no_ci_evidence` catalog), ``"blocking"`` (a gh process
    that never returns), ``"failed"``, ``"delayed"`` or ``"merge-delayed"``. A
    caller that owns its own evidence catalog passes *on_pushed* instead, which
    replaces the outcome seeding; the fixture paths (``hook_marker``-derived sha
    and ready files) are unchanged either way.
    """
    sha_path = hook_marker.with_name(hook_marker.name + " sha")
    ready_path = hook_marker.with_name(hook_marker.name + " ready")
    write_pre_push_sha_hook(project, sha_path=sha_path, ready_path=ready_path, marker=hook_marker)
    errors: list[BaseException] = []
    stop = threading.Event()
    pushed_shas: list[str] = []
    def seed() -> None:
        try:
            sha = _wait_for_pushed_sha(sha_path, stop)
            if sha is None:
                return
            pushed_shas.append(sha)
            if on_pushed is not None:
                on_pushed(sha)
                ready_path.write_text("ready\n")
                return
            if outcome == "no_ci":
                serve_no_ci_evidence(fake_gh, branch=branch, head_sha=sha)
                ready_path.write_text("ready\n")
                return
            pr_row = {
                "number": PR_NUMBER, "html_url": f"https://github.com/{BASE_REPOSITORY}/pull/{PR_NUMBER}",
                "state": "open", "base": {"ref": "main", "repo": {"full_name": BASE_REPOSITORY}},
                "head": {"ref": branch, "sha": sha, "repo": {"full_name": HEAD_REPOSITORY}},
                "merge_commit_sha": "e" * 40 if outcome == "merge-delayed" else None,
            }
            if outcome == "blocking":
                fake_gh.serve_blocking_process(
                    f"GET repos/{BASE_REPOSITORY}/pulls/{PR_NUMBER}",
                    pid_file=hook_marker.with_name(hook_marker.name + " pids"),
                )
                ready_path.write_text("ready\n")
                return
            fake_gh.set_response("GET", f"repos/{BASE_REPOSITORY}/pulls/{PR_NUMBER}", pr_row)
            fake_gh.set_response("GET", f"repos/{BASE_REPOSITORY}/rules/branches/main?per_page=100&page=1", [])
            required_name = "Build [matrix]" if outcome == "failed" else "Build"
            pinned_checks = (
                [{"context": required_name, "app_id": 10}]
                if outcome in {"failed", "delayed", "merge-delayed"}
                else []
            )
            fake_gh.set_response(
                "GET", f"repos/{BASE_REPOSITORY}/branches/main/protection/required_status_checks",
                {"strict": False, "contexts": [], "checks": pinned_checks},
            )
            fake_gh.set_response(
                "GET", f"repos/{BASE_REPOSITORY}/actions/workflows?per_page=100&page=1",
                {"total_count": 0, "workflows": []},
            )
            checks: list[dict[str, object]] = []
            if outcome in {"failed", "delayed", "merge-delayed"}:
                checks.append(
                    {
                        "id": 1, "name": required_name, "head_sha": sha, "app": {"id": 10}, "status": "completed",
                        "conclusion": "failure" if outcome == "failed" else "success",
                        "details_url": (
                            "https://github.com/base-user/project/actions/runs/[matrix]/7"
                            if outcome == "failed"
                            else "https://github.com/base-user/project/actions/runs/7"
                        ),
                        "output": {
                            "title": "Build failed", "summary": "secret=top-secret build failed",
                            "text": "must not persist",
                        },
                    }
                )
            if outcome == "failed":
                checks.append(
                    {
                        "id": 2, "name": "Lint [optional]", "head_sha": sha, "app": {"id": 20}, "status": "completed",
                        "conclusion": "failure",
                        "details_url": "https://github.com/base-user/project/actions/runs/[optional]/8",
                        "output": {
                            "title": "Advisory lint failed", "summary": "advisory failure", "text": "must not persist",
                        },
                    }
                )
            check_key = (
                f"GET repos/{BASE_REPOSITORY}/commits/"
                f"{sha}/check-runs?filter=latest&per_page=100&page=1"
            )
            if outcome == "delayed":
                fake_gh.set_response_sequence(
                    check_key,
                    [
                        {"total_count": 0, "check_runs": []}, {"total_count": 1, "check_runs": checks},
                        {"total_count": 1, "check_runs": checks},
                    ],
                )
            else:
                fake_gh.set_response(
                    "GET", check_key.removeprefix("GET "), {"total_count": len(checks), "check_runs": checks},
                )
            fake_gh.set_response("GET", f"repos/{BASE_REPOSITORY}/commits/{sha}/statuses?per_page=100&page=1", [])
            if outcome == "merge-delayed":
                merge_sha = "e" * 40
                merge_key = (
                    f"GET repos/{BASE_REPOSITORY}/commits/"
                    f"{merge_sha}/check-runs?filter=latest&per_page=100&page=1"
                )
                pending = dict(checks[0], head_sha=merge_sha, status="in_progress", conclusion=None)
                passed = dict(checks[0], head_sha=merge_sha)
                fake_gh.set_response_sequence(
                    merge_key,
                    [
                        {"total_count": 1, "check_runs": [pending]}, {"total_count": 1, "check_runs": [passed]},
                        {"total_count": 1, "check_runs": [passed]},
                    ],
                )
                fake_gh.set_response(
                    "GET", f"repos/{BASE_REPOSITORY}/commits/{merge_sha}/statuses?per_page=100&page=1", [],
                )
            ready_path.write_text("ready\n")
        except BaseException as exc:  # thread failures are re-raised by the test
            errors.append(exc)
            ready_path.write_text("failed\n")
    thread = threading.Thread(target=seed, daemon=True)
    thread.start()
    return RemoteCISeeder(thread, errors, stop, pushed_shas)


def finish_remote_ci_fake(seeder: RemoteCISeeder) -> None:
    """Stop and join a seeder before its owning test releases fixture state."""
    seeder.stop.set()
    seeder.thread.join(timeout=5)
    assert not seeder.thread.is_alive(), "remote-CI seeding thread did not stop"
    assert seeder.errors == []


class NoCIRemote:
    """Route a GitHub-shaped remote to a real bare repo and serve no-CI evidence."""

    pr_number = PR_NUMBER
    base_repository = BASE_REPOSITORY
    head_repository = HEAD_REPOSITORY

    def __init__(self, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,) -> None:
        self._fake_gh = fake_gh
        self._tmp_path = tmp_path
        self._seeders: list[RemoteCISeeder] = []
        monkeypatch.setattr(remote_ci, "DEFAULT_LIMITS",
            remote_ci.RemoteCILimits(poll_seconds=0.01,
                # Real Git/gh subprocess startup under parallel CI exhausted
                # a 5-second window; retain this loaded-host safety margin.
                # Two concurrent 16-worker pytest runs (e.g. overlapping
                # pre-push gates on one host) exhausted even 15s: the
                # seeding threads starve behind 32 saturated workers and
                # the run reports "missing" CI for an evidence payload
                # that arrives moments later. 45s covers the worst
                # observed starvation with room to spare; a genuinely
                # broken harness still fails fast enough for CI.
                # completion >= discovery is a RemoteCILimits invariant;
                # the completion window itself is inert here because the
                # fake gh serves final no-CI evidence during discovery.
                discovery_seconds=45, completion_seconds=90, request_seconds=10,
            ),
        )

    def connect(self, repo: Path, bare: Path, *, remote: str = "origin") -> None:
        """Connect *remote* through a GitHub URL and seed exact-SHA no-CI state."""
        raw_remote = f"https://github.com/{self.head_repository}.git"
        git(repo, "config", f"url.{bare.resolve().as_uri()}.insteadOf", raw_remote)
        remotes = git(repo, "remote").splitlines()
        if remote in remotes:
            git(repo, "remote", "set-url", remote, raw_remote)
        else:
            git(repo, "remote", "add", remote, raw_remote)

        branch = git(repo, "branch", "--show-current")
        initial_sha = git(repo, "rev-parse", "HEAD")
        self._serve_pr(branch=branch, head_sha=initial_sha)

        def seed_pushed(head_sha: str) -> None:
            """Rebind PR identity to the pushed SHA, then serve its no-CI catalog.

            Both calls stay on ``self`` so a test can replace either half (the
            mixed-case identity test overrides both) without touching the seeder.
            """
            self._serve_pr(branch=branch, head_sha=head_sha)
            self._serve_no_ci(branch=branch, head_sha=head_sha)

        marker = self._tmp_path / f"remote-ci-{len(self._seeders)}"
        self._seeders.append(
            start_remote_ci_fake(repo, self._fake_gh, marker, outcome="no_ci", branch=branch, on_pushed=seed_pushed)
        )

    def finish(self) -> None:
        """Join each seeding thread and surface its failure in the owning test."""
        for seeder in self._seeders:
            finish_remote_ci_fake(seeder)
            assert seeder.pushed_shas, "real push never reached the pre-push hook"

    def _serve_pr(self, *, branch: str, head_sha: str) -> None:
        seed_pr_identity(self._fake_gh, branch=branch, head_sha=head_sha, title="Fixture PR")

    def _serve_no_ci(self, *, branch: str, head_sha: str) -> None:
        serve_no_ci_evidence(self._fake_gh, branch=branch, head_sha=head_sha)
