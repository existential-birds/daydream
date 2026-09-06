"""Real-Git/Fake-GitHub harness for successful runs with no configured CI."""

from __future__ import annotations

import re
import shlex
import threading
from pathlib import Path

import pytest

from daydream import remote_ci
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git

_FULL_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


def _wait_for_pushed_sha(
    sha_path: Path,
    stop: threading.Event,
    *,
    poll_seconds: float = 0.01,
) -> str | None:
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


class NoCIRemote:
    """Route a GitHub-shaped remote to a real bare repo and serve no-CI evidence."""

    pr_number = 7
    base_repository = "base-user/project"
    head_repository = "fork-user/project"

    def __init__(
        self,
        fake_gh: FakeGh,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._fake_gh = fake_gh
        self._tmp_path = tmp_path
        self._threads: list[
            tuple[threading.Thread, list[BaseException], threading.Event, list[str]]
        ] = []
        monkeypatch.setattr(
            remote_ci,
            "DEFAULT_LIMITS",
            remote_ci.RemoteCILimits(
                poll_seconds=0.01,
                discovery_seconds=15,
                completion_seconds=20,
                request_seconds=10,
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

        marker = self._tmp_path / f"remote-ci-{len(self._threads)}"
        sha_path = marker.with_suffix(".sha")
        ready_path = marker.with_suffix(".ready")
        hook = repo / ".git" / "hooks" / "pre-push"
        if hook.exists():
            raise AssertionError(f"refusing to replace existing pre-push hook: {hook}")
        sha_temp_prefix = f"{sha_path}.tmp"
        hook.write_text(
            "#!/bin/sh\n"
            "read local_ref local_sha remote_ref remote_sha\n"
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
        errors: list[BaseException] = []
        stop = threading.Event()
        pushed_shas: list[str] = []

        def seed_after_push() -> None:
            try:
                pushed_sha = _wait_for_pushed_sha(sha_path, stop)
                if pushed_sha is None:
                    return
                pushed_shas.append(pushed_sha)
                self._serve_pr(branch=branch, head_sha=pushed_sha)
                self._serve_no_ci(branch=branch, head_sha=pushed_sha)
                ready_path.write_text("ready\n")
            except BaseException as exc:
                errors.append(exc)
                ready_path.write_text("failed\n")

        thread = threading.Thread(target=seed_after_push, daemon=True)
        thread.start()
        self._threads.append((thread, errors, stop, pushed_shas))

    def finish(self) -> None:
        """Join each seeding thread and surface its failure in the owning test."""
        for thread, errors, stop, pushed_shas in self._threads:
            stop.set()
            thread.join(timeout=5)
            assert not thread.is_alive(), "no-CI seeding thread did not stop"
            assert errors == []
            assert pushed_shas, "real push never reached the pre-push hook"

    def _serve_pr(self, *, branch: str, head_sha: str) -> None:
        self._fake_gh.set_response("repo-view", value=self.base_repository)
        self._fake_gh.serve_pr_view(
            {
                "number": self.pr_number,
                "title": "Fixture PR",
                "body": "",
                "state": "OPEN",
                "headRefName": branch,
                "baseRefName": "main",
                "headRefOid": head_sha,
                "url": f"https://github.com/{self.base_repository}/pull/{self.pr_number}",
                "headRepository": {"nameWithOwner": self.head_repository},
                "headRepositoryOwner": {"login": "fork-user"},
            }
        )

    def _serve_no_ci(self, *, branch: str, head_sha: str) -> None:
        pull = {
            "number": self.pr_number,
            "html_url": f"https://github.com/{self.base_repository}/pull/{self.pr_number}",
            "state": "open",
            "base": {"ref": "main", "repo": {"full_name": self.base_repository}},
            "head": {
                "ref": branch,
                "sha": head_sha,
                "repo": {"full_name": self.head_repository},
            },
            "merge_commit_sha": None,
        }
        self._fake_gh.set_response(
            "GET", f"repos/{self.base_repository}/pulls/{self.pr_number}", pull
        )
        self._fake_gh.set_response(
            "GET", f"repos/{self.base_repository}/rules/branches/main?per_page=100&page=1", []
        )
        self._fake_gh.set_response(
            "GET",
            f"repos/{self.base_repository}/branches/main/protection/required_status_checks",
            {"strict": False, "contexts": [], "checks": []},
        )
        self._fake_gh.set_response(
            "GET",
            f"repos/{self.base_repository}/actions/workflows?per_page=100&page=1",
            {"total_count": 0, "workflows": []},
        )
        self._fake_gh.set_response(
            "GET",
            f"repos/{self.base_repository}/commits/{head_sha}/check-runs?filter=latest&per_page=100&page=1",
            {"total_count": 0, "check_runs": []},
        )
        self._fake_gh.set_response(
            "GET",
            f"repos/{self.base_repository}/commits/{head_sha}/statuses?per_page=100&page=1",
            [],
        )
