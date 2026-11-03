"""Real-process tests for bounded asynchronous GitHub API reads."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.git_ops import process as git_process
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import refreshing_session
from tests.harness.processes import fd_count, wait_for_fd_baseline, wait_for_process_group_exit


def _budget(
    *, seconds: float = 30.0, per_request: float = 5.0, monotonic: Callable[[], float] = time.monotonic,
) -> git_ops.GitHubRequestBudget:
    now = monotonic()
    return git_ops.GitHubRequestBudget(deadline=now + seconds, per_request_seconds=per_request, monotonic=monotonic)


def _blocking_auth(
    loop: asyncio.AbstractEventLoop, started: asyncio.Event, release: threading.Event, finished: threading.Event,
) -> Any:
    class BlockingAuth:
        def environment_for_request(self) -> None:
            loop.call_soon_threadsafe(started.set)
            try:
                release.wait(timeout=2)
            finally:
                finished.set()
    return BlockingAuth


def _page_endpoint(endpoint: str, page: int, *, per_page: int = 100) -> str:
    separator = "&" if "?" in endpoint else "?"
    return f"{endpoint}{separator}per_page={per_page}&page={page}"


def _forbid_spawn(monkeypatch: pytest.MonkeyPatch, message: str) -> list[Any]:
    """Patch the ``gh`` subprocess seam to reject any spawn with *message*. Returns the observation list; an empty
    list proves the path under test never reached the process boundary."""
    spawn_calls: list[Any] = []
    async def create_process(*args: Any, **kwargs: Any) -> Any:
        spawn_calls.append((args, kwargs))
        raise AssertionError(message)
    monkeypatch.setattr("daydream.git_ops.process.asyncio.create_subprocess_exec", create_process)
    return spawn_calls


async def test_sync_and_async_requests_share_one_session_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Concurrent sync/async requests share one refresh while independent sessions and snapshots stay isolated."""
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-token-must-not-merge")
    captured: list[dict[str, str] | None] = []
    refresh_calls = 0

    class CompletedProcess:
        returncode = 0
        async def communicate(self) -> tuple[bytes, bytes]:
            return b"{}", b""

    def refresh() -> tuple[git_ops.StaticGitHubAuth, float]:
        nonlocal refresh_calls
        refresh_calls += 1
        time.sleep(0.02)
        return (
            git_ops.StaticGitHubAuth({"PATH": "/tools", "GH_TOKEN": "ghs_shared_fresh_token_1234567890"}), float("inf"),
        )

    def run_process(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.append(kwargs["env"])
        return subprocess.CompletedProcess(args[0], 0, "{}", "")

    async def create_process(*_args: Any, **kwargs: Any) -> CompletedProcess:
        captured.append(kwargs["env"])
        return CompletedProcess()

    auth = git_ops.RefreshingGitHubAuth(
        git_ops.StaticGitHubAuth({"PATH": "/tools", "GH_TOKEN": "ghs_shared_expired_token_1234567890"}), expires_at=0,
        refresh=refresh,
    )
    other_refreshes = {"other": 0}
    other = refreshing_session("other", other_refreshes)
    source = {"PATH": "/tools", "GH_TOKEN": "ghs_static_token_1234567890"}
    static = git_ops.StaticGitHubAuth(source)
    source["GH_TOKEN"] = "mutated"
    monkeypatch.setattr(subprocess, "run", run_process)
    monkeypatch.setattr("daydream.git_ops.process.asyncio.create_subprocess_exec", create_process)
    await asyncio.gather(
        asyncio.to_thread(git_process._run_gh, tmp_path, ["api", "/user"], auth=auth),
        git_process._run_gh_async(tmp_path, ["api", "/user"], auth=auth, budget=_budget()),
        git_process._run_gh_async(tmp_path, ["api", "/user"], auth=other, budget=_budget()),
        asyncio.to_thread(git_process._run_gh, tmp_path, ["api", "/user"], auth=static),
    )
    assert refresh_calls == 1 and other_refreshes == {"other": 1}
    assert len(captured) == 4
    assert captured.count({"PATH": "/tools", "GH_TOKEN": "ghs_shared_fresh_token_1234567890"}) == 2
    assert captured.count({"PATH": "/other/tools", "GH_TOKEN": "ghs_other_fresh_token_1234567890"}) == 1
    assert captured.count({"PATH": "/tools", "GH_TOKEN": "ghs_static_token_1234567890"}) == 1
    assert "ghs_static_token" not in repr(static)
    await git_process._run_gh_async(tmp_path, ["api", "/user"], auth=git_ops.INHERIT_GITHUB_AUTH, budget=_budget())
    assert captured[-1] is None


async def test_async_gh_requests_never_spawn_after_auth_or_budget_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Expired budgets, slow or cancelled auth resolution and spawn failures never reach a running gh request."""
    auth_calls = 0
    class Auth:
        def environment_for_request(self) -> None:
            nonlocal auth_calls
            auth_calls += 1
    spawn_calls = _forbid_spawn(monkeypatch, "a failed auth or budget must not spawn gh")
    budget = git_ops.GitHubRequestBudget(deadline=0.0, per_request_seconds=1.0, monotonic=lambda: 1.0)
    with pytest.raises(git_ops.DeadlineExpired):
        await git_process._run_gh_async(tmp_path, ["api", "/user"], auth=Auth(), budget=budget)
    assert auth_calls == 0
    times = iter((0.0, 2.0))
    budget = git_ops.GitHubRequestBudget(deadline=1.0, per_request_seconds=1.0, monotonic=lambda: next(times))
    with pytest.raises(git_ops.DeadlineExpired):
        await git_process._run_gh_async(tmp_path, ["api", "/user"], auth=git_ops.INHERIT_GITHUB_AUTH, budget=budget)

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    BlockingAuth = _blocking_auth(loop, started, release, finished)
    task = asyncio.create_task(
        git_process._run_gh_async(
            tmp_path, ["api", "/user"], auth=BlockingAuth(), budget=_budget(seconds=2, per_request=1),
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        heartbeat = asyncio.Event()
        loop.call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(git_ops.GitTimeoutError, match="authentication timed out") as timed_out:
            await git_process._run_gh_async(
                tmp_path, ["api", "Authorization: Bearer secret-value"], auth=BlockingAuth(),
                budget=_budget(seconds=2, per_request=0.05),
            )
        assert "secret-value" not in str(timed_out.value) and timed_out.value.__cause__ is None
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 1)
    assert spawn_calls == []

    async def missing_gh(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("gh executable is absent")
    monkeypatch.setattr("daydream.git_ops.process.asyncio.create_subprocess_exec", missing_gh)
    with pytest.raises(git_ops.GitError, match="FileNotFoundError: gh executable is absent"):
        await git_process._run_gh_async(tmp_path, ["api", "/user"], budget=_budget())


async def test_bounded_github_reads_fail_closed_at_shape_and_limit_boundaries(fake_gh: FakeGh, tmp_path: Path) -> None:
    """Pagination limits, envelopes, capacity and absence are never read as a complete or empty answer."""
    with pytest.raises(git_ops.GitError, match="pagination limits must be positive"):
        await git_ops.gh_api_bounded_pages(
            tmp_path, "repos/acme/widgets/statuses", envelope=None,
            limits=git_ops.GitHubPageLimits(per_page=100, max_pages=0), budget=_budget(),
        )
    assert fake_gh.process_calls() == []

    calls = 0
    for index, (response, expected_error) in enumerate([
        ({"total_count": 2, "workflows": [{"id": 1}]}, "incomplete"),
        ({"total_count": 1001, "workflows": []}, "pagination limit"),  # rejected before a second call
        ([], "shape"), ({"workflows": "wrong"}, "shape"),
    ]):
        endpoint = f"repos/acme/widgets{index}/actions/workflows"
        fake_gh.set_response("GET", _page_endpoint(endpoint, 1), response)
        with pytest.raises(git_ops.GitError, match=expected_error):
            await git_ops.gh_api_bounded_pages(
                tmp_path, endpoint, envelope="workflows", limits=git_ops.GitHubPageLimits(), budget=_budget(),
            )
        calls += 1
        assert len(fake_gh.process_calls()) == calls

    max_pages = 2
    endpoint = "repos/acme/widgets/commits/" + "f" * 40 + "/statuses"
    for page in range(1, max_pages + 1):
        fake_gh.set_response(
            "GET", _page_endpoint(endpoint, page, per_page=2), [{"id": page * 2 - 1}, {"id": page * 2}],
        )
    with pytest.raises(git_ops.GitError, match="pagination limit"):
        await git_ops.gh_api_bounded_pages(
            tmp_path, endpoint, envelope=None, limits=git_ops.GitHubPageLimits(per_page=2, max_pages=max_pages),
            budget=_budget(),
        )
    assert len(fake_gh.process_calls()) == calls + max_pages

    protection = "repos/acme/widgets/branches/main/protection/required_status_checks"
    fake_gh.set_response("GET", protection, {"__error__": "gh: Branch not protected (HTTP 404)"})
    assert await git_ops.gh_classic_required_checks(tmp_path, "acme", "widgets", "main", budget=_budget()) is None
    for response, error in [
        ({"__error__": "API rate limit exceeded (HTTP 403)"}, git_ops.RateLimitError), ([], git_ops.GitError),
    ]:
        fake_gh.set_response("GET", protection, response)
        with pytest.raises(error):
            await git_ops.gh_classic_required_checks(tmp_path, "acme", "widgets", "main", budget=_budget())
    fake_gh.set_response("GET", "repos/acme/widgets/pulls/7", [])
    with pytest.raises(git_ops.GitError):
        await git_ops.gh_pr_ci_snapshot(tmp_path, "acme", "widgets", 7, budget=_budget())


async def _published_process_ids(path: Path, *, timeout_s: float = 5.0) -> dict[str, int]:
    """Await the blocking fake gh's direct/grandchild PID JSON."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return {"direct": int(loaded["direct"]), "grandchild": int(loaded["grandchild"])}
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            await asyncio.sleep(0.01)
    raise AssertionError(f"blocking fake gh did not publish {path}")


async def test_timed_out_request_kills_reaps_its_process_group_and_closes_fds(
    fake_gh: FakeGh, tmp_path: Path,
) -> None:
    """A per-request timeout kills the whole gh process group, reaps it and leaks no descriptors."""
    baseline = fd_count()
    endpoint = "repos/acme/widgets/commits/" + "f" * 40 + "/statuses"
    pid_file = tmp_path / "timeout-pids.json"
    fake_gh.serve_blocking_process("GET " + _page_endpoint(endpoint, 1), pid_file=pid_file)
    request = asyncio.create_task(git_ops.gh_api_bounded_pages(
        tmp_path, endpoint, envelope=None, limits=git_ops.GitHubPageLimits(),
        budget=_budget(seconds=5.0, per_request=0.5),
    ))
    try:
        pids = await _published_process_ids(pid_file)
    except BaseException:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        raise
    with pytest.raises(git_ops.GitTimeoutError, match="timed out"):
        await request
    await wait_for_process_group_exit(pids["direct"])
    assert pids["direct"] != pids["grandchild"]
    await wait_for_fd_baseline(baseline, timeout_s=2.0)
    assert len(fake_gh.process_calls()) == 1
