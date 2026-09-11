"""Real-process tests for bounded asynchronous GitHub API reads."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.backends._subprocess import terminate_process
from tests.harness.fake_gh import FakeGh


def _budget(
    *,
    seconds: float = 30.0,
    per_request: float = 5.0,
    monotonic: Callable[[], float] = time.monotonic,
) -> git_ops.GitHubRequestBudget:
    now = monotonic()
    return git_ops.GitHubRequestBudget(
        deadline=now + seconds,
        per_request_seconds=per_request,
        monotonic=monotonic,
    )


def _page_endpoint(endpoint: str, page: int, *, per_page: int = 100) -> str:
    separator = "&" if "?" in endpoint else "?"
    return f"{endpoint}{separator}per_page={per_page}&page={page}"


@pytest.mark.asyncio
async def test_async_requests_keep_refreshing_session_environments_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[dict[str, str] | None] = []
    refresh_calls = {"a": 0, "b": 0}

    class CompletedProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"{}", b""

    async def create_process(*_args: Any, **kwargs: Any) -> CompletedProcess:
        captured.append(kwargs["env"])
        return CompletedProcess()

    def session(name: str) -> git_ops.RefreshingGitHubAuth:
        def refresh() -> tuple[git_ops.StaticGitHubAuth, float]:
            refresh_calls[name] += 1
            return (
                git_ops.StaticGitHubAuth(
                    {
                        "PATH": f"/{name}/tools",
                        "GH_TOKEN": f"ghs_{name}_fresh_token_1234567890",
                    }
                ),
                float("inf"),
            )

        return git_ops.RefreshingGitHubAuth(
            git_ops.StaticGitHubAuth(
                {
                    "PATH": f"/{name}/tools",
                    "GH_TOKEN": f"ghs_{name}_expired_token_1234567890",
                }
            ),
            expires_at=0,
            refresh=refresh,
        )

    monkeypatch.setattr(
        "daydream.git_ops.asyncio.create_subprocess_exec",
        create_process,
    )
    await asyncio.gather(
        git_ops._run_gh_async(
            tmp_path,
            ["api", "/user"],
            auth=session("a"),
            budget=_budget(),
        ),
        git_ops._run_gh_async(
            tmp_path,
            ["api", "/user"],
            auth=session("b"),
            budget=_budget(),
        ),
    )
    await git_ops._run_gh_async(
        tmp_path,
        ["api", "/user"],
        auth=git_ops.INHERIT_GITHUB_AUTH,
        budget=_budget(),
    )

    assert sorted(env["GH_TOKEN"] for env in captured if env is not None) == [
        "ghs_a_fresh_token_1234567890",
        "ghs_b_fresh_token_1234567890",
    ]
    assert captured[-1] is None
    assert refresh_calls == {"a": 1, "b": 1}


@pytest.mark.asyncio
async def test_sync_and_async_requests_share_one_session_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
            git_ops.StaticGitHubAuth(
                {"PATH": "/tools", "GH_TOKEN": "ghs_shared_fresh_token_1234567890"}
            ),
            float("inf"),
        )

    def run_process(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.append(kwargs["env"])
        return subprocess.CompletedProcess(args[0], 0, "{}", "")

    async def create_process(*_args: Any, **kwargs: Any) -> CompletedProcess:
        captured.append(kwargs["env"])
        return CompletedProcess()

    auth = git_ops.RefreshingGitHubAuth(
        git_ops.StaticGitHubAuth(
            {"PATH": "/tools", "GH_TOKEN": "ghs_shared_expired_token_1234567890"}
        ),
        expires_at=0,
        refresh=refresh,
    )
    monkeypatch.setattr(subprocess, "run", run_process)
    monkeypatch.setattr(
        "daydream.git_ops.asyncio.create_subprocess_exec",
        create_process,
    )

    await asyncio.gather(
        asyncio.to_thread(
            git_ops._run_gh,
            tmp_path,
            ["api", "/user"],
            auth=auth,
        ),
        git_ops._run_gh_async(
            tmp_path,
            ["api", "/user"],
            auth=auth,
            budget=_budget(),
        ),
    )

    assert refresh_calls == 1
    assert len(captured) == 2
    assert all(
        environment is not None
        and environment["GH_TOKEN"] == "ghs_shared_fresh_token_1234567890"
        for environment in captured
    )


@pytest.mark.asyncio
async def test_expired_budget_never_resolves_auth_or_spawns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    auth_calls = 0
    spawn_calls = 0

    class Auth:
        def environment_for_request(self) -> None:
            nonlocal auth_calls
            auth_calls += 1

    async def create_process(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("expired budget must not spawn gh")

    monkeypatch.setattr(
        "daydream.git_ops.asyncio.create_subprocess_exec",
        create_process,
    )
    budget = git_ops.GitHubRequestBudget(
        deadline=0.0,
        per_request_seconds=1.0,
        monotonic=lambda: 1.0,
    )

    with pytest.raises(git_ops.DeadlineExpired):
        await git_ops._run_gh_async(
            tmp_path,
            ["api", "/user"],
            auth=Auth(),
            budget=budget,
        )

    assert auth_calls == 0
    assert spawn_calls == 0


@pytest.mark.asyncio
async def test_blocked_auth_resolution_keeps_loop_responsive_and_is_cancellable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    spawn_calls = 0

    class BlockingAuth:
        def environment_for_request(self) -> None:
            loop.call_soon_threadsafe(started.set)
            try:
                release.wait(timeout=2)
            finally:
                finished.set()

    async def create_process(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("cancelled auth resolution must not spawn gh")

    monkeypatch.setattr(
        "daydream.git_ops.asyncio.create_subprocess_exec",
        create_process,
    )
    task = asyncio.create_task(
        git_ops._run_gh_async(
            tmp_path,
            ["api", "/user"],
            auth=BlockingAuth(),
            budget=_budget(seconds=2, per_request=1),
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
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 1)

    assert spawn_calls == 0


@pytest.mark.asyncio
async def test_auth_resolution_timeout_never_spawns_and_redacts_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    spawn_calls = 0

    class BlockingAuth:
        def environment_for_request(self) -> None:
            loop.call_soon_threadsafe(started.set)
            try:
                release.wait(timeout=2)
            finally:
                finished.set()

    async def create_process(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("timed-out auth resolution must not spawn gh")

    monkeypatch.setattr(
        "daydream.git_ops.asyncio.create_subprocess_exec",
        create_process,
    )
    task = asyncio.create_task(
        git_ops._run_gh_async(
            tmp_path,
            ["api", "Authorization: Bearer secret-value"],
            auth=BlockingAuth(),
            budget=_budget(seconds=0.5, per_request=0.5),
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        with pytest.raises(git_ops.GitTimeoutError) as excinfo:
            await task
        assert "secret-value" not in str(excinfo.value)
        assert excinfo.value.__cause__ is None
    finally:
        pending = not task.done()
        if pending:
            task.cancel()
        release.set()
        assert await asyncio.to_thread(finished.wait, 1)
        if pending:
            with pytest.raises(asyncio.CancelledError):
                await task

    assert started.is_set()
    assert spawn_calls == 0


@pytest.mark.asyncio
async def test_auth_resolution_that_consumes_budget_never_spawns_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    times = iter((0.0, 2.0))
    spawn_calls = 0

    async def create_process(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("exhausted budget must not spawn gh")

    monkeypatch.setattr(
        "daydream.git_ops.asyncio.create_subprocess_exec",
        create_process,
    )
    budget = git_ops.GitHubRequestBudget(
        deadline=1.0,
        per_request_seconds=1.0,
        monotonic=lambda: next(times),
    )

    with pytest.raises(git_ops.DeadlineExpired):
        await git_ops._run_gh_async(
            tmp_path,
            ["api", "/user"],
            auth=git_ops.INHERIT_GITHUB_AUTH,
            budget=budget,
        )

    assert spawn_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("per_page", "max_pages"),
    [(0, 10), (-1, 10), (100, 0), (100, -1)],
)
async def test_nonpositive_page_limits_fail_before_spawning_gh(
    fake_gh: FakeGh,
    git_repo: Path,
    per_page: int,
    max_pages: int,
) -> None:
    with pytest.raises(git_ops.GitError, match="pagination limits must be positive"):
        await git_ops.gh_api_bounded_pages(
            git_repo,
            "repos/acme/widgets/statuses",
            envelope=None,
            limits=git_ops.GitHubPageLimits(per_page=per_page, max_pages=max_pages),
            budget=_budget(),
        )

    assert fake_gh.process_calls() == []


@pytest.mark.asyncio
async def test_bounded_pages_collect_list_with_headers_and_cwd(
    fake_gh: FakeGh,
    git_repo: Path,
) -> None:
    sha = "a" * 40
    endpoint = f"repos/acme/widgets/commits/{sha}/statuses"
    fake_gh.set_response("GET", _page_endpoint(endpoint, 1), [{"id": n} for n in range(100)])
    fake_gh.set_response("GET", _page_endpoint(endpoint, 2), [{"id": 100}])

    result = await git_ops.gh_commit_statuses(
        git_repo,
        "acme",
        "widgets",
        sha,
        limits=git_ops.GitHubPageLimits(),
        budget=_budget(),
    )

    assert [row["id"] for row in result] == list(range(101))
    calls = fake_gh.process_calls()
    assert [call.argv[-1] for call in calls] == [
        _page_endpoint(endpoint, 1),
        _page_endpoint(endpoint, 2),
    ]
    for call in calls:
        assert call.cwd == git_repo.resolve()
        assert "--paginate" not in call.argv
        assert call.argv[2:4] == ["-H", "Accept: application/vnd.github+json"]
        assert call.argv[4:6] == ["-H", "X-GitHub-Api-Version: 2022-11-28"]
        assert call.argv[6:8] == ["--method", "GET"]

    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=git_repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wrapper", "endpoint", "envelope"),
    [
        (
            git_ops.gh_commit_check_runs,
            "repos/acme/widgets/commits/{sha}/check-runs?filter=latest",
            "check_runs",
        ),
        (
            git_ops.gh_actions_workflows,
            "repos/acme/widgets/actions/workflows",
            "workflows",
        ),
    ],
)
async def test_enveloped_pagination(
    fake_gh: FakeGh,
    git_repo: Path,
    wrapper: Callable[..., Any],
    endpoint: str,
    envelope: str,
) -> None:
    sha = "b" * 40
    endpoint = endpoint.format(sha=sha)
    fake_gh.set_response(
        "GET",
        _page_endpoint(endpoint, 1),
        {"total_count": 1, envelope: [{"id": 7}]},
    )

    if wrapper is git_ops.gh_commit_check_runs:
        result = await wrapper(
            git_repo,
            "acme",
            "widgets",
            sha,
            limits=git_ops.GitHubPageLimits(),
            budget=_budget(),
        )
    else:
        result = await wrapper(
            git_repo,
            "acme",
            "widgets",
            limits=git_ops.GitHubPageLimits(),
            budget=_budget(),
        )

    assert result == [{"id": 7}]
    assert fake_gh.process_calls()[-1].argv[-1] == _page_endpoint(endpoint, 1)


@pytest.mark.asyncio
async def test_enveloped_full_page_stops_at_declared_total(
    fake_gh: FakeGh,
    git_repo: Path,
) -> None:
    endpoint = "repos/acme/widgets/actions/workflows"
    rows = [{"id": n} for n in range(100)]
    fake_gh.set_response(
        "GET",
        _page_endpoint(endpoint, 1),
        {"total_count": 100, "workflows": rows},
    )

    result = await git_ops.gh_actions_workflows(
        git_repo,
        "acme",
        "widgets",
        limits=git_ops.GitHubPageLimits(),
        budget=_budget(),
    )

    assert result == rows
    assert len(fake_gh.process_calls()) == 1


@pytest.mark.asyncio
async def test_branch_is_encoded_and_pr_snapshot_is_exact(
    fake_gh: FakeGh,
    git_repo: Path,
) -> None:
    branch_endpoint = "repos/acme/widgets/rules/branches/feature%2Fspace%20ship"
    fake_gh.set_response("GET", _page_endpoint(branch_endpoint, 1), [{"id": 3}])
    fake_gh.set_response("GET", "repos/acme/widgets/pulls/7", {"number": 7})

    rules = await git_ops.gh_active_branch_rules(
        git_repo,
        "acme",
        "widgets",
        "feature/space ship",
        limits=git_ops.GitHubPageLimits(),
        budget=_budget(),
    )
    snapshot = await git_ops.gh_pr_ci_snapshot(
        git_repo,
        "acme",
        "widgets",
        7,
        budget=_budget(),
    )

    assert rules == [{"id": 3}]
    assert snapshot == {"number": 7}


@pytest.mark.asyncio
async def test_response_sequence_advances_for_same_endpoint(
    fake_gh: FakeGh,
    git_repo: Path,
) -> None:
    key = "GET repos/acme/widgets/pulls/7"
    fake_gh.set_response_sequence(key, [{"number": 7}, {"number": 8}])

    first = await git_ops.gh_pr_ci_snapshot(
        git_repo, "acme", "widgets", 7, budget=_budget()
    )
    second = await git_ops.gh_pr_ci_snapshot(
        git_repo, "acme", "widgets", 7, budget=_budget()
    )

    assert first == {"number": 7}
    assert second == {"number": 8}


@pytest.mark.asyncio
async def test_response_sequence_can_serve_external_error(
    fake_gh: FakeGh,
    git_repo: Path,
) -> None:
    key = "GET repos/acme/widgets/pulls/7"
    fake_gh.set_response_sequence(
        key,
        [{"__error__": "gh: hidden resource (HTTP 404)"}, {"number": 7}],
    )

    with pytest.raises(git_ops.GitError, match="hidden resource"):
        await git_ops.gh_pr_ci_snapshot(
            git_repo, "acme", "widgets", 7, budget=_budget()
        )
    assert await git_ops.gh_pr_ci_snapshot(
        git_repo, "acme", "widgets", 7, budget=_budget()
    ) == {"number": 7}


@pytest.mark.asyncio
async def test_deadline_recomputed_before_each_spawn_and_partial_discarded(
    fake_gh: FakeGh,
    git_repo: Path,
) -> None:
    endpoint = "repos/acme/widgets/commits/" + "c" * 40 + "/statuses"
    fake_gh.set_response("GET", _page_endpoint(endpoint, 1), [{"id": n} for n in range(100)])
    fake_gh.set_response("GET", _page_endpoint(endpoint, 2), [{"id": 100}])
    # Each request checks the shared deadline before and after auth resolution.
    # The first page gets both reads; the second expires at its pre-auth check.
    readings = iter((0.0, 0.0, 11.0))
    budget = git_ops.GitHubRequestBudget(
        deadline=10.0,
        per_request_seconds=5.0,
        monotonic=lambda: next(readings),
    )

    with pytest.raises(git_ops.DeadlineExpired, match="deadline"):
        await git_ops.gh_api_bounded_pages(
            git_repo,
            endpoint,
            envelope=None,
            limits=git_ops.GitHubPageLimits(),
            budget=budget,
        )

    assert [call.argv[-1] for call in fake_gh.process_calls()] == [
        _page_endpoint(endpoint, 1)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        [],
        {"check_runs": "wrong"},
        {"total_count": True, "check_runs": []},
    ],
)
async def test_malformed_envelope_is_rejected(
    fake_gh: FakeGh,
    git_repo: Path,
    response: Any,
) -> None:
    endpoint = "repos/acme/widgets/commits/" + "d" * 40 + "/check-runs?filter=latest"
    fake_gh.set_response("GET", _page_endpoint(endpoint, 1), response)

    with pytest.raises(git_ops.GitError, match="shape"):
        await git_ops.gh_api_bounded_pages(
            git_repo,
            endpoint,
            envelope="check_runs",
            limits=git_ops.GitHubPageLimits(),
            budget=_budget(),
        )


@pytest.mark.asyncio
async def test_enveloped_pagination_rejects_incomplete_short_page(
    fake_gh: FakeGh,
    git_repo: Path,
) -> None:
    endpoint = "repos/acme/widgets/actions/workflows"
    fake_gh.set_response(
        "GET",
        _page_endpoint(endpoint, 1),
        {"total_count": 2, "workflows": [{"id": 1}]},
    )

    with pytest.raises(git_ops.GitError, match="incomplete"):
        await git_ops.gh_actions_workflows(
            git_repo,
            "acme",
            "widgets",
            limits=git_ops.GitHubPageLimits(),
            budget=_budget(),
        )

    assert len(fake_gh.process_calls()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("max_pages", [1, 2])
async def test_enveloped_exact_capacity_returns_declared_total(
    fake_gh: FakeGh,
    git_repo: Path,
    max_pages: int,
) -> None:
    endpoint = "repos/acme/widgets/actions/workflows"
    expected = [{"id": value} for value in range(1, max_pages * 2 + 1)]
    for page in range(1, max_pages + 1):
        start = (page - 1) * 2
        fake_gh.set_response(
            "GET",
            _page_endpoint(endpoint, page, per_page=2),
            {"total_count": len(expected), "workflows": expected[start : start + 2]},
        )

    result = await git_ops.gh_actions_workflows(
        git_repo,
        "acme",
        "widgets",
        limits=git_ops.GitHubPageLimits(per_page=2, max_pages=max_pages),
        budget=_budget(),
    )

    assert result == expected
    assert [call.argv[-1] for call in fake_gh.process_calls()] == [
        _page_endpoint(endpoint, page, per_page=2)
        for page in range(1, max_pages + 1)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("max_pages", [1, 2])
async def test_envelope_less_full_capacity_remains_ambiguous(
    fake_gh: FakeGh,
    git_repo: Path,
    max_pages: int,
) -> None:
    sha = "f" * 40
    endpoint = f"repos/acme/widgets/commits/{sha}/statuses"
    for page in range(1, max_pages + 1):
        fake_gh.set_response(
            "GET",
            _page_endpoint(endpoint, page, per_page=2),
            [{"id": page * 2 - 1}, {"id": page * 2}],
        )

    with pytest.raises(git_ops.GitError, match="pagination limit"):
        await git_ops.gh_commit_statuses(
            git_repo,
            "acme",
            "widgets",
            sha,
            limits=git_ops.GitHubPageLimits(per_page=2, max_pages=max_pages),
            budget=_budget(),
        )

    assert len(fake_gh.process_calls()) == max_pages


@pytest.mark.asyncio
async def test_total_count_over_capacity_is_rejected_before_second_call(
    fake_gh: FakeGh,
    git_repo: Path,
) -> None:
    endpoint = "repos/acme/widgets/actions/workflows"
    fake_gh.set_response(
        "GET",
        _page_endpoint(endpoint, 1),
        {"total_count": 1001, "workflows": []},
    )

    with pytest.raises(git_ops.GitError, match="pagination limit"):
        await git_ops.gh_actions_workflows(
            git_repo,
            "acme",
            "widgets",
            limits=git_ops.GitHubPageLimits(),
            budget=_budget(),
        )

    assert len(fake_gh.process_calls()) == 1


@pytest.mark.asyncio
async def test_full_tenth_page_is_rejected_at_hard_limit(
    fake_gh: FakeGh,
    git_repo: Path,
) -> None:
    sha = "e" * 40
    endpoint = f"repos/acme/widgets/commits/{sha}/statuses"
    for page in range(1, 11):
        fake_gh.set_response(
            "GET",
            _page_endpoint(endpoint, page),
            [{"id": page * 100 + n} for n in range(100)],
        )

    with pytest.raises(git_ops.GitError, match="pagination limit"):
        await git_ops.gh_commit_statuses(
            git_repo,
            "acme",
            "widgets",
            sha,
            limits=git_ops.GitHubPageLimits(),
            budget=_budget(),
        )

    assert len(fake_gh.process_calls()) == 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected", "error"),
    [
        ({"__error__": "gh: Branch not protected (HTTP 404)"}, None, None),
        ({"__error__": "gh: Not Found (HTTP 404)"}, None, git_ops.GitError),
        (
            {"__error__": "API rate limit exceeded (HTTP 403)"},
            None,
            git_ops.RateLimitError,
        ),
        ([], None, git_ops.GitError),
        ({"contexts": ["ci"]}, {"contexts": ["ci"]}, None),
    ],
)
async def test_classic_required_checks_narrow_absence(
    fake_gh: FakeGh,
    git_repo: Path,
    response: Any,
    expected: Any,
    error: type[BaseException] | None,
) -> None:
    endpoint = "repos/acme/widgets/branches/main/protection/required_status_checks"
    fake_gh.set_response("GET", endpoint, response)

    if error is not None:
        with pytest.raises(error):
            await git_ops.gh_classic_required_checks(
                git_repo, "acme", "widgets", "main", budget=_budget()
            )
    else:
        assert (
            await git_ops.gh_classic_required_checks(
                git_repo, "acme", "widgets", "main", budget=_budget()
            )
            == expected
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [{"__stdout__": "not-json"}, []])
async def test_pr_snapshot_rejects_invalid_json_or_shape(
    fake_gh: FakeGh,
    git_repo: Path,
    response: Any,
) -> None:
    fake_gh.set_response("GET", "repos/acme/widgets/pulls/7", response)

    with pytest.raises(git_ops.GitError):
        await git_ops.gh_pr_ci_snapshot(
            git_repo, "acme", "widgets", 7, budget=_budget()
        )


def _fd_count() -> int | None:
    fd_dir = Path("/dev/fd")
    if not fd_dir.is_dir():
        return None
    return len(list(fd_dir.iterdir()))


async def _wait_for_json(path: Path, *, timeout: float = 5.0) -> dict[str, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = json.loads(path.read_text())
            if (
                isinstance(value, dict)
                and isinstance(value.get("direct"), int)
                and isinstance(value.get("grandchild"), int)
            ):
                return {
                    "direct": value["direct"],
                    "grandchild": value["grandchild"],
                }
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        await asyncio.sleep(0.01)
    raise AssertionError(f"blocking fake gh did not publish {path}")


async def _wait_process_group_gone(pgid: int, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            # EPERM means the pgid was recycled by a foreign-uid process,
            # i.e. our same-uid group exited.
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"process group {pgid} survived cleanup")


async def _wait_for_fd_baseline(baseline: int, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _fd_count() == baseline:
            return
        await asyncio.sleep(0.01)
    assert _fd_count() == baseline


async def _exercise_blocking_request(
    fake_gh: FakeGh,
    git_repo: Path,
    tmp_path: Path,
    *,
    cancel: bool,
) -> dict[str, int]:
    endpoint = "repos/acme/widgets/commits/" + "f" * 40 + "/statuses"
    key = "GET " + _page_endpoint(endpoint, 1)
    pid_file = tmp_path / ("cancel-pids.json" if cancel else "timeout-pids.json")
    fake_gh.serve_blocking_process(key, pid_file=pid_file)
    fake_gh.set_response("GET", _page_endpoint(endpoint, 2), [{"id": 2}])
    budget = _budget(seconds=5.0, per_request=2.0)
    request = asyncio.create_task(
        git_ops.gh_api_bounded_pages(
            git_repo,
            endpoint,
            envelope=None,
            limits=git_ops.GitHubPageLimits(),
            budget=budget,
        )
    )
    try:
        pids = await _wait_for_json(pid_file)
    except BaseException:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        raise

    if cancel:
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
    else:
        with pytest.raises(git_ops.GitTimeoutError, match="timed out"):
            await request

    await _wait_process_group_gone(pids["direct"])
    return pids


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_blocking_process_group_is_killed_reaped_and_fds_closed(
    fake_gh: FakeGh,
    git_repo: Path,
    tmp_path: Path,
    cancel: bool,
) -> None:
    baseline = _fd_count()

    pids = await _exercise_blocking_request(
        fake_gh, git_repo, tmp_path, cancel=cancel
    )

    assert pids["direct"] != pids["grandchild"]
    if baseline is not None:
        await _wait_for_fd_baseline(baseline)
    assert len(fake_gh.process_calls()) == 1


@pytest.mark.asyncio
async def test_process_cleanup_is_idempotent(
    fake_gh: FakeGh,
    git_repo: Path,
    tmp_path: Path,
) -> None:
    endpoint = "repos/acme/widgets/pulls/7"
    pid_file = tmp_path / "idempotent-pids.json"
    fake_gh.serve_blocking_process("GET " + endpoint, pid_file=pid_file)
    proc = await asyncio.create_subprocess_exec(
        "gh",
        "api",
        "--method",
        "GET",
        endpoint,
        cwd=git_repo,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        pids = await _wait_for_json(pid_file)
        await terminate_process(proc)
        await terminate_process(proc)
    finally:
        await terminate_process(proc)

    assert proc.returncode is not None
    await _wait_process_group_gone(pids["direct"])
