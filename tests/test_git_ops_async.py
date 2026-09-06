"""Real-process tests for bounded asynchronous GitHub API reads."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
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
    readings = iter((0.0, 11.0))
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
@pytest.mark.parametrize(
    ("total_count", "rows", "per_page", "max_pages"),
    [
        (2, [{"id": 1}], 100, 10),
        (2, [{"id": 1}, {"id": 2}], 2, 1),
    ],
)
async def test_enveloped_pagination_rejects_incomplete_or_full_limit_page(
    fake_gh: FakeGh,
    git_repo: Path,
    total_count: int,
    rows: list[dict[str, int]],
    per_page: int,
    max_pages: int,
) -> None:
    endpoint = "repos/acme/widgets/actions/workflows"
    fake_gh.set_response(
        "GET",
        _page_endpoint(endpoint, 1, per_page=per_page),
        {"total_count": total_count, "workflows": rows},
    )

    with pytest.raises(git_ops.GitError, match="pagination|incomplete"):
        await git_ops.gh_actions_workflows(
            git_repo,
            "acme",
            "widgets",
            limits=git_ops.GitHubPageLimits(per_page=per_page, max_pages=max_pages),
            budget=_budget(),
        )

    assert len(fake_gh.process_calls()) == 1


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
        except ProcessLookupError:
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
