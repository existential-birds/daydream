from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream.git_ops import GitError, GitHubRequestBudget, gh_pr_ci_snapshot
from daydream.remote_ci import (
    CIObservation,
    GitHubRemoteCIFetcher,
    PRCIBinding,
    RemoteCILimits,
    RemoteCISnapshot,
    RemoteCITarget,
    RemoteCIVerdict,
    RequiredContext,
    RequiredPolicy,
    evaluate_remote_ci,
    local_host_facts,
    parse_active_workflows,
    parse_observations,
    parse_pr_ci_binding,
    parse_required_policy,
    pending_remote_ci_verdict,
    remote_ci_handoff_payload,
    remote_ci_verdict_payload,
    unavailable_remote_ci_verdict,
    wait_for_remote_ci,
    write_remote_ci_verdict,
)
from tests.harness.fake_gh import FakeGh

PUSHED_SHA = "1" * 40
MERGE_SHA = "2" * 40


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_limits_reject_nonfinite_time_values(value: float) -> None:
    with pytest.raises(ValueError):
        RemoteCILimits(request_seconds=value)


def _target(tmp_path: Path) -> RemoteCITarget:
    return RemoteCITarget(
        target_dir=tmp_path,
        base_repository="Example/Project",
        base_ref="main",
        head_repository="Contributor/Fork",
        head_ref="feature/ci",
        pr_number=42,
        pr_url="https://github.com/example/project/pull/42",
        remote="fork",
        pushed_sha=PUSHED_SHA,
    )


def _binding(target: RemoteCITarget, **overrides: object) -> PRCIBinding:
    values: dict[str, object] = {
        "pr_number": target.pr_number,
        "pr_url": target.pr_url,
        "base_repository": target.base_repository,
        "base_ref": target.base_ref,
        "head_repository": target.head_repository,
        "head_ref": target.head_ref,
        "head_sha": target.pushed_sha,
        "merge_sha": None,
        "state": "open",
    }
    values.update(overrides)
    return PRCIBinding(**values)  # type: ignore[arg-type]


def _pr_row(target: RemoteCITarget, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "number": target.pr_number,
        "html_url": target.pr_url,
        "state": "open",
        "base": {
            "ref": target.base_ref,
            "repo": {"full_name": "EXAMPLE/PROJECT"},
        },
        "head": {
            "ref": target.head_ref,
            "sha": target.pushed_sha,
            "repo": {"full_name": "CONTRIBUTOR/FORK"},
        },
        "merge_commit_sha": MERGE_SHA,
    }
    row.update(overrides)
    return row


def _check(
    row_id: int,
    name: str,
    *,
    sha: str = PUSHED_SHA,
    app_id: int = 10,
    status: str = "completed",
    conclusion: str | None = "success",
    url: str | None = "https://github.com/example/project/actions/runs/7",
    summary: str | None = None,
) -> dict[str, object]:
    return {
        "id": row_id,
        "name": name,
        "head_sha": sha,
        "app": {"id": app_id},
        "status": status,
        "conclusion": conclusion,
        "details_url": url,
        "output": {"title": None, "summary": summary, "text": "must not persist"},
    }


def _status(
    row_id: int,
    context: str,
    *,
    sha: str = PUSHED_SHA,
    state: str = "success",
    url: str | None = "https://ci.example.test/run/7",
    description: str | None = None,
) -> dict[str, object]:
    return {
        "id": row_id,
        "context": context,
        "sha": sha,
        "state": state,
        "target_url": url,
        "description": description,
    }


def _observation(
    context: str,
    state: str,
    *,
    source: str = "check_run",
    app_id: int | None = 10,
) -> CIObservation:
    return CIObservation(
        source=source,  # type: ignore[arg-type]
        context=context,
        app_id=app_id,
        state=state,  # type: ignore[arg-type]
        raw_state=state,
        url=None,
        diagnostic=None,
    )


def _snapshot(
    target: RemoteCITarget,
    *,
    policy: RequiredPolicy = RequiredPolicy((), False),
    workflows: tuple[dict[str, str | int], ...] = (),
    head: tuple[CIObservation, ...] = (),
    merge: tuple[CIObservation, ...] = (),
    binding: PRCIBinding | None = None,
) -> RemoteCISnapshot:
    return RemoteCISnapshot(
        target=target,
        binding=binding or _binding(target),
        policy=policy,
        active_workflows=workflows,
        head_observations=head,
        merge_observations=merge,
    )


def test_parse_pr_ci_binding_preserves_full_fixed_identity(tmp_path: Path) -> None:
    target = _target(tmp_path)

    binding = parse_pr_ci_binding(_pr_row(target), target)

    assert binding == PRCIBinding(
        pr_number=42,
        pr_url="https://github.com/example/project/pull/42",
        base_repository="example/project",
        base_ref="main",
        head_repository="contributor/fork",
        head_ref="feature/ci",
        head_sha=PUSHED_SHA,
        merge_sha=MERGE_SHA,
        state="open",
    )


@pytest.mark.parametrize(
    "replacement",
    [
        {"number": 43},
        {"number": True},
        {"html_url": "https://github.com/example/project/pull/43"},
        {"state": "merged"},
        {"merge_commit_sha": "ABC"},
        {"head": {"ref": "feature/ci", "sha": PUSHED_SHA, "repo": None}},
        {"head": {"ref": "other", "sha": PUSHED_SHA, "repo": {"full_name": "contributor/fork"}}},
        {"base": {"ref": "other", "repo": {"full_name": "example/project"}}},
    ],
)
def test_parse_pr_ci_binding_rejects_identity_or_schema_mismatch(
    tmp_path: Path, replacement: dict[str, object]
) -> None:
    target = _target(tmp_path)

    with pytest.raises(ValueError):
        parse_pr_ci_binding(_pr_row(target, **replacement), target)


def test_policy_unions_sources_and_preserves_exact_names_and_app_pins() -> None:
    active = [
        {
            "type": "required_status_checks",
            "parameters": {
                "strict_required_status_checks_policy": True,
                "required_status_checks": [
                    {"context": "Build", "integration_id": None},
                    {"context": "Build", "integration_id": 7},
                    {"context": "build", "integration_id": None},
                    {"context": "Deploy", "integration_id": 8},
                    {"context": "Deploy", "integration_id": 9},
                ],
            },
        }
    ]
    classic = {
        "strict": False,
        "contexts": ["Classic"],
        "checks": [{"context": "Build", "app_id": 7}],
    }

    policy = parse_required_policy(active, classic)

    assert policy.strict is True
    assert policy.contexts == (
        RequiredContext("Build", 7),
        RequiredContext("Classic", None),
        RequiredContext("Deploy", 8),
        RequiredContext("Deploy", 9),
        RequiredContext("build", None),
    )


@pytest.mark.parametrize(
    ("active", "classic"),
    [
        ({}, None),
        ([{"type": "required_status_checks", "parameters": None}], None),
        (
            [
                {
                    "type": "required_status_checks",
                    "parameters": {
                        "strict_required_status_checks_policy": 1,
                        "required_status_checks": [],
                    },
                }
            ],
            None,
        ),
        ([], {"strict": False, "contexts": [""], "checks": []}),
        ([], {"strict": False, "contexts": [], "checks": [{"context": "x", "app_id": 0}]}),
    ],
)
def test_policy_rejects_unknown_or_malformed_shapes(active: object, classic: object) -> None:
    with pytest.raises(ValueError):
        parse_required_policy(active, classic)


def test_observations_reduce_latest_producers_with_source_specific_case_rules() -> None:
    observations = parse_observations(
        [
            _check(1, "Build", conclusion="failure"),
            _check(2, "Build", conclusion="success"),
            _check(3, "build", conclusion="failure", app_id=11),
        ],
        [
            _status(11, "lInT", state="success"),
            _status(10, "Lint", state="failure"),
        ],
        expected_sha=PUSHED_SHA,
        limit=2_000,
    )

    assert [(item.source, item.context, item.app_id, item.state) for item in observations] == [
        ("check_run", "Build", 10, "pass"),
        ("check_run", "build", 11, "fail"),
        ("status", "lInT", None, "pass"),
    ]


@pytest.mark.parametrize("conclusion", ["success", "neutral", "skipped"])
def test_observation_accepts_documented_check_success(conclusion: str) -> None:
    result = parse_observations(
        [_check(1, "Build", conclusion=conclusion)], [], expected_sha=PUSHED_SHA, limit=2_000
    )
    assert result[0].state == "pass"


@pytest.mark.parametrize(
    "conclusion",
    ["failure", "cancelled", "timed_out", "action_required", "stale", "startup_failure"],
)
def test_observation_accepts_documented_check_failures(conclusion: str) -> None:
    result = parse_observations(
        [_check(1, "Build", conclusion=conclusion)], [], expected_sha=PUSHED_SHA, limit=2_000
    )
    assert result[0].state == "fail"


@pytest.mark.parametrize("status", ["queued", "in_progress", "requested", "waiting", "pending"])
def test_observation_accepts_documented_pending_checks(status: str) -> None:
    result = parse_observations(
        [_check(1, "Build", status=status, conclusion=None)],
        [],
        expected_sha=PUSHED_SHA,
        limit=2_000,
    )
    assert result[0].state == "pending"


def test_observation_legacy_error_is_failure_but_check_error_is_unsupported() -> None:
    result = parse_observations(
        [], [_status(1, "Build", state="error")], expected_sha=PUSHED_SHA, limit=2_000
    )
    assert result[0].state == "fail"

    with pytest.raises(ValueError):
        parse_observations(
            [_check(1, "Build", conclusion="error")],
            [],
            expected_sha=PUSHED_SHA,
            limit=2_000,
        )


@pytest.mark.parametrize(
    ("checks", "statuses"),
    [
        ([_check(0, "Build")], []),
        ([_check(1, "Build"), _check(1, "Other")], []),
        ([_check(1, "Build", sha="3" * 40)], []),
        ([_check(1, "Build", app_id=True)], []),
        ([_check(1, "Build", url="ftp://example.test/run")], []),
        ([_check(1, "Build", status="completed", conclusion=None)], []),
        ([], [_status(1, "Build", state="unknown")]),
        ([], [_status(1, "Build", sha="3" * 40)]),
    ],
)
def test_observations_reject_stale_or_malformed_rows(
    checks: object, statuses: object
) -> None:
    with pytest.raises(ValueError):
        parse_observations(checks, statuses, expected_sha=PUSHED_SHA, limit=2_000)


def test_observation_sanitizes_url_and_redacts_bounded_diagnostic() -> None:
    secret = "sk-12345678901234567890"
    result = parse_observations(
        [
            _check(
                1,
                "Build",
                url=(
                    f"https://ci.example.test/run?token={secret}"
                    f"&X-Amz-Signature={secret}&opaque={secret}&page=1"
                ),
                summary=f"Authorization: Bearer {secret} " + ("x" * 500),
            )
        ],
        [],
        expected_sha=PUSHED_SHA,
        limit=120,
    )[0]

    assert secret not in (result.url or "")
    assert secret not in (result.diagnostic or "")
    assert "REDACTED" in (result.url or "")
    assert len(result.diagnostic or "") <= 120


def test_active_workflow_parser_returns_only_strictly_valid_active_rows() -> None:
    rows = [
        {"id": 2, "name": "CI", "path": ".github/workflows/ci.yml", "state": "active"},
        {
            "id": 1,
            "name": "Old",
            "path": ".github/workflows/old.yml",
            "state": "disabled_manually",
        },
    ]

    assert parse_active_workflows(rows) == (
        {"id": 2, "name": "CI", "path": ".github/workflows/ci.yml", "state": "active"},
    )


@pytest.mark.parametrize(
    "rows",
    [
        {},
        [{"id": True, "name": "CI", "path": "ci.yml", "state": "active"}],
        [{"id": 1, "name": "CI", "path": "ci.yml", "state": "mystery"}],
        [
            {"id": 1, "name": "CI", "path": "ci.yml", "state": "active"},
            {"id": 1, "name": "Other", "path": "other.yml", "state": "active"},
        ],
    ],
)
def test_active_workflow_parser_rejects_unknown_or_malformed_rows(rows: object) -> None:
    with pytest.raises(ValueError):
        parse_active_workflows(rows)


def test_evaluate_pinned_context_requires_exact_check_name_and_app(tmp_path: Path) -> None:
    target = _target(tmp_path)
    policy = RequiredPolicy((RequiredContext("Build", 7),), True)
    snapshot = _snapshot(
        target,
        policy=policy,
        head=(
            _observation("Build", "pass", app_id=8),
            _observation("build", "pass", app_id=7),
            _observation("Build", "pass", source="status", app_id=None),
        ),
    )

    verdict = evaluate_remote_ci(snapshot, elapsed=120, stable_polls=2, limits=RemoteCILimits())

    assert verdict.status == "missing"
    assert verdict.missing_contexts == ("Build (app 7)",)


def test_evaluate_unpinned_context_requires_all_latest_matching_producers(tmp_path: Path) -> None:
    target = _target(tmp_path)
    snapshot = _snapshot(
        target,
        policy=RequiredPolicy((RequiredContext("Build", None),), False),
        head=(
            _observation("Build", "pass", app_id=7),
            _observation("bUiLd", "fail", source="status", app_id=None),
        ),
    )

    verdict = evaluate_remote_ci(snapshot, elapsed=1, stable_polls=1, limits=RemoteCILimits())

    assert verdict.status == "failed"
    assert verdict.failing_contexts == ("Build",)


def test_evaluate_required_pending_and_deadlines(tmp_path: Path) -> None:
    target = _target(tmp_path)
    snapshot = _snapshot(
        target,
        policy=RequiredPolicy((RequiredContext("Build", None),), False),
        head=(_observation("Build", "pending"),),
    )

    assert evaluate_remote_ci(
        snapshot, elapsed=1_799.999, stable_polls=1, limits=RemoteCILimits()
    ).status == "pending"
    assert evaluate_remote_ci(
        snapshot, elapsed=1_800, stable_polls=1, limits=RemoteCILimits()
    ).status == "timed_out"


def test_evaluate_missing_required_or_active_workflow_at_discovery(tmp_path: Path) -> None:
    target = _target(tmp_path)
    required = _snapshot(
        target,
        policy=RequiredPolicy((RequiredContext("Build", None),), False),
    )
    workflow = _snapshot(
        target,
        workflows=({"id": 1, "name": "CI", "path": "ci.yml", "state": "active"},),
    )

    assert evaluate_remote_ci(required, elapsed=119.999, stable_polls=2, limits=RemoteCILimits()).status == "pending"
    assert evaluate_remote_ci(required, elapsed=120, stable_polls=2, limits=RemoteCILimits()).status == "missing"
    assert evaluate_remote_ci(workflow, elapsed=120, stable_polls=2, limits=RemoteCILimits()).status == "missing"


def test_evaluate_no_ci_needs_discovery_deadline_and_two_stable_polls(tmp_path: Path) -> None:
    snapshot = _snapshot(_target(tmp_path))

    assert evaluate_remote_ci(snapshot, elapsed=119.999, stable_polls=2, limits=RemoteCILimits()).status == "pending"
    assert evaluate_remote_ci(snapshot, elapsed=120, stable_polls=1, limits=RemoteCILimits()).status == "pending"
    assert evaluate_remote_ci(snapshot, elapsed=120, stable_polls=2, limits=RemoteCILimits()).status == "no_ci"


def test_evaluate_advisory_failure_and_pending_do_not_block_pass(tmp_path: Path) -> None:
    target = _target(tmp_path)
    snapshot = _snapshot(
        target,
        policy=RequiredPolicy((RequiredContext("Build", 7),), True),
        head=(
            _observation("Build", "pass", app_id=7),
            _observation("Advisory failure", "fail", app_id=8),
            _observation("Advisory pending", "pending", app_id=9),
        ),
    )

    verdict = evaluate_remote_ci(snapshot, elapsed=20, stable_polls=2, limits=RemoteCILimits())

    assert verdict.status == "passed"
    assert [(item.context, item.state) for item in verdict.advisory_observations] == [
        ("Advisory failure", "fail"),
        ("Advisory pending", "pending"),
    ]


def test_evaluate_policyless_observations_need_all_terminal(tmp_path: Path) -> None:
    target = _target(tmp_path)
    pending = _snapshot(target, head=(_observation("Advisory", "pending"),))
    terminal = _snapshot(
        target,
        head=(
            _observation("Advisory green", "pass"),
            _observation("Advisory red", "fail"),
        ),
    )

    assert evaluate_remote_ci(pending, elapsed=100, stable_polls=2, limits=RemoteCILimits()).status == "pending"
    assert evaluate_remote_ci(pending, elapsed=1_800, stable_polls=2, limits=RemoteCILimits()).status == "timed_out"
    assert evaluate_remote_ci(terminal, elapsed=10, stable_polls=2, limits=RemoteCILimits()).status == "passed"


def test_evaluate_selects_merge_globally_and_empty_merge_falls_back_to_head(tmp_path: Path) -> None:
    target = _target(tmp_path)
    policy = RequiredPolicy((RequiredContext("Build", 7),), False)
    binding = _binding(target, merge_sha=MERGE_SHA)
    green_head = (_observation("Build", "pass", app_id=7),)
    red_merge = (_observation("Build", "fail", app_id=7),)

    merged = evaluate_remote_ci(
        _snapshot(target, policy=policy, head=green_head, merge=red_merge, binding=binding),
        elapsed=10,
        stable_polls=2,
        limits=RemoteCILimits(),
    )
    fallback = evaluate_remote_ci(
        _snapshot(target, policy=policy, head=green_head, binding=binding),
        elapsed=10,
        stable_polls=2,
        limits=RemoteCILimits(),
    )

    assert (merged.status, merged.evidence_sha) == ("failed", MERGE_SHA)
    assert (fallback.status, fallback.evidence_sha) == ("passed", PUSHED_SHA)


def test_evaluate_closed_or_changed_fixed_identity_is_superseded(tmp_path: Path) -> None:
    target = _target(tmp_path)
    snapshot = _snapshot(target, binding=_binding(target, state="closed"))

    assert evaluate_remote_ci(snapshot, elapsed=1, stable_polls=1, limits=RemoteCILimits()).status == "superseded"


def test_verdict_payload_is_session_sha_and_full_pr_identity_bound(tmp_path: Path) -> None:
    target = _target(tmp_path)
    snapshot = _snapshot(target, head=(_observation("Build", "pass"),))
    verdict = evaluate_remote_ci(snapshot, elapsed=12.5, stable_polls=2, limits=RemoteCILimits())

    payload = remote_ci_verdict_payload(
        verdict,
        session_id="session-1",
        poll_count=3,
        started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:00:12Z",
        discovery_deadline=120.0,
        completion_deadline=1_800.0,
    )

    assert payload["schema_version"] == 1
    assert payload["session_id"] == "session-1"
    assert payload["target"] == {
        "base_repository": "example/project",
        "base_ref": "main",
        "head_repository": "contributor/fork",
        "head_ref": "feature/ci",
        "pr_number": 42,
        "pr_url": "https://github.com/example/project/pull/42",
        "remote": "fork",
        "pushed_sha": PUSHED_SHA,
    }
    assert payload["binding"] == {
        "pr_number": 42,
        "pr_url": "https://github.com/example/project/pull/42",
        "base_repository": "example/project",
        "base_ref": "main",
        "head_repository": "contributor/fork",
        "head_ref": "feature/ci",
        "head_sha": PUSHED_SHA,
        "merge_sha": None,
        "state": "open",
    }
    assert payload["head_sha"] == PUSHED_SHA
    assert payload["merge_sha"] is None
    assert "target_dir" not in json.dumps(payload)
    assert "must not persist" not in json.dumps(payload)


def test_verdict_writer_atomically_replaces_prior_json(tmp_path: Path) -> None:
    target = _target(tmp_path)
    verdict = evaluate_remote_ci(_snapshot(target), elapsed=120, stable_polls=2, limits=RemoteCILimits())
    path = tmp_path / "deep" / "remote-ci-verdict.json"
    path.parent.mkdir()
    path.write_text('{"status":"stale"}\n', encoding="utf-8")

    write_remote_ci_verdict(
        path,
        verdict,
        session_id="session-2",
        poll_count=2,
        started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:02:00Z",
        discovery_deadline=120.0,
        completion_deadline=1_800.0,
    )

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["status"] == "no_ci"
    assert loaded["session_id"] == "session-2"


def test_handoff_payload_is_only_for_non_success_and_preserves_identity(tmp_path: Path) -> None:
    target = _target(tmp_path)
    failed = evaluate_remote_ci(
        _snapshot(
            target,
            policy=RequiredPolicy((RequiredContext("Build", 10),), False),
            head=(_observation("Build", "fail"),),
        ),
        elapsed=12,
        stable_polls=1,
        limits=RemoteCILimits(),
    )
    handoff = remote_ci_handoff_payload(failed, session_id="session-3")

    assert handoff["outcome"] == "failed"
    assert handoff["target"] == remote_ci_verdict_payload(
        failed,
        session_id="session-3",
        poll_count=1,
        started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:00:12Z",
        discovery_deadline=120,
        completion_deadline=1_800,
    )["target"]
    assert handoff["binding"] is not None
    assert "ordinary push" in str(handoff["next_action"])

    passed = evaluate_remote_ci(
        _snapshot(target, head=(_observation("Build", "pass"),)),
        elapsed=12,
        stable_polls=2,
        limits=RemoteCILimits(),
    )
    with pytest.raises(ValueError):
        remote_ci_handoff_payload(passed, session_id="session-3")


def test_pre_target_unavailable_uses_null_identity_without_placeholder() -> None:
    verdict = unavailable_remote_ci_verdict(
        reason="The pushed repository could not be bound to a pull request",
        diagnostic="token=super-secret-value",
    )
    payload = remote_ci_verdict_payload(
        verdict,
        session_id="session-4",
        poll_count=1,
        started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:00:01Z",
        discovery_deadline=120,
        completion_deadline=1_800,
    )
    handoff = remote_ci_handoff_payload(verdict, session_id="session-4")

    assert payload["status"] == "unavailable"
    assert payload["target"] is None
    assert payload["binding"] is None
    assert payload["policy"] is None
    assert payload["active_workflow_count"] is None
    polling = payload["polling"]
    assert isinstance(polling, dict)
    assert polling["stable_polls"] == 0
    assert handoff["target"] is None
    assert "super-secret-value" not in json.dumps(payload)


def test_pending_factory_preserves_target_without_claiming_observed_ci(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    verdict = pending_remote_ci_verdict(target)
    payload = remote_ci_verdict_payload(
        verdict,
        session_id="session-pending",
        poll_count=0,
        started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:00:00Z",
        discovery_deadline=120,
        completion_deadline=1_800,
    )
    handoff = remote_ci_handoff_payload(verdict, session_id="session-pending")

    assert verdict.archive_state == "partial"
    assert payload["status"] == "pending"
    assert payload["reason"] == "discovering remote CI"
    assert payload["target"] is not None
    assert payload["binding"] is None
    assert payload["policy"] is None
    assert payload["active_workflow_count"] is None
    assert payload["evidence_sha"] is None
    assert payload["required_observations"] == []
    assert payload["advisory_observations"] == []
    assert payload["archive_state"] == "partial"
    polling = payload["polling"]
    assert isinstance(polling, dict)
    assert polling["poll_count"] == 0
    assert handoff["outcome"] == "incomplete"
    assert handoff["target"] == payload["target"]
    assert handoff["binding"] is None

    unavailable_payload = remote_ci_verdict_payload(
        unavailable_remote_ci_verdict(reason="unavailable"),
        session_id="session-unavailable",
        poll_count=0,
        started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:00:00Z",
        discovery_deadline=120,
        completion_deadline=1_800,
    )
    unavailable_polling = unavailable_payload["polling"]
    assert isinstance(unavailable_polling, dict)
    assert unavailable_polling["poll_count"] == 0

    passed = evaluate_remote_ci(
        _snapshot(target, head=(_observation("Build", "pass"),)),
        elapsed=1,
        stable_polls=2,
        limits=RemoteCILimits(),
    )
    with pytest.raises(ValueError, match="poll count"):
        remote_ci_verdict_payload(
            passed,
            session_id="session-complete",
            poll_count=0,
            started_at="2026-09-06T12:00:00Z",
            updated_at="2026-09-06T12:00:00Z",
            discovery_deadline=120,
            completion_deadline=1_800,
        )


def test_local_host_facts_report_native_identity_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.release", lambda: "25.0")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("platform.python_implementation", lambda: "CPython")
    monkeypatch.setattr("platform.python_version", lambda: "3.13.5")

    assert local_host_facts() == {
        "system": "Darwin",
        "release": "25.0",
        "machine": "arm64",
        "python_implementation": "CPython",
        "python_version": "3.13.5",
    }


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.value += delay


class _ScriptedFetcher:
    def __init__(
        self,
        actions: list[
            RemoteCISnapshot
            | BaseException
            | Callable[[GitHubRequestBudget], RemoteCISnapshot]
        ],
    ) -> None:
        self.actions = actions
        self.deadlines: list[float] = []
        self.calls = 0

    async def fetch(
        self, target: RemoteCITarget, *, budget: GitHubRequestBudget
    ) -> RemoteCISnapshot:
        del target
        self.calls += 1
        self.deadlines.append(budget.deadline)
        budget.next_timeout()
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        if callable(action):
            return action(budget)
        return action


@pytest.mark.asyncio
async def test_waiter_delayed_registration_and_success_switches_absolute_budget(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    empty = _snapshot(target)
    policy = RequiredPolicy((RequiredContext("Build", 10),), True)
    pending = _snapshot(target, policy=policy, head=(_observation("Build", "pending"),))
    passed = _snapshot(target, policy=policy, head=(_observation("Build", "pass"),))
    fetcher = _ScriptedFetcher([empty, pending, passed, passed])
    clock = _Clock()
    emitted: list[RemoteCIVerdict] = []

    verdict = await wait_for_remote_ci(
        target,
        fetcher=fetcher,
        limits=RemoteCILimits(poll_seconds=10, discovery_seconds=120, completion_seconds=1800),
        monotonic=clock,
        sleep=clock.sleep,
        on_snapshot=emitted.append,
    )

    assert verdict.status == "passed"
    assert [item.status for item in emitted] == ["pending", "pending", "pending", "passed"]
    assert fetcher.deadlines == [120.0, 120.0, 1800.0, 1800.0]


@pytest.mark.asyncio
async def test_waiter_explicit_start_is_the_authoritative_artifact_and_budget_deadline(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    clock = _Clock()
    clock.value = 25.0
    fetcher = _ScriptedFetcher([_snapshot(target)] * 4)
    verdict_path = tmp_path / "verdict.json"
    limits = RemoteCILimits(
        poll_seconds=60,
        discovery_seconds=120,
        completion_seconds=1800,
    )

    def persist(verdict: RemoteCIVerdict) -> None:
        write_remote_ci_verdict(
            verdict_path,
            verdict,
            session_id="deadline-session",
            poll_count=1,
            started_at="2026-09-06T12:00:00Z",
            updated_at="2026-09-06T12:00:01Z",
            discovery_deadline=125.0,
            completion_deadline=1805.0,
        )

    verdict = await wait_for_remote_ci(
        target,
        fetcher=fetcher,
        limits=limits,
        monotonic=clock,
        monotonic_started_at=5.0,
        sleep=clock.sleep,
        on_snapshot=persist,
    )

    artifact = json.loads(verdict_path.read_text())
    assert verdict.status == "no_ci"
    assert verdict.elapsed_seconds == 120.0
    assert fetcher.deadlines == [125.0, 125.0]
    assert artifact["polling"]["discovery_deadline"] == fetcher.deadlines[0]
    assert artifact["polling"]["completion_deadline"] == 1805.0


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [float("inf"), 11.0])
async def test_waiter_rejects_nonfinite_or_future_explicit_start(
    tmp_path: Path, started: float
) -> None:
    clock = _Clock()
    clock.value = 10.0
    with pytest.raises(ValueError, match="monotonic start"):
        await wait_for_remote_ci(
            _target(tmp_path),
            fetcher=_ScriptedFetcher([]),
            monotonic=clock,
            monotonic_started_at=started,
            sleep=clock.sleep,
            on_snapshot=lambda _verdict: None,
        )


@pytest.mark.asyncio
async def test_waiter_policy_and_merge_changes_reset_two_poll_stability(tmp_path: Path) -> None:
    target = _target(tmp_path)
    policy_a = RequiredPolicy((RequiredContext("Build", 10),), False)
    policy_b = RequiredPolicy((RequiredContext("Build", 10), RequiredContext("Lint", 11)), False)
    green_a = _snapshot(target, policy=policy_a, head=(_observation("Build", "pass"),))
    green_b = _snapshot(
        target,
        policy=policy_b,
        head=(_observation("Build", "pass"), _observation("Lint", "pass", app_id=11)),
    )
    merge_a = _snapshot(
        target,
        policy=policy_a,
        binding=_binding(target, merge_sha=MERGE_SHA),
        merge=(_observation("Build", "pass"),),
    )
    replacement_sha = "3" * 40
    merge_b = _snapshot(
        target,
        policy=policy_a,
        binding=_binding(target, merge_sha=replacement_sha),
        merge=(_observation("Build", "pass"),),
    )

    for actions in ([green_a, green_b, green_b], [merge_a, merge_b, merge_b]):
        fetcher = _ScriptedFetcher(list(actions))
        clock = _Clock()
        emitted: list[RemoteCIVerdict] = []
        verdict = await wait_for_remote_ci(
            target,
            fetcher=fetcher,
            limits=RemoteCILimits(poll_seconds=1),
            monotonic=clock,
            sleep=clock.sleep,
            on_snapshot=emitted.append,
        )
        assert verdict.status == "passed"
        assert [item.status for item in emitted] == ["pending", "pending", "passed"]


@pytest.mark.asyncio
async def test_waiter_allows_initial_stale_head_then_supersedes_after_binding(tmp_path: Path) -> None:
    target = _target(tmp_path)
    stale = _snapshot(target, binding=_binding(target, head_sha="4" * 40))
    green = _snapshot(target, head=(_observation("Build", "pass"),))
    clock = _Clock()
    emitted: list[RemoteCIVerdict] = []
    verdict = await wait_for_remote_ci(
        target,
        fetcher=_ScriptedFetcher([stale, green, green]),
        limits=RemoteCILimits(poll_seconds=1),
        monotonic=clock,
        sleep=clock.sleep,
        on_snapshot=emitted.append,
    )
    assert verdict.status == "passed"

    clock = _Clock()
    emitted = []
    verdict = await wait_for_remote_ci(
        target,
        fetcher=_ScriptedFetcher([green, stale]),
        limits=RemoteCILimits(poll_seconds=1),
        monotonic=clock,
        sleep=clock.sleep,
        on_snapshot=emitted.append,
    )
    assert verdict.status == "superseded"
    assert verdict.binding is not None
    assert verdict.binding.head_sha == "4" * 40
    assert [item.status for item in emitted] == ["pending", "superseded"]


@pytest.mark.asyncio
async def test_waiter_classifies_exact_deadlines_from_last_complete_snapshot(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    cases = [
        (_snapshot(target), 120.0, "no_ci"),
        (
            _snapshot(
                target,
                policy=RequiredPolicy((RequiredContext("Build", 10),), False),
            ),
            120.0,
            "missing",
        ),
        (
            _snapshot(
                target,
                policy=RequiredPolicy((RequiredContext("Build", 10),), False),
                head=(_observation("Build", "pending"),),
            ),
            1800.0,
            "timed_out",
        ),
    ]
    for snapshot, expected_deadline, expected_status in cases:
        clock = _Clock()
        fetcher = _ScriptedFetcher([snapshot] * 100)
        verdict = await wait_for_remote_ci(
            target,
            fetcher=fetcher,
            limits=RemoteCILimits(
                poll_seconds=60,
                discovery_seconds=120,
                completion_seconds=1800,
            ),
            monotonic=clock,
            sleep=clock.sleep,
            on_snapshot=lambda _verdict: None,
        )
        assert verdict.status == expected_status
        assert clock.value == expected_deadline
        assert fetcher.deadlines[-1] == expected_deadline
        assert all(deadline <= expected_deadline for deadline in fetcher.deadlines)


@pytest.mark.asyncio
async def test_waiter_advisory_observation_does_not_extend_missing_required_discovery(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    snapshot = _snapshot(
        target,
        policy=RequiredPolicy((RequiredContext("Build", 10),), False),
        head=(_observation("Advisory", "pending", app_id=11),),
    )
    clock = _Clock()
    fetcher = _ScriptedFetcher([snapshot, snapshot])

    verdict = await wait_for_remote_ci(
        target,
        fetcher=fetcher,
        limits=RemoteCILimits(poll_seconds=60),
        monotonic=clock,
        sleep=clock.sleep,
        on_snapshot=lambda _verdict: None,
    )

    assert verdict.status == "missing"
    assert fetcher.calls == 2
    assert fetcher.deadlines == [120.0, 120.0]


@pytest.mark.asyncio
async def test_waiter_incomplete_first_poll_is_unavailable_and_later_uses_last_complete(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)

    def expire(budget: GitHubRequestBudget) -> RemoteCISnapshot:
        clock.value = budget.deadline
        budget.next_timeout()
        raise AssertionError("unreachable")

    clock = _Clock()
    emitted: list[RemoteCIVerdict] = []
    verdict = await wait_for_remote_ci(
        target,
        fetcher=_ScriptedFetcher([expire]),
        limits=RemoteCILimits(poll_seconds=60),
        monotonic=clock,
        sleep=clock.sleep,
        on_snapshot=emitted.append,
    )
    assert verdict.status == "unavailable"
    assert [item.status for item in emitted] == ["unavailable"]

    clock = _Clock()
    fetcher = _ScriptedFetcher([_snapshot(target), expire])
    verdict = await wait_for_remote_ci(
        target,
        fetcher=fetcher,
        limits=RemoteCILimits(poll_seconds=60),
        monotonic=clock,
        sleep=clock.sleep,
        on_snapshot=lambda _verdict: None,
    )
    assert verdict.status == "missing"
    assert fetcher.calls == 2


@pytest.mark.asyncio
async def test_waiter_api_error_is_bounded_unavailable(tmp_path: Path) -> None:
    target = _target(tmp_path)
    emitted: list[RemoteCIVerdict] = []
    verdict = await wait_for_remote_ci(
        target,
        fetcher=_ScriptedFetcher([GitError("secret=top-secret")]),
        monotonic=lambda: 0.0,
        sleep=lambda _delay: asyncio.sleep(0),
        on_snapshot=emitted.append,
    )
    assert verdict.status == "unavailable"
    assert "top-secret" not in (verdict.diagnostic or "")
    assert emitted == [verdict]


def _page(endpoint: str, *, page: int = 1) -> str:
    separator = "&" if "?" in endpoint else "?"
    return f"{endpoint}{separator}per_page=100&page={page}"


def _seed_fetch(fake_gh: FakeGh, target: RemoteCITarget) -> None:
    fake_gh.set_response("GET", "repos/example/project/pulls/42", _pr_row(target, merge_commit_sha=None))
    fake_gh.set_response("GET", _page("repos/example/project/rules/branches/main"), [])
    fake_gh.set_response(
        "GET",
        "repos/example/project/branches/main/protection/required_status_checks",
        {"strict": False, "contexts": ["Build"], "checks": []},
    )
    fake_gh.set_response(
        "GET",
        _page("repos/example/project/actions/workflows"),
        {"total_count": 0, "workflows": []},
    )
    checks_endpoint = f"repos/example/project/commits/{PUSHED_SHA}/check-runs?filter=latest"
    fake_gh.set_response(
        "GET",
        _page(checks_endpoint),
        {"total_count": 1, "check_runs": [_check(1, "Build")]},
    )
    fake_gh.set_response(
        "GET",
        _page(f"repos/example/project/commits/{PUSHED_SHA}/statuses"),
        [],
    )


@pytest.mark.asyncio
async def test_github_fetcher_uses_frozen_boundary_and_returns_only_normalized_rows(
    fake_gh: FakeGh, git_repo: Path
) -> None:
    target = _target(git_repo)
    _seed_fetch(fake_gh, target)
    budget = GitHubRequestBudget(
        deadline=time.monotonic() + 30,
        per_request_seconds=5,
        monotonic=time.monotonic,
    )

    snapshot = await GitHubRemoteCIFetcher().fetch(target, budget=budget)

    assert snapshot.binding.pr_number == target.pr_number
    assert snapshot.policy.contexts == (RequiredContext("Build", None),)
    assert snapshot.head_observations[0].context == "Build"
    assert "must not persist" not in repr(snapshot)
    assert len(fake_gh.process_calls()) == 6


def _fd_count() -> int | None:
    fd_dir = Path("/dev/fd")
    return len(list(fd_dir.iterdir())) if fd_dir.is_dir() else None


async def _wait_for_pids(path: Path) -> dict[str, int]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded.get("direct"), int) and isinstance(loaded.get("grandchild"), int):
                return {
                    "direct": loaded["direct"],
                    "grandchild": loaded["grandchild"],
                }
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            pass
        await asyncio.sleep(0.01)
    raise AssertionError("blocking fake gh did not publish process ids")


async def _wait_for_process_group_exit(pgid: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"process group {pgid} survived cancellation")


@pytest.mark.asyncio
async def test_waiter_cancellation_persists_once_after_real_gh_group_is_reaped(
    fake_gh: FakeGh, git_repo: Path, tmp_path: Path
) -> None:
    target = _target(git_repo)
    pid_file = tmp_path / "remote-ci-pids.json"
    fake_gh.serve_blocking_process(
        "GET repos/example/project/pulls/42", pid_file=pid_file
    )
    verdict_path = tmp_path / "remote-ci-verdict.json"
    emitted: list[RemoteCIVerdict] = []

    def persist(verdict: Any) -> None:
        emitted.append(verdict)
        write_remote_ci_verdict(
            verdict_path,
            verdict,
            session_id="cancel-session",
            poll_count=1,
            started_at="2026-09-06T12:00:00Z",
            updated_at="2026-09-06T12:00:01Z",
            discovery_deadline=120,
            completion_deadline=1800,
        )

    baseline = _fd_count()
    task = asyncio.create_task(
        wait_for_remote_ci(
            target,
            fetcher=GitHubRemoteCIFetcher(),
            on_snapshot=persist,
        )
    )
    pids = await _wait_for_pids(pid_file)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _wait_for_process_group_exit(pids["direct"])
    await asyncio.sleep(0.05)

    assert [item.status for item in emitted] == ["cancelled"]
    assert json.loads(verdict_path.read_text(encoding="utf-8"))["status"] == "cancelled"
    assert len(fake_gh.process_calls()) == 1
    if baseline is not None:
        deadline = time.monotonic() + 2
        while _fd_count() != baseline and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert _fd_count() == baseline


@pytest.mark.asyncio
async def test_later_real_request_timeout_at_discovery_uses_trusted_empty_snapshot(
    fake_gh: FakeGh, git_repo: Path, tmp_path: Path
) -> None:
    target = _target(git_repo)
    pid_file = tmp_path / "deadline-pids.json"
    fake_gh.serve_blocking_process(
        "GET repos/example/project/pulls/42", pid_file=pid_file
    )

    class CompleteTwiceThenRealTimeout:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch(
            self, fixed: RemoteCITarget, *, budget: GitHubRequestBudget
        ) -> RemoteCISnapshot:
            self.calls += 1
            if self.calls <= 2:
                return _snapshot(fixed)
            await gh_pr_ci_snapshot(
                fixed.target_dir,
                "example",
                "project",
                fixed.pr_number,
                budget=budget,
            )
            raise AssertionError("blocking GitHub request unexpectedly returned")

    fetcher = CompleteTwiceThenRealTimeout()
    emitted: list[RemoteCIVerdict] = []
    limits = RemoteCILimits(
        poll_seconds=0.1,
        discovery_seconds=2.0,
        completion_seconds=5.0,
        request_seconds=5.0,
    )

    verdict = await wait_for_remote_ci(
        target,
        fetcher=fetcher,
        limits=limits,
        monotonic=time.monotonic,
        sleep=asyncio.sleep,
        on_snapshot=emitted.append,
    )
    pids = await _wait_for_pids(pid_file)
    await _wait_for_process_group_exit(pids["direct"])

    assert verdict.status == "no_ci"
    assert [item.status for item in emitted] == ["pending", "pending", "no_ci"]
    assert fetcher.calls == 3
    assert len(fake_gh.process_calls()) == 1
    GitHubRemoteCIFetcher,
