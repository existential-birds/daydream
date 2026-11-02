"""The routing record: one artifact, one writer, one meaning (issue #732)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.deep.routing_record import read_routing_record
from daydream.review_profile import ResolvedProfile
from daydream.runner import run
from tests.deep_orchestrator.empty_synthesis_support import empty_review_config
from tests.deep_orchestrator.precision_overlap_support import PrecisionOverlapBackend
from tests.deep_orchestrator.support import _arbiter_stacks
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence
from tests.test_deep_orchestrator import MakeConfig, Mute


async def _run_profile(
    monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute, target: Path, *,
    latency_profile: str, timeout: float = 120, review_profile: ResolvedProfile | None = None,
    parse_severity: str | None = None, parse_by_stack: dict[str, dict[str, Any]] | None = None,
    merge_echo_records: bool = False, diagram: str | None = None,
) -> tuple[StubBackend, Path]:
    """Run a configured stub profile; optionally echo records to expose adjudication results."""
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, target)
    stub.parse_severity = parse_severity
    stub.parse_by_stack = parse_by_stack
    stub.merge_echo_records = merge_echo_records
    mute_side_effects()
    config = make_config(target, latency_profile=latency_profile, diagram=diagram,
        review_profile=review_profile if review_profile is not None else independent_alternatives_profile(),
    )
    with anyio.fail_after(timeout):
        assert await run(config) in (0, 1)
    return stub, target / ".daydream" / "deep"

async def test_failed_group_fails_open_and_is_retried_on_resume(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """A raising group is recorded, blocks the marker, and reruns on resume."""
    stub, deep = await _run_profile(
        monkeypatch, make_config, mute_side_effects, multi_stack_target, latency_profile="balanced",
        parse_by_stack=_arbiter_stacks({"python": "high", "react": "high", "generic": "high"}),
    )
    first = read_routing_record(deep)
    assert first["arbiter"]["sharded"] is True
    groups = first["arbiter"]["groups"]
    assert len(groups) > 1
    assert all(group["reused"] is False for group in groups)
    failed = groups[0]["group_id"]
    (deep / "adjudication-complete.marker").unlink()
    (deep / f"{failed}-complete.marker").unlink()
    stub.arbiter_fail_group = failed
    stub.calls.clear()

    failing_run = make_config(
        multi_stack_target, latency_profile="balanced", review_profile=independent_alternatives_profile(),
        start_at="merge",
    )
    with anyio.fail_after(90):
        assert await run(failing_run) in (0, 1)

    # The run continued (fail-open), but the block is not complete.
    failed_record = read_routing_record(deep)["arbiter"]
    assert failed_record["failed_groups"] == [failed]
    assert not (deep / "adjudication-complete.marker").exists()

    stub.arbiter_fail_group = None
    stub.calls.clear()
    retry_run = make_config(
        multi_stack_target, latency_profile="balanced", review_profile=independent_alternatives_profile(),
        start_at="merge",
    )
    with anyio.fail_after(90):
        assert await run(retry_run) in (0, 1)

    retry_calls = [call for call in stub.calls if "you are the arbiter" in call["prompt"].lower()]
    assert len(retry_calls) == 1
    assert f"{failed}-input.json" in retry_calls[0]["prompt"]
    retried = read_routing_record(deep)["arbiter"]
    assert retried["failed_groups"] == []
    assert (deep / "adjudication-complete.marker").exists()
    resumed_groups = {group["group_id"]: group for group in retried["groups"]}
    assert resumed_groups[failed]["reused"] is False
    for group in groups:
        if group["group_id"] != failed:
            assert resumed_groups[group["group_id"]]["reused"] is True

    # A validated completion marker proves the finalised records; resume skips the arbiter.
    stub.calls.clear()
    with anyio.fail_after(90):
        assert await run(retry_run) in (0, 1)
    assert not [call for call in stub.calls if "you are the arbiter" in call["prompt"].lower()]


async def test_precision_overlap_closes_both_streams_without_publishing_completion(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation must unwind both live adjudication provider streams.

    Existing reviewer-cancellation coverage does not enter the new adjudication
    fan-out, where a leaked sibling could continue writing after run finalization.
    """
    backend = PrecisionOverlapBackend(multi_stack_target, blocked=True)
    backend.fanout_concurrency = 2
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    profile = independent_alternatives_profile()
    config = empty_review_config(multi_stack_target, tmp_path / "cancel-overlap.json", precision_mode=True,
        latency_profile="forensic", review_profile=profile, review_cache_enabled=True)

    async def wait_for_overlap() -> None:
        await backend.started["arbiter"].wait()
        await backend.started["suppression"].wait()

    task = asyncio.create_task(run(config))
    ready = asyncio.create_task(wait_for_overlap())
    try:
        with anyio.fail_after(60):
            done, _pending = await asyncio.wait((task, ready), return_when=asyncio.FIRST_COMPLETED)
            if task in done and ready not in done:
                await task
                pytest.fail("Runner exited before both adjudication streams started")
            await ready
        live = backend.input_paths["suppression"].parent
        assert not (live / "adjudication-complete.marker").exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not ready.done():
            ready.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, ready, return_exceptions=True)
    assert all(event.is_set() for event in backend.closed.values())
    assert not any(event.is_set() for event in backend.release.values())
    assert backend.completed == []
    deep = multi_stack_target / ".daydream/deep"
    assert not (live / "adjudication-complete.marker").exists()
    assert not (deep / "adjudication-complete.marker").exists()
    assert all(json.loads(path.read_text())["unit"] != "arbiter"
               for path in (deep.parent / "review-cache/entries").glob("*/manifest.json"))


def test_sharded_arbiter_plan_names_each_forcing_signal_and_records_effort_pins() -> None:
    """Group effort follows its own signal; an explicit pin overrides the recorded route and says so.

    The runner journeys use all-high fixtures, so contested-only and routine groups
    and the operator pin never reach the routing record there.
    """
    from daydream.deep.adjudication_steps import _arbiter_groups_record
    from daydream.deep.arbiter import ArbiterGroup
    from daydream.deep.latency import PROFILE_ROUTES, arbiter_plan

    groups = [ArbiterGroup(f"arbiter-group-{index}", (index,), (f"python:{index}",)) for index in range(3)]
    records = [{"severity": "high"}, {"severity": "low"}, {"severity": "low"}]
    plan = arbiter_plan(PROFILE_ROUTES["balanced"], groups, records=records, contested={1})
    assert plan.sharded
    assert [(group.effort, group.reason) for group in plan.groups] == [
        ("xhigh", "arbiter-group-0: high severity"), ("xhigh", "arbiter-group-1: contested findings"),
        ("high", "arbiter-group-2: routine group")]
    pinned = _arbiter_groups_record(plan, effort_pin="low", reused={"arbiter-group-1": True})
    assert [(group["effort"], group["reused"]) for group in pinned] == [("low", False), ("low", True), ("low", False)]
    assert all(group["reason"] == "explicit low effort pin overrode the route (A4)" for group in pinned)
