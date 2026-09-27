"""The routing record: one artifact, one writer, one meaning (issue #732)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import anyio
import pytest

from daydream.deep.routing_record import read_routing_record, write_routing_record
from daydream.eval.analyzer import analyze_routing
from daydream.review_profile import ResolvedProfile
from daydream.runner import run
from tests.deep_orchestrator.support import _merged_items
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.stub_backend import install_stub_backend, silence
from tests.test_deep_orchestrator import MakeConfig, Mute


def _arbiter_stacks(severities: dict[str, str]) -> dict[str, dict[str, object]]:
    """Per-stack findings at three distinct ``(file, line)`` locations.

    Three file components mean three arbiter groups whenever the selection
    spans more than one co-located target. The descriptions differ so a dedup
    pass cannot fold the records together.
    """
    locations = {
        "python": ("api.py", "python finding"),
        "react": ("App.tsx", "react finding"),
        "generic": ("README.md", "generic finding"),
    }
    return {
        name: {
            "severity": severities[name],
            "confidence": "high",
            "file": file,
            "line": 1,
            "description": description,
        }
        for name, (file, description) in locations.items()
    }


def _medium_arbitration_profile() -> ResolvedProfile:
    """The independent-alternatives profile with the arbiter floor at medium.

    The default floor is ``high``, which would select only the one high-severity
    record and make this a single-group (unsharded) run. Lowering the floor to
    ``medium`` puts three distinct records into the selection so the fan-out has
    more than one group to shard over.
    """
    base = independent_alternatives_profile()
    pipeline = replace(
        base.profile.pipeline,
        arbitration=replace(base.profile.pipeline.arbitration, min_severity="medium"),
    )
    return replace(base, profile=replace(base.profile, pipeline=pipeline))


def test_routing_record_merges_instead_of_clobbering(tmp_path: Path) -> None:
    dd = tmp_path / "deep"
    dd.mkdir()
    write_routing_record(dd, {"profile": {"selected": "balanced"}, "risk": {"floors": ["security_surface"]}})
    path = write_routing_record(dd, {"wonder": {"outcome": "run", "effort": "medium", "reason": "profile floor"}})

    record = read_routing_record(dd)
    assert path.name == "latency-routing.json"
    assert json.loads(path.read_text(encoding="utf-8"))["profile"]["selected"] == "balanced"
    assert record["profile"]["selected"] == "balanced"
    assert record["risk"]["floors"] == ["security_surface"]
    assert record["wonder"]["reason"] == "profile floor"


def test_absent_record_reads_as_an_empty_mapping(tmp_path: Path) -> None:
    assert read_routing_record(tmp_path) == {}


def test_evaluated_run_carries_its_latency_profile(tmp_path: Path) -> None:
    """SH1: an archived run's profile is readable without opening its directory."""
    dd = tmp_path / ".daydream"
    (dd / "deep").mkdir(parents=True)
    write_routing_record(dd / "deep", {"profile": {"selected": "forensic"}, "risk": {"floors": []}})
    assert analyze_routing(dd)["profile"] == "forensic"
    assert analyze_routing(tmp_path / "nothing-here") == {"profile": None, "decisions": {}}


async def test_skipped_wonder_records_profile_signals_and_reason(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """MH7: the routing record states why wonder did not run."""
    silence(monkeypatch)
    install_stub_backend(monkeypatch, multi_stack_target)
    mute_side_effects()
    config = make_config(
        multi_stack_target,
        latency_profile="fast",
        review_profile=independent_alternatives_profile(),
    )
    with anyio.fail_after(90):
        assert await run(config) in (0, 1)

    record = read_routing_record(multi_stack_target / ".daydream" / "deep")
    assert record["wonder"]["outcome"] == "skip"
    assert record["profile"]["selected"] == "fast"
    assert record["risk"]["floors"] == []
    assert "fast" in record["wonder"]["reason"]


async def test_multi_group_arbiter_applies_every_verdict_and_records_per_group_effort(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """MH4: one sharded call per group, one merged verdict application."""
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_by_stack = _arbiter_stacks(
        {"python": "high", "react": "medium", "generic": "medium"}
    )
    # The merge agent echoes the per-stack records file contents verbatim, so
    # every arbitration verdict is observable in the shipped merged items.
    stub.merge_echo_records = True
    mute_side_effects()
    config = make_config(
        multi_stack_target,
        latency_profile="balanced",
        review_profile=_medium_arbitration_profile(),
    )
    with anyio.fail_after(120):
        assert await run(config) in (0, 1)

    deep = multi_stack_target / ".daydream" / "deep"
    record = read_routing_record(deep)
    assert record["arbiter"]["sharded"] is True
    groups = record["arbiter"]["groups"]
    assert len(groups) > 1
    assert all(group["effort"] in {"high", "xhigh"} for group in groups)
    assert (deep / "arbiter-complete.marker").exists()
    # ARBITRATED is the stub arbiter's revised description prefix: its presence
    # proves `_apply_adjudication_verdicts` reconciled every group's verdicts
    # back onto the run-wide target ordinals after the fan-out.
    assert "ARBITRATED" in json.dumps(_merged_items(deep))
    assert not (deep / "arbiter-input.json").exists()


async def test_resumed_run_reruns_only_incomplete_groups(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Re-entering arbitration after an interrupted fan-out reruns only the gap."""
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_by_stack = _arbiter_stacks(
        {"python": "high", "react": "high", "generic": "high"}
    )
    mute_side_effects()
    config = make_config(
        multi_stack_target,
        latency_profile="balanced",
        review_profile=independent_alternatives_profile(),
    )
    with anyio.fail_after(120):
        assert await run(config) in (0, 1)

    deep = multi_stack_target / ".daydream" / "deep"
    first = read_routing_record(deep)
    assert first["arbiter"]["sharded"] is True
    groups = first["arbiter"]["groups"]
    assert len(groups) > 1
    assert all(group["reused"] is False for group in groups)

    # Simulate an interruption: the whole-block marker and one group's marker
    # are gone, the other groups' verdict artifacts survive on disk.
    rerun = groups[0]["group_id"]
    (deep / "arbiter-complete.marker").unlink()
    (deep / f"{rerun}-complete.marker").unlink()

    stub.calls.clear()
    resumed = make_config(
        multi_stack_target,
        latency_profile="balanced",
        review_profile=independent_alternatives_profile(),
        start_at="merge",
    )
    with anyio.fail_after(90):
        assert await run(resumed) in (0, 1)

    arbiter_calls = [
        call for call in stub.calls if "you are the arbiter" in call["prompt"].lower()
    ]
    assert len(arbiter_calls) == 1
    assert f"{rerun}-input.json" in arbiter_calls[0]["prompt"]
    assert (deep / "arbiter-complete.marker").exists()
    resumed_groups = {
        group["group_id"]: group for group in read_routing_record(deep)["arbiter"]["groups"]
    }
    assert resumed_groups[rerun]["reused"] is False
    for group in groups:
        if group["group_id"] != rerun:
            assert resumed_groups[group["group_id"]]["reused"] is True


async def test_failed_group_fails_open_and_is_retried_on_resume(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """A raising group is recorded, blocks the marker, and reruns on resume."""
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_by_stack = _arbiter_stacks(
        {"python": "high", "react": "high", "generic": "high"}
    )
    mute_side_effects()
    config = make_config(
        multi_stack_target,
        latency_profile="balanced",
        review_profile=independent_alternatives_profile(),
    )
    with anyio.fail_after(120):
        assert await run(config) in (0, 1)

    deep = multi_stack_target / ".daydream" / "deep"
    groups = read_routing_record(deep)["arbiter"]["groups"]
    failed = groups[0]["group_id"]
    (deep / "arbiter-complete.marker").unlink()
    (deep / f"{failed}-complete.marker").unlink()
    stub.arbiter_fail_group = failed
    stub.calls.clear()

    failing_run = make_config(
        multi_stack_target,
        latency_profile="balanced",
        review_profile=independent_alternatives_profile(),
        start_at="merge",
    )
    with anyio.fail_after(90):
        assert await run(failing_run) in (0, 1)

    # The run continued (fail-open), but the block is not complete.
    failed_record = read_routing_record(deep)["arbiter"]
    assert failed_record["failed_groups"] == [failed]
    assert not (deep / "arbiter-complete.marker").exists()

    stub.arbiter_fail_group = None
    stub.calls.clear()
    retry_run = make_config(
        multi_stack_target,
        latency_profile="balanced",
        review_profile=independent_alternatives_profile(),
        start_at="merge",
    )
    with anyio.fail_after(90):
        assert await run(retry_run) in (0, 1)

    retry_calls = [
        call for call in stub.calls if "you are the arbiter" in call["prompt"].lower()
    ]
    assert len(retry_calls) == 1
    assert f"{failed}-input.json" in retry_calls[0]["prompt"]
    retried = read_routing_record(deep)["arbiter"]
    assert retried["failed_groups"] == []
    assert (deep / "arbiter-complete.marker").exists()
