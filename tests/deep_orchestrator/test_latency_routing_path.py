"""The routing record: one artifact, one writer, one meaning (issue #732)."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import AgentEvent, ResultEvent
from daydream.deep.routing_record import read_routing_record, write_routing_record
from daydream.eval.analyzer import analyze_routing
from daydream.review_profile import ResolvedProfile
from daydream.runner import run
from tests.deep_orchestrator.support import _arbiter_stacks, _merged_items
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence
from tests.test_deep_orchestrator import MakeConfig, Mute

#: The pre-#732 ``.daydream/deep/`` artifact names for a forensic run, captured
#: from the pre-change tree through this same stub scenario. ``latency-routing.json``
#: is the one additive artifact issue #732 mandates (A12), so the gate subtracts it
#: before comparing the two sides.
_FORENSIC_BASELINE_DEEP_ARTIFACTS = frozenset({
    "alternatives.json", "adjudication-complete.marker", "arbiter-input.json", "dedup-candidates.json", "diagram.json",
    "diagram.md", "diff-key", "intent.md", "merged-items.json", "review-output.md", "stack-generic-records.json",
    "stack-generic-review.md", "stack-python-records.json", "stack-python-review.md", "stack-react-records.json",
    "stack-react-review.md", "stack-structure-records.json", "stack-structure-review.md",
})

#: Artifacts deliberately added to ``.daydream/deep/`` after the pre-#732 baseline
#: was captured. ``latency-routing.json`` is issue #732's additive routing record
#: (A12); ``adjudication-provenance.json`` is the host-stamped adjudication ledger
#: issue #735 adds; ``test-recipe.json`` is the once-resolved test recipe issue
#: #1408 persists in the preamble. The gate subtracts all three before comparing
#: against the baseline.
_FORENSIC_ADDITIVE_DEEP_ARTIFACTS = frozenset(
    {"review-coverage.json", "latency-routing.json", "adjudication-provenance.json", "test-recipe.json",
     "stage-inputs"}
)

#: The pre-#732 ``arbiter-input.json`` for the exact stub records below: the
#: forensic path must reproduce it byte-for-byte (A12), never a re-ordered or
#: re-shaped selection.
_FORENSIC_BASELINE_ARBITER_INPUT: list[dict[str, object]] = [{
        "arb_id": 1, "confidence": "MEDIUM", "description": "Sample issue", "evidence": "api.py:1", "file": "api.py",
        "line": 1, "rationale": "stub", "severity": "high", "uid": "generic:1",
    },
    {"arb_id": 2, "confidence": "MEDIUM", "description": "Sample issue", "evidence": "api.py:1", "file": "api.py",
        "line": 1, "rationale": "stub", "severity": "high", "uid": "python:1",
    },
    {"arb_id": 3, "confidence": "MEDIUM", "description": "Sample issue", "evidence": "api.py:1", "file": "api.py",
        "line": 1, "rationale": "stub", "severity": "high", "uid": "react:1",
    },
]


def _medium_arbitration_profile() -> ResolvedProfile:
    """Lower the arbiter floor to medium so three selected groups exercise sharding."""
    base = independent_alternatives_profile()
    pipeline = replace(
        base.profile.pipeline, arbitration=replace(base.profile.pipeline.arbitration, min_severity="medium"),
    )
    return replace(base, profile=replace(base.profile, pipeline=pipeline))


async def _run_profile(
    monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute, target: Path, *,
    latency_profile: str, timeout: float = 120, review_profile: ResolvedProfile | None = None,
    parse_severity: str | None = None, parse_by_stack: dict[str, dict[str, Any]] | None = None,
    merge_echo_records: bool = False,
) -> tuple[StubBackend, Path]:
    """Run a configured stub profile; optionally echo records to expose adjudication results."""
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, target)
    stub.parse_severity = parse_severity
    stub.parse_by_stack = parse_by_stack
    stub.merge_echo_records = merge_echo_records
    mute_side_effects()
    config = make_config(target, latency_profile=latency_profile,
        review_profile=review_profile if review_profile is not None else independent_alternatives_profile(),
    )
    with anyio.fail_after(timeout):
        assert await run(config) in (0, 1)
    return stub, target / ".daydream" / "deep"

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
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """MH7: the routing record states why wonder did not run."""
    _, deep = await _run_profile(
        monkeypatch, make_config, mute_side_effects, multi_stack_target, latency_profile="fast", timeout=90,
    )
    record = read_routing_record(deep)
    assert record["wonder"]["outcome"] == "skip"
    assert record["profile"]["selected"] == "fast"
    assert record["risk"]["floors"] == []
    assert "fast" in record["wonder"]["reason"]

async def test_forensic_reproduces_todays_wonder_and_arbiter_artifacts(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """MH2 + A12: same artifacts, same effort values, same marker meaning."""
    _, deep = await _run_profile(
        monkeypatch, make_config, mute_side_effects, multi_stack_target, latency_profile="forensic",
        parse_severity="high",
    )
    assert (deep / "adjudication-complete.marker").exists()
    assert not list(deep.glob("arbiter-group-*-input.json"))
    # A12: the profile work adds the routing record, and issue #735 adds the
    # adjudication provenance ledger; both are deliberate, so the gate subtracts
    # the post-baseline additions before comparing.
    assert {path.name
        for path in deep.iterdir()
        if path.name not in _FORENSIC_ADDITIVE_DEEP_ARTIFACTS
    } == _FORENSIC_BASELINE_DEEP_ARTIFACTS
    assert json.loads((deep / "arbiter-input.json").read_text()) == _FORENSIC_BASELINE_ARBITER_INPUT
    record = read_routing_record(deep)
    assert record["arbiter"]["sharded"] is False
    assert [g["effort"] for g in record["arbiter"]["groups"]] == ["xhigh"]
    assert record["wonder"]["effort"] == "high"

async def test_single_group_sharding_profile_matches_forensic_arbiter_artifacts(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """Co-located targets take one xhigh call and produce artifacts identical to forensic."""
    _, deep = await _run_profile(
        monkeypatch, make_config, mute_side_effects, multi_stack_target, latency_profile="balanced",
        parse_severity="high",
    )
    assert json.loads((deep / "arbiter-input.json").read_text()) == _FORENSIC_BASELINE_ARBITER_INPUT
    assert (deep / "adjudication-complete.marker").exists()
    assert not list(deep.glob("arbiter-group-*-input.json"))
    record = read_routing_record(deep)
    assert record["arbiter"]["sharded"] is False
    assert [g["effort"] for g in record["arbiter"]["groups"]] == ["xhigh"]

async def test_forensic_resume_from_the_whole_block_marker_runs_no_arbiter_call(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """The documented `--start-at merge` contract still trusts adjudication-complete.marker."""
    stub, _ = await _run_profile(
        monkeypatch, make_config, mute_side_effects, multi_stack_target, latency_profile="forensic",
        parse_severity="high",
    )
    base = dict(latency_profile="forensic", review_profile=independent_alternatives_profile())
    stub.calls.clear()
    with anyio.fail_after(60):
        assert await run(make_config(multi_stack_target, start_at="merge", **base)) in (0, 1)
    assert [c for c in stub.calls if "you are the arbiter" in c["prompt"].lower()] == []
    assert (multi_stack_target / ".daydream" / "deep" / "arbiter-input.json").exists()

async def test_multi_group_arbiter_applies_every_verdict_and_records_per_group_effort(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """MH4: one sharded call per group, one merged verdict application."""
    _, deep = await _run_profile(
        monkeypatch, make_config, mute_side_effects, multi_stack_target, latency_profile="balanced",
        parse_by_stack=_arbiter_stacks({"python": "high", "react": "medium", "generic": "medium"}),
        merge_echo_records=True, review_profile=_medium_arbitration_profile(),
    )
    record = read_routing_record(deep)
    assert record["arbiter"]["sharded"] is True
    groups = record["arbiter"]["groups"]
    assert len(groups) > 1
    assert all(group["effort"] in {"high", "xhigh"} for group in groups)
    assert (deep / "adjudication-complete.marker").exists()
    # ARBITRATED is the stub arbiter's revised description prefix: its presence
    # proves `_apply_adjudication_verdicts` reconciled every group's verdicts
    # back onto the run-wide target ordinals after the fan-out.
    assert "ARBITRATED" in json.dumps(_merged_items(deep))
    assert not (deep / "arbiter-input.json").exists()

async def test_resumed_run_reruns_only_incomplete_groups(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """Re-entering arbitration after an interrupted fan-out reruns only the gap."""
    stub, deep = await _run_profile(
        monkeypatch, make_config, mute_side_effects, multi_stack_target, latency_profile="balanced",
        parse_by_stack=_arbiter_stacks({"python": "high", "react": "high", "generic": "high"}),
    )
    first = read_routing_record(deep)
    assert first["arbiter"]["sharded"] is True
    groups = first["arbiter"]["groups"]
    assert len(groups) > 1
    assert all(group["reused"] is False for group in groups)

    # Simulate an interruption: the whole-block marker and one group's marker
    # are gone, the other groups' verdict artifacts survive on disk.
    rerun = groups[0]["group_id"]
    (deep / "adjudication-complete.marker").unlink()
    (deep / f"{rerun}-complete.marker").unlink()

    stub.calls.clear()
    resumed = make_config(
        multi_stack_target, latency_profile="balanced", review_profile=independent_alternatives_profile(),
        start_at="merge",
    )
    with anyio.fail_after(90):
        assert await run(resumed) in (0, 1)

    arbiter_calls = [call for call in stub.calls if "you are the arbiter" in call["prompt"].lower()]
    assert len(arbiter_calls) == 1
    assert f"{rerun}-input.json" in arbiter_calls[0]["prompt"]
    assert (deep / "adjudication-complete.marker").exists()
    resumed_groups = {group["group_id"]: group for group in read_routing_record(deep)["arbiter"]["groups"]}
    assert resumed_groups[rerun]["reused"] is False
    for group in groups:
        if group["group_id"] != rerun:
            assert resumed_groups[group["group_id"]]["reused"] is True

async def test_failed_group_fails_open_and_is_retried_on_resume(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """A raising group is recorded, blocks the marker, and reruns on resume."""
    stub, deep = await _run_profile(
        monkeypatch, make_config, mute_side_effects, multi_stack_target, latency_profile="balanced",
        parse_by_stack=_arbiter_stacks({"python": "high", "react": "high", "generic": "high"}),
    )
    groups = read_routing_record(deep)["arbiter"]["groups"]
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


async def test_resume_retains_confirmed_demoted_findings_without_second_suppression(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity, stub.merge_echo_records, stub.suppression_keep = 'high', True, False
    execute = stub.execute

    async def demote(cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        async for event in execute(cwd, prompt, *args, **kwargs):
            if 'you are the arbiter' in prompt.lower() and isinstance(event, ResultEvent):
                assert isinstance(event.structured_output, dict)
                payload = {'findings': [{**row, 'severity': 'low', 'confidence': 'HIGH'}
                                       for row in event.structured_output['findings']]}
                event = replace(event, structured_output=payload)
            yield event

    monkeypatch.setattr(stub, 'execute', demote)
    mute_side_effects()
    config = make_config(multi_stack_target, latency_profile='balanced', precision_mode=True,
                         review_profile=independent_alternatives_profile())
    assert await run(config) == 0
    deep = multi_stack_target / '.daydream/deep'
    first = _merged_items(deep)
    demoted = [row for row in first if row['description'].startswith('ARBITRATED:')]
    assert demoted and all(row['severity'] == 'low' for row in demoted)
    stub.calls.clear()
    assert await run(replace(config, start_at='merge')) == 0
    assert not any('you are the arbiter' in call['prompt'].lower() or
                   'you are the suppression reviewer' in call['prompt'].lower() for call in stub.calls)
    assert _merged_items(deep) == first


async def test_no_target_completion_reruns_when_selection_policy_changes(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    base = independent_alternatives_profile()
    pipeline = replace(base.profile.pipeline, arbitration=replace(
        base.profile.pipeline.arbitration, min_severity='high', contested_location=False))
    profile = replace(base, profile=replace(base.profile, pipeline=pipeline))
    stub, deep = await _run_profile(monkeypatch, make_config, mute_side_effects, multi_stack_target,
        latency_profile='balanced', review_profile=profile, parse_severity='medium')
    assert not any('you are the arbiter' in call['prompt'].lower() for call in stub.calls)
    assert json.loads((deep / 'adjudication-complete.marker').read_text())['plan'] is None
    changed = replace(profile, profile=replace(profile.profile, pipeline=replace(
        pipeline, arbitration=replace(pipeline.arbitration, min_severity='medium'))))
    stub.calls.clear()
    assert await run(make_config(multi_stack_target, start_at='merge', latency_profile='balanced',
                                 review_profile=changed)) == 0
    assert any('you are the arbiter' in call['prompt'].lower() for call in stub.calls)
