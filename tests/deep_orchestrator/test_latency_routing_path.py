"""The routing record: one artifact, one writer, one meaning (issue #732)."""
from __future__ import annotations

import asyncio
import json
import re
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
from tests.deep_orchestrator.empty_synthesis_support import empty_review_config
from tests.deep_orchestrator.precision_overlap_support import PrecisionOverlapBackend
from tests.deep_orchestrator.support import _arbiter_stacks, _merged_items
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.review_result import saved_coverage
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
#: #1408 persists in the preamble; ``diagram-complete.json`` is issue #1401's
#: snapshot-bound diagram recovery marker. The gate subtracts these additions
#: before comparing against the baseline.
_FORENSIC_ADDITIVE_DEEP_ARTIFACTS = frozenset(
    {"review-coverage.json", "latency-routing.json", "adjudication-provenance.json", "test-recipe.json", "stage-inputs",
     "diagram-complete.json"}
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
        parse_severity="high", diagram="auto",
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


async def test_precision_overlap_reuses_successful_shard_after_real_sibling_failure(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial concurrent pass must retain shard evidence without claiming whole-pass completion.

    Existing shard-resume tests do not run suppression beside the failing fan-out,
    so they cannot detect premature whole-pass publication or suppression shifting identities.
    """
    backend = PrecisionOverlapBackend(multi_stack_target, blocked=True)
    backend.fanout_concurrency = 4
    execute = backend.execute
    fail_docs = True
    arbiter_paths: list[Path] = []
    failed_path: Path | None = None

    async def provider(cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        nonlocal failed_path
        arbiter = "you are the arbiter" in prompt.lower()
        if arbiter:
            match = re.search(r"listed in (\S*arbiter[-\w]*-input\.json)", prompt)
            assert match is not None
            path = Path(match.group(1))
            arbiter_paths.append(path)
            if fail_docs and any(row["file"] == "README.md" for row in json.loads(path.read_text())):
                failed_path = path
                backend.calls.append({"cwd": cwd, "prompt": prompt, **kwargs})
                raise RuntimeError("docs arbiter provider unavailable")
        async for event in execute(cwd, prompt, *args, **kwargs):
            if arbiter and isinstance(event, ResultEvent):
                assert isinstance(event.structured_output, dict)
                # Keep surviving arbiter targets eligible on resume, so completed
                # group evidence must actually be loaded rather than bypassed.
                event = replace(event, structured_output={"findings": [dict(row, severity="high")
                    for row in event.structured_output["findings"]]})
            yield event

    monkeypatch.setattr(backend, "execute", provider)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    config = empty_review_config(multi_stack_target, tmp_path / "partial-overlap.json", precision_mode=True,
        latency_profile="balanced", review_profile=independent_alternatives_profile(), review_cache_enabled=True)
    async def drive() -> None:
        assert await run(config) == 0

    with anyio.fail_after(60):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(drive)
            await backend.started["arbiter"].wait()
            await backend.started["suppression"].wait()
            live = backend.input_paths["suppression"].parent
            backend.release["arbiter"].set()
            await backend.closed["arbiter"].wait()
            await anyio.wait_all_tasks_blocked()
            assert failed_path is not None
            assert not (live / "adjudication-complete.marker").exists()
            assert not backend.closed["suppression"].is_set()
            backend.release["suppression"].set()

    deep = multi_stack_target / ".daydream/deep"
    first = read_routing_record(deep)["arbiter"]
    assert first["sharded"] is True and len(first["groups"]) == 2
    assert len(first["failed_groups"]) == 1
    failed = first["failed_groups"][0]
    successful = next(group["group_id"] for group in first["groups"] if group["group_id"] != failed)
    assert (deep / f"{successful}-complete.marker").is_file()
    assert not (deep / f"{failed}-complete.marker").exists()
    assert not (deep / "adjudication-complete.marker").exists()
    assert saved_coverage(deep).phases["suppression"]["status"] == "complete"
    assert all(json.loads(path.read_text())["unit"] != "arbiter"
               for path in (deep.parent / "review-cache/entries").glob("*/manifest.json"))
    records = [row for path in deep.glob("stack-*-records.json")
               for row in json.loads(path.read_text())["issues"]]
    assert {row["uid"] for row in records} >= {"generic:1", "python:1", "react:2"}
    assert not {"python:2", "react:1"} & {row["uid"] for row in records}

    fail_docs = False
    arbiter_paths.clear()
    assert await run(replace(config, start_at="merge", trajectory_path=tmp_path / "resumed-overlap.json")) == 0
    assert [path.name for path in arbiter_paths] == [f"{failed}-input.json"]
    resumed = read_routing_record(deep)["arbiter"]
    assert resumed["failed_groups"] == []
    assert {group["group_id"]: group["reused"] for group in resumed["groups"]} == {
        failed: False, successful: True,
    }
    assert (deep / "adjudication-complete.marker").is_file()
    assert saved_coverage(deep).phases["arbiter"]["status"] == "complete"


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
async def test_precision_overlap_closes_both_streams_without_publishing_completion(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str,
) -> None:
    """Cancellation and inherited deadlines must unwind both live provider streams.

    Existing reviewer-cancellation coverage does not enter the new adjudication
    fan-out, where a leaked sibling could continue writing after run finalization.
    """
    backend = PrecisionOverlapBackend(multi_stack_target, blocked=True)
    backend.fanout_concurrency = 2
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    profile = independent_alternatives_profile()
    if stop == "deadline":
        setattr(backend, "supports_budget_preamble", True)
        execute = backend.execute

        async def budget_provider(cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            if not ("you are the arbiter" in prompt.lower()
                    or "you are the suppression reviewer" in prompt.lower()):
                # Only the adjudication fixture consumes live budget fields;
                # its inherited provider for other phases has a smaller API.
                kwargs.pop("wall_budget_s", None)
                kwargs.pop("tool_call_budget", None)
            async for event in execute(cwd, prompt, *args, **kwargs):
                yield event

        monkeypatch.setattr(backend, "execute", budget_provider)
        # Adjudication reserves 60 seconds for finalization. This leaves a real
        # four-second shared investigation allowance without mocking host clocks.
        profile = replace(profile, profile=replace(profile.profile,
            pipeline=replace(profile.profile.pipeline, review_wall_budget_s=64)))
    config = empty_review_config(multi_stack_target, tmp_path / f"{stop}-overlap.json", precision_mode=True,
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
        if stop == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with anyio.fail_after(15):
                assert await task == 0
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
    if stop == "deadline":
        coverage = saved_coverage(deep)
        assert all(coverage.phases[phase]["status"] == "incomplete" for phase in ("arbiter", "suppression"))
        allowances = [call["wall_budget_s"] for call in backend.calls
                      if "you are the arbiter" in call["prompt"].lower()
                      or "you are the suppression reviewer" in call["prompt"].lower()]
        assert len(allowances) == 2 and all(0 < allowance <= 4 for allowance in allowances)
        assert allowances[0] == pytest.approx(allowances[1], abs=0.25)
