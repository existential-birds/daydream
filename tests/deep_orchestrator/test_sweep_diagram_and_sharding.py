"""Sweep Diagram And Sharding."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from tests.deep_orchestrator.support import (
    _install_uncovered_sweep_stub,
    _root_phase_events,
    _uncovered_sweep_target,
)
from tests.harness.stub_backend import install_stub_backend
from tests.test_deep_orchestrator import (
    MakeConfig,
    Mute,
    _install_model_capturing_stubs,
    _install_stub_backend,
    _prime_merge_resume,
    _profile_with_pipeline,
    _record,
    _run_deep,
    _silence,
)

if TYPE_CHECKING:
    from daydream.review_profile import ResolvedProfile


@pytest.mark.parametrize("failure_mode", ["exception", "missing-output"])
async def test_uncovered_sweep_failure_is_audited_without_claiming_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
    failure_mode: str,
) -> None:
    """A failed sweep leaves a report and artifacts without claiming the file was covered."""
    from daydream.runner import run

    target = _uncovered_sweep_target(tmp_path)
    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_uncovered_sweep_stub(monkeypatch, target)
    if failure_mode == "exception":
        stub.fail_sweep = True
    else:
        stub.sweep_file = "notes.txt"
        stub.sweep_no_output = True
        stub.merge_echo_records = True

    assert await run(make_config(target, assume="yes", output_mode="loop")) == 0
    deep = target / ".daydream" / "deep"
    stats = json.loads((deep / "coverage-stats.json").read_text())
    assert stats["attempted_files"] == ["notes.txt"]
    assert stats["completed_files"] == []
    assert stats["sweep_finding_count"] == 0
    assert "notes.txt" in stats["sweep_failures"]
    if failure_mode == "missing-output":
        assert stats["sweep_failures"] == {"notes.txt": "no structured output produced"}
    assert (deep / "merged-items.json").is_file()
    assert not (deep / "stack-uncovered-records.json").exists()

    report = (target / ".review-output.md").read_text()
    assert "## Coverage" in report
    assert "Second-pass sweep covered" not in report
    assert "Best-effort sweep failures: notes.txt" in report
    if failure_mode == "exception":
        uncovered = [
            event for event in _root_phase_events(target, "deep") if event.get("metadata") == {"stage": "uncovered"}
        ]
        assert len(uncovered) == 2
        assert uncovered[-1]["status"] == "failed"
        assert uncovered[-1]["reason_code"] == "all_children_failed"
        trajectory = json.loads(
            next((target / ".daydream" / "runs").glob("*/trajectory.json")).read_text(encoding="utf-8")
        )
        dispatch = next(
            step
            for step in trajectory["steps"]
            if step.get("extra", {}).get("daydream_phase") == "deep"
            and "dispatch_id" in step.get("extra", {})
            and any(result["content"] == "Dispatched to deep-uncovered-0" for result in step["observation"]["results"])
        )
        assert dispatch["extra"]["dispatch_status"] == "failed"
        assert dispatch["extra"]["reason_code"] == "all_children_failed"


async def test_uncovered_sweep_disabled_by_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """A profile pipeline with ``uncovered_sweep_enabled = false`` skips the sweep entirely."""
    from daydream.runner import run

    target = _uncovered_sweep_target(tmp_path)
    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_uncovered_sweep_stub(monkeypatch, target)
    stub.sweep_file = "notes.txt"
    stub.merge_echo_records = True

    exit_code = await run(
        make_config(
            target,
            assume="yes",
            output_mode="loop",
            review_profile=_profile_with_pipeline(uncovered_sweep_enabled=False),
        )
    )
    assert exit_code == 0

    deep = target / ".daydream" / "deep"
    assert not (deep / "coverage-stats.json").exists()
    assert not (deep / "stack-uncovered-records.json").exists()
    merged_items = json.loads((deep / "merged-items.json").read_text())
    assert all(item.get("file") != "notes.txt" for item in merged_items["items"])


async def test_uncovered_sweep_noop_on_merge_resume(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``--start-at merge`` resume is a sweep no-op: no sweep artifacts are
    created and the resume still succeeds.
    """
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    _prime_merge_resume(
        multi_stack_target,
        python=[_record(confidence="HIGH")],
        react=[_record()],
        generic=[_record()],
        structure=[_record()],
    )

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0
    deep = multi_stack_target / ".daydream" / "deep"
    assert not (deep / "stack-uncovered-records.json").exists()
    assert not (deep / "coverage-stats.json").exists()


async def test_uncovered_sweep_per_stack_resume_clears_stale_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``--start-at per-stack`` resume drops the prior run's sweep artifacts."""
    target = _uncovered_sweep_target(tmp_path)
    _silence(monkeypatch)

    stub = _install_uncovered_sweep_stub(monkeypatch, target)
    stub.sweep_file = "notes.txt"
    stub.merge_echo_records = True
    assert await _run_deep(target) == 0

    deep = target / ".daydream" / "deep"
    assert (deep / "stack-uncovered-records.json").is_file()
    assert (deep / "coverage-stats.json").is_file()
    assert list(deep.glob("uncovered-*-review.md"))

    stub2 = _install_uncovered_sweep_stub(monkeypatch, target)
    stub2.merge_echo_records = True
    assert (
        await _run_deep(
            target,
            start_at="per-stack",
            review_profile=_profile_with_pipeline(uncovered_sweep_enabled=False),
        )
        == 0
    )

    assert not (deep / "stack-uncovered-records.json").exists()
    assert not (deep / "coverage-stats.json").exists()
    assert not list(deep.glob("uncovered-*-review.md"))


async def test_uncovered_sweep_per_stack_resume_no_findings_writes_empty_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-stack resume whose rerun sweep produces no findings writes ``[]``."""
    target = _uncovered_sweep_target(tmp_path)
    _silence(monkeypatch)

    stub = _install_uncovered_sweep_stub(monkeypatch, target)
    stub.sweep_file = "notes.txt"
    stub.merge_echo_records = True
    assert await _run_deep(target) == 0

    deep = target / ".daydream" / "deep"
    assert json.loads((deep / "stack-uncovered-records.json").read_text())

    stub2 = _install_uncovered_sweep_stub(monkeypatch, target)
    stub2.fail_sweep = True  # the rerun sweep attempts and fails -> no findings
    stub2.merge_echo_records = True
    assert await _run_deep(target, start_at="per-stack") == 0

    assert json.loads((deep / "stack-uncovered-records.json").read_text()) == []


async def test_run_deep_uncovered_sweep_review_without_read_not_claimed_as_covered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """A successful sweep review WITHOUT a file Read is an attempt, not coverage (issue #309 finding 6)."""
    from daydream.runner import run

    target = _uncovered_sweep_target(tmp_path)
    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_uncovered_sweep_stub(monkeypatch, target)
    stub.sweep_file = "notes.txt"
    stub.sweep_no_read = True  # review written, but no Read tool call emitted
    stub.merge_echo_records = True

    exit_code = await run(make_config(target, assume="yes", output_mode="loop"))
    assert exit_code == 0

    deep = target / ".daydream" / "deep"
    stats = json.loads((deep / "coverage-stats.json").read_text())
    pre_sweep = stats["pre_sweep"]
    assert pre_sweep["uncovered_files"] == ["notes.txt"]
    # The review output WAS written -> the file is a completed ATTEMPT.
    assert stats["completed_files"] == ["notes.txt"]
    # But no verified completed read of the file -> never "covered".
    assert stats["covered_files"] == []
    assert stats["sweep_attempt_status"] == {"notes.txt": "reviewed (hunks only)"}
    # The coverage numbers are unchanged by the hunk-only review.
    assert stats["post_sweep"]["files_read_by_reviewers"] == 3
    assert stats["post_sweep"]["coverage_ratio"] == pre_sweep["coverage_ratio"] == 0.75

    report = (target / ".review-output.md").read_text()
    assert "## Coverage" in report
    assert "Second-pass sweep covered" not in report
    assert "Second-pass sweep reviewed (hunks only): notes.txt" in report
    assert "Files read by reviewers: 3" in report
    assert "Coverage ratio: 0.75" in report


async def test_run_deep_uncovered_sweep_structured_output_without_review_file_is_merged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Valid structured sweep output is authoritative without a Markdown sidecar."""
    from daydream.runner import run

    target = _uncovered_sweep_target(tmp_path)
    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_uncovered_sweep_stub(monkeypatch, target)
    stub.sweep_file = "notes.txt"
    stub.sweep_no_review_file = True
    stub.sweep_no_read = True
    stub.merge_echo_records = True

    exit_code = await run(make_config(target, assume="yes", output_mode="loop"))
    assert exit_code == 0

    deep = target / ".daydream" / "deep"
    assert not list(deep.glob("uncovered-*-review.md"))
    records = json.loads((deep / "stack-uncovered-records.json").read_text())
    assert [record["file"] for record in records] == ["notes.txt"]
    merged = json.loads((deep / "merged-items.json").read_text())["items"]
    assert any(item.get("file") == "notes.txt" for item in merged)

    stats = json.loads((deep / "coverage-stats.json").read_text())
    assert stats["completed_files"] == ["notes.txt"]
    assert stats["covered_files"] == []
    assert stats["sweep_attempt_status"] == {"notes.txt": "reviewed (hunks only)"}
    assert stats["sweep_finding_count"] == 1
    assert stats["sweep_failures"] == {}


async def test_uncovered_sweep_per_stack_resume_fails_closed_on_unremovable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-stack resume whose stale sweep artifact cannot be removed STOPS with an actionable error instead of
    continuing (issue #309 finding 8)."""
    target = _uncovered_sweep_target(tmp_path)
    _silence(monkeypatch)

    stub = _install_uncovered_sweep_stub(monkeypatch, target)
    stub.sweep_file = "notes.txt"
    stub.merge_echo_records = True
    assert await _run_deep(target) == 0

    deep = target / ".daydream" / "deep"
    assert (deep / "stack-uncovered-records.json").is_file()

    # Make one artifact unremovable: a directory sharing the artifact name makes
    # Path.unlink() raise IsADirectoryError, and the cleanup must fail closed.
    (deep / "coverage-stats.json").unlink()
    (deep / "coverage-stats.json").mkdir()

    stub2 = _install_stub_backend(monkeypatch, target)
    exit_code = await _run_deep(target, start_at="per-stack")
    assert exit_code == 1
    # The resume aborted BEFORE new per-stack work, so no backend call ran and
    # the unremovable artifact is still present (it was NOT silently skipped).
    assert stub2.calls == []
    assert (deep / "coverage-stats.json").is_dir()


async def test_uncovered_sweep_malformed_stats_does_not_fail_run(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structurally-malformed ``coverage-stats.json`` must NOT abort the run (issue #309 finding 9)."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    deep = _prime_merge_resume(
        multi_stack_target,
        python=[_record(confidence="HIGH")],
        react=[_record()],
        generic=[_record()],
        structure=[_record()],
    )
    # Malformed shape: valid JSON but a list, so stats.get(...) would raise
    # AttributeError without the shape guard.
    (deep / "coverage-stats.json").write_text("[]")

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    report = (multi_stack_target / ".review-output.md").read_text()
    assert "## Coverage" not in report


def test_diagram_step_position_and_phase_key() -> None:
    """Issue #1113: ``diagram`` sits between ``supervise`` and ``findings-out``."""
    from daydream.deep.orchestrator import DIAGRAM_STEPS, STEPS

    names = [step.name for step in STEPS]
    assert names.index("supervise") + 1 == names.index("diagram")
    assert names.index("diagram") + 1 == names.index("findings-out")
    steps = {step.name: step for step in STEPS}
    assert steps["diagram"].phase_key == "diagram"

    # ``post-diagram`` must NOT be in STEPS: ``_register_builtin_flows``
    # derives the deep flow definition from it, so a GitHub write would be
    # spliced into every deep review.
    assert "post-diagram" not in names
    assert [step.name for step in DIAGRAM_STEPS] == ["post-diagram"]
    assert DIAGRAM_STEPS[0].phase_key == "post-diagram"


def test_diagram_flow_is_registered_with_its_three_steps() -> None:
    """The ``diagram`` flow reuses the deep flow's exploration + diagram steps."""
    from daydream.extensions import Registry
    from daydream.extensions.builtins import register_builtins

    registry = Registry()
    register_builtins(registry)
    assert sorted(registry.flow_names()) == ["deep", "diagram", "improve"]
    assert registry.flow("diagram") == ["exploration", "diagram", "post-diagram"]


def test_resolve_mode_maps_diagram_output_mode() -> None:
    """``--diagram-only`` resolves to the ``diagram`` mode and its own flow."""
    from daydream.deep.orchestrator import (
        _flow_kind_for_mode,
        _flow_name_for_mode,
        _resolve_mode,
    )
    from daydream.runner import RunConfig
    from daydream.trajectory import DaydreamRunFlow

    config = RunConfig(target="/tmp", output_mode="diagram", diagram="sequence")
    assert _resolve_mode(config) == "diagram"
    assert _flow_name_for_mode("diagram") == "diagram"
    assert _flow_kind_for_mode("diagram") is DaydreamRunFlow.DIAGRAM
    # ``--shallow`` must not win over an explicit diagram-only request.
    shallow = RunConfig(target="/tmp", output_mode="diagram", diagram="both", shallow=True)
    assert _resolve_mode(shallow) == "diagram"
    for mode in ("loop", "comment", "review", "shallow"):
        assert _flow_name_for_mode(mode) == "deep"


def test_uncovered_sweep_step_resolves_via_parse_phase_key() -> None:
    """The sweep step registers ``config_phase="parse"`` (docs/extensions.md)."""
    from daydream.deep.orchestrator import STEPS

    steps = {s.name: s for s in STEPS}
    assert steps["uncovered-sweep"].phase_key == "parse"
    assert steps["per-stack-parse"].phase_key == "parse"


def test_uncovered_sweep_enabled_resolution(tmp_path: Path) -> None:
    """The sweep toggle resolves from the profile pipeline (M8), not config tiers."""
    from daydream.deep.orchestrator import _uncovered_sweep_enabled
    from daydream.extensions import Registry
    from daydream.flows.engine import FlowContext
    from daydream.runner import RunConfig
    from daydream.workspace import WorkContext

    def _ctx(config: RunConfig, review_profile: "ResolvedProfile | None" = None) -> FlowContext:
        work = WorkContext(
            repo=tmp_path,
            source=tmp_path,
            base_branch="main",
            base_sha="",
            head_branch=None,
            head_sha="",
            is_ephemeral=False,
            run_id="test",
        )
        return FlowContext(
            config=config,
            work=work,
            registry=Registry(),
            review_profile=review_profile,
            data={},
        )

    # Default profile pipeline (uncovered_sweep_enabled True) -> on.
    assert _uncovered_sweep_enabled(_ctx(RunConfig(target=str(tmp_path)))) is True
    # A profile disabling uncovered_sweep_enabled -> off.
    off = _profile_with_pipeline(uncovered_sweep_enabled=False)
    assert _uncovered_sweep_enabled(_ctx(RunConfig(target=str(tmp_path)), review_profile=off)) is False
    # Merge/fix resumes always disable the sweep.
    assert _uncovered_sweep_enabled(_ctx(RunConfig(target=str(tmp_path), start_at="merge"))) is False


def test_uncovered_sweep_numeric_resolution_reads_pipeline(tmp_path: Path) -> None:
    """The sweep numeric caps resolve from the profile pipeline (already host-clamped)."""
    from daydream.deep.review_steps import _uncovered_sweep_max_files, _uncovered_sweep_min_hunk_lines
    from daydream.extensions import Registry
    from daydream.flows.engine import FlowContext
    from daydream.runner import RunConfig
    from daydream.workspace import WorkContext

    work = WorkContext(
        repo=tmp_path,
        source=tmp_path,
        base_branch="main",
        base_sha="",
        head_branch=None,
        head_sha="",
        is_ephemeral=False,
        run_id="test",
    )

    profile = _profile_with_pipeline(uncovered_sweep_max_files=3, uncovered_sweep_min_hunk_lines=6)
    ctx = FlowContext(
        config=RunConfig(target=str(tmp_path)),
        work=work,
        registry=Registry(),
        review_profile=profile,
        data={},
    )
    assert _uncovered_sweep_max_files(ctx) == 3
    assert _uncovered_sweep_min_hunk_lines(ctx) == 6

    # Clamped pipeline values (review_profile._parse_pipeline HOST_CAPS) are
    # what the orchestrator sees: a sub-floor max_files is clamped up to 1.
    clamped = _profile_with_pipeline(uncovered_sweep_max_files=0)
    ctx_clamped = FlowContext(
        config=RunConfig(target=str(tmp_path)),
        work=work,
        registry=Registry(),
        review_profile=clamped,
        data={},
    )
    assert _uncovered_sweep_max_files(ctx_clamped) == 1


def test_deep_shard_enabled_default_off(tmp_path: Path) -> None:
    """Sharding is forensic-off by default: DEFAULT_DEEP_SHARD_ENABLED = False."""
    from daydream.config import DEFAULT_DEEP_SHARD_ENABLED
    from daydream.deep.orchestrator import _deep_shard_enabled
    from daydream.runner import RunConfig

    assert DEFAULT_DEEP_SHARD_ENABLED is False
    # Default off (forensic mode): no RunConfig attr, no file config.
    cfg = RunConfig(target=str(tmp_path))
    assert _deep_shard_enabled(cfg) is False
    # Large diffs opt into the existing sharder unless a caller explicitly
    # selected a value.
    large_diff = "+line\n" * 6_500
    assert _deep_shard_enabled(cfg, diff=large_diff) is True
    assert _deep_shard_enabled(cfg, diff="+line\n" * 1_500) is False
    # RunConfig True (highest tier) enables.
    cfg = RunConfig(target=str(tmp_path), deep_shard_enabled=True)
    assert _deep_shard_enabled(cfg) is True
    # File-config True with no RunConfig override enables.
    from daydream.config_file import DaydreamFileConfig

    fc = DaydreamFileConfig(deep_shard_enabled=True)
    cfg = RunConfig(target=str(tmp_path), file_config=fc)
    assert _deep_shard_enabled(cfg) is True
    # Explicit RunConfig False (highest tier) force-off a file-config-enabled repo.
    cfg = RunConfig(target=str(tmp_path), file_config=fc, deep_shard_enabled=False)
    assert _deep_shard_enabled(cfg) is False
    assert _deep_shard_enabled(cfg, diff=large_diff) is False


def test_deep_shard_max_files_resolves_and_coerces(tmp_path: Path) -> None:
    """The per-shard file bound resolves with RunConfig > file-config > default
    and degrades malformed ints to the named default (never raises)."""
    from daydream.config import DEFAULT_DEEP_SHARD_MAX_FILES
    from daydream.config_file import DaydreamFileConfig
    from daydream.deep.orchestrator import _deep_shard_max_files
    from daydream.runner import RunConfig

    # Default.
    cfg = RunConfig(target=str(tmp_path))
    assert _deep_shard_max_files(cfg) == DEFAULT_DEEP_SHARD_MAX_FILES

    # RunConfig int override wins.
    cfg = RunConfig(target=str(tmp_path), deep_shard_max_files=5)
    assert _deep_shard_max_files(cfg) == 5

    # Float must degrade, not raise.
    cfg = RunConfig(
        target=str(tmp_path),
        deep_shard_max_files=2.5,  # type: ignore[arg-type]
    )
    assert _deep_shard_max_files(cfg) == DEFAULT_DEEP_SHARD_MAX_FILES

    # File-config override applies when no RunConfig attr is set.
    fc = DaydreamFileConfig(deep_shard_max_files=7)
    cfg = RunConfig(target=str(tmp_path), file_config=fc)
    assert _deep_shard_max_files(cfg) == 7


def test_deep_shard_default_bounds_align_with_inline_budget() -> None:
    """Issue #740: the default shard bounds retune to 5 files / 12288 bytes, and
    the byte bound equals INLINE_DIFF_BUDGET_BYTES so shards inline by construction."""
    from daydream.config import DEFAULT_DEEP_SHARD_MAX_BYTES, DEFAULT_DEEP_SHARD_MAX_FILES
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES

    assert DEFAULT_DEEP_SHARD_MAX_FILES == 5
    assert DEFAULT_DEEP_SHARD_MAX_BYTES == INLINE_DIFF_BUDGET_BYTES  # == 12_288


async def test_deep_sweep_skips_inline_grounded_file_when_enabled(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    install_backend: Callable[[object], object],
) -> None:
    """Issue #731: an inline-grounded, finding-referenced file is NOT swept."""
    from daydream.runner import RunConfig, run

    install_stub_backend(monkeypatch, multi_stack_target)
    exit_code = await run(
        RunConfig(
            target=str(multi_stack_target),
            cleanup=False,
            deep_shard_enabled=True,
            deep_shard_max_files=100,
            deep_shard_max_bytes=10**9,
        )
    )
    assert exit_code == 0
    deep = multi_stack_target / ".daydream" / "deep"
    stats = json.loads((deep / "coverage-stats.json").read_text())
    pre_sweep = stats["pre_sweep"]
    # The gate's outcome: api.py was inline-grounded AND referenced by a
    # parsed finding, so it is not uncovered and never swept.
    assert "api.py" not in pre_sweep["uncovered_files"]
    # The gate is selective: App.tsx is inlined too, but no parsed finding
    # references it, so it stays uncovered (evidence requires a finding match,
    # never assignment alone).
    assert "App.tsx" in pre_sweep["uncovered_files"]
    # Per-evidence counts are populated only when the receipts were written
    # and consumed; broken plumbing yields no inline evidence at all.
    assert pre_sweep["coverage_by_evidence"]["inline_hunk_reviewed"] == 1


async def test_deep_default_run_coverage_by_evidence_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Issue #740 AC2/AC3: coverage_by_evidence is non-empty on a DEFAULT (non-sharded) run."""
    from daydream.runner import run

    target = _uncovered_sweep_target(tmp_path)
    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, target)
    stub.per_stack_emit_reads = True
    stub.per_stack_unread = frozenset({"notes.txt"})
    stub.sweep_file = "notes.txt"

    exit_code = await run(make_config(target, assume="yes", output_mode="loop"))
    assert exit_code == 0

    deep = target / ".daydream" / "deep"
    # The receipt file exists on a default (non-sharded) run.
    assert (deep / "coverage-receipts.json").is_file()
    stats = json.loads((deep / "coverage-stats.json").read_text())
    cbe = stats["pre_sweep"]["coverage_by_evidence"]
    assert cbe  # present and non-empty on a default run
    assert cbe["inline_hunk_reviewed"] >= 1  # api.py read -> clean/has_findings -> credited


async def test_deep_large_diff_produces_review_and_record_shards(
    shard_many_python_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    install_backend: Callable[[object], object],
) -> None:
    """A large Python stack produces multiple review and records artifacts."""
    from daydream.runner import RunConfig, run

    install_stub_backend(monkeypatch, shard_many_python_target)
    # Sharding enabled, tiny file bound -> the python stack shards.
    rc = await run(
        RunConfig(
            target=str(shard_many_python_target),
            cleanup=False,
            deep_shard_enabled=True,
            deep_shard_max_files=1,
            deep_shard_max_bytes=10**9,
        )
    )
    assert rc == 0
    deep = shard_many_python_target / ".daydream" / "deep"
    shards = sorted(p for p in deep.glob("stack-python#*-review.md"))
    assert len(shards) >= 2  # >1 review task for one language stack
    records = sorted(p for p in deep.glob("stack-python#*-records.json"))
    assert records


async def test_deep_forensic_mode_keeps_single_agent_per_stack(
    shard_many_python_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    install_backend: Callable[[object], object],
) -> None:
    """Issue #731: forensic (default off) keeps exactly one agent per stack."""
    from daydream.runner import RunConfig, run

    install_stub_backend(monkeypatch, shard_many_python_target)
    rc = await run(
        RunConfig(
            target=str(shard_many_python_target),
            cleanup=False,
            deep_shard_enabled=False,  # default / forensic
            deep_shard_max_files=1,
        )
    )  # bound ignored when off
    assert rc == 0
    deep = shard_many_python_target / ".daydream" / "deep"
    assert not list(deep.glob("stack-python#*-review.md"))  # exactly one agent per stack, as today


async def test_clean_verdict_on_unread_file_is_not_reviewed_not_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC4: per-stack reviewer declares clean for an assigned file it never Read -> that file is recorded
    not_reviewed, never a pass; the assigned and Read file stays clean."""
    from daydream.runner import run

    target = _uncovered_sweep_target(tmp_path)
    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, target)
    # Emit reads for one file, and a parse that declares clean verdicts for
    # two files (one of which is never Read).
    stub.per_stack_emit_reads = True
    stub.per_stack_unread = frozenset({"notes.txt"})  # notes.txt is never Read
    stub.parse_declared_verdicts = [
        {"path": "api.py", "lines_read": 10, "verdict": "clean"},
        {"path": "notes.txt", "lines_read": 8, "verdict": "clean"},
    ]
    # The stub's default parse emits an api.py finding for every stack; the
    # gate's ordering is finding-beats-read, so a python-stack api.py finding
    # would make the api.py verdict has_findings and mask the read-clean
    # assertion under test. Route the python parse's finding off api.py so the
    # read+clean verdict for api.py is observable.
    stub.parse_by_stack = {
        "python": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "App.tsx",
            "description": "python-stack finding routed off api.py",
        },
    }
    exit_code = await run(make_config(target, assume="yes", output_mode="loop"))
    assert exit_code == 0

    deep = target / ".daydream" / "deep"
    # The reconciled per-stack record must not record notes.txt as clean.
    rec = json.loads((deep / "stack-python-records.json").read_text())
    verdicts = {v["path"]: v for v in rec.get("verdicts", [])}
    assert verdicts["api.py"]["verdict"] == "clean"  # the read file stays clean
    assert verdicts["api.py"]["lines_read"] == 10  # declared payload survived reconciliation
    # notes.txt lives in the generic stack's scope; it was declared clean but
    # never Read, so the gate must downgrade it to not_reviewed -- and the
    # declared lines_read payload is preserved on the downgraded entry (the
    # reviewer said it read 8 lines; the gate records it was not read).
    rec_generic = json.loads((deep / "stack-generic-records.json").read_text())
    verdicts_generic = {v["path"]: v for v in rec_generic.get("verdicts", [])}
    assert verdicts_generic["notes.txt"]["verdict"] == "not_reviewed"  # read-gated, never clean
    assert verdicts_generic["notes.txt"]["lines_read"] == 8
    assert verdicts_generic["README.md"]["verdict"] == "clean"  # read, though not declared


async def test_per_stack_prompt_points_at_hunk_index(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reviewer prompts source changed-line ranges from the hunk index, not
    a diff.patch re-read (AC#1)."""
    _silence(monkeypatch)
    prompts = _install_model_capturing_stubs(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    prompt = next(c["prompt"] for c in prompts if "Relevant diff hunks" in c["prompt"])
    assert "hunk-index.json" in prompt or "changed line ranges" in prompt.lower()
    assert "do NOT re-Read diff.patch" in prompt or "diff.patch" not in prompt


async def test_no_parse_phase_and_records_from_output_schema(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC4: no parse-<stack> fork exists; records come from output_schema."""
    _silence(monkeypatch)
    prompts = _install_model_capturing_stubs(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # (1) No parse-<stack> phase: the phase_parse_feedback prompt never fires.
    assert not any("extract only actionable issues" in c["prompt"].lower() for c in prompts), (
        "a parse phase prompt fired; parse-* stage must be removed"
    )
    assert not any("parse" in str(c.get("model", "")).lower() for c in prompts)

    # (2) Per-stack records files exist and hold the reviewer's structured
    # output. Each lens is checked against its own stub wording ("Sample issue"
    # for the language stacks, a distinct description for the structural
    # meta-stack), so the assertion cannot depend on glob ordering.
    deep_dir_path = multi_stack_target / ".daydream" / "deep"
    records = sorted(deep_dir_path.glob("stack-*-records.json"))
    assert records, "expected per-stack records written by the reviewer"
    language = [p for p in records if p.name != "stack-structure-records.json"]
    assert language, "expected at least one language-stack records file"
    loaded = json.loads(language[0].read_text())
    assert loaded["issues"][0]["description"] == "Sample issue"
    structural = deep_dir_path / "stack-structure-records.json"
    assert structural.is_file(), "expected the structural meta-stack records file"
    assert json.loads(structural.read_text())["issues"][0]["description"] == "Structural maintainability concern"


def test_structural_gate_resolver_reads_profile_pipeline() -> None:
    """The structural gate's pre-context resolver reads the profile flag."""
    from daydream.deep.orchestrator import _config_pipeline
    from daydream.runner import RunConfig

    assert _config_pipeline(RunConfig(target="/tmp/x")).structural_enabled is True
    off = _profile_with_pipeline(structural_enabled=False)
    assert _config_pipeline(RunConfig(target="/tmp/x", review_profile=off)).structural_enabled is False


def test_uncovered_sweep_gate_reads_profile_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.deep import orchestrator as o
    from daydream.flows.engine import FlowContext

    class _Ctx:
        class _Cfg:
            start_at = "review"

        class _P:
            uncovered_sweep_enabled = False

        config = _Cfg()

        def pipeline(self) -> Any:
            return _Ctx._P()

    assert o._uncovered_sweep_enabled(cast(FlowContext, _Ctx())) is False
