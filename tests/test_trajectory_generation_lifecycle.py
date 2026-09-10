"""Span lifecycle tests: pending generation drafts, timing, late billing ownership.

P18 Task 2 (binding decisions 1, 4, 5). Exercises the trajectory-layer
pending-generation reducer through the public ``Invocation.observe`` seam
using T1's frozen backend event types: drafts stay UNENDED until billing
ownership resolves, immutable provider choice + timing seal at message_end
before tools, each draft ends exactly once at its sealed historical end,
bounds (512 drafts / 10 MiB retained choice bytes) drain with
structural/unbilled-or-none diagnostics, no age limit rejects the
395.332-second case, terminal/cancel paths drain, native timestamps are
non-bool bounded int ms converted exactly (no clamping, no fake RFC3339),
and the billing owner closes to one of
``unresolved | generation_children | structural_attempt | none`` before any
record is exported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import daydream.trajectory as trajectory_module
from daydream.backends import (
    CostEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    ReasoningChoicePart,
    ResultEvent,
    TextChoicePart,
    ToolCallChoicePart,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.trajectory import DaydreamPhase, Invocation
from tests.harness.trajectory import make_recorder

# Pinned replay constants (Task 0 / resume notes): native request start
# 1788690314289 ms and host provider message-end 1788690709621000000 ns
# = 395.332 seconds. Sanitized values only — no credentials, no real payload.
NATIVE_START_MS = 1788690314289
NATIVE_START_NS = 1788690314289000000
HOST_END_NS = 1788690709621000000
DURATION_NS = 395_332_000_000


def _iq(recorder: Any, phase: DaydreamPhase = DaydreamPhase.REVIEW) -> Invocation:
    """One fresh Invocation bound to a harness recorder (A-invariant: A1)."""
    return Invocation(
        recorder=recorder,
        phase=phase,
        invocation_id=f"inv-{phase.value}",
    )


def _choice_parts() -> tuple[Any, ...]:
    return (
        ReasoningChoicePart(text="THINK_ONE"),
        TextChoicePart(text="TEXT_ONE"),
        ToolCallChoicePart(call_id="call_001", name="read_file", arguments={"path": "src/example.py"}),
    )


def _end_event(
    generation_id: str = "g1",
    *,
    native_start_ms: int | None = NATIVE_START_MS,
    ended_at_ns: int = HOST_END_NS,
    boundary_complete: bool = True,
) -> GenerationEndEvent:
    return GenerationEndEvent(
        generation_id=generation_id,
        native_started_at_unix_ms=native_start_ms,
        ended_at_unix_ns=ended_at_ns,
        end_source="host_observed_message_end",
        choice_parts=_choice_parts(),
        response_id="resp_gen_01",
        model_name="glm-4.6",
        provider_name="nous",
        finish_reason="toolUse",
        boundary_complete=boundary_complete,
    )


def _usage(
    recorder: Any, inv: Invocation, generation_id: str = "g1", *, input_tokens: int = 10, output_tokens: int = 5
) -> None:
    inv.observe(
        MetricsEvent(
            message_id="",
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            cached_tokens=0,
            cost_usd=0.00402781,
            model_name="glm-4.6",
            usage_scope="message",
            measurement_source="turn_end",
            generation_id=generation_id,
        )
    )


def _total(recorder: Any, inv: Invocation, *, input_tokens: int = 10, output_tokens: int = 5) -> None:
    inv.observe(
        CostEvent(
            cost_usd=0.00402781,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=0,
            measurement_source="terminal",
            cost_source="reported",
        )
    )


def _summary(inv: Invocation) -> dict[str, Any]:
    return inv.generation_lifecycle()


class TestPendingDraftLifecycle:
    """Decision 1: drafts stay UNENDED until billing ownership resolves."""

    def test_draft_opens_on_start_and_stays_unended_until_resolution(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        summary = _summary(inv)
        assert summary["drafts"][0]["sealed"] is False
        assert summary["drafts"][0]["ended"] is False
        assert summary["billing_owner"] == "unresolved"

    def test_seal_freezes_choice_and_timing_at_message_end_before_tools(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        # Tool execution starts AFTER message_end; it must not duplicate or
        # alter the sealed choice parts.
        inv.observe(ToolStartEvent(id="call_001", name="read_file", input={"path": "src/example.py"}))
        inv.observe(ToolResultEvent(id="call_001", output='{"content": "FILE"}', is_error=False))
        draft = _summary(inv)["drafts"][0]
        assert draft["sealed"] is True
        assert draft["ended"] is False  # still pending until ownership resolves
        kinds = [part["kind"] for part in draft["choice_parts"]]
        assert kinds == ["reasoning", "text", "tool_call"]
        assert draft["choice_parts"][2]["call_id"] == "call_001"
        assert draft["ended_at_unix_ns"] is None

    def test_end_happens_exactly_once_after_late_usage_and_resolution(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        # Late usage lands after seal but before terminal resolution.
        _usage(recorder, inv)
        _total(recorder, inv)
        inv.finish()
        draft = _summary(inv)["drafts"][0]
        assert draft["ended"] is True
        assert draft["ended_at_unix_ns"] == HOST_END_NS  # sealed historical end
        # Second finish()/drain must not re-end anything.
        inv.finish()
        assert _summary(inv)["drafts"][0]["ended"] is True

    def test_terminal_result_path_drains(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        inv.observe(ResultEvent(structured_output=None, continuation=None))
        inv.finish()
        assert _summary(inv)["drafts"][0]["ended"] is True

    def test_cancel_and_error_paths_drain(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        inv.mark_aborted("wall_budget_exceeded")
        inv.finish()
        assert _summary(inv)["drafts"][0]["ended"] is True

        recorder2 = make_recorder(tmp_path)
        inv2 = _iq(recorder2)
        inv2.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv2.observe(_end_event())
        inv2.mark_errored("error_max_turns")
        inv2.finish()
        assert _summary(inv2)["drafts"][0]["ended"] is True

    def test_no_age_limit_rejects_395_second_case(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        _usage(recorder, inv)
        _total(recorder, inv)
        inv.finish()
        draft = _summary(inv)["drafts"][0]
        assert draft["native_started_at_unix_ns"] == NATIVE_START_NS
        assert draft["sealed_end_unix_ns"] == HOST_END_NS
        assert draft["duration_ns"] == DURATION_NS  # 395.332 s exactly


class TestNativeTimingValidation:
    """Decision 4: non-bool bounded int ms, exact ns conversion, explicit fallbacks."""

    def test_exact_ns_conversion(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        draft = _summary(inv)["drafts"][0]
        assert draft["native_started_at_unix_ms"] == NATIVE_START_MS
        assert draft["native_started_at_unix_ns"] == NATIVE_START_NS

    @pytest.mark.parametrize(
        ("native_start", "fallback"),
        [
            (True, "invalid"),  # bool is not an int timestamp
            (2**63 // 1_000_000 + 1, "invalid"),  # out of int64-ns range
            (-5, "invalid"),
            (None, "missing"),
        ],
    )
    def test_invalid_native_start_falls_back_explicitly(
        self, tmp_path: Path, native_start: int | bool | None, fallback: str
    ) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event(native_start_ms=native_start))
        draft = _summary(inv)["drafts"][0]
        assert draft["start_fallback"] == fallback
        assert draft["native_started_at_unix_ns"] is None  # never clamped
        assert draft["duration_ns"] is None

    def test_reversed_chronology_falls_back_explicitly(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        # Native start AFTER the sealed host end is an explicit incomplete
        # boundary — never reordered, never clamped.
        inv.observe(_end_event(native_start_ms=2_000_000_000, ended_at_ns=1_000_000_000))
        draft = _summary(inv)["drafts"][0]
        assert draft["start_fallback"] == "reversed"
        assert draft["duration_ns"] is None


class TestBillingOwnerResolution:
    """Decision 5: closed owner before export; children|structural|none."""

    def test_zero_children_with_authoritative_total_bills_chain_once(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        # Opaque backends (Claude/Codex/Osprey) emit no generation events; a
        # failed/opaque billed attempt with an authoritative total keeps its
        # structural bill even with zero generations.
        _total(recorder, inv, input_tokens=7, output_tokens=2)
        inv.finish()
        summary = _summary(inv)
        assert summary["billing_owner"] == "structural_attempt"
        assert summary["drafts"] == []

    def test_exact_complete_allocation_bills_children(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        for gid in ("g1", "g2"):
            inv.observe(GenerationStartEvent(generation_id=gid, observed_at_unix_ns=1_000))
            inv.observe(_end_event(generation_id=gid))
        _usage(recorder, inv, "g1", input_tokens=10, output_tokens=5)
        _usage(recorder, inv, "g2", input_tokens=3, output_tokens=2)
        _total(recorder, inv, input_tokens=13, output_tokens=7)
        inv.finish()
        summary = _summary(inv)
        assert summary["billing_owner"] == "generation_children"
        assert [d["billed"] for d in summary["drafts"]] == [True, True]

    def test_partial_children_with_authoritative_total_bills_chain_only(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        for gid in ("g1", "g2"):
            inv.observe(GenerationStartEvent(generation_id=gid, observed_at_unix_ns=1_000))
            inv.observe(_end_event(generation_id=gid))
        # Only g1 has a complete allocation; g2 has none.
        _usage(recorder, inv, "g1", input_tokens=10, output_tokens=5)
        _total(recorder, inv, input_tokens=13, output_tokens=7)
        inv.finish()
        summary = _summary(inv)
        assert summary["billing_owner"] == "structural_attempt"
        assert [d["billed"] for d in summary["drafts"]] == [False, False]

    def test_partial_only_evidence_is_custom_nowhere_billed(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        _usage(recorder, inv)  # child usage, but NO authoritative attempt total
        inv.finish()
        summary = _summary(inv)
        assert summary["billing_owner"] == "none"
        assert summary["drafts"][0]["billed"] is False

    def test_contradictory_totals_fail_closed_without_rewriting(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        _usage(recorder, inv, input_tokens=10, output_tokens=5)
        _total(recorder, inv, input_tokens=13, output_tokens=7)  # contradicts children
        inv.finish()
        summary = _summary(inv)
        assert summary["billing_owner"] == "none"  # fail closed
        assert any("contradiction" in d for d in summary["diagnostics"])
        assert summary["drafts"][0]["billed"] is False

    def test_owner_never_switches_after_resolution(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        for gid in ("g1", "g2"):
            inv.observe(GenerationStartEvent(generation_id=gid, observed_at_unix_ns=1_000))
            inv.observe(_end_event(generation_id=gid))
        _usage(recorder, inv, "g1", input_tokens=10, output_tokens=5)
        _usage(recorder, inv, "g2", input_tokens=3, output_tokens=2)
        inv.finish()
        assert _summary(inv)["billing_owner"] == "none"  # partial-only
        # A late authoritative total must NOT flip the closed owner.
        _total(recorder, inv, input_tokens=13, output_tokens=7)
        assert _summary(inv)["billing_owner"] == "none"
        assert _summary(inv)["drafts"][0]["billed"] is False

    def test_incomplete_boundary_is_never_a_billed_child(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000, boundary_complete=False))
        inv.observe(_end_event(boundary_complete=False))
        _usage(recorder, inv)
        _total(recorder, inv)
        inv.finish()
        summary = _summary(inv)
        assert summary["billing_owner"] == "structural_attempt"
        assert summary["drafts"][0]["billed"] is False

    def test_exact_duplicate_authoritative_total_is_idempotent(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        _usage(recorder, inv, input_tokens=10, output_tokens=5)
        _total(recorder, inv, input_tokens=10, output_tokens=5)
        _total(recorder, inv, input_tokens=10, output_tokens=5)  # exact duplicate: no-op
        inv.finish()
        summary = _summary(inv)
        assert summary["billing_owner"] == "generation_children"
        assert summary["drafts"][0]["billed"] is True
        assert not any("contrad" in d for d in summary["diagnostics"])

    def test_contradictory_duplicate_totals_fail_closed_keep_first(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        _usage(recorder, inv, input_tokens=10, output_tokens=5)
        _total(recorder, inv, input_tokens=13, output_tokens=7)
        _total(recorder, inv, input_tokens=20, output_tokens=7)  # contradictory duplicate
        inv.finish()
        summary = _summary(inv)
        assert summary["billing_owner"] == "none"  # fail closed
        assert any("contrad" in d for d in summary["diagnostics"])
        # First authoritative total is never rewritten.
        assert summary["authoritative_total"]["input_tokens"] == 13
        assert summary["drafts"][0]["billed"] is False


class TestPendingBounds:
    """Decision 1 bounds: 512 drafts / 10 MiB retained choice bytes."""

    def test_draft_count_cap_drains_with_fixed_count_only_diagnostic(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        for i in range(512):
            gid = f"g{i:04d}"
            inv.observe(GenerationStartEvent(generation_id=gid, observed_at_unix_ns=1_000))
            inv.observe(_end_event(generation_id=gid))
        assert _summary(inv)["drafts"][-1]["ended"] is False  # cap not yet hit
        # 513th seal trips the cap: everything drains immediately.
        inv.observe(GenerationStartEvent(generation_id="g512", observed_at_unix_ns=1_000))
        inv.observe(_end_event(generation_id="g512"))
        summary = _summary(inv)
        assert all(d["ended"] for d in summary["drafts"])
        assert all(d["billed"] is False for d in summary["drafts"])
        cap_diags = [d for d in summary["diagnostics"] if "cap" in d]
        assert len(cap_diags) == 1
        assert "512" in cap_diags[0] or cap_diags[0].lower().startswith("generation_pending_cap")
        # Choice content cleared; a later child remains unbilled/custom.
        assert all(d["choice_parts"] == [] for d in summary["drafts"])
        inv.observe(GenerationStartEvent(generation_id="g513", observed_at_unix_ns=1_000))
        inv.observe(_end_event(generation_id="g513"))
        after = _summary(inv)
        assert after["children_after_cap"] is True
        tail = after["drafts"][-1]
        assert tail["ended"] is True
        assert tail["billed"] is False
        assert tail["choice_parts"] == []

    def test_choice_bytes_cap_10_mib(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        huge = TextChoicePart(text="x" * (10 * 1024 * 1024))
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(
            GenerationEndEvent(
                generation_id="g1",
                native_started_at_unix_ms=NATIVE_START_MS,
                ended_at_unix_ns=HOST_END_NS,
                end_source="host_observed_message_end",
                choice_parts=(huge,),
                boundary_complete=True,
            )
        )
        summary = _summary(inv)
        assert summary["drafts"][0]["ended"] is True
        assert summary["drafts"][0]["billed"] is False
        cap_diags = [d for d in summary["diagnostics"] if "cap" in d]
        assert len(cap_diags) == 1


class TestUnbilledOrNoneCapOwner:
    """Cap drain locks ownership to structural (if a later total exists) or none."""

    def test_cap_drain_with_later_authoritative_total_locks_structural(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        for i in range(513):
            gid = f"g{i:04d}"
            inv.observe(GenerationStartEvent(generation_id=gid, observed_at_unix_ns=1_000))
            inv.observe(_end_event(generation_id=gid))
        _total(recorder, inv, input_tokens=1, output_tokens=1)
        inv.finish()
        assert _summary(inv)["billing_owner"] == "structural_attempt"

    def test_cap_drained_drafts_never_bill_even_with_matching_sums(self, tmp_path: Path) -> None:
        """Cap drain locks ownership: matching usage sums must not bill children.

        Post-cap drafts may carry late usage whose input/output sums equal the
        authoritative total. The documented cap invariant (no native bill on
        cap-drained children) must win: the owner stays structural and every
        draft remains custom/unbilled, never ``generation_children``.
        """
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        for i in range(513):
            gid = f"g{i:04d}"
            inv.observe(GenerationStartEvent(generation_id=gid, observed_at_unix_ns=1_000))
            inv.observe(_end_event(generation_id=gid))
            inv.observe(
                MetricsEvent(
                    message_id=gid,
                    prompt_tokens=1,
                    completion_tokens=1,
                    cached_tokens=0,
                    cost_usd=0.001,
                    generation_id=gid,
                )
            )
        _total(recorder, inv, input_tokens=513, output_tokens=513)
        inv.finish()
        summary = _summary(inv)
        assert summary["children_after_cap"] is True
        assert summary["billing_owner"] == "structural_attempt"
        assert all(draft["billed"] is False for draft in summary["drafts"])

    def test_cap_drain_without_total_owner_none(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        for i in range(513):
            gid = f"g{i:04d}"
            inv.observe(GenerationStartEvent(generation_id=gid, observed_at_unix_ns=1_000))
            inv.observe(_end_event(generation_id=gid))
        inv.finish()
        assert _summary(inv)["billing_owner"] == "none"


class TestEventDispatchAndSummary:
    """Dispatch routing, subtrajectory surfacing, no behavior change without generations."""

    def test_no_generation_events_leave_summary_unchanged(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(
            MetricsEvent(
                message_id="m1",
                prompt_tokens=5,
                completion_tokens=3,
                cached_tokens=0,
                cost_usd=0.001,
            )
        )
        inv.observe(TurnEndEvent(message_id="m1"))
        inv.finish()
        summary = _summary(inv)
        assert summary["drafts"] == []
        assert summary["billing_owner"] == "unresolved"
        recorder._register_subtrajectory(inv)
        assert "generation_lifecycle" not in recorder._subtrajectories[-1]

    def test_subtrajectory_summary_surfaces_generation_lifecycle(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        _usage(recorder, inv)
        _total(recorder, inv)
        inv.finish()
        recorder._register_subtrajectory(inv)
        entry = recorder._subtrajectories[-1]["generation_lifecycle"]
        assert entry["billing_owner"] == "generation_children"
        assert entry["drafts"][0]["ended"] is True

    def test_usage_never_invented(self, tmp_path: Path) -> None:
        recorder = make_recorder(tmp_path)
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(_end_event())
        # No usage events at all: the record must not fabricate any numbers.
        inv.finish()
        draft = _summary(inv)["drafts"][0]
        assert "usage" not in draft or draft["usage"] == {}
        assert draft["billed"] is False

    def test_module_symbols_exported(self) -> None:
        assert hasattr(trajectory_module, "MAX_PENDING_GENERATION_DRAFTS")
        assert hasattr(trajectory_module, "MAX_RETAINED_CHOICE_BYTES")
