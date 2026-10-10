"""Generation timing and billing through the public Invocation.observe seam.

Choices seal before tools; drafts end once at historical boundaries after
billing resolves. Tests preserve strict native timestamps, late usage, terminal
and cancellation drains, and the 512-draft/10-MiB fail-closed caps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.backends import (
    CostEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    ReasoningChoicePart,
    TextChoicePart,
    ToolCallChoicePart,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.trajectory import DaydreamPhase, Invocation
from tests.harness.trajectory import make_recorder

# Sanitized replay: native start to sealed host end spans exactly 395.332 seconds.
NATIVE_START_MS = 1788690314289
HOST_END_NS = 1788690709621000000


def _iq(recorder: Any, phase: DaydreamPhase = DaydreamPhase.REVIEW) -> Invocation:
    """One fresh Invocation bound to a harness recorder (A-invariant: A1)."""
    return Invocation(recorder=recorder, phase=phase, invocation_id=f"inv-{phase.value}",)


def _choice_parts() -> tuple[Any, ...]:
    return (ReasoningChoicePart(text="THINK_ONE"), TextChoicePart(text="TEXT_ONE"),
        ToolCallChoicePart(call_id="call_001", name="read_file", arguments={"path": "src/example.py"}),
    )


def _end_event(
    generation_id: str = "g1", *, native_start_ms: int | None = NATIVE_START_MS, ended_at_ns: int = HOST_END_NS,
    boundary_complete: bool = True,
) -> GenerationEndEvent:
    return GenerationEndEvent(
        generation_id=generation_id, native_started_at_unix_ms=native_start_ms, ended_at_unix_ns=ended_at_ns,
        end_source="host_observed_message_end", choice_parts=_choice_parts(), response_id="resp_gen_01",
        model_name="glm-4.6", provider_name="nous", finish_reason="toolUse", boundary_complete=boundary_complete,
    )


def _seal_generations(inv: Invocation, count: int) -> None:
    """Start and seal ``count`` complete generations with deterministic ids."""
    for i in range(count):
        gid = f"g{i:04d}"
        inv.observe(GenerationStartEvent(generation_id=gid, observed_at_unix_ns=1_000))
        inv.observe(_end_event(generation_id=gid))


def _sealed_invocation(tmp_path: Path, *, native_start_ms: int | None = NATIVE_START_MS, ended_at_ns: int = HOST_END_NS,
    boundary_complete: bool = True,
) -> tuple[Any, Invocation]:
    recorder = make_recorder(tmp_path)
    inv = _iq(recorder)
    inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000, boundary_complete=boundary_complete)
    )
    inv.observe(
        _end_event(native_start_ms=native_start_ms, ended_at_ns=ended_at_ns, boundary_complete=boundary_complete)
    )
    return recorder, inv


def _usage(recorder: Any, inv: Invocation, generation_id: str = "g1", *, input_tokens: int = 10, output_tokens: int = 5
) -> None:
    inv.observe(MetricsEvent(
            message_id="", prompt_tokens=input_tokens, completion_tokens=output_tokens, cached_tokens=0,
            cost_usd=0.00402781, model_name="glm-4.6", usage_scope="message", measurement_source="turn_end",
            generation_id=generation_id,
        )
    )


def _total(recorder: Any, inv: Invocation, *, input_tokens: int = 10, output_tokens: int = 5) -> None:
    inv.observe(CostEvent(cost_usd=0.00402781, input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0,
            measurement_source="terminal", cost_source="reported",
        )
    )


def _summary(inv: Invocation) -> dict[str, Any]:
    return inv._generation_ledger.to_dict()

class TestPendingDraftLifecycle:
    """Decision 1: drafts stay UNENDED until billing ownership resolves."""

    def test_draft_opens_on_start_and_stays_unended_until_resolution(self, recorder: Any) -> None:
        inv = _iq(recorder)
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        summary = _summary(inv)
        assert summary["drafts"][0]["sealed"] is False
        assert summary["drafts"][0]["ended"] is False
        assert summary["billing_owner"] == "unresolved"

    def test_seal_freezes_choice_and_timing_at_message_end_before_tools(self, tmp_path: Path) -> None:
        _, inv = _sealed_invocation(tmp_path)
        # Tools cannot alter choices already sealed at message end.
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
        recorder, inv = _sealed_invocation(tmp_path)
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


    def test_cancel_and_error_paths_drain(self, tmp_path: Path) -> None:
        _, inv = _sealed_invocation(tmp_path)
        inv.mark_aborted("wall_budget_exceeded")
        inv.finish()
        assert _summary(inv)["drafts"][0]["ended"] is True

        _, inv2 = _sealed_invocation(tmp_path)
        inv2.mark_errored("error_max_turns")
        inv2.finish()
        assert _summary(inv2)["drafts"][0]["ended"] is True


class TestNativeTimingValidation:
    """Decision 4: non-bool bounded int ms, exact ns conversion, explicit fallbacks."""


    @pytest.mark.parametrize(("native_start", "fallback"),
        [
            (True, "invalid"),  # bool is not an int timestamp
            (2**63 // 1_000_000 + 1, "invalid"),  # out of int64-ns range
            (-5, "invalid"), (None, "missing"),
        ],
    )
    def test_invalid_native_start_falls_back_explicitly(
        self, tmp_path: Path, native_start: int | bool | None, fallback: str
    ) -> None:
        _, inv = _sealed_invocation(tmp_path, native_start_ms=native_start)
        draft = _summary(inv)["drafts"][0]
        assert draft["start_fallback"] == fallback
        assert draft["native_started_at_unix_ns"] is None  # never clamped
        assert draft["duration_ns"] is None

    def test_reversed_chronology_falls_back_explicitly(self, tmp_path: Path) -> None:
        # Reversed native timing remains incomplete, never reordered or clamped.
        _, inv = _sealed_invocation(tmp_path, native_start_ms=2_000_000_000, ended_at_ns=1_000_000_000)
        draft = _summary(inv)["drafts"][0]
        assert draft["start_fallback"] == "reversed"
        assert draft["duration_ns"] is None

class TestBillingOwnerResolution:
    """Decision 5: closed owner before export; children|structural|none."""


    def test_partial_children_with_authoritative_total_bills_chain_only(self, recorder: Any) -> None:
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
        recorder, inv = _sealed_invocation(tmp_path)
        _usage(recorder, inv)  # child usage, but NO authoritative attempt total
        inv.finish()
        summary = _summary(inv)
        assert summary["billing_owner"] == "none"
        assert summary["drafts"][0]["billed"] is False

    def test_contradictory_totals_fail_closed_without_rewriting(self, tmp_path: Path) -> None:
        recorder, inv = _sealed_invocation(tmp_path)
        _usage(recorder, inv, input_tokens=10, output_tokens=5)
        _total(recorder, inv, input_tokens=13, output_tokens=7)  # contradicts children
        inv.finish()
        summary = _summary(inv)
        assert summary["billing_owner"] == "none"  # fail closed
        assert any("contradiction" in d for d in summary["diagnostics"])
        assert summary["drafts"][0]["billed"] is False

    def test_owner_never_switches_after_resolution(self, recorder: Any) -> None:
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
        recorder, inv = _sealed_invocation(tmp_path, boundary_complete=False)
        _usage(recorder, inv)
        _total(recorder, inv)
        inv.finish()
        summary = _summary(inv)
        assert summary["billing_owner"] == "structural_attempt"
        assert summary["drafts"][0]["billed"] is False


    def test_contradictory_duplicate_totals_fail_closed_keep_first(self, tmp_path: Path) -> None:
        recorder, inv = _sealed_invocation(tmp_path)
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

    def test_draft_count_cap_drains_with_fixed_count_only_diagnostic(self, recorder: Any) -> None:
        inv = _iq(recorder)
        _seal_generations(inv, 512)
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

    def test_choice_bytes_cap_10_mib(self, recorder: Any) -> None:
        inv = _iq(recorder)
        huge = TextChoicePart(text="x" * (10 * 1024 * 1024))
        inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000))
        inv.observe(GenerationEndEvent(
                generation_id="g1", native_started_at_unix_ms=NATIVE_START_MS, ended_at_unix_ns=HOST_END_NS,
                end_source="host_observed_message_end", choice_parts=(huge,), boundary_complete=True,
            )
        )
        summary = _summary(inv)
        assert summary["drafts"][0]["ended"] is True
        assert summary["drafts"][0]["billed"] is False
        cap_diags = [d for d in summary["diagnostics"] if "cap" in d]
        assert len(cap_diags) == 1

class TestUnbilledOrNoneCapOwner:
    """Cap drain locks ownership to structural (if a later total exists) or none."""


    def test_cap_drained_drafts_never_bill_even_with_matching_sums(self, recorder: Any) -> None:
        """Cap-drained children stay unbilled even when late usage matches terminal totals."""
        inv = _iq(recorder)
        for i in range(513):
            gid = f"g{i:04d}"
            inv.observe(GenerationStartEvent(generation_id=gid, observed_at_unix_ns=1_000))
            inv.observe(_end_event(generation_id=gid))
            inv.observe(MetricsEvent(
                    message_id=gid, prompt_tokens=1, completion_tokens=1, cached_tokens=0, cost_usd=0.001,
                    generation_id=gid,
                )
            )
        _total(recorder, inv, input_tokens=513, output_tokens=513)
        inv.finish()
        summary = _summary(inv)
        assert summary["children_after_cap"] is True
        assert summary["billing_owner"] == "structural_attempt"
        assert all(draft["billed"] is False for draft in summary["drafts"])

    def test_cap_drain_without_total_owner_none(self, recorder: Any) -> None:
        inv = _iq(recorder)
        _seal_generations(inv, 513)
        inv.finish()
        assert _summary(inv)["billing_owner"] == "none"

class TestEventDispatchAndSummary:
    """Dispatch routing, subtrajectory surfacing, no behavior change without generations."""

    def test_no_generation_events_leave_summary_unchanged(self, recorder: Any) -> None:
        inv = _iq(recorder)
        inv.observe(
            MetricsEvent(message_id="m1", prompt_tokens=5, completion_tokens=3, cached_tokens=0, cost_usd=0.001,)
        )
        inv.observe(TurnEndEvent(message_id="m1"))
        inv.finish()
        summary = _summary(inv)
        assert summary["drafts"] == []
        assert summary["billing_owner"] == "unresolved"
        recorder._register_subtrajectory(inv)
        assert "generation_lifecycle" not in recorder._subtrajectories[-1]


    def test_usage_never_invented(self, tmp_path: Path) -> None:
        _, inv = _sealed_invocation(tmp_path)
        # No usage events at all: the record must not fabricate any numbers.
        inv.finish()
        draft = _summary(inv)["drafts"][0]
        assert "usage" not in draft or draft["usage"] == {}
        assert draft["billed"] is False
