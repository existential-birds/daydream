"""Generation timing and billing through the public Invocation.observe seam.

Choices seal before tools; drafts end once at historical boundaries after
billing resolves. Tests preserve strict native timestamps, late usage, terminal
and cancellation drains, and the 512-draft/10-MiB fail-closed caps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from daydream.backends import (
    CostEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    ReasoningChoicePart,
    TextChoicePart,
    ToolCallChoicePart,
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


def _end_event(generation_id: str = "g1", *, boundary_complete: bool = True) -> GenerationEndEvent:
    return GenerationEndEvent(
        generation_id=generation_id, native_started_at_unix_ms=NATIVE_START_MS, ended_at_unix_ns=HOST_END_NS,
        end_source="host_observed_message_end", choice_parts=_choice_parts(), response_id="resp_gen_01",
        model_name="glm-4.6", provider_name="nous", finish_reason="toolUse", boundary_complete=boundary_complete,
    )


def _seal_generations(inv: Invocation, count: int) -> None:
    """Start and seal ``count`` complete generations with deterministic ids."""
    for i in range(count):
        gid = f"g{i:04d}"
        inv.observe(GenerationStartEvent(generation_id=gid, observed_at_unix_ns=1_000))
        inv.observe(_end_event(generation_id=gid))


def _sealed_invocation(tmp_path: Path, *, boundary_complete: bool = True,
) -> tuple[Any, Invocation]:
    recorder = make_recorder(tmp_path)
    inv = _iq(recorder)
    inv.observe(GenerationStartEvent(generation_id="g1", observed_at_unix_ns=1_000, boundary_complete=boundary_complete)
    )
    inv.observe(_end_event(boundary_complete=boundary_complete))
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


class TestBillingOwnerResolution:
    """Decision 5: closed owner before export; children|structural|none."""


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
        inv.finish()
        assert _summary(inv)["billing_owner"] == "none"

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
