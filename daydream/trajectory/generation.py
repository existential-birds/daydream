"""Provider generation lifecycle and single-owner billing reconciliation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from daydream.backends import CostEvent, MetricsEvent

MAX_PENDING_GENERATION_DRAFTS = 512


MAX_RETAINED_CHOICE_BYTES = 10 * 1024 * 1024


_MAX_NATIVE_UNIX_MS = (2**63 - 1) // 1_000_000


_CAP_DIAGNOSTIC_DRAFTS = f"generation_pending_cap:{MAX_PENDING_GENERATION_DRAFTS}_sealed_drafts"


_CAP_DIAGNOSTIC_BYTES = f"generation_pending_cap:{MAX_RETAINED_CHOICE_BYTES}_retained_choice_bytes"


_CONTRADICTION_DIAGNOSTIC = "billing_contradiction:child_sum_mismatches_terminal_total"


@dataclass
class _GenerationDraft:
    """One provider-generation draft pending billing-ownership resolution."""

    generation_id: str
    sealed: bool = False
    ended: bool = False
    billed: bool = False
    boundary_complete: bool = True
    choice_parts: list[dict[str, Any]] = field(default_factory=list)
    native_started_at_unix_ms: int | None = None
    native_started_at_unix_ns: int | None = None
    sealed_end_unix_ns: int | None = None
    ended_at_unix_ns: int | None = None
    duration_ns: int | None = None
    start_fallback: str | None = None
    response_id: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        if not self.usage:
            record.pop("usage")
        return record


def _usage_dict(event: MetricsEvent | CostEvent) -> dict[str, Any]:
    """Project present usage dimensions without inventing missing values."""
    return {
        key: value
        for key, value in (
            ("input_tokens", event.prompt_tokens if isinstance(event, MetricsEvent) else event.input_tokens),
            ("output_tokens", event.completion_tokens if isinstance(event, MetricsEvent) else event.output_tokens),
            ("cached_tokens", event.cached_tokens),
            ("cost_usd", event.cost_usd),
            ("reasoning_tokens", event.reasoning_tokens),
        )
        if value is not None
    }


class _GenerationLedger:
    """Keep generation choices and timing sealed until invocation-final billing.

    Every terminal path ends drafts exactly once at their historical ends.
    Overflow (512 drafts or 10 MiB) drains retained drafts with a fixed count-only
    diagnostic, locks billing to structural-or-none, and leaves later children unbilled."""

    def __init__(self) -> None:
        self._drafts: dict[str, _GenerationDraft] = {}
        self._sealed_pending = 0
        self._retained_choice_bytes = 0
        self._cap_engaged = False
        self._owner = "unresolved"
        self._resolved = False
        self._authoritative_total: dict[str, Any] | None = None
        self._diagnostics: list[str] = []

    def has_drafts(self) -> bool:
        return bool(self._drafts)

    def open(self, event: Any) -> None:
        generation_id = getattr(event, "generation_id", None)
        if not isinstance(generation_id, str) or not generation_id:
            return
        if generation_id in self._drafts:
            return  # idempotent: a generation opens once
        self._drafts[generation_id] = _GenerationDraft(
            generation_id=generation_id,
            boundary_complete=bool(getattr(event, "boundary_complete", True)),
        )

    def seal(self, event: Any) -> None:
        generation_id = getattr(event, "generation_id", None)
        draft = self._drafts.get(generation_id) if isinstance(generation_id, str) else None
        if draft is None or draft.sealed:
            return
        draft.sealed = True
        draft.boundary_complete = bool(getattr(event, "boundary_complete", True))
        self._seal_timing(draft, event)
        serialized = [asdict(part) for part in (getattr(event, "choice_parts", ()) or ())]
        if self._cap_engaged:
            # Overflow mode: later children stay unbilled/custom and their
            # retained content is never admitted.
            draft.ended = True
            draft.ended_at_unix_ns = draft.sealed_end_unix_ns
            return
        added = sum(len(json.dumps(part, sort_keys=True, separators=(",", ":"))) for part in serialized)
        if self._retained_choice_bytes + added >= MAX_RETAINED_CHOICE_BYTES:
            self._engage_cap(_CAP_DIAGNOSTIC_BYTES)
            return
        draft.choice_parts = serialized
        self._retained_choice_bytes += added
        self._sealed_pending += 1
        if self._sealed_pending > MAX_PENDING_GENERATION_DRAFTS:
            self._engage_cap(_CAP_DIAGNOSTIC_DRAFTS)
            return

    def _seal_timing(self, draft: _GenerationDraft, event: Any) -> None:
        """Freeze strict native timing at the sealed boundary (decision 4)."""
        response_id = getattr(event, "response_id", None)
        model_name = getattr(event, "model_name", None)
        provider_name = getattr(event, "provider_name", None)
        finish_reason = getattr(event, "finish_reason", None)
        draft.response_id = response_id if isinstance(response_id, str) else None
        draft.model_name = model_name if isinstance(model_name, str) else None
        draft.provider_name = provider_name if isinstance(provider_name, str) else None
        draft.finish_reason = finish_reason if isinstance(finish_reason, str) else None
        ended_ns = getattr(event, "ended_at_unix_ns", None)
        if isinstance(ended_ns, bool) or not isinstance(ended_ns, int):
            ended_ns = None
        draft.sealed_end_unix_ns = ended_ns
        native_ms = getattr(event, "native_started_at_unix_ms", None)
        if native_ms is None:
            draft.start_fallback = "missing"
            return
        if type(native_ms) is not int or not 0 <= native_ms <= _MAX_NATIVE_UNIX_MS:
            draft.start_fallback = "invalid"
            return
        native_ns = native_ms * 1_000_000  # exact multiplication, no clamping
        if ended_ns is not None and native_ns > ended_ns:
            # A native start after the sealed host end is an explicit
            # incomplete boundary — never reordered or clamped.
            draft.start_fallback = "reversed"
            return
        draft.native_started_at_unix_ms = native_ms
        draft.native_started_at_unix_ns = native_ns
        if ended_ns is not None:
            draft.duration_ns = ended_ns - native_ns

    def _engage_cap(self, diagnostic: str) -> None:
        if self._cap_engaged:
            return
        self._cap_engaged = True
        self._diagnostics.append(diagnostic)
        self._drain_all(clear_content=True)

    def _drain_all(self, *, clear_content: bool) -> None:
        for draft in self._drafts.values():
            if not draft.ended:
                draft.ended = True
                draft.ended_at_unix_ns = draft.sealed_end_unix_ns
                if clear_content:
                    draft.choice_parts = []
        self._sealed_pending = 0
        self._retained_choice_bytes = 0

    def record_usage(self, event: MetricsEvent) -> None:
        """Record late per-generation usage once, retaining only supplied dimensions."""
        draft = self._drafts.get(event.generation_id or "")
        if draft is not None and not draft.usage:
            draft.usage = _usage_dict(event)

    def record_authoritative_total(self, event: CostEvent) -> None:
        """Record the authoritative terminal total once; contradictions fail closed."""
        incoming = _usage_dict(event)
        if self._authoritative_total is None:
            self._authoritative_total = incoming or None
            return
        # Restated dimensions must match; do not rewrite the stored total.
        if incoming != {key: self._authoritative_total.get(key) for key in incoming}:
            self._diagnostics.append(_CONTRADICTION_DIAGNOSTIC)

    def finalize(self) -> None:
        """Terminal/cancel/error drain: end every draft once, then close ownership."""
        if self._resolved:
            return
        self._drain_all(clear_content=False)
        self._resolve_owner()
        self._resolved = True

    def _resolve_owner(self) -> None:
        total = self._authoritative_total
        if total is None:
            self._owner = "none" if self._drafts else "unresolved"
            return
        self._owner = "structural_attempt"
        drafts = self._drafts.values()
        # Allocation requires complete sealed children with usage and no cap loss.
        if not self._drafts or self._cap_engaged or any(
            not draft.sealed or not draft.boundary_complete or not draft.usage for draft in drafts
        ):
            return
        for dimension in ("input_tokens", "output_tokens"):
            child_total = sum(
                draft.usage[dimension] for draft in drafts if isinstance(draft.usage.get(dimension), int)
            )
            if total.get(dimension) is not None and total[dimension] != child_total:
                # Contradictory evidence bills neither side and remains unchanged.
                self._diagnostics.append(_CONTRADICTION_DIAGNOSTIC)
                self._owner = "none"
                return
        self._owner = "generation_children"
        for draft in drafts:
            draft.billed = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "drafts": [draft.to_dict() for draft in self._drafts.values()],
            "billing_owner": self._owner,
            "resolved": self._resolved,
            "authoritative_total": (dict(self._authoritative_total) if self._authoritative_total else None),
            "diagnostics": list(self._diagnostics),
            "children_after_cap": self._cap_engaged,
        }
