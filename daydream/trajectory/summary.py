"""One checked pass over a frozen run's selected documents, reduced to compact summaries.

Each document is decoded once through its checked read, reduced to its metadata
(everything but ``steps``) plus neutral step facts, and released before the next
read. Timing, evaluation and the archive manifest share one summary; nothing here
is persisted or recorded live.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from daydream.timeutil import parse_iso_timestamp
from daydream.trajectory.layout import PARTIAL_SUFFIX, RUN_DOCUMENT_NAME
from daydream.trajectory.types import RunWriteSnapshot


@dataclass(frozen=True)
class StepFacts:
    """Compact facts gathered from one document's ``steps``.

    The ``*_result`` and ``unpaired_call`` facts correlate tool calls with observation
    results within one step. Gathering never raises: ``unreadable`` names each part
    (``count``, ``tools``, ``timing``, ``content``) that a reader iterating the full
    steps would fail on.
    """

    count: int = 0
    tool_names: dict[str, int] = field(default_factory=dict)
    duration_seconds: float | None = None
    tool_calls: int = 0
    has_reasoning: bool = False
    reasoning_chars: int = 0
    agent_message_chars: int = 0
    blank_result: bool = False
    error_result: bool = False
    interrupted_result: bool = False
    unpaired_call: bool = False
    orphan_result: bool = False
    diagnostic_codes: frozenset[str] = frozenset()
    unreadable: frozenset[str] = frozenset()

    def require(self, part: str) -> StepFacts:
        """Return these facts, or raise where reading the full steps for *part* would fail."""
        if part in self.unreadable:
            raise ValueError(f"trajectory steps are malformed for {part}")
        return self


@dataclass(frozen=True)
class DocumentSummary:
    """One selected document: identity, logical source name, metadata without ``steps``, and step facts."""

    trajectory_id: str
    source_file: str
    metadata: dict[str, Any]
    steps: StepFacts


@dataclass(frozen=True)
class RunSummary:
    """Compact summaries of one frozen run snapshot, in snapshot document order."""

    status: Literal["complete", "partial"]
    cutoff_at: str
    root_trajectory_id: str
    documents: tuple[DocumentSummary, ...]

    @property
    def root(self) -> DocumentSummary:
        return next(document for document in self.documents if document.trajectory_id == self.root_trajectory_id)

    @property
    def forked(self) -> tuple[DocumentSummary, ...]:
        return tuple(document for document in self.documents if document.trajectory_id != self.root_trajectory_id)


def _count(steps: Any) -> dict[str, Any]:
    return {"count": len(steps)}


def _tool_names(steps: Any) -> dict[str, Any]:
    names = Counter(call["function_name"] for step in steps for call in step.get("tool_calls") or [])
    if not all(isinstance(name, str) for name in names):
        raise TypeError("tool function names must be strings")
    return {"tool_names": dict(names)}


def _timing(steps: Any) -> dict[str, Any]:
    stamps = [parse_iso_timestamp(step["timestamp"]) for step in steps if step.get("timestamp")]
    return {"duration_seconds": (stamps[-1] - stamps[0]).total_seconds() if len(stamps) >= 2 else None}


def _content(steps: Any) -> dict[str, Any]:
    has_reasoning = any(step.get("reasoning_content") for step in steps)
    tool_calls = sum(len(step.get("tool_calls") or []) for step in steps)
    reasoning_chars = sum(len(step.get("reasoning_content") or "") for step in steps)
    message_chars = sum(
        len(step["message"]) for step in steps
        if step.get("source") == "agent" and isinstance(step.get("message"), str)
    )
    blank_result = False
    for step in steps:
        observation = step.get("observation")
        if observation:
            for result in observation.get("results", []):
                content = result.get("content", "")
                if isinstance(content, str) and content.strip() == "":
                    blank_result = True
                    break
    return {"tool_calls": tool_calls, "has_reasoning": has_reasoning, "reasoning_chars": reasoning_chars,
            "agent_message_chars": message_chars, "blank_result": blank_result, **_pairing(steps)}


def _pairing(steps: Any) -> dict[str, Any]:
    error = interrupted = unpaired = orphan = False
    codes: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        call_counts: Counter[str] = Counter()
        malformed_calls = 0
        raw_calls = step.get("tool_calls")
        if isinstance(raw_calls, list):
            for call in raw_calls:
                call_id = call.get("tool_call_id") if isinstance(call, dict) else None
                if isinstance(call_id, str) and call_id:
                    call_counts[call_id] += 1
                else:
                    malformed_calls += 1

        paired_counts: Counter[str] = Counter()
        observation = step.get("observation")
        raw_results = observation.get("results") if isinstance(observation, dict) else None
        if isinstance(raw_results, list):
            for result in raw_results:
                if not isinstance(result, dict):
                    continue
                source_call_id = result.get("source_call_id")
                raw_extra = result.get("extra")
                extra = raw_extra if isinstance(raw_extra, dict) else {}
                marked_interrupted = extra.get("status") == "interrupted" or extra.get("cancelled") is True
                if isinstance(source_call_id, str) and source_call_id:
                    if source_call_id not in call_counts:
                        orphan = True
                        continue
                    paired_counts[source_call_id] += 1
                    if marked_interrupted:
                        interrupted = True
                    elif extra.get("is_error") is True:
                        error = True
                elif marked_interrupted and (call_counts or malformed_calls):
                    # Recorder-generated interruption markers carry a null source id;
                    # they mark this step's unpaired calls rather than an orphan result.
                    interrupted = True

        if malformed_calls or any(paired_counts[call_id] < count for call_id, count in call_counts.items()):
            unpaired = True

        step_extra = step.get("extra")
        if not isinstance(step_extra, dict):
            continue
        unmatched = step_extra.get("unmatched_tool_results")
        if isinstance(unmatched, list) and unmatched:
            orphan = True
        diagnostics = step_extra.get("backend_diagnostics")
        if isinstance(diagnostics, list):
            codes.update(
                code for diagnostic in diagnostics
                if isinstance(diagnostic, dict) and isinstance(code := diagnostic.get("code"), str)
            )
    return {"error_result": error, "interrupted_result": interrupted, "unpaired_call": unpaired,
            "orphan_result": orphan, "diagnostic_codes": frozenset(codes)}


_STEP_PARTS: tuple[tuple[str, Callable[[Any], dict[str, Any]]], ...] = (
    ("count", _count), ("tools", _tool_names), ("timing", _timing), ("content", _content),
)


def _step_facts(steps: Any) -> StepFacts:
    facts: dict[str, Any] = {}
    unreadable: set[str] = set()
    for part, gather in _STEP_PARTS:
        try:
            facts.update(gather(steps))
        except Exception:  # noqa: BLE001 - only a reader of this part fails on malformed steps
            unreadable.add(part)
    return StepFacts(**facts, unreadable=frozenset(unreadable))


def summarize_run(write_snapshot: RunWriteSnapshot) -> RunSummary:
    """Read each selected document once through its identity-checked read and summarize it.

    Undecodable, non-object or misidentified documents and a missing or duplicate
    root raise ``ValueError``; a changed reference raises its integrity error.
    """
    root_id = write_snapshot.root_trajectory_id
    documents: list[DocumentSummary] = []
    for document in write_snapshot.documents:
        metadata = document.validated_payload(root_id)
        source_file = (RUN_DOCUMENT_NAME if document.trajectory_id == root_id
                       else document.path.name.removesuffix(PARTIAL_SUFFIX))
        documents.append(
            DocumentSummary(document.trajectory_id, source_file, metadata, _step_facts(metadata.pop("steps", [])))
        )
    roots = sum(summary.trajectory_id == root_id for summary in documents)
    if roots != 1:
        raise ValueError("duplicate frozen root trajectory" if roots else "frozen root trajectory is missing")
    return RunSummary(write_snapshot.status, write_snapshot.cutoff_at, root_id, tuple(documents))
