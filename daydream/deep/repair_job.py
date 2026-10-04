"""The durable repair job record (issue #1210).

One artifact, one writer, one meaning — the read-modify-write shape of
``deep/routing_record.py``. ``repair-job.json`` is the job's own state: which
execution ran last, what the job has consumed, whether the turn asked for more
authority, and every named degradation of the host's accounting (including a
checkpoint that could not be persisted). A later execution merges into it, so a
resume never erases an earlier execution's evidence.

It is *not* an input to the dispatch decision in this module: the checkpoint
(``phases/repair_checkpoint.py``) owns the captured work, and the coordinator
owns the decision. This record exists so the decision and the outcome survive
the process.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daydream.deep.artifacts import DeepArtifact
from daydream.json_utils import atomic_write_json, read_json_object

#: Bump whenever the job record shape changes so a stale file can never be read
#: as the current contract (mirrors ``EVIDENCE_REUSE_FORMAT``).
REPAIR_JOB_FORMAT: int = 1

#: The job states a record can hold. ``active`` is the only non-terminal one; a
#: terminal state names why the job stopped, in ``last_transition_reason``.
JOB_ACTIVE = "active"


@dataclass(frozen=True)
class RepairJobRecord:
    """What one repair job has run, what it has consumed, and what it still owes.

    ``diagnostics`` and ``no_progress_evidence`` are names, never contents: the
    bounded evidence itself lives in the checkpoint this record points at.
    """

    job_id: str
    state: str = JOB_ACTIVE
    execution_count: int = 0
    checkpoint_ref: str | None = None
    scope_request: Mapping[str, Any] | None = None
    no_progress_evidence: tuple[str, ...] = ()
    last_transition_reason: str | None = None
    consumed_budget: Mapping[str, float] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        """JSON-serializable form carrying the format version."""
        return {
            "format_version": REPAIR_JOB_FORMAT,
            "job_id": self.job_id,
            "state": self.state,
            "execution_count": self.execution_count,
            "checkpoint_ref": self.checkpoint_ref,
            "scope_request": dict(self.scope_request) if self.scope_request is not None else None,
            "no_progress_evidence": list(self.no_progress_evidence),
            "last_transition_reason": self.last_transition_reason,
            "consumed_budget": dict(self.consumed_budget),
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> RepairJobRecord:
        """Rebuild a record from a stored payload; the job id is the only requirement."""
        job_id = raw.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("repair job record is missing its job_id")
        state = raw.get("state")
        scope_request = raw.get("scope_request")
        return cls(
            job_id=job_id,
            state=state if isinstance(state, str) and state else JOB_ACTIVE,
            execution_count=int(raw.get("execution_count") or 0),
            checkpoint_ref=raw.get("checkpoint_ref") if isinstance(raw.get("checkpoint_ref"), str) else None,
            scope_request=dict(scope_request) if isinstance(scope_request, Mapping) else None,
            no_progress_evidence=_str_tuple(raw.get("no_progress_evidence")),
            last_transition_reason=(raw.get("last_transition_reason")
                if isinstance(raw.get("last_transition_reason"), str) else None),
            consumed_budget=_float_map(raw.get("consumed_budget")),
            diagnostics=_str_tuple(raw.get("diagnostics")),
        )


def read_repair_job(deep_dir_path: Path) -> RepairJobRecord | None:
    """Return the stored job record, or ``None`` when absent, foreign, or corrupt.

    A corrupt job record is deliberately not an empty one: the caller decides
    whether an unreadable record blocks recovery, so this returns ``None`` and
    leaves that judgement to the coordinator rather than inventing state.
    """
    raw = read_json_object(DeepArtifact.REPAIR_JOB.at(deep_dir_path))
    if not raw or raw.get("format_version") != REPAIR_JOB_FORMAT:
        return None
    try:
        return RepairJobRecord.from_payload(raw)
    except (TypeError, ValueError):
        return None


def write_repair_job(deep_dir_path: Path, job: RepairJobRecord) -> Path:
    """Replace the job record and return the written path."""
    path = DeepArtifact.REPAIR_JOB.at(deep_dir_path)
    atomic_write_json(path, job.payload(), trailing_newline=True)
    return path


def merge_repair_job(deep_dir_path: Path, updates: Mapping[str, Any]) -> RepairJobRecord:
    """Merge top-level ``updates`` into the job record and persist the result.

    Mappings merge one level deep (as ``write_routing_record`` does) so a later
    execution can add ``consumed_budget`` without dropping an earlier slice.
    Returns the merged record, which is also what the next execution reads.
    """
    stored = read_repair_job(deep_dir_path)
    current = stored.payload() if stored is not None else {}
    # Start from the stored payload so a partial merge keeps every earlier slice.
    merged: dict[str, Any] = {key: value for key, value in current.items() if key != "format_version"}
    merged["format_version"] = REPAIR_JOB_FORMAT
    for key, value in updates.items():
        existing = current.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            nested = dict(existing)
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    job = RepairJobRecord.from_payload(merged)
    write_repair_job(deep_dir_path, job)
    return job


def record_diagnostic(deep_dir_path: Path, job_id: str, diagnostic: str) -> RepairJobRecord | None:
    """Append one named host degradation to the job record; return it, or ``None``.

    Used for failures the job cannot recover from inside its own turn — notably a
    repair checkpoint that could not be persisted. The write is best-effort by
    design only because its *caller* is the blocking outcome: this returns
    ``None`` when even the diagnostic could not be recorded.
    """
    existing = read_repair_job(deep_dir_path)
    diagnostics = (*(existing.diagnostics if existing is not None else ()), diagnostic)
    try:
        return merge_repair_job(deep_dir_path, {"job_id": job_id, "diagnostics": list(diagnostics)})
    except OSError:
        return None


def _str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: float(item)
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, (int, float)) and not isinstance(item, bool)
    }


__all__ = [
    "JOB_ACTIVE",
    "REPAIR_JOB_FORMAT",
    "RepairJobRecord",
    "merge_repair_job",
    "read_repair_job",
    "record_diagnostic",
    "write_repair_job",
]
