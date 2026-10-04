"""The durable repair checkpoint: the only copy of an interrupted repair's work.

A repair turn that stalls or dies still leaves authorized edits on disk, and the
very next step of this phase restores files. Cleanup is therefore the one thing
that can erase the only copy, so the capture is read from the *live tree* at the
capture site inside ``_launch_fix`` — before confinement, before the
generated-file guard, and before any rerun — rather than from ``run_agent``'s
returned partial, which the agent discards at the moment of interruption.

The record carries structured facts, digests, and bounded redacted excerpts only.
It deliberately does **not** carry arbitrary prose, environment values, or an
unbounded command line: that exclusion mirrors ``evidence_reuse.audit_payload``
(``deep/evidence_reuse.py:128``), which records names and digests precisely so a
persisted artifact can never become a credential leak.

Reading is deliberately asymmetric. ``read_json_object`` returns ``{}`` for an
absent, corrupt, and non-object payload alike, so a caller cannot tell "no repair
ever happened" from "the repair record is unreadable". :func:`read_repair_checkpoint`
separates those cases itself and returns a *blocked* result for the second: a
corrupt checkpoint is a recovery blocker, never an empty job.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daydream.deep.artifacts import DeepArtifact
from daydream.json_utils import atomic_write_json
from daydream.phases.repair_outcome import RepairOutcome
from daydream.redaction import redact_text

#: Bump whenever the checkpoint shape changes so a stale file can never be read
#: as the current contract (mirrors ``EVIDENCE_REUSE_FORMAT``).
REPAIR_CHECKPOINT_FORMAT: int = 1

#: A checkpoint outlives the process that wrote it, so a failure excerpt is a
#: redacted, length-capped tail rather than the unbounded output itself.
CHECKPOINT_EXCERPT_MAX_CHARS = 2000


def bounded_result_excerpt(text: str, *, max_chars: int = CHECKPOINT_EXCERPT_MAX_CHARS) -> str:
    """Return a redacted, length-capped tail of *text* (``""`` when there is none)."""
    excerpt = redact_text(text.strip())
    if len(excerpt) > max_chars:
        excerpt = excerpt[-max_chars:]
    return excerpt


def command_label(argv: Sequence[str], *, max_chars: int = CHECKPOINT_EXCERPT_MAX_CHARS) -> str:
    """Return a redacted, length-capped rendering of a test command.

    The command is recorded because a resuming job must know what failed, not
    because its contents are trusted: redaction runs first, and the label is
    capped exactly like any other prose excerpt.
    """
    return bounded_result_excerpt(" ".join(argv), max_chars=max_chars)


def failure_identity(failure_output: str) -> str:
    """Name the failure a repair worked from by digest, never by its raw text."""
    digest = hashlib.sha256(failure_output.encode("utf-8", errors="surrogateescape")).hexdigest()
    return f"sha256:{digest}"


# The bounded follow-up each interrupted turn owes the job that resumes it. These
# are host-authored names, not model prose: the next job decides which to run.
_NEXT_EXPERIMENT: dict[RepairOutcome, str | None] = {
    RepairOutcome.CANDIDATE_COMPLETE: None,
    RepairOutcome.BUDGET_INTERRUPTED: "resume_execution_with_bounded_allowance",
    RepairOutcome.DIAGNOSIS_UNRESOLVED: "continue_diagnosis_from_recorded_excerpt",
    RepairOutcome.EXECUTION_ERROR: "retry_execution_after_transport_failure",
    RepairOutcome.SCOPE_BLOCKED: None,
}


def next_experiment_for(outcome: RepairOutcome) -> str | None:
    """Return the named next experiment for *outcome*, or ``None`` when none applies."""
    return _NEXT_EXPERIMENT[outcome]


@dataclass(frozen=True)
class RepairCheckpoint:
    """One repair turn's authorized work, captured before any restoration ran.

    Every field is a fact the host can read off the tree or its own accounting;
    a value the host cannot produce stays empty rather than being invented, and
    the resuming job is the component that decides what an empty one means.
    """

    job_id: str
    execution_id: str
    #: Authorized-path-only patch, read from the live tree at the capture site.
    candidate_patch: str
    base_tree_key: str = ""
    retained_tree_key: str = ""
    authorized_scope: tuple[str, ...] = ()
    policy_revision: int = 0
    failure_identity: str = ""
    test_command: str = ""
    #: Run-relative to the repository root; the host's absolute cwd is an
    #: environment value and is deliberately not persisted.
    test_cwd: str = "."
    focused_results: tuple[str, ...] = ()
    completed_experiments: tuple[str, ...] = ()
    disproven_hypotheses: tuple[str, ...] = ()
    next_experiment: str | None = None
    backend_name: str = ""
    model: str = ""
    consumed_budget: Mapping[str, float] = field(default_factory=dict)

    @property
    def patch_digest(self) -> str:
        """Content key of the captured patch; a stored digest must match it."""
        return hashlib.sha256(self.candidate_patch.encode("utf-8", errors="surrogateescape")).hexdigest()

    def payload(self) -> dict[str, Any]:
        """JSON-serializable form carrying the format version and the patch digest.

        The digest is derived here rather than stored by the caller, so a record
        cannot claim a digest for a patch it does not carry.
        """
        return {
            "format_version": REPAIR_CHECKPOINT_FORMAT,
            "patch_digest": self.patch_digest,
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "candidate_patch": self.candidate_patch,
            "base_tree_key": self.base_tree_key,
            "retained_tree_key": self.retained_tree_key,
            "authorized_scope": list(self.authorized_scope),
            "policy_revision": self.policy_revision,
            "failure_identity": self.failure_identity,
            "test_command": self.test_command,
            "test_cwd": self.test_cwd,
            "focused_results": list(self.focused_results),
            "completed_experiments": list(self.completed_experiments),
            "disproven_hypotheses": list(self.disproven_hypotheses),
            "next_experiment": self.next_experiment,
            "backend_name": self.backend_name,
            "model": self.model,
            "consumed_budget": dict(self.consumed_budget),
        }

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> RepairCheckpoint:
        """Rebuild a checkpoint from a stored payload, requiring its identity fields.

        ``format_version`` and ``patch_digest`` are deliberately *not* trusted
        here: :func:`read_repair_checkpoint` verifies them before calling this, so
        a record that fails them can never be resurrected as a usable checkpoint.
        """
        job_id = _required_str(raw, "job_id")
        return cls(
            job_id=job_id,
            execution_id=_required_str(raw, "execution_id"),
            # An authorized turn that changed nothing is a real capture, so an
            # empty patch is content -- its absence from the record is not.
            candidate_patch=_text(raw, "candidate_patch"),
            base_tree_key=_str(raw, "base_tree_key"),
            retained_tree_key=_str(raw, "retained_tree_key"),
            authorized_scope=tuple(_str_list(raw, "authorized_scope")),
            policy_revision=int(raw.get("policy_revision") or 0),
            failure_identity=_str(raw, "failure_identity"),
            test_command=_str(raw, "test_command"),
            test_cwd=_str(raw, "test_cwd") or ".",
            focused_results=tuple(_str_list(raw, "focused_results")),
            completed_experiments=tuple(_str_list(raw, "completed_experiments")),
            disproven_hypotheses=tuple(_str_list(raw, "disproven_hypotheses")),
            next_experiment=_str(raw, "next_experiment") or None,
            backend_name=_str(raw, "backend_name"),
            model=_str(raw, "model"),
            consumed_budget=_float_map(raw.get("consumed_budget")),
        )


@dataclass(frozen=True)
class CheckpointRead:
    """The outcome of reading a checkpoint: decoded, absent, or blocked.

    ``blocked`` means a checkpoint *should* exist and cannot be trusted, which
    is a recovery blocker; ``checkpoint is None`` without ``blocked`` means no
    repair was ever captured.
    """

    checkpoint: RepairCheckpoint | None
    blocked: bool
    reason: str | None = None


def write_repair_checkpoint(deep_dir_path: Path, checkpoint: RepairCheckpoint) -> Path:
    """Publish *checkpoint* atomically and verify the read-back; return its path.

    Raises ``OSError`` when the bytes cannot be published and ``ValueError`` when
    they cannot be read back: an unverifiable checkpoint is exactly as
    unrecoverable as an unwritten one, so the caller must treat both as a
    blocking outcome rather than proceeding with no evidence.
    """
    path = DeepArtifact.REPAIR_CHECKPOINT.at(deep_dir_path)
    atomic_write_json(path, checkpoint.payload(), trailing_newline=True)
    written = read_repair_checkpoint(deep_dir_path)
    if written.blocked:
        raise ValueError(f"repair checkpoint could not be verified: {written.reason}")
    if written.checkpoint != checkpoint:
        raise ValueError("repair checkpoint read back with different contents")
    return path


def read_repair_checkpoint(deep_dir_path: Path) -> CheckpointRead:
    """Decode the stored checkpoint, distinguishing absent from untrustworthy."""
    path = DeepArtifact.REPAIR_CHECKPOINT.at(deep_dir_path)
    if not path.exists():
        return CheckpointRead(None, False)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return CheckpointRead(None, True, f"checkpoint malformed: unreadable ({type(exc).__name__}: {exc})")
    if not isinstance(raw, dict):
        return CheckpointRead(None, True, "checkpoint malformed: payload is not a JSON object")
    version = raw.get("format_version")
    if version != REPAIR_CHECKPOINT_FORMAT:
        return CheckpointRead(
            None, True,
            f"checkpoint malformed: unsupported format_version {version!r}",
        )
    try:
        checkpoint = RepairCheckpoint.from_payload(raw)
    except (TypeError, ValueError) as exc:
        return CheckpointRead(None, True, f"checkpoint malformed: {exc}")
    if checkpoint.patch_digest != raw.get("patch_digest"):
        return CheckpointRead(None, True, "checkpoint malformed: patch digest does not match candidate_patch")
    return CheckpointRead(checkpoint, False)


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing or empty required field {key!r}")
    return value


def _text(raw: Mapping[str, Any], key: str) -> str:
    """Return a present string field, empty content included; absent is not empty."""
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"missing or non-text required field {key!r}")
    return value


def _str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    return value if isinstance(value, str) else ""


def _str_list(raw: Mapping[str, Any], key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: float(item)
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, (int, float)) and not isinstance(item, bool)
    }


__all__ = [
    "CHECKPOINT_EXCERPT_MAX_CHARS",
    "REPAIR_CHECKPOINT_FORMAT",
    "CheckpointRead",
    "RepairCheckpoint",
    "bounded_result_excerpt",
    "command_label",
    "failure_identity",
    "next_experiment_for",
    "read_repair_checkpoint",
    "write_repair_checkpoint",
]
