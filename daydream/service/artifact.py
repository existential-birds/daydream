"""Strictly-passive worker artifact envelope (WorkerArtifactV1).

The worker's *output* contract: every observable outcome of a service review
turn — terminal state, completed/missing lenses, process outcome, findings,
artifact hashes, timestamps — carried as a frozen, validated object.

It is STRICTLY passive. It carries no Sprite/Coder/pod/VM/lease and no
worker-asserted infrastructure identity (no executor kind, no opaque handle,
no attempt binding). Those live in the controller's separate ``ExecutionRef``,
owned by another Plan-008 leaf. Nothing in this module shells out, reads git,
or reaches the network.

Fail-closed invariants (enforced in ``__post_init__`` and guaranteed by the
differentiated constructors):

* ``terminal == "clean"`` requires empty ``missing_lenses`` and no blocking
  findings.
* Any blocking finding (high/medium severity) keeps ``terminal == "findings"``
  even when the process exited 0.
* ``terminal == "findings"`` requires at least one blocking finding.
* ``terminal == "infra_error"`` is never producible for a clean/missing-free
  process outcome — it must name a real failure.
* ``terminal == "cancelled"`` requires ``process_outcome == "cancelled"``.
* Findings are bounded, strict (allowed keys and types only), and homogeneous
  (every finding carries the identical key set).
* ``hashes`` are digests, never pointers to infrastructure (URLs are rejected).

Exports:
    MAX_FINDINGS: Hard cap on the findings tuple.
    WorkerArtifactV1: The passive envelope with differentiated constructors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from daydream.service.models import ReviewJobV1

MAX_FINDINGS = 200
_MAX_HASHES = 64
_MAX_TIMESTAMPS = 64

_TERMINALS: frozenset[str] = frozenset({"clean", "findings", "infra_error", "cancelled"})

# Documented process-outcome vocabulary plus the service-leaf extensions. A
# value outside this set is rejected so the envelope can never claim a
# process state the worker cannot produce.
_PROCESS_OUTCOMES: frozenset[str] = frozenset(
    {
        "exited_0",
        "exited_nonzero",
        "budget_exhausted",
        "parse_loss",
        "process_loss",
        "mutation_detected",
        "cancelled",
        "tool_vetoed",
        "incomplete_lenses",
        "lens_unavailable",
        "findings_overflow",
        "git_preflight_failed",
        "state_capture_failed",
    }
)

# Findings that block a clean approval (severity policy for this leaf).
BLOCKING_SEVERITIES: frozenset[str] = frozenset({"high", "medium"})

_FINDING_FIELDS: frozenset[str] = frozenset(
    {"id", "lens", "file", "line", "severity", "confidence", "title", "body"}
)
_FINDING_SEVERITIES: frozenset[str] = frozenset({"high", "medium", "low"})
_FINDING_CONFIDENCES: frozenset[str] = frozenset({"HIGH", "MEDIUM"})


def _blocking_findings(findings: tuple[dict[str, Any], ...]) -> bool:
    return any(f.get("severity") in BLOCKING_SEVERITIES for f in findings)


def _validate_findings(findings: tuple[dict[str, Any], ...]) -> None:
    """Enforce bounded, strict, homogeneous findings."""
    if len(findings) > MAX_FINDINGS:
        raise ValueError(f"findings exceeds the {MAX_FINDINGS}-entry bound")
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("every finding must be a dict")
        keys = set(finding)
        if keys != _FINDING_FIELDS:
            raise ValueError(
                f"finding must carry exactly {sorted(_FINDING_FIELDS)}, got {sorted(keys)}"
            )
        if not isinstance(finding["id"], int):
            raise ValueError("finding 'id' must be an integer")
        if not isinstance(finding["lens"], str) or not finding["lens"]:
            raise ValueError("finding 'lens' must be a non-empty string")
        if not isinstance(finding["file"], str):
            raise ValueError("finding 'file' must be a string")
        line = finding["line"]
        if line is not None and not isinstance(line, int):
            raise ValueError("finding 'line' must be an integer or null")
        if finding["severity"] not in _FINDING_SEVERITIES:
            raise ValueError(f"finding 'severity' must be one of {sorted(_FINDING_SEVERITIES)}")
        if finding["confidence"] not in _FINDING_CONFIDENCES:
            raise ValueError(
                f"finding 'confidence' must be one of {sorted(_FINDING_CONFIDENCES)}"
            )
        if not isinstance(finding["title"], str):
            raise ValueError("finding 'title' must be a string")
        if not isinstance(finding["body"], str):
            raise ValueError("finding 'body' must be a string")


def _validate_string_map(
    mapping: dict[str, str],
    name: str,
    cap: int,
    *,
    iso: bool = False,
    pointer_free: bool = False,
) -> None:
    if len(mapping) > cap:
        raise ValueError(f"{name} exceeds the {cap}-entry bound")
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError(f"{name}[{key!r}] must be a string")
        if iso:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{name}[{key!r}] is not an ISO 8601 timestamp") from exc
        if pointer_free and ("://" in value or value.startswith("/")):
            raise ValueError(f"{name}[{key!r}] must be a digest, never an infra pointer")


def _validate_lens_sets(
    completed_lenses: tuple[str, ...], missing_lenses: tuple[str, ...]
) -> None:
    for lens in (*completed_lenses, *missing_lenses):
        if not isinstance(lens, str) or not lens:
            raise ValueError("lens names must be non-empty strings")
    if set(completed_lenses) & set(missing_lenses):
        raise ValueError("completed_lenses and missing_lenses must be disjoint")


@dataclass(frozen=True)
class WorkerArtifactV1:
    """Strictly-passive validated outcome of one service review turn.

    Attributes:
        job_id: Job identifier from the immutable job.
        idempotency_key: Dedup key from the immutable job.
        terminal: ``"clean"``, ``"findings"``, ``"infra_error"``, or
            ``"cancelled"``.
        completed_lenses: Subset of required lenses that completed with output.
        missing_lenses: Subset of required lenses that did not complete.
        process_outcome: One of the documented process-outcome vocabulary.
        findings: Bounded, strict, homogeneous finding dicts.
        hashes: Bounded artifact digests (e.g. lens -> digest); never pointers
            to infrastructure.
        timestamps: Bounded ISO 8601 UTC timestamps.
    """

    job_id: str
    idempotency_key: str
    terminal: str
    completed_lenses: tuple[str, ...]
    missing_lenses: tuple[str, ...]
    process_outcome: str | None
    findings: tuple[dict[str, Any], ...]
    hashes: dict[str, str]
    timestamps: dict[str, str]

    def __post_init__(self) -> None:
        if self.terminal not in _TERMINALS:
            raise ValueError(f"unknown terminal {self.terminal!r}")
        if self.process_outcome is not None and self.process_outcome not in _PROCESS_OUTCOMES:
            raise ValueError(f"unknown process_outcome {self.process_outcome!r}")
        _validate_lens_sets(self.completed_lenses, self.missing_lenses)
        _validate_findings(self.findings)
        _validate_string_map(self.hashes, "hashes", _MAX_HASHES, pointer_free=True)
        _validate_string_map(self.timestamps, "timestamps", _MAX_TIMESTAMPS, iso=True)

        blocking = _blocking_findings(self.findings)
        if self.terminal == "clean":
            if self.missing_lenses:
                raise ValueError("clean terminal requires empty missing_lenses")
            if blocking:
                raise ValueError("clean terminal cannot carry blocking findings")
            if self.process_outcome not in (None, "exited_0"):
                raise ValueError("clean terminal cannot report a failed process")
        elif self.terminal == "findings":
            if not blocking:
                raise ValueError("findings terminal requires at least one blocking finding")
            if self.process_outcome not in (None, "exited_0"):
                raise ValueError("findings terminal cannot report a failed process")
        elif self.terminal == "infra_error":
            if self.process_outcome in (None, "exited_0"):
                raise ValueError("infra_error terminal must name a real failure")
        elif self.terminal == "cancelled":
            if self.process_outcome != "cancelled":
                raise ValueError("cancelled terminal requires process_outcome='cancelled'")

    # --- Differentiated constructors -----------------------------------------

    @classmethod
    def complete(
        cls,
        job: ReviewJobV1,
        *,
        completed_lenses: tuple[str, ...],
        findings: tuple[dict[str, Any], ...],
        process_outcome: str = "exited_0",
        hashes: dict[str, str] | None = None,
        timestamps: dict[str, str] | None = None,
    ) -> WorkerArtifactV1:
        """Build the terminal for a fully-completed turn.

        Blocking findings (high/medium) keep ``terminal="findings"`` even when
        the process exited 0; a missing-free run with no blocking findings is
        ``terminal="clean"``. Never produces ``infra_error``.
        """
        terminal = "findings" if _blocking_findings(findings) else "clean"
        return cls(
            job_id=job.job_id,
            idempotency_key=job.idempotency_key,
            terminal=terminal,
            completed_lenses=tuple(completed_lenses),
            missing_lenses=(),
            process_outcome=process_outcome,
            findings=tuple(findings),
            hashes=dict(hashes or {}),
            timestamps=dict(timestamps or {}),
        )

    @classmethod
    def infra_error(
        cls,
        job: ReviewJobV1,
        *,
        process_outcome: str,
        completed_lenses: tuple[str, ...] = (),
        missing_lenses: tuple[str, ...] = (),
        findings: tuple[dict[str, Any], ...] = (),
        hashes: dict[str, str] | None = None,
        timestamps: dict[str, str] | None = None,
    ) -> WorkerArtifactV1:
        """Build a fail-closed ``infra_error`` terminal naming a real failure."""
        return cls(
            job_id=job.job_id,
            idempotency_key=job.idempotency_key,
            terminal="infra_error",
            completed_lenses=tuple(completed_lenses),
            missing_lenses=tuple(missing_lenses),
            process_outcome=process_outcome,
            findings=tuple(findings),
            hashes=dict(hashes or {}),
            timestamps=dict(timestamps or {}),
        )

    @classmethod
    def cancelled(
        cls,
        job: ReviewJobV1,
        *,
        completed_lenses: tuple[str, ...] = (),
        missing_lenses: tuple[str, ...] = (),
        hashes: dict[str, str] | None = None,
        timestamps: dict[str, str] | None = None,
    ) -> WorkerArtifactV1:
        """Build the ``cancelled`` terminal (distinct from a fault)."""
        return cls(
            job_id=job.job_id,
            idempotency_key=job.idempotency_key,
            terminal="cancelled",
            completed_lenses=tuple(completed_lenses),
            missing_lenses=tuple(missing_lenses),
            process_outcome="cancelled",
            findings=(),
            hashes=dict(hashes or {}),
            timestamps=dict(timestamps or {}),
        )

    # --- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "job_id": self.job_id,
            "idempotency_key": self.idempotency_key,
            "terminal": self.terminal,
            "completed_lenses": list(self.completed_lenses),
            "missing_lenses": list(self.missing_lenses),
            "process_outcome": self.process_outcome,
            "findings": [dict(f) for f in self.findings],
            "hashes": dict(self.hashes),
            "timestamps": dict(self.timestamps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerArtifactV1:
        """Deserialize strictly: unknown fields raise, and every invariant in
        ``__post_init__`` re-runs on the decoded values."""
        unknown = set(data) - _ARTIFACT_KEYS
        if unknown:
            raise ValueError(f"unexpected fields on WorkerArtifactV1: {sorted(unknown)}")
        return cls(
            job_id=data["job_id"],
            idempotency_key=data["idempotency_key"],
            terminal=data["terminal"],
            completed_lenses=tuple(data["completed_lenses"]),
            missing_lenses=tuple(data["missing_lenses"]),
            process_outcome=data.get("process_outcome"),
            findings=tuple(dict(f) for f in data["findings"]),
            hashes=dict(data["hashes"]),
            timestamps=dict(data["timestamps"]),
        )


_ARTIFACT_KEYS = frozenset(WorkerArtifactV1.__dataclass_fields__)

__all__ = ["BLOCKING_SEVERITIES", "MAX_FINDINGS", "WorkerArtifactV1"]
