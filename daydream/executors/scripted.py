"""Scripted executor: an in-memory, step-based lifecycle conformance adapter.

``ScriptedExecutor`` is the *second, structurally different* hermetic adapter
for the DAYDREAM_SERVICE_V1 conformance suite. Where :class:`LocalExecutor`
drives lifecycle with a real asyncio task and real on-disk workspace,
``ScriptedExecutor``:

- holds all state in an in-memory **store** (a dict, keyed by opaque handle);
- advances the lifecycle **a step at a time** — each ``inspect`` moves the
  execution one step along its declared *script* (``starting -> running ->
  collecting -> evaluated``) instead of elapsing wall-clock time;
- is fully deterministic and free of I/O, asyncio tasks, and clock dependence.

The store is owned by the caller and can be shared across two instances, which
models the ``restart_reconciliation`` trait: a fresh instance given the same
store can ``inspect`` / ``collect`` / ``release`` a reference the earlier
instance started. Cancellation jumps the script straight to ``cancelled`` and
release removes the entry, modelling ``strong_cancel`` + ``deterministic_release``.

Simulated outcomes come from ``job.payload``:

- ``outcome`` (str): ``clean`` (default), ``findings``, ``infra_error_direct``,
  or ``infra_error_vendor`` (maps a simulated vendor exception to neutral
  ``infra_error``).
- ``lenses`` (list[str]) and ``script`` (list[str], optional) control the lens
  inventory and the step plan walked per inspection.

No vendor types, SDK objects, or handles appear in store state: the store
holds only neutral status strings and lens names.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import MutableMapping
from typing import Any

from daydream.executors.contract import (
    REQUIRED_CAPABILITIES,
    ArtifactEnvelope,
    ExecutionOutcome,
    ExecutionRef,
    ExecutionSnapshot,
    ExecutionStatus,
    ExecutorCapability,
    ExecutorError,
    ExecutorJob,
    UnknownExecutionError,
    map_vendor_error,
    require_capabilities,
)
from daydream.executors.protocol import ReviewExecutor

_DEFAULT_SCRIPT = [
    ExecutionStatus.STARTING,
    ExecutionStatus.RUNNING,
    ExecutionStatus.COLLECTING,
    ExecutionStatus.EVALUATED,
]


class ScriptedExecutor(ReviewExecutor):
    """In-memory, step-based, script-planned conformance executor."""

    kind = "scripted"
    adapter_version = 1
    capabilities: frozenset[ExecutorCapability] = REQUIRED_CAPABILITIES

    def __init__(self, store: MutableMapping[str, dict[str, Any]] | None = None) -> None:
        require_capabilities(set(self.capabilities), kind=self.kind)
        self._store: MutableMapping[str, dict[str, Any]] = store if store is not None else {}

    # -- lifecycle --------------------------------------------------------

    async def start(self, job: ExecutorJob) -> ExecutionRef:
        opaque = self._find_existing(job)
        if opaque is None:
            opaque = uuid.uuid4().hex
        ref = ExecutionRef(
            executor_kind=self.kind,
            adapter_version=self.adapter_version,
            opaque_handle=opaque,
            attempt_id=job.attempt_id,
        )
        if opaque not in self._store:
            script = _coerce_script(job.payload.get("script"))
            self._store[opaque] = {
                "kind": self.kind,
                "adapter_version": self.adapter_version,
                "attempt_id": job.attempt_id,
                "idempotency_key": job.idempotency_key,
                "script_index": 0,
                "script": [s.value for s in script],
                "outcome": None,
                "lenses": [],
                "error": None,
                "payload": dict(job.payload),
            }
        return ref

    async def inspect(self, ref: ExecutionRef) -> ExecutionSnapshot:
        entry = self._entry(ref)
        self._advance(entry)
        return ExecutionSnapshot(
            ref=ref,
            status=ExecutionStatus(entry["script"][min(entry["script_index"], len(entry["script"]) - 1)]),
        )

    async def cancel(self, ref: ExecutionRef) -> None:
        entry = self._entry(ref)
        entry["script"] = [ExecutionStatus.CANCELLED.value]
        entry["script_index"] = 0
        entry["outcome"] = ExecutionOutcome.CANCELLED.value

    async def collect(self, ref: ExecutionRef) -> ArtifactEnvelope:
        entry = self._entry(ref)
        status = ExecutionStatus(entry["script"][entry["script_index"]])
        if status not in _TERMINAL:
            raise ExecutorError(f"cannot collect a non-terminal execution (status={status.value})")
        outcome = ExecutionOutcome(entry["outcome"] or ExecutionOutcome.CANCELLED.value)
        return ArtifactEnvelope(
            ref=ref,
            outcome=outcome,
            completed_lenses=tuple(entry.get("lenses", [])),
            artifact_sha256=_hash_lenses(entry.get("lenses", [])),
        )

    async def release(self, ref: ExecutionRef, disposition: str) -> None:
        self._store.pop(ref.opaque_handle, None)

    # -- script machinery -------------------------------------------------

    def _advance(self, entry: dict[str, Any]) -> None:
        """Advance one step; when reaching the script tail, materialize the outcome."""
        if entry["script_index"] >= len(entry["script"]) - 1:
            return
        entry["script_index"] += 1
        if ExecutionStatus(entry["script"][entry["script_index"]]) == ExecutionStatus.EVALUATED:
            self._materialize_outcome(entry)

    def _materialize_outcome(self, entry: dict[str, Any]) -> None:
        payload = entry.get("payload", {})
        outcome_key = payload.get("outcome", "clean")
        if outcome_key == "infra_error_direct":
            entry["outcome"] = ExecutionOutcome.INFRA_ERROR.value
            entry["script"][entry["script_index"]] = ExecutionStatus.INFRA_ERROR.value
            return
        if outcome_key == "infra_error_vendor":
            exc = RuntimeError("simulated vendor failure")
            neutral = map_vendor_error("scripted adapter vendor simulation", exc)
            entry["error"] = str(neutral)
            entry["outcome"] = ExecutionOutcome.INFRA_ERROR.value
            entry["script"][entry["script_index"]] = ExecutionStatus.INFRA_ERROR.value
            return
        if outcome_key == "findings":
            entry["outcome"] = ExecutionOutcome.FINDINGS.value
        else:
            entry["outcome"] = ExecutionOutcome.CLEAN.value
        entry["lenses"] = _coerce_lenses(payload.get("lenses"))

    def _entry(self, ref: ExecutionRef) -> dict[str, Any]:
        try:
            return self._store[ref.opaque_handle]
        except KeyError:
            raise UnknownExecutionError(f"execution {ref.opaque_handle!r} is unknown to executor 'scripted'") from None

    def _find_existing(self, job: ExecutorJob) -> str | None:
        for opaque, entry in self._store.items():
            if (
                entry.get("attempt_id") == job.attempt_id
                and (not job.idempotency_key or entry.get("idempotency_key") == job.idempotency_key)
            ):
                return opaque
        return None


_TERMINAL = frozenset({ExecutionStatus.EVALUATED, ExecutionStatus.CANCELLED, ExecutionStatus.INFRA_ERROR})


def _coerce_script(raw: object) -> list[ExecutionStatus]:
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        statuses = []
        for item in raw:
            try:
                statuses.append(ExecutionStatus(item))
            except ValueError:
                pass
        if statuses:
            return statuses
    return list(_DEFAULT_SCRIPT)


def _coerce_lenses(raw: object) -> list[str]:
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return [item for item in raw if isinstance(item, str)]
    return ["python"]


def _hash_lenses(lenses: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(lenses)).encode()).hexdigest()
