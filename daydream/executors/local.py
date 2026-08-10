"""Hermetic Local executor: real filesystem workspace + time-based async lifecycle.

``LocalExecutor`` is the in-repo reference conformance adapter for
DAYDREAM_SERVICE_V1 and the development/test executor.

- Each execution gets its own workspace directory under a shared *root*,
  giving it the ``exclusive_workspace`` trait with genuinely separate on-disk
  state.
- Lifecycle is driven by an asyncio task that performs simulated review work
  (a bounded delay + artifact write) and advances the neutral status through
  ``starting -> running -> evaluated`` (or a terminal side exit).
- State is persisted to disk, so a *fresh* ``LocalExecutor`` pointed at the
  same root can ``inspect`` / ``collect`` / ``release`` a reference started by
  an earlier instance — the ``restart_reconciliation`` trait proven against
  real durable storage rather than faked in memory.
- ``release`` deterministically deletes the execution's workspace in a fixed
  order (artifacts then state then directory), modelling ``deterministic_release``.

This adapter is development/test infrastructure. It is NOT automatically safe
for merge-authorizing untrusted code: nothing here sandboxes ambient
credentials, source writes, or egress. Use it for conformance and local
experimentation only.

Simulated outcomes are driven by ``job.payload`` keys:

- ``outcome`` (str): ``clean`` (default), ``findings``,
  ``infra_error_vendor`` (raises a simulated vendor exception that the adapter
  maps to neutral ``infra_error``), or ``infra_error_direct``.
- ``lenses`` (list[str]): lens names recorded in the artifact envelope.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import uuid
from pathlib import Path
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
    ExecutorInfrastructureError,
    ExecutorJob,
    UnknownExecutionError,
    map_vendor_error,
    require_capabilities,
)
from daydream.executors.protocol import ReviewExecutor

_STATE_FILE = "state.json"
_ARTIFACT_DIR = "artifacts"
_IDEMPOTENCY_FILE = "idempotency.json"
_TERMINAL = frozenset(
    {ExecutionStatus.EVALUATED, ExecutionStatus.CANCELLED, ExecutionStatus.INFRA_ERROR, ExecutionStatus.RELEASED}
)


def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _coerce_lenses(raw: object) -> list[str]:
    """Coerce ``job.payload["lenses"]`` into a str list, defaulting to ``["python"]``."""
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return [item for item in raw if isinstance(item, str)]
    return ["python"]


class LocalExecutor(ReviewExecutor):
    """Filesystem-backed, time-based conformance/local executor."""

    kind = "local"
    adapter_version = 1
    capabilities: frozenset[ExecutorCapability] = REQUIRED_CAPABILITIES

    def __init__(self, root: Path, *, work_seconds: float = 0.02) -> None:
        require_capabilities(set(self.capabilities), kind=self.kind)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.work_seconds = work_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._shutdown = False

    # -- lifecycle --------------------------------------------------------

    async def start(self, job: ExecutorJob) -> ExecutionRef:
        self._ensure_online()
        existing = self._lookup_by_key(job)
        if existing is not None:
            return existing
        opaque = uuid.uuid4().hex
        ref = ExecutionRef(
            executor_kind=self.kind,
            adapter_version=self.adapter_version,
            opaque_handle=opaque,
            attempt_id=job.attempt_id,
        )
        state: dict[str, Any] = {
            "status": ExecutionStatus.STARTING.value,
            "outcome": None,
            "lenses": [],
            "payload": dict(job.payload),
        }
        self._write_state(ref, state)
        self._remember_key(job, opaque)
        task = asyncio.create_task(self._run(ref, job))
        self._tasks[opaque] = task
        return ref

    async def inspect(self, ref: ExecutionRef) -> ExecutionSnapshot:
        state = self._read_state(ref)
        status = ExecutionStatus(state["status"])
        return ExecutionSnapshot(
            ref=ref,
            status=status,
            started_at_iso=_iso_now() if status in (ExecutionStatus.STARTING, ExecutionStatus.RUNNING) else None,
            completed_at_iso=_iso_now() if status in _TERMINAL else None,
        )

    async def cancel(self, ref: ExecutionRef) -> None:
        self._ensure_online()
        state = self._read_state(ref)
        current = ExecutionStatus(state["status"])
        if current in _TERMINAL:
            return
        task = self._tasks.get(ref.opaque_handle)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        state = {"status": ExecutionStatus.CANCELLED.value, "outcome": ExecutionOutcome.CANCELLED.value, "lenses": []}
        self._write_state(ref, state)

    async def collect(self, ref: ExecutionRef) -> ArtifactEnvelope:
        state = self._read_state(ref)
        status = ExecutionStatus(state["status"])
        if status not in _TERMINAL:
            raise ExecutorError(f"cannot collect a non-terminal execution (status={status.value})")
        outcome = ExecutionOutcome(state["outcome"] or ExecutionOutcome.CANCELLED.value)
        lens_dir = self._exec_dir(ref) / _ARTIFACT_DIR
        hashes = []
        if lens_dir.is_dir():
            for path in sorted(lens_dir.iterdir()):
                if path.is_file():
                    hashes.append(_sha256(path.read_bytes()))
        sha = hashlib.sha256("\n".join(hashes).encode()).hexdigest() if hashes else None
        return ArtifactEnvelope(
            ref=ref,
            outcome=outcome,
            completed_lenses=tuple(state.get("lenses", [])),
            artifact_sha256=sha,
        )

    async def release(self, ref: ExecutionRef, disposition: str) -> None:
        self._ensure_online()
        state = self._read_state(ref)
        state["status"] = ExecutionStatus.RELEASED.value
        self._write_state(ref, state)
        # Deterministic cleanup order: artifacts, then state, then the dir itself.
        self._remove_execution_dir(ref)
        self._tasks.pop(ref.opaque_handle, None)

    # -- simulated work ---------------------------------------------------

    async def _run(self, ref: ExecutionRef, job: ExecutorJob) -> None:
        state = self._read_state(ref)
        final: dict[str, Any] = {"lenses": [], "outcome": ExecutionOutcome.CLEAN.value}
        try:
            state["status"] = ExecutionStatus.RUNNING.value
            self._write_state(ref, state)
            await asyncio.sleep(self.work_seconds)
            outcome_key = job.payload.get("outcome", "clean")
            if outcome_key == "infra_error_vendor":
                try:
                    raise RuntimeError("simulated vendor failure")
                except RuntimeError as exc:
                    neutral = map_vendor_error("local adapter vendor simulation", exc)
                    raise ExecutorInfrastructureError(str(neutral), vendor_cause=exc) from exc
            if outcome_key == "infra_error_direct":
                raise ExecutorInfrastructureError("simulated direct infra error")
            if outcome_key == "findings":
                final["outcome"] = ExecutionOutcome.FINDINGS.value
            else:
                final["outcome"] = ExecutionOutcome.CLEAN.value
            final["lenses"] = _coerce_lenses(job.payload.get("lenses"))
            self._write_artifacts(ref, final["lenses"])
            final["status"] = ExecutionStatus.EVALUATED.value
        except asyncio.CancelledError:
            raise
        except ExecutorInfrastructureError as exc:
            final["status"] = ExecutionStatus.INFRA_ERROR.value
            final["outcome"] = ExecutionOutcome.INFRA_ERROR.value
            state["error"] = str(exc)
        self._write_state(ref, final | {"payload": dict(job.payload)})

    # -- storage helpers --------------------------------------------------

    def _exec_dir(self, ref: ExecutionRef) -> Path:
        return self.root / ref.opaque_handle

    def _read_state(self, ref: ExecutionRef) -> dict[str, Any]:
        path = self._exec_dir(ref) / _STATE_FILE
        if not path.is_file():
            raise UnknownExecutionError(f"execution {ref.opaque_handle!r} is unknown to executor 'local'")
        return json.loads(path.read_text())

    def _write_state(self, ref: ExecutionRef, state: dict[str, Any]) -> None:
        exec_dir = self._exec_dir(ref)
        exec_dir.mkdir(parents=True, exist_ok=True)
        (exec_dir / _STATE_FILE).write_text(json.dumps(state, sort_keys=True))

    def _write_artifacts(self, ref: ExecutionRef, lenses: list[str]) -> None:
        lens_dir = self._exec_dir(ref) / _ARTIFACT_DIR
        lens_dir.mkdir(parents=True, exist_ok=True)
        for lens in lenses:
            (lens_dir / f"{lens}.review.json").write_text(json.dumps({"lens": lens, "ok": True}))

    def _remember_key(self, job: ExecutorJob, opaque: str) -> None:
        key = self._key_for(job)
        if not key:
            return
        idem = self._load_idempotency()
        idem[key] = opaque
        self._write_idempotency(idem)

    def _lookup_by_key(self, job: ExecutorJob) -> ExecutionRef | None:
        key = self._key_for(job)
        if not key:
            return None
        opaque = self._load_idempotency().get(key)
        if opaque is None:
            return None
        ref = ExecutionRef(
            executor_kind=self.kind,
            adapter_version=self.adapter_version,
            opaque_handle=opaque,
            attempt_id=job.attempt_id,
        )
        if not (self._exec_dir(ref) / _STATE_FILE).is_file():
            return None
        return ref

    @staticmethod
    def _key_for(job: ExecutorJob) -> str:
        if job.idempotency_key:
            return f"{job.idempotency_key}::{job.attempt_id}"
        return ""

    def _load_idempotency(self) -> dict[str, str]:
        path = self.root / _IDEMPOTENCY_FILE
        if not path.is_file():
            return {}
        return dict(json.loads(path.read_text()))

    def _write_idempotency(self, mapping: dict[str, str]) -> None:
        (self.root / _IDEMPOTENCY_FILE).write_text(json.dumps(mapping, sort_keys=True))

    def _remove_execution_dir(self, ref: ExecutionRef) -> None:
        target = self._exec_dir(ref)
        if target.is_dir():
            for child in target.iterdir():
                if child.is_dir():
                    for leaf in child.iterdir():
                        leaf.unlink(missing_ok=True)
                    child.rmdir()
                else:
                    child.unlink(missing_ok=True)
            target.rmdir()

    def _ensure_online(self) -> None:
        if self._shutdown:
            raise ExecutorError("executor is shut down")
