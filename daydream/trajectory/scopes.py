"""Invocation and fork context lifetimes, including failure-safe recorder cleanup."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

import anyio

from daydream import timeutil, ui
from daydream.trajectory.context import _RECORDER_VAR
from daydream.trajectory.invocation import Invocation
from daydream.trajectory.lifecycle import DispatchHandle, ForkIdentity, _CompletedFork
from daydream.trajectory.types import DaydreamPhase

if TYPE_CHECKING:
    from daydream.trajectory.recorder import EvidenceRecorder

_console = ui.create_console()


@asynccontextmanager
async def invocation_scope(recorder: EvidenceRecorder, phase: DaydreamPhase) -> AsyncIterator[Invocation]:
    """Register an active invocation, then append its terminal evidence and steps."""
    invocation = Invocation(recorder=recorder, phase=phase, invocation_id=recorder._next_invocation_id())
    invocation.started_at = timeutil.now_iso()
    recorder._active_invocations.append(invocation)
    try:
        yield invocation
    except anyio.get_cancelled_exc_class():
        # Cancellation is a terminal attempt outcome, not a backend error.
        # An earlier budget/veto reason remains authoritative during cleanup.
        if invocation._stop_reason is None:
            invocation.mark_aborted("cancelled")
        raise
    except Exception as exc:
        # Record escaping backend errors before the final step closes, without
        # masking the original error if recording its subtype fails.
        try:
            invocation.mark_errored(str(getattr(exc, "subtype", None) or type(exc).__name__))
        except Exception:  # noqa: BLE001 - recording must not mask the body failure
            pass
        raise
    finally:
        try:
            invocation.finish()
            invocation.ended_at = timeutil.now_iso()
            recorder._register_subtrajectory(invocation)
        finally:
            with suppress(ValueError):
                recorder._active_invocations.remove(invocation)


@asynccontextmanager
async def fork_scope(
    parent: EvidenceRecorder, descriptor: str, dispatch: DispatchHandle | None = None,
) -> AsyncIterator[EvidenceRecorder]:
    """Bind a child recorder and fold only sealed, nonempty child evidence."""
    from daydream.trajectory.recorder import EvidenceRecorder, _admitted_descriptor

    descriptor = _admitted_descriptor(descriptor)
    identity: ForkIdentity | None = None
    if dispatch is not None:
        if dispatch._recorder is not parent:
            raise RuntimeError("dispatch belongs to a different recorder")
        identity = dispatch.register_fork(descriptor)
    registry = parent._signal_registry
    if registry is None:
        raise RuntimeError("cannot enter a fork without an active parent recorder")
    child = EvidenceRecorder(
        path=parent._sibling_path_for(descriptor, identity),
        run_flow=parent.run_flow,
        target_dir=parent.target_dir,
        agent_model_name=parent.agent_model_name,
        session_id=parent.session_id,
        pr_number=parent.pr_number,
        pr_repo=parent.pr_repo,
        immutable_input=parent.immutable_input,
        backend_name=parent.backend_name,
        review_backend_name=parent.review_backend_name,
        fix_backend_name=parent.fix_backend_name,
        test_backend_name=parent.test_backend_name,
        artifact_run_dir=parent.artifact_run_dir,
    )
    child.parent = parent
    child.descriptor = descriptor
    if identity is not None:
        child._trajectory_id = identity.fork_id
    child._signal_registry = registry
    entered_at = timeutil.now_iso()
    child._run_started_at = entered_at
    child._open_document()
    registry.register(child)
    child._previous_token = _RECORDER_VAR.set(child)
    try:
        yield child
    except BaseException:
        child._aborted = True
        raise
    finally:
        try:
            exited_at = timeutil.now_iso()
            child._run_ended_at = exited_at
            sealed = False
            try:
                child._write_final()
                sealed = True
            except Exception as exc:  # noqa: BLE001 - recording must never crash a run
                registry.failed_documents.append(child.trajectory_id)
                ui.print_warning(_console, f"Sibling evidence write failed: {type(exc).__name__}: {exc}")
            if sealed and child._step_count:
                # The root ledger is whole-run truth: fold the fork's total once so
                # manifest/eval consumers read one number. The fork keeps its own share.
                parent.usage.fold(child.usage)
                registry.retain(child._reference())
                phase = DaydreamPhase.FIX.value if child._first_phase is None else child._first_phase[1]
                parent._register_fork_subtrajectory(
                    child=child,
                    phase=phase,
                    identity=identity,
                    started_at=entered_at,
                    ended_at=exited_at,
                    sibling_trajectory_ref=parent._logical_child_trajectory_ref(child.path),
                )
                if dispatch is not None and identity is not None:
                    dispatch._record_completed(
                        _CompletedFork(identity=identity, path=child.path, trajectory_id=child.trajectory_id)
                    )
            else:
                if sealed:
                    # A child that produced no steps keeps no document, as before.
                    with suppress(OSError):
                        child.path.unlink()
                if dispatch is not None:
                    dispatch._record_write_failure()
        finally:
            registry.unregister(child)
            child._signal_registry = None
            if child._previous_token is not None:
                _RECORDER_VAR.reset(child._previous_token)
                child._previous_token = None
