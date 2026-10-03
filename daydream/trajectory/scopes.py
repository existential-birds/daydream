"""Invocation and fork context lifetimes, including failure-safe recorder cleanup."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

from daydream import timeutil, ui
from daydream.trajectory.context import _RECORDER_VAR
from daydream.trajectory.invocation import Invocation
from daydream.trajectory.lifecycle import DispatchHandle, ForkIdentity, _CompletedFork
from daydream.trajectory.types import DaydreamPhase

if TYPE_CHECKING:
    from daydream.trajectory.recorder import TrajectoryRecorder

_console = ui.create_console()


@asynccontextmanager
async def invocation_scope(recorder: TrajectoryRecorder, phase: DaydreamPhase) -> AsyncIterator[Invocation]:
    """Register an active invocation, then persist its terminal evidence and steps."""
    invocation = Invocation(recorder=recorder, phase=phase, invocation_id=recorder._next_invocation_id())
    invocation.started_at = timeutil.now_iso()
    recorder._active_invocations.append(invocation)
    try:
        yield invocation
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
    parent: TrajectoryRecorder, descriptor: str, dispatch: DispatchHandle | None = None,
) -> AsyncIterator[TrajectoryRecorder]:
    """Bind a child recorder and fold only successfully written child evidence."""
    from daydream.trajectory.recorder import TrajectoryRecorder

    identity: ForkIdentity | None = None
    if dispatch is not None:
        if dispatch._recorder is not parent:
            raise RuntimeError("dispatch belongs to a different recorder")
        identity = dispatch.register_fork(descriptor)
    child = TrajectoryRecorder(
        path=parent._sibling_path_for(descriptor, identity),
        run_flow=parent.run_flow,
        target_dir=parent.target_dir,
        agent_model_name=parent.agent_model_name,
        redactor=parent.redactor,
        session_id=parent.session_id,
        pr_number=parent.pr_number,
        pr_repo=parent.pr_repo,
        backend_name=parent.backend_name,
        review_backend_name=parent.review_backend_name,
        fix_backend_name=parent.fix_backend_name,
        test_backend_name=parent.test_backend_name,
        artifact_run_dir=parent.artifact_run_dir,
        document_writer=parent.document_writer,
    )
    child.parent = parent
    child.descriptor = descriptor
    if identity is not None:
        child._trajectory_id = identity.fork_id
    registry = parent._signal_registry
    if registry is None:
        raise RuntimeError("cannot enter a fork without an active parent recorder")
    child._signal_registry = registry
    registry.register(child)
    child._previous_token = _RECORDER_VAR.set(child)
    entered_at = timeutil.now_iso()
    child._run_started_at = entered_at
    try:
        yield child
    except BaseException:
        child._aborted = True
        raise
    finally:
        write_ok = False
        try:
            exited_at = timeutil.now_iso()
            child._run_ended_at = exited_at
            try:
                child._write()
                write_ok = bool(child.steps)
            except Exception as exc:  # noqa: BLE001 - recording must never crash a run
                ui.print_warning(
                    _console,
                    f"Sibling trajectory write failed: {type(exc).__name__}: {exc}",
                )
            if write_ok and child.parent is not None:
                # The root trajectory's final_metrics is whole-run truth: fold the
                # fork's totals in so manifest/eval consumers read one number
                # instead of re-summing sibling files. The fork file keeps its own
                # share. A failed child write folds nothing (the error already
                # degrades the record, D-11).
                child.parent._accumulate_metrics(
                    prompt_tokens=child._final_totals["prompt"],
                    completion_tokens=child._final_totals["completion"],
                    cached_tokens=child._final_totals["cached"],
                    cost_usd=(child._final_totals["cost"] if child._final_totals["any_cost_seen"] else None),
                )
                child.parent._folded_fork_totals = True
                sibling_ref = child.parent._logical_child_trajectory_ref(child.path)
                child.parent._register_fork_subtrajectory(
                    child=child,
                    identity=identity,
                    sibling_trajectory_ref=sibling_ref,
                )
                if dispatch is not None and identity is not None:
                    dispatch._record_completed(
                        _CompletedFork(
                            identity=identity,
                            path=child.path,
                            trajectory_id=child.trajectory_id,
                        )
                    )
            elif dispatch is not None:
                dispatch._record_write_failure()
        finally:
            registry = child._signal_registry
            if registry is not None:
                registry.unregister(child)
                child._signal_registry = None
            if child._previous_token is not None:
                _RECORDER_VAR.reset(child._previous_token)
                child._previous_token = None
