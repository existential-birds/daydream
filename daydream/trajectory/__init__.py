"""Record backend events as ATIF trajectories.

The recorder owns persistence; invocation buffers events; lifecycle brackets
phases and fan-outs. Layout, frozen timing analysis, and billing each have one
owner. Credential redaction is shared independently by daydream.redaction.
"""

from daydream.redaction import (
    redact_structured_text as redact_structured_text,
    redact_text as redact_text,
    redact_value as redact_value,
)
from daydream.timeutil import now_iso as now_iso
from daydream.trajectory.context import (
    current_session_id as current_session_id,
    get_current_recorder as get_current_recorder,
)
from daydream.trajectory.generation import (
    MAX_PENDING_GENERATION_DRAFTS as MAX_PENDING_GENERATION_DRAFTS,
    MAX_RETAINED_CHOICE_BYTES as MAX_RETAINED_CHOICE_BYTES,
)
from daydream.trajectory.invocation import INCOMPLETE_CALL_CONTENT as INCOMPLETE_CALL_CONTENT, Invocation as Invocation
from daydream.trajectory.layout import (
    PARTIAL_SUFFIX as PARTIAL_SUFFIX,
    RUN_DOCUMENT_NAME as RUN_DOCUMENT_NAME,
    RUNS_DIRNAME as RUNS_DIRNAME,
    SIBLINGS_DIRNAME as SIBLINGS_DIRNAME,
    default_trajectory_path as default_trajectory_path,
    partial_document_path as partial_document_path,
    run_directory as run_directory,
    run_document_path as run_document_path,
    sibling_document_path as sibling_document_path,
    siblings_directory as siblings_directory,
)
from daydream.trajectory.lifecycle import (
    DispatchHandle as DispatchHandle,
    ForkIdentity as ForkIdentity,
    HostPhaseHandle as HostPhaseHandle,
    PhaseEvent as PhaseEvent,
    PhaseScopeHandle as PhaseScopeHandle,
    dispatch_scope as dispatch_scope,
    finish_partial_or_failed as finish_partial_or_failed,
    host_phase_scope as host_phase_scope,
    maybe_fork as maybe_fork,
    partial_or_failed_terminal as partial_or_failed_terminal,
    phase_scope as phase_scope,
)
from daydream.trajectory.recorder import (
    TrajectoryDocumentWriter as TrajectoryDocumentWriter,
    TrajectoryRecorder as TrajectoryRecorder,
    TrajectoryWriteCallback as TrajectoryWriteCallback,
    flush_active_signal_recorders as flush_active_signal_recorders,
)
from daydream.trajectory.redactor import Redactor as Redactor
from daydream.trajectory.timing import (
    TimingSummary as TimingSummary,
    compute_timing_summary as compute_timing_summary,
    snapshot_trajectories as snapshot_trajectories,
)
from daydream.trajectory.types import (
    DaydreamPhase as DaydreamPhase,
    DaydreamRunFlow as DaydreamRunFlow,
    LifecycleReasonCode as LifecycleReasonCode,
    LifecycleStatus as LifecycleStatus,
    RunWriteSnapshot as RunWriteSnapshot,
    TrajectoryDocumentSnapshot as TrajectoryDocumentSnapshot,
)
