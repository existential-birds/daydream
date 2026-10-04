"""Capture run identity, own recorder writes, and finalize immutable evidence."""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

from daydream.agent import console
from daydream.artifact_visibility import (
    ArtifactDisposition,
    ArtifactSession,
    ArtifactVisibilityError,
    PrivateWorkspaceOwner,
    RoutedDestination,
    TrajectoryOutputRoute,
    artifact_session_active,
)
from daydream.backends import BackendExecutionInput
from daydream.config import DEFAULT_PI_MODEL
from daydream.extensions import Registry, UnresolvedExtensionError, get_registry
from daydream.observability.runtime import associate_run_trajectory
from daydream.review_profile import resolve_from_runconfig
from daydream.run_config import (
    DEEP_FLOW_ALIASES,
    RunConfig,
    _default_backend_name,
    _resolved_backend_name,
    _resolved_model,
    _resolved_review_backend_name,
)
from daydream.run_snapshot import ArchiveRunSnapshot, ManifestRunIdentity, RunPhaseCapabilities, RunProfileIdentity
from daydream.trajectory import (
    DaydreamRunFlow,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    TrajectoryRecorder,
    default_trajectory_path,
    get_current_recorder,
)
from daydream.ui import print_error
from daydream.workspace import WorkContext


class _RunSnapshotCaptureError(RuntimeError):
    """A recorder callback supplied malformed run-wide immutable evidence."""


@dataclass
class _RunWriteCapture:
    """Synchronous, non-raising retention boundary for P07 write snapshots."""

    session_id: str
    run_flow: DaydreamRunFlow | None = None
    partial: RunWriteSnapshot | None = None
    final: RunWriteSnapshot | None = None
    validation_error: _RunSnapshotCaptureError | None = None
    manifest_identity: ManifestRunIdentity | None = None

    def retain(self, recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot) -> None:
        try:
            if recorder.session_id != self.session_id:
                raise _RunSnapshotCaptureError("recorder session identity mismatch")
            if self.run_flow is not None and recorder.run_flow is not self.run_flow:
                raise _RunSnapshotCaptureError("recorder run flow identity mismatch")
            snapshot.validate(self.session_id)
            self.run_flow = recorder.run_flow
            if snapshot.status == "complete":
                self.final = snapshot
                self.validation_error = None
            else:
                self.partial = snapshot
        except Exception as exc:
            self.validation_error = (
                exc if isinstance(exc, _RunSnapshotCaptureError)
                else _RunSnapshotCaptureError(f"snapshot validation failed ({type(exc).__name__})")
            )


@dataclass(frozen=True)
class _RunArtifacts:
    """One outer host session and its pre-model registered output routes."""

    session: ArtifactSession
    owner: PrivateWorkspaceOwner
    trajectory: TrajectoryOutputRoute
    capture: _RunWriteCapture
    dump: RoutedDestination | None
    execution_input: BackendExecutionInput | None = field(default=None, kw_only=True, repr=False, compare=False)

    def write_trajectory_document(
        self, document: TrajectoryDocumentSnapshot, status: Literal["complete", "partial"]
    ) -> None:
        """Write through the host sink and retain a closed failure disposition."""
        try:
            self.session.write_trajectory_document(self.trajectory, document, status)
        except Exception as exc:
            self.capture.validation_error = _RunSnapshotCaptureError(
                f"trajectory {status} output failed ({type(exc).__name__})"
            )
            raise


def _open_recorder(
    *,
    config: RunConfig,
    target_dir: Path,
    work: WorkContext | None,
    flow_kind: DaydreamRunFlow,
    run_artifacts: _RunArtifacts | None = None,
    allow_standalone: bool = False,
) -> TrajectoryRecorder:
    """Open every flow's recorder with consistent identity, paths, and retention.

    Flows must use this factory so dump-artifacts and strict archive finalization
    retain their snapshots. Direct standalone callers opt in with
    ``allow_standalone=True`` and no artifact session; they retain and archive none.
    """
    if run_artifacts is None:
        if not allow_standalone:
            raise ArtifactVisibilityError("standalone recorder requires allow_standalone=True")
        if artifact_session_active():
            raise ArtifactVisibilityError("standalone recorder cannot run inside an active artifact session")
        session_id = str(uuid.uuid4())
        trajectory_path = config.trajectory_path or default_trajectory_path(target_dir, session_id)
    else:
        session_id = run_artifacts.session.layout.session_id
        if (
            work is None
            or work.source != run_artifacts.owner.source
            or work.source != run_artifacts.session.provenance.public_source
            or session_id != run_artifacts.capture.session_id
            or session_id != run_artifacts.session.provenance.session_id
        ):
            raise ArtifactVisibilityError("recorder artifact identity mismatch")
        # The recorder's target is the public source once a session owns the run.
        target_dir = work.source
        if run_artifacts.trajectory.full.write_path is None:
            raise ArtifactVisibilityError("trajectory route has no writable destination")
        trajectory_path = run_artifacts.trajectory.full.write_path
        if run_artifacts.capture.manifest_identity is not None:
            raise ArtifactVisibilityError("manifest run identity was already captured")
        _resolve_review_profile(config)
        run_artifacts.capture.manifest_identity = capture_manifest_run_identity(
            config, flow_kind, get_registry(), work.repo,
            execution_input=run_artifacts.execution_input,
        )
    # Trajectory labels retain their representative per-flow phase mapping;
    # manifest identity separately records the general default and capabilities.
    names = _recorder_backend_names(config, flow_kind)
    recorder = TrajectoryRecorder(
        path=trajectory_path,
        run_flow=flow_kind,
        target_dir=target_dir,
        artifact_run_dir=None if run_artifacts is None else run_artifacts.trajectory.run_dir,
        document_writer=None if run_artifacts is None else run_artifacts.write_trajectory_document,
        agent_model_name="",
        session_id=session_id,
        explicit_path=config.trajectory_path is not None,
        pr_number=config.pr_number,
        pr_repo=config.pr_repo,
        backend_name=names.backend,
        review_backend_name=names.backend,
        fix_backend_name=names.fix,
        test_backend_name=names.test,
        on_write=None if run_artifacts is None else run_artifacts.capture.retain,
    )
    associate_run_trajectory(recorder.session_id)
    return recorder


def _resolve_review_profile(config: RunConfig) -> None:
    """Resolve and validate the review profile once at the runner composition root."""
    if config.review_profile is None:
        config.review_profile = resolve_from_runconfig(config)
    _record_review_profile(config)


def _record_review_profile(config: RunConfig) -> None:
    """Record profile version, name, source, and digest when both profile and recorder exist.

    Deep dispatch resolves before opening its recorder, then calls this again from
    inside the recorder scope to capture the resolved policy.
    """
    if config.review_profile is None:
        return
    recorder = get_current_recorder()
    if recorder is None:
        return
    recorder.record_profile(
        schema_version=config.review_profile.profile.schema_version,
        name=config.review_profile.name,
        source_kind=config.review_profile.source_kind,
        digest=config.review_profile.digest,
    )


class RecorderBackendNames(NamedTuple):
    """Representative, fix, and test backend identities for trajectory and manifest.

    Empty fix/test names mean the phase is absent and are omitted on serialization.
    """

    backend: str
    fix: str
    test: str


def _recorder_backend_names(
    config: RunConfig, flow_kind: DaydreamRunFlow
) -> RecorderBackendNames:
    """Resolve identities only for phases statically owned by each flow.

    Custom compositions cannot promise fix/test execution. Empty names suppress
    those fields; the representative phase for each built-in is listed below.
    """
    representative, runs_fix, runs_test = {
        DaydreamRunFlow.NORMAL: ("per_stack_review", True, True),
        DaydreamRunFlow.DEEP: ("per_stack_review", True, True),
        DaydreamRunFlow.TTT: ("per_stack_review", False, False),
        DaydreamRunFlow.IMPROVE: ("recon", False, False),
        DaydreamRunFlow.DIAGRAM: ("diagram", False, False),
        DaydreamRunFlow.CUSTOM: ("review", False, False),
        DaydreamRunFlow.PR: ("review", True, False),
    }.get(flow_kind, ("review", True, True))
    return RecorderBackendNames(
        backend=_resolved_backend_name(config, representative),
        fix=_resolved_backend_name(config, "fix") if runs_fix else "",
        test=_resolved_backend_name(config, "test") if runs_test else "",
    )


def capture_manifest_run_identity(
    config: RunConfig, flow_kind: DaydreamRunFlow, registry: Registry, cwd: Path,
    *, execution_input: BackendExecutionInput | None = None,
) -> ManifestRunIdentity:
    """Freeze manifest identity from resolved runner policy before model execution.

    The registry is the run's resolved registry, including built-in overrides.
    Mode and resume exceptions preserve the historical archive capabilities;
    these labels do not evaluate each step's runtime predicate. Backend/model
    precedence stays owned by the same helpers used to construct backends.
    """
    runtime_flow_name: str | None
    if flow_kind is DaydreamRunFlow.IMPROVE:
        runtime_flow_name = "improve"
    elif flow_kind is DaydreamRunFlow.DIAGRAM:
        runtime_flow_name = "diagram"
    elif flow_kind is DaydreamRunFlow.CUSTOM:
        runtime_flow_name = config.flow_name
    else:
        runtime_flow_name = "deep"
    step_names: set[str] = set()
    if runtime_flow_name:
        try:
            entries = registry.flow(runtime_flow_name)
        except UnresolvedExtensionError:
            entries = []
        for entry in entries:
            if isinstance(entry, str):
                step_names.add(entry)
            else:
                step_names.update(entry.steps)

    if flow_kind is DaydreamRunFlow.TTT:
        runs_fix, runs_test = False, False
    elif flow_kind is DaydreamRunFlow.PR:
        runs_fix, runs_test = True, False
    else:
        runs_fix, runs_test = "fix" in step_names, "test" in step_names
    if flow_kind in (DaydreamRunFlow.PR, DaydreamRunFlow.IMPROVE, DaydreamRunFlow.DIAGRAM):
        runs_merge = False
    elif flow_kind is DaydreamRunFlow.CUSTOM:
        runs_merge = any("merge" in step for step in step_names)
    else:
        runs_merge = True
    runs_per_stack_review = (
        (config.flow_name is None or config.flow_name in DEEP_FLOW_ALIASES)
        and config.start_at not in ("merge", "fix")
        and flow_kind is not DaydreamRunFlow.DIAGRAM
    )
    phases = RunPhaseCapabilities(
        per_stack_review=runs_per_stack_review,
        merge=runs_merge and config.start_at != "fix",
        fix=runs_fix,
        test=runs_test,
        push=flow_kind not in (DaydreamRunFlow.TTT, DaydreamRunFlow.PR) and "commit" in step_names,
        remote_ci=flow_kind not in (DaydreamRunFlow.TTT, DaydreamRunFlow.PR) and "remote-ci" in step_names,
    )
    per_stack_backend: str | None = None
    per_stack_model: str | None = None
    if phases.per_stack_review:
        per_stack_backend = _resolved_backend_name(config, "per_stack_review")
        per_stack_model = _resolved_model(config, "per_stack_review")
        if per_stack_model is None and per_stack_backend == "pi":
            from daydream.backends.pi import _configured_pi_model

            configured_model = (
                _configured_pi_model(cwd) if execution_input is None
                else _configured_pi_model(cwd, agent_dir=execution_input.pi_agent_dir)
            )
            per_stack_model = configured_model or DEFAULT_PI_MODEL
    profile = config.review_profile
    return ManifestRunIdentity(
        skill=config.stack,
        model=None,
        backend=_default_backend_name(config),
        review_backend=_resolved_review_backend_name(config),
        fix_backend=_resolved_backend_name(config, "fix") if phases.fix else None,
        test_backend=_resolved_backend_name(config, "test") if phases.test else None,
        per_stack_review_backend=per_stack_backend,
        per_stack_review_model=per_stack_model,
        review_only=config.output_mode == "review",
        deep=not config.shallow,
        profile=None if profile is None else RunProfileIdentity(
            schema_version=profile.profile.schema_version,
            name=profile.name,
            source_kind=profile.source_kind,
            digest=profile.digest,
        ),
        phases=phases,
    )


def _finalize_run_artifacts(
    run_artifacts: _RunArtifacts, *, selected: RunWriteSnapshot, config: RunConfig,
    work: WorkContext, successful: bool,
) -> None:
    """Freeze once, collect optional data, and publish the joined runtime outputs."""
    from daydream.archive import (
        ArchiveFinalizationError,
        ArchiveIntegrityError,
        ArchivePublicationError,
        finalize_archive_run,
    )
    from daydream.archive.manifest import archive_recorder_provenance_from_snapshot

    if run_artifacts.capture.run_flow is None:
        raise ArtifactVisibilityError("run flow provenance was not retained")
    identity = run_artifacts.capture.manifest_identity
    if identity is None:
        raise ArtifactVisibilityError("manifest run identity was not captured")
    snapshot = run_artifacts.session.freeze(selected)
    recorder_provenance = archive_recorder_provenance_from_snapshot(
        write_snapshot=selected, run_flow=run_artifacts.capture.run_flow
    )
    dump_path = (
        None
        if run_artifacts.dump is None
        else run_artifacts.session.finalization_merge_path(run_artifacts.dump, snapshot=snapshot)
    )
    try:
        finalize_archive_run(
            run=ArchiveRunSnapshot(recorder_provenance, identity, selected), artifacts=snapshot,
            artifact_provenance=run_artifacts.session.provenance, config=config,
            work=work, upload=successful, dump_path=dump_path,
        )
    except (ArchiveIntegrityError, ArchivePublicationError) as exc:
        raise ArtifactVisibilityError(str(exc)) from exc
    except ArchiveFinalizationError as exc:
        # Exception text can include credentials, repository content, or private
        # runtime paths. Collection diagnostics report only the error category.
        print_error(console, "Data Collection", f"Run data could not be persisted ({type(exc).__name__}).")
        if dump_path is not None and dump_path.is_dir() and not any(dump_path.iterdir()):
            dump_path.rmdir()
    disposition = ArtifactDisposition.COMPLETE if successful else ArtifactDisposition.PARTIAL_EVIDENCE
    run_artifacts.session.finalize_frozen(snapshot, disposition=disposition)
