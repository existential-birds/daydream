"""Centralized run archive for daydream.

Automatically copies the full artifact bundle (trajectory, review output,
deep artifacts, diff) to ``~/.daydream/archive/runs/{session_id}/`` after
every run, writes a ``manifest.json``, and indexes the run in a SQLite
database for cross-project querying.

Exports:
    archive_run: Top-level entry point called from the TrajectoryRecorder
        on_write callback.
    get_archive_dir: Returns the archive root directory, creating it on
        first access.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from daydream.archive.git_context import capture_git_context
from daydream.archive.index import upsert_run
from daydream.archive.manifest import (
    ArchiveRecorderProvenance,
    _flow_fix_test_steps,
    _flow_phase_steps,
    _runtime_flow_name,
    archive_recorder_provenance_from_snapshot,
    build_manifest_from_snapshot,
)
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.trajectory import DaydreamRunFlow

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from daydream.artifact_visibility import (
        ArtifactEvidenceProvenance,
        ArtifactTreeSnapshot,
    )
    from daydream.runner import RunConfig
    from daydream.trajectory import RunWriteSnapshot, TrajectoryRecorder
    from daydream.workspace import WorkContext


class ArchiveFinalizationError(RuntimeError):
    """Strict host archive finalization did not complete successfully."""


def _warn(message: str) -> None:
    """Print a one-line warning through the daydream console (never raises)."""
    from daydream.ui import create_console, print_warning

    print_warning(create_console(), message)


def _flow_runs_merge(flow: DaydreamRunFlow, flow_name: str | None) -> bool:
    """Whether the executed flow runs the deep cross-stack/single-stack merge.

    The deep pipeline's merge step runs for normal and review flows. Improve-only,
    diagram-only (issue #1113) and the legacy PR compatibility label do not run
    the merge spine. Diagram-only must answer False explicitly: it reuses the
    deep preamble and deliberately keeps a prior deep run's
    ``.daydream/deep/`` artifacts on disk (D21), so a True here would make it
    adopt that run's ``merged-items.json`` as its own pipeline state.
    Custom flows are classified from their registered pipeline (a fork
    composing the built-in deep merge step is detected as it runs), mirroring
    ``_flow_fix_test_steps`` in ``archive.manifest``.
    """
    if flow is DaydreamRunFlow.PR:
        return False
    if flow is DaydreamRunFlow.IMPROVE:
        return False
    if flow is DaydreamRunFlow.DIAGRAM:
        return False
    if flow is DaydreamRunFlow.CUSTOM:
        from daydream.archive.manifest import _flow_phase_steps, _runtime_flow_name

        steps = _flow_phase_steps(_runtime_flow_name(flow, flow_name))
        return any("merge" in step for step in steps)
    return True


def _flow_push_remote_steps(
    flow: DaydreamRunFlow, flow_name: str | None
) -> tuple[bool, bool]:
    """Return exact commit/push and remote-CI capabilities for this run."""
    if flow in {DaydreamRunFlow.TTT, DaydreamRunFlow.PR}:
        return False, False
    steps = _flow_phase_steps(_runtime_flow_name(flow, flow_name))
    return ("commit" in steps, "remote-ci" in steps)


def get_archive_dir() -> Path:
    """Return the archive root directory, creating it on first access.

    Respects ``DAYDREAM_ARCHIVE_DIR`` env var. Default: ``~/.daydream/archive/``.

    Returns:
        Path to the archive root directory.
    """
    env = os.environ.get("DAYDREAM_ARCHIVE_DIR")
    if env:
        archive_dir = Path(env)
    else:
        archive_dir = Path.home() / ".daydream" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "runs").mkdir(exist_ok=True)
    return archive_dir


def archive_run(
    *,
    recorder: TrajectoryRecorder,
    write_snapshot: RunWriteSnapshot,
    target_dir: Path,
    config: RunConfig,
    run_eval: bool = True,
    work: WorkContext | None = None,
    upload: bool = True,
) -> None:
    """Copy artifact bundle to archive and index in SQLite.

    Called from the ``on_write`` callback on ``TrajectoryRecorder``. Wraps
    its entire body in try/except so archive failure never affects the
    primary run.

    Args:
        recorder: Immutable run identity/configuration context.
        write_snapshot: Canonical trajectory bytes and run status frozen before
            the first write. Archive consumers never reread live recorder data.
        target_dir: Target directory that was reviewed.
        config: The RunConfig for this run.
        run_eval: Whether to run deterministic evaluation analysis.
        work: Optional WorkContext with pre-resolved git metadata. When
            provided, ``base_branch`` and ``base_sha`` are taken from the
            workspace snapshot instead of re-deriving them (which can fail
            when the default-branch probe or merge-base computation fails
            at archive time).
        upload: Whether to run the opt-in HF bundle upload. Disabled for
            signal-flush (partial) archives so a blocking network call never
            runs inside the SIGINT/SIGTERM handler path.
    """
    try:
        _archive_run_inner(
            recorder=recorder,
            write_snapshot=write_snapshot,
            target_dir=target_dir,
            config=config,
            run_eval=run_eval,
            work=work,
            upload=upload,
        )
    except Exception:  # noqa: BLE001 - archive failure must never affect the run
        # Import lazily to avoid circular imports at module level
        try:
            from daydream.ui import create_console, print_warning

            print_warning(create_console(), "Run archive failed (non-fatal)")
        except Exception:  # noqa: BLE001
            pass


def _validate_frozen_artifacts(artifacts: ArtifactTreeSnapshot) -> None:
    """Reject a frozen tree that changed before a strict host consumer."""
    from daydream.artifact_visibility import _manifest

    if _manifest(artifacts.root) != artifacts.manifest:
        raise ArchiveFinalizationError("frozen artifact tree changed before archive")


def _frozen_destination_path(
    *,
    artifacts: ArtifactTreeSnapshot,
    artifact_provenance: ArtifactEvidenceProvenance,
    label: str,
) -> Path | None:
    """Map one registered private destination into the immutable tree."""
    from daydream.artifact_visibility import OutputLabel

    expected = OutputLabel(label)
    matches = [route for route in artifacts.destinations if route.label is expected]
    if not matches:
        return None
    if len(matches) != 1 or matches[0].frozen_path is None:
        raise ArchiveFinalizationError("frozen artifact destination is malformed")
    live_root = Path(*artifact_provenance.live_components)
    try:
        relative = matches[0].frozen_path.relative_to(live_root)
    except ValueError as exc:
        raise ArchiveFinalizationError("frozen artifact destination escaped its run") from exc
    return artifacts.root / relative


def _copy_snapshot_bundle(
    *,
    artifacts: ArtifactTreeSnapshot,
    artifact_provenance: ArtifactEvidenceProvenance,
    run_dir: Path,
    recorder_provenance: ArchiveRecorderProvenance,
    write_snapshot: RunWriteSnapshot,
) -> None:
    """Assemble an archive only from frozen tree and immutable document bytes."""
    root_path = run_dir / "trajectory.json"
    seen: set[Path] = set()
    for document in write_snapshot.documents:
        try:
            payload = json.loads(document.json_bytes)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ArchiveFinalizationError("frozen trajectory document is malformed") from exc
        if not isinstance(payload, dict) or (
            payload.get("trajectory_id") != document.trajectory_id
            or payload.get("session_id") != recorder_provenance.session_id
        ):
            raise ArchiveFinalizationError("frozen trajectory identity is malformed")
        if document.trajectory_id == recorder_provenance.session_id:
            destination = root_path
        else:
            name = document.path.name.removesuffix(".partial")
            if not name.endswith(".json") or name in {".", ".."}:
                raise ArchiveFinalizationError("frozen trajectory path is malformed")
            destination = run_dir / "trajectories" / name
        if destination in seen:
            raise ArchiveFinalizationError("frozen trajectory destination is duplicated")
        seen.add(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(document.json_bytes)

    frozen_daydream = artifacts.root / ".daydream"
    deep_dir = frozen_daydream / "deep"
    if recorder_provenance.run_flow is DaydreamRunFlow.DIAGRAM:
        for name in ("diagram.json", "diagram.md"):
            source = deep_dir / name
            if source.is_file():
                destination = run_dir / "deep" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    elif deep_dir.is_dir():
        shutil.copytree(deep_dir, run_dir / "deep", dirs_exist_ok=True)

    if recorder_provenance.run_flow is not DaydreamRunFlow.DIAGRAM:
        review = artifacts.root / REVIEW_OUTPUT_FILE
        if review.is_file():
            shutil.copy2(review, run_dir / "review-output.md")
    for name in ("diff.patch", "recommended.patch"):
        source = frozen_daydream / name
        if source.is_file() and not (
            name == "recommended.patch"
            and recorder_provenance.run_flow is DaydreamRunFlow.DIAGRAM
        ):
            shutil.copy2(source, run_dir / name)

    findings = _frozen_destination_path(
        artifacts=artifacts,
        artifact_provenance=artifact_provenance,
        label="findings_output",
    )
    if findings is not None and findings.is_file():
        shutil.copy2(findings, run_dir / "findings.json")


def _run_eval_strict(
    artifact_root: Path,
    session_id: str,
    run_dir: Path,
    write_snapshot: RunWriteSnapshot,
    *,
    artifact_provenance: ArtifactEvidenceProvenance,
    code_workspace: Path,
) -> dict[str, Any]:
    """Run evaluation without the legacy warning-and-None disposition."""
    from daydream.eval.analyzer import analyze_session
    from daydream.trajectory import snapshot_trajectories

    try:
        result = analyze_session(
            artifact_root / ".daydream",
            session_id=session_id,
            frozen_trajectories=snapshot_trajectories(write_snapshot),
            artifact_provenance=artifact_provenance,
            code_workspace=code_workspace,
        )
        if not isinstance(result, dict) or "error" in result:
            raise ValueError("evaluation returned an incomplete result")
        (run_dir / "evaluation.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        return result
    except ArchiveFinalizationError:
        raise
    except Exception as exc:
        raise ArchiveFinalizationError("evaluation finalization failed") from exc


def finalize_archive_run(
    *,
    recorder_provenance: ArchiveRecorderProvenance,
    artifacts: ArtifactTreeSnapshot,
    artifact_provenance: ArtifactEvidenceProvenance,
    config: RunConfig,
    write_snapshot: RunWriteSnapshot,
    work: WorkContext | None,
    upload: bool = True,
    dump_path: Path | None = None,
) -> None:
    """Strictly archive one immutable run or raise a closed typed error."""
    if (
        artifacts.session_id != recorder_provenance.session_id
        or artifacts.session_id != write_snapshot.root_trajectory_id
        or artifacts.workspace_key != artifact_provenance.workspace_key
        or artifacts.session_id != artifact_provenance.session_id
        or (
            work is not None
            and artifact_provenance.public_source != work.source
        )
    ):
        raise ArchiveFinalizationError("archive identity mismatch")
    _validate_frozen_artifacts(artifacts)
    if not config.archive and not config.dump_artifacts:
        return
    archive_dir: Path | None = None
    run_dir: Path | None = None
    assembly_dir: Path | None = None
    completed = False
    assembly_created = False
    archive_installed = False
    dump_started = False
    try:
        archive_dir = get_archive_dir()
        run_dir = archive_dir / "runs" / recorder_provenance.session_id
        assembly_dir = run_dir.with_name(f".{run_dir.name}.finalizing")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        if (
            run_dir.exists()
            or run_dir.is_symlink()
            or assembly_dir.exists()
            or assembly_dir.is_symlink()
        ):
            raise ArchiveFinalizationError("archive session destination already exists")
        assembly_dir.mkdir()
        assembly_created = True
        _copy_snapshot_bundle(
            artifacts=artifacts,
            artifact_provenance=artifact_provenance,
            run_dir=assembly_dir,
            recorder_provenance=recorder_provenance,
            write_snapshot=write_snapshot,
        )
        _validate_frozen_artifacts(artifacts)
        status = write_snapshot.status
        target_dir = artifacts.root
        git_target = work.repo if work is not None else target_dir
        git_ctx = capture_git_context(git_target)
        _validate_frozen_artifacts(artifacts)
        if work is not None:
            git_ctx.base_branch = work.base_branch
            if git_ctx.base_sha is None:
                git_ctx.base_sha = work.base_sha
        runs_fix, runs_test = _flow_fix_test_steps(
            recorder_provenance.run_flow,
            config.flow_name,
        )
        evaluation: dict[str, Any] | None = None
        if config.run_eval and recorder_provenance.run_flow is not DaydreamRunFlow.DIAGRAM:
            evaluation = _run_eval_strict(
                target_dir,
                recorder_provenance.session_id,
                assembly_dir,
                write_snapshot,
                artifact_provenance=artifact_provenance,
                code_workspace=(
                    work.repo if work is not None else artifact_provenance.public_source
                ),
            )
            _validate_frozen_artifacts(artifacts)
        fix_failures = _read_fix_failures(target_dir) if runs_fix else None
        fix_leftover = _read_fix_leftover_untracked(target_dir) if runs_fix else None
        fix_quality = (
            _read_fix_quality_gate(target_dir, recorder_provenance.session_id)
            if runs_fix
            else None
        )
        recommended = (
            _read_recommended_capture(target_dir, recorder_provenance.session_id)
            if runs_fix
            else None
        )
        _validate_frozen_artifacts(artifacts)
        if fix_failures:
            status = "partial"
        from daydream.archive.pipeline import derive_phase_states, derive_pipeline_status
        from daydream.archive.provenance import capture_executable_provenance

        runs_merge = (
            _flow_runs_merge(recorder_provenance.run_flow, config.flow_name)
            and getattr(config, "start_at", None) != "fix"
        )
        runs_push, runs_remote_ci = _flow_push_remote_steps(
            recorder_provenance.run_flow,
            config.flow_name,
        )
        from daydream.trajectory import snapshot_trajectories

        frozen_root = snapshot_trajectories(write_snapshot).get("main")
        frozen_extra = frozen_root.get("extra") if isinstance(frozen_root, dict) else None
        if isinstance(frozen_extra, dict) and frozen_extra.get("partial") is True:
            status = "partial"
        phase_states = derive_phase_states(
            target_dir,
            phase_events=(frozen_extra or {}).get("phase_events") if isinstance(frozen_extra, dict) else None,
            runs_merge=runs_merge,
            runs_fix=runs_fix,
            runs_test=runs_test,
            runs_push=runs_push,
            runs_remote_ci=runs_remote_ci,
            session_id=recorder_provenance.session_id,
            pr_repo=recorder_provenance.pr_repo,
            pr_number=recorder_provenance.pr_number,
        )
        _validate_frozen_artifacts(artifacts)
        pipeline_status = derive_pipeline_status(
            status,
            fix_failures,
            phase_states,
            runs_merge=runs_merge,
            runs_fix=runs_fix,
            runs_test=runs_test,
        )
        manifest = build_manifest_from_snapshot(
            recorder_provenance=recorder_provenance,
            write_snapshot=write_snapshot,
            config=config,
            git_ctx=git_ctx,
            status=status,
            archive_path=run_dir,
            evaluation=evaluation,
            source_path=str(work.source) if work is not None else None,
            cwd=str(work.repo) if work is not None else None,
            fix_failures=fix_failures,
            fix_leftover_untracked=fix_leftover,
            fix_quality_gate=fix_quality,
            recommended_capture=(recommended or {}).get("capture_point"),
            pipeline_status=pipeline_status,
            phase_states=phase_states,
            provenance=capture_executable_provenance(),
        )
        (assembly_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2),
            encoding="utf-8",
        )
        _validate_frozen_artifacts(artifacts)
        if config.archive and upload:
            from daydream.archive import hub

            hub_repo_id = hub.resolve_hub_repo(config)
            _validate_frozen_artifacts(artifacts)
            if hub_repo_id:
                uploaded = hub.upload_run_bundle(
                    assembly_dir,
                    hub_repo_id,
                    recorder_provenance.session_id,
                )
                _validate_frozen_artifacts(artifacts)
                if not uploaded:
                    raise ArchiveFinalizationError("archive upload failed")
        if config.dump_artifacts:
            if dump_path is None:
                raise ArchiveFinalizationError("dump finalization path is missing")
            from daydream.archive import scan

            scan_result = scan.scan_run_dir(assembly_dir)
            _validate_frozen_artifacts(artifacts)
            if not scan_result.clean:
                raise ArchiveFinalizationError("dump artifact secret scan refused publication")
            dump_started = True
            shutil.copytree(assembly_dir, dump_path, dirs_exist_ok=True)
        _validate_frozen_artifacts(artifacts)
        os.replace(assembly_dir, run_dir)
        assembly_created = False
        archive_installed = True
        _validate_frozen_artifacts(artifacts)
        upsert_run(archive_dir, manifest)
        completed = True
    except BaseException as exc:
        owned_paths = (
            (assembly_dir, assembly_created),
            (run_dir, archive_installed),
        )
        for owned, created in owned_paths:
            if (
                not completed
                and created
                and owned is not None
                and owned.exists()
                and not owned.is_symlink()
                and owned.is_dir()
            ):
                try:
                    shutil.rmtree(owned)
                except OSError as cleanup_error:
                    exc.add_note(
                        "strict archive cleanup retained private evidence "
                        f"({type(cleanup_error).__name__})"
                    )
        if dump_started and dump_path is not None and dump_path.exists() and dump_path.is_dir():
            try:
                shutil.rmtree(dump_path)
                dump_path.mkdir(mode=0o700)
            except OSError as cleanup_error:
                exc.add_note(
                    "strict dump cleanup retained private evidence "
                    f"({type(cleanup_error).__name__})"
                )
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, ArchiveFinalizationError):
            raise
        raise ArchiveFinalizationError("archive finalization failed") from exc


def _archive_run_inner(
    *,
    recorder: TrajectoryRecorder,
    write_snapshot: RunWriteSnapshot,
    target_dir: Path,
    config: RunConfig,
    run_eval: bool,
    work: WorkContext | None = None,
    upload: bool = True,
) -> None:
    """Core archive logic, not exception-wrapped."""
    from daydream.trajectory import snapshot_trajectories

    frozen_root = snapshot_trajectories(write_snapshot).get("main")
    if not isinstance(frozen_root, dict):
        raise ValueError("frozen root trajectory is missing")
    if (
        frozen_root.get("trajectory_id") != write_snapshot.root_trajectory_id
        or write_snapshot.root_trajectory_id != recorder.session_id
        or frozen_root.get("session_id") != recorder.session_id
    ):
        raise ValueError("frozen root trajectory does not match archive session")

    archive_dir = get_archive_dir()
    run_dir = archive_dir / "runs" / recorder.session_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy artifact bundle
    _copy_bundle(
        target_dir,
        run_dir,
        recorder,
        config,
        write_snapshot=write_snapshot,
    )
    status = write_snapshot.status

    # 2. Capture git context — prefer pre-resolved WorkContext over re-deriving
    #    from disk (HEAD may have moved; base-branch detection fails in worktrees).
    git_ctx = capture_git_context(target_dir)
    if work is not None:
        git_ctx.base_branch = work.base_branch
        if git_ctx.base_sha is None:
            git_ctx.base_sha = work.base_sha
            # Backfill changed_files when base_sha was injected from WorkContext
            if git_ctx.base_sha and git_ctx.head_sha and not git_ctx.changed_files:
                from daydream import git_ops
                from daydream.git_ops import GitError

                try:
                    git_ctx.changed_files = git_ops.diff_name_only(
                        target_dir, git_ctx.base_sha, git_ctx.head_sha,
                    )
                except GitError:
                    git_ctx.changed_files = []

    runs_fix, runs_test = _flow_fix_test_steps(recorder.run_flow, config.flow_name)

    # 3. Optionally run deterministic evaluation
    evaluation: dict[str, Any] | None = None
    if run_eval and recorder.run_flow is not DaydreamRunFlow.DIAGRAM:
        evaluation = _run_eval(
            target_dir,
            recorder.session_id,
            run_dir,
            write_snapshot,
        )

    # 3b. Surface dropped fix groups. A deep fix run that hit per-group failures
    #     left partial/reverted edits in the tree; the run is NOT "complete".
    #     Read from the source deep dir (written by the orchestrator before it
    #     returned, so it is reliably present here), force status to "partial".
    fix_failures = _read_fix_failures(target_dir) if runs_fix else None
    fix_leftover_untracked = _read_fix_leftover_untracked(target_dir) if runs_fix else None
    fix_quality_gate = (
        _read_fix_quality_gate(target_dir, recorder.session_id) if runs_fix else None
    )
    recommended_capture = (
        _read_recommended_capture(target_dir, recorder.session_id) if runs_fix else None
    )
    if fix_failures:
        status = "partial"

    # 3c. Executable provenance + pipeline outcome. Best-effort by contract:
    #     capture_executable_provenance never raises (per-field "unknown"), and
    #     derive_phase_states/derive_pipeline_status never raise on bad artifacts
    #     (absent/malformed -> absent/neutral). A failure here must never abort
    #     the archive (the surrounding archive_run try/except is a second net).
    from daydream.archive.pipeline import derive_phase_states, derive_pipeline_status
    from daydream.archive.provenance import capture_executable_provenance

    provenance = capture_executable_provenance()
    # Gate derivation to phases this registered flow can execute. Session-bound
    # artifacts prevent a non-deep or interrupted run from adopting prior state.
    runs_merge = (
        _flow_runs_merge(recorder.run_flow, config.flow_name)
        and getattr(config, "start_at", None) != "fix"
    )
    runs_push, runs_remote_ci = _flow_push_remote_steps(
        recorder.run_flow, config.flow_name
    )
    raw_frozen_extra = frozen_root.get("extra") if isinstance(frozen_root, dict) else None
    frozen_extra: dict[str, Any] = raw_frozen_extra if isinstance(raw_frozen_extra, dict) else {}
    if frozen_extra.get("partial") is True:
        status = "partial"
    frozen_phase_events = frozen_extra.get("phase_events")
    recorder_provenance = archive_recorder_provenance_from_snapshot(
        write_snapshot=write_snapshot,
        run_flow=recorder.run_flow,
    )
    phase_states = derive_phase_states(
        target_dir,
        phase_events=frozen_phase_events,
        runs_merge=runs_merge,
        runs_fix=runs_fix,
        runs_test=runs_test,
        runs_push=runs_push,
        runs_remote_ci=runs_remote_ci,
        session_id=recorder_provenance.session_id,
        pr_repo=recorder_provenance.pr_repo,
        pr_number=recorder_provenance.pr_number,
    )
    pipeline_status = derive_pipeline_status(
        status,
        fix_failures,
        phase_states,
        runs_merge=runs_merge,
        runs_fix=runs_fix,
        runs_test=runs_test,
    )

    # 4. Build and write manifest
    source_path = str(work.source) if work is not None else str(target_dir)
    manifest = build_manifest_from_snapshot(
        recorder_provenance=recorder_provenance,
        write_snapshot=write_snapshot,
        config=config,
        git_ctx=git_ctx,
        status=status,
        archive_path=run_dir,
        evaluation=evaluation,
        source_path=source_path,
        cwd=str(work.repo) if work is not None else None,
        fix_failures=fix_failures,
        fix_leftover_untracked=fix_leftover_untracked,
        fix_quality_gate=fix_quality_gate,
        recommended_capture=(recommended_capture or {}).get("capture_point"),
        pipeline_status=pipeline_status,
        phase_states=phase_states,
        provenance=provenance,
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    # 5. Index in SQLite
    upsert_run(archive_dir, manifest)

    # 5b. Opt-in HuggingFace dataset repo upload of the complete bundle. Fires
    #     only when centralized archiving is enabled AND this is not a
    #     signal-flush (partial) archive — a blocking network upload (create_repo
    #     + upload_folder, retries) must never run inside the SIGINT/SIGTERM
    #     handler path. The uploader is internally non-fatal (never raises), and
    #     the surrounding archive try/except is a second net.
    from daydream.archive import hub

    if config.archive and upload:
        hub_repo_id = hub.resolve_hub_repo(config)
        if hub_repo_id:
            hub.upload_run_bundle(run_dir, hub_repo_id, recorder.session_id)

    # 6. Optionally copy the fully-assembled bundle to a user-specified directory
    #    (``--dump-artifacts``) so CI can upload it. Copied wholesale from run_dir
    #    so it includes the manifest and evaluation written above.
    if config.dump_artifacts:
        dest = Path(config.dump_artifacts)
        from daydream.archive import scan

        # Fail-closed: a bundle whose serialized artifacts carry a credential is
        # never copied to a user-specified directory. The run itself is already
        # archived; the dump is skipped with a value-free warning (M11/M12).
        scan_result = scan.scan_run_dir(run_dir)
        if not scan_result.clean:
            _warn(
                f"Refusing --dump-artifacts copy of {recorder.session_id}: bundle "
                f"secret scan found problems ({scan_result.summary()})"
            )
        else:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_dir, dest, dirs_exist_ok=True)


def _read_json_artifact(path: Path, expected_type: type) -> Any | None:
    """Read a JSON artifact from *path*, returning ``None`` when absent, empty, or malformed."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, expected_type) or not data:
        return None
    return data


def _read_fix_failures(target_dir: Path) -> dict[str, str] | None:
    """Read ``deep/fix-failures.json`` from the source tree, if present.

    Written by the deep orchestrator when ``phase_fix_parallel`` dropped one or
    more file-groups. Returns the parsed ``{file_group: reason}`` map, or
    ``None`` when the file is absent, empty, or malformed — any of which means
    "no recorded fix failures" and leaves the run status untouched.
    """
    # Imported here (not at module level) to avoid pulling the deep package into
    # the archive import graph for non-deep runs.
    from daydream.deep.artifacts import fix_failures_path

    data = _read_json_artifact(fix_failures_path(target_dir / ".daydream" / "deep"), dict)
    if data is None:
        return None
    return {str(k): str(v) for k, v in data.items()}


def _read_fix_leftover_untracked(target_dir: Path) -> list[str] | None:
    """Read ``deep/fix-leftover-untracked.json`` from the source tree, if present.

    Written by the deep orchestrator alongside ``fix-failures.json`` when a
    failed fix pass left untracked files behind. Returns the parsed sorted list
    of paths, or ``None`` when the file is absent, empty, or malformed.
    """
    from daydream.deep.artifacts import fix_leftover_untracked_path

    data = _read_json_artifact(fix_leftover_untracked_path(target_dir / ".daydream" / "deep"), list)
    if data is None:
        return None
    return [str(p) for p in data]


def _read_session_bound_json_artifact(
    target_dir: Path, session_id: str | None, resolver: Callable[[Path], Path]
) -> dict[str, Any] | None:
    """Read a session-bound ``deep/*.json`` sidecar if it matches this run.

    Shared reader for the deep-flow sidecar artifacts (fix-quality-gate,
    recommended-capture): parses the JSON written at ``resolver`` under
    ``<target_dir>/.daydream/deep`` and returns it only when its
    ``session_id`` matches the current run's -- an artifact left behind by a
    DIFFERENT session (e.g. a prior deep run on the same target repo) must not
    be attributed to this run (#329). Returns ``None`` when the file is
    absent, empty, malformed, unbound (no ``session_id`` key), or bound to
    another session.
    """
    if session_id is None:
        return None
    data = _read_json_artifact(resolver(target_dir / ".daydream" / "deep"), dict)
    if data is None:
        return None
    if not isinstance(data, dict) or data.get("session_id") != session_id:
        return None
    return data


def _read_fix_quality_gate(target_dir: Path, session_id: str | None) -> dict[str, Any] | None:
    """Read ``deep/fix-quality-gate.json`` from the source tree, if present.

    Written by the deep orchestrator's fix-phase anti-degradation gate (#315):
    ``{"enabled": bool, "session_id": ..., "rounds": [...]}`` carrying per-file
    before/after erosion + verbosity deltas over the files the fix phase
    edited. Returns the parsed dict only when its ``session_id`` matches the
    current run's -- an artifact left behind by a DIFFERENT session (e.g. a
    prior deep run on the same target repo) must not be attributed to this run
    (#329). Returns ``None`` when the file is absent, empty, malformed, unbound
    (no ``session_id`` key), or bound to another session.
    """
    from daydream.deep.artifacts import fix_quality_gate_path

    return _read_session_bound_json_artifact(target_dir, session_id, fix_quality_gate_path)


def _read_recommended_capture(target_dir: Path, session_id: str | None) -> dict[str, Any] | None:
    """Read ``deep/recommended-capture.json`` from the source tree, if present.

    Written by the deep orchestrator's best-effort post-test re-capture
    (#743): ``{"session_id": ..., "capture_point": "post_test"}`` recording
    which tree produced the archived ``recommended.patch``. Returns the parsed
    dict only when its ``session_id`` matches the current run's -- an artifact
    left behind by a DIFFERENT session must not be attributed to this run.
    Returns ``None`` when the file is absent, empty, malformed, unbound, or
    bound to another session (mirrors :func:`_read_fix_quality_gate`).
    """
    from daydream.deep.artifacts import recommended_capture_path

    return _read_session_bound_json_artifact(target_dir, session_id, recommended_capture_path)


def _copy_bundle(
    target_dir: Path,
    run_dir: Path,
    recorder: TrajectoryRecorder,
    config: RunConfig,
    *,
    write_snapshot: RunWriteSnapshot | None = None,
) -> None:
    """Copy ``.daydream/`` artifacts to the archive run directory.

    Production callbacks project the exact frozen ``RunWriteSnapshot`` bytes
    into ``trajectory.json`` and ``trajectories/*.json``; direct legacy callers
    without a snapshot retain the prior live-tree copy behavior. Other artifacts
    (``review-output.md``, ``deep/``, ``diff.patch``, ``findings.json``)
    keep their existing copy logic. Diagram-only runs retain prior deep-review
    state in the live tree, so they archive only ``diagram.json`` and
    ``diagram.md`` from that directory. Missing files are silently skipped.
    """
    daydream_dir = target_dir / ".daydream"

    live_run_dir = daydream_dir / "runs" / recorder.session_id
    if write_snapshot is None:
        # Compatibility path for direct bundle-copy callers. Production archive
        # callbacks always supply the immutable run snapshot.
        if live_run_dir.is_dir():
            shutil.copytree(live_run_dir, run_dir, dirs_exist_ok=True)
        if recorder.explicit_path and recorder.path.is_file():
            try:
                resolved = recorder.path.resolve()
                inside_run_dir = resolved.is_relative_to(live_run_dir.resolve())
            except (OSError, ValueError):
                inside_run_dir = False
            if not inside_run_dir:
                shutil.copy2(recorder.path, run_dir / "trajectory.json")
    else:
        root_path = run_dir / "trajectory.json"
        root_path.unlink(missing_ok=True)
        shutil.rmtree(run_dir / "trajectories", ignore_errors=True)
        seen_destinations: set[Path] = set()
        for document in write_snapshot.documents:
            payload = json.loads(document.json_bytes)
            if not isinstance(payload, dict) or payload.get("trajectory_id") != document.trajectory_id:
                raise ValueError("invalid frozen trajectory document identity")
            if document.trajectory_id == write_snapshot.root_trajectory_id:
                destination = root_path
            else:
                name = document.path.name
                if name.endswith(".json.partial"):
                    name = name.removesuffix(".partial")
                if not name.endswith(".json") or name in {".", ".."}:
                    raise ValueError("invalid frozen trajectory document path")
                destination = run_dir / "trajectories" / name
            if destination in seen_destinations:
                raise ValueError("duplicate frozen trajectory destination")
            seen_destinations.add(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(document.json_bytes)

    diagram_only = recorder.run_flow is DaydreamRunFlow.DIAGRAM
    deep_dir = daydream_dir / "deep"
    if diagram_only:
        from daydream.deep.artifacts import diagram_markdown_path, diagram_path

        for source in (diagram_path(deep_dir), diagram_markdown_path(deep_dir)):
            if source.is_file():
                destination = run_dir / "deep" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    elif deep_dir.is_dir():
        shutil.copytree(deep_dir, run_dir / "deep", dirs_exist_ok=True)

    # Review output (in target root, not .daydream/)
    review_output = target_dir / REVIEW_OUTPUT_FILE
    if not diagram_only and review_output.is_file():
        shutil.copy2(review_output, run_dir / "review-output.md")

    # Diff patch (the PR-under-review diff, captured before fixes)
    diff_patch = daydream_dir / "diff.patch"
    if diff_patch.is_file():
        shutil.copy2(diff_patch, run_dir / "diff.patch")

    # Recommended-change patch (daydream's proposed diff, captured after fixes)
    recommended_patch = daydream_dir / "recommended.patch"
    if not diagram_only and recommended_patch.is_file():
        shutil.copy2(recommended_patch, run_dir / "recommended.patch")

    # Findings artifact (``--findings-out`` / Phase A). Archived so the corpus
    # harvest per-finding join has a fingerprint source for real PR runs —
    # without it ``_row_recorded_fingerprints`` always returns ``[]`` and the
    # per-finding supervision never reaches the corpus. The writer resolves
    # ``config.findings_out`` against CWD (the repo root in the review-bot
    # workflow), so fall back to target_dir-relative for runs invoked elsewhere.
    findings_out = config.findings_out
    if findings_out:
        findings_src = Path(findings_out)
        if not findings_src.is_absolute() and not findings_src.is_file():
            findings_src = target_dir / findings_out
        if findings_src.is_file():
            shutil.copy2(findings_src, run_dir / "findings.json")


def _run_eval(
    target_dir: Path,
    session_id: str,
    run_dir: Path,
    write_snapshot: RunWriteSnapshot,
) -> dict[str, Any] | None:
    """Run deterministic evaluation analysis and write results to the archive."""
    try:
        from daydream.eval.analyzer import analyze_session
        from daydream.trajectory import snapshot_trajectories

        daydream_dir = target_dir / ".daydream"
        result = analyze_session(
            daydream_dir,
            session_id=session_id,
            frozen_trajectories=snapshot_trajectories(write_snapshot),
        )
        eval_path = run_dir / "evaluation.json"
        eval_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except Exception:  # noqa: BLE001 - eval failure should not block archive
        logger.exception(
            "eval analysis failed for session %s; archive missing evaluation.json",
            session_id,
        )
        from daydream.ui import create_console, print_warning

        print_warning(
            create_console(),
            f"Evaluation failed for session {session_id}; archive missing evaluation.json",
        )
        return None
