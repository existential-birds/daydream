"""Centralized run archive for daydream.

Copies the full artifact bundle (trajectory, review output, deep artifacts,
diff) from one run's frozen artifact tree to
``~/.daydream/archive/runs/{session_id}/``, writes a ``manifest.json``, and
indexes the run in a SQLite database for cross-project querying. Assembly is
strict and transactional: it either publishes a complete, attested bundle or
raises :class:`ArchiveFinalizationError` and leaves nothing behind.

Exports:
    finalize_archive_run: The single archive entry point, called once per run
        from the runner's artifact finalization boundary.
    get_archive_dir: Returns the archive root directory, creating it on
        first access.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from daydream.archive.git_context import capture_git_context
from daydream.archive.index import upsert_run
from daydream.archive.manifest import build_manifest_from_snapshot
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.run_snapshot import ArchiveRunSnapshot
from daydream.trajectory import DaydreamRunFlow

if TYPE_CHECKING:
    from daydream.artifact_visibility import (
        ArtifactEvidenceProvenance,
        ArtifactTreeSnapshot,
    )
    from daydream.runner import RunConfig
    from daydream.trajectory import RunWriteSnapshot
    from daydream.workspace import WorkContext


class ArchiveFinalizationError(RuntimeError):
    """Strict host archive finalization did not complete successfully."""


def _warn(message: str) -> None:
    """Print a one-line warning through the daydream console (never raises).

    Lazily imported (mirroring ``hub._warn``) so the archive import graph does
    not pull the UI package in for callers that never warn.
    """
    from daydream.ui import create_console, print_warning

    print_warning(create_console(), message)


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


def _validate_frozen_artifacts(artifacts: ArtifactTreeSnapshot) -> None:
    """Reject a frozen tree that changed before a strict host consumer."""
    from daydream.artifact_visibility import manifest_tree

    if manifest_tree(artifacts.root) != artifacts.manifest:
        raise ArchiveFinalizationError("frozen artifact tree changed before archive")


def _copy_snapshot_bundle(
    *,
    run: ArchiveRunSnapshot,
    artifacts: ArtifactTreeSnapshot,
    artifact_provenance: ArtifactEvidenceProvenance,
    run_dir: Path,
) -> None:
    """Assemble an archive only from frozen tree and immutable document bytes.

    The registered findings destination is relocated from its live write path
    into the frozen tree; public output paths are never reconstructed.
    """
    from daydream.artifact_visibility import OutputLabel

    recorder_provenance = run.recorder_provenance
    _project_documents(
        run.trajectories,
        run_dir,
        session_id=recorder_provenance.session_id,
    )
    findings_routes = [
        route for route in artifacts.destinations if route.label is OutputLabel.FINDINGS_OUTPUT
    ]
    findings_src: Path | None = None
    if findings_routes:
        if len(findings_routes) != 1 or findings_routes[0].frozen_path is None:
            raise ArchiveFinalizationError("frozen artifact destination is malformed")
        try:
            relative = findings_routes[0].frozen_path.relative_to(artifact_provenance.live_root)
        except ValueError as exc:
            raise ArchiveFinalizationError("frozen artifact destination escaped its run") from exc
        findings_src = artifacts.root / relative
    _copy_run_artifacts(
        artifacts.root,
        run_dir,
        diagram_only=recorder_provenance.run_flow is DaydreamRunFlow.DIAGRAM,
        findings_src=findings_src,
    )


def _manifest_state(
    *,
    target_dir: Path,
    run: ArchiveRunSnapshot,
    frozen_extra: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the status, fix, and pipeline manifest fields for one run tree.

    Derivation is gated to the phases this registered flow can execute, and
    every sidecar read is session-bound, so a non-deep or interrupted run never
    adopts prior state. The ``derive_*`` helpers never raise on absent or
    malformed artifacts, so this can never abort an archive.
    """
    from daydream.archive.pipeline import derive_phase_states, derive_pipeline_status

    recorder_provenance = run.recorder_provenance
    phases = run.identity.phases
    session_id = recorder_provenance.session_id
    status = run.trajectories.status
    runs_merge = phases.merge
    runs_fix = phases.fix
    runs_test = phases.test
    runs_push = phases.push
    runs_remote_ci = phases.remote_ci
    # A deep fix run that hit per-group failures left partial/reverted edits in
    # the tree; the run is NOT "complete".
    fix_failures = _read_fix_failures(target_dir) if runs_fix else None
    recommended = _read_recommended_capture(target_dir, session_id) if runs_fix else None
    if fix_failures or frozen_extra.get("partial") is True:
        status = "partial"
    phase_states = derive_phase_states(
        target_dir,
        phase_events=frozen_extra.get("phase_events"),
        runs_merge=runs_merge,
        runs_fix=runs_fix,
        runs_test=runs_test,
        runs_push=runs_push,
        runs_remote_ci=runs_remote_ci,
        session_id=session_id,
        pr_repo=recorder_provenance.pr_repo,
        pr_number=recorder_provenance.pr_number,
    )
    return {
        "status": status,
        "fix_failures": fix_failures,
        "fix_leftover_untracked": _read_fix_leftover_untracked(target_dir) if runs_fix else None,
        "fix_quality_gate": _read_fix_quality_gate(target_dir, session_id) if runs_fix else None,
        "recommended_capture": (recommended or {}).get("capture_point"),
        "phase_states": phase_states,
        "pipeline_status": derive_pipeline_status(
            status,
            fix_failures,
            phase_states,
            runs_merge=runs_merge,
            runs_fix=runs_fix,
            runs_test=runs_test,
        ),
    }


def _discard_or_note(path: Path, exc: BaseException, stage: str, *, recreate: bool = False) -> None:
    """Remove one owned private path, noting a failure on the pending error."""
    try:
        shutil.rmtree(path)
        if recreate:
            path.mkdir(mode=0o700)
    except OSError as cleanup_error:
        exc.add_note(
            f"strict {stage} cleanup retained private evidence ({type(cleanup_error).__name__})"
        )


def finalize_archive_run(
    *,
    run: ArchiveRunSnapshot,
    artifacts: ArtifactTreeSnapshot,
    artifact_provenance: ArtifactEvidenceProvenance,
    config: RunConfig,
    work: WorkContext | None,
    upload: bool = True,
    dump_path: Path | None = None,
) -> None:
    """Strictly archive one immutable run or raise a closed typed error."""
    recorder_provenance = run.recorder_provenance
    write_snapshot = run.trajectories
    session_id = recorder_provenance.session_id
    if (
        session_id != artifacts.session_id
        or session_id != write_snapshot.root_trajectory_id
        or session_id != artifact_provenance.session_id
        or artifacts.workspace_key != artifact_provenance.workspace_key
        or (work is not None and artifact_provenance.public_source != work.source)
    ):
        raise ArchiveFinalizationError("archive identity mismatch")
    _validate_frozen_artifacts(artifacts)
    if not config.archive and not config.dump_artifacts:
        return
    run_dir: Path | None = None
    assembly_dir: Path | None = None
    assembly_created = False
    archive_installed = False
    dump_started = False
    try:
        from daydream.archive.provenance import capture_executable_provenance
        from daydream.trajectory import snapshot_trajectories

        archive_dir = get_archive_dir()
        run_dir = archive_dir / "runs" / session_id
        assembly_dir = run_dir.with_name(f".{run_dir.name}.finalizing")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        for owned in (run_dir, assembly_dir):
            if owned.exists() or owned.is_symlink():
                raise ArchiveFinalizationError("archive session destination already exists")
        assembly_dir.mkdir()
        assembly_created = True
        _copy_snapshot_bundle(
            run=run,
            artifacts=artifacts,
            artifact_provenance=artifact_provenance,
            run_dir=assembly_dir,
        )
        target_dir = artifacts.root
        git_ctx = capture_git_context(work.repo if work is not None else target_dir)
        if work is not None:
            git_ctx.base_branch = work.base_branch
            if git_ctx.base_sha is None:
                git_ctx.base_sha = work.base_sha
        frozen = snapshot_trajectories(write_snapshot)
        evaluation: dict[str, Any] | None = None
        if config.run_eval and recorder_provenance.run_flow is not DaydreamRunFlow.DIAGRAM:
            # A failed evaluation is a closed archive failure, not the legacy
            # warning-and-None disposition, and the evaluator must not be able to
            # change the frozen tree it read.
            try:
                from daydream.eval.analyzer import analyze_session

                evaluation = analyze_session(
                    target_dir / ".daydream",
                    session_id=session_id,
                    frozen_trajectories=frozen,
                    artifact_provenance=artifact_provenance,
                    code_workspace=work.repo if work is not None else artifact_provenance.public_source,
                )
                if "error" in evaluation:
                    raise ValueError("evaluation returned an incomplete result")
            except Exception as exc:
                raise ArchiveFinalizationError("evaluation finalization failed") from exc
            (assembly_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
            _validate_frozen_artifacts(artifacts)
        frozen_root = frozen.get("main")
        frozen_extra = frozen_root.get("extra") if isinstance(frozen_root, dict) else None
        manifest = build_manifest_from_snapshot(
            run=run,
            git_ctx=git_ctx,
            archive_path=run_dir,
            evaluation=evaluation,
            source_path=str(work.source) if work is not None else None,
            provenance=capture_executable_provenance(),
            **_manifest_state(
                target_dir=target_dir,
                run=run,
                frozen_extra=frozen_extra if isinstance(frozen_extra, dict) else {},
            ),
        )
        (assembly_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2),
            encoding="utf-8",
        )
        if config.archive and upload:
            from daydream.archive import hub

            hub_repo_id = hub.resolve_hub_repo(config)
            _validate_frozen_artifacts(artifacts)
            if hub_repo_id:
                # A False disposition is a skip (no HF_TOKEN, no huggingface_hub),
                # a fail-closed secret refusal, or a transport failure — and
                # upload_run_bundle has already warned with the reason in every
                # case. None of them is a reason to discard a completed review:
                # issue #981 requires refusing the upload "while preserving the
                # local run", and the refusal itself happens inside the callee,
                # before anything reaches the Hub. Raising here would only throw
                # the local bundle away without containing anything extra.
                hub.upload_run_bundle(assembly_dir, hub_repo_id, session_id)
        if config.dump_artifacts:
            if dump_path is None:
                raise ArchiveFinalizationError("dump finalization path is missing")
            from daydream.archive import scan

            scan_result = scan.scan_run_dir(assembly_dir)
            if scan_result.blocking:
                raise ArchiveFinalizationError(
                    f"dump artifact secret scan refused publication ({scan_result.summary()})"
                )
            if scan_result.findings:
                _warn(
                    "Publishing the dump for "
                    f"{recorder_provenance.session_id} with advisory secret-scan findings "
                    f"({scan_result.summary()})"
                )
            dump_started = True
            shutil.copytree(assembly_dir, dump_path, dirs_exist_ok=True)
        _validate_frozen_artifacts(artifacts)
        os.replace(assembly_dir, run_dir)
        assembly_created = False
        archive_installed = True
        upsert_run(archive_dir, manifest)
    except BaseException as exc:
        owned_paths: tuple[Path | None, ...] = (
            assembly_dir if assembly_created else None,
            run_dir if archive_installed else None,
        )
        for private in owned_paths:
            if private is not None and private.is_dir() and not private.is_symlink():
                _discard_or_note(private, exc, "archive")
        if dump_started and dump_path is not None and dump_path.is_dir():
            _discard_or_note(dump_path, exc, "dump", recreate=True)
        if isinstance(exc, ArchiveFinalizationError) or not isinstance(exc, Exception):
            raise
        raise ArchiveFinalizationError("archive finalization failed") from exc


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


def _project_documents(
    write_snapshot: RunWriteSnapshot,
    run_dir: Path,
    *,
    session_id: str,
) -> None:
    """Project the exact frozen trajectory bytes into one archive bundle.

    ``session_id`` additionally binds every document to the archived run.
    """
    root_path = run_dir / "trajectory.json"
    root_path.unlink(missing_ok=True)
    shutil.rmtree(run_dir / "trajectories", ignore_errors=True)
    seen: set[Path] = set()
    for document in write_snapshot.documents:
        payload = json.loads(document.json_bytes)
        if (
            not isinstance(payload, dict)
            or payload.get("trajectory_id") != document.trajectory_id
            or payload.get("session_id") != session_id
        ):
            raise ValueError("invalid frozen trajectory document identity")
        if document.trajectory_id == write_snapshot.root_trajectory_id:
            destination = root_path
        else:
            name = document.path.name.removesuffix(".partial")
            if not name.endswith(".json") or name in {".", ".."}:
                raise ValueError("invalid frozen trajectory document path")
            destination = run_dir / "trajectories" / name
        if destination in seen:
            raise ValueError("duplicate frozen trajectory destination")
        seen.add(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(document.json_bytes)


def _copy_run_artifacts(
    target_dir: Path,
    run_dir: Path,
    *,
    diagram_only: bool,
    findings_src: Path | None,
) -> None:
    """Copy one run's non-trajectory artifacts to the archive run directory.

    Diagram-only runs retain prior deep-review state in the tree, so they
    archive only ``diagram.json`` and ``diagram.md`` from that directory, and
    neither the review output nor the recommended patch. Missing files are
    silently skipped.
    """
    daydream_dir = target_dir / ".daydream"
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
    # per-finding supervision never reaches the corpus.
    if findings_src is not None and findings_src.is_file():
        shutil.copy2(findings_src, run_dir / "findings.json")
