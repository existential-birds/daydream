"""Finalize frozen run bundles and publish optional archive outputs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.archive.bundle import _copy_snapshot_bundle, _validate_frozen_artifacts
from daydream.archive.errors import (
    ArchiveFinalizationError,
    ArchiveIntegrityError,
    ArchivePublicationError,
)
from daydream.archive.git_context import capture_git_context
from daydream.archive.index import upsert_run
from daydream.archive.manifest import build_manifest_from_snapshot
from daydream.archive.pipeline import _manifest_state
from daydream.run_snapshot import ArchiveRunSnapshot
from daydream.trajectory import (
    RUNS_DIRNAME,
    DaydreamRunFlow,
    run_directory,
    snapshot_trajectories,
)

if TYPE_CHECKING:
    from daydream.artifact_visibility import (
        ArtifactEvidenceProvenance,
        ArtifactTreeSnapshot,
    )
    from daydream.run_config import RunConfig
    from daydream.workspace import WorkContext


def get_archive_dir() -> Path:
    """Create the archive root and runs directory.

    Use ``DAYDREAM_ARCHIVE_DIR`` when set, otherwise ``~/.daydream/archive``.
    """
    env = os.environ.get("DAYDREAM_ARCHIVE_DIR")
    if env:
        archive_dir = Path(env)
    else:
        archive_dir = Path.home() / ".daydream" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / RUNS_DIRNAME).mkdir(exist_ok=True)
    return archive_dir


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
        raise ArchiveIntegrityError("archive identity mismatch")
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

        archive_dir = get_archive_dir()
        run_dir = run_directory(archive_dir, session_id)
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
            # Evaluation failure aborts the archive; revalidate its frozen input afterward.
            try:
                from daydream.eval.analyzer import analyze_session

                evaluation = analyze_session(
                    target_dir / ".daydream",
                    write_snapshot=write_snapshot,
                    artifact_provenance=artifact_provenance,
                    code_workspace=work.repo if work is not None else artifact_provenance.public_source,
                )
                if "error" in evaluation:
                    raise ValueError("evaluation returned an incomplete result")
            except Exception as exc:
                _validate_frozen_artifacts(artifacts)
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
        if config.dump_artifacts:
            if dump_path is None:
                raise ArchivePublicationError("dump finalization path is missing")
            # Mark copying started before I/O so publication failures clean the stage.
            dump_started = True
            try:
                shutil.copytree(assembly_dir, dump_path, dirs_exist_ok=True)
            except Exception as exc:
                raise ArchivePublicationError("dump publication failed") from exc
        if config.archive and upload:
            from daydream.archive import hub
            from daydream.archive._console import warn

            try:
                hub_repo_id = hub.resolve_hub_repo(config)
                _validate_frozen_artifacts(artifacts)
                if hub_repo_id:
                    hub.upload_run_bundle(assembly_dir, hub_repo_id, session_id)
            except Exception as exc:
                _validate_frozen_artifacts(artifacts)
                if isinstance(exc, ArchiveIntegrityError):
                    raise
                warn(f"Data Collection: run upload failed ({type(exc).__name__})")
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
        if (
            dump_started and dump_path is not None and dump_path.is_dir()
            and (isinstance(exc, (ArchiveIntegrityError, ArchivePublicationError)) or not isinstance(exc, Exception))
        ):
            _discard_or_note(dump_path, exc, "dump", recreate=True)
        if isinstance(exc, ArchiveFinalizationError) or not isinstance(exc, Exception):
            raise
        raise ArchiveFinalizationError("archive finalization failed") from exc
