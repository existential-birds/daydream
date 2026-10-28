"""Finalize frozen run bundles and publish optional archive outputs."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.archive.bundle import _copy_snapshot_bundle, _validate_frozen_artifacts
from daydream.archive.errors import (
    ArchiveFinalizationError,
    ArchiveIntegrityError,
    ArchivePublicationError,
)
from daydream.archive.git_context import GitContext, capture_git_context
from daydream.archive.index import upsert_run
from daydream.archive.manifest import build_manifest_from_snapshot
from daydream.archive.pipeline import _manifest_state
from daydream.artifacts.models import ArtifactSourceUnavailableError
from daydream.observability.runtime import current_session
from daydream.observability.spans import SpanScope
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
    from daydream.artifacts.ownership import CodeWorkspace
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


def _validate_evaluation(
    evaluation: dict[str, Any], *, session_id: str, trajectory_count: int, display_path: Path,
) -> None:
    """Require a complete analyzer result bound to the archive's selected evidence."""
    sections = (
        "agent", "cost", "timing", "tools", "findings", "location", "routing", "training_signals", "quality", "derived",
    )
    if (
        not isinstance(evaluation, dict)
        or "error" in evaluation
        or evaluation.get("session_id") != session_id
        or type(evaluation.get("trajectory_count")) is not int
        or evaluation["trajectory_count"] != trajectory_count
        or evaluation.get("daydream_dir") != str(display_path)
        or "latency_profile" not in evaluation
        or evaluation["latency_profile"] is not None and not isinstance(evaluation["latency_profile"], str)
        or any(not isinstance(evaluation.get(section), dict) for section in sections)
    ):
        raise ValueError("evaluation returned a malformed or unbound result")
    findings = evaluation["findings"]
    duplicates = findings.get("shipped_duplication")
    if not isinstance(duplicates, dict):
        raise ValueError("evaluation returned malformed finding metrics")
    for section, key, nullable in ((findings, "total", False), (duplicates, "near_duplicate_pairs", True)):
        value = section.get(key)
        if (
            key not in section or value is None and not nullable
            or value is not None and (type(value) is not int or not 0 <= value < 2**63)
        ):
            raise ValueError("evaluation returned a malformed count")
    quality = evaluation["quality"]
    # Validate the nullable numeric outcomes consumed by the manifest/index.
    metrics = (
        (evaluation["timing"], "total_wall_clock_seconds", None),
        (evaluation["location"], "in_hunk_rate", 1),
        (quality, "erosion", 1), (quality, "verbosity", 1),
        (evaluation["derived"], "cost_per_finding_usd", None),
    )
    for section, key, maximum in metrics:
        value = section.get(key)
        if key not in section or value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            or maximum is not None and value > maximum
        ):
            raise ValueError("evaluation returned a malformed metric")
    if (
        not isinstance(quality.get("per_file"), dict)
        or not isinstance(quality.get("calibration"), dict)
        or type(quality.get("scoped_files")) is not int
        or quality["scoped_files"] < 0
        or "unavailable" in quality and type(quality["unavailable"]) is not bool
        or quality.get("unavailable") is True and (
            quality["erosion"] is not None or quality["verbosity"] is not None
            or quality["per_file"] or quality["scoped_files"] != 0
            or not isinstance(quality.get("error"), str) or not quality["error"].strip()
        )
    ):
        raise ValueError("evaluation returned malformed quality")


def finalize_archive_run(
    *,
    run: ArchiveRunSnapshot,
    artifacts: ArtifactTreeSnapshot,
    artifact_provenance: ArtifactEvidenceProvenance,
    config: RunConfig,
    work: WorkContext | None,
    dump_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
    code_workspace: CodeWorkspace | None = None,
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

        if progress is not None:
            progress("Assembling run archive")
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
        try:
            git_ctx = (
                code_workspace.run(partial(capture_git_context, code_workspace.path))
                if code_workspace is not None
                else capture_git_context(work.repo if work is not None else target_dir)
            )
        except ArtifactSourceUnavailableError:
            git_ctx = GitContext(
                branch=work.head_branch if work is not None else None,
                head_sha=work.head_sha if work is not None else None,
            )
        if work is not None:
            git_ctx.base_branch = work.base_branch
            if git_ctx.base_sha is None:
                git_ctx.base_sha = work.base_sha
        frozen = snapshot_trajectories(write_snapshot)
        evaluation: dict[str, Any] | None = None
        if config.run_eval and recorder_provenance.run_flow is not DaydreamRunFlow.DIAGRAM:
            if progress is not None:
                progress("Evaluating run")
            # Evaluation failure aborts the archive; revalidate its frozen input afterward.
            try:
                from daydream.eval.analyzer import analyze_session

                # Omit optional labels so this worker inherits selected-stack attribution.
                with SpanScope(
                    current_session(), "daydream.step.evaluation-analysis", "step",
                    {"daydream.step": "evaluation-analysis", "daydream.phase": "evaluation-analysis"},
                ) as observed:
                    evaluation = analyze_session(
                        target_dir / ".daydream",
                        write_snapshot=write_snapshot,
                        artifact_provenance=artifact_provenance,
                        code_workspace=code_workspace or (
                            work.repo if work is not None else artifact_provenance.public_source
                        ),
                    )
                    _validate_evaluation(
                        evaluation, session_id=session_id, trajectory_count=len(write_snapshot.documents),
                        display_path=artifact_provenance.public_daydream_dir,
                    )
                    observed.finish()
                if evaluation.get("quality", {}).get("unavailable") and progress is not None:
                    progress("Live-source quality unavailable; frozen analysis retained")
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
            if progress is not None:
                progress("Preparing artifact dump")
            if dump_path is None:
                raise ArchivePublicationError("dump finalization path is missing")
            # Mark copying started before I/O so publication failures clean the stage.
            dump_started = True
            try:
                shutil.copytree(assembly_dir, dump_path, dirs_exist_ok=True)
            except Exception as exc:
                raise ArchivePublicationError("dump publication failed") from exc
        _validate_frozen_artifacts(artifacts)

        if progress is not None:
            progress("Publishing run archive")
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
