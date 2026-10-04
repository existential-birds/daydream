"""Assemble archive bundles from frozen artifacts and trajectory documents."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from daydream.archive.errors import ArchiveIntegrityError
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.run_snapshot import ArchiveRunSnapshot
from daydream.trajectory import (
    PARTIAL_SUFFIX,
    DaydreamRunFlow,
    run_document_path,
    sibling_document_path,
    siblings_directory,
)

if TYPE_CHECKING:
    from daydream.artifact_visibility import (
        ArtifactEvidenceProvenance,
        ArtifactTreeSnapshot,
    )
    from daydream.trajectory import RunWriteSnapshot


def _validate_frozen_artifacts(artifacts: ArtifactTreeSnapshot) -> None:
    """Reject a frozen tree that changed before a strict host consumer."""
    from daydream.artifact_visibility import ArtifactVisibilityError, manifest_tree

    try:
        actual = manifest_tree(artifacts.root)
    except (ArtifactVisibilityError, OSError) as exc:
        raise ArchiveIntegrityError("frozen artifact tree could not be validated") from exc
    if actual != artifacts.manifest:
        raise ArchiveIntegrityError("frozen artifact tree changed before archive")


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
    try:
        _project_documents(run.trajectories, run_dir, session_id=recorder_provenance.session_id)
    except (ValueError, UnicodeError) as exc:
        raise ArchiveIntegrityError("frozen trajectory projection failed") from exc
    findings_routes = [
        route for route in artifacts.destinations if route.label is OutputLabel.FINDINGS_OUTPUT
    ]
    findings_src: Path | None = None
    if findings_routes:
        if len(findings_routes) != 1 or findings_routes[0].frozen_path is None:
            raise ArchiveIntegrityError("frozen artifact destination is malformed")
        try:
            relative = findings_routes[0].frozen_path.relative_to(artifact_provenance.live_root)
        except ValueError as exc:
            raise ArchiveIntegrityError("frozen artifact destination escaped its run") from exc
        findings_src = artifacts.root / relative
    _copy_run_artifacts(
        artifacts.root,
        run_dir,
        diagram_only=recorder_provenance.run_flow is DaydreamRunFlow.DIAGRAM,
        findings_src=findings_src,
    )


def _project_documents(
    write_snapshot: RunWriteSnapshot,
    run_dir: Path,
    *,
    session_id: str,
) -> None:
    """Project the exact frozen trajectory bytes into one archive bundle.

    ``session_id`` additionally binds every document to the archived run.
    """
    root_path = run_document_path(run_dir)
    root_path.unlink(missing_ok=True)
    shutil.rmtree(siblings_directory(run_dir), ignore_errors=True)
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
            name = document.path.name.removesuffix(PARTIAL_SUFFIX)
            if not name.endswith(".json") or name in {".", ".."}:
                raise ValueError("invalid frozen trajectory document path")
            destination = sibling_document_path(run_dir, name)
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
        from daydream.deep.artifacts import DeepArtifact

        for source in (DeepArtifact.DIAGRAM.at(deep_dir), DeepArtifact.DIAGRAM_MARKDOWN.at(deep_dir)):
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

    # Corpus harvest needs these fingerprints to join per-finding supervision.
    if findings_src is not None and findings_src.is_file():
        shutil.copy2(findings_src, run_dir / "findings.json")
