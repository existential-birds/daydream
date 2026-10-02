"""Artifact path helpers for the improve advisor flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.artifact_visibility import ArtifactSession, artifact_dir_for
from daydream.trajectory import DaydreamPhase, get_current_recorder


def improve_dir(
    target: Path, *, session: ArtifactSession | None = None, allow_standalone: bool = False,
) -> Path:
    """Return the target's ``.daydream/improve`` directory, creating it."""
    directory = artifact_dir_for(target, session=session, allow_standalone=allow_standalone) / "improve"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


RECON_FILENAME = 'recon.json'
COVERAGE_FILENAME = 'coverage.json'
VETTED_FINDINGS_FILENAME = 'vetted-findings.json'
REPORT_FILENAME = 'report.md'
PLAN_WRITE_DIAGNOSTICS_FILENAME = 'plan-write-diagnostics.json'
PUBLISHED_ISSUES_FILENAME = 'published-issues.json'


def _artifact_provenance(*, phase: DaydreamPhase) -> dict[str, str]:
    """Return host-authored identity tying an improve artifact to this run."""
    recorder = get_current_recorder()
    if recorder is None:
        return {"session_id": "unrecorded", "phase": phase.value}
    try:
        trajectory_path = recorder.path.relative_to(recorder.target_dir).as_posix()
    except ValueError:
        trajectory_path = str(recorder.path)
    return {
        "session_id": recorder.session_id,
        "phase": phase.value,
        "trajectory_path": trajectory_path,
    }


def write_artifact(path: Path, payload: dict[str, Any], *, phase: DaydreamPhase) -> dict[str, Any]:
    """Persist one Improve JSON document with host-owned run/phase identity."""
    document = {"artifact_provenance": _artifact_provenance(phase=phase), **payload}
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document
