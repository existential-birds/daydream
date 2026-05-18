"""Fixture archive builder for daydream.training tests.

Materializes the SPEC §9 matrix on disk: ``index.db`` via real
``upsert_run`` calls, one ``manifest.json`` per session, and a minimal
valid ``trajectory.json`` per session. Tests build this archive into a
``tmp_path`` and then query it through the same public helpers the real
exporter will use — no SQLite-, archive-, or filesystem-level mocking.

The matrix is exported as ``FIXTURE_SESSIONS`` so test modules can
reference the expected session IDs / repos / labels without hard-coding
them in two places.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from daydream.archive.index import upsert_run
from daydream.archive.manifest import Manifest

ARCHIVED_AT = "2026-05-17T00:00:00+00:00"


@dataclass(frozen=True)
class FixtureSession:
    """One row of the §9 fixture matrix."""

    session_id: str
    repo_slug: str
    skill: str
    grounding_rate: float
    outcome_labels: tuple[str, ...]
    status: str = "complete"
    notes: str = ""


def _react_stratification_sessions() -> list[FixtureSession]:
    """Generate the 8 ``fff-react-NNN`` sessions used by stratification tests."""
    sessions: list[FixtureSession] = []
    for n in range(1, 9):
        sessions.append(
            FixtureSession(
                session_id=f"fff-react-{n:03d}",
                repo_slug="someorg/react-app",
                skill="beagle-react:review-frontend",
                grounding_rate=0.9,
                outcome_labels=("accepted",),
                notes="Stratification (react > 60%)",
            )
        )
    return sessions


FIXTURE_SESSIONS: list[FixtureSession] = [
    FixtureSession(
        session_id="aaa-python-accepted",
        repo_slug="someorg/python-app",
        skill="beagle-python:review-python",
        grounding_rate=0.9,
        outcome_labels=("accepted",),
        notes="Happy path (python)",
    ),
    FixtureSession(
        session_id="bbb-react-rejected",
        repo_slug="someorg/react-app",
        skill="beagle-react:review-frontend",
        grounding_rate=0.6,
        outcome_labels=("rejected",),
        notes="Filter-out",
    ),
    FixtureSession(
        session_id="ccc-python-low-grounding",
        repo_slug="someorg/python-app",
        skill="beagle-python:review-python",
        grounding_rate=0.3,
        outcome_labels=("accepted",),
        notes="min-grounding cutoff",
    ),
    FixtureSession(
        session_id="ddd-on-exclusion",
        repo_slug="getsentry/sentry",
        skill="beagle-python:review-python",
        grounding_rate=0.95,
        outcome_labels=("accepted",),
        notes="C5 — must never appear",
    ),
    FixtureSession(
        session_id="eee-copyleft",
        repo_slug="gnu/coreutils",
        skill="beagle-python:review-python",
        grounding_rate=0.9,
        outcome_labels=("accepted",),
        notes="C8 — needs --allow-copyleft",
    ),
    *_react_stratification_sessions(),
]


_MINIMAL_TRAJECTORY: dict[str, object] = {
    "schema_version": "1.6",
    "steps": [
        {
            "step_id": 1,
            "source": "agent",
            "reasoning_content": "thinking",
            "message": "",
            "tool_calls": [{"name": "Bash", "arguments": {}}],
        }
    ],
}


def _build_manifest(session: FixtureSession, session_dir: Path) -> Manifest:
    """Construct a Manifest for one fixture session."""
    return Manifest(
        session_id=session.session_id,
        archived_at=ARCHIVED_AT,
        status=session.status,
        skill=session.skill,
        repo_slug=session.repo_slug,
        branch="feat/x",
        base_branch="main",
        head_sha="abc123",
        grounding_rate=session.grounding_rate,
        outcome_labels=json.dumps(list(session.outcome_labels)),
        archive_path=str(session_dir),
    )


def build_fixture_archive(root: Path) -> None:
    """Materialize the §9 fixture matrix under *root*.

    Creates ``root/runs/<session_id>/`` for every entry of
    ``FIXTURE_SESSIONS``, writes ``manifest.json`` + a minimal
    ``trajectory.json`` to each, and upserts the row into
    ``root/index.db`` via the production ``upsert_run`` helper.

    Args:
        root: Directory that will play the role of the daydream archive
            root (e.g. a pytest ``tmp_path``). Created on demand.
    """
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for session in FIXTURE_SESSIONS:
        session_dir = runs_dir / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        manifest = _build_manifest(session, session_dir)
        (session_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2),
            encoding="utf-8",
        )
        (session_dir / "trajectory.json").write_text(
            json.dumps(_MINIMAL_TRAJECTORY, indent=2),
            encoding="utf-8",
        )
        upsert_run(root, manifest)
