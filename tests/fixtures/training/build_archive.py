"""SPEC §9 fixture matrix for daydream.training tests.

The matrix is exported as ``FIXTURE_SESSIONS`` so test modules can
reference the expected session IDs / repos / labels without hard-coding
them in two places.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixtureSession:
    """One row of the §9 fixture matrix.

    Attributes:
        session_id: Unique identifier for this fixture session.
        repo_slug: GitHub-style ``org/repo`` slug.
        skill: Beagle review skill invocation string.
        grounding_rate: Fraction of findings grounded in actual code (0.0–1.0).
        outcome_labels: Tuple of outcome tags applied to the session.
        status: Archive status, e.g. ``"complete"``.
        notes: Free-text annotation describing the fixture's purpose.
    """

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
