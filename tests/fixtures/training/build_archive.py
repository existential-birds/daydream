"""Shared training fixture sessions, exported as FIXTURE_SESSIONS for consistent IDs, repositories, and
labels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixtureSession:
    """One training fixture session with repository, review skill, and outcome labels."""

    session_id: str
    repo_slug: str
    skill: str
    outcome_labels: tuple[str, ...]


FIXTURE_SESSIONS: list[FixtureSession] = [FixtureSession(
        session_id="aaa-python-accepted", repo_slug="someorg/python-app", skill="beagle-python:review-python",
        outcome_labels=("accepted",),
    ),
    FixtureSession(session_id="bbb-react-rejected", repo_slug="someorg/react-app", skill="beagle-react:review-frontend",
        outcome_labels=("rejected",),
    ),
    FixtureSession(
        session_id="ccc-python-low-grounding", repo_slug="someorg/python-app", skill="beagle-python:review-python",
        outcome_labels=("accepted",),
    ),
]


_MINIMAL_TRAJECTORY: dict[str, object] = {"schema_version": "1.6",
    "steps": [{"step_id": 1, "source": "agent", "reasoning_content": "thinking", "message": "",
            "tool_calls": [{"name": "Bash", "arguments": {}}],
        }
    ],
}
