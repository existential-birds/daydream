"""Tests for live, session-owned PR run-info acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

import daydream.pr_run_info as provider
from daydream.artifact_visibility import (
    ArtifactSession,
)
from daydream.atif import Trajectory
from daydream.pr_comment_renderer import FALLBACK_NOTE
from daydream.pr_run_info import (
    LiveRunInfoSource,
    render_live_run_info,
)
from daydream.trajectory import (
    TrajectoryDocumentSnapshot,
    TrajectoryRecorder,
)
from tests.harness.trajectory import make_recorder


def _trajectory(
    session_id: str, trajectory_id: str, *, phase: str = "review", model: str = "gpt-5.5", prompt_tokens: int = 100,
    cost_usd: float | None = 0.01,
) -> Trajectory:
    return Trajectory.model_validate({
            "schema_version": "ATIF-v1.7", "session_id": session_id, "trajectory_id": trajectory_id,
            "agent": {"name": "daydream", "version": "test", "model_name": model},
            "steps": [{"step_id": 1, "timestamp": "2026-09-11T00:00:00Z", "source": "user", "message": "review",
                    "extra": {"daydream_phase": phase},
                }, {"step_id": 2, "timestamp": "2026-09-11T00:00:01Z", "source": "agent", "message": "done",
                    "model_name": model,
                    "metrics": {"prompt_tokens": prompt_tokens, "completion_tokens": 10, "cached_tokens": 20,
                        "cost_usd": cost_usd,
                    }, "extra": {"daydream_phase": phase},
                },
            ],
        }
    )


def _recorder(tmp_path: Path, session_id: str = "run-info", *, model: str = "gpt-5.5", prompt_tokens: int = 100,
    cost_usd: float | None = 0.01,
) -> TrajectoryRecorder:
    recorder = make_recorder(tmp_path, path=tmp_path / "trajectory.json", agent_model_name=model, session_id=session_id,
    )
    recorder.steps.extend(
        _trajectory(session_id, session_id, model=model, prompt_tokens=prompt_tokens, cost_usd=cost_usd,).steps
    )
    return recorder

@pytest.mark.parametrize(("source", "diagnostic"),
    [
        (LiveRunInfoSource(recorder=_recorder(Path(".")), artifacts=None), "run info: artifact session unavailable",),
    ],
)
def test_missing_live_source_returns_bounded_unavailable_result(source: LiveRunInfoSource, diagnostic: str,) -> None:
    result = render_live_run_info(source)

    assert FALLBACK_NOTE in result.markdown
    assert result.diagnostic == diagnostic

class _SnapshotSource:
    def __init__(self, snapshots: tuple[TrajectoryDocumentSnapshot, ...] = (), *, error: Exception | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.error = error

    def snapshot_completed_sibling_trajectories(self, *, session_id: str,) -> tuple[TrajectoryDocumentSnapshot, ...]:
        if self.error is not None:
            raise self.error
        return self.snapshots


def _source_with_snapshots(
    tmp_path: Path, snapshots: tuple[TrajectoryDocumentSnapshot, ...] = (), *, error: Exception | None = None,
    recorder: TrajectoryRecorder | None = None,
) -> LiveRunInfoSource:
    return LiveRunInfoSource(recorder=recorder or _recorder(tmp_path),
        artifacts=cast(ArtifactSession, _SnapshotSource(snapshots, error=error)),
    )

def test_snapshot_failure_returns_fixed_diagnostic_without_exception_text(tmp_path: Path,) -> None:
    secret = "private/session/path/and-bad-json"
    source = _source_with_snapshots(tmp_path, error=RuntimeError(secret))
    result = render_live_run_info(source)

    assert result.diagnostic == "run info: trajectory snapshot unavailable"
    assert secret not in result.markdown
    assert secret not in (result.diagnostic or "")

def test_non_root_recorder_is_rejected_before_acquisition(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder._trajectory_id = "child-recorder"
    result = render_live_run_info(_source_with_snapshots(tmp_path, recorder=recorder))

    assert result.diagnostic == "run info: trajectory identity invalid"


@pytest.mark.parametrize(("snapshot_id", "document_session", "document_id"),
    [("child", "other-session", "child"), ("child", "run-info", "other-child"), ],
)
def test_invalid_sibling_identity_returns_unavailable(
    tmp_path: Path, snapshot_id: str, document_session: str, document_id: str,
) -> None:
    document = _trajectory(document_session, document_id)
    snapshot = TrajectoryDocumentSnapshot(snapshot_id, tmp_path / "ignored.json", document.model_dump_json().encode(),)
    result = render_live_run_info(_source_with_snapshots(tmp_path, (snapshot,)))

    assert result.diagnostic == "run info: trajectory identity invalid"
    assert FALLBACK_NOTE in result.markdown

def test_malformed_sibling_and_duplicate_ids_return_unavailable(tmp_path: Path) -> None:
    malformed = TrajectoryDocumentSnapshot("child", tmp_path / "ignored.json", b"{bad json",)
    malformed_result = render_live_run_info(_source_with_snapshots(tmp_path, (malformed,)))
    child = _trajectory("run-info", "child")
    duplicate = TrajectoryDocumentSnapshot("child", tmp_path / "ignored.json", child.model_dump_json().encode(),)
    duplicate_result = render_live_run_info(_source_with_snapshots(tmp_path, (duplicate, duplicate)))

    assert malformed_result.diagnostic == "run info: trajectory document invalid"
    assert duplicate_result.diagnostic == "run info: trajectory identity invalid"

@pytest.mark.parametrize(("operation", "message"), [
    ("render_run_info", "secret renderer failure"),
    ("load_user_prices", "private prices path"),
])
def test_pricing_or_render_failure_returns_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, message: str,
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(message)

    monkeypatch.setattr(provider, operation, fail)
    source = _source_with_snapshots(tmp_path)
    result = render_live_run_info(source)

    assert result.diagnostic == "run info: rendering unavailable"
    assert FALLBACK_NOTE in result.markdown
