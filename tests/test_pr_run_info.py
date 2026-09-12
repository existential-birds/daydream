"""Tests for live, session-owned PR run-info acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from daydream.artifact_visibility import (
    ArtifactSession,
    open_artifact_session,
    private_root_locations,
    resolve_private_workspace_owner,
)
from daydream.atif import Trajectory
from daydream.pr_comment_renderer import FALLBACK_NOTE
from daydream.pr_run_info import (
    LiveRunInfoSource,
    RunInfoStatus,
    render_live_run_info,
)
from daydream.pricing import ModelPrice
from daydream.trajectory import (
    DaydreamRunFlow,
    TrajectoryDocumentSnapshot,
    TrajectoryRecorder,
)
from daydream.workspace import WorkContext
from tests.conftest import _make_repo_with_main
from tests.harness.git_helpers import git


def _trajectory(
    session_id: str,
    trajectory_id: str,
    *,
    phase: str = "review",
    model: str = "gpt-5.5",
    prompt_tokens: int = 100,
    cost_usd: float | None = 0.01,
) -> Trajectory:
    return Trajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "agent": {
                "name": "daydream",
                "version": "test",
                "model_name": model,
            },
            "steps": [
                {
                    "step_id": 1,
                    "timestamp": "2026-09-11T00:00:00Z",
                    "source": "user",
                    "message": "review",
                    "extra": {"daydream_phase": phase},
                },
                {
                    "step_id": 2,
                    "timestamp": "2026-09-11T00:00:01Z",
                    "source": "agent",
                    "message": "done",
                    "model_name": model,
                    "metrics": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 10,
                        "cached_tokens": 20,
                        "cost_usd": cost_usd,
                    },
                    "extra": {"daydream_phase": phase},
                },
            ],
        }
    )


def _recorder(
    tmp_path: Path,
    session_id: str = "run-info",
    *,
    model: str = "gpt-5.5",
    prompt_tokens: int = 100,
    cost_usd: float | None = 0.01,
) -> TrajectoryRecorder:
    recorder = TrajectoryRecorder(
        path=tmp_path / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name=model,
        session_id=session_id,
    )
    recorder.steps.extend(
        _trajectory(
            session_id,
            session_id,
            model=model,
            prompt_tokens=prompt_tokens,
            cost_usd=cost_usd,
        ).steps
    )
    return recorder


def _work(repo: Path) -> WorkContext:
    sha = git(repo, "rev-parse", "HEAD")
    return WorkContext(
        repo=repo,
        source=repo,
        base_branch="main",
        base_sha=sha,
        head_branch="main",
        head_sha=sha,
        is_ephemeral=False,
        run_id="run-info",
    )


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [
        (LiveRunInfoSource(recorder=None, artifacts=None), "run info: recorder unavailable"),
        (
            LiveRunInfoSource(recorder=_recorder(Path(".")), artifacts=None),
            "run info: artifact session unavailable",
        ),
    ],
)
def test_missing_live_source_returns_bounded_unavailable_result(
    source: LiveRunInfoSource,
    diagnostic: str,
) -> None:
    result = render_live_run_info(source)

    assert result.status is RunInfoStatus.UNAVAILABLE
    assert FALLBACK_NOTE in result.markdown
    assert result.diagnostic == diagnostic


async def test_live_provider_renders_parent_and_real_retained_complete_sibling(
    tmp_path: Path,
) -> None:
    repo = _make_repo_with_main(tmp_path)
    session_id = "live-run-info"
    owner = resolve_private_workspace_owner(repo, locations=private_root_locations())
    async with open_artifact_session(
        _work(repo),
        session_id=session_id,
        owner=owner,
    ) as artifacts:
        route = artifacts.register_trajectory_output(None)
        assert route.full.write_path is not None
        recorder = _recorder(repo, session_id)
        recorder.path = route.full.write_path
        recorder.artifact_run_dir = route.run_dir
        child = _trajectory(session_id, "child", phase="fix")
        artifacts.write_trajectory_document(
            route,
            TrajectoryDocumentSnapshot(
                "child",
                route.run_dir / "trajectories" / "child.json",
                child.model_dump_json().encode(),
            ),
            "complete",
        )

        result = render_live_run_info(
            LiveRunInfoSource(recorder=recorder, artifacts=artifacts)
        )

    assert result.status is RunInfoStatus.RENDERED
    assert result.diagnostic is None
    assert "| Review |" in result.markdown
    assert "| Fix |" in result.markdown
    assert "- **Steps / tool calls:** 2 / 0" in result.markdown


class _SnapshotSource:
    def __init__(
        self,
        snapshots: tuple[TrajectoryDocumentSnapshot, ...] = (),
        *,
        error: Exception | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.error = error

    def snapshot_completed_sibling_trajectories(
        self,
        *,
        session_id: str,
    ) -> tuple[TrajectoryDocumentSnapshot, ...]:
        if self.error is not None:
            raise self.error
        return self.snapshots


def _source_with_snapshots(
    tmp_path: Path,
    snapshots: tuple[TrajectoryDocumentSnapshot, ...],
) -> LiveRunInfoSource:
    return LiveRunInfoSource(
        recorder=_recorder(tmp_path),
        artifacts=cast(ArtifactSession, _SnapshotSource(snapshots)),
    )


def test_snapshot_failure_returns_fixed_diagnostic_without_exception_text(
    tmp_path: Path,
) -> None:
    secret = "private/session/path/and-bad-json"
    source = LiveRunInfoSource(
        recorder=_recorder(tmp_path),
        artifacts=cast(ArtifactSession, _SnapshotSource(error=RuntimeError(secret))),
    )

    result = render_live_run_info(source)

    assert result.status is RunInfoStatus.UNAVAILABLE
    assert result.diagnostic == "run info: trajectory snapshot unavailable"
    assert secret not in result.markdown
    assert secret not in (result.diagnostic or "")


def test_non_root_recorder_is_rejected_before_acquisition(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder._trajectory_id = "child-recorder"
    result = render_live_run_info(
        LiveRunInfoSource(
            recorder=recorder,
            artifacts=cast(ArtifactSession, _SnapshotSource()),
        )
    )

    assert result.status is RunInfoStatus.UNAVAILABLE
    assert result.diagnostic == "run info: trajectory identity invalid"


def test_parent_build_failure_returns_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _recorder(tmp_path)

    def fail() -> Trajectory:
        raise RuntimeError("private parent contents")

    monkeypatch.setattr(recorder, "build_trajectory", fail)
    result = render_live_run_info(
        LiveRunInfoSource(
            recorder=recorder,
            artifacts=cast(ArtifactSession, _SnapshotSource()),
        )
    )

    assert result.status is RunInfoStatus.UNAVAILABLE
    assert result.diagnostic == "run info: parent trajectory unavailable"


@pytest.mark.parametrize(
    ("snapshot_id", "document_session", "document_id"),
    [
        ("child", "other-session", "child"),
        ("child", "run-info", "other-child"),
        ("run-info", "run-info", "run-info"),
    ],
)
def test_invalid_sibling_identity_returns_unavailable(
    tmp_path: Path,
    snapshot_id: str,
    document_session: str,
    document_id: str,
) -> None:
    document = _trajectory(document_session, document_id)
    snapshot = TrajectoryDocumentSnapshot(
        snapshot_id,
        tmp_path / "ignored.json",
        document.model_dump_json().encode(),
    )

    result = render_live_run_info(_source_with_snapshots(tmp_path, (snapshot,)))

    assert result.status is RunInfoStatus.UNAVAILABLE
    assert result.diagnostic == "run info: trajectory identity invalid"
    assert FALLBACK_NOTE in result.markdown


def test_malformed_sibling_and_duplicate_ids_return_unavailable(tmp_path: Path) -> None:
    malformed = TrajectoryDocumentSnapshot(
        "child",
        tmp_path / "ignored.json",
        b"{bad json",
    )
    malformed_result = render_live_run_info(
        _source_with_snapshots(tmp_path, (malformed,))
    )
    child = _trajectory("run-info", "child")
    duplicate = TrajectoryDocumentSnapshot(
        "child",
        tmp_path / "ignored.json",
        child.model_dump_json().encode(),
    )
    duplicate_result = render_live_run_info(
        _source_with_snapshots(tmp_path, (duplicate, duplicate))
    )

    assert malformed_result.status is RunInfoStatus.UNAVAILABLE
    assert malformed_result.diagnostic == "run info: trajectory document invalid"
    assert duplicate_result.status is RunInfoStatus.UNAVAILABLE
    assert duplicate_result.diagnostic == "run info: trajectory identity invalid"


def test_pricing_or_render_failure_returns_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import daydream.pr_run_info as provider

    def fail(
        _trajectories: Any,
        *,
        prices: dict[str, ModelPrice] | None = None,
    ) -> str:
        raise RuntimeError("secret renderer failure")

    monkeypatch.setattr(provider, "render_run_info", fail)
    source = LiveRunInfoSource(
        recorder=_recorder(tmp_path),
        artifacts=cast(ArtifactSession, _SnapshotSource()),
    )

    result = render_live_run_info(source)

    assert result.status is RunInfoStatus.UNAVAILABLE
    assert result.diagnostic == "run info: rendering unavailable"
    assert FALLBACK_NOTE in result.markdown


def test_price_lookup_failure_returns_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import daydream.pr_run_info as provider

    def fail() -> dict[str, Any]:
        raise RuntimeError("private prices path")

    monkeypatch.setattr(provider, "load_user_prices", fail)
    source = LiveRunInfoSource(
        recorder=_recorder(tmp_path),
        artifacts=cast(ArtifactSession, _SnapshotSource()),
    )

    result = render_live_run_info(source)

    assert result.status is RunInfoStatus.UNAVAILABLE
    assert result.diagnostic == "run info: rendering unavailable"
    assert FALLBACK_NOTE in result.markdown


def test_live_provider_honors_user_price_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_model = "custom-live-model"
    prices_file = tmp_path / "prices.toml"
    prices_file.write_text(
        f'[prices."{custom_model}"]\n'
        "input = 2.0\n"
        "cached_input = 0.5\n"
        "output = 8.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(prices_file))
    source = LiveRunInfoSource(
        recorder=_recorder(
            tmp_path,
            model=custom_model,
            prompt_tokens=1_000_000,
            cost_usd=None,
        ),
        artifacts=cast(ArtifactSession, _SnapshotSource()),
    )

    result = render_live_run_info(source)

    assert result.status is RunInfoStatus.RENDERED
    assert "- **Cost:** $2.00" in result.markdown
