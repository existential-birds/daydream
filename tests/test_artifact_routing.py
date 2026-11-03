"""Real-runner regressions for explicit artifact ownership and durable handoffs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from daydream import run_artifacts, runner
from daydream.agent import run_agent
from daydream.artifact_visibility import (
    ArtifactVisibilityError,
    open_artifact_session,
    private_root_locations,
    resolve_private_workspace_owner,
)
from daydream.backends import ResultEvent, TextEvent
from daydream.deep.orchestrator import run_deep
from daydream.run_config import RunConfig
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import git


async def test_recorder_public_output_requires_standalone_opt_in(tmp_path: Path, make_config: Callable[..., RunConfig],
) -> None:
    config = make_config(tmp_path)
    with pytest.raises(ArtifactVisibilityError, match="allow_standalone=True"):
        run_artifacts._open_recorder(config=config, target_dir=tmp_path, work=None, flow_kind=DaydreamRunFlow.CUSTOM,)
    assert not (tmp_path / ".daydream").exists()

    recorder = run_artifacts._open_recorder(
        config=config, target_dir=tmp_path, work=None, flow_kind=DaydreamRunFlow.CUSTOM, allow_standalone=True,
    )
    async with recorder:
        await run_agent(ScriptedBackend(
                events=[TextEvent(text="Standalone result"), ResultEvent(structured_output=None, continuation=None)],
            ), tmp_path, "Record one standalone turn.", phase=DaydreamPhase.REVIEW,
        )
    assert recorder.path.is_file()
    assert json.loads(recorder.path.read_text())["session_id"] == recorder.session_id

async def test_standalone_entry_rejects_borrowing_a_bound_artifact_session(
    tiny_diff_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., RunConfig],
    make_work: Callable[..., WorkContext],
) -> None:
    repo = tiny_diff_target
    work = make_work(
        repo, base_sha=git(repo, "rev-parse", "main"), head_sha=git(repo, "rev-parse", "HEAD"), head_branch="feature",
    )
    prior = repo / ".daydream" / "deep" / "prior.txt"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"prior session evidence\n")
    owner = resolve_private_workspace_owner(
        repo, locations=private_root_locations(base=(tmp_path / "private").resolve()),
    )
    backend = ScriptedBackend(events=[ResultEvent(structured_output=None, continuation=None)])
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)
    config = make_config(repo, start_at="fix")

    async with open_artifact_session(work, session_id="bound-session", owner=owner) as session:
        live = session.daydream_dir
        before = sorted(path.relative_to(live) for path in live.rglob("*"))
        with pytest.raises(ArtifactVisibilityError, match="standalone.*active artifact session"):
            await run_deep(config, work, allow_standalone=True)
        with pytest.raises(ArtifactVisibilityError, match="standalone.*active artifact session"):
            run_artifacts._open_recorder(
                config=config, target_dir=repo, work=work, flow_kind=DaydreamRunFlow.CUSTOM, allow_standalone=True,
            )
        assert not (repo / ".daydream").exists()
        assert (live / "deep" / "prior.txt").read_bytes() == b"prior session evidence\n"
        assert sorted(path.relative_to(live) for path in live.rglob("*")) == before
        assert backend.call_count == 0

    assert prior.read_bytes() == b"prior session evidence\n"
