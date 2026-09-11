"""Real-runner regressions for explicit artifact ownership and durable handoffs."""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream import runner
from daydream.agent import run_agent
from daydream.artifact_visibility import (
    ArtifactVisibilityError,
    open_artifact_session,
    private_root_locations,
    resolve_private_workspace_owner,
)
from daydream.backends import ResultEvent, TextEvent
from daydream.deep.orchestrator import run_deep
from daydream.runner import RunConfig
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import bare_remote, git


async def test_recorder_public_output_requires_standalone_opt_in(
    tmp_path: Path, make_config: Callable[..., RunConfig],
) -> None:
    config = make_config(tmp_path)
    with pytest.raises(ArtifactVisibilityError, match="allow_standalone=True"):
        runner._open_recorder(
            config=config, target_dir=tmp_path, work=None, flow_kind=DaydreamRunFlow.CUSTOM,
        )
    assert not (tmp_path / ".daydream").exists()

    recorder = runner._open_recorder(
        config=config, target_dir=tmp_path, work=None, flow_kind=DaydreamRunFlow.CUSTOM,
        allow_standalone=True,
    )
    async with recorder:
        await run_agent(
            ScriptedBackend(
                events=[TextEvent(text="Standalone result"), ResultEvent(structured_output=None, continuation=None)],
            ),
            tmp_path,
            "Record one standalone turn.",
            phase=DaydreamPhase.REVIEW,
        )
    assert recorder.path.is_file()
    assert json.loads(recorder.path.read_text())["session_id"] == recorder.session_id


@pytest.mark.parametrize("entrypoint", ["deep", "recorder"])
async def test_standalone_entry_rejects_borrowing_a_bound_artifact_session(
    tiny_diff_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., RunConfig],
    make_work: Callable[..., WorkContext],
    entrypoint: str,
) -> None:
    repo = tiny_diff_target
    work = make_work(
        repo, base_sha=git(repo, "rev-parse", "main"),
        head_sha=git(repo, "rev-parse", "HEAD"), head_branch="feature",
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

    async with open_artifact_session(work, session_id=f"bound-{entrypoint}", owner=owner) as session:
        live = session.daydream_dir
        before = sorted(path.relative_to(live) for path in live.rglob("*"))
        with pytest.raises(ArtifactVisibilityError, match="standalone.*active artifact session"):
            if entrypoint == "deep":
                await run_deep(config, work, allow_standalone=True)
            else:
                runner._open_recorder(
                    config=config, target_dir=repo, work=work,
                    flow_kind=DaydreamRunFlow.CUSTOM, allow_standalone=True,
                )
        assert not (repo / ".daydream").exists()
        assert (live / "deep" / "prior.txt").read_bytes() == b"prior session evidence\n"
        assert sorted(path.relative_to(live) for path in live.rglob("*")) == before
        assert backend.call_count == 0

    assert prior.read_bytes() == b"prior session evidence\n"


@pytest.mark.parametrize(
    "relative_destination", ["operator evidence/trajectory.json", ".daydream/custom trajectory.json"],
)
async def test_ephemeral_failure_handoff_retains_explicit_trajectory_without_archive(
    tiny_diff_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    make_config: Callable[..., RunConfig],
    relative_destination: str,
) -> None:
    repo = tiny_diff_target
    origin = bare_remote(tmp_path / "origin.git")
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-u", "origin", "main")
    git(repo, "push", "-u", "origin", "feature")
    explicit = repo / relative_destination
    private_base = (tmp_path / "private artifact roots").resolve()
    head_before = git(repo, "rev-parse", "HEAD")
    ext_dir.write_module(
        "from daydream import git_ops\n"
        "from daydream.extensions import FlowStep, Stop\n"
        "from daydream.fix_footprint import AuthorizedFixFootprint\n"
        "from daydream.phases import phase_test_and_heal\n"
        "async def _handoff(ctx):\n"
        "    assert ctx.artifacts is not None\n"
        "    def capture_tree():\n"
        "        return git_ops.tree_key(git_ops.snapshot_worktree_delta(\n"
        "            ctx.work.repo, 'HEAD', preexisting_untracked={},\n"
        "        ))\n"
        "    result = await phase_test_and_heal(\n"
        "        ctx.backend_for('test'), ctx.work, config=ctx.config,\n"
        "        session_id=ctx.artifacts.layout.session_id,\n"
        "        capture_tree_key=capture_tree,\n"
        "        footprint=AuthorizedFixFootprint.build(ctx.work.repo, set(), []),\n"
        "        run_context=ctx.run_context,\n"
        "        artifact_session=ctx.artifacts, allow_standalone=False,\n"
        "    )\n"
        "    return Stop(0 if result.passed else 1)\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='artifact-handoff', run=_handoff))\n"
        "    registry.set_flow('artifact-handoff', ['artifact-handoff'])\n"
    )
    # An empty, valid response exercises the deterministic host handoff.
    backend = ScriptedBackend(
        events=[ResultEvent(structured_output={"handoff_prompt": ""}, continuation=None)],
    )
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)
    config = make_config(
        repo,
        flow_name="artifact-handoff",
        force_worktree=True,
        trajectory_path=explicit,
        archive=False,
        run_eval=False,
        test_command=shlex.join(
            [sys.executable, "-c", "import sys; print('1 failed, 0 passed'); sys.exit(1)"]
        ),
    )

    result = await runner.run(config, private_roots=private_root_locations(base=private_base))

    assert result == 1
    assert explicit.is_file()
    trajectory = json.loads(explicit.read_text())
    handoffs = list(repo.glob(".daydream/runs/*/handoff.md"))
    assert len(handoffs) == 1
    assert handoffs[0].parent.name == trajectory["session_id"]
    body = handoffs[0].read_text()
    assert str(explicit) in body
    assert str(repo / ".daydream" / "diff.patch") in body
    assert str(private_base) not in body
    assert "1 failed, 0 passed" in body
    assert backend.call_count == 1
    assert backend.read_only_calls == [True]
    model_cwd = backend.calls[0]["cwd"]
    assert model_cwd != repo
    assert str(model_cwd) not in body
    assert not model_cwd.exists()
    assert git(repo, "rev-parse", "HEAD") == head_before

    # A later real run must accept the prior session's published destinations
    # and preserve their bytes when writing its own default trajectory.
    published = explicit.read_bytes()
    second_config = replace(
        config,
        trajectory_path=None,
        test_command=shlex.join([sys.executable, "-c", "print('1 passed')"]),
    )
    assert await runner.run(
        second_config, private_roots=private_root_locations(base=private_base),
    ) == 0
    assert explicit.read_bytes() == published
    assert handoffs[0].read_text() == body
    assert backend.call_count == 1
    assert git(repo, "rev-parse", "HEAD") == head_before
