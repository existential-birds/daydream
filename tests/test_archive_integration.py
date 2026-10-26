"""Recorder callback, strict archive round-trip, and CLI archive/evaluation integration."""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest

from daydream.agent import run_agent
from daydream.archive import finalize_archive_run
from daydream.archive.manifest import archive_recorder_provenance_from_snapshot
from daydream.artifact_visibility import (
    ArtifactEvidenceProvenance,
    ArtifactTreeSnapshot,
)
from daydream.artifacts.filesystem import manifest_tree
from daydream.atif import Step, validate as atif_validate
from daydream.backends import AgentEvent, ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.run_config import RunConfig
from daydream.run_snapshot import (
    ArchiveRunSnapshot,
    ManifestRunIdentity,
    RunPhaseCapabilities,
)
from daydream.runner import run
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    LifecycleReasonCode,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    TrajectoryRecorder,
    now_iso,
)
from tests.harness.backend import ScriptedBackend
from tests.harness.review_profile import independent_exploration_profile
from tests.harness.stub_backend import StubBackend
from tests.harness.trajectory import make_recorder


class _SecretFailureBackend(StubBackend):
    """External backend failure containing synthetic, intentionally private data."""

    secrets = ("sk-abcdefghijklmnopqrstuvwxyz1234567890", "https://private-user:private-password@example.invalid/model",
        "/Users/private-lifecycle-user/work/client-private.py",
    )

    def __init__(self, target: Path) -> None:
        super().__init__(target)
        self.failures = 0

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if "you are the **dependency-tracer** specialist" in prompt.lower():
            self.failures += 1
            raise RuntimeError("backend rejected " + " ".join(self.secrets))
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


async def test_runner_lifecycle_reason_redaction_reaches_evaluation_and_archive(
    shard_many_python_target: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real caught backend failure emits closed codes, not its private message."""

    target = shard_many_python_target
    backend = _SecretFailureBackend(target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    assert await run(RunConfig(
        target=str(target), cleanup=False, non_interactive=True, review_profile=independent_exploration_profile(),
        test_command=shlex.join([sys.executable, "-c", "pass"]),
    )) == 0
    assert backend.failures == 1
    roots = list((target / ".daydream/runs").glob("*/trajectory.json"))
    assert len(roots) == 1
    root = json.loads(roots[0].read_bytes())
    dispatches = [step for step in root["steps"] if "dispatch_id" in step.get("extra", {})]
    partials = [step for step in dispatches if step["extra"]["dispatch_status"] == "partial"]
    assert len(partials) == 1
    assert partials[0]["extra"]["reason_code"] == "some_children_failed"
    refs = [ref for step in dispatches for result in step["observation"]["results"]
            for ref in result["subagent_trajectory_ref"]]
    assert refs
    live_paths = [roots[0], *(target / ".daydream" / ref["trajectory_path"] for ref in refs)]
    archive = archive_dir / "runs" / root["session_id"]
    manifest = archive / "manifest.json"
    evaluation = archive / "evaluation.json"
    assert manifest.is_file() and evaluation.is_file()
    archived_paths = list(archive.rglob("*.json"))
    assert len(archived_paths) >= len(live_paths) + 2
    reason_codes: list[str] = []

    def collect_reasons(value: Any) -> None:
        if isinstance(value, dict):
            if "reason_code" in value and value["reason_code"] is not None:
                reason_codes.append(value["reason_code"])
            for child in value.values():
                collect_reasons(child)
        elif isinstance(value, list):
            for child in value:
                collect_reasons(child)

    for path in [*live_paths, *archived_paths]:
        text = path.read_text(encoding="utf-8")
        for secret in backend.secrets:
            assert secret not in text, f"private exception text leaked to {path.name}"
        collect_reasons(json.loads(text))
    assert "some_children_failed" in reason_codes
    assert set(reason_codes) <= {reason.value for reason in LifecycleReasonCode}


def _add_user_step(recorder: TrajectoryRecorder) -> None:
    """Append a minimal user Step so the recorder has at least one step and won't skip _write."""
    step = Step(step_id=recorder._next_step_id(), timestamp=now_iso(), source="user", message="test prompt",
        extra={"daydream_phase": DaydreamPhase.REVIEW.value, "daydream_run_flow": DaydreamRunFlow.NORMAL.value,},
    )
    recorder.steps.append(step)


async def _hold_archive_fork(parent: TrajectoryRecorder, name: str, entered: anyio.Event, release: anyio.Event,
) -> None:
    async with parent.fork(name) as child:
        for call in ("first", "second"):
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                invocation.observe_user_step(prompt=f"{name}-{call}")
        async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            invocation.observe_user_step(prompt=f"{name}-blocked")
            entered.set()
            await release.wait()


def _finalize_strict_archive(recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot, config: Any, target: Path, *,
    destinations: tuple[Any, ...] = (), dump_path: Path | None = None,
) -> None:
    """Run strict finalization; the re-attested target tree must exclude the archive directory."""
    session_id = recorder.session_id
    finalize_archive_run(run=ArchiveRunSnapshot(recorder_provenance=archive_recorder_provenance_from_snapshot(
                write_snapshot=snapshot, run_flow=recorder.run_flow,
            ),
            identity=ManifestRunIdentity(skill="python", model=None, backend="claude", review_backend=None,
                fix_backend="claude" if recorder.run_flow is DaydreamRunFlow.NORMAL else None,
                test_backend="claude" if recorder.run_flow is DaydreamRunFlow.NORMAL else None,
                per_stack_review_backend=None, per_stack_review_model=None, review_only=False,
                deep=recorder.run_flow is DaydreamRunFlow.NORMAL, profile=None,
                phases=RunPhaseCapabilities(per_stack_review=False, merge=recorder.run_flow is DaydreamRunFlow.NORMAL,
                    fix=recorder.run_flow is DaydreamRunFlow.NORMAL, test=recorder.run_flow is DaydreamRunFlow.NORMAL,
                    push=False, remote_ci=False,
                ),
            ), trajectories=snapshot,
        ),
        artifacts=ArtifactTreeSnapshot(
            session_id=session_id, workspace_key="workspace", root=target, manifest=manifest_tree(target),
            destinations=destinations,
        ),
        artifact_provenance=ArtifactEvidenceProvenance(
            workspace_key="workspace", session_id=session_id, public_source=target,
            # Live-root-relative routes retain their paths inside the frozen copy.
            live_root=target,
        ), config=config, work=None, dump_path=dump_path,
    )


def _strict_archive_callback(
    config: Any, target: Path, *, destinations: tuple[Any, ...] = (),
    dump_path: Path | None = None,
) -> Any:
    """Finalize only the first recorder snapshot, emulating the runner's one-time boundary."""
    finalized: list[str] = []

    def _finalize(recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot) -> None:
        if finalized:
            return
        finalized.append(snapshot.status)
        _finalize_strict_archive(
            recorder, snapshot, config, target, destinations=destinations, dump_path=dump_path,
        )

    return _finalize


@pytest.mark.parametrize(("status", "lifecycle", "expected_wall"), [
    ("complete", True, 10.0), ("partial", True, 4.0), ("complete", False, None),
])
def test_finalization_evaluates_selected_snapshot_after_files_change(tmp_path: Path, archive_dir: Path,
    monkeypatch: pytest.MonkeyPatch, status: Literal["complete", "partial"], lifecycle: bool,
    expected_wall: float | None,
) -> None:
    from daydream.eval.analyzer import analyze_session

    target = tmp_path / "frozen"
    target.mkdir()
    session = "selected-snapshot"
    path = target / ".daydream" / "runs" / session / "trajectory.json"
    path.parent.mkdir(parents=True)
    extra = {"run_started_at": "2026-01-01T00:00:00Z", "run_ended_at": "2026-01-01T00:00:10Z",
        "snapshot_at": "2026-01-01T00:00:04Z", "partial": False,
    } if lifecycle else {}
    root: dict[str, Any] = {"session_id": session, "trajectory_id": session,
        "steps": [{"timestamp": "2026-01-01T00:00:02Z"}, {"timestamp": "2026-01-01T00:00:03Z"}],
        "extra": extra, "final_metrics": {"total_cost_usd": 0.5},
    }
    cutoff = "2026-01-01T00:00:04Z" if status == "partial" else "2026-01-01T00:00:10Z"
    # A delayed sibling and step timestamps outside the lifecycle must not widen root wall time.
    child_path = path.parent / "trajectories" / "delayed.json"
    child_path.parent.mkdir()
    child = {
        "session_id": session,
        "trajectory_id": "delayed",
        "extra": {},
        "steps": [{"timestamp": "2026-01-01T00:01:00Z"}, {"timestamp": "2026-01-01T00:01:02Z"}],
    }
    child_path.write_text(json.dumps(child))
    root["steps"] = [{"timestamp": "2026-01-01T00:00:40Z"}, {"timestamp": "2026-01-01T00:00:41Z"}]
    snapshot = RunWriteSnapshot(
        status=status,
        cutoff_at=cutoff,
        root_trajectory_id=session,
        documents=(
            TrajectoryDocumentSnapshot(session, path, json.dumps(root).encode()),
            TrajectoryDocumentSnapshot("delayed", child_path, json.dumps(child).encode()),
        ),
    )
    snapshot.validate(session)
    path.write_bytes(snapshot.documents[0].json_bytes)
    root["final_metrics"]["total_cost_usd"] = 99.0
    root["extra"]["run_ended_at"] = "2026-01-01T00:01:00Z"
    path.write_text(json.dumps(root))
    evaluated: list[RunWriteSnapshot] = []

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        evaluated.append(kwargs["write_snapshot"])
        return analyze_session(*args, **kwargs)

    monkeypatch.setattr("daydream.eval.analyzer.analyze_session", evaluate)
    recorder = make_recorder(target, session_id=session, path=path)

    _finalize_strict_archive(recorder, snapshot, RunConfig(run_eval=True), target)

    assert evaluated == [snapshot]
    assert evaluated[0] is snapshot
    archived = archive_dir / "runs" / session
    evaluation = json.loads((archived / "evaluation.json").read_bytes())
    manifest = json.loads((archived / "manifest.json").read_bytes())
    assert (archived / "trajectory.json").read_bytes() == snapshot.documents[0].json_bytes
    assert evaluation["cost"]["total_cost_usd"] == 0.5
    assert evaluation["timing"]["total_wall_clock_seconds"] == expected_wall
    assert evaluation["timing"]["by_agent"] == [
        {"agent": "delayed", "duration_seconds": 2.0},
        {"agent": "main", "duration_seconds": 1.0},
    ]
    assert (archived / "trajectories" / "delayed.json").read_bytes() == snapshot.documents[1].json_bytes
    assert manifest["metrics"]["wall_clock_seconds"] == expected_wall


async def test_on_write_failure_does_not_raise(tmp_path: Path) -> None:
    def on_write_boom(recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot,) -> None:
        raise RuntimeError("archive exploded")
    recorder = make_recorder(tmp_path, on_write=on_write_boom)
    async with recorder:
        _add_user_step(recorder)
    traj_path = tmp_path / ".daydream" / "trajectory.json"
    assert traj_path.exists()
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    assert data["session_id"] == recorder.session_id
    assert len(data["steps"]) == 1


async def test_archive_callback_keeps_configured_hub_work_local(
    tmp_path: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: hub)
    target = tmp_path / "project"
    target.mkdir()
    (target / ".review-output.md").write_text("# Review\nLooks good.\n", encoding="utf-8")
    config = RunConfig(run_eval=False, trajectory_hub_repo="acme/dd-trajectories", archive=True)
    recorder = make_recorder(target, on_write=_strict_archive_callback(config, target))
    _add_user_step(recorder)
    async with recorder:
        pass

    assert hub.commits == []
    assert (archive_dir / "runs" / recorder.session_id / "manifest.json").is_file()


async def test_signal_flush_archive_uses_one_immutable_cutoff_for_all_documents(
    tmp_path: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate partial/complete archive roots preserve single-finalization while testing both fork states."""

    target = tmp_path / "frozen"
    target.mkdir()
    config = RunConfig(target=str(target), archive=True, run_eval=True)
    snapshots: list[RunWriteSnapshot] = []
    roots = {status: tmp_path / f"archive-{status}" for status in ("partial", "complete")}

    def callback(recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot) -> None:
        snapshots.append(snapshot)
        monkeypatch.setenv("DAYDREAM_ARCHIVE_DIR", str(roots[snapshot.status]))
        _finalize_strict_archive(recorder, snapshot, config, target)

    recorder = make_recorder(target, on_write=callback)
    entered = {name: anyio.Event() for name in ("a", "b")}
    release = {name: anyio.Event() for name in entered}
    async with recorder:
        _add_user_step(recorder)
        async with anyio.create_task_group() as task_group:
            for name in entered:
                task_group.start_soon(_hold_archive_fork, recorder, name, entered[name], release[name],)
                await entered[name].wait()
            recorder.write_partial()
            partial = snapshots[-1]
            assert partial.status == "partial"
            assert len(partial.documents) == 3
            assert {json.loads(document.json_bytes)["extra"]["snapshot_at"] for document in partial.documents} == {
                partial.cutoff_at
            }
            frozen = tuple(document.json_bytes for document in partial.documents)
            run_dir = roots["partial"] / "runs" / recorder.session_id
            manifest = json.loads((run_dir / "manifest.json").read_text())
            evaluation = json.loads((run_dir / "evaluation.json").read_text())
            assert manifest["metrics"]["wall_clock_seconds"] == evaluation["timing"]["total_wall_clock_seconds"]
            assert evaluation["timing"]["agent_completeness"] == {"total": 6, "attributed": 0, "unattributed": 6,}
            assert evaluation["timing"]["diagnostics"]["malformed_invocation"] == 2
            assert manifest["metrics"]["timing_coverage"]["agent_completeness"] == evaluation["timing"][
                "agent_completeness"
            ]
            assert len(list((run_dir / "trajectories").glob("*.json"))) == 2
            for event in release.values():
                event.set()

        assert tuple(document.json_bytes for document in partial.documents) == frozen

    complete = snapshots[-1]
    assert complete.status == "complete"
    assert complete.cutoff_at > partial.cutoff_at
    assert all("run_ended_at" in json.loads(document.json_bytes)["extra"] for document in complete.documents)
    for document in complete.documents:
        payload = json.loads(document.json_bytes)
        assert payload["session_id"] == recorder.session_id
        assert atif_validate(payload, validate_images=False)
        assert [step["step_id"] for step in payload["steps"]] == list(range(1, len(payload["steps"]) + 1))
    assert {document.trajectory_id for document in complete.documents} == {
        recorder.session_id,
        f"{recorder.session_id}:a",
        f"{recorder.session_id}:b",
    }
    final_evaluation = json.loads((roots["complete"] / "runs" / recorder.session_id / "evaluation.json").read_text())
    assert final_evaluation["timing"]["agent_completeness"] == {"total": 6, "attributed": 0, "unattributed": 6,}
    assert final_evaluation["timing"]["diagnostics"]["malformed_invocation"] == 0


async def test_archive_callback_no_archive_dump_artifacts_skips_upload(
    tmp_path: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: hub)
    target = tmp_path / "project"
    target.mkdir()
    (target / ".review-output.md").write_text("# Review\nLooks good.\n", encoding="utf-8")
    config = RunConfig(run_eval=False, trajectory_hub_repo="acme/dd-trajectories", archive=False,
                       dump_artifacts=str(dump_dir))
    recorder = make_recorder(target, on_write=_strict_archive_callback(config, target, dump_path=dump_dir))
    _add_user_step(recorder)
    async with recorder:
        pass

    assert hub.commits == []
    assert (dump_dir / "manifest.json").is_file()
    assert (dump_dir / "review-output.md").is_file()


async def test_runner_archive_round_trip_redacts_structured_tool_credentials(tmp_path: Path, archive_dir: Path,
) -> None:
    """Live and archived structured tool credentials must redact identically and remain valid ATIF."""

    sentinel = "opaque-test-only-sentinel"
    target_dir = tmp_path / "project"
    target_dir.mkdir()

    backend = ScriptedBackend(events=(TextEvent(text="hello"), ToolStartEvent(id="t1", name="ListDir",
                input={"dir": "/tmp", "apiKey": {"nested": sentinel}, "displayName": "visible",},
            ),
            ToolResultEvent(id="t1", output='{"status": "ok", "token": "opaque-test-only-sentinel"}', is_error=False,),
            ResultEvent(structured_output=None, continuation=None),
        )
    )

    config = RunConfig(target=str(target_dir), archive=True, run_eval=False)
    recorder = make_recorder(
        target_dir, run_flow=DaydreamRunFlow.NORMAL, on_write=_strict_archive_callback(config, target_dir),
    )

    async with recorder:
        await run_agent(backend, target_dir, "inspect configuration", phase=DaydreamPhase.REVIEW)

    live = json.loads(recorder.path.read_text(encoding="utf-8"))
    run_dir = archive_dir / "runs" / recorder.session_id
    archived = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))

    assert live == archived
    assert atif_validate(live) is True
    assert any(step["source"] == "agent" and step.get("message") == "hello" for step in live["steps"])
    blob = json.dumps(live)
    assert sentinel not in blob
    assert "[REDACTED_CREDENTIAL]" in blob
    assert "visible" in blob
