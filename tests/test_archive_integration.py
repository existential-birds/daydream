"""Integration tests for the TrajectoryRecorder on_write callback and archive pipeline.

Verifies that the on_write callback fires at the right times, that the full
archive round-trip produces valid bundles, and that CLI flags for --no-archive
and --no-eval are parsed correctly into RunConfig.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.atif import Step
from daydream.backends import AgentEvent
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    RunWriteSnapshot,
    TrajectoryRecorder,
    now_iso,
)
from tests.harness.config import TARGET_HUB_KEY_CONFIG
from tests.harness.stub_backend import StubBackend
from tests.harness.trajectory import make_recorder


class _SecretFailureBackend(StubBackend):
    """External backend failure containing synthetic, intentionally private data."""

    secrets = (
        "sk-abcdefghijklmnopqrstuvwxyz1234567890",
        "https://private-user:private-password@example.invalid/model",
        "/Users/private-lifecycle-user/work/client-private.py",
    )

    def __init__(self, target: Path) -> None:
        super().__init__(target)
        self.failures = 0

    async def execute(
        self, cwd: Path, prompt: str, output_schema: Any = None,
        continuation: Any = None, agents: Any = None,
        max_turns: Any = None, read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        if "you are the **dependency-tracer** specialist" in prompt.lower():
            self.failures += 1
            raise RuntimeError("backend rejected " + " ".join(self.secrets))
        async for event in super().execute(cwd, prompt, output_schema, continuation, agents, max_turns, read_only):
            yield event


async def test_runner_lifecycle_reason_redaction_reaches_evaluation_and_archive(
    shard_many_python_target: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real caught backend failure emits closed codes, not its private message."""
    from daydream.runner import RunConfig, run
    from daydream.trajectory import LifecycleReasonCode

    target = shard_many_python_target
    backend = _SecretFailureBackend(target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    assert await run(RunConfig(target=str(target), cleanup=False, non_interactive=True)) == 0
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
    step = Step(
        step_id=recorder._next_step_id(),
        timestamp=now_iso(),
        source="user",
        message="test prompt",
        extra={
            "daydream_phase": DaydreamPhase.REVIEW.value,
            "daydream_run_flow": DaydreamRunFlow.NORMAL.value,
        },
    )
    recorder.steps.append(step)


async def _hold_archive_fork(
    parent: TrajectoryRecorder,
    name: str,
    entered: anyio.Event,
    release: anyio.Event,
) -> None:
    async with parent.fork(name) as child:
        for call in ("first", "second"):
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                invocation.observe_user_step(prompt=f"{name}-{call}")
        async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            invocation.observe_user_step(prompt=f"{name}-blocked")
            entered.set()
            await release.wait()


# --- shared round-trip setup for the strict archive finalization tests ---
def _finalize_strict_archive(
    recorder: TrajectoryRecorder,
    snapshot: RunWriteSnapshot,
    config: Any,
    target: Path,
    *,
    destinations: tuple[Any, ...] = (),
    upload: bool = True,
    dump_path: Path | None = None,
) -> None:
    """Drive the production strict archive finalizer over one frozen tree.

    The tree handed over is attested after every stage, so ``target`` must not
    contain the archive directory itself.
    """
    from daydream.archive import finalize_archive_run
    from daydream.archive.manifest import archive_recorder_provenance_from_snapshot
    from daydream.artifact_visibility import (
        ArtifactEvidenceProvenance,
        ArtifactTreeSnapshot,
        _manifest,
    )

    session_id = recorder.session_id
    finalize_archive_run(
        recorder_provenance=archive_recorder_provenance_from_snapshot(
            write_snapshot=snapshot, run_flow=recorder.run_flow,
        ),
        artifacts=ArtifactTreeSnapshot(
            session_id=session_id,
            workspace_key="workspace",
            root=target,
            manifest=_manifest(target),
            destinations=destinations,
        ),
        artifact_provenance=ArtifactEvidenceProvenance(
            workspace_key="workspace",
            session_id=session_id,
            public_source=target,
            # The frozen root is a copy of the live root, so route paths
            # relative to one resolve unchanged inside the other.
            live_root=target,
        ),
        config=config,
        write_snapshot=snapshot,
        work=None,
        upload=upload,
        dump_path=dump_path,
    )


def _strict_archive_callback(
    config: Any,
    target: Path,
    *,
    destinations: tuple[Any, ...] = (),
    unsuccessful: bool = False,
    dump_path: Path | None = None,
) -> Any:
    """An on_write hook that stands in for the runner's finalization boundary.

    The runner retains snapshots during the run and calls
    ``finalize_archive_run`` exactly once with ``upload=successful``; this hook
    finalizes the first snapshot it sees so an archive can be assembled from a
    single recorder without opening a real workspace.
    """
    finalized: list[str] = []

    def _finalize(recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot) -> None:
        if finalized:
            return
        finalized.append(snapshot.status)
        _finalize_strict_archive(
            recorder,
            snapshot,
            config,
            target,
            destinations=destinations,
            upload=not unsuccessful,
            dump_path=dump_path,
        )

    return _finalize


def _findings_route(live_root: Path) -> Any:
    """The registered ``--findings-out`` route the strict bundle relocates."""
    from daydream.artifact_visibility import (
        DestinationDelivery,
        OutputLabel,
        RoutedDestination,
    )

    private = live_root / ".explicit" / "0000" / "findings.json"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text(
        '{"schema_version": 1, "findings": [{"fingerprint": "deadbeef"}]}'
    )
    return RoutedDestination(
        label=OutputLabel.FINDINGS_OUTPUT,
        requested=Path("findings/findings.json"),
        write_path=private,
        frozen_path=private,
        delivery=DestinationDelivery.DEFERRED,
    )


def _make_round_trip_fixture(
    tmp_path: Path,
    run_flow: DaydreamRunFlow,
    *,
    run_eval: bool = False,
) -> TrajectoryRecorder:
    """Build the full archive round-trip fixture: a frozen tree with minimal
    .daydream/ scaffolding, a registered findings route, a RunConfig, the strict
    archive hook, and a recorder primed with lifecycle stamps 8.5s apart so the
    derived wall-clock span is deterministic.

    ``run_eval`` turns on the real deterministic eval pass (production default),
    so the archived bundle carries a genuine ``evaluation.json`` whose metrics
    the manifest projection reads."""
    from daydream.runner import RunConfig

    target = tmp_path / "frozen"
    target.mkdir()
    (target / ".review-output.md").write_text("# Review\nLooks good.\n")
    findings = _findings_route(target)

    config = RunConfig(
        target=str(target),
        stack="python",
        backend="claude",
        archive=True,
        run_eval=run_eval,
        findings_out=str(findings.write_path),
    )

    recorder = make_recorder(
        target,
        run_flow=run_flow,
        on_write=_strict_archive_callback(config, target, destinations=(findings,)),
    )
    # Two steps spaced 8.5s apart so the derived span is deterministic.
    for ts in ("2026-05-31T10:00:00.000000Z", "2026-05-31T10:00:08.500000Z"):
        recorder.steps.append(
            Step(
                step_id=recorder._next_step_id(),
                timestamp=ts,
                source="agent",
                message="step",
                extra={
                    "daydream_phase": DaydreamPhase.REVIEW.value,
                    "daydream_run_flow": run_flow.value,
                },
            )
        )
    return recorder


def _assert_round_trip_bundle(
    archive_dir: Path,
    recorder: TrajectoryRecorder,
    *,
    fix_backend: str | None,
    test_backend: str | None,
) -> None:
    """Shared manifest + SQLite assertions for the archive round-trip.

    fix_backend/test_backend parameterize the divergent branch: flows that never
    run fix/test (IMPROVE) omit the keys and leave the SQL columns NULL (None),
    while deep-family flows (NORMAL) record the resolved backend."""
    run_dir = archive_dir / "runs" / recorder.session_id
    assert run_dir.is_dir()

    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["session_id"] == recorder.session_id
    assert manifest["status"] == "complete"
    assert manifest["run"]["flow"] == recorder.run_flow.value
    assert manifest["run"]["skill"] == "python"
    assert manifest["metrics"]["wall_clock_seconds"] == 8.5
    if fix_backend is None:
        assert "fix_backend" not in manifest["run"]
        assert "test_backend" not in manifest["run"]
    else:
        assert manifest["run"]["fix_backend"] == fix_backend
        assert manifest["run"]["test_backend"] == test_backend

    archived = run_dir / "findings.json"
    assert archived.is_file()
    assert (
        json.loads(archived.read_text(encoding="utf-8"))["findings"][0]["fingerprint"]
        == "deadbeef"
    )

    db_path = archive_dir / "index.db"
    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM runs WHERE session_id = ?",
            (recorder.session_id,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "complete"
        assert row["run_flow"] == recorder.run_flow.value
        assert row["fix_backend"] == fix_backend
        assert row["test_backend"] == test_backend
    finally:
        conn.close()


# on_write does NOT fire on empty trajectory
async def test_on_write_does_not_fire_on_empty_trajectory(tmp_path: Path) -> None:
    """Empty trajectories skip _write entirely, so on_write must not be called."""
    callback_calls: list[tuple[str, str]] = []

    def on_write(recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot) -> None:
        callback_calls.append((recorder.session_id, snapshot.status))

    recorder = make_recorder(tmp_path, on_write=on_write)
    async with recorder:
        pass

    assert len(callback_calls) == 0
    assert not (tmp_path / ".daydream" / "trajectory.json").exists()


@pytest.mark.parametrize("run_flow", [DaydreamRunFlow.NORMAL, DaydreamRunFlow.IMPROVE])
async def test_full_archive_round_trip_fix_test_backend_columns(
    tmp_path: Path,
    archive_dir: Path,
    run_flow: DaydreamRunFlow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path round-trip: IMPROVE omits fix/test_backend (keys + NULL SQL
    columns); NORMAL (deep-family) records both (keys present + non-NULL)."""
    recorder = _make_round_trip_fixture(tmp_path, run_flow)
    lifecycle_times = iter(("2026-05-31T10:00:00.000000Z", "2026-05-31T10:00:08.500000Z"))
    monkeypatch.setattr("daydream.trajectory.now_iso", lambda: next(lifecycle_times))

    async with recorder:
        pass

    # Collapse the IMPROVE/NORMAL branch pair into a single expected value: the
    # shared assert helper applies it to both the manifest keys and the SQL columns.
    backend: str | None = None if run_flow is DaydreamRunFlow.IMPROVE else "claude"
    _assert_round_trip_bundle(
        archive_dir, recorder, fix_backend=backend, test_backend=backend,
    )


async def test_archive_round_trip_projects_eval_location_metrics(
    tmp_path: Path,
    archive_dir: Path,
) -> None:
    """#1106: the location/duplication eval axes reach manifest.json and index.db.

    Real path: the runner's archive callback runs the deterministic eval pass,
    writes evaluation.json, builds the manifest from it, and indexes the row.
    The two new headline metrics must be present in the manifest metrics block
    and in the SQL row, carrying exactly the value evaluation.json reported
    (including ``None``/NULL when the axis is undefined for this run) — never
    an invented 0.
    """
    recorder = _make_round_trip_fixture(tmp_path, DaydreamRunFlow.NORMAL, run_eval=True)

    async with recorder:
        pass

    run_dir = archive_dir / "runs" / recorder.session_id
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    expected_rate = evaluation.get("location", {}).get("in_hunk_rate")
    expected_pairs = (
        evaluation.get("findings", {}).get("shipped_duplication", {}).get("near_duplicate_pairs")
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = manifest["metrics"]
    assert "location_in_hunk_rate" in metrics  # the axis has a surface at all
    assert "shipped_duplicate_pairs" in metrics
    assert metrics["location_in_hunk_rate"] == expected_rate
    assert metrics["shipped_duplicate_pairs"] == expected_pairs

    conn = sqlite3.connect(str(archive_dir / "index.db"))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT location_in_hunk_rate, shipped_duplicate_pairs FROM runs WHERE session_id = ?",
            (recorder.session_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["location_in_hunk_rate"] == expected_rate
    assert row["shipped_duplicate_pairs"] == expected_pairs


# on_write failure does not raise
async def test_on_write_failure_does_not_raise(tmp_path: Path) -> None:
    """If on_write raises, the context manager exits cleanly and trajectory is still written."""

    def on_write_boom(
        recorder: TrajectoryRecorder,
        snapshot: RunWriteSnapshot,
    ) -> None:
        raise RuntimeError("archive exploded")

    recorder = make_recorder(tmp_path, on_write=on_write_boom)
    async with recorder:
        _add_user_step(recorder)

    # Trajectory should still be on disk despite the callback failure
    traj_path = tmp_path / ".daydream" / "trajectory.json"
    assert traj_path.exists()
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    assert data["session_id"] == recorder.session_id
    assert len(data["steps"]) == 1


# CLI --no-archive flag
def test_cli_no_archive_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-archive sets config.archive to False."""
    from daydream.cli import _parse_args

    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/fake", "--no-archive"])
    config = _parse_args()
    assert config.archive is False


# CLI --no-eval flag
def test_cli_no_eval_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-eval opts out: sets config.run_eval to False."""
    from daydream.cli import _parse_args

    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/fake", "--no-eval"])
    config = _parse_args()
    assert config.run_eval is False


# CLI defaults for archive and eval
def test_cli_defaults_archive_and_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --no-archive or --no-eval, archive=True and run_eval=True (eval on by default)."""
    from daydream.cli import _parse_args

    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/fake"])
    config = _parse_args()
    assert config.archive is True
    assert config.run_eval is True


# HF upload hook fires through the archive callback when configured
async def test_archive_callback_uploads_to_hub_when_configured(
    tmp_path: Path,
    archive_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured trajectory_hub_repo makes the finalizer call the uploader after manifest write."""
    from daydream.runner import RunConfig

    uploaded: list[tuple[Any, ...]] = []

    def _fake_upload(run_dir: Path, repo_id: str, session_id: str) -> bool:
        uploaded.append((str(run_dir), repo_id, session_id))
        return True

    monkeypatch.setattr("daydream.archive.hub.upload_run_bundle", _fake_upload)

    target_dir = tmp_path / "project"
    target_dir.mkdir()
    daydream_dir = target_dir / ".daydream"
    daydream_dir.mkdir()
    (target_dir / ".review-output.md").write_text("# Review\nLooks good.\n", encoding="utf-8")

    config = RunConfig(
        trajectory_hub_repo="acme/dd-trajectories",
        archive=True,
        run_eval=False,
        dump_artifacts=None,
    )
    recorder = make_recorder(
        target_dir, on_write=_strict_archive_callback(config, target_dir)
    )
    _add_user_step(recorder)
    async with recorder:
        pass

    assert len(uploaded) == 1
    repo_id, session = uploaded[0][1], uploaded[0][2]
    assert repo_id == "acme/dd-trajectories"
    assert session == recorder.session_id
    # The bundle is uploaded from the private assembly directory, before it is
    # installed under its session id — so the manifest the hook waited for is
    # in the installed run directory once finalization completes.
    assert Path(uploaded[0][0]) != archive_dir / "runs" / recorder.session_id
    assert (archive_dir / "runs" / recorder.session_id / "manifest.json").is_file()


@pytest.mark.parametrize("filename,body,set_hf_token", [
    (None, None, False),
    ("pyproject.toml", TARGET_HUB_KEY_CONFIG, True),
    (".daydream.toml", 'trajectory_hub_repo = "evil/repo"\n', True),
])
async def test_archive_callback_does_not_upload_when_unconfigured(
    tmp_path: Path,
    archive_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str | None,
    body: str | None,
    set_hf_token: bool,
) -> None:
    """Without an operator-configured trajectory_hub_repo, the uploader is never called.

    Covers the bare unconfigured case plus a target checkout setting the key in
    pyproject.toml or .daydream.toml — the ignored key never reaches the
    uploader even with HF_TOKEN present."""
    from daydream.config_file import load_file_config
    from daydream.runner import RunConfig

    calls: list[Any] = []
    if set_hf_token:
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)

    def _fake_upload(*args: object, **kwargs: object) -> bool:
        calls.append(args)
        return True

    monkeypatch.setattr("daydream.archive.hub.upload_run_bundle", _fake_upload)

    target_dir = tmp_path / "project"
    target_dir.mkdir()
    if filename is not None and body is not None:
        (target_dir / filename).write_text(body, encoding="utf-8")
    config = RunConfig(
        archive=True,
        run_eval=False,
        file_config=load_file_config(target_dir) if filename is not None else None,
    )
    recorder = make_recorder(
        target_dir, on_write=_strict_archive_callback(config, target_dir)
    )
    _add_user_step(recorder)
    async with recorder:
        pass

    assert calls == []


# Signal-flush (partial) archives must never trigger the blocking HF upload
@pytest.mark.parametrize("successful", [False, True])
async def test_archive_upload_tracks_run_success(
    tmp_path: Path,
    archive_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    successful: bool,
) -> None:
    """The bundle is always archived locally; only a successful run uploads.

    The runner finalizes once and passes ``upload=successful``, so an
    interrupted or failed run is archived without the blocking HF upload — that
    call must never hang a SIGINT/SIGTERM shutdown on a network round trip."""
    from daydream.runner import RunConfig

    uploaded: list[tuple[Any, ...]] = []

    def _fake_upload(run_dir: Path, repo_id: str, session_id: str) -> bool:
        uploaded.append((str(run_dir), repo_id, session_id))
        return True

    monkeypatch.setattr("daydream.archive.hub.upload_run_bundle", _fake_upload)

    target_dir = tmp_path / "project"
    target_dir.mkdir()
    daydream_dir = target_dir / ".daydream"
    daydream_dir.mkdir()
    (target_dir / ".review-output.md").write_text("# Review\nLooks good.\n", encoding="utf-8")

    config = RunConfig(
        trajectory_hub_repo="acme/dd-trajectories",
        archive=True,
        run_eval=False,
        dump_artifacts=None,
    )
    recorder = make_recorder(
        target_dir,
        on_write=_strict_archive_callback(
            config, target_dir, unsuccessful=not successful
        ),
    )
    _add_user_step(recorder)
    async with recorder:
        pass

    assert (archive_dir / "runs" / recorder.session_id / "manifest.json").is_file()
    if successful:
        assert [row[1:] for row in uploaded] == [
            ("acme/dd-trajectories", recorder.session_id)
        ]
    else:
        assert uploaded == []


async def test_signal_flush_archive_uses_one_immutable_cutoff_for_all_documents(
    tmp_path: Path,
    archive_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each frozen snapshot archives as one immutable, self-consistent bundle.

    The runner finalizes exactly one snapshot per run, so each status is
    finalized into its own archive root here: the mid-flight partial (whose
    documents must all share one cutoff, and whose still-running forks read as
    malformed invocations) and the completed run (whose forks have closed)."""
    from daydream.runner import RunConfig

    target = tmp_path / "frozen"
    target.mkdir()
    config = RunConfig(target=str(target), archive=True, run_eval=True)
    snapshots: list[RunWriteSnapshot] = []
    roots = {
        status: tmp_path / f"archive-{status}" for status in ("partial", "complete")
    }

    def callback(recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot) -> None:
        snapshots.append(snapshot)
        monkeypatch.setenv("DAYDREAM_ARCHIVE_DIR", str(roots[snapshot.status]))
        _finalize_strict_archive(
            recorder, snapshot, config, target, upload=snapshot.status == "complete"
        )

    recorder = make_recorder(target, on_write=callback)
    entered = {name: anyio.Event() for name in ("a", "b")}
    release = {name: anyio.Event() for name in entered}
    async with recorder:
        _add_user_step(recorder)
        async with anyio.create_task_group() as task_group:
            for name in entered:
                task_group.start_soon(
                    _hold_archive_fork,
                    recorder,
                    name,
                    entered[name],
                    release[name],
                )
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
            assert evaluation["timing"]["agent_completeness"] == {
                "total": 6,
                "attributed": 0,
                "unattributed": 6,
            }
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
    final_evaluation = json.loads(
        (roots["complete"] / "runs" / recorder.session_id / "evaluation.json").read_text()
    )
    assert final_evaluation["timing"]["agent_completeness"] == {
        "total": 6,
        "attributed": 0,
        "unattributed": 6,
    }
    assert final_evaluation["timing"]["diagnostics"]["malformed_invocation"] == 0


# --no-archive + --dump-artifacts: bundle still dumped, upload never fires
async def test_archive_callback_no_archive_dump_artifacts_skips_upload(
    tmp_path: Path,
    archive_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-archive with --dump-artifacts still copies the bundle to the dump dir but never uploads."""
    from daydream.runner import RunConfig

    uploaded: list[Any] = []

    def _fake_upload(*args: object, **kwargs: object) -> bool:
        uploaded.append(args)
        return True

    monkeypatch.setattr("daydream.archive.hub.upload_run_bundle", _fake_upload)

    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    daydream_dir = target_dir / ".daydream"
    daydream_dir.mkdir()
    (target_dir / ".review-output.md").write_text("# Review\nLooks good.\n", encoding="utf-8")

    config = RunConfig(
        trajectory_hub_repo="acme/dd-trajectories",
        archive=False,
        run_eval=False,
        dump_artifacts=str(dump_dir),
    )
    recorder = make_recorder(
        target_dir,
        on_write=_strict_archive_callback(config, target_dir, dump_path=dump_dir),
    )
    _add_user_step(recorder)
    async with recorder:
        pass

    assert uploaded == []
    # Existing --dump-artifacts behavior is preserved: the full bundle lands in
    # the dump dir even though centralized archiving + the HF upload are off.
    assert (dump_dir / "manifest.json").is_file()
    assert (dump_dir / "review-output.md").is_file()


# Recorder → archive redaction parity (issue #455, Task 3)
async def test_runner_archive_round_trip_redacts_structured_tool_credentials(
    tmp_path: Path,
    archive_dir: Path,
) -> None:
    """A structured tool call under a sensitive key is redacted identically in the
    live trajectory and the archived copy; both pass the ATIF validator."""
    from daydream.agent import run_agent
    from daydream.atif import validate as atif_validate
    from daydream.backends import ResultEvent, ToolResultEvent, ToolStartEvent
    from daydream.runner import RunConfig
    from tests.harness.backend import ScriptedBackend

    sentinel = "opaque-test-only-sentinel"
    target_dir = tmp_path / "project"
    target_dir.mkdir()

    backend = ScriptedBackend(
        events=(
            ToolStartEvent(
                id="t1",
                name="ListDir",
                input={
                    "dir": "/tmp",
                    "apiKey": {"nested": sentinel},
                    "displayName": "visible",
                },
            ),
            ToolResultEvent(
                id="t1",
                output='{"status": "ok", "token": "opaque-test-only-sentinel"}',
                is_error=False,
            ),
            ResultEvent(structured_output=None, continuation=None),
        )
    )

    config = RunConfig(target=str(target_dir), archive=True, run_eval=False)
    recorder = make_recorder(
        target_dir,
        run_flow=DaydreamRunFlow.NORMAL,
        on_write=_strict_archive_callback(config, target_dir),
    )

    async with recorder:
        await run_agent(backend, target_dir, "inspect configuration", phase=DaydreamPhase.REVIEW)

    live = json.loads(recorder.path.read_text(encoding="utf-8"))
    run_dir = archive_dir / "runs" / recorder.session_id
    archived = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))

    assert live == archived
    assert atif_validate(live) is True
    blob = json.dumps(live)
    assert sentinel not in blob
    assert "[REDACTED_CREDENTIAL]" in blob
    assert "visible" in blob
