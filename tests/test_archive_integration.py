# tests/test_archive_integration.py
"""Integration tests for the TrajectoryRecorder on_write callback and archive pipeline.

Verifies that the on_write callback fires at the right times, that the full
archive round-trip produces valid bundles, and that CLI flags for --no-archive
and --no-eval are parsed correctly into RunConfig.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest

from daydream.atif import Step
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    TrajectoryRecorder,
    now_iso,
)
from tests.harness.config import TARGET_HUB_KEY_CONFIG
from tests.harness.trajectory import make_recorder

if TYPE_CHECKING:
    from daydream.runner import RunConfig


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


# --- shared round-trip setup for the archive on_write tests ---
def _make_round_trip_fixture(
    tmp_path: Path,
    run_flow: DaydreamRunFlow,
) -> tuple[RunConfig, Callable[[TrajectoryRecorder, str], None], TrajectoryRecorder]:
    """Build the full archive round-trip fixture: minimal .daydream/ scaffolding,
    RunConfig, archive callback, and a recorder primed with two steps spaced 8.5s
    apart so the derived wall-clock span is deterministic."""
    from daydream.runner import RunConfig, _make_archive_callback

    target_dir = tmp_path / "project"
    target_dir.mkdir()
    daydream_dir = target_dir / ".daydream"
    daydream_dir.mkdir()
    (target_dir / ".review-output.md").write_text("# Review\nLooks good.\n")
    findings_src = target_dir / "findings" / "findings.json"
    findings_src.parent.mkdir(parents=True)
    findings_src.write_text(
        '{"schema_version": 1, "findings": [{"fingerprint": "deadbeef"}]}'
    )

    config = RunConfig(
        target=str(target_dir),
        skill="python",
        backend="claude",
        archive=True,
        run_eval=False,
        findings_out="findings/findings.json",
    )
    callback = _make_archive_callback(config, target_dir)
    assert callback is not None

    recorder = TrajectoryRecorder(
        path=daydream_dir / "trajectory.json",
        run_flow=run_flow,
        target_dir=target_dir,
        agent_model_name="opus",
        session_id="test",
        on_write=callback,
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
    return config, callback, recorder


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

    def on_write(recorder: TrajectoryRecorder, status: str) -> None:
        callback_calls.append((recorder.session_id, status))

    recorder = make_recorder(tmp_path, on_write=on_write)
    async with recorder:
        pass

    assert len(callback_calls) == 0
    assert not (tmp_path / ".daydream" / "trajectory.json").exists()


# Full archive round-trip via on_write
async def test_full_archive_round_trip(tmp_path: Path, archive_dir: Path) -> None:
    """_make_archive_callback wires archive_run through on_write, producing manifest + SQLite row."""
    _, _, recorder = _make_round_trip_fixture(tmp_path, DaydreamRunFlow.NORMAL)

    async with recorder:
        pass

    assert recorder.path.exists()
    _assert_round_trip_bundle(
        archive_dir, recorder, fix_backend="claude", test_backend="claude",
    )


@pytest.mark.parametrize("run_flow", [DaydreamRunFlow.NORMAL, DaydreamRunFlow.IMPROVE])
async def test_full_archive_round_trip_fix_test_backend_columns(
    tmp_path: Path, archive_dir: Path, run_flow: DaydreamRunFlow,
) -> None:
    """Real-path round-trip: IMPROVE omits fix/test_backend (keys + NULL SQL
    columns); NORMAL (deep-family) records both (keys present + non-NULL)."""
    _, _, recorder = _make_round_trip_fixture(tmp_path, run_flow)

    async with recorder:
        pass

    # Collapse the IMPROVE/NORMAL branch pair into a single expected value: the
    # shared assert helper applies it to both the manifest keys and the SQL columns.
    backend: str | None = None if run_flow is DaydreamRunFlow.IMPROVE else "claude"
    _assert_round_trip_bundle(
        archive_dir, recorder, fix_backend=backend, test_backend=backend,
    )


# on_write failure does not raise
async def test_on_write_failure_does_not_raise(tmp_path: Path) -> None:
    """If on_write raises, the context manager exits cleanly and trajectory is still written."""

    def on_write_boom(recorder: TrajectoryRecorder, status: str) -> None:
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
    tmp_path: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured trajectory_hub_repo makes _archive_run_inner call the uploader after manifest write."""
    from daydream.runner import RunConfig, _make_archive_callback

    uploaded: list[tuple] = []

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
    cb = _make_archive_callback(config, target_dir)
    assert cb is not None

    from tests.harness.trajectory import make_recorder
    recorder = make_recorder(tmp_path, on_write=cb)
    from tests.test_archive_integration import _add_user_step  # reuse the step helper
    _add_user_step(recorder)
    async with recorder:
        pass

    assert len(uploaded) == 1
    repo_id, session = uploaded[0][1], uploaded[0][2]
    assert repo_id == "acme/dd-trajectories"
    assert session == recorder.session_id
    # run_dir on disk contains the manifest the hook must wait for
    run_dir = Path(uploaded[0][0])
    assert (run_dir / "manifest.json").is_file()


@pytest.mark.parametrize("filename,body,set_hf_token", [
    (None, None, False),
    ("pyproject.toml", TARGET_HUB_KEY_CONFIG, True),
    (".daydream.toml", 'trajectory_hub_repo = "evil/repo"\n', True),
])
async def test_archive_callback_does_not_upload_when_unconfigured(
    tmp_path: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
    filename: str | None, body: str | None, set_hf_token: bool,
) -> None:
    """Without an operator-configured trajectory_hub_repo, the uploader is never called.

    Covers the bare unconfigured case plus a target checkout setting the key in
    pyproject.toml or .daydream.toml — the ignored key never reaches the
    uploader even with HF_TOKEN present."""
    from daydream.config_file import load_file_config
    from daydream.runner import RunConfig, _make_archive_callback
    from tests.harness.trajectory import make_recorder
    from tests.test_archive_integration import _add_user_step

    calls: list = []
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
    cb = _make_archive_callback(config, target_dir)
    recorder = make_recorder(tmp_path, on_write=cb)
    _add_user_step(recorder)
    async with recorder:
        pass

    assert calls == []


# Signal-flush (partial) archives must never trigger the blocking HF upload
async def test_archive_callback_partial_status_skips_hf_upload(
    tmp_path: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial (signal-flush) archive never uploads; a complete one does."""
    from daydream.runner import RunConfig, _make_archive_callback
    from tests.harness.trajectory import make_recorder
    from tests.test_archive_integration import _add_user_step

    uploaded: list[tuple] = []

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
    cb = _make_archive_callback(config, target_dir)
    assert cb is not None

    recorder = make_recorder(tmp_path, on_write=cb)
    _add_user_step(recorder)

    # Signal flush fires on_write("partial") synchronously; the blocking HF
    # upload must be skipped so Ctrl-C/Ctrl-\ shutdown never hangs on a network call.
    recorder.write_partial()
    assert uploaded == []

    # Normal completion fires on_write("complete"); the upload must run.
    async with recorder:
        pass

    assert len(uploaded) == 1
    assert uploaded[0][1] == "acme/dd-trajectories"
    assert uploaded[0][2] == recorder.session_id


# --no-archive + --dump-artifacts: bundle still dumped, upload never fires
async def test_archive_callback_no_archive_dump_artifacts_skips_upload(
    tmp_path: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-archive with --dump-artifacts still copies the bundle to the dump dir but never uploads."""
    from daydream.runner import RunConfig, _make_archive_callback
    from tests.harness.trajectory import make_recorder
    from tests.test_archive_integration import _add_user_step

    uploaded: list = []

    def _fake_upload(*args: object, **kwargs: object) -> bool:
        uploaded.append(args)
        return True

    monkeypatch.setattr("daydream.archive.hub.upload_run_bundle", _fake_upload)

    dump_dir = tmp_path / "dump"
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
    cb = _make_archive_callback(config, target_dir)
    assert cb is not None

    recorder = make_recorder(tmp_path, on_write=cb)
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
    tmp_path: Path, archive_dir: Path,
) -> None:
    """A structured tool call under a sensitive key is redacted identically in the
    live trajectory and the archived copy; both pass the ATIF validator."""
    from daydream.agent import run_agent
    from daydream.atif import validate as atif_validate
    from daydream.backends import ResultEvent, ToolResultEvent, ToolStartEvent
    from daydream.runner import RunConfig, _open_recorder
    from tests.harness.backend import ScriptedBackend

    sentinel = "opaque-test-only-sentinel"
    target_dir = tmp_path / "project"
    target_dir.mkdir()

    backend = ScriptedBackend(events=(
        ToolStartEvent(
            id="t1", name="ListDir",
            input={"dir": "/tmp", "apiKey": {"nested": sentinel}, "displayName": "visible"},
        ),
        ToolResultEvent(
            id="t1",
            output='{"status": "ok", "token": "opaque-test-only-sentinel"}',
            is_error=False,
        ),
        ResultEvent(structured_output=None, continuation=None),
    ))

    config = RunConfig(target=str(target_dir), archive=True, run_eval=False)
    recorder = _open_recorder(
        config=config, target_dir=target_dir, work=None, flow_kind=DaydreamRunFlow.NORMAL,
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
