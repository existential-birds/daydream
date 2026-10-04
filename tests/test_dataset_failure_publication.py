"""Optional persistence faults preserve the runner's protected output publication."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from daydream import runner
from daydream.archive import hub
from daydream.run_config import RunConfig
from tests.conftest import ExtDir
from tests.test_runner import _ControlledBackend, _feature_repo, _run_private, _write_probe_flow

_SECRET_FAILURE = "credential=SECRET_COLLECTION_FAILURE /private/runtime"
_REVIEW = "# Completed review\nNo findings.\n"


def _write_review_flow(ext_dir: ExtDir, *, exit_code: int = 0) -> None:
    """Write completed outputs through the real runner-owned session routes."""
    ext_dir.write_module(
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep, Stop\n"
        "from daydream.trajectory import DaydreamPhase\n"
        "async def _review(ctx):\n"
        "    assert ctx.artifacts is not None\n"
        "    await run_agent(ctx.backend_for('review'), ctx.work.repo, 'REVIEW', "
        "phase=DaydreamPhase.REVIEW)\n"
        "    output = ctx.artifacts.live_path_for(ctx.work.source / '.review-output.md', repo=ctx.work.repo)\n"
        f"    output.write_text({_REVIEW!r}, encoding='utf-8')\n"
        f"    return Stop({exit_code})\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='review', run=_review))\n"
        "    registry.set_flow('dataset-review', ['review'])\n"
    )


def _assert_sanitized_diagnostic(output: str) -> None:
    assert "Data Collection" in output
    assert "SECRET_COLLECTION_FAILURE" not in output
    assert "/private/runtime" not in output


def _inject_collection_fault(monkeypatch: pytest.MonkeyPatch, archive_dir: Path, fault: str) -> list[str]:
    reached: list[str] = []
    if fault == "evaluation":
        def fail_evaluation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            reached.append(fault)
            raise ValueError(_SECRET_FAILURE)

        monkeypatch.setattr("daydream.eval.analyzer.analyze_session", fail_evaluation)
    elif fault == "index":
        connect: Callable[..., sqlite3.Connection] = sqlite3.connect

        def fail_index(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
            if Path(database) == archive_dir / "index.db":
                reached.append(fault)
                raise sqlite3.OperationalError(_SECRET_FAILURE)
            return connect(database, *args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", fail_index)
    else:
        assert fault == "filesystem"
        mkdir = Path.mkdir

        def fail_archive_mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
            if path.parent == archive_dir / "runs" and path.name.endswith(".finalizing"):
                reached.append(fault)
                raise PermissionError(_SECRET_FAILURE)
            mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_archive_mkdir)
    return reached


@pytest.mark.parametrize("fault", ["evaluation", "index", "filesystem"])
async def test_collection_failure_preserves_completed_review_and_explicit_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, archive_dir: Path,
    capsys: pytest.CaptureFixture[str], fault: str,
) -> None:
    repo = _feature_repo(tmp_path)
    _write_review_flow(ext_dir)
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: _ControlledBackend())
    reached = _inject_collection_fault(monkeypatch, archive_dir, fault)
    external = tmp_path / "explicit-trajectory.json"
    external.write_text("prior full\n")
    dump = tmp_path / "diagnostic-dump"
    dump.mkdir()
    (dump / "operator.txt").write_text("prior diagnostic\n")
    result = await _run_private(
        RunConfig(
            target=str(repo), base="main", flow_name="dataset-review", trajectory_path=external,
            archive=True, run_eval=fault == "evaluation", dump_artifacts=str(dump), non_interactive=True,
        ), tmp_path,
    )
    assert result == 0
    assert reached == [fault]
    public_runs = list((repo / ".daydream" / "runs").iterdir())
    assert len(public_runs) == 1
    assert (repo / ".review-output.md").read_text() == _REVIEW
    assert external.read_bytes() == (public_runs[0] / "trajectory.json").read_bytes()
    assert not external.with_suffix(".json.partial").exists()
    if fault == "index":
        assert (dump / "trajectory.json").read_bytes() == external.read_bytes()
        assert json.loads((dump / "manifest.json").read_text())["session_id"] == public_runs[0].name
    else:
        assert (dump / "operator.txt").read_text() == "prior diagnostic\n"
        assert not (dump / "trajectory.json").exists()
    _assert_sanitized_diagnostic(capsys.readouterr().out)


async def test_hf_network_failure_preserves_completed_review_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, archive_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _feature_repo(tmp_path)
    _write_review_flow(ext_dir)
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: _ControlledBackend())
    monkeypatch.setenv("HF_TOKEN", "test-token")
    attempts: list[Path] = []

    class FailingHub:
        def create_repo(self, **_kwargs: Any) -> None:
            pass

        def repo_info(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(private=True)

        def upload_folder(self, *, folder_path: str, **_kwargs: Any) -> None:
            attempts.append(Path(folder_path))
            raise ConnectionError(_SECRET_FAILURE)

    monkeypatch.setattr(hub, "HfApi", FailingHub)
    external = tmp_path / "explicit-trajectory.json"
    result = await _run_private(
        RunConfig(
            target=str(repo), base="main", flow_name="dataset-review", trajectory_path=external,
            archive=True, run_eval=False, trajectory_hub_repo="test/new-runs", non_interactive=True,
        ), tmp_path,
    )
    assert result == 0
    assert len(attempts) == 1
    public_runs = list((repo / ".daydream" / "runs").iterdir())
    assert len(public_runs) == 1
    public = public_runs[0]
    assert (repo / ".review-output.md").read_text() == _REVIEW
    assert external.read_bytes() == (public / "trajectory.json").read_bytes()
    archived = archive_dir / "runs" / public.name
    assert (archived / "trajectory.json").read_bytes() == external.read_bytes()
    assert json.loads((archived / "manifest.json").read_text())["archive_status"] == "complete"
    _assert_sanitized_diagnostic(capsys.readouterr().out)


@pytest.mark.parametrize("outcome", ["failed", "interrupted", "partial"])
async def test_collection_failure_preserves_primary_and_partial_output_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, archive_dir: Path,
    capsys: pytest.CaptureFixture[str], outcome: str,
) -> None:
    repo = _feature_repo(tmp_path)
    baseline = b"prior completed review\x00\n"
    (repo / ".review-output.md").write_bytes(baseline)
    external = tmp_path / "explicit-trajectory.json"
    external.write_bytes(b"prior full\n")
    if outcome == "partial":
        _write_review_flow(ext_dir, exit_code=3)
        backend = _ControlledBackend()
        flow_name = "dataset-review"
    else:
        flow_name = "dataset-probe"
        _write_probe_flow(ext_dir, flow_name)
        backend = _ControlledBackend(
            "hang" if outcome == "interrupted" else "raise", error=RuntimeError("model failed"),
        )
    monkeypatch.setattr(runner, "create_backend", lambda *_args, **_kwargs: backend)
    reached = _inject_collection_fault(monkeypatch, archive_dir, "evaluation")
    config = RunConfig(
        target=str(repo), base="main", flow_name=flow_name, trajectory_path=external,
        archive=True, run_eval=True, non_interactive=True,
    )
    if outcome == "failed":
        with pytest.raises(RuntimeError) as raised:
            await _run_private(config, tmp_path)
        assert raised.value is backend.error
    elif outcome == "interrupted":
        caught: list[BaseException] = []

        async def invoke() -> None:
            try:
                await _run_private(config, tmp_path)
            except BaseException as exc:
                caught.append(exc)
                raise

        with anyio.fail_after(20):
            async with anyio.create_task_group() as group:
                group.start_soon(invoke)
                await backend.entered.wait()
                group.cancel_scope.cancel()
        assert len(caught) == 1
        assert isinstance(caught[0], anyio.get_cancelled_exc_class())
        assert backend.cancelled is True
    else:
        assert await _run_private(config, tmp_path) == 3
    assert reached == ["evaluation"]
    expected_review = _REVIEW.encode() if outcome == "partial" else baseline
    assert (repo / ".review-output.md").read_bytes() == expected_review
    public_runs = list((repo / ".daydream" / "runs").iterdir())
    assert len(public_runs) == 1
    # The recorder's final write is complete even when its body failed or was
    # cancelled; the document carries the partial marker and uses the full route.
    assert external.read_bytes() == (public_runs[0] / "trajectory.json").read_bytes()
    assert not external.with_suffix(".json.partial").exists()
    assert json.loads(external.read_text())["extra"].get("partial", False) is (outcome != "partial")
    assert not list((archive_dir / "runs").iterdir())
    _assert_sanitized_diagnostic(capsys.readouterr().out)
