"""Regression: ``scripts/run_demo_python.py --skip-setup`` must reuse an existing repo."""

import sys
import types
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import _demo_common  # noqa: E402
import run_demo_python  # noqa: E402


class _SubprocessRecorder:
    """Captures the daydream subprocess command without executing it."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def run(self, *args: object, **kwargs: object) -> types.SimpleNamespace:
        self.calls.append(args)
        return types.SimpleNamespace(returncode=0)


def test_run_demo_python_skip_setup_reuses_existing_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--skip-setup`` must not create or touch an existing repo, and must run daydream on it."""
    target = tmp_path / "existing-repo"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("precious-user-data\n")

    def _fail_on_create(*_args: object, **_kwargs: object) -> None:
        pytest.fail("create_test_repo must not run when --skip-setup is passed")

    monkeypatch.setattr(_demo_common, "create_test_repo", _fail_on_create)
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(_demo_common.subprocess, "run", recorder.run)
    monkeypatch.setattr(sys, "argv", ["run_demo_python.py", str(target), "--skip-setup"])

    assert run_demo_python.main() == 0
    assert sentinel.read_text() == "precious-user-data\n"
    assert len(recorder.calls) == 1
    cmd = recorder.calls[0][0]
    assert isinstance(cmd, list)
    assert str(target) in [str(c) for c in cmd]
