"""Regression: ``scripts/run_demo_python.py --skip-setup`` must reuse an existing repo."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script_module(name: str) -> types.ModuleType:
    """Load a scripts/ module without mutating interpreter sys.path."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_run_demo_python_skip_setup_reuses_existing_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--skip-setup`` must not create or touch an existing repo, and must run daydream on it."""
    # Load scripts/ modules via importlib so we never mutate interpreter sys.path.
    _demo_common = _load_script_module("_demo_common")
    run_demo_python = _load_script_module("run_demo_python")

    class _SubprocessRecorder:
        """Captures the daydream subprocess command without executing it."""

        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def run(self, *args: object, **kwargs: object) -> types.SimpleNamespace:
            self.calls.append(args)
            return types.SimpleNamespace(returncode=0)

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