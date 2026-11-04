"""Shared fan-out admission across the CLI backends."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from daydream.backends import (
    effective_fanout_concurrency,
    resolve_fanout_concurrency,
)
from tests.harness.protocol_cli import install_protocol_cli


@pytest.mark.parametrize("raw,expected", [("6", 6), ("0", 10)])
def test_endpoint_fanout_admission_and_workflow_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    raw: str,
    expected: int,
) -> None:
    """The shared positive rule owns admission; actual backend journeys own defaults and forwarding."""
    monkeypatch.setenv("DAYDREAM_PI_FANOUT_CONCURRENCY", raw)
    hint = resolve_fanout_concurrency("DAYDREAM_PI_FANOUT_CONCURRENCY", 10)
    assert hint == expected
    assert effective_fanout_concurrency(2, SimpleNamespace(fanout_concurrency=hint)) == 2
    assert ("must be positive; using default 10" in caplog.text) is (raw == "0")


def test_fresh_process_loads_only_the_selected_native_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI help and per-phase selection load only the selected adapter; public adapter names stay importable.

    Only a fresh interpreter proves the module graph: the suite has already loaded every adapter.
    """
    fixture = install_protocol_cli(tmp_path / "external", "pi")
    target = tmp_path / "repo"
    target.mkdir()
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PI_PROVIDER", "nous")
    monkeypatch.delenv("PI_API_KEY", raising=False)
    script = """
import asyncio
import contextlib
import io
import sys
from pathlib import Path

assert not any(name == "daydream" or name.startswith("daydream.") for name in sys.modules)
# Ordinary Pi runs need no host admission endpoint; only the selected adapter may load an SDK.
sys.modules["daydream.host_tools_server"] = None
sys.modules["claude_agent_sdk"] = None
ADAPTERS = {f"daydream.backends.{name}" for name in ("claude", "codex", "pi", "osprey")}

def loaded():
    return ADAPTERS & set(sys.modules)

from daydream.cli import main
with contextlib.redirect_stdout(io.StringIO()):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0, exc.code
    else:
        raise AssertionError("--help returned")
assert loaded() == set(), loaded()

from daydream.backends import ResultEvent, TextEvent
from daydream.run_config import RunConfig
from daydream.runner import _resolve_backend

target = Path(sys.argv[1])
backends = {}
for name in ("pi", "osprey", "codex"):
    before = loaded()
    config = RunConfig(target=str(target), backend="claude", review_backend=name, review_model="fixture-model")
    backends[name] = _resolve_backend(config, "review", cwd=target)
    assert loaded() - before == {f"daydream.backends.{name}"}, (name, loaded() - before)
assert sys.modules["claude_agent_sdk"] is None

async def run_pi():
    events = [event async for event in backends["pi"].execute(target, "Ordinary Pi call", persist_session=False)]
    assert any(isinstance(event, ResultEvent) for event in events)
    assert [event.text for event in events if isinstance(event, TextEvent)] == ["CURRENT_REASONING_CANARY"]

asyncio.run(run_pi())
assert sys.modules["daydream.host_tools_server"] is None

del sys.modules["claude_agent_sdk"]
import daydream.backends as package
from daydream.backends import ClaudeBackend, MaxTurnsError, OspreyBackend, PiBackend
from daydream.backends import claude, osprey, pi

exported = (ClaudeBackend, MaxTurnsError, OspreyBackend, PiBackend)
owned = (claude.ClaudeBackend, claude.MaxTurnsError, osprey.OspreyBackend, pi.PiBackend)
assert all(left is right for left, right in zip(exported, owned, strict=True))
assert not hasattr(package, "NotABackend")
assert isinstance(_resolve_backend(RunConfig(target=str(target)), "review"), ClaudeBackend)
assert "claude_agent_sdk" in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script, str(target)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    (observation,) = fixture.read_observations()
    assert observation["process_outcome"] == "success"
