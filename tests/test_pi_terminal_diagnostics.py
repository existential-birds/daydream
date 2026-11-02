"""Public Pi failure evidence through real staged review and frozen record storage."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from daydream.agent import run_agent
from daydream.backends.pi import PiBackend, PiError
from daydream.trajectory import DaydreamPhase
from tests.harness.trajectory import make_recorder


@pytest.mark.parametrize("failure_kind", ["native-error", ])
async def test_pi_failure_records_redacted_process_tail_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str,
) -> None:
    """A real child preserves late stderr and distinguishes host cleanup from exit.

    Mock process replays cannot prove late merged output is drained or that
    exit/cleanup observations describe the actual process on both paths.
    The graceful child ignores repeated SIGTERM during interpreter shutdown
    so cleanup still observes its promised zero exit.
    """
    binary = tmp_path / "bin"
    binary.mkdir()
    child = binary / "pi"
    child.write_text(f"#!{sys.executable}\n" + '''
import json, signal, sys, time
def stop(*_):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print('late cleanup diagnostic', file=sys.stderr, flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
for index in range(25):
    print(f'early diagnostic {index}', file=sys.stderr)
print('TOKEN=private-stderr-canary; ' + 'safe ' * 1000, file=sys.stderr, flush=True)
print('-----BEGIN PRIVATE KEY-----', file=sys.stderr)
for index in range(30):
    print(f'private-pem-body-canary-{index}', file=sys.stderr)
''' + ("print('-----END PRIVATE KEY-----', file=sys.stderr, flush=True)\n" if failure_kind == "native-error" else "")
        + ('''print(json.dumps({'type': 'turn_end', 'message': {
    'role': 'assistant', 'stopReason': 'error', 'errorMessage': 'terminated',
    'usage': {'input': 10, 'output': 2, 'cost': {'total': 0.01}}}}), flush=True)
time.sleep(300)
''' if failure_kind == "native-error" else "raise SystemExit(7)\n"))
    child.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    backend = PiBackend(model="fixture-model")
    backend.retry_attempts = 0
    recorder = make_recorder(tmp_path)
    async with recorder:
        with pytest.raises(PiError) as failure:
            await run_agent(backend, tmp_path, "Review", phase=DaydreamPhase.REVIEW,
                            progress_callback=lambda _: None)
    assert failure.value.category == ("STREAM_DROP" if failure_kind == "native-error" else "PROCESS_EXIT")
    document = json.loads(recorder.path.read_text())
    diagnostic = next(item for step in document["steps"]
                      for item in step.get("extra", {}).get("backend_diagnostics", [])
                      if item["code"] == "pi_process_failure")
    metadata = diagnostic["metadata"]
    assert metadata["process_running_at_cleanup"] is (failure_kind == "native-error")
    assert metadata["process_returncode"] == (0 if failure_kind == "native-error" else 7)
    assert len(metadata["diagnostic_tail"]) == 20 and metadata["diagnostic_tail_truncated"] is True
    assert all(len(line) <= 1024 for line in metadata["diagnostic_tail"])
    assert "private-pem-body-canary" not in json.dumps(document)
    assert "private-stderr-canary" not in json.dumps(document)
    assert any("[REDACTED_ENV_VAR]" in line for line in metadata["diagnostic_tail"])
    assert any("[REDACTED_PEM_KEY]" in line for line in metadata["diagnostic_tail"])
    if failure_kind == "native-error":
        assert metadata["diagnostic_tail"][-1] == "late cleanup diagnostic"
        assert document["final_metrics"]["total_cost_usd"] == 0.01
    lifecycle = document["extra"]["subtrajectories"][0]["backend_lifecycle"]
    assert lifecycle["process_returncode"] == metadata["process_returncode"]
    assert lifecycle["teardown_started_s"] is not None and lifecycle["teardown_completed_s"] is not None
    assert lifecycle["stdout_eof_s"] is not None
    if failure_kind == "native-error":
        assert lifecycle["stdout_eof_s"] >= lifecycle["teardown_started_s"]
