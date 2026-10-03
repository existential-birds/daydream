"""#1221: shared subprocess ownership and provider-owned failure checks across the CLI backends."""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from daydream.backends import _parsed_nonnegative_float, _parsed_nonnegative_int
from daydream.backends._transport import (
    PROCESS_EXIT_EXCERPT_MAX_LINES,
    CliTransport,
    process_exit_message,
    teardown,
)
from daydream.backends.codex import CodexBackend, CodexError
from daydream.backends.osprey import OspreyBackend, OspreyConfig, OspreyError
from daydream.backends.pi import (
    _PI_DEFAULT_RETRY_ATTEMPTS,
    PiBackend,
    PiError,
    _pi_retry_attempts,
)
from tests.harness.fake_cli_process import FakeCliProcess
from tests.harness.osprey_jsonl import osprey_session

DIAG = [f"diag-{i:02d}" for i in range(1, 26)]  # 25 > every capture window (codex/pi 20, osprey 10)

def _transport(monkeypatch: pytest.MonkeyPatch, proc: FakeCliProcess) -> CliTransport:
    """A real CliTransport whose child is *proc* (the transport seam is patched, not the transport)."""
    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
    return CliTransport("codex", ["codex"], limit=1024)

@pytest.mark.parametrize(("returncode", "expected"), [(0, 0), (1, 1), (-15, -15)])
async def test_wait_returns_actual_process_status(
    monkeypatch: pytest.MonkeyPatch, returncode: int, expected: int
) -> None:
    proc = FakeCliProcess([], exit_code=returncode)
    transport = _transport(monkeypatch, proc)
    await transport.start()
    assert await transport.wait() == expected
    assert proc.reaped

async def test_teardown_is_idempotent_and_drops_the_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = FakeCliProcess([], hang=True)  # returncode stays None until the teardown signals it
    transport = _transport(monkeypatch, proc)
    await transport.start()
    transports = [transport]
    await teardown(transport, transports)
    await teardown(transport, transports)  # safe to invoke twice (osprey's finally hits this path)
    assert proc.terminate_calls == 1
    assert proc.reaped and proc.stdin.closed and proc._transport.closed
    assert transports == []

def test_process_exit_message_reports_the_count_it_prints() -> None:
    assert process_exit_message(display="Codex", returncode=1, lines=DIAG) == (
        "Codex CLI exited with return code 1.\n"
        f"Codex CLI output (last {PROCESS_EXIT_EXCERPT_MAX_LINES} non-JSON lines):\n"
        + "\n".join(DIAG[-PROCESS_EXIT_EXCERPT_MAX_LINES:])
    )
    assert process_exit_message(display="Pi", returncode=1, lines=DIAG[10:20]) == (
        "Pi CLI exited with return code 1.\n"
        "Pi CLI output (last 10 non-JSON lines):\n" + "\n".join(DIAG[10:20])
    )
    assert process_exit_message(display="Pi", returncode=2, lines=[]) == (
        "Pi CLI exited with return code 2.\n"
        "(no non-JSON output captured — pi may have crashed before writing to stdout)"
    )

# Parser and facade coverage.

@pytest.mark.parametrize(
    ("raw", "expected", "warning"),
    [
        (None, 20, None),                                     # absent: silent default
        ("5", 5, None),                                       # valid override
        ("", 20, "DAYDREAM_PI_RETRY_ATTEMPTS='' is not a valid integer; using default 20"),
        ("-1", 20, "DAYDREAM_PI_RETRY_ATTEMPTS='-1' is negative; using default 20"),
        ("2.5", 20, "DAYDREAM_PI_RETRY_ATTEMPTS='2.5' is not a valid integer; using default 20"),
    ], ids=["absent", "override", "empty", "negative", "not-an-int"],
)
def test_shared_nonnegative_int_parser(
    caplog: pytest.LogCaptureFixture, raw: str | None, expected: int, warning: str | None
) -> None:
    environment = {} if raw is None else {"DAYDREAM_PI_RETRY_ATTEMPTS": raw}
    with caplog.at_level(logging.WARNING):
        assert _parsed_nonnegative_int(environment, "DAYDREAM_PI_RETRY_ATTEMPTS", 20) == expected
    if warning is None:
        assert caplog.text == ""          # a valid or absent value must not warn
    else:
        assert warning in caplog.text

@pytest.mark.parametrize(
    ("raw", "expected", "warning"),
    [
        (None, 10.0, None), ("0.5", 0.5, None),
        ("", 10.0, "DAYDREAM_PI_RETRY_BASE_DELAY_S='' is not a valid float; using default 10"),
        ("nan", 10.0, "DAYDREAM_PI_RETRY_BASE_DELAY_S='nan' is not finite; using default 10"),
        ("inf", 10.0, "DAYDREAM_PI_RETRY_BASE_DELAY_S='inf' is not finite; using default 10"),
        ("-1", 10.0, "DAYDREAM_PI_RETRY_BASE_DELAY_S='-1' is negative; using default 10"),
    ], ids=["absent", "override", "empty", "nan", "inf", "negative"],
)
def test_shared_nonnegative_float_parser(
    caplog: pytest.LogCaptureFixture, raw: str | None, expected: float, warning: str | None
) -> None:
    environment = {} if raw is None else {"DAYDREAM_PI_RETRY_BASE_DELAY_S": raw}
    with caplog.at_level(logging.WARNING):
        value = _parsed_nonnegative_float(environment, "DAYDREAM_PI_RETRY_BASE_DELAY_S", 10.0)
    assert value == pytest.approx(expected)
    if warning is None:
        assert caplog.text == ""          # the check order is pinned: nan/inf are caught before the sign check
    else:
        assert warning in caplog.text

@pytest.mark.parametrize(
    ("env_value", "expected"), [(None, _PI_DEFAULT_RETRY_ATTEMPTS), ("5", 5),
     ("", _PI_DEFAULT_RETRY_ATTEMPTS), ("-1", _PI_DEFAULT_RETRY_ATTEMPTS)],
    ids=["default", "override", "empty-warns", "negative-warns"],
)
def test_pi_facades_delegate_to_the_shared_parsers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, env_value: str | None, expected: int
) -> None:
    if env_value is not None:
        monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", env_value)
    with caplog.at_level(logging.WARNING):
        assert _pi_retry_attempts() == expected
    if env_value not in (None, "5"):
        assert f"DAYDREAM_PI_RETRY_ATTEMPTS={env_value!r}" in caplog.text   # the warning still names the knob

# Real drivers with only OS process spawning replaced.

async def _drive(
    backend: Any, stdout_lines: list[str], *, exit_code: int = 0, stderr_lines: list[str] | None = None,
) -> tuple[list[Any], FakeCliProcess]:
    """Return events and the fake child. Osprey receives JSONL stdout separately from diagnostic stderr;
    Codex/Pi merge their streams.
    """
    proc = FakeCliProcess(stdout_lines, exit_code=exit_code, stderr_lines=stderr_lines)
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=proc):
        events = [event async for event in backend.execute(Path("/tmp"), "p")]
    return events, proc

def _assert_clean_lifecycle(backend: Any, proc: FakeCliProcess) -> None:
    assert proc.returncode == 0
    assert proc.reaped, "the shared reap must await the child even on a clean exit"
    assert proc.stdin.closed, "the shared teardown must close the stdin pipe"
    assert proc._transport.closed, "the shared teardown must release the pipe fds"
    assert backend._transports == [], "the shared teardown must drop the transport from the backend list"

def _osprey_stream() -> list[str]:
    return [json.dumps(event) for event in osprey_session()]

async def test_codex_process_exit_message_anchor_and_count() -> None:
    with pytest.raises(CodexError) as exc_info:
        await _drive(CodexBackend(model="fixture-model"), DIAG, exit_code=1)
    assert str(exc_info.value) == (
        "Codex CLI exited with return code 1.\nCodex CLI output (last 10 non-JSON lines):\n"
        + "\n".join(DIAG[15:25])  # rolling tail-20 keeps 6..25; the excerpt is the stream's last ten
    )
    assert exc_info.value.category == "PROCESS_EXIT"
    assert not hasattr(exc_info.value, "retryable")  # codex passes no retryable= kwarg

async def test_pi_process_exit_message_anchor_and_count() -> None:
    with pytest.raises(PiError) as exc_info:
        await _drive(PiBackend(model="glm-5.2"), DIAG, exit_code=1)
    assert str(exc_info.value) == (
        "Pi CLI exited with return code 1.\nPi CLI output (last 10 non-JSON lines):\n"
        + "\n".join(DIAG[10:20])  # head-20 capture, last ten of it: positions 11-20 of the stream
    )
    assert exc_info.value.category == "PROCESS_EXIT"
    assert exc_info.value.retryable is False

async def test_pi_process_exit_retryable_for_oom_exit_code() -> None:
    with pytest.raises(PiError) as exc_info:
        await _drive(PiBackend(model="glm-5.2"), DIAG, exit_code=137)
    assert exc_info.value.retryable is True
    assert str(exc_info.value).startswith("Pi CLI exited with return code 137.\n")

async def test_osprey_process_exit_message_anchor_and_count() -> None:
    with pytest.raises(OspreyError) as exc_info:
        await _drive(
            OspreyBackend(OspreyConfig(osprey_binary="fake")),
            _osprey_stream(),  # valid protocol/session_start/…/session_end JSONL
            exit_code=1,
            stderr_lines=DIAG,  # 25 drained stderr lines; the sink caps at 10
        )
    assert str(exc_info.value) == (
        "Osprey CLI exited with return code 1: " + "\n".join(DIAG[:10])
    )
    assert exc_info.value.category == "PROCESS_EXIT"
    assert exc_info.value.retryable is False

async def test_codex_clean_exit_lifecycle() -> None:
    backend = CodexBackend(model="fixture-model")
    _events, proc = await _drive(backend, [json.dumps({"type": "turn.completed", "usage": {}})])
    _assert_clean_lifecycle(backend, proc)

async def test_pi_clean_exit_lifecycle() -> None:
    backend = PiBackend(model="glm-5.2")
    _events, proc = await _drive(backend, [])
    _assert_clean_lifecycle(backend, proc)

async def test_osprey_clean_exit_lifecycle() -> None:
    backend = OspreyBackend(OspreyConfig(osprey_binary="fake"))
    _events, proc = await _drive(backend, _osprey_stream())
    _assert_clean_lifecycle(backend, proc)

# Providers interpret statuses; only the transport may spawn, read, or reap children.

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKENDS_DIR = _REPO_ROOT / "daydream" / "backends"


def _native_process_calls(source: str) -> list[tuple[str, int]]:
    tree = ast.parse(source)
    return [
        (node.func.attr, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and (
            node.func.attr == "create_subprocess_exec"
            or node.func.attr in {"readline", "wait"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"proc", "stdout", "stderr"}
        )
    ]


@pytest.mark.parametrize("source", [
    "asyncio.create_subprocess_exec('cli')",
    "proc.wait()",
    "stdout.readline()",
    "stderr.readline()",
])
def test_process_ownership_guard_detects_native_operations(source: str) -> None:
    assert _native_process_calls(source)


def test_adapters_delegate_native_process_operations_to_transport() -> None:
    adapters = [_BACKENDS_DIR / f"{name}.py" for name in ("codex", "pi", "osprey")]
    assert all(path.is_file() for path in adapters)
    assert not {
        path.name: calls for path in adapters
        if (calls := _native_process_calls(path.read_text(encoding="utf-8")))
    }
