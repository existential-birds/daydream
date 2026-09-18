"""#1221: one shared reap / exit-check / teardown across the CLI backends."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from daydream.backends._transport import (
    PROCESS_EXIT_EXCERPT_MAX_LINES,
    CliTransport,
    process_exit_message,
    raise_for_exit,
    reap,
    teardown,
)
from daydream.backends.codex import CodexBackend, CodexError
from daydream.backends.osprey import OspreyBackend, OspreyError
from daydream.backends.pi import PiBackend, PiError
from tests.harness.fake_cli_process import FakeCliProcess

DIAG = [f"diag-{i:02d}" for i in range(1, 26)]  # 25 > every capture window (codex/pi 20, osprey 10)


def _transport(monkeypatch: pytest.MonkeyPatch, proc: FakeCliProcess) -> CliTransport:
    """A real CliTransport whose child is *proc* (the transport seam is patched, not the transport)."""
    monkeypatch.setattr(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    )
    return CliTransport("codex", ["codex"], limit=1024)


class _Recorder(Exception):
    """Captures the kwargs the shared raise path used to construct the adapter error."""

    def __init__(self, message: str, **kwargs: object) -> None:
        super().__init__(message)
        self.kwargs = kwargs


@pytest.mark.parametrize(("returncode", "expected"), [(0, 0), (1, 1), (-15, -15)])
async def test_reap_suppresses_transport_exit_and_returns_the_code(
    monkeypatch: pytest.MonkeyPatch, returncode: int, expected: int
) -> None:
    proc = FakeCliProcess([], exit_code=returncode)
    transport = _transport(monkeypatch, proc)
    await transport.start()

    assert await reap(transport) == expected  # a non-zero exit must not escape as TransportExitError
    assert proc.reaped


def test_raise_for_exit_kwarg_shape_is_per_adapter() -> None:
    built: list[dict[str, object]] = []

    def build_message(returncode: int) -> str:
        return f"exited {returncode}"

    raise_for_exit(0, error_type=_Recorder, category="PROCESS_EXIT", build_message=build_message)
    assert built == []  # nothing raised, no error constructed
    for kwargs, retryable in (({}, None), ({"retryable": True}, True)):
        with pytest.raises(_Recorder, match="exited 1") as exc_info:
            raise_for_exit(
                1, error_type=_Recorder, category="PROCESS_EXIT",
                build_message=build_message, retryable=retryable,
            )
        built.append(exc_info.value.kwargs)
    # codex/osprey pass no retryable= kwarg; pi's is forwarded verbatim.
    assert built == [{"category": "PROCESS_EXIT"}, {"category": "PROCESS_EXIT", "retryable": True}]


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


# ---------------------------------------------------------------------------
# Per-adapter lifecycle (requirement 13): drive the real backends through the
# transport seam (only the OS fork is replaced) and pin the observable outcome.
# ---------------------------------------------------------------------------


async def _drive(
    backend: Any,
    stdout_lines: list[str],
    *,
    exit_code: int = 0,
    stderr_lines: list[str] | None = None,
) -> tuple[list[Any], FakeCliProcess]:
    """Drive *backend* with a fake child; return the emitted events and the child.

    stdout_lines are raw lines (the adapters decode/parse them); osprey callers
    pass JSON-encoded events in stdout_lines and its diagnostics in stderr_lines,
    because osprey is the only adapter on StderrPolicy.DRAIN_TASK.
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


_OSPREY_SESSION: list[dict[str, object]] = [
    {"event": "protocol", "version": 2},
    {
        "event": "session_start",
        "session_id": "s-137",
        "started_at": "2026-08-15T00:00:00Z",
        "model": "custom-model",
        "provider": "openai-compatible",
    },
    {
        "event": "session_end",
        "total_turns": 1,
        "session_wallclock_ms": 15,
        "total_cost_usd": None,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cached_tokens": None,
        "total_cache_write_tokens": None,
        "total_thinking_tokens": 0,
        "total_oom_kills": 0,
        "p50_turn_ms": 15,
        "p99_turn_ms": 15,
        "avg_turn_cost_usd": None,
        "structured_output": None,
        "outcome": "completed",
        "verification": None,
        "exit_code": 0,
    },
]


def _osprey_stream() -> list[str]:
    """The minimal valid session, as the raw JSON lines ``_drive`` feeds osprey."""
    return [json.dumps(event) for event in _OSPREY_SESSION]


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
            OspreyBackend(osprey_binary="fake"),
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
    _events, proc = await _drive(
        backend,
        [json.dumps({"type": "turn.completed", "usage": {}})],
    )
    _assert_clean_lifecycle(backend, proc)


async def test_pi_clean_exit_lifecycle() -> None:
    backend = PiBackend(model="glm-5.2")
    _events, proc = await _drive(backend, [])
    _assert_clean_lifecycle(backend, proc)


async def test_osprey_clean_exit_lifecycle() -> None:
    backend = OspreyBackend(osprey_binary="fake")
    _events, proc = await _drive(backend, _osprey_stream())
    _assert_clean_lifecycle(backend, proc)


# ---------------------------------------------------------------------------
# Structural guard: the reap / raise / teardown sequence lives once, in the
# owner module. The detectors are liveness-proven (each is asserted to fire on
# a synthetic offender) so an empty scan can never pass because it is broken.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKENDS_DIR = _REPO_ROOT / "daydream" / "backends"
_OWNER = "daydream/backends/_transport.py"
_OWNER_SURFACE = frozenset({"reap", "raise_for_exit", "teardown"})


def _forbidden_sites(source: str) -> list[tuple[str, int]]:
    """(label, lineno) for every reap/raise shape that must live only in the owner."""
    tree = ast.parse(source)
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
            func = node.value.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "wait"
                and isinstance(func.value, ast.Name)
                and func.value.id == "transport"
            ):
                found.append(("awaits transport.wait()", node.lineno))
        if (
            isinstance(node, ast.ExceptHandler)
            and isinstance(node.type, ast.Name)
            and node.type.id == "TransportExitError"
        ):
            found.append(("except TransportExitError", node.lineno))
        if (
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and any(
                kw.arg == "category"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "PROCESS_EXIT"
                for kw in node.exc.keywords
            )
        ):
            found.append(('raise with category="PROCESS_EXIT"', node.lineno))
    return found


@pytest.mark.parametrize(
    ("source", "label"),
    [
        ("async def f(transport):\n    await transport.wait()\n", "awaits transport.wait()"),
        ("try:\n    pass\nexcept TransportExitError:\n    pass\n", "except TransportExitError"),
        ('raise AdapterError("x", category="PROCESS_EXIT")\n', 'raise with category="PROCESS_EXIT"'),
    ],
)
def test_the_guard_detects_each_forbidden_shape(source: str, label: str) -> None:
    """Liveness: an empty scan can never pass silently — each detector is proven to fire."""
    assert label in [found for found, _line in _forbidden_sites(source)]


def test_no_adapter_owns_the_reap_or_the_process_exit_raise() -> None:
    offenders: dict[str, list[tuple[str, int]]] = {}
    modules = sorted(_BACKENDS_DIR.glob("*.py"))
    assert modules, "the scan found no production modules"  # liveness: mis-rooted scan fails
    for path in modules:
        relative = path.relative_to(_REPO_ROOT).as_posix()
        if relative == _OWNER:
            continue
        sites = _forbidden_sites(path.read_text(encoding="utf-8"))
        if sites:
            offenders[relative] = sites
    assert not offenders, f"the reap / PROCESS_EXIT raise must live in {_OWNER}: {offenders}"


def test_the_owner_still_carries_the_shared_surface() -> None:
    """Liveness: the exemption is a property of the owner's contents, not of its filename."""
    tree = ast.parse((_REPO_ROOT / _OWNER).read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert _OWNER_SURFACE <= defined, f"{_OWNER} no longer defines {sorted(_OWNER_SURFACE - defined)}"
    assert any(
        isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "TransportExitError"
        for node in ast.walk(tree)
    ), "the suppression handler must live in _OWNER"
