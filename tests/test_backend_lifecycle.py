"""#1221: one shared reap / exit-check / teardown across the CLI backends."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from daydream.backends._transport import (
    PROCESS_EXIT_EXCERPT_MAX_LINES,
    CliTransport,
    process_exit_message,
    raise_for_exit,
    reap,
    teardown,
)
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
