"""Focused tests for run-local interaction and backend lifecycle state."""

from __future__ import annotations

import signal
from typing import Any

import pytest

from daydream import run_context
from daydream.run_context import (
    InteractionPolicy,
    RunContext,
    active_backends,
    resolve_run_context,
)
from tests.harness.backend import ScriptedBackend


class _EqualBackend(ScriptedBackend):
    """Unhashable backend that compares equal to every sibling instance."""

    def __eq__(self, _other: object) -> bool:
        return True


def test_backend_registration_is_identity_aware_and_reference_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _EqualBackend()
    sibling = _EqualBackend()
    first = resolve_run_context()
    second = resolve_run_context()
    assert first is not second
    assert first.policy == InteractionPolicy()
    assert second.policy == InteractionPolicy()
    with first.backend_registration(backend):
        with first.backend_registration(backend):
            with second.backend_registration(backend):
                with second.backend_registration(sibling):
                    assert first.active_backends() == (backend,)
                    assert first.active_backends()[0] is backend
                    local = second.active_backends()
                    process = active_backends()
                    assert len(local) == len(process) == 2
                    assert local[0] is process[0] is backend
                    assert local[1] is process[1] is sibling
            assert first.active_backends() == (backend,)
            assert first.active_backends()[0] is backend
            assert active_backends() == (backend,)
            assert active_backends()[0] is backend
        assert active_backends() == (backend,)
        assert active_backends()[0] is backend
    assert first.active_backends() == ()
    assert active_backends() == ()
    # A same-thread signal snapshot must neither deadlock nor leak membership.
    interrupted = ScriptedBackend()

    def interrupt(_signum: int, _frame: object) -> None:
        assert run_context.active_backends() == (interrupted,)
        raise KeyboardInterrupt

    class InterruptingRegistry(dict[object, Any]):
        def __setitem__(self, key: object, value: Any) -> None:
            super().__setitem__(key, value)
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(run_context, "_active_registrations", InterruptingRegistry())
    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        with pytest.raises(KeyboardInterrupt), first.backend_registration(interrupted):
            raise AssertionError("registration enter should have been interrupted")
    finally:
        signal.signal(signal.SIGINT, previous)
    assert first.active_backends() == ()
    assert active_backends() == ()


async def test_overlapping_runs_keep_prompt_and_console_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interactive sibling task cannot enable stdin or disable log redaction for a logged run."""
    import anyio

    from daydream.agent import console
    from daydream.run_context import bind_run_context, current_run_context

    sentinel = "ghp_" + "x" * 16
    input_calls: list[str] = []

    def answer() -> str:
        input_calls.append("read")
        return "interactive-answer"

    monkeypatch.setattr("builtins.input", answer)
    logged = RunContext(InteractionPolicy(log_mode=True, quiet=True, interactive=False))
    plain = RunContext(InteractionPolicy(assume="yes"))
    choices: dict[str, str] = {}
    logged_bound, plain_bound = anyio.Event(), anyio.Event()

    async def run(label: str, context: RunContext, mine: anyio.Event, sibling: anyio.Event) -> None:
        with bind_run_context(context):
            mine.set()
            await sibling.wait()  # Both runs are bound while each one prompts and prints.
            choices[label] = resolve_run_context().choice(
                f"{label} choice", default="safe-default", safe_default="safe-default", console=console)
            console.print(f"{label}-token={sentinel}", markup=False, highlight=False)

    with console.capture() as captured, anyio.fail_after(5):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run, "logged", logged, logged_bound, plain_bound)
            tasks.start_soon(run, "plain", plain, plain_bound, logged_bound)

    assert choices == {"logged": "safe-default", "plain": "interactive-answer"}
    assert input_calls == ["read"]
    output = captured.get()
    assert f"logged-token={sentinel}" not in output and "logged-token=[REDACTED_API_KEY]" in output
    assert f"plain-token={sentinel}" in output
    assert current_run_context() is None
