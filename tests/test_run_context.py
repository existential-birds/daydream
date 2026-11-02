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



def test_choice_forces_only_supplied_assume_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []
    def prompt(_console: Any, message: str, _default: str) -> str:
        prompts.append(message)
        return "typed"
    monkeypatch.setattr("daydream.run_context._prompt_user", prompt)
    assumed = RunContext(InteractionPolicy(assume="yes"))
    assert assumed.choice("menu", default="3", safe_default="4", assume_yes="2") == "2"
    assert assumed.choice("free form", default=".", safe_default=".") == "typed"
    unattended = RunContext(InteractionPolicy(assume="yes", interactive=False))
    assert unattended.choice("free form", default=".", safe_default="safe") == "safe"
    assert prompts == ["free form"]

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
