"""Focused tests for run-local interaction and backend lifecycle state."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import Any

import pytest

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




def test_resolve_run_context_creates_a_fresh_standalone_default() -> None:
    first = resolve_run_context()
    second = resolve_run_context()

    assert first is not second
    assert first.policy == InteractionPolicy()
    assert second.policy == InteractionPolicy()



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

def test_backend_registration_is_identity_aware_and_reference_counted() -> None:
    backend = ScriptedBackend()
    sibling = ScriptedBackend()
    first = RunContext(InteractionPolicy())
    second = RunContext(InteractionPolicy())
    with first.backend_registration(backend):
        with first.backend_registration(backend):
            with second.backend_registration(backend):
                with second.backend_registration(sibling):
                    assert first.active_backends() == (backend,)
                    assert second.active_backends() == (backend, sibling)
                    assert active_backends() == (backend, sibling)
            assert first.active_backends() == (backend,)
            assert active_backends() == (backend,)
        assert active_backends() == (backend,)
    assert first.active_backends() == ()
    assert active_backends() == ()

def test_backend_registration_does_not_collapse_equal_distinct_objects() -> None:
    first_backend = _EqualBackend()
    second_backend = _EqualBackend()
    context = RunContext(InteractionPolicy())
    with context.backend_registration(first_backend), context.backend_registration(second_backend):
        local = context.active_backends()
        process = active_backends()
        assert len(local) == len(process) == 2
        assert local[0] is process[0] is first_backend
        assert local[1] is process[1] is second_backend
    assert active_backends() == ()


def test_signal_snapshot_reenters_registration_and_interrupted_enter_cleans_up() -> None:
    """A same-thread signal snapshot must neither deadlock nor leak membership."""
    script = textwrap.dedent(
        """
        import signal

        import daydream.run_context as runtime

        backend = object()
        context = runtime.RunContext(runtime.InteractionPolicy())

        def interrupt(_signum, _frame):
            assert runtime.active_backends() == (backend,)
            raise KeyboardInterrupt

        class InterruptingRegistry(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                signal.raise_signal(signal.SIGINT)

        signal.signal(signal.SIGINT, interrupt)
        runtime._active_registrations = InterruptingRegistry()
        try:
            with context.backend_registration(backend):
                raise AssertionError("registration enter should have been interrupted")
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("signal did not interrupt registration enter")

        assert context.active_backends() == ()
        assert runtime.active_backends() == ()
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=5,)
