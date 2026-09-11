"""Focused tests for run-local interaction and backend lifecycle state."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from daydream.run_context import (
    InteractionPolicy,
    RunContext,
    active_backends,
    bind_run_context,
    current_run_context,
    resolve_gate,
    resolve_run_context,
)
from tests.harness.backend import ScriptedBackend


class _EqualBackend(ScriptedBackend):
    """Unhashable backend that compares equal to every sibling instance."""

    def __eq__(self, _other: object) -> bool:
        return True


def test_interaction_policy_is_frozen_and_keyword_only() -> None:
    policy = InteractionPolicy(assume="yes", interactive=False, quiet=True, log_mode=True)

    with pytest.raises(FrozenInstanceError):
        policy.quiet = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        InteractionPolicy("yes")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("assume", "interactive", "safe_default", "expected"),
    [
        ("yes", True, False, True),
        ("no", True, True, False),
        (None, False, True, True),
        (None, True, False, None),
    ],
)
def test_resolve_gate_remains_pure(
    assume: str | None, interactive: bool, safe_default: bool, expected: bool | None
) -> None:
    assert resolve_gate(assume=assume, interactive=interactive, safe_default=safe_default) is expected


def test_nested_binding_restores_the_previous_context() -> None:
    outer = RunContext(InteractionPolicy(quiet=True))
    inner = RunContext(InteractionPolicy(log_mode=True))

    assert current_run_context() is None
    with bind_run_context(outer):
        assert current_run_context() is outer
        assert resolve_run_context() is outer
        with bind_run_context(inner):
            assert current_run_context() is inner
            assert resolve_run_context(outer) is outer
        assert current_run_context() is outer
    assert current_run_context() is None


def test_resolve_run_context_creates_a_fresh_standalone_default() -> None:
    first = resolve_run_context()
    second = resolve_run_context()

    assert first is not second
    assert first.policy == InteractionPolicy()
    assert second.policy == InteractionPolicy()


def test_confirm_uses_forced_and_unattended_answers_without_prompting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[tuple[Any, str, str]] = []

    def prompt(console: Any, message: str, default: str) -> str:
        prompts.append((console, message, default))
        return "y"

    monkeypatch.setattr("daydream.run_context._prompt_user", prompt)

    assert RunContext(InteractionPolicy(assume="yes")).confirm("continue?", safe_default=False)
    assert not RunContext(InteractionPolicy(assume="no")).confirm("continue?", safe_default=True)
    assert RunContext(InteractionPolicy(interactive=False)).confirm("continue?", safe_default=True)
    assert prompts == []


def test_confirm_prompts_through_the_single_gateway_with_its_context_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(InteractionPolicy(log_mode=True))
    observed: list[RunContext | None] = []

    def prompt(_console: Any, _message: str, _default: str) -> str:
        observed.append(current_run_context())
        return "YES"

    monkeypatch.setattr("daydream.run_context._prompt_user", prompt)

    assert context.confirm("continue?", safe_default=False, default="n")
    assert observed == [context]
    assert current_run_context() is None


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


def test_backend_registration_cleans_up_after_exception() -> None:
    backend = ScriptedBackend()
    context = RunContext(InteractionPolicy())

    with pytest.raises(RuntimeError, match="boom"):
        with context.backend_registration(backend):
            assert active_backends() == (backend,)
            raise RuntimeError("boom")

    assert context.active_backends() == ()
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

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
