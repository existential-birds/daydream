"""Fix Gate Cleanup And Precision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.config import REVIEW_OUTPUT_FILE
from daydream.deep import fix_steps
from daydream.git_ops import GitTimeoutError
from tests.test_deep_orchestrator import (
    _force_interactive,
    _install_stub_backend,
    _run_deep,
    _silence,
)


def test_closed_stdin_resolves_a_gate_prompt_to_its_safe_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """EOF at an interactive gate (closed stdin) is caught and resolves to the default, never a crash."""
    from io import StringIO

    from rich.console import Console

    from daydream.ui.messages import _read_user_input

    answers: list[Any] = [EOFError("simulated closed stdin"), "", "y"]

    def scripted_input(*_a: Any, **_kw: Any) -> str:
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return str(answer)

    monkeypatch.setattr("builtins.input", scripted_input)
    output = StringIO()
    console = Console(file=output, width=200)
    assert [_read_user_input(console, "Apply fixes?", "n") for _ in range(3)] == ["n", "n", "y"]
    assert "stdin closed (EOF)" in output.getvalue()


async def test_cleanup_removes_the_review_report_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit cleanup flags win; otherwise the run policy decides, and unattended runs keep the report."""
    from dataclasses import replace

    from daydream.deep.fix_steps import _perform_cleanup
    from daydream.run_context import InteractionPolicy, RunContext
    from tests.deep_orchestrator.support import _base_repo, _direct_fix_context

    ctx = _direct_fix_context(_base_repo(tmp_path, "cleanup"), [], changed_files=set())
    report = ctx.work.repo / REVIEW_OUTPUT_FILE
    for cleanup, assume, kept in ((False, "yes", True), (None, None, True), (None, "yes", False), (True, None, False)):
        report.write_text("review\n")
        ctx.config = replace(ctx.config, cleanup=cleanup)
        ctx.run_context = RunContext(InteractionPolicy(assume=assume, interactive=False))
        await _perform_cleanup(ctx)
        assert report.exists() is kept, (cleanup, assume)


async def test_deep_run_reports_persistent_git_timeout_distinctly(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)


    def always_timeout(*args: Any, **kwargs: Any) -> str:
        raise GitTimeoutError("git diff main...HEAD timed out after 30s (3 attempts)")

    monkeypatch.setattr("daydream.git_ops.diff", always_timeout)

    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "daydream.deep.orchestrator.print_error", lambda console, title, msg, *a, **kw: errors.append((title, msg)),
    )

    exit_code = await _run_deep(multi_stack_target)

    # Aborts with the timeout-specific title, NOT the misleading base-branch error.
    assert exit_code == 1
    titles = [t for t, _ in errors]
    assert "Git Timeout" in titles, f"expected a distinct Git Timeout error, got {errors!r}"
    assert "Git Error" not in titles, f"a timeout was misreported as the generic base-branch error: {errors!r}"

def test_verdict_buckets_reject_unchecked_or_missing_selection() -> None:
    cases: tuple[Any, ...] = (None, [], [{}], [{"item_uid": "item:4", "selected": 1, "reason_code": "selected"}],
                              [{"item_uid": "item:4", "selected": True, "reason_code": "selected"}] * 2)
    for decisions in cases:
        payload = {"selection": {"decisions": decisions}}
        with pytest.raises(ValueError):
            fix_steps._verdict_buckets([{"id": 4, "item_uid": "item:4", "lens": "structural"}], payload)
