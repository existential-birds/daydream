"""Real-path integration test for the ``feedback`` SUBCOMMAND argv -> body.

Where ``test_pr_feedback_integration.py`` enters at ``run_feedback`` with a
hand-built ``RunConfig``, this test enters one layer earlier: it drives raw
argv through the real parser (``daydream.cli._parse_args`` feedback branch ->
``_build_feedback_config``) and then into the REAL five-phase body via
``runner.run_feedback``. Nothing in the routing/dispatch/body path is stubbed —
only the external SDK backend seam (``daydream.runner.create_backend``) and the
interactive UI prompts. This proves subcommand parsing populates bot/pr/target
AND that those parsed values wire all the way through dispatch into the real
PR-feedback body, together.

The mock backend and observable assertions are shared with Task 1 by importing
its prompt-dispatching stub — there is exactly one stub, no parallel copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_pr_feedback_integration import (
    BOT,
    FIX_MARKER,
    PR_NUMBER,
    _git,
    _PRFeedbackStubBackend,
    _silence,
)


async def test_feedback_subcommand_argv_to_body(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-path: argv -> _parse_args(feedback) -> run_feedback -> full body.

    Drives raw argv through the real feedback subparser into the real body.
    Mocks ONLY the Backend and silences UI prompts. Asserts the parse routed
    bot/pr/target correctly AND the same observable body outcomes as Task 1:
    exit 0, the fix marker on disk, a new commit carrying the Daydream-Run
    trailer, and respond-pr-feedback invoked with the right pr/bot.
    """
    from daydream import cli
    from daydream.runner import run_feedback

    # 1. Parse raw argv through the REAL feedback subparser.
    config = cli._parse_args(
        ["feedback", str(PR_NUMBER), "--bot", BOT, str(multi_stack_target)]
    )

    # 2. Assert the subcommand parse routed correctly into RunConfig.
    assert config.bot == BOT
    assert config.pr_number == PR_NUMBER
    assert config.target == str(multi_stack_target)

    # 3. Patch ONLY the Backend seam + silence interactive prompts.
    _silence(monkeypatch)
    stub = _PRFeedbackStubBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: stub)

    head_before = _git(multi_stack_target, "rev-parse", "HEAD")

    # 4. Call the REAL body — no _run_pr_feedback / _dispatch / phase stub.
    exit_code = await run_feedback(config, config.pr_number)

    # 5. Observable outcomes (same as Task 1).
    assert exit_code == 0

    api_text = (multi_stack_target / "api.py").read_text()
    assert FIX_MARKER.strip() in api_text

    head_after = _git(multi_stack_target, "rev-parse", "HEAD")
    assert head_after != head_before
    commit_msg = _git(multi_stack_target, "log", "-1", "--format=%B")
    assert "Daydream-Run:" in commit_msg

    assert len(stub.respond_calls) == 1
    respond_prompt = stub.respond_calls[0]
    assert f"--pr {PR_NUMBER}" in respond_prompt
    assert f"--bot {BOT}" in respond_prompt


async def test_feedback_subcommand_non_interactive_argv_to_body(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-path: ``daydream feedback ... --non-interactive`` parses AND runs.

    Regression guard for the gap where ``--non-interactive`` lived only on the
    main parser: the feedback subparser rejected it (``unrecognized arguments``)
    and ``_build_feedback_config`` never set ``non_interactive``. This drives raw
    argv WITH the flag through the real feedback subparser into the real body.

    Crucially it does NOT silence prompts — instead ``builtins.input`` is patched
    to raise on ANY call, so a stray stdin read anywhere in the unattended
    feedback path fails the test loudly rather than being masked.
    """
    from daydream import cli
    from daydream.agent import get_non_interactive, reset_state
    from daydream.runner import run_feedback

    # 1. Parse argv WITH --non-interactive through the REAL feedback subparser.
    #    Pre-fix this raised SystemExit (unrecognized argument).
    config = cli._parse_args(
        ["feedback", str(PR_NUMBER), "--bot", BOT, "--non-interactive", str(multi_stack_target)]
    )

    # 2. The subparser accepted the flag AND threaded it into RunConfig.
    assert config.non_interactive is True
    assert config.bot == BOT
    assert config.pr_number == PR_NUMBER

    # 3. Patch ONLY the Backend seam. No _silence — stdin must never be touched.
    stub = _PRFeedbackStubBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: stub)

    def _forbidden_input(*_a: object, **_kw: object) -> str:
        raise AssertionError("input() was called -- feedback path must not touch stdin")

    monkeypatch.setattr("builtins.input", _forbidden_input)

    head_before = _git(multi_stack_target, "rev-parse", "HEAD")

    # 4. Call the REAL body. run() applies set_non_interactive(config.non_interactive).
    #    reset_state() in finally so the global non_interactive flag never bleeds
    #    into a later test (run() sets but does not reset module state).
    try:
        exit_code = await run_feedback(config, config.pr_number)
        # 5. Observable outcomes: the flag was honored end-to-end and the body ran.
        assert exit_code == 0
        assert get_non_interactive() is True, "set_non_interactive was not applied from the feedback config"
    finally:
        reset_state()

    api_text = (multi_stack_target / "api.py").read_text()
    assert FIX_MARKER.strip() in api_text

    head_after = _git(multi_stack_target, "rev-parse", "HEAD")
    assert head_after != head_before
    commit_msg = _git(multi_stack_target, "log", "-1", "--format=%B")
    assert "Daydream-Run:" in commit_msg
