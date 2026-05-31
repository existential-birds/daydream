"""Real-path integration tests for ``daydream.cli.main`` exit-code propagation.

``cli.main()`` is the true process entrypoint: it installs signal handlers,
routes subcommands, parses argv via ``_parse_args``, then drives
``anyio.run(run, config)`` and finally ``sys.exit(exit_code)``. Until now only
``_parse_args`` was unit-tested; ``main()`` itself — including its
``anyio.run`` ownership of the event loop and its dedicated except clauses —
had zero coverage.

These tests INVOKE ``cli.main()`` for real and assert the PROCESS EXIT CODE:

  * a clean default deep run -> ``0`` (the code returned by ``runner.run``
    must flow through ``anyio.run`` -> ``sys.exit``), and
  * the ``WrongBranchError`` guard -> ``1`` (exercising ``main()``'s dedicated
    ``except git_ops.WrongBranchError`` clause).

They are deliberately SYNC ``def test_...`` functions, NOT ``async def``.
``cli.main()`` calls ``anyio.run(...)``, which starts its own event loop. Under
``asyncio_mode = "auto"`` an ``async def`` test already runs inside a running
loop, so calling ``anyio.run`` from there raises "Already running asyncio in
this thread". A plain sync test lets ``anyio.run`` own the loop — which is the
exact production code path we need to cover.

Only the external seams are mocked: the network/SDK ``Backend`` (via
``create_backend``), the ``gh``-shelling detection helpers, and interactive
UI prompts/heroes. ``run``, ``_dispatch``, ``run_deep`` and every ``phase_*``
run for real against a real temp git worktree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from daydream import cli

# Reuse the deep-pipeline stub from the exemplar instead of duplicating it.
from tests.test_deep_orchestrator import (
    _install_stub_backend,
    _silence,
)


def _silence_cli_and_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock only the external seams cli.main touches before/around the loop.

    - gh-detection helpers (`_auto_detect_pr_number`/`_detect_repo_slug`) shell
      out to ``gh`` against ``Path.cwd()``; pin them to ``None`` for
      determinism (no network, no dependency on the host's git checkout).
    - ``runner.print_phase_hero`` renders the DAYDREAM banner on the real run
      path; silence it so the test output stays clean. (It does not block, but
      patching keeps the captured output focused.)
    - signal-handler install is a no-op concern here; leave it real — it is a
      cheap, side-effect-free part of the production entrypoint we want covered.
    """
    monkeypatch.setattr("daydream.cli._auto_detect_pr_number", lambda: None)
    monkeypatch.setattr("daydream.cli._detect_repo_slug", lambda: None)
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)


def test_cli_main_clean_deep_run_exits_0(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean default deep run drives cli.main -> anyio.run -> sys.exit(0).

    ``multi_stack_target`` is a real git repo checked out on ``feature`` with a
    real cross-stack diff. Driving ``sys.argv`` with just the target positional
    exercises the production default (deep multi-stack) pipeline end to end. The
    only mocks are the Backend (stub) and the gh/UI seams — ``run``,
    ``_dispatch``, ``run_deep`` and the ``phase_*`` functions all run for real.
    """
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    monkeypatch.setattr(sys, "argv", ["daydream", str(multi_stack_target)])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    # SystemExit.code is the int returned by runner.run, propagated through
    # anyio.run -> sys.exit. Not a hardcoded 0 (see TDD proof in the PR).
    assert exc.value.code == 0


def test_cli_main_wrong_branch_exits_1(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The WrongBranch guard drives cli.main's dedicated except clause -> exit 1.

    ``git_repo`` is a real repo checked out on ``main`` (the base branch) with a
    single commit and no feature branch. Running ``daydream <repo>`` with no
    ``--branch``/``--worktree`` hits the ``_dispatch`` guard, which raises
    ``git_ops.WrongBranchError``; ``runner.run`` re-raises it and
    ``cli.main``'s ``except git_ops.WrongBranchError`` clause calls
    ``sys.exit(1)``. The stub backend is installed but never reached.
    """
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    _install_stub_backend(monkeypatch, git_repo)
    # The error path renders a panel via print_error; silence it.
    monkeypatch.setattr("daydream.cli.print_error", lambda *a, **kw: None)

    monkeypatch.setattr(sys, "argv", ["daydream", str(git_repo)])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
