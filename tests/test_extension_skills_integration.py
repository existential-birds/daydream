"""Real-path tests: extension skill slots reach the pr-feedback, shallow, and deep flows.

Drives the production entrypoints (``runner.run_feedback`` / ``runner.run``)
against a real temp git repo, mocking ONLY the backend seam
(``daydream.runner.create_backend``) per the testing standard — the same shape
as ``tests/test_pr_feedback_integration.py``. A ``daydream_ext`` package
written by the ``ext_dir`` fixture overrides skill slots; assertions are on
the prompts the backend actually received and the exit code.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream import runner
from daydream.backends import ResultEvent, TextEvent
from daydream.runner import RunConfig
from tests.conftest import ExtDir


class RecordingBackend:
    """Prompt-recording stub modelled on ``_PRFeedbackStubBackend``.

    Dispatches on prompt content just enough to drive each flow to a clean
    exit: writes the review-output file for review/fetch prompts, returns an
    empty issues list for parse prompts, and reports passing tests.
    """

    model = "mock-model"
    fanout_concurrency = 4

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        self.prompts.append(prompt)
        pl = prompt.lower()

        if "fetch-pr-feedback" in pl or "review the changes" in pl:
            (cwd / ".review-output.md").write_text("# Review\n\nNo issues found.\n")
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        if "extract only actionable issues" in pl:
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"issues": []}, continuation=None)
            return

        if "test suite" in pl:
            yield TextEvent(text="All tests passed")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        yield TextEvent(text="")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        # Mirror ClaudeBackend: append args so the test can read the slot from the prompt.
        result = f"/{skill_key}"
        if args:
            result = f"{result} {args}"
        return result


async def test_fork_overrides_pr_feedback_fetch_skill(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daydream_ext override of the pr-feedback-fetch slot reaches the fetch prompt.

    Observable outcomes: exit 0, the overridden skill string appears in a
    prompt the backend received, and the built-in literal appears in none.
    """
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 1\n"
        "def register(r):\n"
        "    r.override_skill('pr-feedback-fetch', 'ro-core:fetch-pr-feedback')\n"
    )
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    rc = await runner.run_feedback(
        RunConfig(target=str(multi_stack_target), bot="x[bot]", non_interactive=True), pr=1
    )

    assert rc == 0
    assert any("ro-core:fetch-pr-feedback" in p for p in backend.prompts)
    assert not any("beagle-core:fetch-pr-feedback" in p for p in backend.prompts)


async def test_stack_slot_override_reaches_shallow_skill_flag(
    ext_dir: ExtDir, feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--skill python`` resolves through the ``stack:python`` slot, not SKILL_MAP.

    With a daydream_ext override of ``stack:python``, the shallow flow's
    review prompt must carry the overridden invocation and never the built-in
    ``beagle-python:review-python`` literal.
    """
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 1\n"
        "def register(r):\n"
        "    r.override_skill('stack:python', 'ro-python:review-python')\n"
    )
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    config = RunConfig(
        target=str(feature_branch_repo),
        shallow=True,
        skill="python",
        non_interactive=True,
        cleanup=False,
    )
    rc = await runner.run(config)

    assert rc == 0
    assert any("ro-python:review-python" in p for p in backend.prompts)
    assert not any("beagle-python:review-python" in p for p in backend.prompts)


async def test_fork_stack_rule_routes_deep_per_stack_review(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daydream_ext ``add_stack(StackRule(...))`` reaches the deep per-stack review.

    Commits a ``.proto`` file into the multi-stack diff so ``detect_stacks``
    (deep flow only) sees a file matching the fork glob, then drives the full
    deep pipeline through ``runner.run``. Observable outcomes: exit 0 and a
    per-stack review prompt carrying the fork skill invocation AND the routed
    ``.proto`` file reached the backend.
    """
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    ext_dir.write_module(
        "from daydream.extensions import StackRule\n"
        "DAYDREAM_EXT_API = 1\n"
        "def register(r):\n"
        "    r.add_stack(StackRule('proto', ('*.proto',), 'ro-proto:review-proto'))\n"
    )
    (multi_stack_target / "api.proto").write_text('syntax = "proto3";\n')
    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "add", "."], cwd=multi_stack_target, capture_output=True, check=True
    )
    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "commit", "-m", "add proto"], cwd=multi_stack_target, capture_output=True, check=True
    )
    backend = _install_stub_backend(monkeypatch, multi_stack_target)
    _silence(monkeypatch)

    # The PR post runs before the fix gate; stub the non-idempotent GitHub write.
    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )

    assert rc == 0
    proto_prompts = [c["prompt"] for c in backend.calls if "/ro-proto:review-proto" in c["prompt"]]
    assert proto_prompts and "api.proto" in proto_prompts[0]


async def test_phase_review_slot_supplies_shallow_skill(
    ext_dir: ExtDir, feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound phase:review slot replaces the non-interactive "Missing --skill" error.

    Runs the shallow flow through ``runner.run`` non-interactively WITHOUT
    ``--skill`` — today that config exits 1 with "Missing --skill". With the
    slot bound, the run must exit 0 and the review prompt must carry the
    bound skill invocation.
    """
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 1\n"
        "def register(r):\n"
        "    r.override_skill('phase:review', 'ro-python:review-python')\n"
    )
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    config = RunConfig(
        target=str(feature_branch_repo),
        shallow=True,
        non_interactive=True,
        cleanup=False,
    )
    rc = await runner.run(config)

    assert rc == 0
    assert any("ro-python:review-python" in p for p in backend.prompts)
