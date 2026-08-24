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
from collections.abc import Callable
from pathlib import Path

import pytest

from daydream import runner
from daydream.backends import ResultEvent, TextEvent
from daydream.runner import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend

_CLEAN_TURN = (
    TextEvent(text=""),
    ResultEvent(structured_output={"issues": []}, continuation=None),
)


async def test_fork_overrides_pr_feedback_fetch_skill(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
) -> None:
    """A daydream_ext override of the pr-feedback-fetch slot reaches the fetch prompt.

    Observable outcomes: exit 0, the overridden skill string appears in a
    prompt the backend received, and the built-in literal appears in none.
    """
    ext_dir.write_module(
        "def register(r):\n"
        "    r.override_skill('pr-feedback-fetch', 'ro-core:fetch-pr-feedback')\n"
    )
    backend = ScriptedBackend(events=_CLEAN_TURN)
    install_backend(backend)

    rc = await runner.run_feedback(make_config(multi_stack_target, bot="x[bot]"), pr=1)

    assert rc == 0
    assert any("ro-core:fetch-pr-feedback" in p for p in backend.prompts)
    assert not any("beagle-core:fetch-pr-feedback" in p for p in backend.prompts)


async def test_shallow_stack_uses_native_profile_strategy_no_skill(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    mute_side_effects: Callable[..., None],
) -> None:
    """``--stack python`` renders the native per-stack strategy with no skill token.

    Built-in stacks have no ``stack:*`` skill slot (M1/M2); the shallow flow's
    review prompt must carry the profile-owned ``discovery.per_stack`` strategy
    and never a Beagle/skill invocation.
    """
    from daydream import review_profile as rp

    backend = ScriptedBackend(events=_CLEAN_TURN)
    install_backend(backend)
    mute_side_effects("daydream.deep.orchestrator")

    rc = await runner.run(make_config(feature_branch_repo, shallow=True, stack="python"))

    assert rc == 0
    strategy = rp.build_default_profile().strategies["discovery.per_stack"].content
    assert any(strategy in p for p in backend.prompts)
    assert not any("/beagle-" in p or "/ro-" in p or "/skill:" in p for p in backend.prompts)


async def test_fork_stack_rule_routes_deep_per_stack_review(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    make_config: Callable[..., RunConfig],
    mute_side_effects: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daydream_ext ``add_stack(StackRule(...))`` reaches the deep per-stack review.

    Commits a ``.proto`` file into the multi-stack diff so ``detect_stacks``
    (deep flow only) sees a file matching the fork glob, then drives the full
    deep pipeline through ``runner.run``. Observable outcomes: exit 0, the
    routed ``.proto`` file reaches the per-stack reviewer, and the prompt is
    native (profile strategy, no fork skill invocation).
    """
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    ext_dir.write_module(
        "from daydream.extensions import StackRule\n"
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
    mute_side_effects("daydream.deep.orchestrator", heal=False, commit=False)

    rc = await runner.run(make_config(multi_stack_target))

    assert rc == 0
    # The fork StackRule routes the .proto file to the proto stack (scope
    # metadata), but the native per-stack prompt carries the profile strategy
    # with no skill invocation (M2/M12).
    proto_prompts = [
        c["prompt"] for c in backend.calls
        if "you are reviewing the proto stack" in c["prompt"].lower()
    ]
    assert proto_prompts and "api.proto" in proto_prompts[0]
    assert "/ro-proto:review-proto" not in proto_prompts[0]
    assert "/beagle-" not in proto_prompts[0]
