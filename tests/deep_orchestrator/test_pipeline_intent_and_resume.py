"""Pipeline Intent And Resume."""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream.runner import run
from tests.test_deep_orchestrator import (
    PR_SENTINEL,
    MakeConfig,
    _assert_authoritative_rule_gated,
    _install_stub_backend,
    _intent_prompt,
    _silence,
)


async def test_pr_body_reaches_intent_prompt(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    _silence(monkeypatch)
    monkeypatch.setattr(
        "daydream.git_ops.gh_pr_view", lambda repo, pr=None, **_kwargs: {"number": 7, "body": PR_SENTINEL},
    )
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    # High severity so the scoped arbiter fires and all five builders are covered.
    stub.parse_severity = "high"
    rc = await run(make_config(multi_stack_target, pr_number=7))
    assert rc == 0
    assert PR_SENTINEL in _intent_prompt(stub)
    _assert_authoritative_rule_gated(stub, expect_present=True)
