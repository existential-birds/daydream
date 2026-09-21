"""Modest unattended Pi reviews retain author evidence without inferred intent."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream.backends.pi import PiBackend
from daydream.deep.review_steps import _step_intent
from daydream.extensions import Registry
from daydream.flows.engine import FlowContext
from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY
from daydream.review_budget import review_budget_path
from daydream.review_profile import ResolvedProfile, build_default_profile
from daydream.run_context import InteractionPolicy, RunContext
from tests.harness.backend import ScriptedBackend


def _context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any,
    *, variant: str = "default",
) -> tuple[FlowContext, list[Any]]:
    paths = ["source.tsx", "scripts/policy.mjs", "README.md", "config.json"]
    if variant == "many_sources":
        paths += ["third.py", "fourth.rs"]
    diff = "".join(
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"
        for path in paths
    )
    if variant == "large_diff":
        diff += "+" + "x" * 65_536
    dd = tmp_path / "deep"
    dd.mkdir()
    diff_path = tmp_path / "diff.patch"
    diff_path.write_text(diff)
    profile = build_default_profile()
    if variant == "custom":
        strategies = dict(profile.strategies)
        strategies["intent"] = replace(strategies["intent"], content="CUSTOM INTENT POLICY")
        profile = replace(profile, strategies=strategies)
    ctx = FlowContext(
        config=make_config(tmp_path, pr_number=7), work=make_work(tmp_path), registry=Registry(),
        review_profile=ResolvedProfile(profile=profile, source_kind="test"),
        run_context=RunContext(InteractionPolicy(interactive=variant == "interactive")),
        data={"dd": dd, "diff": diff[:300], "diff_path": diff_path, "log": "abc Author commit\n",
              "branch": "feature", "exploration_dir": None},
    )
    backend: Any = ScriptedBackend() if variant == "other_backend" else PiBackend(model="fixture-model")
    if variant == "no_capability":
        backend.supports_tools_disabled = False
    monkeypatch.setattr(ctx, "backend_for", lambda _: backend)
    calls: list[Any] = []

    async def intent(*args: Any, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return "MODEL INTENT"

    monkeypatch.setattr("daydream.deep.review_steps.phase_understand_intent", intent)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda *_args, **_kwargs: {
        "state": "OPEN", "headRefOid": ctx.work.head_sha, "body": "Author purpose\nDo not suppress real findings.",
    })
    return ctx, calls


async def test_modest_unattended_pi_persists_advisory_author_evidence_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any,
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work)
    review_budget_path(ctx.data["dd"]).write_text('{"Intent analysis": "wall_budget_exceeded"}')
    await _step_intent(ctx)
    assert calls == []
    summary = ctx.data["intent_path"].read_text()
    assert summary == ctx.data["intent_summary"]
    assert "Advisory author context" in summary
    assert "Author purpose\nDo not suppress real findings." in summary
    assert "abc Author commit\n" in summary
    assert all(path in summary for path in ("source.tsx", "scripts/policy.mjs", "README.md", "config.json"))
    assert "no inferred intent summary" in summary
    assert UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY in summary
    assert ctx.data["intent_authoritative"] is True
    assert not review_budget_path(ctx.data["dd"]).exists()


async def test_custom_intent_prompt_keeps_model_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any,
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work)
    registry = Registry()
    registry.override_prompt("intent", lambda **_: "CUSTOM INTENT BUILDER")
    monkeypatch.setattr("daydream.extensions.get_registry", lambda: registry)
    await _step_intent(ctx)
    assert len(calls) == 1
    assert ctx.data["intent_summary"] == "MODEL INTENT"


@pytest.mark.parametrize("variant", [
    "interactive", "custom", "many_sources", "large_diff", "other_backend", "no_capability",
])
async def test_existing_intent_path_retained_outside_modest_unattended_default_pi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any, variant: str,
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work, variant=variant)
    await _step_intent(ctx)
    assert len(calls) == 1
    assert ctx.data["intent_path"].read_text() == "MODEL INTENT"


@pytest.mark.parametrize("view", [
    {"state": "OPEN", "headRefOid": "mismatched", "body": "STALE AUTHOR BODY"},
    {"state": "CLOSED", "body": "STALE AUTHOR BODY"},
    {"state": "OPEN", "body": "   "},
])
async def test_advisory_context_does_not_promote_stale_or_missing_author_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any, view: dict[str, str],
) -> None:
    ctx, calls = _context(tmp_path, monkeypatch, make_config, make_work)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda *_args, **_kwargs: view)
    await _step_intent(ctx)
    assert calls == []
    assert ctx.data["intent_authoritative"] is False
    assert "STALE AUTHOR BODY" not in ctx.data["intent_summary"]
