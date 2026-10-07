"""Pi discovery uses admitted diff references even for small changes."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ResultEvent
from daydream.backends.pi import PiBackend
from daydream.deep.detection import StackAssignment
from daydream.run_context import InteractionPolicy, RunContext
from tests.harness.review_result import review_scopes
from tests.harness.stub_backend import review_stage_result, review_stage_state


async def test_small_pi_review_keeps_structural_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Any,
) -> None:
    (tmp_path / "app.py").write_text("value = 'DIFF_SENTINEL'\n")
    diff = tmp_path / "diff.patch"
    diff.write_text("diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
                    "@@ -1 +1 @@\n-value = 0\n+value = 'DIFF_SENTINEL'\n")
    intent = tmp_path / "intent.md"
    intent.write_text("Change the value")
    calls: list[str] = []

    async def review(
        self: PiBackend, cwd: Path, prompt: str, *args: Any, **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        assert str(diff) in prompt
        assert "DIFF_SENTINEL" not in prompt
        assert kwargs["read_only"] is True
        assert not kwargs.get("tools_disabled")
        assert "Host review stage:" in prompt
        assert "INVESTIGATION HAS ENDED" not in prompt
        calls.append(prompt)
        yield ResultEvent(structured_output=review_stage_result(prompt, []), continuation=None)

    monkeypatch.setattr(PiBackend, "execute", review)
    results, failures = await review_scopes(PiBackend(model="fixture"), make_work(tmp_path),
        [StackAssignment("python", ["app.py"]), StackAssignment("structure", ["app.py"])],
        diff_path=diff, diff_text=diff.read_text(), intent_path=intent,
        alternatives_path=tmp_path / "alternatives.json", allow_standalone=True,
        run_context=RunContext(InteractionPolicy(interactive=False)),
    )
    assert failures == {}
    assert set(results) == {"python", "structure"}
    stages = [review_stage_state(prompt) for prompt in calls]
    assert sorted((stage["scope_id"], stage["stage"], stage["assigned_files"])
                  for stage in stages if stage is not None) == [
        ("python", "first_pass", ["app.py"]), ("structure", "integration", ["app.py"]),
    ]
