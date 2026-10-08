"""Pi discovery uses admitted diff references even for small changes."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ResultEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.pi import PiBackend
from daydream.deep.detection import StackAssignment
from daydream.hunk_index import write_hunk_index
from daydream.run_context import InteractionPolicy, RunContext
from tests.deep_orchestrator.test_review_capture_and_retry import supporting_contents
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.review_result import review_scopes
from tests.harness.stub_backend import review_stage_result, review_stage_state
from tests.test_deep_orchestrator import _sanctioned_inputs


async def test_small_pi_review_keeps_structural_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Any,
) -> None:
    (tmp_path / "app.py").write_text("value = 'DIFF_SENTINEL'\n")
    seed_feature_branch(tmp_path, base={'app.py': 'value = 0\n'},
                        feature={'app.py': "value = 'DIFF_SENTINEL'\n"})
    diff = tmp_path / "diff.patch"
    diff.write_text("diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
                    "@@ -1 +1 @@\n-value = 0\n+value = 'DIFF_SENTINEL'\n")
    write_hunk_index(tmp_path, diff.read_text())
    intent = tmp_path / "intent.md"
    intent.write_text("Change the value")
    calls: list[str] = []

    async def review(
        self: PiBackend, cwd: Path, prompt: str, *args: Any, **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        assert str(diff) not in prompt
        assert "review-assignment" in _sanctioned_inputs(prompt)
        state = review_stage_state(prompt)
        assert state is not None
        captured_diff = supporting_contents(prompt)['diff'] if state['scope_id'] == 'python' else ''.join(
            Path(part['path']).read_text() for part in state['supporting_parts'])
        assert "DIFF_SENTINEL" in captured_diff
        assert "DIFF_SENTINEL" not in prompt
        assert kwargs["read_only"] is True
        assert not kwargs.get("tools_disabled")
        assert "Host review stage:" in prompt
        assert "INVESTIGATION HAS ENDED" not in prompt
        calls.append(prompt)
        yield ToolStartEvent(id="source-app", name="Read", input={"file_path": "app.py"})
        yield ToolResultEvent(id="source-app", output=(cwd / "app.py").read_text(), is_error=False)
        yield ResultEvent(structured_output=review_stage_result(prompt, []), continuation=None)

    monkeypatch.setattr(PiBackend, "execute", review)
    work = make_work(tmp_path, base_sha=git(tmp_path, "rev-parse", "main"),
                     head_sha=git(tmp_path, "rev-parse", "HEAD"), head_branch="feature")
    results, failures = await review_scopes(PiBackend(model="fixture"), work,
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
