"""Pi discovery uses admitted diff references even for small changes."""

from pathlib import Path
from typing import Any

import pytest

from daydream.backends.pi import PiBackend
from daydream.deep.detection import StackAssignment
from daydream.run_context import InteractionPolicy, RunContext
from tests.harness.review_result import review_scopes


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

    async def review(*args: Any, **kwargs: Any) -> Any:
        prompt = args[2]
        assert str(diff) in prompt
        assert "DIFF_SENTINEL" not in prompt
        assert kwargs["read_only"] is True
        assert not kwargs.get("tools_disabled")
        inputs = kwargs["sanctioned_inputs"]
        inputs.revalidate(args[0], args[1], True)
        assert "DIFF_SENTINEL" not in inputs.finalization_text(args[0], args[1], True)
        assert "DIFF_SENTINEL" not in repr(kwargs["finalization_context"])
        calls.append(prompt)
        return {"issues": []}, None, None

    monkeypatch.setattr("daydream.agent.run_agent", review)
    results, failures = await review_scopes(PiBackend(model="fixture"), make_work(tmp_path),
        [StackAssignment("python", ["app.py"]), StackAssignment("structure", ["app.py"])],
        diff_path=diff, diff_text=diff.read_text(), intent_path=intent,
        alternatives_path=tmp_path / "alternatives.json", allow_standalone=True,
        run_context=RunContext(InteractionPolicy(interactive=False)),
    )
    assert failures == {}
    assert set(results) == {"python", "structure"}
    assert len(calls) == 2
    deep = tmp_path / ".daydream/deep"
    assert not (deep / "coverage-receipts.json").exists()
