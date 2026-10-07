"""External-provider fixture for real-run empty synthesis regressions."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable, Iterable
from pathlib import Path
from typing import Any

from daydream.backends import AgentEvent, ResultEvent, ToolStartEvent
from daydream.config_file import DaydreamFileConfig
from daydream.run_config import RunConfig
from tests.harness.stub_backend import StubBackend, review_stage_result


class EmptyReviewBackend(StubBackend):
    """Return empty reviewer findings and reject unnecessary synthesis calls."""

    def __init__(
        self, target: Path, *, forbid_merge: bool = True, forbid_supervise: bool = True,
        review_by_stack: dict[str, list[dict[str, Any]]] | None = None,
        fail_stack: str | None = None, stack_error: Exception | None = None,
        alternatives: list[dict[str, Any]] | None = None,
        incomplete_stack: str | None = None,
        responder: Callable[[str], Iterable[AgentEvent | BaseException] | None] | None = None,
    ) -> None:
        super().__init__(target)
        self.forbid_merge = forbid_merge
        self.forbid_supervise = forbid_supervise
        self.merge_items = []
        self.review_by_stack = review_by_stack or {}
        self.fail_stack = fail_stack
        self.stack_error = stack_error or RuntimeError("review provider unavailable")
        self.alternatives = alternatives or []
        self.incomplete_stack = incomplete_stack
        self.responder = responder

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if self.responder is not None:
            events = self.responder(prompt)
            if events is not None:
                self.calls.append({"prompt": prompt, "model": self.model})
                for event in events:
                    if isinstance(event, BaseException):
                        raise event
                    yield event
                return
        lower = prompt.lower()
        merge = "cross-stack merge agent" in lower
        supervise = "supervisor adjudication" in lower
        if (merge and self.forbid_merge) or (supervise and self.forbid_supervise):
            self.calls.append({"prompt": prompt, "model": self.model})
            raise AssertionError("eligible empty synthesis must not dispatch to a provider")
        stack_match = re.search(r"you are reviewing the (\S+) stack", lower)
        stack = stack_match.group(1) if stack_match else None
        if "you are the structural reviewer" in lower:
            stack = "structure"
        if stack is not None:
            self.calls.append({"prompt": prompt, "model": self.model})
            if stack == self.incomplete_stack:
                yield ToolStartEvent(id="vetoed-review", name="Bash", input={"command": "blocked-review-tool"})
                return
            if stack == self.fail_stack:
                raise self.stack_error
            yield ResultEvent(structured_output=review_stage_result(prompt, self.review_by_stack.get(stack, [])),
                              continuation=None)
            return
        if "would you have done this differently" in lower or "evaluate the implementation" in lower:
            self.calls.append({"prompt": prompt, "model": self.model})
            yield ResultEvent(structured_output={"issues": self.alternatives}, continuation=None)
            return
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


def empty_review_config(target: Path, trajectory: Path, **overrides: Any) -> RunConfig:
    """Use production review/report routing without a publishing or fix gate."""
    fields: dict[str, Any] = {
        "target": str(target),
        "base": "main",
        "output_mode": "review",
        "assume": "yes",
        "non_interactive": True,
        "cleanup": False,
        "archive": False,
        "run_eval": False,
        "diagram": "off",
        "review_cache_enabled": False,
        "shallow_fanout_threshold": 0,
        "file_config": DaydreamFileConfig(supervisor="llm"),
        "trajectory_path": trajectory,
    }
    fields.update(overrides)
    return RunConfig(**fields)
