"""External-provider fixture for real-run terminal review journeys."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable
from pathlib import Path
from typing import Any

from daydream.backends import AgentEvent
from daydream.config_file import DaydreamFileConfig
from daydream.run_config import RunConfig
from tests.harness.stub_backend import StubBackend


class EmptyReviewBackend(StubBackend):
    """Review without proposing a claim unless a test scripts one; a responder may act on or fail any prompt."""

    def __init__(
        self, target: Path, *, responder: Callable[[str], Iterable[AgentEvent | BaseException] | None] | None = None,
    ) -> None:
        super().__init__(target)
        self.review_claims = []
        self.responder = responder

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        events = self.responder(prompt) if self.responder is not None else None
        if events is not None:
            self.calls.append({"prompt": prompt, "model": self.model})
            for event in events:
                if isinstance(event, BaseException):
                    raise event
                yield event
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
        "file_config": DaydreamFileConfig(),
        "trajectory_path": trajectory,
    }
    fields.update(overrides)
    return RunConfig(**fields)
