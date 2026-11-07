"""Scripted shallow-flow backend that dispatches review, fix-verify and fix prompts.

Native review consumes per-iteration issue lists; other prompts get canned replies."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any

from daydream.backends import AgentEvent, ResultEvent, TextEvent
from tests.harness.stub_backend import review_stage_result, review_stage_state


def _shape_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Supply grounded defaults while preserving explicit per-issue overrides."""
    grounded: dict[str, Any] = {"confidence": "HIGH", "rationale": "harness fixture", "evidence": "",
                                "severity": "medium"}
    return [{**grounded, "evidence": f"{it.get('file') or 'harness.py'}:{it.get('line') or 1}", **it} for it in issues]


class PhaseDispatchBackend:
    """Dispatch native review, fix-verify and fix prompts for real-path tests."""

    model = "mock-model"

    def __init__(self, parse_results: list[list[dict[str, Any]]] | None = None) -> None:
        """Set per-discovery review issue lists (empty after exhaustion)."""
        self._parse_results = parse_results or []
        self._review_call = 0

    async def execute(self, cwd: Any, prompt: str, output_schema: Any=None, continuation: Any=None, agents: Any=None,
        read_only: Any=False, persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        prompt_lower = prompt.lower()
        # Recognize native review instructions.
        structured: dict[str, Any] | None = None
        _review_markers = (
            "full change spans",  # structural runtime line
            "language-agnostic review practices",  # generic-fallback strategy
            "assigned to this stack",  # authored per-stack strategy
            "repository-wide interactions",  # authored structural strategy
        )
        if any(marker in prompt_lower for marker in _review_markers):
            yield TextEvent(text="Review complete.")
            if output_schema is not None:
                # Native review emits structured records directly.
                state = review_stage_state(prompt)
                first_discovery = state is None or (state["stage"] in {"first_pass", "integration"}
                                                    and not state["progress"])
                issues = []
                if first_discovery:
                    issues = (self._parse_results[self._review_call] if self._review_call < len(self._parse_results)
                        else []
                    )
                    self._review_call += 1
                structured = review_stage_result(prompt, _shape_issues(issues))
        elif "post-fix fix-verifier agent" in prompt_lower:
            ids = [int(value) for value in re.findall(r"(?m)^(\d+)\. \[", prompt)]
            yield TextEvent(text="")
            structured = {"verdicts": [
                {"issue_id": issue_id, "verdict": "resolved", "reason": "harness fix accepted"}
                for issue_id in ids
            ]}
        elif "fix this issue" in prompt_lower or prompt_lower.startswith("fix these"):
            yield TextEvent(text="Fixed.")
        else:
            yield TextEvent(text="OK")
        yield ResultEvent(structured_output=structured, continuation=None)

    async def cancel(self) -> None:
        pass
