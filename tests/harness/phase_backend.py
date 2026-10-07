"""Scripted shallow-flow backend with optional raw event replay.

Dispatch mode consumes per-iteration issue lists and records protocol calls."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from daydream.backends import AgentEvent, ResultEvent, TextEvent
from tests.harness.stub_backend import review_stage_result


def _shape_issues(issues: list[dict[str, Any]], severity: str | None = None,) -> list[dict[str, Any]]:
    """Supply grounded defaults while preserving explicit per-issue overrides."""
    grounded: dict[str, Any] = {"confidence": "HIGH", "rationale": "harness fixture", "evidence": ""}
    if severity is not None:
        grounded["severity"] = severity
    return [{**grounded, "evidence": f"{it.get('file') or 'harness.py'}:{it.get('line') or 1}", **it} for it in issues]


class PhaseDispatchBackend:
    """Dispatch review/fix/test prompts and record calls for real-path tests.

    review_prompts and prompts retain full text; call_log keeps 80-character
    lowercase prefixes. Review and explicit extraction use independent queues."""

    model = "mock-model"

    def __init__(
        self, parse_results: list[list[dict[str, Any]]] | None = None, *, events: list[AgentEvent] | None = None,
        tests_pass: bool = True,
    ) -> None:
        """Set issue lists (empty after exhaustion), optional raw events, and suite outcome."""
        self._parse_results = parse_results or []
        self._events = events
        self._tests_pass = tests_pass
        self._parse_call = 0
        self._review_call = 0
        self.call_log: list[str] = []
        self.review_prompts: list[str] = []
        self.last_prompt: str = ""
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def parse_calls(self) -> int:
        """Number of explicit extract-JSON dispatches, separate from native review."""
        return self._parse_call

    @property
    def prompts(self) -> list[str]:
        """Full prompt of each recorded call, in call order."""
        return [call["prompt"] for call in self.calls]

    async def execute(self, cwd: Any, prompt: str, output_schema: Any=None, continuation: Any=None, agents: Any=None,
        max_turns: Any=None, read_only: Any=False, persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        self.last_prompt = prompt
        self.call_count += 1
        self.calls.append({
                "cwd": cwd, "prompt": prompt, "output_schema": output_schema, "agents": agents, "model": self.model,
                "continuation": continuation, "max_turns": max_turns, "read_only": read_only,
                "persist_session": persist_session,
            }
        )

        if self._events is not None:
            for event in self._events:
                yield event
            return

        prompt_lower = prompt.lower()
        self.call_log.append(prompt_lower[:80])

        # Recognize both explicit skill prompts and native review instructions.
        structured: dict[str, Any] | None = None
        _review_markers = (
            "inclusion obligation",  # _stack_scope_instruction (pre-threading)
            "full change spans",  # structural runtime line (pre-threading)
            "language-agnostic review practices",  # generic-fallback strategy
            "assigned to this stack",  # authored per-stack strategy (post-threading)
            "repository-wide interactions",  # authored structural strategy (post-threading)
        )
        if ("beagle-" in prompt_lower
            and "review" in prompt_lower
            or any(marker in prompt_lower for marker in _review_markers)
        ):
            self.review_prompts.append(prompt)
            yield TextEvent(text="Review complete.")
            if output_schema is not None:
                # Native review emits structured records directly.
                first_discovery = True
                marker = "Host review stage:\n"
                if marker in prompt:
                    state = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]
                    first_discovery = state["stage"] == "first_pass" and not state["progress"]
                issues = []
                if first_discovery:
                    issues = (self._parse_results[self._review_call]
                        if self._review_call < len(self._parse_results)
                        else []
                    )
                    self._review_call += 1
                issues = _shape_issues(issues, severity="medium")
                structured = review_stage_result(prompt, issues)
        elif "extract" in prompt_lower and "json" in prompt_lower:
            issues = (self._parse_results[self._parse_call] if self._parse_call < len(self._parse_results) else [])
            self._parse_call += 1
            issues = _shape_issues(issues)
            yield TextEvent(text="Parsed.")
            structured = {"issues": issues}
        elif "post-fix fix-verifier agent" in prompt_lower:
            ids = [int(value) for value in re.findall(r"(?m)^(\d+)\. \[", prompt)]
            yield TextEvent(text="")
            structured = {"verdicts": [
                {"issue_id": issue_id, "verdict": "resolved", "reason": "harness fix accepted"}
                for issue_id in ids
            ]}
        elif "fix this issue" in prompt_lower or prompt_lower.startswith("fix these"):
            yield TextEvent(text="Fixed.")
        elif "test suite" in prompt_lower or "run the project" in prompt_lower:
            if self._tests_pass:
                yield TextEvent(text="All 1 tests passed. 0 failed.")
            else:
                yield TextEvent(text="1 test failed.")
        elif "the daydream changes are already staged" in prompt_lower and "do not push" in prompt_lower:
            yield TextEvent(text="Committed iteration changes.")
        elif "commit-push" in prompt_lower:
            yield TextEvent(text="Committed.")
        else:
            yield TextEvent(text="OK")
        yield ResultEvent(structured_output=structured, continuation=None)

    async def cancel(self) -> None:
        pass
