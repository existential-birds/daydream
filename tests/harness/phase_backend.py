"""Shared phase-dispatch fake ``Backend`` for shallow review-fix-test tests.

Consolidates the two genuinely-redundant prompt-heuristic dispatch mocks that
the shallow loop and single-pass integration tests grew independently
(``loop_mock_backend`` in ``tests/test_loop.py`` and ``MockBackend`` in
``tests/test_integration.py``). Both classified the same four shallow phases off
the same prompt substrings; this is the unified implementation.

``PhaseDispatchBackend`` is a *dispatch* fake, not a replay fake: it routes by
prompt heuristic (``beagle-*``/``extract json``/``fix this issue``/``test
suite``) and serves a per-iteration ``parse_results`` queue. The per-iteration
queue is the one place the shallow loop legitimately needs *sequence* (issues on
iteration 1, clean on iteration 2 → exit 0) — deliberately kept OUT of the
phase-keyed *replay* harness (``tests/harness/phase_replay.py``), which serves
one fixture per phase per firing.

It implements the 3-method ``Backend`` interface (``execute``/``cancel``/
Backend protocol methods (``execute`` / ``cancel``). Two response modes:

* ``events``: a raw pre-built ``AgentEvent`` list yielded verbatim (the
  tool-panel / single-pass-fixture mode the old ``MockBackend(events=...)``
  supported).
* default: prompt-heuristic dispatch with a per-iteration parse queue.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from daydream.backends import AgentEvent, CostEvent, ResultEvent, TextEvent


def _shape_issues(
    issues: list[dict[str, Any]],
    severity: str | None = None,
) -> list[dict[str, Any]]:
    """Shape a parsed issue list into harness ground-shaped structured records.

    Shared by the review and extract-json dispatch branches (both were
    near-identical duplicate comprehensions). ``**it`` stays last so explicit
    per-item fields still win over the grounded defaults. Only the review
    branch pins a default ``severity``.
    """
    grounded: dict[str, Any] = {
        "confidence": "HIGH",
        "rationale": "harness fixture",
        "evidence": "",
    }
    if severity is not None:
        grounded["severity"] = severity
    return [
        {
            **grounded,
            "evidence": f"{it.get('file') or 'harness.py'}:{it.get('line') or 1}",
            **it,
        }
        for it in issues
    ]


class PhaseDispatchBackend:
    """Prompt-heuristic dispatch fake with a per-iteration parse-results queue.

    Attributes:
        model: Stable fake model name (satisfies the ``Backend`` surface).
        parse_calls: Number of parse-feedback phases dispatched (observable
            proof of how many loop iterations reached review/parse).
        call_log: Truncated lowercased prompts, in dispatch order.
        commit_calls: Iteration-commit prompts seen (one per successful fix
            iteration in loop mode).
        review_prompts: Full review-phase prompts, in order (lets diff-base
            assertions inspect ``git diff`` targets per iteration).
        last_prompt: The most recent prompt passed to ``execute``.
        call_count: Total ``execute`` invocations.
    """

    model = "mock-model"

    def __init__(
        self,
        parse_results: list[list[dict[str, Any]]] | None = None,
        *,
        events: list[AgentEvent] | None = None,
        tests_pass: bool = True,
        emit_cost: bool = False,
    ) -> None:
        """Configure the fake.

        Args:
            parse_results: One issue-list per iteration; the parse phase returns
                ``{"issues": parse_results[n]}`` on the n-th parse, then ``[]``
                once the queue is exhausted. ``None`` => always empty.
            events: When set, ``execute`` yields this raw event list verbatim
                and skips dispatch (tool-panel / fixed-fixture mode).
            tests_pass: Controls the test-suite phase's pass/fail text.
            emit_cost: When True (and not in ``events`` mode), emit a
                ``CostEvent`` on the non-structured branches (matches the old
                integration ``MockBackend`` default-event shape).
        """
        self._parse_results = parse_results or []
        self._events = events
        self._tests_pass = tests_pass
        self._emit_cost = emit_cost
        self._parse_call = 0
        self._review_call = 0
        self.call_log: list[str] = []
        self.commit_calls: list[str] = []
        self.review_prompts: list[str] = []
        self.last_prompt: str = ""
        self.call_count = 0

    @property
    def parse_calls(self) -> int:
        """Number of parse-feedback phases dispatched so far (observable)."""
        return self._parse_call

    async def execute(
        self,
        cwd: Any,
        prompt: str,
        output_schema: Any=None,
        continuation: Any=None,
        agents: Any=None,
        max_turns: Any=None,
        read_only: Any=False,
    ) -> AsyncGenerator[AgentEvent, None]:
        self.last_prompt = prompt
        self.call_count += 1

        if self._events is not None:
            for event in self._events:
                yield event
            return

        prompt_lower = prompt.lower()
        self.call_log.append(prompt_lower[:80])

        # Native review prompts are skill-free (#886): the per-stack / structural /
        # generic-fallback builders no longer carry a ``beagle-*`` invocation, so
        # dispatch on their distinctive judgment-prose markers instead (covering
        # the pre- and post-strategy-threading wording).
        _review_markers = (
            "inclusion obligation",  # _stack_scope_instruction (pre-threading)
            "full change spans",  # structural runtime line (pre-threading)
            "language-agnostic review practices",  # generic-fallback strategy
            "assigned to this stack",  # authored per-stack strategy (post-threading)
            "repository-wide interactions",  # authored structural strategy (post-threading)
        )
        if (
            "beagle-" in prompt_lower
            and "review" in prompt_lower
            or any(marker in prompt_lower for marker in _review_markers)
        ):
            self.review_prompts.append(prompt)
            yield TextEvent(text="Review complete.")
            if output_schema is not None:
                # Issue #745 (AC4): the per-stack reviewer emits
                # PER_STACK_RECORD_SCHEMA structured output directly (the
                # deep-shallow spine no longer has a separate parse step).
                issues = (
                    self._parse_results[self._review_call]
                    if self._review_call < len(self._parse_results)
                    else []
                )
                self._review_call += 1
                issues = _shape_issues(issues, severity="medium")
                yield ResultEvent(
                    structured_output={"issues": issues, "verdicts": []},
                    continuation=None,
                )
            else:
                yield ResultEvent(structured_output=None, continuation=None)
        elif "extract" in prompt_lower and "json" in prompt_lower:
            issues = (
                self._parse_results[self._parse_call]
                if self._parse_call < len(self._parse_results)
                else []
            )
            self._parse_call += 1
            issues = _shape_issues(issues)
            yield TextEvent(text="Parsed.")
            # Issue #742: the deep per-stack parse schema requires a
            # ``verdicts`` property (Codex strict-mode output), so the parse
            # payload always carries it (empty when not exercised).
            yield ResultEvent(
                structured_output={"issues": issues, "verdicts": []}, continuation=None
            )
        elif "fix this issue" in prompt_lower or prompt_lower.startswith("fix these"):
            yield TextEvent(text="Fixed.")
            yield ResultEvent(structured_output=None, continuation=None)
        elif "test suite" in prompt_lower or "run the project" in prompt_lower:
            if self._tests_pass:
                yield TextEvent(text="All 1 tests passed. 0 failed.")
            else:
                yield TextEvent(text="1 test failed.")
            yield ResultEvent(structured_output=None, continuation=None)
        elif "the daydream changes are already staged" in prompt_lower and "do not push" in prompt_lower:
            self.commit_calls.append(prompt_lower)
            yield TextEvent(text="Committed iteration changes.")
            yield ResultEvent(structured_output=None, continuation=None)
        elif "commit-push" in prompt_lower:
            yield TextEvent(text="Committed.")
            yield ResultEvent(structured_output=None, continuation=None)
        else:
            yield TextEvent(text="OK")
            if self._emit_cost:
                yield CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None)
            yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass
