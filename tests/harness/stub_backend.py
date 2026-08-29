"""Shared stub backend for deep-pipeline integration tests.

Canonical home of the prompt-dispatching stub backend and its install helpers.
Previously these lived as ``_``-private symbols at the top of
``tests/test_deep_orchestrator.py`` and were imported across modules; lifting
them here gives every consumer a single public, documented stub to import.

Public surface:

* ``StubBackend`` -- prompt-dispatching mock backend. Writes realistic per-stack
  review output and a merged report so the orchestrator progresses through every
  stage, and records every call so tests can assert ordering, ``agents=``
  absence, and per-stack isolation.
* ``install_stub_backend`` -- patch ``create_backend`` to return one stub
  instance, with optional exploration pinning.
* ``silence`` -- silence noise-only UI helpers in the deep orchestrator and
  phases (``prompts=False`` leaves the real prompt seam in place).
* ``force_interactive`` -- pin a TTY stdin and unset ``CI`` so a test drives the
  real interactive prompt path.
"""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    MaxTurnsError,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.eval.analyzer import _records_issues_or_empty

PARTIAL_FIX_MARKER = "// PARTIAL BROKEN EDIT -- max turns exhausted mid-fix\n"


class StubBackend:
    """MockBackend that dispatches on prompt content.

    Writes realistic per-stack review outputs and a merged report so the
    orchestrator can progress through every stage. Records every call so
    tests can assert ordering, agents-kwarg absence, and per-stack isolation.
    """

    model = "mock-model"

    def __init__(
        self,
        target: Path,
        *,
        model: str = "mock-model",
        shared_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self._target = target
        # When set, every execute() call is also appended (model-tagged) to this
        # shared list so a per-(name, model) factory can capture which model ran
        # each phase even though backends are cached by (name, model) (#168).
        self._shared = shared_calls
        # #168 knobs: per-stack parse severity to emit (drives arbiter selection),
        # and whether the merge agent echoes the on-disk per-stack records (so the
        # rendered artifact reflects arbiter revisions instead of fixed items).
        self.parse_severity: str | None = None
        self.merge_echo_records: bool = False
        # When True, the arbiter branch returns an empty findings list (omits every
        # verdict), simulating a truncated/lazy Opus response so a test can assert
        # the selected high-severity record fails open and survives (#175).
        self.arbiter_omit_verdicts: bool = False
        # #232 knobs: per-stack parse override (drives suppression selection with a
        # distinct file/severity/confidence per stack, so a HIGH and a borderline
        # LOW finding coexist at NON-colliding locations -> uncontested), and the
        # keep verdict the suppression-reviewer stub returns for every sup_id.
        self.parse_by_stack: dict[str, dict[str, Any]] | None = None
        self.suppression_keep: bool = True
        self.calls: list[dict[str, Any]] = []
        # Verdict the recommendation-verifier branch emits for issue_id=1; flip to
        # "contradicts" to exercise verdict propagation into the phase_fix prompt.
        self.verifier_verdict: str = "consistent"
        self.verifier_unverified_assumptions: list[str] = []
        # Post-fix fix-verifier knobs (#744). fix_verify_verdicts overrides the
        # verdict for a finding id (default: resolved) on round 1 only; when
        # None every dispatched finding resolves. fix_verify_requires_read_only
        # pins the phase's read_only=True contract (default True).
        # fix_verify_resolve_after_round returns unresolved for rounds below
        # the threshold, resolved thereafter (drives the re-dispatch loop).
        self.fix_verify_verdicts: dict[int, dict[str, Any]] | None = None
        self.fix_verify_requires_read_only: bool = True
        self.fix_verify_resolve_after_round: int = 1
        # Counts test-suite invocations so a test can fail the FIRST run (driving
        # the heal loop into choice "2") and pass the SECOND.
        self.test_suite_calls: int = 0
        # When True, the test-suite branch fails the first call and passes after.
        self.fail_first_test_run: bool = False
        # When True, EVERY test-suite run fails (permanently-red suite).
        self.fail_all_test_runs: bool = False
        # Override for the merge agent's item list (None -> default three-item payload).
        self.merge_items: list[dict[str, Any]] | None = None
        # When set, the cross-stack merge branch emits unparseable prose text
        # (no structured output) -- drives the "got str" salvage path (#361).
        self.merge_emit_str: str | None = None
        # When set, the cross-stack merge branch emits a BARE item list (not
        # wrapped in {"items": [...]}) -- the R1 parseable-list shape (#361).
        self.merge_emit_bare_list: list[dict[str, Any]] | None = None
        # Optional LLM supervisor verdicts keyed by canonical item id.
        self.supervise_verdicts: dict[int, dict[str, Any]] | None = None
        # Optional deferred Write tool pairs for the built-in tool-supervisor tests.
        self.deferred_write_pairs: list[str] | None = None
        # When set, the fix branch appends the prompt's marker token to this file
        # via read-modify-write with an anyio.sleep(0) interleave point -- a
        # per-ITEM fan-out would lose/reorder appends; correct per-file
        # serialization preserves marker order.
        self.fix_append_path: Path | None = None
        # When set, the fix branch raises for the matching file, isolating one
        # file-group's failure so a test can assert the others still applied.
        self.fix_fail_file: str | None = None
        # When set, ONLY the batched fix turn ("Fix these N issues in <file>")
        # for the matching file raises, forcing phase_fix_parallel into its
        # per-finding fallback loop -- the #186 pattern the group budget bounds.
        # The per-finding retries ("Fix this issue") for that file then succeed.
        self.fail_batched_fix_file: str | None = None
        # When set, ONLY the batched fix turn for the matching file emits a slow
        # runaway burst (real per-event sleep, never a ResultEvent) so run_agent's
        # OWN per-invocation WALL budget trips and returns a budget_reason -- the
        # batched call then raises and phase_fix_parallel falls back to per-finding
        # fixes. Unlike fail_batched_fix_file (a synchronous stub raise), this
        # exercises the real budget_reason -> raise path AND burns real wall-time
        # that carries into the fallback via the shared FileGroupBudget (#201).
        # Per-finding fallback turns for that file are NOT runaway.
        self.runaway_batched_fix_file: str | None = None
        self.runaway_batched_sleep_s: float = 0.05
        # When set, the fix branch WRITES a broken partial edit to the matching
        # file and THEN raises MaxTurnsError -- simulating an agent that mutated
        # the tree before exhausting its turn budget, so a test can assert the
        # orchestrator reverts that partial edit and saves a recovery patch.
        self.fix_partial_then_maxturns: str | None = None
        # Turns this stub's fix agent needs to finish. Models the CLI contract:
        # a turn ceiling below this raises MaxTurnsError instead of applying the
        # edit. Set above a former ceiling to prove the fix is no longer capped.
        self.fix_turns_needed: int = 0
        # Repo-relative path of a stray untracked file the failing group creates
        # before raising (e.g. "store/uuid.go") -- NOT the group's key file, so
        # it survives tree-protection and must surface in fix_leftover_untracked.
        self.fix_orphan_file: str | None = None
        # Repo-relative generated path created by a successful fix turn.
        self.fix_new_generated: str | None = None
        # Repo-relative historical generated path modified by a test-healing
        # fix turn, plus an optional new generated path created by that turn.
        self.heal_fix_generated: str | None = None
        self.heal_fix_new_generated: str | None = None
        # When True, the fix branch yields a long burst of ToolStartEvents and
        # NEVER emits a ResultEvent -- simulating a runaway turn. Without the
        # in-loop tool-call budget in run_agent this stream never completes and
        # the run hangs; with it, the loop aborts after tool_call_budget calls.
        self.runaway_fix: bool = False
        # Real per-event sleep for the runaway burst (default 0.0 == anyio.sleep(0),
        # an interleave point with no wall time). A small positive value lets the
        # wall-clock budget trip before the tool-call budget in the wall real-path test.
        self.runaway_fix_sleep_s: float = 0.0
        # When True, the test-suite branch emits a Postgres-unreachable signature
        # (infra down, not a code bug) so phase_test_and_heal's
        # is_environmental_failure() short-circuit fires. The heal-fix branch then
        # writes ``.daydream-heal-fix-applied`` -- a sentinel that MUST be absent
        # when the short-circuit aborts before re-entering a fix turn (AC#6b).
        self.environmental_test_failure: bool = False
        # When set, the fix branch APPENDS this text to the fixed TRACKED file
        # (in addition to the sentinels), producing a real tracked-tree change so
        # a test can assert the recommended-change patch captures daydream's edit.
        self.fix_edit_line: str | None = None
        # Runaway knobs mirroring ``runaway_fix`` for the deep-review phases: an
        # unbounded ToolStartEvent burst with no ResultEvent, so run_agent's
        # tool-call budget trips and returns a budget_reason.
        # ``runaway_alternatives``: the wonder turn.
        # ``runaway_stack``: the per-stack review turn for that stack name.
        # ``runaway_parse``: the parse turn for that stack name.
        self.runaway_alternatives: bool = False
        self.runaway_stack: str | None = None
        self.runaway_parse: str | None = None
        # ``runaway_test``: the test-suite turn (a hung suite, never a result).
        self.runaway_test: bool = False
        # When >0, the wonder turn emits this many ToolStartEvents and THEN its
        # normal structured result -- a long but terminating turn, unlike the
        # runaway knobs. Set above a tool-call ceiling to prove the ceiling does
        # (or no longer does) truncate a legitimately exploratory pass.
        self.alternatives_tool_calls: int = 0
        # When True, the alternatives branch raises instead of answering.
        self.fail_alternatives: bool = False
        # When set, the arbiter branch mints a ContinuationToken carrying this
        # session id, so a test can assert the merge call resumes it.
        self.arbiter_session_id: str | None = None
        # When set, the pattern-scanner specialist names this convention, so a
        # test can prove WHICH run produced the on-disk exploration artifacts.
        self.exploration_sentinel: str | None = None
        # When True, the exploration specialists raise instead of answering, so
        # the real pre_scan degrade path runs: specialist_failed -> completed
        # False -> exploration dir materialized without a cache-key.
        self.fail_exploration: bool = False
        # Issue #309 (uncovered-file sweep): knobs for the per-stack review
        # branches' simulated reads and the sweep branch/parse.
        # When True, the per-stack / generic review branch emits a Read
        # ToolStartEvent per file in its scope (except `per_stack_unread`), so
        # analyze_coverage sees per-stack reviewers as having read their own
        # files -- leaving only genuinely-unread files uncovered.
        self.per_stack_emit_reads: bool = False
        # Files in a stack's scope to NOT emit a read for, even when
        # per_stack_emit_reads is on (the uncovered-file-sweep test uses this to
        # leave one diff file unread by every reviewer).
        self.per_stack_unread: frozenset[str] = frozenset()
        # Issue #742: when set, the per-stack parse branch emits these as the
        # declared per-file verdicts in its structured_output
        # (``{"issues": issues, "verdicts": self.parse_declared_verdicts}``),
        # so the orchestrator's ``include_verdicts=True`` parse surfaces them
        # and the clean-verdict gate reconciles them against completed reads.
        # Default ``None`` keeps the existing parse output (no ``verdicts``
        # key), so all current tests are unchanged.
        self.parse_declared_verdicts: list[dict[str, Any]] | None = None
        # When set, the sweep parse branch emits its finding for this file
        # (instead of the default api.py), so stack-uncovered-records.json names
        # the swept file.
        self.sweep_file: str | None = None
        # When True, the uncovered-file-sweep branch returns success without
        # either structured output or a review file -- a backend can return
        # normally while producing nothing (issue #309 finding 7). A successful
        # return must NOT be recorded as completed coverage.
        self.sweep_no_output: bool = False
        # When True, the sweep still returns valid structured output but omits
        # the legacy Markdown review sidecar. Real structured-output backends
        # can behave this way; the host-owned records artifact remains the
        # authoritative finding source.
        self.sweep_no_review_file: bool = False
        # When True, the uncovered-file-sweep branch writes its review output
        # but emits NO Read tool call -- a successful hunk-only review (issue
        # #309 finding 6). The file must be recorded as a completed ATTEMPT
        # ("reviewed (hunks only)"), never as covered.
        self.sweep_no_read: bool = False
        # When True, the uncovered-file-sweep branch raises -- exercising the
        # sweep's fail-open contract.
        self.fail_sweep: bool = False

    @staticmethod
    def _stack_scope_files(prompt: str) -> list[str]:
        """Extract the file list from a per-stack/generic scope instruction.

        ``_stack_scope_instruction`` renders the files comma-joined on the
        ``Assigned files:`` marker line (``  Assigned files: api.py, README.md``),
        so the single line is split on commas.
        """
        m = re.search(r"Assigned files:\s*([^\n]+)", prompt)
        if m is None:
            return []
        return [part.strip() for part in m.group(1).split(",") if part.strip()]

    def _is_runaway(self, prompt: str, pl: str) -> bool:
        """Whether this turn should emit the unbounded budget-tripping burst."""
        if self.runaway_alternatives and (
            "would you have done this differently" in pl or "evaluate the implementation" in pl
        ):
            return True
        stack_match = re.search(r"stack-(\S+?)-review\.md", prompt)
        stack_name = stack_match.group(1) if stack_match else None
        if (
            self.runaway_stack is not None
            and "you are reviewing the" in pl
            and f"you are reviewing the {self.runaway_stack} stack" in pl
        ):
            return True
        if self.runaway_test and "run the project's test suite" in pl:
            return True
        return (
            self.runaway_parse is not None
            and "extract only actionable issues" in pl
            and stack_name == self.runaway_parse
        )

    def _apply_parse_by_stack_override(
        self, prompt: str, issue: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """#232: per-stack override keyed off the review-file path the prompt
        points at (``stack-<name>-review.md``). Lets one stack emit a HIGH
        finding and another a borderline LOW one at DISTINCT locations, so they
        stay uncontested and drive the suppression predicate.

        Shared by the per-stack review branch and the parse-feedback branch so
        the ``parse_by_stack`` knob is honored identically in both (issue #745
        split them; this hoist re-unifies the duplicated body). Returns the
        issue list -- the base issue, optionally with an ``extra`` sibling at the
        same (file, line).
        """
        if self.parse_by_stack is None:
            return [issue]
        sm = re.search(r"stack-(\S+?)-review\.md", prompt)
        if sm is None or sm.group(1) not in self.parse_by_stack:
            return [issue]
        ov = self.parse_by_stack[sm.group(1)]
        issue["severity"] = ov["severity"]
        issue["confidence"] = ov["confidence"]
        issue["file"] = ov.get("file", issue["file"])
        issue["line"] = ov.get("line", issue["line"])
        issue["description"] = ov.get("description", issue["description"])
        issue["evidence"] = f"{issue['file']}:{issue['line']}"
        issue["rationale"] = "stub"
        issues: list[dict[str, Any]] = [issue]
        # #232: an ``extra`` sibling lets ONE stack emit a second finding at the
        # SAME (file, line) as its HIGH finding. Single stack -> uncontested, so
        # only the HIGH one is an arbiter target; the borderline sibling must
        # still reach suppression, which only holds if exclusion is keyed by
        # record identity, not by (file, line).
        extra = ov.get("extra")
        if extra is not None:
            ex_file = extra.get("file", issue["file"])
            ex_line = extra.get("line", issue["line"])
            issues.append(
                {
                    "id": 2,
                    "description": extra.get("description", "extra finding"),
                    "file": ex_file,
                    "line": ex_line,
                    "severity": extra["severity"],
                    "confidence": extra["confidence"],
                    "rationale": "stub",
                    "evidence": f"{ex_file}:{ex_line}",
                }
            )
        return issues

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        call = {
            "cwd": cwd,
            "prompt": prompt,
            "output_schema": output_schema,
            "agents": agents,
            "model": self.model,
            "continuation": continuation,
            "max_turns": max_turns,
            "read_only": read_only,
        }
        self.calls.append(call)
        if self._shared is not None:
            self._shared.append(call)
        pl = prompt.lower()

        if self._is_runaway(prompt, pl):
            # Unbounded burst, never a ResultEvent -- run_agent's tool-call
            # budget is the only thing that ends this stream.
            for n in range(500):
                yield ToolStartEvent(id=f"tc-{n}", name="Bash", input={"command": "find /"})
                await anyio.sleep(0)
            return

        # TTT alternative-review -> structured output. Checked BEFORE intent: the
        # alt prompt embeds the intent summary, defeating a naive substring check.
        if "would you have done this differently" in pl or "evaluate the implementation" in pl:
            if self.fail_alternatives:
                raise RuntimeError("alternatives blew up")
            for n in range(self.alternatives_tool_calls):
                yield ToolStartEvent(id=f"alt-tc-{n}", name="Read", input={"file_path": "api.py"})
                await anyio.sleep(0)
            yield TextEvent(text="")
            yield ResultEvent(
                structured_output={
                    "issues": [
                        {
                            "id": 1,
                            "title": "Inconsistent greeting wording",
                            "description": "'universe' diverges from 'world' in docs",
                            "recommendation": "align copy",
                            "severity": "low",
                            "files": ["api.py", "README.md"],
                        }
                    ]
                },
                continuation=None,
            )
            return

        # Exploration specialists. Each returns its envelope sub-dict as
        # structured_output (keyed into results[name] by _run_specialist) plus a
        # TextEvent of raw JSON the production gate must suppress from the terminal.
        if "you are the **pattern-scanner** specialist" in pl:
            if self.fail_exploration:
                raise RuntimeError("exploration unavailable")
            payload = {
                "conventions": [
                    {
                        "name": self.exploration_sentinel or "OpenAPI First",
                        "description": "openapi.yaml is the HTTP contract",
                        "source": "CLAUDE.md",
                    }
                ],
                "guidelines": [],
            }
            yield TextEvent(text=json.dumps({"conventions": payload["conventions"], "guidelines": []}))
            yield ResultEvent(structured_output=payload, continuation=None)
            return
        if "you are the **dependency-tracer** specialist" in pl:
            if self.fail_exploration:
                raise RuntimeError("exploration unavailable")
            payload = {
                "affected_files": [],
                "dependencies": [
                    {
                        "source": "App.tsx",
                        "target": "api.py",
                        "relationship": self.exploration_sentinel or "calls",
                    }
                ],
            }
            yield TextEvent(text=json.dumps(payload))
            yield ResultEvent(structured_output=payload, continuation=None)
            return
        if "you are the **test-mapper** specialist" in pl:
            payload = {"affected_files": []}
            yield TextEvent(text=json.dumps(payload))
            yield ResultEvent(structured_output=payload, continuation=None)
            return

        # TTT intent phase -> plain text. Discriminator unique to build_intent_prompt.
        if "understand the intent of these changes" in pl:
            # Echo the author's PR description (when build_intent_prompt injected
            # one) into the returned intent summary so it lands verbatim in the
            # on-disk confirmed-intent file (intent_p) and downstream stages —
            # including the fix prompt — read it back.
            summary = "The PR updates greetings across stacks."
            _tag = "<pr_description>\n"
            if _tag in prompt:
                tail = prompt.split(_tag, 1)[1]
                pr_body = tail.split("\n</pr_description>", 1)[0]
                summary += f"\nConfirmed author intent: {pr_body}"
            yield TextEvent(text=summary)
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Issue #309: uncovered-file sweep reviewer. Dispatch marker phrase is
        # unique to build_uncovered_sweep_prompt. Emits a Read of the swept
        # file (so post-run analyze_coverage counts it as read) and writes a
        # review the sweep parse pass turns into a PER_STACK_RECORD_SCHEMA
        # finding for that file. fail_sweep raises to exercise the fail-open
        # contract.
        if "uncovered file sweep" in pl:
            if self.fail_sweep:
                raise RuntimeError("stub: uncovered-file sweep blew up")
            file_match = re.search(r"changed file (\S+) was NOT read", prompt)
            swept_file = file_match.group(1) if file_match else "notes.txt"
            if not self.sweep_no_output and not self.sweep_no_review_file:
                out_match = re.search(r"write your full review to (\S+)", prompt, flags=re.IGNORECASE)
                if out_match is not None:
                    out_path = Path(out_match.group(1).rstrip("."))
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        f"# Review (uncovered)\n\n## Issues\n\n"
                        f"1. [{swept_file}:1] Uncovered-file finding for {swept_file}\n"
                    )
            if not self.sweep_no_read:
                yield ToolStartEvent(
                    id=f"sweep-read-{swept_file}", name="Read", input={"file_path": swept_file}
                )
                yield ToolResultEvent(
                    id=f"sweep-read-{swept_file}", output="sweep read returned", is_error=False
                )
            # Issue #745 (AC4): the sweep reviewer emits UNCOVERED_SWEEP_SCHEMA
            # structured output directly (no parse-uncovered-<n> fork).
            yield TextEvent(text="")
            if self.sweep_no_output:
                # Backend succeeds but produces nothing: no structured output
                # either, so the sweep must not claim coverage (issue #309 f7).
                yield ResultEvent(structured_output=None, continuation=None)
                return
            yield ResultEvent(
                structured_output={
                    "issues": [
                        {
                            "id": 1,
                            "description": f"Sweep finding for {swept_file}",
                            "file": swept_file,
                            "line": 1,
                            "severity": self.parse_severity or "low",
                            "confidence": "MEDIUM",
                            "rationale": "stub",
                            "evidence": f"{swept_file}:1",
                        }
                    ]
                },
                continuation=None,
            )
            return

        # Per-stack review -> write a markdown file + emit done.
        m = re.search(r"you are reviewing the (\S+) stack", pl)
        if m is None:
            m = re.search(r"you are reviewing the (generic-fallback) stack", pl)
        if m is None and "you are the structural reviewer" in pl:
            # Structural meta-stack: same review-file contract, no language label.
            class _M:
                @staticmethod
                def group(_: int) -> str:
                    return "structure"

            m = _M()  # type: ignore[assignment]
        if m is not None:
            if self.per_stack_emit_reads:
                for scope_file in self._stack_scope_files(prompt):
                    if scope_file in self.per_stack_unread:
                        continue
                    yield ToolStartEvent(
                        id=f"read-{scope_file}", name="Read", input={"file_path": scope_file}
                    )
                    # Completed read: a ToolResultEvent paired with the start,
                    # so the sweep's coverage computation counts the file as read.
                    yield ToolResultEvent(
                        id=f"read-{scope_file}", output="file content", is_error=False
                    )
            out_match = re.search(r"write your full review to (\S+)", prompt, flags=re.IGNORECASE)
            if out_match is not None:
                raw = out_match.group(1).rstrip(".")
                out_path = Path(raw)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                stack = m.group(1)
                out_path.write_text(
                    f"# Review ({stack})\n\n## Issues\n\n1. [api.py:1] Sample issue for {stack}\n"
                )
            # Issue #745 (AC4): the per-stack reviewer emits PER_STACK_RECORD_SCHEMA
            # structured output directly (no separate parse-<stack> fork). Build a
            # schema-valid payload with every required issue field.
            issue: dict[str, Any] = {
                "id": 1,
                "description": "Sample issue",
                "file": "api.py",
                "line": 1,
                "severity": self.parse_severity or "medium",
                "confidence": "MEDIUM",
                "rationale": "stub",
                "evidence": "api.py:1",
            }
            issues: list[dict[str, Any]] = self._apply_parse_by_stack_override(
                prompt, issue
            )
            yield TextEvent(text="")
            yield ResultEvent(
                structured_output={
                    "issues": issues,
                    "verdicts": self.parse_declared_verdicts or [],
                },
                continuation=None,
            )
            return

        if "extract only actionable issues" in pl:  # phase_parse_feedback
            # Evidence gate (#227): every parsed finding carries a grounded
            # citation so it survives _is_evidenced downstream (structural items
            # and the tiny-diff bypass route these records straight to the gate).
            parse_issue: dict[str, Any] = {
                "id": 1,
                "description": "Sample issue",
                "file": "api.py",
                "line": 1,
                "evidence": "api.py:1",
            }
            # Deep per-stack parse requests severity (PER_STACK_RECORD_SCHEMA,
            # #168); every stack -- language or the structural meta-stack
            # (#314) -- parses with this schema, so all emit the configured
            # severity so a test can drive arbiter selection.
            if self.parse_severity is not None and "severity" in pl:
                parse_issue["severity"] = self.parse_severity
                parse_issue["confidence"] = "MEDIUM"
                parse_issue["rationale"] = "stub"
            # Issue #309: the uncovered-sweep parse (prompt points at an
            # `uncovered-<n>-review.md`) emits its finding for the swept file.
            if self.sweep_file is not None and re.search(r"uncovered-\d+-review\.md", prompt):
                parse_issue["file"] = self.sweep_file
                parse_issue["line"] = 1
                parse_issue["description"] = f"Sweep finding for {self.sweep_file}"
                parse_issue["evidence"] = f"{self.sweep_file}:1"
                if "severity" in pl:
                    parse_issue["severity"] = self.parse_severity or "low"
                    parse_issue["confidence"] = "MEDIUM"
                    parse_issue["rationale"] = "stub"
            # #232: per-stack override keyed off the review-file path the parse
            # prompt points at (``stack-<name>-review.md``). Lets one stack emit a
            # HIGH finding and another a borderline LOW one at DISTINCT locations,
            # so they stay uncontested and drive the suppression predicate.
            parse_issues: list[dict[str, Any]] = self._apply_parse_by_stack_override(
                prompt, parse_issue
            )
            yield TextEvent(text="")
            # Issue #742: PER_STACK_RECORD_SCHEMA requires a ``verdicts``
            # property (Codex strict-mode output schemas list every key in
            # ``required``), so the parse payload always carries the key --
            # declared per-file verdicts when the knob is set, else empty.
            parse_payload: dict[str, Any] = {
                "issues": parse_issues,
                "verdicts": self.parse_declared_verdicts or [],
            }
            yield ResultEvent(structured_output=parse_payload, continuation=None)
            return

        # Scoped Opus arbiter (#168). Reads the arbiter-input.json path the prompt
        # points at, echoes every arb_id back with keep=true, and stamps the
        # description so the arbitrated finding is observable downstream.
        if "you are the arbiter" in pl:
            in_match = re.search(r"listed in (\S+arbiter-input\.json)", prompt)
            findings: list[dict[str, Any]] = []
            if in_match is not None and not self.arbiter_omit_verdicts:
                arb_inputs = json.loads(Path(in_match.group(1)).read_text())
                for entry in arb_inputs:
                    findings.append(
                        {
                            "arb_id": entry["arb_id"],
                            "keep": True,
                            "severity": entry.get("severity") or "high",
                            "confidence": entry.get("confidence") or "HIGH",
                            "description": f"ARBITRATED: {entry.get('description')}",
                            "rationale": "arbiter second opinion",
                        }
                    )
            yield TextEvent(text="")
            yield ResultEvent(
                structured_output={"findings": findings},
                continuation=(
                    ContinuationToken(
                        backend="claude", data={"session_id": self.arbiter_session_id}
                    )
                    if self.arbiter_session_id
                    else None
                ),
            )
            return

        # Precision-mode suppression reviewer (#232). Reads suppression-input.json,
        # echoes every sup_id back with keep=self.suppression_keep. keep=False drops
        # the borderline finding (fail-closed apply); keep=True with cited evidence
        # retains it. Dispatch phrase must not collide with the arbiter or merge
        # branches above/below.
        if "you are the suppression reviewer" in pl:
            in_match = re.search(r"listed in (\S+suppression-input\.json)", prompt)
            sup_findings: list[dict[str, Any]] = []
            if in_match is not None:
                sup_inputs = json.loads(Path(in_match.group(1)).read_text())
                for entry in sup_inputs:
                    sup_findings.append(
                        {
                            "sup_id": entry["sup_id"],
                            "keep": self.suppression_keep,
                            "severity": entry.get("severity") or "low",
                            "confidence": entry.get("confidence") or "LOW",
                            "description": entry.get("description") or "finding",
                            "rationale": (
                                "confirmed by code" if self.suppression_keep else "no confirming evidence"
                            ),
                            "evidence": entry.get("evidence") or "",
                        }
                    )
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"findings": sup_findings}, continuation=None)
            return

        # Cross-stack merge -> schema-validated item list; the host appends
        # structural findings, normalizes ids, and renders review-output.md.
        if "cross-stack merge agent" in pl:
            yield TextEvent(text="")
            if self.merge_emit_str is not None:
                yield TextEvent(text=self.merge_emit_str)
                yield ResultEvent(structured_output=None, continuation=None)
                return
            if self.merge_emit_bare_list is not None:
                yield ResultEvent(structured_output=self.merge_emit_bare_list, continuation=None)
                return
            if self.merge_echo_records:
                # Echo the on-disk per-stack records as merged items so the
                # rendered artifact reflects any arbiter revisions (#168).
                # Issue #742: fresh-run records files carry the dict shape
                # {"issues": [...], "verdicts": [...]}; normalize to the
                # bare issues list (legacy files stay bare lists).
                echoed: list[dict[str, Any]] = []
                next_id = 1
                for path_str in re.findall(r"  - (\S+-records\.json)", prompt):
                    loaded = json.loads(Path(path_str).read_text())
                    recs = _records_issues_or_empty(loaded)
                    for rec in recs:
                        echoed.append(
                            {
                                "id": next_id,
                                "lens": "per-stack",
                                "file": rec.get("file"),
                                "line": rec.get("line"),
                                "severity": rec.get("severity", "medium"),
                                "description": rec.get("description"),
                                "confidence": rec.get("confidence", "MEDIUM"),
                                "rationale": rec.get("rationale", "rationale"),
                                "evidence": rec.get("evidence", "api.py:1"),
                            }
                        )
                        next_id += 1
                yield ResultEvent(structured_output={"items": echoed}, continuation=None)
                return
            if self.merge_items is not None:
                yield ResultEvent(
                    structured_output={"items": self.merge_items},
                    continuation=None,
                )
                return
            yield ResultEvent(
                structured_output={
                    "items": [
                        {
                            "id": 1,
                            "lens": "per-stack",
                            "file": "api.py",
                            "line": 1,
                            "severity": "medium",
                            "description": "Python issue",
                            "confidence": "MEDIUM",
                            "rationale": "rationale",
                            "evidence": "api.py:1",
                        },
                        {
                            "id": 2,
                            "lens": "per-stack",
                            "file": "App.tsx",
                            "line": 1,
                            "severity": "medium",
                            "description": "React issue",
                            "confidence": "MEDIUM",
                            "rationale": "rationale",
                            "evidence": "App.tsx:1",
                        },
                        {
                            "id": 3,
                            "lens": "cross-stack",
                            "file": "api.py",
                            "line": 1,
                            "severity": "high",
                            "description": "Contract drift between Python handler and React caller",
                            "confidence": "HIGH",
                            "rationale": "rationale",
                            "evidence": "api.py:1",
                        },
                    ]
                },
                continuation=None,
            )
            return

        if "supervisor adjudication" in pl:
            in_match = re.search(r"listed in (\S+supervise-input\.json)", prompt)
            input_items = json.loads(Path(in_match.group(1)).read_text()) if in_match else []
            verdicts = []
            for item in input_items:
                verdict = (self.supervise_verdicts or {}).get(item["id"])
                if verdict is not None:
                    verdicts.append({"id": item["id"], **verdict})
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"verdicts": verdicts}, continuation=None)
            return

        # phase_fix -> "apply" the edit by writing a sentinel file, the observable
        # consequence the --yes real-path test asserts the fix gate auto-approved.
        if pl.startswith("fix this issue") or pl.startswith("fix these"):
            if self.runaway_fix:
                # Emit a long burst of tool calls and NEVER a ResultEvent. A
                # generator that never returns models the 1.5-5h time-tail the
                # tool-call budget exists to cut; the budget breaks the loop.
                for n in range(500):
                    yield ToolStartEvent(id=f"tc-{n}", name="Bash", input={"command": "find /"})
                    await anyio.sleep(self.runaway_fix_sleep_s)
                return
            # Single-finding prompts carry "File: <path>"; batched prompts name the
            # one target file in their "Fix these N issues in <path>:" header.
            m = re.search(r"^File: (.+)$", prompt, re.M)
            batched_hdr = re.search(r"^Fix these \d+ issues in (.+):$", prompt, re.M)
            if m is None:
                m = batched_hdr
            fixed_file = m.group(1).strip() if m else "unknown"
            # phase_fix emits an absolute path when the file exists on disk; the stub keys fixes by basename.
            fixed_name = Path(fixed_file).name
            # Runaway ONLY the batched turn for the marked file: burn real wall so
            # run_agent's per-invocation wall budget trips, returns a budget_reason,
            # and phase_fix_batched raises into the per-finding fallback (#201).
            if (
                self.runaway_batched_fix_file is not None
                and batched_hdr is not None
                and fixed_name == self.runaway_batched_fix_file
            ):
                for n in range(500):
                    yield ToolStartEvent(id=f"btc-{n}", name="Bash", input={"command": "find /"})
                    await anyio.sleep(self.runaway_batched_sleep_s)
                return
            # Fail ONLY the batched turn for the marked file so the group falls
            # back to per-finding fixes (the #186 pattern under budget test).
            if (
                self.fail_batched_fix_file is not None
                and batched_hdr is not None
                and fixed_name == self.fail_batched_fix_file
            ):
                raise RuntimeError(f"stub: batched fix failure for {fixed_name}")
            if self.fix_partial_then_maxturns is not None and fixed_name == self.fix_partial_then_maxturns:
                edit_target = Path(fixed_file) if Path(fixed_file).is_absolute() else (cwd / fixed_file)
                edit_target.write_text(PARTIAL_FIX_MARKER)
                if self.fix_orphan_file is not None:
                    orphan = cwd / self.fix_orphan_file
                    orphan.parent.mkdir(parents=True, exist_ok=True)
                    orphan.write_text("// stray file from a dead fix agent\n")
                raise MaxTurnsError(f"stub: max turns exhausted mid-fix for {fixed_name}")
            if self.fix_fail_file is not None and fixed_name == self.fix_fail_file:
                raise RuntimeError(f"stub fix failure for {fixed_name}")
            if max_turns is not None and self.fix_turns_needed > max_turns:
                raise MaxTurnsError(
                    f"stub: fix for {fixed_name} needs {self.fix_turns_needed} turns, capped at {max_turns}"
                )
            if self.deferred_write_pairs is not None:
                for index, path in enumerate(self.deferred_write_pairs, start=1):
                    edit_target = Path(path) if Path(path).is_absolute() else cwd / path
                    yield ToolStartEvent(
                        id=f"deferred-write-{index}",
                        name="Write",
                        input={"file_path": str(edit_target), "content": "backend resumed"},
                    )
                    edit_target.write_text("backend resumed")
                yield TextEvent(text="Applied the deferred writes.")
                yield ResultEvent(structured_output=None, continuation=None)
                return
            (cwd / ".daydream-fix-applied").write_text("applied\n")  # legacy sentinel
            (cwd / f".fixed-{fixed_name.replace('.', '_')}").write_text("applied\n")
            if self.fix_new_generated is not None:
                generated = cwd / self.fix_new_generated
                generated.parent.mkdir(parents=True, exist_ok=True)
                generated.write_text("-- new migration\n")
            if self.fix_edit_line is not None:
                edit_target = Path(fixed_file) if Path(fixed_file).is_absolute() else (cwd / fixed_file)
                if edit_target.exists():
                    edit_target.write_text(edit_target.read_text() + self.fix_edit_line)
            if self.fix_append_path is not None and fixed_name == self.fix_append_path.name:
                # A batched fix turn addresses EVERY finding it is handed, so append
                # each marker the prompt names (in prompt/severity order), not just
                # the first. A single-finding prompt names exactly one marker.
                toks = re.findall(r"marker-\d+", prompt) or ["?"]
                cur = self.fix_append_path.read_text() if self.fix_append_path.exists() else ""
                await anyio.sleep(0)  # deterministic interleave point
                self.fix_append_path.write_text(cur + "".join(t + "\n" for t in toks))
            yield TextEvent(text="Applied the fix.")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Recommendation verifier (#83). Discriminator is the verifier's role
        # sentence — the schema dump it used to key off was removed from the
        # prompt (the schema reaches backends via output_schema). The stub only
        # emits a well-formed payload; the phase persists it to
        # recommendation-verdicts.json itself.
        if "you are the recommendation-verifier agent" in pl:
            yield TextEvent(text="")
            yield ResultEvent(
                structured_output={
                    "verdicts": [
                        {
                            "issue_id": 1,
                            "verdict": self.verifier_verdict,
                            "evidence": "stub",
                            "unverified_assumptions": list(self.verifier_unverified_assumptions),
                        }
                    ]
                },
                continuation=None,
            )
            return

        # Post-fix fix-verifier (issue #744). Discriminator is the role sentence
        # of build_fix_verify_prompt. Returns one verdict per dispatched finding
        # id rendered in the prompt (default resolved), honoring the
        # fix_verify_verdicts override map. Read-only enforcement is pinned:
        # when fix_verify_requires_read_only and the call arrived
        # read_only=False, raise (the phase must ALWAYS pass read_only=True).
        if "post-fix fix-verifier agent" in pl:
            if self.fix_verify_requires_read_only and not read_only:
                raise AssertionError("fix-verify turn must arrive read_only=True")
            round_match = re.search(r"Round (\d+) of up to 3 check passes", prompt)
            round_num = int(round_match.group(1)) if round_match else 1
            ids = [int(i) for i in re.findall(r"(?m)^(\d+)\. \[", prompt)]
            verdicts = []
            for i in ids:
                override = (self.fix_verify_verdicts or {}).get(i)
                if override is not None and round_num == 1:
                    verdicts.append(dict(override))
                elif round_num < self.fix_verify_resolve_after_round:
                    verdicts.append({"issue_id": i, "verdict": "unresolved", "reason": "stub: round still fumbling"})
                else:
                    verdicts.append({"issue_id": i, "verdict": "resolved", "reason": "stub"})
            yield TextEvent(text="")
            yield ResultEvent(
                structured_output={"verdicts": verdicts},
                continuation=None,
            )
            return

        # Heal-loop fix turn (prompt starts with "The tests failed."). Writes a
        # distinct sentinel so a test can assert the heal loop DID re-enter a fix
        # turn -- and, by its ABSENCE, that the environmental short-circuit aborted
        # before any fix turn ran (AC#6b).
        if pl.startswith("the tests failed"):
            (cwd / ".daydream-heal-fix-applied").write_text("healed\n")
            if self.heal_fix_generated is not None:
                generated = cwd / self.heal_fix_generated
                if generated.exists():
                    generated.write_text(generated.read_text() + "\n-- FORBIDDEN HEAL EDIT\n")
            if self.heal_fix_new_generated is not None:
                new_generated = cwd / self.heal_fix_new_generated
                new_generated.parent.mkdir(parents=True, exist_ok=True)
                new_generated.write_text("-- new healing migration\n")
            yield TextEvent(text="Attempted to fix the test failures.")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Test-and-heal run. The prompt is constant, so a call counter drives the
        # result: with fail_first_test_run set, the FIRST run fails (heal loop
        # reaches choice "2") and subsequent runs pass. With
        # environmental_test_failure set, every run emits a Postgres-unreachable
        # signature -- detect_test_success() is False AND is_environmental_failure()
        # is True, so the heal loop must abort before a fix turn.
        if "run the project's test suite" in pl:
            self.test_suite_calls += 1
            if self.environmental_test_failure:
                yield TextEvent(
                    text=(
                        "could not connect to server: Connection refused\n"
                        "\tIs the server running on host localhost (127.0.0.1) "
                        "and accepting TCP/IP connections on port 5432?\n"
                        "The dev Postgres container is not running."
                    )
                )
            elif self.fail_all_test_runs:
                yield TextEvent(text="1 failed, 0 passed")
            elif self.fail_first_test_run and self.test_suite_calls == 1:
                yield TextEvent(text="1 failed, 0 passed")
            else:
                yield TextEvent(text="2 passed, 0 failed")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Default: empty.
        yield TextEvent(text="")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass


def silence(monkeypatch: pytest.MonkeyPatch, *, prompts: bool = True) -> None:
    """Silence noise-only UI helpers in deep orchestrator + phases.

    ``prompts=False`` leaves the real ``prompt_user`` in place at both seams, for
    tests that drive a genuine gate.
    """
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_verification_summary", lambda *a, **kw: None)
    if prompts:
        monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")
        # resolve_or_prompt routes through agent.prompt_user; patch it too so those
        # gates don't block on stdin.
        monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "n")


def force_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the run's interactivity axis to interactive for prompt-path tests.

    ``runner.run`` now auto-resolves non-interactive from a non-TTY stdin or a
    truthy ``CI`` env var (Task 4). Under pytest, stdin is not a TTY (and ``CI``
    is set in CI), so a test that drives the REAL interactive prompt path must
    explicitly establish a TTY stdin and unset ``CI`` -- otherwise the gate
    short-circuits to its safe default and the interactive branch never runs.
    """
    monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)


def install_stub_backend(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    *,
    pin_skill_availability: bool = True,
    enable_exploration: bool = False,
) -> StubBackend:
    """Patch create_backend to return a single stub backend instance.

    Args:
        pin_skill_availability: When True (default), disables the exploration
            pre-scan so it doesn't add unexpected backend calls. Pass False when
            a test wants to leave ``EXPLORATION_AVAILABLE`` at its module value.
            Kept for call-site compatibility; stack detection is now
            registry-independent, so there is no skill-availability gate to pin.
        enable_exploration: When True, leaves ``EXPLORATION_AVAILABLE`` True so
            the real ``pre_scan`` branch runs and the stub answers the
            specialist prompts. Default False preserves the existing behavior
            (exploration disabled) that the rest of the suite relies on.
    """
    stub = StubBackend(target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    if pin_skill_availability:
        if enable_exploration:
            # Pin True so the pre_scan branch runs regardless of ambient module state.
            monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", True)
        else:
            # Disable exploration pre-scan so it doesn't add extra backend calls.
            monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)
    return stub
