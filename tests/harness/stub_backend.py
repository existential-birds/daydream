"""Prompt-dispatching backend and UI helpers for deep-pipeline integration tests."""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
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
from daydream.deep.records import record_issues_or_empty, record_uid, stack_name_from_uid
from tests.harness.review_result import merge_result

PARTIAL_FIX_MARKER = "// PARTIAL BROKEN EDIT -- max turns exhausted mid-fix\n"


class _StubRetryableError(RuntimeError):
    """Default transport-shaped failure for ``fix_retryable_failures``."""

    retryable = True


class StubBackend:
    """Emit realistic review/fix results and record calls for pipeline assertions."""

    model = "mock-model"

    def __init__(self, target: Path, *, model: str = "mock-model", shared_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self._target = target
        # Backend fan-out hint; tests override it for deterministic group ordering.
        self.fanout_concurrency: int = 4
        # Optional model-tagged call log shared across cached backend instances.
        self._shared = shared_calls
        # Parse severity selects arbiter targets; merge_echo_records exposes revisions.
        self.parse_severity: str | None = None
        self.merge_echo_records: bool = False
        # Omit all arbiter verdicts to exercise fail-open retention of selected records.
        self.arbiter_omit_verdicts: bool = False
        # Fail only this shard (e.g. arbiter-group-1) for fail-open/resume tests.
        self.arbiter_fail_group: str | None = None
        # Per-stack findings can vary file/severity/confidence to control suppression
        # selection. suppression_keep applies to every suppression-review verdict.
        self.parse_by_stack: dict[str, dict[str, Any]] | None = None
        self.suppression_keep: bool = True
        self.calls: list[dict[str, Any]] = []
        # Recommendation verdict for issue 1; "contradicts" changes the fix prompt.
        self.verifier_verdict: str = "consistent"
        self.verifier_unverified_assumptions: list[str] = []
        # Per-finding verdict overrides apply only on round 1. Otherwise findings
        # resolve at fix_verify_resolve_after_round and remain unresolved before it.
        self.fix_verify_verdicts: dict[int, dict[str, Any]] | None = None
        self.fix_verify_resolve_after_round: int = 1
        # Test invocation count supports first-run failures followed by healing.
        self.test_suite_calls: int = 0
        # Fail the first test call, then pass.
        self.fail_first_test_run: bool = False
        # Fail every test call.
        self.fail_all_test_runs: bool = False
        # Override for the merge agent's item list (None -> default three-item payload).
        self.merge_items: list[dict[str, Any]] | None = None
        # Emit unstructured prose to exercise merge salvage.
        self.merge_emit_str: str | None = None
        # Emit a bare item list instead of the {"items": [...]} envelope.
        self.merge_emit_bare_list: list[dict[str, Any]] | None = None
        # Optional LLM supervisor verdicts keyed by canonical item id.
        self.supervise_verdicts: dict[int, dict[str, Any]] | None = None
        # Optional deferred Write tool pairs for the built-in tool-supervisor tests.
        self.deferred_write_pairs: list[str] | None = None
        # Append prompt markers with a read/write interleave: per-file serialization
        # must preserve order, while per-item fan-out would lose or reorder appends.
        self.fix_append_path: Path | None = None
        # Fail this file's fix while allowing sibling groups to finish.
        self.fix_fail_file: str | None = None
        # Fail this file's batched turn; its per-finding fallback turns succeed.
        self.fail_batched_fix_file: str | None = None
        # Run away only during this file's batch. The real run_agent wall timeout
        # raises into per-finding fallback, which shares the time already consumed.
        # Events sleep runaway_batched_sleep_s or advance the injected clock.
        self.runaway_batched_fix_file: str | None = None
        self.runaway_batched_sleep_s: float = 0.05
        # Run away only for a single-item turn naming this file, testing a group
        # deadline without a fallback loop.
        self.runaway_single_fix_file: str | None = None
        # Yield no events and sleep(0) until this gate opens, so a sibling can
        # finish deterministically before the runaway begins.
        self.runaway_gate: Callable[[], bool] | None = None
        # Completed fix basenames, recorded after ResultEvent consumption. Unlike
        # sentinel files, this evidence survives the fix-footprint guard.
        self.completed_fix_files: list[str] = []
        # Charge clock_advance_per_event_s through the injected clock instead of
        # runaway sleeps; never both. Only run_agent enforces deadlines.
        self.clock_advance: Callable[[float], None] | None = None
        self.clock_advance_per_event_s: float = 0.0
        # Fail each file's first N fix turns with fix_retryable_error (or the
        # default). Each failure charges one clock step before raising; later turns
        # succeed, unless the real group deadline cuts the retry ladder first.
        self.fix_retryable_failures: int = 0
        self.fix_retryable_error: Exception | None = None
        # Restrict retry failures to this basename; sibling files succeed at once.
        self.fix_retryable_file: str | None = None
        self._fix_retry_counts: dict[str, int] = {}
        # Write a broken partial edit, then raise MaxTurnsError. The orchestrator
        # must restore the file and save a recovery patch.
        self.fix_partial_then_maxturns: str | None = None
        # Required fix turns; a lower max_turns raises before applying the edit.
        self.fix_turns_needed: int = 0
        # Stray untracked path created by the failing group, outside its key file;
        # it survives tree protection and must appear in fix_leftover_untracked.
        self.fix_orphan_file: str | None = None
        # Existing untracked user file damaged by the same failed turn.  The
        # run-wide terminal guard must restore its exact pre-run bytes.
        self.fix_damage_protected_file: str | None = None
        # Repo-relative generated path created by a successful fix turn.
        self.fix_new_generated: str | None = None
        # Repo-relative historical generated path modified by a test-healing
        # fix turn, plus an optional new generated path created by that turn.
        self.heal_fix_generated: str | None = None
        self.heal_fix_new_generated: str | None = None
        # Repo-relative tracked path outside the reviewed diff edited by a
        # test-healing turn, with the line it appends. Exercises the early
        # full confinement that must run before the suite is rerun.
        self.heal_fix_unauthorized: str | None = None
        self.heal_fix_unauthorized_line: str = "\n# unauthorized healing edit\n"
        # Emit a runaway ToolStartEvent burst without a result to exercise budgets.
        self.runaway_fix: bool = False
        # Runaway pacing; zero still yields via sleep(0). Positive values can trip
        # the wall limit before the tool limit. An injected clock replaces sleeps.
        self.runaway_fix_sleep_s: float = 0.0
        # Emit a Postgres-unreachable test failure. The environmental short-circuit
        # must abort before healing writes .daydream-heal-fix-applied.
        self.environmental_test_failure: bool = False
        # Append a real tracked edit so recommended-change patches can capture it.
        self.fix_edit_line: str | None = None
        # Runaway wonder or named per-stack review: tool events without a result.
        self.runaway_alternatives: bool = False
        self.runaway_stack: str | None = None
        # Runaway test-suite turn.
        self.runaway_test: bool = False
        # Emit this many tool calls before a normal wonder result, testing whether
        # a tool-call ceiling truncates a long but terminating review.
        self.alternatives_tool_calls: int = 0
        # When True, the alternatives branch raises instead of answering.
        self.fail_alternatives: bool = False
        # Arbiter continuation session that merge can resume.
        self.arbiter_session_id: str | None = None
        # Convention name identifying the run that wrote exploration artifacts.
        self.exploration_sentinel: str | None = None
        # Fail exploration specialists: pre_scan must degrade to incomplete
        # artifacts without a cache key.
        self.fail_exploration: bool = False
        # Paired source reads let recovery tests exercise real review evidence.
        self.per_stack_emit_reads: bool = False
        # Per-kind author/repair spec queue; repeat its last entry when exhausted.
        # An absent queue returns an empty spec, which grounds to an omission.
        self.diagram_specs: dict[str, list[dict[str, Any]]] = {}
        # Author continuation required for a repair turn to run.
        self.diagram_session_id: str | None = None
        # Diagram kinds whose author turn raises (the fail-open/exit-1 knob).
        self.diagram_fail: frozenset[str] = frozenset()
        # Turn counter per kind, so the repair turn reads the next queued spec.
        self.diagram_turns: dict[str, int] = {}
        # Force INLINE sanctioned-input transport independently of read_only.
        self.sandbox: bool = False

    @staticmethod
    def _prompt_record_uid_groups(prompt: str) -> list[list[str]]:
        """Read nonempty UID groups from prompted records files in prompt order.

        Structural records are host-appended and absent from the merge prompt.
        Grouping by file supports both per-stack and cross-stack provenance.
        Missing or malformed files contribute nothing, allowing phase tests to
        use unwritten prompt paths.
        """

        groups: list[list[str]] = []
        for path_str in re.findall(r"  - (\S+-records\.json)", prompt):
            try:
                loaded = json.loads(Path(path_str).read_text())
            except (OSError, json.JSONDecodeError):
                continue
            uids = [uid
                for rec in record_issues_or_empty(loaded)
                if isinstance(rec, dict) and (uid := record_uid(rec))
            ]
            if uids:
                groups.append(uids)
        return groups

    @staticmethod
    def _diagram_dispatch(pl: str) -> tuple[str, bool] | None:
        """Return (kind, is_repair) from stable role/repair prompt openers."""
        for kind in ("sequence", "flowchart"):
            if f"diagram repair turn ({kind})" in pl:
                return kind, True
        if "you are the sequence-diagram author for this pull request." in pl:
            return "sequence", False
        if "you are the flowchart author for this pull request." in pl:
            return "flowchart", False
        return None


    @staticmethod
    def _stack_scope_files(prompt: str) -> list[str]:
        """Split the comma-separated Assigned files marker in a scope instruction."""
        m = re.search(r"Assigned files:\s*([^\n]+)", prompt)
        if m is None:
            return []
        return [part.strip() for part in m.group(1).split(",") if part.strip()]

    def _tick(self) -> None:
        """Charge one configured event step to the injected clock, if present."""
        if self.clock_advance is not None:
            self.clock_advance(self.clock_advance_per_event_s)

    def _is_runaway(self, prompt: str, pl: str) -> bool:
        """Whether this turn should emit the unbounded budget-tripping burst."""
        if self.runaway_alternatives and (
            "would you have done this differently" in pl or "evaluate the implementation" in pl
        ):
            return True
        if (self.runaway_stack is not None
            and "you are reviewing the" in pl
            and f"you are reviewing the {self.runaway_stack} stack" in pl
        ):
            return True
        if self.runaway_test and "run the project's test suite" in pl:
            return True
        return False

    def _apply_parse_by_stack_override(self, prompt: str, issue: dict[str, Any]) -> list[dict[str, Any]]:
        """Apply the named stack's override and optional same-location extra issue.

        Review and parse dispatch share this contract; native scope instructions
        and legacy stack-<name>-review.md paths identify the stack.
        """
        if self.parse_by_stack is None:
            return [issue]
        sm = (re.search(r"you are reviewing the (\S+) stack", prompt, re.IGNORECASE)
              or re.search(r"stack-(\S+?)-review\.md", prompt))
        scope = sm.group(1) if sm is not None else (
            "structure" if "you are the structural reviewer" in prompt.lower() else None
        )
        if scope == "generic-fallback":
            scope = "generic"
        if scope not in self.parse_by_stack:
            return [issue]
        ov = self.parse_by_stack[scope]
        payload = ov.get("issue")
        if isinstance(payload, dict):
            issue.update(payload)
        issue["severity"] = ov["severity"]
        issue["confidence"] = ov["confidence"]
        issue["file"] = ov.get("file", issue["file"])
        issue["line"] = ov.get("line", issue["line"])
        issue["description"] = ov.get("description", issue["description"])
        issue["evidence"] = f"{issue['file']}:{issue['line']}"
        issue["rationale"] = "stub"
        issues: list[dict[str, Any]] = [issue]
        # A same-location sibling tests record-identity exclusion: selecting the
        # HIGH finding for arbitration must not exclude its borderline sibling
        # from suppression.
        extra = ov.get("extra")
        if extra is not None:
            ex_file = extra.get("file", issue["file"])
            ex_line = extra.get("line", issue["line"])
            issues.append({
                    "id": 2, "description": extra.get("description", "extra finding"), "file": ex_file, "line": ex_line,
                    "severity": extra["severity"], "confidence": extra["confidence"], "rationale": "stub",
                    "evidence": f"{ex_file}:{ex_line}",
                }
            )
        return issues

    async def _runaway_burst(self, prefix: str, sleep_s: float) -> AsyncIterator[AgentEvent]:
        """Emit 500 tool calls without a result, advancing the injected clock or sleeping."""
        for n in range(500):
            yield ToolStartEvent(id=f"{prefix}-{n}", name="Bash", input={"command": "find /"})
            if self.clock_advance is not None:
                self._tick()
            else:
                await anyio.sleep(sleep_s)

    async def execute(
        self, cwd: Path, prompt: str, output_schema: Any = None, continuation: Any = None, agents: Any = None,
        max_turns: Any = None, read_only: bool = False, persist_session: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        call = {"cwd": cwd, "prompt": prompt, "output_schema": output_schema, "agents": agents, "model": self.model,
            "continuation": continuation, "max_turns": max_turns, "read_only": read_only,
            "persist_session": persist_session,
        }
        self.calls.append(call)
        if self._shared is not None:
            self._shared.append(call)
        pl = prompt.lower()

        if self._is_runaway(prompt, pl):
            # No ResultEvent: run_agent must terminate this burst through its budgets.
            async for event in self._runaway_burst("tc", 0):
                yield event
            return

        # Check alternatives before intent: their prompt embeds the intent summary.
        if "would you have done this differently" in pl or "evaluate the implementation" in pl:
            if self.fail_alternatives:
                raise RuntimeError("alternatives blew up")
            for n in range(self.alternatives_tool_calls):
                yield ToolStartEvent(id=f"alt-tc-{n}", name="Read", input={"file_path": "api.py"})
                await anyio.sleep(0)
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"issues": [{"id": 1, "title": "Inconsistent greeting wording",
                            "description": "'universe' diverges from 'world' in docs", "recommendation": "align copy",
                            "severity": "low", "files": ["api.py", "README.md"],
                            "confidence": "MEDIUM", "rationale": "repository copy diverges",
                            "evidence": "api.py:1 and README.md:1 use different greetings",
                        }
                    ]
                }, continuation=None,
            )
            return

        # Return the queued diagram spec and optional repair continuation.
        dispatch = self._diagram_dispatch(pl)
        if dispatch is not None:
            kind, is_repair = dispatch
            if kind in self.diagram_fail and not is_repair:
                raise RuntimeError(f"stub: diagram author for {kind} blew up")
            turn = self.diagram_turns.get(kind, 0)
            self.diagram_turns[kind] = turn + 1
            queue = self.diagram_specs.get(kind) or []
            if queue:
                spec = queue[min(turn, len(queue) - 1)]
            elif kind == "sequence":
                spec = {"participants": [], "messages": [], "blocks": []}
            else:
                spec = {"root": None, "nodes": [], "edges": []}
            yield TextEvent(text="")
            yield ResultEvent(structured_output=spec,
                continuation=(ContinuationToken(backend="claude", data={"session_id": self.diagram_session_id})
                    if self.diagram_session_id
                    else None
                ),
            )
            return

        # Specialists emit structured payloads plus raw JSON text, which the
        # production display gate must suppress.
        if "you are the **pattern-scanner** specialist" in pl:
            if self.fail_exploration:
                raise RuntimeError("exploration unavailable")
            payload = {"conventions": [{"name": self.exploration_sentinel or "OpenAPI First",
                        "description": "openapi.yaml is the HTTP contract", "source": "CLAUDE.md",
                    }
                ], "guidelines": [],
            }
            yield TextEvent(text=json.dumps({"conventions": payload["conventions"], "guidelines": []}))
            yield ResultEvent(structured_output=payload, continuation=None)
            return
        if "you are the **dependency-tracer** specialist" in pl:
            if self.fail_exploration:
                raise RuntimeError("exploration unavailable")
            payload = {"affected_files": [],
                "dependencies": [
                    {"source": "App.tsx", "target": "api.py", "relationship": self.exploration_sentinel or "calls"}
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
            # Echo the PR description into persisted intent for downstream prompts.
            summary = "The PR updates greetings across stacks."
            _tag = "<pr_description>\n"
            if _tag in prompt:
                tail = prompt.split(_tag, 1)[1]
                pr_body = tail.split("\n</pr_description>", 1)[0]
                summary += f"\nConfirmed author intent: {pr_body}"
            yield TextEvent(text=summary)
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Per-stack review -> write a markdown file + emit done.
        stack_match = re.search(r"you are reviewing the (\S+) stack", pl)
        stack_label = stack_match.group(1) if stack_match else None
        if stack_label == "generic-fallback":
            stack_label = "generic"
        if stack_label is None and "you are the structural reviewer" in pl:
            stack_label = "structure"
        if stack_label is not None:
            if self.per_stack_emit_reads:
                scope_files = self._stack_scope_files(prompt)
                for scope_file in scope_files:
                    yield ToolStartEvent(id=f"read-{scope_file}", name="Read", input={"file_path": scope_file})
                    # Budget recovery retains only completed source reads.
                    yield ToolResultEvent(id=f"read-{scope_file}", output="file content", is_error=False)
            out_match = re.search(r"write your full review to (\S+)", prompt, flags=re.IGNORECASE)
            if out_match is not None:
                raw = out_match.group(1).rstrip(".")
                out_path = Path(raw)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    f"# Review ({stack_label})\n\n## Issues\n\n1. [api.py:1] Sample issue for {stack_label}\n"
                )
            # Reviewers emit schema-valid records directly. Structural findings use a
            # distinct description so host dedup does not fold them into language
            # findings and erase the structural-lens test case.
            issue: dict[str, Any] = {"id": 1,
                "description": ("Structural maintainability concern"
                    if stack_label == "structure"
                    else "Sample issue"
                ), "file": "api.py", "line": 1, "severity": self.parse_severity or "medium", "confidence": "MEDIUM",
                "rationale": "stub", "evidence": "api.py:1",
            }
            issues: list[dict[str, Any]] = self._apply_parse_by_stack_override(prompt, issue)
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"issues": issues}, continuation=None,)
            return

        # Echo arbiter IDs with keep=True; stamp descriptions so revisions remain
        # observable in downstream artifacts.
        if "you are the arbiter" in pl:
            # Read either the unsharded or group-specific arbiter input path.
            in_match = re.search(r"listed in (\S*arbiter[-\w]*input\.json)", prompt)
            if self.arbiter_fail_group is not None and in_match is not None:
                group_match = re.search(r"(arbiter-group-\d+)", in_match.group(1))
                if group_match is not None and group_match.group(1) == self.arbiter_fail_group:
                    raise RuntimeError(f"stub: forced failure of {self.arbiter_fail_group}")
            findings: list[dict[str, Any]] = []
            if in_match is not None and not self.arbiter_omit_verdicts:
                arb_inputs = json.loads(Path(in_match.group(1)).read_text())
                for entry in arb_inputs:
                    findings.append({
                            "arb_id": entry["arb_id"], "keep": True, "severity": entry.get("severity") or "high",
                            "confidence": entry.get("confidence") or "HIGH",
                            "description": f"ARBITRATED: {entry.get('description')}",
                            "rationale": "arbiter second opinion",
                            "evidence": entry.get("evidence") or "api.py:1",
                        }
                    )
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"findings": findings},
                continuation=(ContinuationToken(backend="claude", data={"session_id": self.arbiter_session_id})
                    if self.arbiter_session_id
                    else None
                ),
            )
            return

        # Echo suppression IDs. keep=False drops borderline findings; keep=True
        # retains cited findings. Its role sentence uniquely selects this branch.
        if "you are the suppression reviewer" in pl:
            in_match = re.search(r"listed in (\S+suppression-input\.json)", prompt)
            sup_findings: list[dict[str, Any]] = []
            if in_match is not None:
                sup_inputs = json.loads(Path(in_match.group(1)).read_text())
                for entry in sup_inputs:
                    sup_findings.append({"sup_id": entry["sup_id"], "keep": self.suppression_keep,
                            "severity": entry.get("severity") or "low", "confidence": entry.get("confidence") or "LOW",
                            "description": entry.get("description") or "finding",
                            "rationale": ("confirmed by code" if self.suppression_keep else "no confirming evidence"),
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
                # Echo arbiter revisions from current envelope or legacy bare records.
                echoed: list[dict[str, Any]] = []
                next_id = 1
                for path_str in re.findall(r"  - (\S+-records\.json)", prompt):
                    loaded = json.loads(Path(path_str).read_text())
                    recs = record_issues_or_empty(loaded)
                    for rec in recs:
                        echoed.append({
                                "id": next_id, "lens": "per-stack", "file": rec.get("file"), "line": rec.get("line"),
                                "severity": rec.get("severity", "medium"), "description": rec.get("description"),
                                "confidence": rec.get("confidence", "MEDIUM"),
                                "rationale": rec.get("rationale", "rationale"),
                                "evidence": rec.get("evidence", "api.py:1"),
                                # Copy host-owned record provenance verbatim through source_uids.
                                "source_uids": [uid] if (uid := record_uid(rec)) else [],
                            }
                        )
                        next_id += 1
                yield ResultEvent(structured_output=merge_result(echoed), continuation=None)
                return
            if self.merge_items is not None:
                yield ResultEvent(structured_output=merge_result(self.merge_items), continuation=None,)
                return
            # Default items cite real record UIDs: per-stack items cite their named
            # stack and cross-stack items cite every stack. Collapsed generic runs
            # fall back to the first available stack.
            lead_uids = [group[0] for group in self._prompt_record_uid_groups(prompt)]
            leads_by_stack = {stack_name_from_uid(uid): uid for uid in reversed(lead_uids)}

            def _lead(stack: str) -> list[str]:
                uid = leads_by_stack.get(stack) or (lead_uids[0] if lead_uids else "")
                return [uid] if uid else []

            yield ResultEvent(structured_output=merge_result([{
                            "id": 1, "lens": "per-stack", "file": "api.py", "line": 1, "severity": "medium",
                            "description": "Python issue", "confidence": "MEDIUM", "rationale": "rationale",
                            "evidence": "api.py:1", "source_uids": _lead("python"),
                        }, {"id": 2, "lens": "per-stack", "file": "App.tsx", "line": 1, "severity": "medium",
                            "description": "React issue", "confidence": "MEDIUM", "rationale": "rationale",
                            "evidence": "App.tsx:1", "source_uids": _lead("react"),
                        }, {"id": 3, "lens": "cross-stack", "file": "api.py", "line": 1, "severity": "high",
                            "description": "Contract drift between Python handler and React caller",
                            "confidence": "HIGH", "rationale": "rationale", "evidence": "api.py:1",
                            "source_uids": list(lead_uids),
                        },
                    ]), continuation=None,
            )
            return

        if "supervisor adjudication" in pl:
            in_match = re.search(r"listed in (\S+supervise-input\.json)", prompt)
            input_items = json.loads(Path(in_match.group(1)).read_text()) if in_match else []
            verdicts = []
            for item in input_items:
                verdict = ({"action": "allow", "reason": "confirmed by supplied evidence"}
                           if self.supervise_verdicts is None else self.supervise_verdicts.get(item["id"]))
                if verdict is not None:
                    verdicts.append({"id": item["id"], "action": "allow", "reason": "confirmed",
                                     "severity": None, "confidence": None, "description": None,
                                     "rationale": None, "evidence": None, **verdict})
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"verdicts": verdicts}, continuation=None)
            return

        # Fix sentinels make gate authorization observable in real-path tests.
        if pl.startswith("fix this issue") or pl.startswith("fix these"):
            if self.runaway_fix:
                # A missing result lets the real budgets terminate the fix turn.
                async for event in self._runaway_burst("tc", self.runaway_fix_sleep_s):
                    yield event
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
            # Charge failed backend time before raising; the real invocation deadline
            # must bound retries. Later attempts apply the normal fix.
            if ((self.fix_retryable_file is None or self.fix_retryable_file == fixed_name)
                and self._fix_retry_counts.get(fixed_name, 0) < self.fix_retryable_failures
            ):
                self._fix_retry_counts[fixed_name] = self._fix_retry_counts.get(fixed_name, 0) + 1
                self._tick()
                raise self.fix_retryable_error or _StubRetryableError(f"stub: retryable fix failure for {fixed_name}")
            # Run away for this file only when no batched header is present.
            if (self.runaway_single_fix_file is not None
                and m is not None
                and batched_hdr is None
                and fixed_name == self.runaway_single_fix_file
            ):
                while self.runaway_gate is not None and not self.runaway_gate():
                    await anyio.sleep(0)
                async for event in self._runaway_burst("stc", self.runaway_fix_sleep_s):
                    yield event
                return
            # The real wall timeout raises into per-finding fallback.
            if (self.runaway_batched_fix_file is not None
                and batched_hdr is not None
                and fixed_name == self.runaway_batched_fix_file
            ):
                async for event in self._runaway_burst("btc", self.runaway_batched_sleep_s):
                    yield event
                return
            # Fail this file's batch so its per-finding fallback runs.
            if (self.fail_batched_fix_file is not None
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
                if self.fix_damage_protected_file is not None:
                    (cwd / self.fix_damage_protected_file).write_bytes(b"damaged by failed fixer")
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
                    yield ToolStartEvent(id=f"deferred-write-{index}", name="Write",
                        input={"file_path": str(edit_target), "content": "backend resumed"},
                    )
                    self._tick()
                    edit_target.write_text("backend resumed")
                yield TextEvent(text="Applied the deferred writes.")
                self._tick()
                yield ResultEvent(structured_output=None, continuation=None)
                self.completed_fix_files.append(fixed_name)
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
                # Append all handed-off markers in prompt/severity order.
                toks = re.findall(r"marker-\d+", prompt) or ["?"]
                append_path = cwd / self.fix_append_path.relative_to(self._target)
                cur = append_path.read_text() if append_path.exists() else ""
                await anyio.sleep(0)  # deterministic interleave point
                append_path.write_text(cur + "".join(t + "\n" for t in toks))
            yield TextEvent(text="Applied the fix.")
            self._tick()
            yield ResultEvent(structured_output=None, continuation=None)
            self._tick()
            self.completed_fix_files.append(fixed_name)
            return

        # Role-based recommendation dispatch; the phase persists the verdicts.
        if "you are the recommendation-verifier agent" in pl:
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"verdicts": [{
                            "issue_id": 1, "verdict": self.verifier_verdict, "evidence": "stub",
                            "unverified_assumptions": list(self.verifier_unverified_assumptions),
                        }
                    ]
                }, continuation=None,
            )
            return

        # Post-fix verification must be read-only and covers every dispatched ID.
        if "post-fix fix-verifier agent" in pl:
            if not read_only:
                raise AssertionError("fix-verify turn must arrive read_only=True")
            round_match = re.search(r"(?:Round (\d+) of up to 3 check passes|Verification pass (\d+))", prompt,)
            round_num = int(next(group for group in round_match.groups() if group)) if round_match else 1
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
            yield ResultEvent(structured_output={"verdicts": verdicts}, continuation=None,)
            return

        # The healing sentinel proves re-entry; environmental failures must leave
        # it absent.
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
            if self.heal_fix_unauthorized is not None:
                unauthorized = cwd / self.heal_fix_unauthorized
                unauthorized.write_text(
                    unauthorized.read_text() + self.heal_fix_unauthorized_line
                )
            yield TextEvent(text="Attempted to fix the test failures.")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Test failures can be first-only, permanent, or environmental. The
        # Postgres signature must stop healing before another fix turn.
        if "run the project's test suite" in pl:
            self.test_suite_calls += 1
            if self.environmental_test_failure:
                yield TextEvent(text=(
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
    """Silence noise-only UI helpers at their deep-flow owners.

    ``prompts=False`` leaves the real interaction gateway input in place, for
    tests that drive a genuine gate.
    """
    monkeypatch.setattr("daydream.deep.review_steps.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.merge_steps.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.fix_steps.print_verification_summary", lambda *a, **kw: None)
    if prompts:
        def answer(_console: Any, message: str, default: str = "") -> str:
            return "y" if "understanding correct" in message.lower() else "n"

        monkeypatch.setattr("daydream.run_context._prompt_user", answer)


def force_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set TTY stdin and clear CI so prompt tests exercise real interactive gates."""
    monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)


def install_stub_backend(monkeypatch: pytest.MonkeyPatch, target: Path, *, pin_skill_availability: bool = True,
    enable_exploration: bool = False,
) -> StubBackend:
    """Install one backend instance and optionally pin exploration availability.

    With pin_skill_availability=True, enable_exploration selects whether the
    pre-scan runs (default False). Otherwise leave module availability intact.
    The historical parameter name remains for call-site compatibility.
    """
    stub = StubBackend(target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    if pin_skill_availability:
        if enable_exploration:
            # Pin True so the pre_scan branch runs regardless of ambient module state.
            monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", True)
        else:
            # Disable exploration pre-scan so it doesn't add extra backend calls.
            monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    return stub
