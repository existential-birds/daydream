"""Prompt-dispatching backend and UI helpers for deep-pipeline integration tests."""
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
    ResultEvent,
    TextEvent,
    ToolStartEvent,
)
from daydream.deep.records import record_issues_or_empty, record_uid, stack_name_from_uid
from tests.harness.review_result import merge_result


def review_stage_state(prompt: str) -> dict[str, Any] | None:
    """Decode the host state before any trailing invocation instructions."""
    _, marker, payload = prompt.partition("Host review stage:\n")
    state: dict[str, Any] | None = json.JSONDecoder().raw_decode(payload)[0] if marker else None
    return state


def stage_result(stage: dict[str, Any], *, candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    def references(target: str) -> list[dict[str, Any]]:
        return [{'source_id': option['source_id'], 'basis': 'supplied_diff', 'source_range': None}
                for option in stage.get('source_options', {}).get(target, [])[:1]]
    return {"targets": [{"target_id": target, "status": "reviewed", "reason": ""}
                    | {'evidence_refs': references(target)}
                    for target in stage["assigned_target_ids"]],
        "notes": "Assigned changed behavior reviewed", "supporting_files": [],
        "candidates": candidates or [], "contradictions": []}


def review_stage_result(prompt: str, issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Emit fixture findings once, in the first discovery batch; retain legacy output."""
    state = review_stage_state(prompt)
    if state is None:
        return {"issues": issues}
    candidates = []
    if state["stage"] in {"first_pass", "integration"} and not state["progress"]:
        candidates = [{"candidate_id": "", "file": issue["file"], "line": issue["line"],
            "trigger": "The changed path is exercised", "consequence": issue["description"],
            "grounds": issue["evidence"], "disposition": "confirmed", "finding": issue} for issue in issues]
    elif state["stage"] == "triage":
        candidates = [dict(candidate, disposition="confirmed") for candidate in state["candidates"]
                      if candidate["candidate_id"] in state["assigned_candidate_ids"]]
    return stage_result(state, candidates=candidates)


class StubBackend:
    """Emit realistic review/fix results and record calls for pipeline assertions."""

    model = "mock-model"

    def __init__(self, target: Path, *, model: str = "mock-model") -> None:
        self.model = model
        self._target = target
        # Backend fan-out hint; tests override it for deterministic group ordering.
        self.fanout_concurrency: int = 4
        # Parse severity selects arbiter targets; merge_echo_records exposes revisions.
        self.parse_severity: str | None = None
        self.merge_echo_records: bool = False
        # Fail only this shard (e.g. arbiter-group-1) for fail-open/resume tests.
        self.arbiter_fail_group: str | None = None
        # Per-stack findings can vary file/severity/confidence to control suppression selection.
        self.parse_by_stack: dict[str, dict[str, Any]] | None = None
        self.calls: list[dict[str, Any]] = []
        # Forbidden legacy TEST dispatch count: real suites use host commands.
        self.test_suite_calls: int = 0
        # Override for the merge agent's item list (None -> default three-item payload).
        self.merge_items: list[dict[str, Any]] | None = None
        # Emit unstructured prose to exercise merge salvage.
        self.merge_emit_str: str | None = None
        # Optional LLM supervisor verdicts keyed by canonical item id.
        self.supervise_verdicts: dict[int, dict[str, Any]] | None = None
        # Fail this file's batched turn; its per-finding fallback turns succeed.
        self.fail_batched_fix_file: str | None = None
        # Repo-relative generated path created by a successful fix turn.
        self.fix_new_generated: str | None = None
        # Repo-relative historical generated path modified by a test-healing
        # fix turn, plus an optional new generated path created by that turn.
        self.heal_fix_generated: str | None = None
        self.heal_fix_new_generated: str | None = None
        # Text a healing turn emits as its partial diagnosis and then never
        # finishes: the real run_agent wall budget ends the turn, so the host
        # records an interrupted repair rather than a completed one.
        self.heal_fix_partial: str | None = None
        # Authorized tracked path the healing turn edits, with the line it
        # appends. Lands before any interruption check, so an interrupted turn
        # leaves the work it was mid-way through on disk.
        self.heal_fix_authorized: str | None = None
        self.heal_fix_authorized_line: str = "\n# healing edit\n"
        # Partial-repair runaway pacing; zero still yields via sleep(0). Positive
        # values can trip the wall limit before the tool limit.
        self.runaway_fix_sleep_s: float = 0.0
        # Append a real tracked edit so recommended-change patches can capture it.
        self.fix_edit_line: str | None = None
        # Runaway wonder: tool events without a result.
        self.runaway_alternatives: bool = False
        # When True, the alternatives branch raises instead of answering.
        self.fail_alternatives: bool = False
        # Arbiter continuation session that merge can resume.
        self.arbiter_session_id: str | None = None
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

        Retained language and structural records both appear in merge prompts.
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
        """Emit 500 tool calls without a result, sleeping after each."""
        for n in range(500):
            yield ToolStartEvent(id=f"{prefix}-{n}", name="Bash", input={"command": "find /"})
            await anyio.sleep(sleep_s)

    async def execute(
        self, cwd: Path, prompt: str, output_schema: Any = None, continuation: Any = None, agents: Any = None,
        read_only: bool = False, persist_session: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        call = {"cwd": cwd, "prompt": prompt, "output_schema": output_schema, "agents": agents, "model": self.model,
            "continuation": continuation, "read_only": read_only,
            "persist_session": persist_session,
        }
        self.calls.append(call)
        pl = prompt.lower()

        # Check alternatives before intent: their prompt embeds the intent summary.
        if "would you have done this differently" in pl or "evaluate the implementation" in pl:
            if self.runaway_alternatives:
                # No ResultEvent: run_agent must terminate this burst through its budgets.
                async for event in self._runaway_burst("tc", 0):
                    yield event
                return
            if self.fail_alternatives:
                raise RuntimeError("alternatives blew up")
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
            payload = {"conventions": [{"name": "OpenAPI First",
                        "description": "openapi.yaml is the HTTP contract", "source": "AGENTS.md",
                    }
                ], "guidelines": [],
            }
            yield TextEvent(text=json.dumps({"conventions": payload["conventions"], "guidelines": []}))
            yield ResultEvent(structured_output=payload, continuation=None)
            return
        if "you are the **dependency-tracer** specialist" in pl:
            payload = {"affected_files": [],
                "dependencies": [{"source": "App.tsx", "target": "api.py", "relationship": "calls"}],
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
            state = review_stage_state(prompt)
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
            if state is not None and not (cwd / 'api.py').is_file():
                issue.update(file=state['assigned_files'][0], evidence=f"{state['assigned_files'][0]}:1")
            issues: list[dict[str, Any]] = self._apply_parse_by_stack_override(prompt, issue)
            payload = review_stage_result(prompt, issues)
            yield TextEvent(text="")
            yield ResultEvent(structured_output=payload, continuation=None,)
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
            if in_match is not None:
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

        # Echo suppression IDs with keep=True, retaining cited findings. Its role
        # sentence uniquely selects this branch.
        if "you are the suppression reviewer" in pl:
            in_match = re.search(r"listed in (\S+suppression-input\.json)", prompt)
            sup_findings: list[dict[str, Any]] = []
            if in_match is not None:
                sup_inputs = json.loads(Path(in_match.group(1)).read_text())
                for entry in sup_inputs:
                    sup_findings.append({"sup_id": entry["sup_id"], "keep": True,
                            "severity": entry.get("severity") or "low", "confidence": entry.get("confidence") or "LOW",
                            "description": entry.get("description") or "finding", "rationale": "confirmed by code",
                            "evidence": entry.get("evidence") or "",
                        }
                    )
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"findings": sup_findings}, continuation=None)
            return

        # Cross-stack merge -> complete model-authored items; the host validates
        # evidence/provenance, normalizes ids, and renders review-output.md.
        if "cross-stack merge agent" in pl:
            yield TextEvent(text="")
            if self.merge_emit_str is not None:
                yield TextEvent(text=self.merge_emit_str)
                yield ResultEvent(structured_output=None, continuation=None)
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
                                "id": next_id, "lens": "structural" if record_uid(rec).startswith("structure:")
                                else "per-stack", "file": rec.get("file"), "line": rec.get("line"),
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
            # Default synthesis retains structural concerns as independent items.
            structural_items = []
            for path_str in re.findall(r"  - (\S+-records\.json)", prompt):
                for rec in record_issues_or_empty(json.loads(Path(path_str).read_text())):
                    if record_uid(rec).startswith("structure:"):
                        structural_items.append({
                            key: rec[key] for key in ("id", "file", "line", "severity", "confidence", "description",
                                                     "rationale", "evidence") if key in rec
                        } | {"lens": "structural", "source_uids": [record_uid(rec)]})
            lead_uids = [group[0] for group in self._prompt_record_uid_groups(prompt)
                         if not group[0].startswith("structure:")]
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
                    ] + structural_items), continuation=None,
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
            # Single-finding prompts carry "File: <path>"; batched prompts name the
            # one target file in their "Fix these N issues in <path>:" header.
            m = re.search(r"^File: (.+)$", prompt, re.M)
            batched_hdr = re.search(r"^Fix these \d+ issues in (.+):$", prompt, re.M)
            if m is None:
                m = batched_hdr
            fixed_file = m.group(1).strip() if m else "unknown"
            # phase_fix emits an absolute path when the file exists on disk; the stub keys fixes by basename.
            fixed_name = Path(fixed_file).name
            # Fail this file's batch so its per-finding fallback runs.
            if (self.fail_batched_fix_file is not None
                and batched_hdr is not None
                and fixed_name == self.fail_batched_fix_file
            ):
                raise RuntimeError(f"stub: batched fix failure for {fixed_name}")
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
            yield TextEvent(text="Applied the fix.")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Role-based recommendation dispatch; the phase persists the verdicts.
        if "you are the recommendation-verifier agent" in pl:
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"verdicts": [{
                            "issue_id": 1, "verdict": "consistent", "evidence": "stub",
                            "unverified_assumptions": [],
                        }
                    ]
                }, continuation=None,
            )
            return

        # Post-fix verification must be read-only and covers every dispatched ID.
        if "post-fix fix-verifier agent" in pl:
            if not read_only:
                raise AssertionError("fix-verify turn must arrive read_only=True")
            verdicts = [{"issue_id": int(i), "verdict": "resolved", "reason": "stub"}
                        for i in re.findall(r"(?m)^(\d+)\. \[", prompt)]
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"verdicts": verdicts}, continuation=None,)
            return

        # The healing sentinel proves re-entry; environmental failures must leave
        # it absent.
        if pl.startswith("the tests failed"):
            if self.heal_fix_authorized is not None:
                edited = cwd / self.heal_fix_authorized
                if edited.exists():
                    edited.write_text(edited.read_text() + self.heal_fix_authorized_line)
            if self.heal_fix_partial is not None:
                # Diagnosis, then no result: only the host budget can end this
                # turn, which is exactly the interrupted repair under test.
                yield TextEvent(text=self.heal_fix_partial)
                async for event in self._runaway_burst("htc", self.runaway_fix_sleep_s):
                    yield event
                return
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

        # Keep the old prompt recognizable as a negative dispatch observable.
        if "run the project's test suite" in pl:
            self.test_suite_calls += 1
            raise AssertionError("TEST agents cannot execute the canonical suite")

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
