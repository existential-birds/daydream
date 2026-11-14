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
    ResultEvent,
    TextEvent,
    ToolStartEvent,
)


class StubBackend:
    """Emit realistic review/fix results and record calls for pipeline assertions."""

    model = "mock-model"
    supports_host_tools = True

    def __init__(self, target: Path, *, model: str = "mock-model") -> None:
        self.model = model
        self._target = target
        # Claims the scripted review discovers, each ``{"title", "file", "line"}`` cited on the new
        # side unless ``"side": "old"``; None proposes
        # one "Sample issue" on the first changed target. ``review_verdict`` is the challenge's answer
        # and ``review_undecided`` leaves that many trailing targets without a decision.
        self.review_claims: list[dict[str, Any]] | None = None
        self.review_verdict: str = "confirmed"
        self.review_undecided: int = 0
        # Raised by every review assignment after one observed tool start, before any submission.
        self.review_failure: BaseException | None = None
        # Called with the assignment kind after the host accepted its submission and before native
        # settlement; raising there is a native failure after acceptance.
        self.review_after_acceptance: Callable[[str], None] | None = None
        # Backend fan-out hint; tests override it for deterministic group ordering.
        self.fanout_concurrency: int = 4
        self.calls: list[dict[str, Any]] = []
        # Forbidden legacy TEST dispatch count: real suites use host commands.
        self.test_suite_calls: int = 0
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


    async def _runaway_burst(self, prefix: str, sleep_s: float) -> AsyncIterator[AgentEvent]:
        """Emit 500 tool calls without a result, sleeping after each."""
        for n in range(500):
            yield ToolStartEvent(id=f"{prefix}-{n}", name="Bash", input={"command": "find /"})
            await anyio.sleep(sleep_s)

    def bind_review_context(self, original_source: Path) -> dict[str, object]:
        """As a native adapter does, keep the binding for every later invocation of this instance."""
        del original_source
        self.review_bound = True
        return {"configuration_identity": "0" * 64}

    def _scripted_review(self, prompt: str, binding: Any) -> dict[str, Any]:
        """Answer one committed-review assignment through its real host binding.

        The assignment is the first message's second host line. Discovery decides its targets from the
        host lookup and proposes the configured claims; challenge returns ``review_verdict`` for every
        assigned claim revision.
        """
        token = prompt.split(" ", 1)[0]
        assignment = json.loads(prompt.split("\n")[1].removeprefix(f"{token} Committed review assignment: "))
        if assignment["kind"] == "challenge":
            return {"claims": [], "verdicts": [
                {"uid": claim["uid"], "revision": claim["revision"], "verdict": self.review_verdict,
                 "grounds": "Captured source was searched for counterevidence.", "evidence_refs": []}
                for claim in assignment["claims"]]}
        lookup = binding.host_tools["review_source"].callback
        targets = [lookup({"target_id": target, "path": None, "cursor": None})["target"]
                   for target in assignment["work"]["targets"]["ids"]]
        decided = targets[:len(targets) - self.review_undecided] if self.review_undecided else targets
        wanted = self.review_claims
        if wanted is None:
            wanted = [{"title": "Sample issue", "file": targets[0]["file"], "line": None}] if targets else []
        claims = []
        for claim in wanted:
            option = next(option for target in targets if target["file"] == claim["file"]
                          for option in target["source_options"] if option["side"] == claim.get("side", "new"))
            line = claim["line"]
            if line is None and option["source_range"] is not None:
                line = option["source_range"]["start"]
            claims.append({"uid": "", "revision": 0, "source_uids": [], "title": claim["title"], "payload": {
                "claim": claim.get("claim", "The changed behavior is incorrect."),
                "trigger": "The changed path is exercised.", "consequence": "Callers observe the defect.",
                "grounds": f"{claim['file']} changes this behavior.", "file": claim["file"], "line": line,
                "evidence_refs": [{key: option[key] for key in ("source_id", "basis", "source_range")}]}})
        return {"claims": claims, "targets": [
            {"target_id": target["target_id"], "status": "reviewed", "reason": "Captured change inspected.",
             "evidence_refs": []} for target in decided]}

    async def execute(
        self, cwd: Path, prompt: str, output_schema: Any = None, continuation: Any = None, agents: Any = None,
        read_only: bool = False, persist_session: bool = True, review_instructions: str | None = None,
        host_tools: Any = None,
    ) -> AsyncIterator[AgentEvent]:
        call = {"cwd": cwd, "prompt": prompt, "output_schema": output_schema, "agents": agents, "model": self.model,
            "continuation": continuation, "read_only": read_only,
            "persist_session": persist_session, "review_instructions": review_instructions,
            "review_bound": getattr(self, "review_bound", False),
        }
        self.calls.append(call)
        if host_tools is not None:
            assert review_instructions is not None
            host_tools.configure_schema(output_schema)
            if self.review_failure is not None:
                yield ToolStartEvent(id="failed-tool", name="Read", input={"file_path": "api.py"})
                raise self.review_failure
            async with host_tools.session():
                value = self._scripted_review(prompt, host_tools)
                host_tools.admit("terminal", "structured_output", value, terminal=True)
                accepted = host_tools.submit("terminal", value)
                if self.review_after_acceptance is not None:
                    self.review_after_acceptance("challenge" if "verdicts" in value else "discovery")
                host_tools.native_settled = True
            yield ResultEvent(structured_output=accepted, continuation=None)
            return
        pl = prompt.lower()

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
    monkeypatch.setattr("daydream.deep.fix_steps.print_verification_summary", lambda *a, **kw: None)
    if prompts:
        def answer(_console: Any, message: str, default: str = "") -> str:
            return "n"

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
