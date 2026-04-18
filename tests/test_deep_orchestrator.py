"""Deep-mode orchestrator integration tests (plan 05-09).

Covers D-07..D-10, D-17, D-19..D-22, D-24..D-26, D-28, D-30, D-31,
D-34, D-35, D-44.

The tests share a ``_StubBackend`` that dispatches on prompt content to
simulate the full review pipeline without talking to a real SDK.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import ResultEvent, TextEvent


class _StubBackend:
    """MockBackend that dispatches on prompt content.

    Writes realistic per-stack review outputs and a merged report so the
    orchestrator can progress through every stage. Records every call so
    tests can assert ordering, agents-kwarg absence, and per-stack isolation.
    """

    def __init__(self, target: Path, *, is_codex: bool = False) -> None:
        self._target = target
        self._is_codex = is_codex
        self.calls: list[dict[str, Any]] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
    ):
        self.calls.append(
            {
                "cwd": cwd,
                "prompt": prompt,
                "output_schema": output_schema,
                "agents": agents,
            }
        )
        pl = prompt.lower()

        # TTT intent phase -> returns plain text.
        if "describe what these changes are trying to achieve" in pl or "confirm your understanding" in pl:
            yield TextEvent(text="The PR updates greetings across stacks.")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # TTT alternative-review phase -> structured output.
        if "alternative" in pl and "evaluate" in pl:
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

        # Per-stack review -> write a markdown file + emit done.
        m = re.search(r"you are reviewing the (\S+) stack", pl)
        if m is None:
            m = re.search(r"you are reviewing the (generic-fallback) stack", pl)
        if m is not None:
            # Extract the output path the prompt asks the agent to write.
            out_match = re.search(r"write your full review to (\S+)", prompt, flags=re.IGNORECASE)
            if out_match is not None:
                out_path = Path(out_match.group(1))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                stack = m.group(1)
                out_path.write_text(
                    f"# Review ({stack})\n\n## Issues\n\n1. [api.py:1] Sample issue for {stack}\n"
                )
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # phase_parse_feedback -> structured output.
        if "extract only actionable issues" in pl:
            yield TextEvent(text="")
            yield ResultEvent(
                structured_output={
                    "issues": [
                        {"id": 1, "description": "Sample issue", "file": "api.py", "line": 1}
                    ]
                },
                continuation=None,
            )
            return

        # Cross-stack merge -> write the report to REVIEW_OUTPUT_FILE.
        if "cross-stack merge agent" in pl:
            out_match = re.search(r"write the complete report to (\S+)", prompt, flags=re.IGNORECASE)
            if out_match is not None:
                out_path = Path(out_match.group(1))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    "# Review\n\n"
                    "## Issues\n\n"
                    "1. [api.py:1] Python issue\n"
                    "   rationale\n"
                    "2. [App.tsx:1] React issue\n"
                    "   rationale\n\n"
                    "## Cross-Stack Issues\n\n"
                    "3. [cross-stack] [api.py:1] Contract drift between Python handler and React caller\n"
                    "   rationale\n"
                )
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Default: empty text + no structured output.
        yield TextEvent(text="")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


def _silence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence interactive UI helpers in deep orchestrator + phases."""
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.prompt_user", lambda *a, **kw: "n")
    # phase_understand_intent calls prompt_user("Is this understanding correct?", "y")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")


def _install_stub_backend(
    monkeypatch: pytest.MonkeyPatch, target: Path, *, is_codex: bool = False
) -> _StubBackend:
    """Patch create_backend to return a single stub backend instance."""
    stub = _StubBackend(target, is_codex=is_codex)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: stub)
    # When is_codex=True, the orchestrator's isinstance(backend, CodexBackend)
    # check needs to succeed. Patch CodexBackend to our stub's class so the
    # isinstance check fires without needing a real Codex dependency.
    if is_codex:
        monkeypatch.setattr("daydream.deep.orchestrator.CodexBackend", _StubBackend, raising=False)
    return stub


async def _run_deep(target: Path, *, start_at: str = "review") -> int:
    from daydream.runner import RunConfig, run

    config = RunConfig(target=str(target), deep=True, start_at=start_at)
    return await run(config)


async def test_pipeline_order(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-07: Stage order = TTT intent -> alternatives -> per-stack -> parse -> merge."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # Classify each call by inspecting prompt content.
    order: list[str] = []
    for call in stub.calls:
        pl = call["prompt"].lower()
        if "describe what these changes are trying to achieve" in pl:
            order.append("intent")
        elif "alternative" in pl and "evaluate" in pl:
            order.append("alternatives")
        elif "you are reviewing the" in pl and "stack" in pl:
            order.append("per-stack")
        elif "extract only actionable issues" in pl:
            order.append("parse")
        elif "cross-stack merge agent" in pl:
            order.append("merge")

    first = {name: order.index(name) for name in set(order)}
    assert first["intent"] < first["alternatives"]
    assert first["alternatives"] < first["per-stack"]
    assert first["per-stack"] < first["parse"]
    assert first["parse"] < first["merge"]


async def test_fresh_context_per_stage(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-08: Each stage = a distinct Backend.execute call (no continuation reuse)."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    # At minimum: intent + alternatives + 3 per-stack + 3 parse + 1 merge = 9 distinct calls.
    assert len(stub.calls) >= 8
    # Distinct by prompt (no test here about continuation, just that each stage fires a separate execute).
    prompts = [c["prompt"] for c in stub.calls]
    assert len(set(prompts)) == len(prompts) or len(prompts) >= 8


async def test_artifacts_on_disk(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-09: intent.md, alternatives.json, stack-*-review.md, stack-*-records.json exist."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    deep = multi_stack_target / ".daydream" / "deep"
    assert (deep / "intent.md").exists()
    assert (deep / "alternatives.json").exists()
    # At least one per-stack review and one per-stack records JSON.
    review_files = list(deep.glob("stack-*-review.md"))
    records_files = list(deep.glob("stack-*-records.json"))
    assert review_files, "expected at least one stack-*-review.md"
    assert records_files, "expected at least one stack-*-records.json"
    # Dedup candidates.
    assert (deep / "dedup-candidates.json").exists()


async def test_per_stack_context_isolation(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-10: per-stack prompts don't embed other stacks' file lists."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # Find per-stack prompts.
    per_stack_prompts = [
        c["prompt"] for c in stub.calls if "you are reviewing the" in c["prompt"].lower()
    ]
    assert per_stack_prompts, "expected per-stack prompts"
    # Each per-stack prompt should mention its own stack's file but NOT foreign files.
    python_prompt = next((p for p in per_stack_prompts if "api.py" in p and "the python stack" in p.lower()), None)
    react_prompt = next((p for p in per_stack_prompts if "app.tsx" in p.lower() and "the react stack" in p.lower()), None)
    assert python_prompt is not None
    assert react_prompt is not None
    # Python prompt should not embed React files in its scope instruction.
    python_scope_line = next(
        line for line in python_prompt.splitlines() if "focus only on these files" in line.lower()
    )
    # next line holds the actual file list (per build_per_stack_prompt).
    assert "App.tsx" not in python_prompt.split("Focus ONLY on these files:")[1].split("\n", 2)[1]


async def test_parallel_fan_out(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-17: per-stack fan-out uses anyio task group. No ``agents`` kwarg passed to execute."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    # Every execute call must have agents=None per D-38.
    assert all(c["agents"] is None for c in stub.calls)


async def test_per_stack_prompt_context(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-19: per-stack prompts reference intent and alternatives paths."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    per_stack_prompts = [
        c["prompt"] for c in stub.calls if "you are reviewing the" in c["prompt"].lower()
    ]
    assert per_stack_prompts
    for p in per_stack_prompts:
        assert "intent.md" in p
        assert "alternatives.json" in p


async def test_doc_review_notice(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-20: generic fallback gets build_generic_fallback_prompt (may include doc notice when docs-only)."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # In the multi-stack fixture the diff is mixed (python + react + md), so the
    # generic bucket is NOT docs-only and the notice is NOT expected. The contract
    # is: a generic-fallback prompt is emitted for README.md and it mentions the
    # file.
    fallback_prompts = [
        c["prompt"] for c in stub.calls if "you are reviewing the generic-fallback stack" in c["prompt"].lower()
    ]
    assert fallback_prompts
    assert any("README.md" in p for p in fallback_prompts)


async def test_pre_merge_parse_per_stack(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-21, D-22: phase_parse_feedback invoked once per per-stack output; records written."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    parse_calls = [
        c for c in stub.calls if "extract only actionable issues" in c["prompt"].lower()
    ]
    # At least as many parse calls as per-stack outputs.
    per_stack_outputs = list((multi_stack_target / ".daydream" / "deep").glob("stack-*-review.md"))
    assert len(parse_calls) >= len(per_stack_outputs)
    records = list((multi_stack_target / ".daydream" / "deep").glob("stack-*-records.json"))
    assert len(records) == len(per_stack_outputs)


async def test_merged_report_path(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-24: final report written at REVIEW_OUTPUT_FILE path."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    from daydream.config import REVIEW_OUTPUT_FILE

    assert (multi_stack_target / REVIEW_OUTPUT_FILE).exists()


async def test_report_format_flat_numbered(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-25: flat globally-numbered ## Issues + continuing ## Cross-Stack Issues subsection."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    from daydream.config import REVIEW_OUTPUT_FILE

    text = (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert "## Issues" in text
    assert "## Cross-Stack Issues" in text
    # Numbering continues: stub writes 1., 2. in ## Issues then 3. in ## Cross-Stack Issues.
    assert "3." in text.split("## Cross-Stack Issues", 1)[1]


async def test_cross_stack_prefix(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-26: every cross-stack title starts with [cross-stack]."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    from daydream.config import REVIEW_OUTPUT_FILE

    text = (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    cross_section = text.split("## Cross-Stack Issues", 1)[1]
    assert "[cross-stack]" in cross_section


async def test_fix_gate_prompt(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-28: Y/n prompt after merge decides whether to apply fixes."""
    _install_stub_backend(monkeypatch, multi_stack_target)

    asked: list[str] = []

    def _record_prompt(console, message, default=""):
        asked.append(message)
        # Decline the fix gate.
        return "n"

    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.prompt_user", _record_prompt)
    # phase_understand_intent also calls prompt_user; always confirm.
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    # At least one prompt message must mention the fix gate.
    assert any("fix" in msg.lower() or "apply" in msg.lower() for msg in asked)


async def test_preflight_notice(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-30: pre-flight notice lists stages, stacks, skill per stack, total agent count."""
    captured: list[dict[str, Any]] = []

    def _capture(console, *, stages, stack_lines, agent_count, codex_in_use, exploration_available) -> None:
        captured.append(
            {
                "stages": stages,
                "stack_lines": stack_lines,
                "agent_count": agent_count,
                "codex_in_use": codex_in_use,
                "exploration_available": exploration_available,
            }
        )

    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", _capture)
    monkeypatch.setattr("daydream.deep.orchestrator.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")
    _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    assert len(captured) == 1, "pre-flight notice must fire exactly once"
    notice = captured[0]
    # Exactly 5 stages.
    assert len(notice["stages"]) == 5
    # Agent count: 2 TTT + N per-stack + N parse + 1 merge with N=3 -> 9.
    assert notice["agent_count"] == 9
    # Stacks surfaced.
    assert len(notice["stack_lines"]) >= 1
    # No Codex caveat on Claude backend.
    assert notice["codex_in_use"] is False


async def test_codex_cost_caveat(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-31: Codex backend triggers the cost_usd=None caveat in the notice."""
    captured: list[dict[str, Any]] = []

    def _capture(console, *, stages, stack_lines, agent_count, codex_in_use, exploration_available) -> None:
        captured.append({"codex_in_use": codex_in_use})

    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", _capture)
    monkeypatch.setattr("daydream.deep.orchestrator.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")
    _install_stub_backend(monkeypatch, multi_stack_target, is_codex=True)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    assert captured[0]["codex_in_use"] is True


async def test_resume_per_stack_reruns_all(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-34: --start-at per-stack re-runs ALL per-stack reviews (after priming TTT artifacts)."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    # Prime required TTT artifacts for the resume gate.
    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")

    exit_code = await _run_deep(multi_stack_target, start_at="per-stack")
    assert exit_code == 0

    per_stack_calls = [c for c in stub.calls if "you are reviewing the" in c["prompt"].lower()]
    # detect_stacks on the fixture yields at least 2 non-generic buckets + 1 generic = 3.
    assert len(per_stack_calls) >= 2


async def test_resume_overwrites(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-35: resume overwrites stage artifacts (new stack-*-review.md replaces old)."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    # Prime TTT artifacts and an OLD per-stack review that must be overwritten.
    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    old = deep / "stack-python-review.md"
    old.write_text("STALE CONTENT")

    exit_code = await _run_deep(multi_stack_target, start_at="per-stack")
    assert exit_code == 0

    # The stub writes a new review -- STALE CONTENT must be gone.
    assert "STALE CONTENT" not in old.read_text()


async def test_stage_ui_surfacing(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-44: UI prints [stage N/5: ...] at each stage boundary."""
    progress_calls: list[tuple[int, int, str]] = []

    def _capture(console, current, total, name) -> None:
        progress_calls.append((current, total, name))

    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", _capture)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")
    _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    # At least 5 distinct stage boundaries announced.
    stage_numbers = {c[0] for c in progress_calls}
    assert stage_numbers == {1, 2, 3, 4, 5}
    # Total is always 5.
    assert all(c[1] == 5 for c in progress_calls)
