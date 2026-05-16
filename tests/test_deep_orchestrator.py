"""Deep-mode orchestrator integration tests (plan 05-09).

Covers D-07..D-10, D-17, D-19..D-22, D-24..D-26, D-28, D-30, D-31,
D-34, D-35, D-44.

The tests share a ``_StubBackend`` that dispatches on prompt content to
simulate the full review pipeline without talking to a real SDK.
"""

from __future__ import annotations

import re
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

    model = "mock-model"

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
        max_turns: Any = None,
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

        # TTT alternative-review phase -> structured output.
        # (Checked BEFORE intent because the alt prompt embeds the intent summary
        # which contains the word "intent", defeating a naive substring check.)
        if "would you have done this differently" in pl or "evaluate the implementation" in pl:
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

        # TTT intent phase -> plain text. Discriminator: "understand the intent"
        # + "commit log:" are both unique to build_intent_prompt.
        if "understand the intent of these changes" in pl:
            yield TextEvent(text="The PR updates greetings across stacks.")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Per-stack review -> write a markdown file + emit done.
        m = re.search(r"you are reviewing the (\S+) stack", pl)
        if m is None:
            m = re.search(r"you are reviewing the (generic-fallback) stack", pl)
        if m is not None:
            # Extract the output path the prompt asks the agent to write.
            out_match = re.search(r"write your full review to (\S+)", prompt, flags=re.IGNORECASE)
            if out_match is not None:
                raw = out_match.group(1).rstrip(".")
                out_path = Path(raw)
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
                raw = out_match.group(1).rstrip(".")
                out_path = Path(raw)
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
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    *,
    is_codex: bool = False,
    pin_skill_availability: bool = True,
) -> _StubBackend:
    """Patch create_backend to return a single stub backend instance.

    Args:
        pin_skill_availability: When True (default), patches
            ``get_installed_skills`` to return ``None`` (optimistic fallback
            giving all SKILL_MAP stacks) and disables the exploration
            pre-scan. This isolates tests from the local machine's Beagle
            plugin registry and prevents exploration from adding unexpected
            backend calls. Pass False when a test explicitly controls skill
            availability (e.g. via ``CLAUDE_CONFIG_DIR``).
    """
    stub = _StubBackend(target, is_codex=is_codex)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: stub)
    # When is_codex=True, the orchestrator's isinstance(backend, CodexBackend)
    # check needs to succeed. Patch CodexBackend to our stub's class so the
    # isinstance check fires without needing a real Codex dependency.
    if is_codex:
        monkeypatch.setattr("daydream.deep.orchestrator.CodexBackend", _StubBackend, raising=False)
    if pin_skill_availability:
        # Return None -> orchestrator falls back to set(SKILL_MAP.keys())
        monkeypatch.setattr("daydream.deep.orchestrator.get_installed_skills", lambda: None)
        # Disable exploration pre-scan so it doesn't add extra backend calls
        monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)
    return stub


async def _run_deep(target: Path, *, start_at: str = "review") -> int:
    from daydream.runner import RunConfig, run

    # cleanup=False suppresses the interactive cleanup prompt in runner.run().
    # Deep is the default; no shallow flag set.
    config = RunConfig(target=str(target), start_at=start_at, cleanup=False)
    return await run(config)


async def test_pipeline_order(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-07: Stage order = TTT intent -> alternatives -> per-stack -> parse -> merge."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # Classify each call by inspecting prompt content. Alt is checked before
    # intent because the alt prompt embeds the intent summary text.
    order: list[str] = []
    for call in stub.calls:
        pl = call["prompt"].lower()
        if "would you have done this differently" in pl or "evaluate the implementation" in pl:
            order.append("alternatives")
        elif "understand the intent of these changes" in pl:
            order.append("intent")
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
    assert len(stub.calls) >= 9
    # Each stage fires a distinct Backend.execute call -- prompts must be unique.
    prompts = [c["prompt"] for c in stub.calls]
    assert len(set(prompts)) == len(prompts)


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
    python_prompt = next(
        (p for p in per_stack_prompts if "api.py" in p and "the python stack" in p.lower()),
        None,
    )
    react_prompt = next(
        (p for p in per_stack_prompts if "app.tsx" in p.lower() and "the react stack" in p.lower()),
        None,
    )
    assert python_prompt is not None
    assert react_prompt is not None
    # The scope instruction's file-list line (right after the "Focus ONLY on these files:" header)
    # must not embed React files in the Python stack prompt.
    python_scope_files_line = python_prompt.split("Focus ONLY on these files:")[1].split("\n", 2)[1]
    assert "App.tsx" not in python_scope_files_line


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


async def test_resume_merge_consumes_saved_records(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--start-at merge loads stack-*-records.json and does NOT re-parse reviews.

    Regression: previously the merge branch always re-ran phase_parse_feedback
    against reconstructed stack-*-review.md paths, so resume failed when those
    markdown files were absent even though the validated records.json existed.
    """
    import json

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    # Prime saved records but intentionally NOT the review.md files -- the
    # resume path must consume records.json directly.
    (deep / "stack-python-records.json").write_text(
        json.dumps([{"id": 1, "description": "py issue", "file": "api.py", "line": 1}])
    )
    (deep / "stack-react-records.json").write_text(
        json.dumps([{"id": 1, "description": "tsx issue", "file": "App.tsx", "line": 1}])
    )
    # Markdown routes to the generic bucket; prime its records too so the
    # merge-resume validation (every detected stack must have records or be
    # in failed_stacks) passes.
    (deep / "stack-generic-records.json").write_text(
        json.dumps([{"id": 1, "description": "docs issue", "file": "README.md", "line": 1}])
    )

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    # Parse phase must NOT have been invoked (records already on disk).
    parse_calls = [c for c in stub.calls if "extract only actionable issues" in c["prompt"].lower()]
    assert parse_calls == [], f"unexpected parse invocations on merge resume: {len(parse_calls)}"

    # Merge agent must have run and produced the merged report.
    merge_calls = [c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert len(merge_calls) == 1
    from daydream.config import REVIEW_OUTPUT_FILE
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).exists()


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


def _write_plugin_registry(config_dir: Path, plugin_names: list[str]) -> None:
    registry = config_dir / "plugins" / "installed_plugins.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        '{"version": 2, "plugins": {'
        + ", ".join(f'"{name}@marketplace": []' for name in plugin_names)
        + "}}"
    )


def test_get_installed_skills_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All per-stack beagle plugins present -> full SKILL_MAP coverage."""
    from daydream.config import SKILL_MAP
    from daydream.deep.orchestrator import get_installed_skills

    plugin_names = [skill.split(":", 1)[0] for skill in SKILL_MAP.values()]
    _write_plugin_registry(tmp_path, plugin_names)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    assert get_installed_skills() == set(SKILL_MAP.keys())


def test_get_installed_skills_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing beagle-go plugin -> go is excluded from availability."""
    from daydream.deep.orchestrator import get_installed_skills

    _write_plugin_registry(tmp_path, ["beagle-python", "beagle-react"])
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    result = get_installed_skills()
    assert result == {"python", "react"}


def test_get_installed_skills_missing_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing registry file -> None (signals 'unknown' to the caller)."""
    from daydream.deep.orchestrator import get_installed_skills

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    assert get_installed_skills() is None


def test_get_installed_skills_malformed_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unparseable registry -> None (fall back to optimistic availability)."""
    from daydream.deep.orchestrator import get_installed_skills

    registry = tmp_path / "plugins" / "installed_plugins.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("not json {{{")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    assert get_installed_skills() is None


def test_run_deep_routes_missing_skill_to_generic(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When beagle-react is absent, React files route to the generic bucket.

    Regression: previously orchestrator passed ``set(SKILL_MAP.keys())`` as
    availability, so detect_stacks kept React as its own stack, the per-stack
    agent raised MissingSkillError, and phase_per_stack_reviews silently
    dropped the React findings.
    """
    import anyio

    from daydream.deep import detection as _detection

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target, pin_skill_availability=False)
    # Registry with only python installed -- react and markdown should route to generic.
    _write_plugin_registry(tmp_path, ["beagle-python"])
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    captured: dict[str, list[_detection.StackAssignment]] = {}
    real_detect = _detection.detect_stacks

    def _spy(files: list[str], **kwargs: Any) -> list[_detection.StackAssignment]:
        result = real_detect(files, **kwargs)
        captured["stacks"] = result
        return result

    monkeypatch.setattr("daydream.deep.orchestrator.detect_stacks", _spy)

    exit_code = anyio.run(_run_deep, multi_stack_target)
    assert exit_code == 0

    stacks = {s.stack_name for s in captured["stacks"]}
    # Python remains, but React (no skill installed) must have fallen through to generic.
    assert "python" in stacks
    assert "react" not in stacks
    assert "generic" in stacks


def test_diff_changed_files_rename_single_entry() -> None:
    """Rename diff contributes only the destination path, not both sides."""
    from daydream.deep.orchestrator import _diff_changed_files

    rename_diff = (
        "diff --git a/foo.py b/foo.ts\n"
        "similarity index 85%\n"
        "rename from foo.py\n"
        "rename to foo.ts\n"
        "--- a/foo.py\n"
        "+++ b/foo.ts\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+const x = 1;\n"
    )
    assert _diff_changed_files(rename_diff) == ["foo.ts"]


def test_diff_changed_files_handles_modify_add_delete_binary() -> None:
    """Non-rename diff shapes emit exactly one path each."""
    from daydream.deep.orchestrator import _diff_changed_files

    mixed = (
        "diff --git a/keep.py b/keep.py\n"
        "--- a/keep.py\n"
        "+++ b/keep.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
        "diff --git a/new.py b/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
        "diff --git a/old.py b/old.py\n"
        "deleted file mode 100644\n"
        "--- a/old.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-x = 1\n"
        "diff --git a/logo.png b/logo.png\n"
        "index 1234..5678 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    assert _diff_changed_files(mixed) == ["keep.py", "new.py", "old.py", "logo.png"]


async def test_merge_prompt_lists_records_in_sorted_order(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-merge parse iterates sorted(per_stack_outputs.items()) so the merge
    prompt's records list is stable across runs regardless of which per-stack
    task completed first."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    merge_prompts = [c["prompt"] for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert merge_prompts, "merge agent was not invoked"
    prompt = merge_prompts[0]

    # build_merge_prompt writes records under "Per-stack parsed records:" as
    # "  - <path>" lines.
    lines = prompt.splitlines()
    start = next((i for i, line in enumerate(lines) if "per-stack parsed records:" in line.lower()), None)
    assert start is not None, "merge prompt missing per-stack records block"

    record_paths: list[str] = []
    for line in lines[start + 1:]:
        if line.startswith("  - "):
            record_paths.append(line[4:].strip())
        elif line.strip() == "":
            break
        else:
            break

    assert record_paths, "no record paths found in merge prompt"
    assert record_paths == sorted(record_paths), (
        f"records not in sorted order: {record_paths}"
    )


async def test_failed_per_stack_surfaces_to_merge_prompt_and_persists(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-stack agent failure must:
      1) persist to per-stack-failures.json under .daydream/deep/,
      2) appear in the merge prompt under an 'Uncovered stacks' block,
    so the merge agent can call it out instead of silently ignoring the gap.
    """
    import json as _json

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    # Wrap the stub's execute so the REACT per-stack prompt raises. Everything
    # else (TTT, parse, merge, other stacks) keeps the stub's normal behavior.
    original_execute = stub.execute

    def _maybe_fail(cwd, prompt, output_schema=None, continuation=None, agents=None, max_turns=None):
        pl = prompt.lower()
        if "you are reviewing the react stack" in pl:
            async def _fail():
                raise RuntimeError("simulated react failure")
                yield  # pragma: no cover -- unreachable; satisfies async-gen typing
            return _fail()
        return original_execute(cwd, prompt, output_schema, continuation, agents, max_turns=max_turns)

    stub.execute = _maybe_fail  # type: ignore[method-assign]

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    failures_p = multi_stack_target / ".daydream" / "deep" / "per-stack-failures.json"
    assert failures_p.is_file(), "failures file should be persisted for merge-resume"
    failures_payload = _json.loads(failures_p.read_text())
    assert "react" in failures_payload
    assert "simulated react failure" in failures_payload["react"]

    merge_prompts = [
        c["prompt"] for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()
    ]
    assert merge_prompts, "merge agent was not invoked"
    prompt = merge_prompts[0]
    assert "Uncovered stacks" in prompt
    assert "react" in prompt
    assert "simulated react failure" in prompt


def test_get_installed_skills_non_dict_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-dict root JSON -> None (fall back to optimistic availability).

    Regression: previously `data.get("plugins", {})` raised AttributeError
    when the registry parsed to a non-dict, aborting deep mode instead of
    returning None.
    """
    from daydream.deep.orchestrator import get_installed_skills

    registry = tmp_path / "plugins" / "installed_plugins.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("[]")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    assert get_installed_skills() is None


def test_get_installed_skills_non_dict_plugins_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`plugins` field is not a mapping -> None.

    Regression: previously iterating ``data.get("plugins", {})`` raised
    TypeError when the `plugins` field was e.g. a list, aborting deep
    mode instead of returning None.
    """
    from daydream.deep.orchestrator import get_installed_skills

    registry = tmp_path / "plugins" / "installed_plugins.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text('{"version": 2, "plugins": ["beagle-python@marketplace"]}')
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    assert get_installed_skills() is None


async def test_resume_merge_errors_on_missing_stack_records(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--start-at merge must fail loudly when a detected stack has no records.

    Regression: previously the merge branch globbed whatever ``stack-*-records.json``
    files happened to exist on disk, so a detected stack with no prior records
    would silently disappear from the merged report.
    """
    import json

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    # Prime records for python only; react and generic are missing.
    (deep / "stack-python-records.json").write_text(
        json.dumps([{"id": 1, "description": "py issue", "file": "api.py", "line": 1}])
    )

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 1

    # Merge agent must NOT have run -- the orchestrator bailed before it.
    merge_calls = [c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert merge_calls == []


async def test_resume_merge_allows_missing_records_for_failed_stacks(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stack listed in per-stack-failures.json is allowed to be missing.

    The merge agent still runs and the missing bucket is surfaced as an
    uncovered stack rather than being flagged as a records-file gap.
    """
    import json

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    (deep / "stack-python-records.json").write_text(
        json.dumps([{"id": 1, "description": "py issue", "file": "api.py", "line": 1}])
    )
    (deep / "stack-react-records.json").write_text(
        json.dumps([{"id": 1, "description": "tsx issue", "file": "App.tsx", "line": 1}])
    )
    # No records for the generic bucket, but it's listed as a prior failure.
    (deep / "per-stack-failures.json").write_text(
        json.dumps({"generic": "simulated generic failure"})
    )

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    merge_calls = [c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert len(merge_calls) == 1


async def test_resume_fix_skips_pr_post(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--start-at fix must not call post_review_to_pr_from_report.

    Regression: posting is a non-idempotent GitHub write. Calling it on every
    fix resume would produce duplicate inline reviews on the same PR.
    """
    from daydream.config import REVIEW_OUTPUT_FILE

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    post_calls: list[dict[str, Any]] = []

    async def _spy(
        target_dir: Path, report_path: Path, *, console: Any
    ) -> None:
        post_calls.append({"target_dir": target_dir, "report_path": report_path})

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _spy)

    # Prime every artifact the fix-resume gate needs.
    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    (multi_stack_target / REVIEW_OUTPUT_FILE).write_text(
        "# Review\n\n## Issues\n\n1. [api.py:1] primed issue\n   rationale\n"
    )

    exit_code = await _run_deep(multi_stack_target, start_at="fix")
    assert exit_code == 0
    assert post_calls == [], (
        f"post_review_to_pr_from_report should be skipped on --start-at fix, got {len(post_calls)} call(s)"
    )


async def test_resolve_backend_called_with_each_phase_in_deep_flow(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deep orchestrator must call _resolve_backend with each spec phase,
    not just 'review'. This is a wiring test, not a model-value test.

    Drives a full deep flow (TTT -> per-stack -> parse -> merge -> fix gate
    accepted -> fix-loop -> test -> commit) with the stub backend, and asserts
    every expected phase string appears in the captured call list.
    """
    from daydream import runner as _runner

    seen_phases: list[str] = []
    original = _runner._resolve_backend

    def spy(config, phase, cache=None):
        seen_phases.append(phase)
        return original(config, phase, cache)

    # The deep orchestrator does `from daydream.runner import _resolve_backend`
    # inside run_deep, so patching daydream.runner._resolve_backend intercepts
    # every call site once the orchestrator is wired to per-phase resolution.
    monkeypatch.setattr("daydream.runner._resolve_backend", spy)

    # Silence interactive UI but accept the fix gate so fix/test/commit run.
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.prompt_user", lambda *a, **kw: "y")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    _install_stub_backend(monkeypatch, multi_stack_target)

    # Suppress the PR post side effect.
    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

    # Stub the fix, test_and_heal, and commit_push phases so they don't try to
    # mutate the workspace / run tests, but still trigger their resolver call.
    async def _stub_fix(backend, work, item, idx, total):  # noqa: ARG001
        return None

    async def _stub_test(backend, work):  # noqa: ARG001
        return (True, 0)

    async def _stub_commit(backend, work):  # noqa: ARG001
        return None

    monkeypatch.setattr("daydream.deep.orchestrator.phase_fix", _stub_fix)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", _stub_test)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _stub_commit)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    expected_phases = {"intent", "wonder", "review", "parse", "merge", "fix", "test"}
    captured = set(seen_phases)
    missing = expected_phases - captured
    assert not missing, (
        f"Deep orchestrator missing per-phase resolver calls for {missing}; "
        f"got {sorted(captured)}"
    )
