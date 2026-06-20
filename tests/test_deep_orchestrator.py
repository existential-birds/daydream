"""Deep-mode orchestrator integration tests (plan 05-09).

Covers D-07..D-10, D-17, D-19..D-22, D-24..D-26, D-28, D-30, D-31,
D-34, D-35, D-44.

The tests share a ``_StubBackend`` that dispatches on prompt content to
simulate the full review pipeline without talking to a real SDK.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import ResultEvent, TextEvent


class _StubBackend:
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
        is_codex: bool = False,
        shared_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self._target = target
        self._is_codex = is_codex
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
        self.calls: list[dict[str, Any]] = []
        # Verdict the recommendation-verifier branch emits for issue_id=1; flip to
        # "contradicts" to exercise verdict propagation into the phase_fix prompt.
        self.verifier_verdict: str = "consistent"
        self.verifier_unverified_assumptions: list[str] = []
        # Counts test-suite invocations so a test can fail the FIRST run (driving
        # the heal loop into choice "2") and pass the SECOND.
        self.test_suite_calls: int = 0
        # When True, the test-suite branch fails the first call and passes after.
        self.fail_first_test_run: bool = False
        # Override for the merge agent's item list (None -> default three-item payload).
        self.merge_items: list[dict[str, Any]] | None = None
        # When set, the fix branch appends the prompt's marker token to this file
        # via read-modify-write with an anyio.sleep(0) interleave point -- a
        # per-ITEM fan-out would lose/reorder appends; correct per-file
        # serialization preserves marker order.
        self.fix_append_path: Path | None = None
        # When set, the fix branch raises for the matching file, isolating one
        # file-group's failure so a test can assert the others still applied.
        self.fix_fail_file: str | None = None

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        call = {
            "cwd": cwd,
            "prompt": prompt,
            "output_schema": output_schema,
            "agents": agents,
            "model": self.model,
        }
        self.calls.append(call)
        if self._shared is not None:
            self._shared.append(call)
        pl = prompt.lower()

        # TTT alternative-review -> structured output. Checked BEFORE intent: the
        # alt prompt embeds the intent summary, defeating a naive substring check.
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

        # Exploration specialists. Each returns its envelope sub-dict as
        # structured_output (keyed into results[name] by _run_specialist) plus a
        # TextEvent of raw JSON the production gate must suppress from the terminal.
        if "you are the **pattern-scanner** specialist" in pl:
            payload = {
                "conventions": [
                    {
                        "name": "OpenAPI First",
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
            payload = {
                "affected_files": [],
                "dependencies": [
                    {"source": "App.tsx", "target": "api.py", "relationship": "calls"}
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

        if "extract only actionable issues" in pl:  # phase_parse_feedback
            issue: dict[str, Any] = {"id": 1, "description": "Sample issue", "file": "api.py", "line": 1}
            # Deep per-stack parse requests severity (PER_STACK_RECORD_SCHEMA, #168);
            # emit the configured severity so a test can drive arbiter selection.
            # The structural stack parses with FEEDBACK_SCHEMA (no severity prompt),
            # so it never gets one -- matching production.
            if self.parse_severity is not None and "severity" in pl:
                issue["severity"] = self.parse_severity
                issue["confidence"] = "MEDIUM"
                issue["rationale"] = "stub"
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"issues": [issue]}, continuation=None)
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
            yield ResultEvent(structured_output={"findings": findings}, continuation=None)
            return

        # Cross-stack merge -> schema-validated item list; the host appends
        # structural findings, normalizes ids, and renders review-output.md.
        if "cross-stack merge agent" in pl:
            yield TextEvent(text="")
            if self.merge_echo_records:
                # Echo the on-disk per-stack records as merged items so the
                # rendered artifact reflects any arbiter revisions (#168).
                echoed: list[dict[str, Any]] = []
                next_id = 1
                for path_str in re.findall(r"  - (\S+-records\.json)", prompt):
                    for rec in json.loads(Path(path_str).read_text()):
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
                        },
                    ]
                },
                continuation=None,
            )
            return

        # phase_fix -> "apply" the edit by writing a sentinel file, the observable
        # consequence the --yes real-path test asserts the fix gate auto-approved.
        if pl.startswith("fix this issue"):
            m = re.search(r"^File: (.+)$", prompt, re.M)
            fixed_file = m.group(1).strip() if m else "unknown"
            # phase_fix emits an absolute path when the file exists on disk; the stub keys fixes by basename.
            fixed_name = Path(fixed_file).name
            if self.fix_fail_file is not None and fixed_name == self.fix_fail_file:
                raise RuntimeError(f"stub fix failure for {fixed_name}")
            (cwd / ".daydream-fix-applied").write_text("applied\n")  # legacy sentinel
            (cwd / f".fixed-{fixed_name.replace('.', '_')}").write_text("applied\n")
            if self.fix_append_path is not None and fixed_name == self.fix_append_path.name:
                tok = re.search(r"marker-\d+", prompt)
                cur = self.fix_append_path.read_text() if self.fix_append_path.exists() else ""
                await anyio.sleep(0)  # deterministic interleave point
                self.fix_append_path.write_text(cur + (tok.group(0) if tok else "?") + "\n")
            yield TextEvent(text="Applied the fix.")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Recommendation verifier (#83). Discriminator is the schema constant name
        # embedded by build_verification_prompt — structural, so rewording won't
        # break this branch. The stub only emits a well-formed payload; the phase
        # persists it to recommendation-verdicts.json itself.
        if "RECOMMENDATION_VERDICTS_SCHEMA" in prompt:
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

        # Test-and-heal run. The prompt is constant, so a call counter drives the
        # result: with fail_first_test_run set, the FIRST run fails (heal loop
        # reaches choice "2") and subsequent runs pass.
        if "run the project's test suite" in pl:
            self.test_suite_calls += 1
            if self.fail_first_test_run and self.test_suite_calls == 1:
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

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


def _silence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence interactive UI helpers in deep orchestrator + phases."""
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")
    # resolve_or_prompt routes through agent.prompt_user; patch it too so those
    # gates don't block on stdin.
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "n")


def _force_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the run's interactivity axis to interactive for prompt-path tests.

    ``runner.run`` now auto-resolves non-interactive from a non-TTY stdin or a
    truthy ``CI`` env var (Task 4). Under pytest, stdin is not a TTY (and ``CI``
    is set in CI), so a test that drives the REAL interactive prompt path must
    explicitly establish a TTY stdin and unset ``CI`` -- otherwise the gate
    short-circuits to its safe default and the interactive branch never runs.
    """
    monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)


def _install_stub_backend(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    *,
    is_codex: bool = False,
    pin_skill_availability: bool = True,
    enable_exploration: bool = False,
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
        enable_exploration: When True, leaves ``EXPLORATION_AVAILABLE`` True so
            the real ``pre_scan`` branch runs and the stub answers the
            specialist prompts. Default False preserves the existing behavior
            (exploration disabled) that the rest of the suite relies on.
    """
    stub = _StubBackend(target, is_codex=is_codex)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: stub)
    # Patch CodexBackend to the stub's class so the orchestrator's isinstance check
    # fires without a real Codex dependency.
    if is_codex:
        monkeypatch.setattr("daydream.deep.orchestrator.CodexBackend", _StubBackend, raising=False)
    if pin_skill_availability:
        # None -> orchestrator falls back to set(SKILL_MAP.keys()).
        monkeypatch.setattr("daydream.deep.orchestrator.get_installed_skills", lambda: None)
        if not enable_exploration:
            # Disable exploration pre-scan so it doesn't add extra backend calls.
            monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)
    return stub


def _install_model_capturing_stubs(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    *,
    parse_severity: str | None = None,
    merge_echo_records: bool = False,
    arbiter_omit_verdicts: bool = False,
) -> list[dict[str, Any]]:
    """Patch create_backend with a per-(name, model) stub factory (#168).

    Each phase resolves its own model, so the orchestrator's (name, model)
    backend cache produces a distinct stub instance per model. Every instance
    shares one model-tagged call list, letting a test assert which model ran
    each phase — the observable proof that the per-stack fan-out runs on Sonnet,
    the merge on Opus, and the arbiter on Opus exactly when it should.

    Returns the shared, model-tagged call list (one dict per execute()).
    """
    shared_calls: list[dict[str, Any]] = []

    def factory(name: str, model: str | None = None) -> _StubBackend:
        stub = _StubBackend(target, model=model or "mock-model", shared_calls=shared_calls)
        stub.parse_severity = parse_severity
        stub.merge_echo_records = merge_echo_records
        stub.arbiter_omit_verdicts = arbiter_omit_verdicts
        return stub

    monkeypatch.setattr("daydream.runner.create_backend", factory)
    monkeypatch.setattr("daydream.deep.orchestrator.get_installed_skills", lambda: None)
    monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)
    return shared_calls


async def _run_deep(target: Path, *, start_at: str = "review") -> int:
    from daydream.runner import RunConfig, run

    # cleanup=False suppresses the interactive cleanup prompt; deep is the default.
    config = RunConfig(target=str(target), start_at=start_at, cleanup=False)
    return await run(config)


def _merge_item(item_id: int, file: str, severity: str, *, desc: str | None = None) -> dict[str, Any]:
    """Build a validated 8-key merged item (shape copied from the stub default)."""
    return {
        "id": item_id,
        "lens": "per-stack",
        "file": file,
        "line": 1,
        "severity": severity,
        "description": desc if desc is not None else f"{severity} issue in {file}",
        "confidence": "MEDIUM",
        "rationale": "rationale",
    }


async def _ok(*_a: Any, **_k: Any) -> tuple[bool, int]:
    """Async stand-in for phase_test_and_heal that always passes."""
    return (True, 0)


async def _noop_commit(*_a: Any, **_k: Any) -> None:
    """Async no-op stand-in for phase_commit_push."""
    return None


async def test_run_deep_renders_prescan_summary_not_json(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-path: the pre-scan summary renders as a readable panel, not raw JSON.

    Drives ``run`` through ``run_deep`` with EXPLORATION_AVAILABLE left True so
    the real ``pre_scan`` branch executes and the stub answers the specialist
    prompts. Asserts the convention surfaces in the rendered summary and that
    no raw structured-output JSON envelope leaks to the terminal.
    """
    from rich.console import Console

    from daydream.runner import RunConfig, run

    # Add a 4th changed file so select_tier() -> "parallel" (the pattern-scanner
    # runs and its conventions reach the rendered summary).
    (multi_stack_target / "extra.py").write_text("VALUE = 2\n")
    subprocess.run(["git", "add", "."], cwd=multi_stack_target, capture_output=True, check=True)  # noqa: S603, S607 - arguments are not user-controlled
    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "commit", "-m", "add extra"], cwd=multi_stack_target, capture_output=True, check=True
    )

    _silence(monkeypatch)
    rec = Console(record=True, force_terminal=True, width=120)
    monkeypatch.setattr("daydream.deep.orchestrator.console", rec)
    _install_stub_backend(monkeypatch, multi_stack_target, enable_exploration=True)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", lambda *a, **k: _ok())
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)

    exit_code = await run(
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
    )
    assert exit_code == 0
    out = rec.export_text()
    assert "OpenAPI First" in out  # convention surfaced by the summary
    assert '{"conventions"' not in out and "pattern-scanner" not in out  # no raw JSON envelope


async def test_parallel_fix_applies_all_disjoint_files(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#3: every disjoint-file group is applied (each per-file sentinel lands).

    A serial loop + single-sentinel stub would fail this -- it asserts EVERY
    per-file ``.fixed-*`` marker, not just the last one written.
    """
    from daydream.runner import RunConfig, run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    files = ["f1.py", "f2.py", "f3.py", "f4.py"]
    stub.merge_items = [_merge_item(i + 1, f, "high") for i, f in enumerate(files)]
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", lambda *a, **k: _ok())
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)
    exit_code = await run(
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
    )
    assert exit_code == 0
    for f in files:
        assert (multi_stack_target / f".fixed-{f.replace('.', '_')}").exists()


async def test_parallel_fix_same_file_no_race(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 items on ONE file (read-modify-write append) + 1 on another. Correct
    per-file serialization keeps all markers in severity order; a per-ITEM
    fan-out would lose/interleave appends (the anyio.sleep(0) makes that race
    deterministic).

    Discrimination validated: temporarily fanning out ``phase_fix_parallel``
    per item (one task per item) makes this FAIL -- the racing read-modify-write
    append + anyio.sleep(0) drops/reorders markers in shared.py; the correct
    per-file-group serialization keeps them ordered. Reverting restores PASS.
    """
    from daydream.runner import RunConfig, run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    shared = multi_stack_target / "shared.py"
    stub.fix_append_path = shared
    stub.merge_items = [
        _merge_item(1, "shared.py", "high", desc="marker-1"),
        _merge_item(2, "shared.py", "medium", desc="marker-2"),
        _merge_item(3, "shared.py", "low", desc="marker-3"),
        _merge_item(4, "other.py", "high", desc="other"),
    ]
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", lambda *a, **k: _ok())
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)
    exit_code = await run(
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
    )
    assert exit_code == 0
    assert shared.read_text().split() == ["marker-1", "marker-2", "marker-3"]


async def test_parallel_fix_failure_isolated_returns_nonzero(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#5: a failed fix group is isolated, surfaced, and exits nonzero.

    The bad.py group raises; its sentinel must be absent while the other groups
    apply, a warning naming bad.py must surface (non-silent), commit must be
    skipped, and the run must exit 1 (locked nonzero-on-failure decision).
    """
    from daydream.runner import RunConfig, run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.fix_fail_file = "bad.py"
    stub.merge_items = [
        _merge_item(1, "good1.py", "high"),
        _merge_item(2, "bad.py", "high"),
        _merge_item(3, "good2.py", "low"),
    ]
    warnings: list[str] = []
    monkeypatch.setattr(
        "daydream.deep.orchestrator.print_warning",
        lambda console, msg, *a, **k: warnings.append(msg),
    )
    commit_calls: list[int] = []

    async def _spy_commit(backend: Any, work: Any) -> None:
        commit_calls.append(1)

    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _spy_commit)
    exit_code = await run(
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
    )
    assert exit_code == 1  # decision: nonzero on failure
    assert (multi_stack_target / ".fixed-good1_py").exists()  # other groups applied
    assert (multi_stack_target / ".fixed-good2_py").exists()
    assert not (multi_stack_target / ".fixed-bad_py").exists()  # failed group did not apply
    assert any("bad.py" in m for m in warnings)  # non-silent
    assert commit_calls == []  # no commit on failure


async def test_parallel_fix_commit_runs_once_after_all(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#6: commit stays serial and runs exactly once, after every parallel fix lands.

    Each fix writes its own ``.fixed-*`` sentinel; the spy commit records whether ALL
    sentinels already exist at commit time. ``seen_at_commit == [True]`` proves a single
    commit that observed every fix -- a regression moving commit inside the fan-out would
    see a partial set (False) or commit more than once.
    """
    from daydream.runner import RunConfig, run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    files = ["f1.py", "f2.py", "f3.py"]
    stub.merge_items = [_merge_item(i + 1, f, "high") for i, f in enumerate(files)]
    seen_at_commit: list[bool] = []

    async def _spy_commit(backend: Any, work: Any) -> None:
        seen_at_commit.append(
            all((multi_stack_target / f".fixed-{f.replace('.', '_')}").exists() for f in files)
        )

    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", lambda *a, **k: _ok())
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _spy_commit)
    exit_code = await run(
        RunConfig(target=str(multi_stack_target), assume="yes", output_mode="loop", cleanup=False)
    )
    assert exit_code == 0
    assert seen_at_commit == [True]  # exactly one commit, and every fix already landed


INTENT_SENTINEL = "SKIP_IF_NO_QUERY_IS_A_DELIBERATE_GUARD"


def _fix_prompts(stub: _StubBackend) -> list[str]:
    return [c["prompt"] for c in stub.calls if c["prompt"].startswith("Fix this issue")]


async def test_confirmed_intent_reaches_fix_prompt(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The confirmed author intent reaches every deep fix prompt so a fixer
    can't undo a deliberate decision.

    Real-path through ``runner.run`` to the fix gate (``assume="yes"``). The PR
    body carries ``INTENT_SENTINEL``; the stub's intent branch echoes it into
    the confirmed-intent file (intent_p), and the fix phase must inline that
    file's text plus the "don't undo deliberate intent" rule into each fix
    prompt. Asserts on observable fix-prompt content, not that a call happened.
    """
    from daydream.runner import RunConfig, run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    monkeypatch.setattr(
        "daydream.git_ops.gh_pr_view",
        lambda repo, pr=None: {"body": INTENT_SENTINEL},
    )
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", lambda *a, **k: _ok())
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)

    rc = await run(
        RunConfig(
            target=str(multi_stack_target),
            pr_number=7,
            assume="yes",
            output_mode="loop",
            cleanup=False,
        )
    )
    assert rc == 0
    fix_prompts = _fix_prompts(stub)
    assert fix_prompts, "expected at least one fix prompt"
    joined = "\n".join(fix_prompts)
    assert INTENT_SENTINEL in joined
    low = joined.lower()
    assert "deliberate" in low and ("do not" in low or "don't" in low)


async def test_pipeline_order(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-07: Stage order = TTT intent -> alternatives -> per-stack -> parse -> merge."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # Alt checked before intent: the alt prompt embeds the intent summary text.
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


PR_SENTINEL = "DELIBERATE_RATIO_PASS_THROUGH_IS_INTENTIONAL"


def _intent_prompt(stub: _StubBackend) -> str:
    """Recover the intent-phase prompt by its stable instruction text."""
    return next(c["prompt"] for c in stub.calls if "understand the intent of these changes" in c["prompt"].lower())


async def test_pr_body_reaches_intent_prompt(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PR description body is threaded into the initial intent prompt."""
    from daydream.runner import RunConfig, run

    _silence(monkeypatch)
    monkeypatch.setattr(
        "daydream.git_ops.gh_pr_view",
        lambda repo, pr=None: {"number": 7, "body": PR_SENTINEL},
    )
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    rc = await run(RunConfig(target=str(multi_stack_target), pr_number=7, start_at="review", cleanup=False))
    assert rc == 0
    assert PR_SENTINEL in _intent_prompt(stub)


async def test_no_pr_body_degrades_cleanly(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No PR body -> intent prompt is byte-for-byte today's behavior."""
    from daydream.runner import RunConfig, run

    _silence(monkeypatch)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda repo, pr=None: None)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    rc = await run(RunConfig(target=str(multi_stack_target), pr_number=7, start_at="review", cleanup=False))
    assert rc == 0
    assert PR_SENTINEL not in _intent_prompt(stub)
    assert "pull request description" not in _intent_prompt(stub).lower()


async def test_non_interactive_intent_prompt_carries_pr_body(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-path: the unattended (non-interactive) deep run auto-accepts the
    proposed intent with no human corrector -- and STILL threads the PR body
    into the intent prompt.

    This locks the path where the bug actually bit. The body must be wired at
    prompt-build time in ``build_intent_prompt`` (before ``run_agent``),
    independent of the confirm gate -- so it survives auto-accept. The real
    ``prompt_user`` is left intact; ``builtins.input`` is a forbidden sentinel
    proving stdin is never touched in non-interactive mode.
    """
    from daydream.agent import get_non_interactive, reset_state
    from daydream.runner import RunConfig, run

    _silence_gate_noise(monkeypatch)
    monkeypatch.setattr(
        "daydream.git_ops.gh_pr_view",
        lambda repo, pr=None: {"number": 7, "body": PR_SENTINEL},
    )
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

    def _forbidden_input(*_a: Any, **_kw: Any) -> str:
        raise AssertionError("input() was called in non-interactive mode -- stdin must not be touched")

    monkeypatch.setattr("builtins.input", _forbidden_input)

    reset_state()
    rc = -1
    try:
        assert get_non_interactive() is False
        config = RunConfig(
            target=str(multi_stack_target),
            pr_number=7,
            start_at="review",
            cleanup=False,
            non_interactive=True,
        )
        rc = await run(config)
        assert get_non_interactive() is True
    finally:
        reset_state()

    assert rc == 0
    assert PR_SENTINEL in _intent_prompt(stub)


@pytest.mark.asyncio
async def test_non_open_pr_state_suppresses_pr_body(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When gh_pr_view returns a non-OPEN state (CLOSED or MERGED), the
    orchestrator must NOT thread the PR body into the intent prompt — trusting
    a stale description would be wrong.  Asserts on the observable prompt
    content, not on internal state."""
    from daydream.runner import RunConfig, run

    _silence(monkeypatch)
    for state in ("CLOSED", "MERGED"):
        monkeypatch.setattr(
            "daydream.git_ops.gh_pr_view",
            lambda repo, pr=None, _s=state: {"number": 7, "body": PR_SENTINEL, "state": _s},
        )
        stub = _install_stub_backend(monkeypatch, multi_stack_target)

        rc = await run(RunConfig(target=str(multi_stack_target), pr_number=7, start_at="review", cleanup=False))
        assert rc == 0
        intent = _intent_prompt(stub)
        assert PR_SENTINEL not in intent, f"PR body must be suppressed when state={state!r}"
        assert "pull request description" not in intent.lower(), (
            f"PR section header must be absent when state={state!r}"
        )


async def test_fresh_context_per_stage(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-08: Each stage = a distinct Backend.execute call (no continuation reuse)."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    # At minimum: intent + alternatives + 3 per-stack + 3 parse + 1 merge = 9 distinct calls.
    assert len(stub.calls) >= 9
    # Each stage fires a distinct execute call -- prompts must be unique.
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
    review_files = list(deep.glob("stack-*-review.md"))
    records_files = list(deep.glob("stack-*-records.json"))
    assert review_files, "expected at least one stack-*-review.md"
    assert records_files, "expected at least one stack-*-records.json"
    assert (deep / "dedup-candidates.json").exists()


async def test_per_stack_context_isolation(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-10: per-stack prompts don't embed other stacks' file lists."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    per_stack_prompts = [
        c["prompt"] for c in stub.calls if "you are reviewing the" in c["prompt"].lower()
    ]
    assert per_stack_prompts, "expected per-stack prompts"
    # Each prompt should mention its own stack's file but NOT foreign files.
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

    # The fixture's diff is mixed, so the generic bucket is NOT docs-only (no
    # notice). Contract: a generic-fallback prompt is emitted for README.md.
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
    # Numbering continues: 1., 2. in ## Issues then 3. in ## Cross-Stack Issues.
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
    # The fix gate short-circuits to decline under non-TTY/CI; this test asserts
    # the interactive prompt path, so pin interactivity on.
    _force_interactive(monkeypatch)

    asked: list[str] = []

    def _record_prompt(console, message, default=""):
        asked.append(message)
        return "n"  # decline the fix gate

    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    # resolve_or_prompt routes through agent.prompt_user; capture it there.
    monkeypatch.setattr("daydream.agent.prompt_user", _record_prompt)
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    assert any("fix" in msg.lower() or "apply" in msg.lower() for msg in asked)


async def test_yes_auto_applies_fix(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Task 6 real-path: ``--yes`` (assume="yes") auto-applies fixes without prompting.

    Drives ``runner.run`` through the deep orchestrator's fix gate with
    ``assume="yes"``. The gate must NOT call ``prompt_user`` and MUST proceed to
    ``phase_fix`` — the observable consequence is the sentinel file the stub
    writes when it receives a fix prompt.
    """
    from daydream.runner import RunConfig, run

    _install_stub_backend(monkeypatch, multi_stack_target)

    fix_marker = multi_stack_target / ".daydream-fix-applied"
    assert not fix_marker.exists()

    prompt_calls: list[tuple[Any, ...]] = []

    def _record_prompt(console, message, default=""):
        prompt_calls.append((message, default))
        return default

    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    # The fix gate routes through agent.prompt_user; under --yes it must never
    # be reached. The intent gate must also be suppressed -- fail loudly if hit.
    monkeypatch.setattr("daydream.agent.prompt_user", _record_prompt)
    monkeypatch.setattr(
        "daydream.phases.prompt_user",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("phases.prompt_user called under --yes")),
    )

    config = RunConfig(
        target=str(multi_stack_target),
        assume="yes",
        output_mode="loop",
        cleanup=False,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert not any(
        "apply" in msg.lower() or "fix" in msg.lower() for msg, _ in prompt_calls
    ), f"fix gate prompted under --yes: {prompt_calls}"
    # Observable consequence: the fix landed.
    assert fix_marker.exists(), "phase_fix never ran -> --yes did not auto-apply"


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
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")
    _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    assert len(captured) == 1, "pre-flight notice must fire exactly once"
    notice = captured[0]
    assert len(notice["stages"]) == 5
    # Agent count = 2 TTT + N per-stack + N parse + 1 merge + 1 arbiter;
    # fixture yields N=4 (python + react + generic + structure), so 2 + 2*4 + 1 + 1 = 12.
    assert notice["agent_count"] == 12
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
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "n")
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
    # Fixture yields >= 2 non-generic buckets + 1 generic.
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
    # Prime records but NOT the review.md files -- resume must consume records.json.
    (deep / "stack-python-records.json").write_text(
        json.dumps([{"id": 1, "description": "py issue", "file": "api.py", "line": 1}])
    )
    (deep / "stack-react-records.json").write_text(
        json.dumps([{"id": 1, "description": "tsx issue", "file": "App.tsx", "line": 1}])
    )
    # Markdown routes to the generic bucket; prime its records so merge-resume
    # validation (every detected stack must have records or be a failure) passes.
    (deep / "stack-generic-records.json").write_text(
        json.dumps([{"id": 1, "description": "docs issue", "file": "README.md", "line": 1}])
    )
    # Structure meta-stack also runs in production -- prime its records.
    (deep / "stack-structure-records.json").write_text(
        json.dumps(
            [{"id": 1, "description": "structural issue", "file": "api.py", "line": 1}]
        )
    )

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    # Parse phase must NOT run (records already on disk).
    parse_calls = [c for c in stub.calls if "extract only actionable issues" in c["prompt"].lower()]
    assert parse_calls == [], f"unexpected parse invocations on merge resume: {len(parse_calls)}"

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
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")
    _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0
    stage_numbers = {c[0] for c in progress_calls}
    assert stage_numbers == {1, 2, 3, 4, 5}
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

    # Records appear under "Per-stack parsed records:" as "  - <path>" lines.
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

    # Wrap execute so only the REACT per-stack prompt raises; everything else
    # keeps the stub's normal behavior.
    original_execute = stub.execute

    def _maybe_fail(
        cwd, prompt, output_schema=None, continuation=None, agents=None,
        max_turns=None, read_only=False,
    ):
        pl = prompt.lower()
        if "you are reviewing the react stack" in pl:
            async def _fail():
                raise RuntimeError("simulated react failure")
                yield  # pragma: no cover -- unreachable; satisfies async-gen typing
            return _fail()
        return original_execute(
            cwd, prompt, output_schema, continuation, agents,
            max_turns=max_turns, read_only=read_only,
        )

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
    # Structure meta-stack also runs in production -- prime its records.
    (deep / "stack-structure-records.json").write_text(
        json.dumps(
            [{"id": 1, "description": "structural issue", "file": "api.py", "line": 1}]
        )
    )
    # No records for the generic bucket, but it's listed as a prior failure.
    (deep / "per-stack-failures.json").write_text(
        json.dumps({"generic": "simulated generic failure"})
    )

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    merge_calls = [c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert len(merge_calls) == 1


async def test_orchestrator_threads_structural_records_to_merge(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structural records ride the merge prompt as a separate input and are
    excluded from the dedup pre-filter so they don't get silently collapsed
    against language-stack findings.

    Drives the merge resume path (start_at="merge") with pre-written records
    JSONs including a structure record carrying a sentinel description, then
    asserts (a) the merge prompt receives structural_records_path pointing at
    the structure records file, (b) the dedup input lists do NOT contain the
    sentinel structural record.
    """
    import json as _json

    from daydream.deep import dedup as _dedup
    from daydream.deep import prompts as _prompts

    _real_build_merge = _prompts.build_merge_prompt

    captured_merge: dict = {}
    captured_dedup_records: dict = {}
    captured_record_dedup: dict = {}

    real_build_dedup = _dedup.build_dedup_candidates
    real_build_record_dedup = _dedup.build_record_dedup_candidates

    def _capture_merge(**kwargs):
        captured_merge.update(kwargs)
        return _real_build_merge(**kwargs)

    def _capture_dedup(records, alt_issues):
        captured_dedup_records["records"] = list(records)
        return real_build_dedup(records, alt_issues)

    def _capture_record_dedup(records, sources):
        captured_record_dedup["records"] = list(records)
        captured_record_dedup["sources"] = list(sources)
        return real_build_record_dedup(records, sources=sources)

    monkeypatch.setattr("daydream.deep.prompts.build_merge_prompt", _capture_merge)
    monkeypatch.setattr(
        "daydream.deep.orchestrator.build_dedup_candidates", _capture_dedup
    )
    monkeypatch.setattr(
        "daydream.deep.orchestrator.build_record_dedup_candidates",
        _capture_record_dedup,
    )

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    (deep / "stack-python-records.json").write_text(
        _json.dumps(
            [{"id": "py-1", "description": "py issue", "file": "api.py", "line": 1}]
        )
    )
    (deep / "stack-react-records.json").write_text(
        _json.dumps(
            [{"id": "react-1", "description": "tsx issue", "file": "App.tsx", "line": 1}]
        )
    )
    (deep / "stack-generic-records.json").write_text(
        _json.dumps(
            [{"id": "generic-1", "description": "docs issue", "file": "README.md", "line": 1}]
        )
    )
    # Structural record carries a sentinel id so we can verify it never lands
    # in the dedup input lists.
    (deep / "stack-structure-records.json").write_text(
        _json.dumps(
            [
                {
                    "id": "structure-1",
                    "description": "1000-line file budget violated",
                    "file": "api.py",
                    "line": 1,
                }
            ]
        )
    )

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    # (1) Merge prompt received structural_records_path; per_stack_records_paths
    #     must NOT include the structural file (it rides as its own argument).
    assert captured_merge.get("structural_records_path") is not None
    assert captured_merge["structural_records_path"].name == "stack-structure-records.json"
    per_stack_paths = captured_merge["per_stack_records_paths"]
    assert all(
        p.name != "stack-structure-records.json" for p in per_stack_paths
    ), f"structural records must be partitioned out: {per_stack_paths}"

    # (2) The structural sentinel record must NOT appear in either dedup input.
    def _has_structure(records: list) -> bool:
        return any(str(r.get("id", "")).startswith("structure") for r in records)

    assert not _has_structure(captured_dedup_records["records"]), (
        f"structural records leaked into build_dedup_candidates: "
        f"{captured_dedup_records['records']}"
    )
    assert not _has_structure(captured_record_dedup["records"]), (
        f"structural records leaked into build_record_dedup_candidates: "
        f"{captured_record_dedup['records']}"
    )
    # And the sources list must stay parallel to the filtered records list.
    assert len(captured_record_dedup["sources"]) == len(captured_record_dedup["records"])


async def test_orchestrator_threads_structural_records_to_merge_fresh_run(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh-run path (no start_at) applies the same structural partition.

    Mirrors ``test_orchestrator_threads_structural_records_to_merge`` but lets
    the pipeline execute the pre-merge parse loop instead of the resume loop,
    so a divergence between the two code paths would surface here.
    """
    from daydream.deep import dedup as _dedup
    from daydream.deep import prompts as _prompts

    _real_build_merge = _prompts.build_merge_prompt

    captured_merge: dict = {}
    captured_record_dedup: dict = {}

    real_build_dedup = _dedup.build_dedup_candidates
    real_build_record_dedup = _dedup.build_record_dedup_candidates

    def _capture_merge(**kwargs):
        captured_merge.update(kwargs)
        return _real_build_merge(**kwargs)

    def _capture_dedup(records, alt_issues):
        return real_build_dedup(records, alt_issues)

    def _capture_record_dedup(records, sources):
        captured_record_dedup["records"] = list(records)
        captured_record_dedup["sources"] = list(sources)
        return real_build_record_dedup(records, sources=sources)

    monkeypatch.setattr("daydream.deep.prompts.build_merge_prompt", _capture_merge)
    monkeypatch.setattr(
        "daydream.deep.orchestrator.build_dedup_candidates", _capture_dedup
    )
    monkeypatch.setattr(
        "daydream.deep.orchestrator.build_record_dedup_candidates",
        _capture_record_dedup,
    )

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # Structural records file lives under the deep artifact dir.
    assert captured_merge.get("structural_records_path") is not None
    assert captured_merge["structural_records_path"].name == "stack-structure-records.json"
    per_stack_paths = captured_merge["per_stack_records_paths"]
    assert all(
        p.name != "stack-structure-records.json" for p in per_stack_paths
    ), f"structural records must be partitioned out (fresh run): {per_stack_paths}"

    # Fresh-run populates record_sources with stack_name, so the partition drops
    # every entry whose source == 'structure'; sources stay parallel to records.
    assert "structure" not in captured_record_dedup["sources"]
    assert len(captured_record_dedup["sources"]) == len(captured_record_dedup["records"])


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

    # Prime the fix-resume artifacts: the verifier and fix gate both read the
    # canonical merged-items.json, so prime it alongside the markdown report.
    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    (multi_stack_target / REVIEW_OUTPUT_FILE).write_text(
        "# Review\n\n## Issues\n\n1. [api.py:1] primed issue\n   rationale\n"
    )
    (deep / "merged-items.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": 1,
                        "lens": "per-stack",
                        "file": "api.py",
                        "line": 1,
                        "severity": "medium",
                        "description": "primed issue",
                        "confidence": "MEDIUM",
                        "rationale": "rationale",
                    }
                ]
            }
        )
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

    # run_deep imports _resolve_backend from daydream.runner, so patching it there
    # intercepts every call site under per-phase resolution.
    monkeypatch.setattr("daydream.runner._resolve_backend", spy)

    # Accept the fix gate so fix/test/commit run; pin interactivity so the "y"
    # stub is honoured instead of the unattended decline default.
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    _install_stub_backend(monkeypatch, multi_stack_target)

    # Suppress the PR post side effect.
    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

    # Stub fix/test_and_heal/commit_push so they don't mutate the workspace or run
    # tests, but still trigger their resolver call.
    async def _stub_fix(backend, work, item, idx, total, **kwargs):  # noqa: ARG001
        return None

    async def _stub_test(backend, work, feedback_items=None):  # noqa: ARG001
        return (True, 0)

    async def _stub_commit(backend, work):  # noqa: ARG001
        return None

    monkeypatch.setattr("daydream.phases.phase_fix", _stub_fix)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", _stub_test)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _stub_commit)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    expected_phases = {"intent", "wonder", "review", "parse", "merge", "fix", "test", "verify"}
    captured = set(seen_phases)
    missing = expected_phases - captured
    assert not missing, (
        f"Deep orchestrator missing per-phase resolver calls for {missing}; "
        f"got {sorted(captured)}"
    )


async def test_verifier_runs_after_merge_before_fix(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recommendation verifier runs as a sub-step of the fix gate.

    Asserts:
      1. The verifier prompt was dispatched through the stub backend.
      2. Ordering: merge call index < verifier call index < first fix call.
      3. The verdicts JSON lands on disk at the expected artifacts path.

    Requires the y/N gate to accept ("y") so the fix loop runs and the
    fix-call index exists to compare against.
    """
    from daydream.deep.artifacts import verdicts_path

    # Accept the fix gate so the fix loop runs; pin interactivity so the gate
    # honours the "y" stub.
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_verification_summary", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    # Suppress the PR post and don't mutate the workspace in test_and_heal /
    # commit_push. phase_fix stays REAL so verdict propagation is observable.
    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    async def _stub_test(backend, work, feedback_items=None):  # noqa: ARG001
        return (True, 0)

    async def _stub_commit(backend, work):  # noqa: ARG001
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", _stub_test)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _stub_commit)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    merge_idx: int | None = None
    verifier_idx: int | None = None
    first_fix_idx: int | None = None
    for idx, call in enumerate(stub.calls):
        pl = call["prompt"].lower()
        if merge_idx is None and "cross-stack merge agent" in pl:
            merge_idx = idx
        elif verifier_idx is None and "recommendation-verifier" in pl:
            verifier_idx = idx
        elif first_fix_idx is None and pl.startswith("fix this issue:"):
            first_fix_idx = idx

    assert verifier_idx is not None, "verifier prompt was not dispatched"
    assert merge_idx is not None, "merge prompt was not dispatched"
    assert first_fix_idx is not None, "no fix prompt dispatched -- fix loop did not run"
    assert merge_idx < verifier_idx < first_fix_idx, (
        f"expected merge ({merge_idx}) < verifier ({verifier_idx}) < first fix "
        f"({first_fix_idx})"
    )

    # Verdicts JSON lands on disk at the orchestrator-controlled path.
    expected_path = verdicts_path(multi_stack_target / ".daydream" / "deep")
    assert expected_path == multi_stack_target / ".daydream" / "deep" / "recommendation-verdicts.json"
    assert expected_path.is_file(), f"verdicts file missing at {expected_path}"

    import json as _json
    payload = _json.loads(expected_path.read_text())
    assert payload == {
        "verdicts": [
            {
                "issue_id": 1,
                "verdict": "consistent",
                "evidence": "stub",
                "unverified_assumptions": [],
            }
        ]
    }


async def test_verifier_contradicts_propagates_to_fix_prompt(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the verifier returns `contradicts` for an issue_id matching a parsed
    feedback item, the orchestrator attaches the verdict and phase_fix inlines
    `Verifier verdict: contradicts` into the fix-agent prompt.
    """
    # Pin interactivity so the fix gate honours the "y" stub.
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_verification_summary", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    # Parsed feedback uses id=1, so this verdict matches and the orchestrator
    # attaches it to that item.
    stub.verifier_verdict = "contradicts"
    stub.verifier_unverified_assumptions = [
        "assumes endpoint returns JSON",
        "assumes caller is authenticated",
    ]

    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    async def _stub_test(backend, work, feedback_items=None):  # noqa: ARG001
        return (True, 0)

    async def _stub_commit(backend, work):  # noqa: ARG001
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", _stub_test)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _stub_commit)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    fix_prompts = [c["prompt"] for c in stub.calls if c["prompt"].lower().startswith("fix this issue:")]
    assert fix_prompts, "no fix prompt dispatched -- fix loop did not run"
    assert any("Verifier verdict: contradicts" in p for p in fix_prompts), (
        "contradicts verdict did not propagate into the fix prompt; "
        f"fix prompts seen: {fix_prompts!r}"
    )
    assert any(
        "Unverified assumptions: assumes endpoint returns JSON; "
        "assumes caller is authenticated." in p
        for p in fix_prompts
    ), (
        "unverified_assumptions did not propagate into the fix prompt; "
        f"fix prompts seen: {fix_prompts!r}"
    )


async def test_heal_loop_receives_feedback_items_in_fix_prompt(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deep mode threads parsed feedback_items into phase_test_and_heal so the
    heal loop's fix prompt names the changed files.

    Drives the REAL phase_test_and_heal: the first test-suite run reports a
    failure (so detect_test_success() is False), the heal menu reaches choice
    "2", _build_fix_prompt() runs with the feedback_items, and the second
    test-suite run reports a pass so the run completes. Asserts the resulting
    heal fix prompt (the one starting with "The tests failed.") names the
    feedback file "api.py" and carries the "Focus on the files listed above."
    scope instruction -- the observable consequence of feedback_items flowing
    parse -> orchestrator -> phase_test_and_heal -> _build_fix_prompt.
    """
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_verification_summary", lambda *a, **kw: None)
    # Drives the REAL interactive heal menu; pin interactivity so non-TTY pytest
    # stdin doesn't auto-resolve to non-interactive and bypass it.
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")

    # phases.prompt_user is shared: intent-confirmation needs "y"; the heal menu
    # ("Choice") needs "2" (fix-and-retry). Dispatch on the message arg.
    def _phases_prompt(console: Any, message: str, default: str = "") -> str:  # noqa: ARG001
        return "2" if "Choice" in message else "y"

    monkeypatch.setattr("daydream.phases.prompt_user", _phases_prompt)

    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.fail_first_test_run = True  # first run fails, second passes

    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    async def _stub_commit(backend, work):  # noqa: ARG001
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)
    # phase_test_and_heal stays REAL so feedback_items must flow through it.
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _stub_commit)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0, "deep run did not complete -- heal loop should pass on the second test run"

    # Without the orchestrator threading feedback_items, the _build_fix_prompt
    # output would lack "api.py" and the scope instruction -- the regression check.
    heal_prompts = [c["prompt"] for c in stub.calls if c["prompt"].startswith("The tests failed.")]
    assert heal_prompts, "heal loop did not dispatch a fix prompt -- choice '2' path not reached"
    heal_prompt = heal_prompts[0]
    assert "api.py" in heal_prompt, (
        "feedback file 'api.py' missing from heal fix prompt -- feedback_items "
        f"did not reach _build_fix_prompt; prompt was: {heal_prompt!r}"
    )
    assert "Focus on the files listed above." in heal_prompt, (
        "scope instruction missing from heal fix prompt -- feedback_items not "
        f"honored; prompt was: {heal_prompt!r}"
    )
    # First call failed, second passed: exactly two runs.
    assert stub.test_suite_calls == 2, (
        f"expected 2 test-suite runs (fail then pass), saw {stub.test_suite_calls}"
    )


async def test_structural_finding_reaches_fix_loop(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix gate feeds the canonical merged-items.json (structural included),
    severity-ordered, into phase_fix -- never the LLM re-parse that dropped
    structural findings.

    Observable consequence: every item that reaches phase_fix is captured. The
    structural item (lens="structural") MUST appear (not silently dropped by a
    markdown re-parse), and items MUST arrive severity-ordered (high before low,
    stable within a tier).
    """
    # Pin interactivity so the fix gate honours the "y" stub.
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_verification_summary", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    # One per-stack(high) + one per-stack(low); phase_cross_stack_merge appends
    # the structure meta-stack as structural(high), giving the required mix.
    stub.merge_items = [
        {
            "id": 1,
            "lens": "per-stack",
            "file": "api.py",
            "line": 1,
            "severity": "high",
            "description": "High-severity per-stack issue",
            "confidence": "HIGH",
            "rationale": "rationale",
        },
        {
            "id": 2,
            "lens": "per-stack",
            "file": "App.tsx",
            "line": 1,
            "severity": "low",
            "description": "Low-severity per-stack issue",
            "confidence": "LOW",
            "rationale": "rationale",
        },
    ]

    fixed: list[dict[str, Any]] = []

    async def _capture_fix(backend, work, item, idx, total, **kwargs):  # noqa: ARG001
        fixed.append(item)

    async def _stub_test(backend, work, feedback_items=None):  # noqa: ARG001
        return (True, 0)

    async def _stub_commit(backend, work):  # noqa: ARG001
        return None

    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    monkeypatch.setattr("daydream.phases.phase_fix", _capture_fix)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", _stub_test)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _stub_commit)
    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    assert any(i.get("lens") == "structural" for i in fixed), (
        "structural finding never reached phase_fix -- it was dropped before the "
        f"fix loop; items fixed: {[(i.get('lens'), i.get('severity')) for i in fixed]!r}"
    )
    sev = [str(i["severity"]) for i in fixed]
    assert sev == sorted(sev, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])


async def test_start_at_fix_recovers_merged_items(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--start-at fix with ONLY the deep-dir merged-items.json present (canonical
    repo review-output.md ABSENT) still loads items and reaches phase_fix.

    The fix gate reads merged_items_path(dd) directly -- the canonical markdown
    is render-only. The missing-input guard must distinguish "no JSON at all"
    (fail loudly) from "canonical markdown absent but JSON present" (proceed).
    This test pins the proceed case: the recovered item must reach phase_fix
    even though no review-output.md exists in the repo or the deep dir.
    """
    from daydream.config import REVIEW_OUTPUT_FILE

    # Pin interactivity so the fix gate honours the "y" stub.
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_verification_summary", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    _install_stub_backend(monkeypatch, multi_stack_target)

    fixed: list[dict[str, Any]] = []

    async def _capture_fix(backend, work, item, idx, total, **kwargs):  # noqa: ARG001
        fixed.append(item)

    async def _stub_test(backend, work, feedback_items=None):  # noqa: ARG001
        return (True, 0)

    async def _stub_commit(backend, work):  # noqa: ARG001
        return None

    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    monkeypatch.setattr("daydream.phases.phase_fix", _capture_fix)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", _stub_test)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _stub_commit)
    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

    # Prime fix-resume prerequisites EXCEPT the canonical markdown report -- only
    # the deep-dir merged-items.json exists, no review-output.md anywhere.
    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    (deep / "merged-items.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": 1,
                        "lens": "per-stack",
                        "file": "api.py",
                        "line": 1,
                        "severity": "high",
                        "description": "recovered issue",
                        "confidence": "HIGH",
                        "rationale": "rationale",
                    }
                ]
            }
        )
    )
    assert not (multi_stack_target / REVIEW_OUTPUT_FILE).exists()
    assert not (deep / "review-output.md").exists()

    exit_code = await _run_deep(multi_stack_target, start_at="fix")
    assert exit_code == 0
    assert len(fixed) >= 1, (
        "no items reached phase_fix on --start-at fix; the recovery guard bailed "
        "on the missing canonical markdown instead of loading the deep-dir "
        f"merged-items.json; items fixed: {fixed!r}"
    )
    assert fixed[0].get("description") == "recovered issue", (
        "phase_fix received an item that did not originate from the deep-dir "
        f"merged-items.json; got {fixed!r}"
    )


# Real-path integration: non-interactive / EOF-safe apply-fixes gate.
# Both tests drive the REAL deep pipeline to the apply-fixes prompt with the real
# ui.prompt_user (NOT mocked): non-interactive must short-circuit on
# get_non_interactive(); interactive must catch EOF on stdin. Both resolve to the
# safe default. Only the backend and PR post are mocked. A phase_fix spy proves
# the fix loop never ran; builtins.input fails the test if stdin is touched.


def _silence_gate_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence noise-only UI in the deep path WITHOUT mocking prompt_user.

    Unlike ``_silence``, this deliberately leaves the real ``prompt_user`` in
    both the orchestrator and phases so the apply-fixes gate runs the genuine
    production code path under test.
    """
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_verification_summary", lambda *a, **kw: None)


async def test_apply_fixes_gate_non_interactive_takes_safe_default(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-path: non-interactive deep run declines fixes and exits 0 without
    reading stdin.

    Drives ``runner.run`` -> ``run_deep`` (deep is the default dispatch) with a
    mock backend to the real ``prompt_user`` apply-fixes gate. With
    ``config.non_interactive=True`` propagated by ``run``, the gate must
    short-circuit to its "n" default -- never touching stdin -- so the run
    declines fixes and returns 0.
    """
    from daydream.agent import get_non_interactive, reset_state
    from daydream.config import REVIEW_OUTPUT_FILE
    from daydream.runner import RunConfig, run

    _silence_gate_noise(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    # The PR post runs before the gate; stub the non-idempotent GitHub write.
    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

    # Spy on phase_fix to prove fixes are NOT applied when the gate declines.
    fix_calls: list[Any] = []

    async def _spy_fix(backend, work, item, idx, total, **kwargs):  # noqa: ARG001
        fix_calls.append(item)
        return None

    monkeypatch.setattr("daydream.phases.phase_fix", _spy_fix)

    # Any stdin read in non-interactive mode is a bug -- fail loudly.
    def _forbidden_input(*_a: Any, **_kw: Any) -> str:
        raise AssertionError("input() was called in non-interactive mode -- stdin must not be touched")

    monkeypatch.setattr("builtins.input", _forbidden_input)

    reset_state()
    try:
        assert get_non_interactive() is False
        config = RunConfig(target=str(multi_stack_target), cleanup=False, non_interactive=True)
        exit_code = await run(config)
        assert get_non_interactive() is True
    finally:
        reset_state()

    assert exit_code == 0
    assert fix_calls == [], f"phase_fix ran despite the gate declining: {fix_calls!r}"
    # The gate's "report written ... exiting" path ran (report on disk before return 0).
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).is_file(), (
        "merged report missing -- the apply-fixes gate's success/exit path did not run"
    )


async def test_apply_fixes_gate_eof_declines_cleanly_no_crash(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-path: an EOF on stdin at the apply-fixes gate is caught and resolved
    to the safe default -- the deep run declines fixes and returns 0, no crash.

    This is the interactive path (``non_interactive`` False): the production
    ``prompt_user`` reaches ``input()``, which raises ``EOFError`` (closed
    stdin). The gate must catch it, return the "n" default, and exit 0 -- proving
    EOF-safety end-to-end through the real orchestrator, not just the unit
    ``prompt_user``.
    """
    from daydream.agent import get_non_interactive, reset_state
    from daydream.config import REVIEW_OUTPUT_FILE
    from daydream.runner import RunConfig, run

    _silence_gate_noise(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

    fix_calls: list[Any] = []

    async def _spy_fix(backend, work, item, idx, total, **kwargs):  # noqa: ARG001
        fix_calls.append(item)
        return None

    monkeypatch.setattr("daydream.phases.phase_fix", _spy_fix)

    # Every stdin read raises EOFError (closed stdin without the non_interactive flag).
    def _eof_input(*_a: Any, **_kw: Any) -> str:
        raise EOFError("simulated closed stdin")

    monkeypatch.setattr("builtins.input", _eof_input)

    # Pin interactivity ON so this exercises the interactive EOF branch, not the
    # auto non-interactive short-circuit non-TTY pytest stdin would trigger.
    _force_interactive(monkeypatch)

    reset_state()
    try:
        assert get_non_interactive() is False
        config = RunConfig(target=str(multi_stack_target), cleanup=False, non_interactive=False)
        # If the gate did not catch EOFError, this await would raise.
        exit_code = await run(config)
    finally:
        reset_state()

    assert exit_code == 0
    assert fix_calls == [], f"phase_fix ran despite EOF at the gate: {fix_calls!r}"
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).is_file(), (
        "merged report missing -- the apply-fixes gate's success/exit path did not run"
    )


# Git timeout under load (issue #120)


async def test_deep_run_recovers_from_transient_git_timeout(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for #120: a transient git timeout no longer fails the run.

    Under heavy host load a trivial git command in the deep preamble would
    exceed its 5s timeout, collapse to a generic ``GitError``, and the run
    would exit 1 -- making every ``assert exit_code == 0`` deep test flaky.
    With the bounded retry in ``git_ops._run_git`` the timeout is retried and
    the run completes normally.

    Drives the real production path: ``runner.run`` -> deep orchestrator ->
    ``git_ops.diff`` -> ``_run_git`` -> ``subprocess.run`` (only the backend is
    stubbed). The first git subprocess call raises ``TimeoutExpired``; every
    later call delegates to the real ``subprocess.run``.
    """
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    real_run = subprocess.run
    state = {"timed_out_once": False}

    def flaky_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        cmd = args[0] if args else kwargs.get("args", [])
        # Trip only on a real `git` invocation (these retry); leave `gh` untouched
        # so the test stays deterministic.
        is_git = isinstance(cmd, (list, tuple)) and len(cmd) and cmd[0] == "git"
        if is_git and not state["timed_out_once"]:
            state["timed_out_once"] = True
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)
        return real_run(*args, **kwargs)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", flaky_run)

    exit_code = await _run_deep(multi_stack_target)

    from daydream.config import REVIEW_OUTPUT_FILE

    assert state["timed_out_once"], "the injected git timeout never fired"
    # Survived the timeout, exited cleanly, and progressed past the diff preamble.
    assert exit_code == 0
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).is_file()


async def test_deep_run_reports_persistent_git_timeout_distinctly(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout that survives retries is surfaced as a distinct 'Git Timeout'.

    A genuine bad-ref ``GitError`` reports 'Unable to determine base branch for
    diff'. A timeout is a different failure mode (transient host load) and must
    not be misreported as that deterministic-sounding ref error (#120). Drives
    ``runner.run`` to the real orchestrator branch and captures the rendered
    error title.
    """
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    from daydream.git_ops import GitTimeoutError

    def always_timeout(*args: Any, **kwargs: Any) -> str:
        raise GitTimeoutError("git diff main...HEAD timed out after 30s (3 attempts)")

    monkeypatch.setattr("daydream.git_ops.diff", always_timeout)

    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "daydream.deep.orchestrator.print_error",
        lambda console, title, msg, *a, **kw: errors.append((title, msg)),
    )

    exit_code = await _run_deep(multi_stack_target)

    # Aborts with the timeout-specific title, NOT the misleading base-branch error.
    assert exit_code == 1
    titles = [t for t, _ in errors]
    assert "Git Timeout" in titles, f"expected a distinct Git Timeout error, got {errors!r}"
    assert "Git Error" not in titles, (
        f"a timeout was misreported as the generic base-branch error: {errors!r}"
    )


# Issue #168: Sonnet-first per-stack review with a scoped Opus arbiter.


async def test_per_stack_sonnet_merge_opus_and_arbiter_on_high_severity(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#168 real-path: drive runner.run through the production entrypoint and
    assert observable model targeting + the arbitrated finding on disk.

    The per-stack parse emits ``high`` severity, so the scoped arbiter must fire
    exactly once on Opus; the rendered merge artifact must reflect its revision.
    """
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(
        monkeypatch, multi_stack_target, parse_severity="high", merge_echo_records=True
    )

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    def models_where(predicate: Any) -> list[str | None]:
        return [c["model"] for c in calls if predicate(c["prompt"].lower())]

    # (a) Per-stack fan-out created with a Sonnet model id (N>1 multi-stack).
    per_stack_models = models_where(lambda pl: "you are reviewing the" in pl and "stack" in pl)
    assert len(per_stack_models) >= 2, f"expected an N>1 fan-out, got {per_stack_models!r}"
    assert set(per_stack_models) == {"claude-sonnet-4-6"}

    # (b) Merge backend created with an Opus model id.
    assert models_where(lambda pl: "cross-stack merge agent" in pl) == ["claude-opus-4-8"]

    # (c) Opus arbiter created exactly once when a high-severity record exists.
    assert models_where(lambda pl: "you are the arbiter" in pl) == ["claude-opus-4-8"]

    # The rendered merge artifact on disk reflects the arbitrated finding.
    report = (multi_stack_target / ".review-output.md").read_text()
    assert "ARBITRATED:" in report, f"arbitrated finding missing from report:\n{report}"


async def test_arbiter_missing_verdict_retains_high_severity_finding(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#175 real-path: a truncated/lazy arbiter that omits every verdict must NOT
    delete the high-severity finding it was selected to protect.

    The arbiter fires (high severity), but its stub returns an empty findings
    list -- no verdict for any arb_id. Fail-open means the original record is
    retained unchanged and survives into the rendered merge artifact.
    """
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(
        monkeypatch,
        multi_stack_target,
        parse_severity="high",
        merge_echo_records=True,
        arbiter_omit_verdicts=True,
    )

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # The arbiter still ran (high severity selects it) ...
    arbiter_calls = [c for c in calls if "you are the arbiter" in c["prompt"].lower()]
    assert arbiter_calls, "arbiter must run on a high-severity finding"

    # ... but with no verdict returned, the finding is retained, not dropped.
    report = (multi_stack_target / ".review-output.md").read_text()
    # The un-arbitrated description survives (no ARBITRATED: prefix was applied).
    assert "ARBITRATED:" not in report
    deep_dir = multi_stack_target / ".daydream" / "deep"
    records = [
        rec
        for path in deep_dir.glob("stack-*-records.json")
        for rec in json.loads(path.read_text())
    ]
    assert any(r.get("severity") == "high" for r in records), (
        f"high-severity record must survive a missing arbiter verdict:\n{records}"
    )


async def test_no_arbiter_when_all_findings_low_and_uncontested(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#168 real-path: when every per-stack finding is low/uncontested, NO Opus
    arbiter backend is created — but Sonnet still runs the per-stack fan-out."""
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(
        monkeypatch, multi_stack_target, parse_severity="low", merge_echo_records=True
    )

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    arbiter_calls = [c for c in calls if "you are the arbiter" in c["prompt"].lower()]
    assert arbiter_calls == [], "arbiter must not run on low/uncontested findings"

    per_stack_models = {
        c["model"] for c in calls if "you are reviewing the" in c["prompt"].lower() and "stack" in c["prompt"].lower()
    }
    assert per_stack_models == {"claude-sonnet-4-6"}


def _prime_merge_resume_records(deep: Path, *, python_severity: str | None) -> None:
    """Write the per-stack records a `--start-at merge` resume needs on disk.

    Every detected stack (python, react, generic, structure) must have a records
    file or be a recorded failure, else the resume guard returns 1. The python
    record optionally carries ``python_severity`` to drive arbiter selection.
    """
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    py_record: dict[str, Any] = {"id": 1, "description": "py issue", "file": "api.py", "line": 1}
    if python_severity is not None:
        py_record["severity"] = python_severity
        py_record["confidence"] = "HIGH"
        py_record["rationale"] = "stub"
    (deep / "stack-python-records.json").write_text(json.dumps([py_record]))
    (deep / "stack-react-records.json").write_text(
        json.dumps([{"id": 1, "description": "tsx issue", "file": "App.tsx", "line": 1}])
    )
    (deep / "stack-generic-records.json").write_text(
        json.dumps([{"id": 1, "description": "docs issue", "file": "README.md", "line": 1}])
    )
    (deep / "stack-structure-records.json").write_text(
        json.dumps([{"id": 1, "description": "structural issue", "file": "api.py", "line": 1}])
    )


async def test_merge_resume_reruns_arbiter_when_marker_absent(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#175 real-path: a `--start-at merge` resume whose on-disk records carry a
    high-severity finding and NO completion marker must re-run the arbiter.

    A crash between the parse write and the arbiter rewrite leaves unarbitrated
    high-severity records on disk; trusting them at merge would bypass the
    quality gate exactly on the riskiest findings.
    """
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(monkeypatch, multi_stack_target, merge_echo_records=True)

    deep = multi_stack_target / ".daydream" / "deep"
    _prime_merge_resume_records(deep, python_severity="high")
    assert not (deep / "arbiter-complete.marker").exists()

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    arbiter_calls = [c for c in calls if "you are the arbiter" in c["prompt"].lower()]
    assert arbiter_calls, "arbiter must re-run on merge resume when no completion marker exists"
    assert (deep / "arbiter-complete.marker").is_file(), "completion marker must be written after arbitration"

    report = (multi_stack_target / ".review-output.md").read_text()
    assert "ARBITRATED:" in report, f"arbitrated finding missing from merge-resume report:\n{report}"


async def test_merge_resume_skips_arbiter_when_marker_present(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#175 real-path: when the completion marker proves the records were already
    finalised, a `--start-at merge` resume must NOT re-run the arbiter."""
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(monkeypatch, multi_stack_target, merge_echo_records=True)

    deep = multi_stack_target / ".daydream" / "deep"
    _prime_merge_resume_records(deep, python_severity="high")
    (deep / "arbiter-complete.marker").write_text("")

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    arbiter_calls = [c for c in calls if "you are the arbiter" in c["prompt"].lower()]
    assert arbiter_calls == [], "arbiter must not re-run when the completion marker is present"
