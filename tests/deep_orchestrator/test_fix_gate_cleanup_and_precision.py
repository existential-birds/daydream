"""Fix Gate Cleanup And Precision."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream.config import REVIEW_OUTPUT_FILE
from daydream.deep import orchestrator as deep_orchestrator
from daydream.extensions.api import Stop
from daydream.git_ops import GitTimeoutError
from daydream.run_context import current_run_context
from daydream.runner import run
from tests.deep_orchestrator.support import (
    _forbidden_input,
    _install_accept_gate_pipeline,
    _merged_item_descriptions,
    _scan_trajectory_extra,
    _silence_gate_noise,
)
from tests.test_deep_orchestrator import (
    _SUPPRESSION_COLLISION_STACKS,
    MakeConfig,
    Mute,
    _accept_intent_decline_other,
    _fix_prompts,
    _force_interactive,
    _install_model_capturing_stubs,
    _install_stub_backend,
    _prime_merge_resume,
    _record_issues,
    _run_deep,
    _severity_sort_key,
    _silence,
)


async def test_verifier_contradicts_propagates_to_fix_prompt(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """When the verifier returns `contradicts` for an issue_id matching a parsed feedback item, the orchestrator
    attaches the verdict and phase_fix inlines `Verifier verdict: contradicts` into the fix-agent prompt."""
    stub = _install_accept_gate_pipeline(monkeypatch, multi_stack_target, mute_side_effects)
    # Parsed feedback uses id=1, so this verdict matches and the orchestrator
    # attaches it to that item.
    stub.verifier_verdict = "contradicts"
    stub.verifier_unverified_assumptions = [
        "assumes endpoint returns JSON",
        "assumes caller is authenticated",
    ]

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # api.py carries two same-file findings, so its fix turn is batched
    # ("Fix these N issues"); the verdict for id=1 rides in that batched prompt.
    fix_prompts = [c["prompt"] for c in stub.calls if c["prompt"].lower().startswith(("fix this issue:", "fix these"))]
    assert fix_prompts, "no fix prompt dispatched -- fix loop did not run"
    assert any("Verifier verdict: contradicts" in p for p in fix_prompts), (
        f"contradicts verdict did not propagate into the fix prompt; fix prompts seen: {fix_prompts!r}"
    )
    assert any(
        "Unverified assumptions: assumes endpoint returns JSON; assumes caller is authenticated." in p
        for p in fix_prompts
    ), f"unverified_assumptions did not propagate into the fix prompt; fix prompts seen: {fix_prompts!r}"
    # AC2 regression guard: when the gate accepts, verify runs and the verdicts
    # artifact lands on disk (same path asserted by the ordering test).
    assert (multi_stack_target / ".daydream" / "deep" / "recommendation-verdicts.json").is_file(), (
        "verdicts file missing -- verify did not run when the gate accepted"
    )


async def test_heal_loop_receives_feedback_items_in_fix_prompt(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """Deep mode threads parsed feedback_items into phase_test_and_heal so the heal loop's fix prompt names the
    changed files."""
    _silence(monkeypatch, prompts=False)
    # Drives the REAL interactive heal menu; pin interactivity so non-TTY pytest
    # stdin doesn't auto-resolve to non-interactive and bypass it.
    _force_interactive(monkeypatch)

    # The single gateway handles both intent confirmation and the heal menu.
    def _prompt(_console: Any, message: str, _default: str = "") -> str:
        return "2" if "Choice" in message else "y"

    monkeypatch.setattr("daydream.run_context._prompt_user", _prompt)

    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.fail_first_test_run = True  # first run fails, second passes

    # phase_test_and_heal stays REAL so feedback_items must flow through it.
    mute_side_effects(heal=False)

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
        f"scope instruction missing from heal fix prompt -- feedback_items not honored; prompt was: {heal_prompt!r}"
    )
    # The heal stub also writes an unauthorized sentinel. The post-heal guard
    # removes it, invalidating the retry's tree identity, so one shared no-heal
    # final test is mandatory: fail, healed pass, stable pass.
    assert stub.test_suite_calls == 3, (
        f"expected 3 test-suite runs (fail, healed pass, stable pass), saw {stub.test_suite_calls}"
    )


async def test_structural_finding_reaches_fix_loop(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """The fix gate feeds the canonical merged-items.json (structural included), severity-ordered, into phase_fix
    -- never the LLM re-parse that dropped structural findings."""
    stub = _install_accept_gate_pipeline(monkeypatch, multi_stack_target, mute_side_effects)
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
            "evidence": "api.py:1",
        },
        {
            "id": 2,
            "lens": "per-stack",
            "file": "App.tsx",
            "line": 1,
            "severity": "low",
            "description": "Low-severity per-stack issue",
            "confidence": "MEDIUM",
            "rationale": "rationale",
            "evidence": "App.tsx:1",
        },
    ]

    fixed: list[dict[str, Any]] = []

    # Capture at the batched dispatch point: phase_fix_parallel now hands every
    # file-group (single- or multi-item) to phase_fix_batched, so this is where
    # every item that reaches the fix loop is observable.
    async def _capture_fix(backend: Any, work: Any, items: Any, item_nums: Any, total: Any, **kwargs: Any) -> None:  # noqa: ARG001
        fixed.extend(items)

    monkeypatch.setattr("daydream.phases.phase_fix_batched", _capture_fix)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    assert any(i.get("lens") == "structural" for i in fixed), (
        "structural finding never reached phase_fix -- it was dropped before the "
        f"fix loop; items fixed: {[(i.get('lens'), i.get('severity')) for i in fixed]!r}"
    )
    sev = [str(i["severity"]) for i in fixed]
    assert sev == sorted(sev, key=_severity_sort_key)


def test_severity_sort_key_names_unknown_value() -> None:
    """Should-Have (R3): an unknown/absent severity in a fixture errors with a
    message that names the value, instead of a bare ``KeyError``."""
    with pytest.raises(ValueError, match="bogus"):
        _severity_sort_key("bogus")
    with pytest.raises(ValueError, match="unexpected severity in fixture"):
        _severity_sort_key("")


async def test_start_at_fix_recovers_merged_items(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """--start-at fix with ONLY the deep-dir merged-items.json present (canonical repo review-output.md ABSENT)
    still loads items and reaches phase_fix."""

    _install_accept_gate_pipeline(monkeypatch, multi_stack_target, mute_side_effects)

    fixed: list[dict[str, Any]] = []

    async def _capture_fix(backend: Any, work: Any, item: Any, idx: Any, total: Any, **kwargs: Any) -> None:  # noqa: ARG001
        fixed.append(item)

    monkeypatch.setattr("daydream.phases.phase_fix", _capture_fix)

    # Prime fix-resume prerequisites EXCEPT the canonical markdown report -- only
    # the deep-dir merged-items.json exists, no review-output.md anywhere.
    deep = _prime_merge_resume(multi_stack_target)
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
        f"phase_fix received an item that did not originate from the deep-dir merged-items.json; got {fixed!r}"
    )
    # AC5 regression guard: a --start-at fix resume that applies fixes (gate
    # accepted) still produces verdicts -- verify runs post-gate-accept on resume.
    assert (multi_stack_target / ".daydream" / "deep" / "recommendation-verdicts.json").is_file(), (
        "verdicts file missing on --start-at fix resume -- verify must run when the gate accepts and fixes are applied"
    )


async def test_apply_fixes_gate_non_interactive_takes_safe_default(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real-path: non-interactive deep run declines fixes and exits 0 without reading stdin."""

    _silence_gate_noise(monkeypatch)
    # The PR post runs before the gate; stub the non-idempotent GitHub write.
    mute_side_effects()
    _install_stub_backend(monkeypatch, multi_stack_target)

    # Spy on phase_fix to prove fixes are NOT applied when the gate declines.
    fix_calls: list[Any] = []

    async def _spy_fix(backend: Any, work: Any, item: Any, idx: Any, total: Any, **kwargs: Any) -> None:  # noqa: ARG001
        fix_calls.append(item)
        return None

    monkeypatch.setattr("daydream.phases.phase_fix", _spy_fix)

    # Any stdin read in non-interactive mode is a bug -- fail loudly.
    monkeypatch.setattr("builtins.input", _forbidden_input)

    traj = tmp_path / "trajectory.json"
    assert current_run_context() is None
    exit_code = await run(make_config(multi_stack_target, trajectory_path=traj))
    assert current_run_context() is None

    assert exit_code == 0
    assert fix_calls == [], f"phase_fix ran despite the gate declining: {fix_calls!r}"
    # The gate's "report written ... exiting" path ran (report on disk before return 0).
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).is_file(), (
        "merged report missing -- the apply-fixes gate's success/exit path did not run"
    )

    # AC1: the verifier never ran because the gate declined. No verify phase in
    # the trajectory, and no recommendation-verdicts.json artifact on disk.
    run_root = multi_stack_target / ".daydream"
    phases = _scan_trajectory_extra(run_root, traj, "daydream_phase")
    assert "verify" not in phases, f"verify phase ran despite the gate declining; phases: {phases!r}"
    verdicts_file = multi_stack_target / ".daydream" / "deep" / "recommendation-verdicts.json"
    assert not verdicts_file.exists(), (
        f"recommendation-verdicts.json exists despite declined gate -- verify must "
        f"not run when fixes are not applied; found {verdicts_file}"
    )


async def test_apply_fixes_gate_eof_declines_cleanly_no_crash(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real-path: an EOF on stdin at the apply-fixes gate is caught and resolved to the safe default -- the deep
    run declines fixes and returns 0, no crash."""

    _silence_gate_noise(monkeypatch)
    mute_side_effects()
    _install_stub_backend(monkeypatch, multi_stack_target)

    fix_calls: list[Any] = []

    async def _spy_fix(backend: Any, work: Any, item: Any, idx: Any, total: Any, **kwargs: Any) -> None:  # noqa: ARG001
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

    assert current_run_context() is None
    # If the gate did not catch EOFError, this await would raise.
    exit_code = await run(make_config(multi_stack_target, non_interactive=False))
    assert current_run_context() is None

    assert exit_code == 0
    assert fix_calls == [], f"phase_fix ran despite EOF at the gate: {fix_calls!r}"
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).is_file(), (
        "merged report missing -- the apply-fixes gate's success/exit path did not run"
    )


async def test_apply_fixes_gate_interactive_yes_applies_fixes(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real-path: a typed ``y`` at the apply-fixes gate runs the fix loop."""

    _silence_gate_noise(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    _force_interactive(monkeypatch)

    reads: list[str] = []

    def _yes_input(*_a: Any, **_kw: Any) -> str:
        reads.append("y")
        return "y"

    monkeypatch.setattr("builtins.input", _yes_input)

    assert current_run_context() is None
    exit_code = await run(
        make_config(
            multi_stack_target,
            non_interactive=False,
            precision_mode=True,
            output_mode="loop",
        )
    )
    assert current_run_context() is None

    assert exit_code == 0
    assert reads, "the apply-fixes gate never reached input() -- no prompt was answered"
    assert _fix_prompts(stub), (
        "a typed 'y' at the apply-fixes gate did not reach phase_fix -- the gate declined despite an affirmative answer"
    )


@pytest.mark.parametrize(
    ("shallow", "cleanup", "report_exists"),
    [
        pytest.param(True, True, False, id="shallow-cleanup"),
        pytest.param(True, False, True, id="shallow-keep"),
        pytest.param(False, True, False, id="deep-cleanup"),
    ],
)
async def test_cleanup_flag_controls_review_report_on_success(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
    shallow: bool,
    cleanup: bool,
    report_exists: bool,
) -> None:
    """Successful shallow and deep runs follow the explicit cleanup flag."""

    _install_stub_backend(monkeypatch, multi_stack_target)
    mute_side_effects()
    assert await run(make_config(multi_stack_target, shallow=shallow, cleanup=cleanup, assume="yes")) == 0
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).exists() is report_exists


async def test_cleanup_none_unattended_defaults_to_keep(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """#330 R2/#6: an unspecified cleanup flag on an unattended run defaults to KEEPING the report (the old
    ``safe_default=False``), without touching stdin."""

    _install_stub_backend(monkeypatch, multi_stack_target)
    mute_side_effects()

    def _forbidden_input(*_a: Any, **_kw: Any) -> str:
        raise AssertionError("input() called in unattended mode -- cleanup gate must not read stdin")

    monkeypatch.setattr("builtins.input", _forbidden_input)

    report = multi_stack_target / REVIEW_OUTPUT_FILE
    exit_code = await run(make_config(multi_stack_target, output_mode="review", cleanup=None))

    assert exit_code == 0
    assert report.exists(), "cleanup=None unattended must keep the report (safe_default=False)"


async def test_cleanup_none_interactive_prompts_before_keeping(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """#330 R2/#6: with cleanup unspecified and interactive stdin, the terminal
    step prompts the user (the old shallow preamble's question); a "n" answer
    keeps the report."""

    _install_stub_backend(monkeypatch, multi_stack_target)
    mute_side_effects()
    # Pin interactivity ON so the cleanup gate reaches the real prompt_user seam
    # (review mode has no fix cycle, so this is the run's only prompt).
    _force_interactive(monkeypatch)

    asked: list[str] = []

    def _record_prompt(_console: Any, message: str, _default: str = "") -> str:
        if "understanding correct" in message.lower():
            return "y"
        asked.append(message)
        return "n"  # decline cleanup

    monkeypatch.setattr("daydream.run_context._prompt_user", _record_prompt)

    report = multi_stack_target / REVIEW_OUTPUT_FILE
    exit_code = await run(make_config(multi_stack_target, output_mode="review", cleanup=None, non_interactive=False))

    assert exit_code == 0
    assert any("cleanup" in msg.lower() for msg in asked), (
        f"cleanup=None interactive run must prompt; saw prompts: {asked!r}"
    )
    assert report.exists(), "declining the cleanup prompt must keep .review-output.md"


@pytest.mark.parametrize(("cleanup", "expected_exists"), [(True, False), (False, True)])
async def test_cleanup_gate_declines_honors_cleanup_flag(
    cleanup: bool,
    expected_exists: bool,
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """#335 real-path: ``--cleanup`` is honored even when the fix gate declines."""

    _install_stub_backend(monkeypatch, multi_stack_target)
    mute_side_effects()
    # Pin interactivity ON so the fix-gate prompt path runs (resolve_gate
    # returns None when interactive with no assumption, then prompts).
    _force_interactive(monkeypatch)

    monkeypatch.setattr("daydream.run_context._prompt_user", _accept_intent_decline_other)

    report = multi_stack_target / REVIEW_OUTPUT_FILE
    exit_code = await run(make_config(multi_stack_target, cleanup=cleanup, non_interactive=False))

    assert exit_code == 0
    assert report.exists() is expected_exists, (
        f"cleanup={cleanup} must {'remove' if expected_exists is False else 'keep'} "
        ".review-output.md when the fix gate declines"
    )


async def test_cleanup_skips_on_failure_keeps_evidence(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """#335 real-path: a non-zero exit skips ``--cleanup`` so evidence survives."""


    _install_stub_backend(monkeypatch, multi_stack_target)
    mute_side_effects()

    async def _fail_fix_gate(_ctx: object) -> Stop:
        return Stop(1)

    # build_registry() reads deep.STEPS fresh on every run(), so swapping the
    # fix-gate step's body makes the real runner entry fail cleanly after the
    # report is written, without touching the flow engine or any other step.
    patched_steps = tuple(
        dataclasses.replace(step, run=_fail_fix_gate) if step.name == "fix-gate" else step
        for step in deep_orchestrator.STEPS
    )
    monkeypatch.setattr(deep_orchestrator, "STEPS", patched_steps)

    report = multi_stack_target / REVIEW_OUTPUT_FILE
    exit_code = await run(make_config(multi_stack_target, cleanup=True))

    assert exit_code == 1
    assert report.exists(), "a failed run must keep .review-output.md as evidence even with --cleanup"


async def test_deep_run_recovers_from_transient_git_timeout(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for #120: a transient git timeout no longer fails the run."""
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


    assert state["timed_out_once"], "the injected git timeout never fired"
    # Survived the timeout, exited cleanly, and progressed past the diff preamble.
    assert exit_code == 0
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).is_file()


async def test_deep_run_reports_persistent_git_timeout_distinctly(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout that survives retries is surfaced as a distinct 'Git Timeout'."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)


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
    assert "Git Error" not in titles, f"a timeout was misreported as the generic base-branch error: {errors!r}"


async def test_per_stack_sonnet_merge_opus_and_arbiter_on_high_severity(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#168 real-path: drive runner.run through the production entrypoint and assert observable model targeting +
    the arbitrated finding on disk."""
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
    assert set(per_stack_models) == {"claude-sonnet-5"}

    # (b) Merge backend created with an Opus model id.
    assert models_where(lambda pl: "cross-stack merge agent" in pl) == ["claude-opus-5"]

    # (c) Opus arbiter created exactly once when a high-severity record exists.
    assert models_where(lambda pl: "you are the arbiter" in pl) == ["claude-opus-5"]

    # The rendered merge artifact on disk reflects the arbitrated finding.
    report = (multi_stack_target / ".review-output.md").read_text()
    assert "ARBITRATED:" in report, f"arbitrated finding missing from report:\n{report}"


async def test_arbiter_missing_verdict_retains_high_severity_finding(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#175 real-path: a truncated/lazy arbiter that omits every verdict must NOT delete the high-severity finding
    it was selected to protect."""
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
        rec for path in deep_dir.glob("stack-*-records.json") for rec in _record_issues(json.loads(path.read_text()))
    ]
    assert any(r.get("severity") == "high" for r in records), (
        f"high-severity record must survive a missing arbiter verdict:\n{records}"
    )


async def test_no_arbiter_when_all_findings_low_and_uncontested(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    assert per_stack_models == {"claude-sonnet-5"}


async def test_precision_suppresses_low_sibling_sharing_high_finding_location(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#232 Comment 2: a borderline LOW finding sharing a (file, line) with an arbitrated HIGH finding from the
    same stack must STILL be suppression-reviewed and dropped. A (file, line)-keyed exclusion excluded both
    siblings, letting the LOW one survive unreviewed; the per-record-identity key fixes it."""
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(
        monkeypatch,
        multi_stack_target,
        merge_echo_records=True,
        parse_by_stack=_SUPPRESSION_COLLISION_STACKS,
        suppression_keep=False,
    )

    exit_code = await _run_deep(multi_stack_target, precision_mode=True)
    assert exit_code == 0

    # The LOW sibling reached suppression despite sharing a location with the
    # arbitrated HIGH finding -- exactly one batched suppression call. Under the
    # (file, line)-keyed bug it was excluded, so the pass had zero targets.
    sup_calls = [c for c in calls if "you are the suppression reviewer" in c["prompt"].lower()]
    assert len(sup_calls) == 1, f"the LOW sibling must reach suppression, got {len(sup_calls)} calls"

    descriptions = _merged_item_descriptions(multi_stack_target)
    # keep=false -> the reviewed LOW sibling is dropped ...
    assert not any("borderline sibling" in d for d in descriptions), (
        f"the LOW sibling sharing the HIGH location must be suppressed:\n{descriptions}"
    )
    # ... while the arbitrated HIGH finding at the same location survives.
    assert any("ARBITRATED" in d and "HIGH finding" in d for d in descriptions), (
        f"the HIGH finding at the shared location must survive:\n{descriptions}"
    )
