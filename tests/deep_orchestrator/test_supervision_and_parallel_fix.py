"""Supervision And Parallel Fix."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from daydream.config_file import load_file_config
from daydream.runner import run
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.deep_orchestrator.support import (
    _capture_warnings,
    _only_archived_run,
    _scan_phase_events,
)
from tests.harness.review_profile import independent_exploration_profile
from tests.harness.stub_backend import StubBackend
from tests.test_deep_orchestrator import (
    MakeConfig,
    Mute,
    _add_to_reviewed_diff,
    _fix_prompts,
    _force_interactive,
    _install_model_capturing_stubs,
    _install_stub_backend,
    _merge_item,
    _pin_findings_pr,
    _profile_with_pipeline,
    _silence,
)


async def test_cold_empty_supervision_retains_canonical_schema_and_skips_fix_work(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An auto-accepted real fix gate consumes the deterministic empty schema."""
    backend = EmptyReviewBackend(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    before = (multi_stack_target / "api.py").read_bytes()
    trajectory = tmp_path / "trajectory.json"

    assert await run(empty_review_config(multi_stack_target, trajectory, output_mode="loop")) == 0

    prompts = [call["prompt"].lower() for call in backend.calls]
    assert not any("supervisor adjudication" in prompt for prompt in prompts)
    assert _fix_prompts(backend) == []
    assert not any("run the project's test suite" in prompt for prompt in prompts)
    deep = multi_stack_target / ".daydream" / "deep"
    assert json.loads((deep / "supervise-input.json").read_text()) == []
    assert json.loads((deep / "merged-items.json").read_text()) == {"items": [], "held": []}
    assert (multi_stack_target / "api.py").read_bytes() == before
    assert "No actionable items" in capsys.readouterr().out
    assert not (deep / "fix-outcomes.json").exists()
    events = json.loads(trajectory.read_text())["extra"]["phase_events"]
    lifecycle = [event for event in events if event.get("metadata", {}).get("stage") == "supervise"]
    assert [event["event"] for event in lifecycle] == ["phase_start", "phase_end"]
    assert lifecycle[-1]["status"] == "succeeded"


async def test_cold_empty_custom_supervision_still_dispatches(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    backend = EmptyReviewBackend(multi_stack_target, forbid_supervise=False)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    resolved = _profile_with_pipeline()
    strategies = dict(resolved.profile.strategies)
    strategies["supervision"] = replace(
        strategies["supervision"], content=strategies["supervision"].content + "\nCustom supervision policy.",
    )
    profile = replace(resolved, profile=replace(resolved.profile, strategies=strategies))

    assert await run(empty_review_config(multi_stack_target, tmp_path / "trajectory.json", review_profile=profile)) == 0

    assert sum("supervisor adjudication" in call["prompt"].lower() for call in backend.calls) == 1
    deep = multi_stack_target / ".daydream" / "deep"
    assert json.loads((deep / "supervise-input.json").read_text()) == []
    assert json.loads((deep / "merged-items.json").read_text()) == {"items": [], "held": []}


def _prepare_fix_stub(target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute) -> StubBackend:
    """Drive fix-cycle tests through the same interactive real-path setup."""
    _force_interactive(monkeypatch)
    return _supervision_stub(target, monkeypatch, mute_side_effects)


@pytest.fixture
def stub(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute) -> StubBackend:
    """Interactive stub backend for this module's real-path fix-cycle tests."""
    return _prepare_fix_stub(multi_stack_target, monkeypatch, mute_side_effects)


def _supervision_stub(
    target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute | None = None, *, pin_pr: bool = False,
    **install_kwargs: Any,
) -> StubBackend:
    """Silence UI, optionally mute/pin findings-out, then install the stub backend."""
    _silence(monkeypatch)
    if mute_side_effects is not None:
        mute_side_effects()
    if pin_pr:
        _pin_findings_pr(monkeypatch, target)
    return _install_stub_backend(monkeypatch, target, **install_kwargs)


def _findings_out_config(
    make_config: MakeConfig, target: Path, out: Path, trajectory: Path | None = None,
    *, monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Bind the supervision configuration and PR to a clean committed snapshot."""
    _add_to_reviewed_diff(target, [".daydream.toml"])
    _pin_findings_pr(monkeypatch, target)
    kwargs: dict[str, Any] = {"pr_number": 7, "findings_out": str(out), "file_config": load_file_config(target)}
    if trajectory is not None:
        kwargs["trajectory_path"] = trajectory
    return make_config(target, **kwargs)


def _forbid_pr_post(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a findings-out PR poster that fails if it is ever invoked."""
    async def _post_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("findings-out must not post to the PR")
    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _post_forbidden)

@pytest.mark.parametrize("policy,items,verdicts,action,expected", [
    pytest.param('supervisor = "rules"\nsupervisor_deny_globs = ["vendor/**"]\n',
                 [("vendor/generated.py", "high", "drop this finding"), ("src/app.py", "low", "keep this finding")],
                 {}, "drop", [("keep this finding", "low")], id="rules-drop-denied-file"),
    pytest.param('supervisor = "llm"\n', [("api.py", "high", "drop by llm"), ("App.tsx", "low", "keep by llm")],
                 {1: {"action": "drop", "reason": "duplicate"}, 2: {"action": "allow", "reason": "confirmed"}},
                 "drop", [("keep by llm", "low")], id="llm-drop-canonical-id"),
    pytest.param('supervisor = "llm"\n', [("api.py", "high", "downgrade me")],
                 {1: {"action": "edit", "reason": "less severe", "severity": "low"}},
                 "edit", [("downgrade me", "low")], id="llm-edit-severity"),
    pytest.param('supervisor = "rules"\nsupervisor_deny_globs = ["**"]\n',
                 [("api.py", "high", "drop everything")], {}, "drop", [], id="rules-drop-all"),
])
async def test_supervision_transforms_canonical_items_before_findings_export(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
    mute_side_effects: Mute, policy: str, items: list[tuple[str, str, str]], verdicts: dict[int, dict[str, Any]],
    action: str, expected: list[tuple[str, str]],
) -> None:
    stub = _supervision_stub(multi_stack_target, monkeypatch, mute_side_effects, pin_pr=True)
    stub.merge_items = [_merge_item(index, file, severity, desc=description)
                        for index, (file, severity, description) in enumerate(items, 1)]
    stub.supervise_verdicts = verdicts
    (multi_stack_target / ".daydream.toml").write_text(policy)
    out, trajectory = multi_stack_target / "findings.json", tmp_path / "trajectory.json"
    _forbid_pr_post(monkeypatch)
    assert await run(_findings_out_config(
        make_config, multi_stack_target, out, trajectory, monkeypatch=monkeypatch,
    )) == 0
    payload = json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())
    descriptions = {description for _, _, description in items}
    assert [(item["description"], item["severity"]) for item in payload["items"]
            if item["description"] in descriptions] == expected
    findings = json.loads(out.read_text())["findings"]
    assert [(finding["title"], finding["severity"]) for finding in findings
            if finding["title"] in descriptions] == expected
    if not expected:
        assert findings == []
    events = _scan_phase_events(multi_stack_target / ".daydream", trajectory, "supervisor_verdict")
    assert any(event.get("metadata", {}).get("finding_id") == 1
               and event["metadata"].get("action") == action for event in events)
    starts = _scan_phase_events(multi_stack_target / ".daydream", trajectory, "phase_start")
    assert any(event.get("metadata", {}).get("stage") == "supervise" for event in starts) is ('"llm"' in policy)

async def test_supervise_hold_excluded_but_rendered(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """Held findings leave the actionable items but remain visible in the report."""

    stub = _supervision_stub(multi_stack_target, monkeypatch, mute_side_effects)
    stub.merge_items = [_merge_item(1, "vendor/generated.py", "high", desc="hold this finding"),
        _merge_item(2, "src/app.py", "low", desc="keep this finding"),
    ]
    stub.supervise_verdicts = {
        1: {"action": "hold", "reason": "needs human review"}, 2: {"action": "allow", "reason": "confirmed"},
    }
    (multi_stack_target / ".daydream.toml").write_text('supervisor = "llm"\n')

    rc = await run(make_config(multi_stack_target, file_config=load_file_config(multi_stack_target)))

    assert rc == 0
    payload = json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())
    actionable = [item["description"] for item in payload["items"]]
    held = [item["description"] for item in payload["held"]]
    assert "keep this finding" in actionable
    assert "hold this finding" in held
    report = (multi_stack_target / ".review-output.md").read_text()
    assert "Held Findings" in report
    assert "hold this finding" in report




async def test_supervise_off_byte_identical(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    tmp_path: Path,
) -> None:

    stub = _supervision_stub(multi_stack_target, monkeypatch, mute_side_effects, pin_pr=True)
    stub.merge_items = [
        _merge_item(1, "api.py", "high", desc="first finding"), _merge_item(2, "App.tsx", "low", desc="second finding"),
    ]
    out = tmp_path / "findings.json"
    empty_config = load_file_config(multi_stack_target)
    first_rc = await run(make_config(multi_stack_target, pr_number=7, findings_out=str(out), file_config=empty_config))
    first_items = (multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_bytes()
    first_findings = json.loads(out.read_text())["findings"]

    (multi_stack_target / ".daydream.toml").write_text('supervisor = "off"\n')
    second_rc = await run(_findings_out_config(make_config, multi_stack_target, out, monkeypatch=monkeypatch))
    second_items = (multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_bytes()
    second_findings = json.loads(out.read_text())["findings"]

    assert first_rc == second_rc == 0
    assert first_items == second_items
    assert first_findings == second_findings

async def test_supervise_dropped_finding_never_reaches_fix(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:

    stub = _supervision_stub(multi_stack_target, monkeypatch, mute_side_effects)
    stub.merge_items = [_merge_item(1, "api.py", "high", desc="drop before fix"),
        _merge_item(2, "App.tsx", "low", desc="fix this survivor"),
    ]
    (multi_stack_target / ".daydream.toml").write_text('supervisor = "rules"\nsupervisor_deny_globs = ["api.py"]\n')
    source_before = (multi_stack_target / "api.py").read_bytes()
    rc = await run(make_config(
            multi_stack_target, assume="yes", output_mode="loop", file_config=load_file_config(multi_stack_target),
        )
    )

    assert rc == 0
    assert (multi_stack_target / "api.py").read_bytes() == source_before
    prompts = "\n".join(_fix_prompts(stub))
    assert "drop before fix" not in prompts

async def test_run_deep_renders_prescan_summary_not_json(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:


    # Add a 4th changed file so select_tier() -> "parallel" (the pattern-scanner
    # runs and its conventions reach the rendered summary).
    (multi_stack_target / "extra.py").write_text("VALUE = 2\n")
    subprocess.run(["git", "add", "."], cwd=multi_stack_target, capture_output=True, check=True)  # noqa: S603, S607 - arguments are not user-controlled
    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "commit", "-m", "add extra"], cwd=multi_stack_target, capture_output=True, check=True
    )

    _silence(monkeypatch)
    mute_side_effects()
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=120)
    monkeypatch.setattr("daydream.deep.review_steps.console", rec)
    _supervision_stub(multi_stack_target, monkeypatch, enable_exploration=True)

    exit_code = await run(make_config(
        multi_stack_target, assume="yes", output_mode="loop", review_profile=independent_exploration_profile(),
    ))
    assert exit_code == 0
    out = rec.export_text()
    assert "OpenAPI First" in out  # convention surfaced by the summary
    assert '{"conventions"' not in out and "pattern-scanner" not in out  # no raw JSON envelope

async def test_parallel_fix_applies_all_disjoint_files(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    stub: StubBackend,
) -> None:
    files = ["f1.py", "f2.py", "f3.py", "f4.py"]
    _add_to_reviewed_diff(multi_stack_target, files)
    stub.merge_items = [_merge_item(i + 1, f, "high") for i, f in enumerate(files)]
    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False))
    assert exit_code == 0
    prompts = [call["prompt"] for call in stub.calls if call["prompt"].lower().startswith("fix this")]
    for f in files:
        assert any(f in prompt for prompt in prompts), f"no fixer dispatched for {f}"

async def test_long_fix_is_not_turn_capped(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, make_config: MakeConfig,
    mute_side_effects: Mute, stub: StubBackend,
) -> None:

    stub.fix_turns_needed = 200
    stub.merge_items = [
        _merge_item(1, "api.py", "high"), _merge_item(2, "App.tsx", "high"), _merge_item(3, "App.tsx", "medium"),
    ]

    exit_code = await run(
        make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False, archive=True,)
    )

    assert exit_code == 0
    fix_calls = [call for call in stub.calls if call["prompt"].lower().startswith(("fix this", "fix these"))]
    assert any("api.py" in call["prompt"] for call in fix_calls)
    assert any("App.tsx" in call["prompt"] for call in fix_calls)
    assert all(call["max_turns"] is None for call in fix_calls)

    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert not manifest["fix_failures"]

async def test_parallel_fix_same_file_no_race(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    stub: StubBackend,
) -> None:
    """3 items on ONE file + 1 on another. The 3 same-file findings collapse into ONE batched fix turn that
    addresses every marker in severity order, while the other file's group runs concurrently. The
    read-modify-write append + anyio.sleep(0) makes any cross-file race deterministic; per-file partitioning
    keeps shared.py's markers ordered and intact."""

    shared = multi_stack_target / "shared.py"
    _add_to_reviewed_diff(multi_stack_target, ["shared.py", "other.py"])
    stub.fix_append_path = shared
    stub.merge_items = [
        _merge_item(1, "shared.py", "high", desc="marker-1"), _merge_item(2, "shared.py", "medium", desc="marker-2"),
        _merge_item(3, "shared.py", "low", desc="marker-3"), _merge_item(4, "other.py", "high", desc="other"),
    ]
    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False))
    assert exit_code == 0
    assert shared.read_text().split() == ["marker-1", "marker-2", "marker-3"]

async def test_parallel_fix_footprint_intersection_dispatches_to_one_agent(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    stub: StubBackend,
) -> None:
    """A finding whose footprint intersects another group is dispatched to ONE
    agent owning both -- observable as ONE batched fix turn covering both files,
    not two per-file turns."""

    _add_to_reviewed_diff(multi_stack_target, ["a.py", "b.py", "c.py"])
    stub.merge_items = [_merge_item(1, "a.py", "high"), {**_merge_item(2, "b.py", "high"), "related_files": ["a.py"]},
        _merge_item(3, "c.py", "high"),
    ]
    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False))
    assert exit_code == 0
    # a.py and b.py findings fixed in ONE batched turn (footprint union);
    # c.py is a separate per-finding turn. (The fixture also dispatches its own
    # in-scope structural item -- api.py "Sample issue" -- as its own turn.)
    fix_calls = [c
        for c in stub.calls
        if c["prompt"].lower().startswith("fix this issue") or c["prompt"].lower().startswith("fix these")
    ]
    batched = [c for c in fix_calls if "fix these" in c["prompt"].lower()]
    assert len(batched) == 1  # a.py and b.py fixed together, not two per-file turns
    assert "issues in " in batched[0]["prompt"]
    assert len(fix_calls) == 3  # merged {a,b} group + c.py + fixture structural item

async def test_fix_verify_turn_is_read_only(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    stub: StubBackend,
) -> None:
    """AC: verification is strictly read-only. The stub records ``read_only``
    per call (stub_backend.py:276), so the real-path run must show every
    fix-verify turn arriving with ``read_only=True``."""
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False))
    assert exit_code == 0
    verify_calls = [c for c in stub.calls if "fix-verify" in c["prompt"]]
    assert verify_calls, "expected a fix-verify turn"
    assert all(c["read_only"] is True for c in verify_calls)

async def test_fix_verify_uses_verify_backend_key_through_runner(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:

    (multi_stack_target / ".daydream.toml").write_text(
        '[phases.verify]\nmodel = "verify-model-sentinel"\n[phases.fix-verify]\nmodel = "registered-step-sentinel"\n'
    )
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    calls = _install_model_capturing_stubs(monkeypatch, multi_stack_target)
    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False,
            file_config=load_file_config(multi_stack_target),
        )
    )

    assert exit_code == 0
    verifier_calls = [call for call in calls if "fix-verify" in call["prompt"]]
    assert verifier_calls
    assert {call["model"] for call in verifier_calls} == {"verify-model-sentinel"}
    assert all(call["model"] != "registered-step-sentinel" for call in verifier_calls)
    outcomes = json.loads((multi_stack_target / ".daydream" / "deep" / "fix-outcomes.json").read_text())
    assert outcomes["outcomes"]

async def test_fix_verify_writes_outcomes_and_breaks_on_resolved(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    stub: StubBackend,
) -> None:
    """Spec: every dispatched finding has a recorded terminal outcome; all
    resolved -> BreakLoop on round 1 (one fix pass per group, no re-dispatch)."""

    stub.merge_items = [_merge_item(1, "api.py", "high"), _merge_item(2, "App.tsx", "medium")]
    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False))
    assert exit_code == 0
    outcomes_p = multi_stack_target / ".daydream" / "deep" / "fix-outcomes.json"
    assert outcomes_p.exists()
    outcomes = json.loads(outcomes_p.read_text())
    # every dispatched finding (incl. structural) has exactly one recorded outcome
    assert outcomes["session_id"]
    assert sorted(v["issue_id"] for v in outcomes["outcomes"].values()) == [1, 2, 3]
    assert all(v["verdict"] == "resolved" for v in outcomes["outcomes"].values())
    # one round only (all resolved -> BreakLoop on round 1)
    fix_calls = [c for c in stub.calls if "fix this" in c["prompt"].lower() or "fix these" in c["prompt"].lower()]
    assert len(fix_calls) == 2  # one per file group, no re-dispatch

async def test_fix_verify_loop_redispatch_resolves_second_round(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    stub: StubBackend,
) -> None:
    """Spec AC#2: a partial first fix verifies unresolved, re-dispatches in a
    second round, verifies resolved -> the loop ran twice and the outcome is
    resolved."""
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    stub.fix_verify_resolve_after_round = 2  # round 1 -> unresolved, round 2 -> resolved
    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False))
    assert exit_code == 0
    outcomes_p = multi_stack_target / ".daydream" / "deep" / "fix-outcomes.json"
    outcomes = json.loads(outcomes_p.read_text())
    assert outcomes["outcomes"]["item:1"]["verdict"] == "resolved"
    fix_calls = [c for c in stub.calls if "fix this" in c["prompt"].lower() or "fix these" in c["prompt"].lower()]
    assert len(fix_calls) == 2  # loop ran twice: round 1 + re-dispatch round 2

async def test_fix_verify_wrong_target_retargets_within_scope(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    stub: StubBackend,
) -> None:
    """Spec: a wrong_target retarget re-dispatches to the corrected file, but
    only inside the allowed edit set (#336 net never widens)."""

    item = _merge_item(1, "api.py", "high")
    item["related_files"] = ["App.tsx"]
    stub.merge_items = [item]
    # round 1 verifier says: defect really lives in App.tsx
    stub.fix_verify_verdicts = {1: {"issue_id": 1, "verdict": "wrong_target", "path": "App.tsx", "reason": "moved"}}
    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False))
    assert exit_code == 0
    outcomes = json.loads((multi_stack_target / ".daydream" / "deep" / "fix-outcomes.json").read_text())
    assert outcomes["outcomes"]["item:1"]["verdict"] == "resolved"
    assert outcomes["outcomes"]["item:1"]["path"] == "App.tsx"
    audit = json.loads((multi_stack_target / ".daydream/deep/fix-footprint.json").read_text())
    accepted = [event for event in audit["events"] if event["action"] == "authorize" and event["origin"] == "retarget"]
    assert len(accepted) == 1
    assert accepted[0]["path"] == "App.tsx"
    assert accepted[0]["item_uid"] == "item:1"
    assert accepted[0]["round_number"] == 2
    assert audit["policy_revision"] == 1

async def test_unresolved_finding_reported_attempted_not_fixed(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    capsys: pytest.CaptureFixture[str], stub: StubBackend,
) -> None:
    """Spec: a finding still unresolved after the last round appears as
    attempted-not-fixed, never counted/shown as fixed."""
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    stub.fix_verify_resolve_after_round = 99  # never resolves -> attempted-not-fixed
    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False))
    assert exit_code == 0
    assert "Attempted, not fixed" in capsys.readouterr().out
    outcomes = json.loads((multi_stack_target / ".daydream" / "deep" / "fix-outcomes.json").read_text())
    assert outcomes["outcomes"]["item:1"]["verdict"] == "unresolved"
    # fix applied was NOT asserted for the unresolved finding (no "Fix applied" line for it)
    assert all(v["verdict"] == "unresolved" for v in outcomes["outcomes"].values())

async def test_parallel_fix_failure_isolated_returns_nonzero(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    capsys: pytest.CaptureFixture[str],
) -> None:

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects(commit=False)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    _add_to_reviewed_diff(multi_stack_target, ["good1.py", "bad.py", "good2.py"])
    stub.fix_fail_file = "bad.py"
    stub.fix_edit_line = "# retained successful group\n"
    stub.merge_items = [
        _merge_item(1, "good1.py", "high"), _merge_item(2, "bad.py", "high"), _merge_item(3, "good2.py", "low"),
    ]
    warnings = _capture_warnings(monkeypatch, "daydream.deep.fix_steps.print_warning")
    commit_calls: list[int] = []

    async def _spy_commit(backend: Any, work: Any, **kwargs: Any) -> None:
        commit_calls.append(1)

    monkeypatch.setattr("daydream.deep.fix_steps.phase_commit_push", _spy_commit)
    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False))
    assert exit_code == 1  # decision: nonzero on failure
    assert "# retained successful group" in (multi_stack_target / "good1.py").read_text()
    assert "# retained successful group" in (multi_stack_target / "good2.py").read_text()
    assert not (multi_stack_target / ".fixed-good1_py").exists()
    assert not (multi_stack_target / ".fixed-good2_py").exists()
    assert not (multi_stack_target / ".fixed-bad_py").exists()  # failed group did not apply
    assert any("bad.py" in m for m in warnings)  # non-silent
    assert commit_calls == []  # no commit on failure
    output = capsys.readouterr().out
    assert "complete group was restored" in output
    assert "this file's changes are left uncommitted" not in output
