"""Precision Budgets And Tiers."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from daydream import remote_ci
from daydream.backends import AgentEvent, ResultEvent, TextEvent
from daydream.config_file import DaydreamFileConfig
from daydream.deep.orchestrator import DEFAULT_SHALLOW_FANOUT_THRESHOLD, _shallow_fanout_threshold
from daydream.deep.settings import _resolve_opt_in
from daydream.runner import RunConfig, run
from tests.deep_orchestrator.support import (
    _batched_group_size,
    _install_accept_gate_pipeline,
    _install_post_recorder,
    _merged_item_files,
    _prime_merge_resume_records,
    _scan_phase_events,
    _scan_trajectory_extra,
    _single_fix_calls_for,
)
from tests.harness.git_helpers import git as _git
from tests.harness.remote_ci import NoCIRemote
from tests.harness.review_profile import independent_alternatives_profile
from tests.test_deep_orchestrator import (
    _CONFIDENCE_KNOB_STACKS,
    _PRECISION_STACKS,
    MakeConfig,
    Mute,
    _add_bare_remote,
    _force_interactive,
    _install_model_capturing_stubs,
    _install_stub_backend,
    _merge_item,
    _profile_with_pipeline,
    _prompt_ref,
    _PromptHookStub,
    _run_deep,
    _sanctioned_inputs,
    _silence,
)


@pytest.mark.parametrize(
    ("precision_mode", "suppression_keep", "low_survives", "suppression_count"),
    [
        pytest.param(True, False, False, 1, id="on-drops-unconfirmed"),
        pytest.param(False, False, True, 0, id="off-keeps-low"),
        pytest.param(True, True, True, 1, id="on-keeps-confirmed"),
    ],
)
async def test_precision_routes_borderline_low_finding(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    precision_mode: bool,
    suppression_keep: bool,
    low_survives: bool,
    suppression_count: int,
) -> None:
    """Precision reviews borderline low findings only when enabled, then applies its verdict."""
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(
        monkeypatch,
        multi_stack_target,
        merge_echo_records=True,
        parse_by_stack=_PRECISION_STACKS,
        suppression_keep=suppression_keep,
    )
    assert await _run_deep(multi_stack_target, precision_mode=precision_mode) == 0

    files = _merged_item_files(multi_stack_target)
    assert ("App.tsx" in files) is low_survives, files
    assert "api.py" in files
    suppression = [c for c in calls if "you are the suppression reviewer" in c["prompt"].lower()]
    assert len(suppression) == suppression_count
    if suppression:
        assert suppression[0]["model"] == "claude-sonnet-5"
    arbiters = [c for c in calls if "you are the arbiter" in c["prompt"].lower()]
    assert len(arbiters) == 1
    assert arbiters[0]["model"] == "claude-opus-5"


@pytest.mark.parametrize("widened", [False, True], ids=["default-low-only", "include-medium"])
async def test_precision_confidence_classes_control_medium_suppression(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    widened: bool,
) -> None:
    """The confidence setting decides whether an uncontested medium finding is reviewed."""
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(
        monkeypatch,
        multi_stack_target,
        merge_echo_records=True,
        parse_by_stack=_CONFIDENCE_KNOB_STACKS,
        suppression_keep=False,
    )
    profile = _profile_with_pipeline(suppression_confidence_classes=("LOW", "MEDIUM")) if widened else None
    assert await _run_deep(multi_stack_target, precision_mode=True, review_profile=profile) == 0

    files = _merged_item_files(multi_stack_target)
    assert ("App.tsx" in files) is not widened, files
    assert "api.py" in files
    suppression = [c for c in calls if "you are the suppression reviewer" in c["prompt"].lower()]
    assert len(suppression) == int(widened)


@pytest.mark.parametrize("enabled", [False, True], ids=["default", "opt-in"])
async def test_deep_flow_forwards_approve_on_clean(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    """The deep flow forwards the default or opted-in approval flag to posting."""
    _silence(monkeypatch)
    _install_model_capturing_stubs(monkeypatch, multi_stack_target)
    received: list[bool] = []
    _install_post_recorder(monkeypatch, received)
    exit_code = (
        await _run_deep(multi_stack_target, approve_on_clean=True) if enabled else await _run_deep(multi_stack_target)
    )
    assert exit_code == 0
    assert received == [enabled]


_OPT_IN_FLAGS = ("precision_mode", "approve_on_clean", "scope_issue_filing")


@pytest.mark.parametrize("flag", _OPT_IN_FLAGS)
@pytest.mark.parametrize(
    ("cli_tier", "file_value", "expected"),
    [
        pytest.param("true", None, True, id="T1-cli-true-file-absent"),
        pytest.param("unset", None, False, id="T2-cli-default-file-absent"),
        pytest.param("true", True, True, id="T3-cli-true-file-true"),
        pytest.param("true", False, True, id="T4-cli-true-outranks-file-false"),
        pytest.param("unset", True, True, id="T5-file-true-beats-cli-default"),
        pytest.param("unset", False, False, id="T6-file-false-falls-through"),
        pytest.param("false", True, True, id="T7-explicit-cli-false-is-unset"),
    ],
)
def test_opt_in_tiers_resolve_cli_then_file(
    flag: str, cli_tier: str, file_value: bool | None, expected: bool
) -> None:
    """#1225: pin the precedence rule of ``daydream/deep/settings.py:_resolve_opt_in``.

    One table over all three deep-mode opt-ins: a truthy ``RunConfig`` attr (CLI tier)
    outranks a truthy ``DaydreamFileConfig`` attr, which outranks the built-in ``False``;
    an explicit file-config ``False`` falls through to the default rather than forcing it
    off. T5 and T7 are the two rows that separate this truthiness rule from the
    ``is not None`` sentinel rule of the sibling ``_resolve_config_value``: both put the
    CLI tier at its built-in ``False`` while the file tier says ``True``. T7 constructs the
    CLI tier as an explicit ``False`` and T5 as an unset field; because ``RunConfig``'s
    fields are ``bool = False`` with no ``None`` sentinel, the two are indistinguishable by
    design — which is why a CLI ``False`` cannot mask a repo that opted in. T4 covers the
    opposite inversion (a rule where the file tier outranks an explicit CLI ``True``).
    """
    run_kwargs: dict[str, Any] = {"target": "/t"}
    if file_value is not None:
        file_kwargs: dict[str, Any] = {flag: file_value}
        run_kwargs["file_config"] = DaydreamFileConfig(**file_kwargs)
    if cli_tier != "unset":
        run_kwargs[flag] = cli_tier == "true"
    assert _resolve_opt_in(RunConfig(**run_kwargs), flag) is expected


def test_approve_on_clean_resolves_from_file_config() -> None:
    """#343 file-config tier: with NO CLI flag but ``approve_on_clean = true`` in
    the repo config, the opt-in resolver returns True; with no opt-in anywhere
    it stays False (default off)."""

    file_only = RunConfig(target="/t", file_config=DaydreamFileConfig(approve_on_clean=True))
    assert _resolve_opt_in(file_only, "approve_on_clean") is True

    unset = RunConfig(target="/t")
    assert _resolve_opt_in(unset, "approve_on_clean") is False


def test_scope_issue_filing_resolves_precedence() -> None:
    """#1056 precedence: CLI tier over file config over built-in default False."""

    cli = RunConfig(target="/t", scope_issue_filing=True)
    assert _resolve_opt_in(cli, "scope_issue_filing") is True

    file_only = RunConfig(target="/t", file_config=DaydreamFileConfig(scope_issue_filing=True))
    assert _resolve_opt_in(file_only, "scope_issue_filing") is True

    unset = RunConfig(target="/t")
    assert _resolve_opt_in(unset, "scope_issue_filing") is False


async def test_merge_resume_reruns_arbiter_when_marker_absent(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#175 real-path: a `--start-at merge` resume whose on-disk records carry a high-severity finding and NO
    completion marker must re-run the arbiter."""
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(monkeypatch, multi_stack_target, merge_echo_records=True)

    deep = _prime_merge_resume_records(multi_stack_target, python_severity="high")
    assert not (deep / "arbiter-complete.marker").exists()

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    arbiter_calls = [c for c in calls if "you are the arbiter" in c["prompt"].lower()]
    assert arbiter_calls, "arbiter must re-run on merge resume when no completion marker exists"
    assert (deep / "arbiter-complete.marker").is_file(), "completion marker must be written after arbitration"

    report = (multi_stack_target / ".review-output.md").read_text()
    assert "ARBITRATED:" in report, f"arbitrated finding missing from merge-resume report:\n{report}"


async def test_merge_resume_skips_arbiter_when_marker_present(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#175 real-path: when the completion marker proves the records were already
    finalised, a `--start-at merge` resume must NOT re-run the arbiter."""
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(monkeypatch, multi_stack_target, merge_echo_records=True)

    deep = _prime_merge_resume_records(multi_stack_target, python_severity="high")
    (deep / "arbiter-complete.marker").write_text("")

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    arbiter_calls = [c for c in calls if "you are the arbiter" in c["prompt"].lower()]
    assert arbiter_calls == [], "arbiter must not re-run when the completion marker is present"


@pytest.mark.parametrize(
    ("budget_attr", "budget_value", "sleep_s", "stop_reason"),
    [
        pytest.param("DEFAULT_TOOL_CALL_BUDGET", 3, 0.0, "tool_call_budget_exceeded", id="tool-calls"),
        pytest.param("DEFAULT_WALL_BUDGET_S", 0.3, 0.05, "wall_budget_exceeded", id="wall-clock"),
    ],
)
async def test_run_terminates_under_fix_turn_budget(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
    budget_attr: str,
    budget_value: int | float,
    sleep_s: float,
    stop_reason: str,
) -> None:
    """A runaway fix records the specific tool-call or wall-clock budget that stopped it."""

    _silence(monkeypatch)
    monkeypatch.setattr("daydream.phases." + budget_attr, budget_value)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_fix = True
    stub.runaway_fix_sleep_s = sleep_s
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))
    assert isinstance(exit_code, int)
    stop_reasons = _scan_trajectory_extra(multi_stack_target / ".daydream", traj, "stop_reason")
    assert stop_reason in stop_reasons, stop_reasons


async def test_run_caps_runaway_file_group_serial_fixes(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    no_ci_remote: NoCIRemote,
) -> None:
    """#201 real-path: a runaway file group is capped by the serial-item budget."""

    _silence(monkeypatch)
    # Lower the group serial-item ceiling at the binding the orchestrator resolves
    # (it imported the constant by name, so patching daydream.config alone is inert).
    monkeypatch.setattr("daydream.deep.fix_steps.DEFAULT_GROUP_MAX_SERIAL_ITEMS", 3)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(i, "api.py", "high") for i in range(1, 7)] + [_merge_item(7, "App.tsx", "high")]
    stub.fail_batched_fix_file = "api.py"  # force the per-finding fallback for api.py
    retained_marker = "# retained before budget stop\n"
    stub.fix_edit_line = retained_marker
    bare = _add_bare_remote(multi_stack_target)
    no_ci_remote.connect(multi_stack_target, bare)
    head_before = _git(multi_stack_target, "rev-parse", "HEAD")

    traj = tmp_path / "trajectory.json"
    # Preserve the original work watchdog in addition to the newly required,
    # separately bounded remote-CI wait. The no-CI harness keeps its truthful
    # discovery window; reducing it previously failed under parallel load.
    with anyio.fail_after(30 + remote_ci.DEFAULT_LIMITS.completion_seconds):
        exit_code = await run(
            make_config(
                multi_stack_target,
                trajectory_path=traj,
                assume="yes",
                output_mode="loop",
                pr_number=no_ci_remote.pr_number,
                pr_repo=no_ci_remote.base_repository,
            )
        )
    assert exit_code == 0
    assert retained_marker in (multi_stack_target / "api.py").read_text()
    assert stub.test_suite_calls >= 1
    head_after = _git(multi_stack_target, "rev-parse", "HEAD")
    assert head_after != head_before
    remote = multi_stack_target.parent / "multi_stack-remote.git"
    assert _git(remote, "rev-parse", "refs/heads/feature") == head_after

    # The failed batched turn names the api.py group size (the pipeline may add a
    # structural finding, so derive N rather than hard-coding it).
    group_size = _batched_group_size(stub, "api.py")
    assert group_size >= 6

    # Only 3 fallback fixes ran (the ceiling) before the group budget tripped --
    # NOT the full group, which is the runaway #186 behaviour the guard bounds.
    api_singles = _single_fix_calls_for(stub, "api.py")
    assert len(api_singles) == 3, f"expected 3 fallback fixes, got {len(api_singles)}"

    # The skipped group is recorded as a budget failure (surfaces to the user).
    fix_failures_p = multi_stack_target / ".daydream" / "deep" / "fix-failures.json"
    assert fix_failures_p.is_file(), "budget-skipped group must write the fix-failures artifact"
    recorded = json.loads(fix_failures_p.read_text())
    assert "api.py" in recorded
    assert recorded["api.py"].startswith("file_group_budget_exceeded: group_serial_item_limit")

    # The trajectory carries the budget event with processed/skipped accounting.
    events = _scan_phase_events(multi_stack_target / ".daydream", traj, "file_group_budget_exceeded")
    assert events, "no file_group_budget_exceeded event emitted"
    meta = events[0]["metadata"]
    assert meta["file"] == "api.py"
    assert meta["reason"] == "group_serial_item_limit"
    assert meta["items_processed"] == 3
    assert meta["items_skipped"] == group_size - 3


async def test_run_leaves_small_file_group_unbudgeted(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """#201 real-path: under a high ceiling the group budget is purely additive."""

    _silence(monkeypatch)
    monkeypatch.setattr("daydream.deep.fix_steps.DEFAULT_GROUP_MAX_SERIAL_ITEMS", 20)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(i, "api.py", "high") for i in range(1, 7)] + [_merge_item(7, "App.tsx", "high")]
    stub.fail_batched_fix_file = "api.py"
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))
    assert isinstance(exit_code, int)

    group_size = _batched_group_size(stub, "api.py")
    api_singles = _single_fix_calls_for(stub, "api.py")
    assert len(api_singles) == group_size, f"all {group_size} fallback fixes should run, got {len(api_singles)}"

    events = _scan_phase_events(multi_stack_target / ".daydream", traj, "file_group_budget_exceeded")
    assert events == [], "no budget event should fire when the group stays within budget"


async def test_run_batched_wall_trip_carries_into_group_fallback(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """#201 real-path: a batched turn's OWN wall trip carries into the fallback."""

    _silence(monkeypatch)
    # Tiny per-invocation wall so the batched turn (scaled to N * 0.3s) trips after
    # ~1.8s of real wall; patch the binding read at the fix call site.
    monkeypatch.setattr("daydream.phases.DEFAULT_WALL_BUDGET_S", 0.3)
    # Group wall ceiling below the batched turn's scaled per-invocation budget, so
    # the wall the batched turn already burned guarantees the fallback's first
    # check trips (deterministic: 1.0 < 0.3 * 6).
    monkeypatch.setattr("daydream.deep.fix_steps.DEFAULT_GROUP_MAX_WALL_S", 1.0)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(i, "api.py", "high") for i in range(1, 7)] + [_merge_item(7, "App.tsx", "high")]
    stub.runaway_batched_fix_file = "api.py"  # batched api.py turn trips its own wall budget
    stub.runaway_batched_sleep_s = 0.05
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))
    assert isinstance(exit_code, int)

    # The batched api.py turn actually ran (and is where the wall was burned).
    group_size = _batched_group_size(stub, "api.py")
    assert group_size >= 6

    # ZERO fallback fixes ran: the batched turn's carried-over wall tripped the
    # group budget on the fallback's very first check -- the #186 runaway is fully
    # bounded, not merely trimmed.
    api_singles = _single_fix_calls_for(stub, "api.py")
    assert len(api_singles) == 0, f"expected 0 fallback fixes (wall carried over), got {len(api_singles)}"

    # The skipped group is recorded as a WALL budget failure (surfaces to the user).
    fix_failures_p = multi_stack_target / ".daydream" / "deep" / "fix-failures.json"
    assert fix_failures_p.is_file(), "budget-skipped group must write the fix-failures artifact"
    recorded = json.loads(fix_failures_p.read_text())
    assert "api.py" in recorded
    assert recorded["api.py"].startswith("file_group_budget_exceeded: group_wall_budget_exceeded")

    # The trajectory carries the budget event: 0 processed, the whole group skipped.
    events = _scan_phase_events(multi_stack_target / ".daydream", traj, "file_group_budget_exceeded")
    assert events, "no file_group_budget_exceeded event emitted"
    meta = events[0]["metadata"]
    assert meta["file"] == "api.py"
    assert meta["reason"] == "group_wall_budget_exceeded"
    assert meta["items_processed"] == 0
    assert meta["items_skipped"] == group_size

    # Discriminator: the batched turn failed via run_agent's REAL per-invocation
    # wall budget (a budget_reason), not a synchronous stub raise -- its aborted
    # ATIF step carries stop_reason == wall_budget_exceeded.
    run_root = multi_stack_target / ".daydream"
    stop_reasons = _scan_trajectory_extra(run_root, traj, "stop_reason")
    assert "wall_budget_exceeded" in stop_reasons, "batched turn did not trip its own per-invocation wall budget"


async def test_run_batches_same_file_findings_into_one_fix_turn(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """#202 real-path: N findings on ONE file collapse to a single FIX run_agent turn."""

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    setattr(stub, "concise_fix_prompts", True)
    stub.merge_items = [
        _merge_item(1, "api.py", "high"),
        _merge_item(2, "api.py", "medium"),
        _merge_item(3, "api.py", "low"),
        _merge_item(4, "App.tsx", "high"),
    ]
    mute_side_effects()

    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False))

    assert exit_code == 0

    fix_prompts = [c["prompt"] for c in stub.calls if c["prompt"].lower().startswith(("fix this issue", "fix these"))]
    # Two file-groups -> two fix turns, regardless of how many findings each holds
    # (the pre-#202 per-finding loop would emit one turn per finding instead).
    assert len(fix_prompts) == 2
    batched = [p for p in fix_prompts if p.lower().startswith("fix these")]
    singles = [p for p in fix_prompts if p.lower().startswith("fix this issue")]
    assert len(batched) == 1 and len(singles) == 1
    # The batched api.py turn collapses all three of my api.py findings (the host
    # may add a structural finding to the same file, so assert >= 3) into one turn.
    m = re.search(r"^Fix these (\d+) issues in (.+):$", batched[0], re.M)
    assert m is not None
    assert int(m.group(1)) >= 3
    assert Path(m.group(2)).name == "api.py"
    assert "CONCISE MODE" in batched[0]


async def test_environmental_failure_aborts_heal_loop(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC#6b real-path: an environmental test failure aborts heal without a fix turn."""
    _silence(monkeypatch, prompts=False)
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")

    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.environmental_test_failure = True  # every test run reports infra-down

    # phase_test_and_heal stays REAL so the environmental short-circuit runs.
    mute_side_effects(heal=False)

    traj = tmp_path / "trajectory.json"
    exit_code = await run(make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"))

    # Environmental failure is not healable -> run reports failure, not success.
    assert isinstance(exit_code, int)
    assert exit_code != 0, "environmental failure must surface as a non-zero exit, not be healed"

    # The observable proof: the heal loop NEVER re-entered a fix turn, so the
    # heal-fix sentinel was never written. (Discriminating: without the
    # short-circuit, choice "2" would write this file.)
    heal_sentinel = multi_stack_target / ".daydream-heal-fix-applied"
    assert not heal_sentinel.exists(), (
        "heal-fix sentinel exists -- the environmental short-circuit did not abort before re-entering a fix turn"
    )
    # And no heal-fix prompt was ever dispatched to the backend.
    heal_prompts = [c for c in stub.calls if c["prompt"].lower().startswith("the tests failed")]
    assert not heal_prompts, "a heal fix prompt was dispatched despite environmental abort"

    # The environmental outcome is observable: the test phase ran (the suite was
    # invoked) and a TEST-phase trajectory step was recorded for this run.
    assert stub.test_suite_calls >= 1, "test suite never ran -- heal phase not reached"
    run_root = multi_stack_target / ".daydream"
    saw_test_step = "test" in _scan_trajectory_extra(run_root, traj, "daydream_phase")
    assert saw_test_step, "no TEST-phase trajectory step recorded -- heal phase not reached"


@pytest.mark.parametrize(
    ("trajectory_mode", "response_kind"),
    [
        pytest.param("default", "clean", id="default-clean"),
        pytest.param("custom-public", "clean", id="custom-public-clean"),
        pytest.param("external", "clean", id="external-clean"),
        pytest.param("default", "known-leaf", id="default-known-leaf-private-fallback"),
        pytest.param(
            "custom-public",
            "known-leaf",
            id="custom-public-known-leaf-private-fallback",
        ),
        pytest.param("external", "known-leaf", id="external-known-leaf-preserved"),
        pytest.param("default", "unknown-private", id="unknown-private-fallback"),
    ],
)
async def test_ephemeral_failure_handoff_projects_public_refs_without_private_paths(
    multi_stack_target: Path,
    tmp_path: Path,
    artifact_runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
    trajectory_mode: str,
    response_kind: str,
) -> None:
    """The real runner reads live bytes and persists only durable handoff paths."""
    _silence(monkeypatch, prompts=False)
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")
    summarizer_observations: list[dict[str, str]] = []
    private_partial_payloads: list[bytes] = []

    class _HandoffReadingStub(_PromptHookStub):
        def intercept(self, cwd: Path, prompt: str) -> Sequence[AgentEvent] | None:
            if response_kind == "unknown-private" and "run the project's test suite" in prompt.lower():
                self.test_suite_calls += 1
                return [
                    TextEvent(text=f"1 failed at {cwd / '.daydream' / 'unreported.log'}"),
                    ResultEvent(structured_output=None, continuation=None),
                ]
            if "read-only failure-summarizer" not in prompt.lower():
                return None
            private_partial = Path(_prompt_ref(prompt, "trajectory-partial"))
            partial_payload = json.loads(private_partial.read_text(encoding="utf-8"))
            changed_relative = Path(".daydream-heal-fix-applied")
            sanctioned = _sanctioned_inputs(prompt)
            assert sanctioned["trajectory-partial"] == private_partial
            assert all(path.is_file() for path in sanctioned.values())
            future_trajectory = _prompt_ref(prompt, "trajectory")
            future_children = _prompt_ref(prompt, "sub-trajectories")
            model_body = (
                "# Daydream handoff\n\nHANDOFF_STRUCTURED_SUCCESS\n\n"
                "## Artifacts\n\n"
                f"- trajectory: {future_trajectory}\n"
                f"- sub-trajectories: {future_children}\n\n"
                "## Changed files\n\n"
                f"- {multi_stack_target / changed_relative}\n"
            )
            if response_kind == "known-leaf":
                model_body += f"\nExact evidence: {private_partial}\n"
            elif response_kind == "unknown-private":
                model_body += f"\nUnknown evidence: {private_partial}.unknown\n"
            private_partial_payloads.append(private_partial.read_bytes())
            summarizer_observations.append(
                {
                    "private_partial": str(private_partial),
                    "cwd": str(cwd),
                    "session_id": str(partial_payload["session_id"]),
                    "changed_body": (cwd / changed_relative).read_text(encoding="utf-8"),
                    "prompt": prompt,
                    "future_trajectory": future_trajectory,
                    "future_children": future_children,
                    "model_body": model_body,
                }
            )
            return [ResultEvent(structured_output={"handoff_prompt": model_body}, continuation=None)]

    stub = _HandoffReadingStub(multi_stack_target)
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None, **kwargs: stub,
    )
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    stub.fail_all_test_runs = True
    mute_side_effects(heal=False)
    _add_bare_remote(multi_stack_target)

    trajectory_path = (
        tmp_path / "external trajectory.json"
        if trajectory_mode == "external"
        else multi_stack_target / ".daydream" / "custom trajectory.json"
        if trajectory_mode == "custom-public"
        else None
    )
    exit_code = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            force_worktree=True,
            trajectory_path=trajectory_path,
        )
    )

    assert exit_code != 0
    handoffs = list(multi_stack_target.glob(".daydream/runs/*/handoff.md"))
    assert len(handoffs) == 1
    body = handoffs[0].read_text(encoding="utf-8")
    public_run = handoffs[0].parent
    expected_trajectory = trajectory_path if trajectory_path is not None else public_run / "trajectory.json"
    expected_partial = expected_trajectory.with_suffix(expected_trajectory.suffix + ".partial")
    assert str(expected_trajectory) in body
    assert expected_trajectory.is_file()
    assert str(artifact_runtime_root.parent) not in body
    assert len(summarizer_observations) == 1
    observation = summarizer_observations[0]
    assert observation["changed_body"] == "healed\n"
    assert observation["session_id"] == handoffs[0].parent.name
    if trajectory_mode == "external":
        assert observation["private_partial"] == str(expected_trajectory.with_suffix(".json.partial"))
    else:
        assert str(artifact_runtime_root.parent) in observation["private_partial"]
    assert "Future handoff links (not readable evidence during this turn)" in observation["prompt"]
    assert "## On-disk artifacts (read these first" not in observation["prompt"]
    assert "- .daydream-heal-fix-applied" in observation["prompt"]
    assert str(multi_stack_target / ".daydream-heal-fix-applied") in observation["prompt"]
    assert observation["future_children"] == str(public_run / "trajectories")
    private_partial = observation["private_partial"]
    if response_kind == "clean":
        assert body == observation["model_body"]
    elif response_kind == "known-leaf":
        if trajectory_mode == "external":
            assert body == observation["model_body"]
        else:
            assert "HANDOFF_STRUCTURED_SUCCESS" not in body
            assert "Tests did not report success" in body
            assert "1 failed" in body
            assert private_partial not in body
        assert expected_partial.is_file()
        assert expected_partial.read_bytes() == private_partial_payloads[0]
    else:
        assert "HANDOFF_STRUCTURED_SUCCESS" not in body
        assert "Tests did not report success" in body
        assert ".daydream/unreported.log" not in body
    if trajectory_mode != "external":
        assert private_partial not in body
    assert observation["cwd"] not in body


@pytest.mark.parametrize(
    ("target_fixture", "independent", "alternatives_run"),
    [
        pytest.param("feature_branch_repo", True, False, id="independent-one-file-skips"),
        pytest.param("multi_stack_target", True, True, id="independent-multi-file-runs"),
        pytest.param("multi_stack_target", False, False, id="default-multi-file-folds"),
    ],
)
async def test_alternatives_phase_follows_diff_size(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
    target_fixture: str,
    independent: bool,
    alternatives_run: bool,
) -> None:
    """Independent wonder remains tiered; the default shares structural review."""

    target = cast(Path, request.getfixturevalue(target_fixture))
    stub = _install_accept_gate_pipeline(monkeypatch, target, mute_side_effects)
    traj = tmp_path / "trajectory.json"
    assert (
        await run(make_config(
            target, trajectory_path=traj, assume="yes", output_mode="loop", non_interactive=False,
            review_profile=independent_alternatives_profile() if independent else None,
        ))
        == 0
    )

    phases = _scan_trajectory_extra(target / ".daydream", traj, "daydream_phase")
    assert ("alternatives" in phases) is alternatives_run, phases
    wonder_calls = [
        c
        for c in stub.calls
        if "would you have done this differently" in c["prompt"].lower()
        or "evaluate the implementation" in c["prompt"].lower()
    ]
    assert bool(wonder_calls) is alternatives_run, wonder_calls
    if not independent:
        assert any(
            "you are the structural reviewer" in call["prompt"].lower()
            and "Within this same boundary review, check design choices" in call["prompt"]
            for call in stub.calls
        )
    if not alternatives_run:
        artifact = target / ".daydream" / "deep" / "alternatives.json"
        assert json.loads(artifact.read_text()) == []
        assert "intent" in phases


def test_shallow_fanout_threshold_precedence() -> None:
    """AC7: SHALLOW_FANOUT_THRESHOLD honors CLI (RunConfig) > config file > default."""

    # Default: no CLI field, no file_config.
    assert _shallow_fanout_threshold(RunConfig()) == DEFAULT_SHALLOW_FANOUT_THRESHOLD
    # Explicit 0 on RunConfig disables the short-circuit (must NOT be ignored as falsy).
    assert _shallow_fanout_threshold(RunConfig(shallow_fanout_threshold=0)) == 0
    # CLI value wins.
    assert _shallow_fanout_threshold(RunConfig(shallow_fanout_threshold=5)) == 5
    # File-config value beats default.
    fc = DaydreamFileConfig(shallow_fanout_threshold=3)
    assert _shallow_fanout_threshold(RunConfig(file_config=fc)) == 3
    # File-config value of 0 (disable) is honored, not treated as falsy.
    fc_zero = DaydreamFileConfig(shallow_fanout_threshold=0)
    assert _shallow_fanout_threshold(RunConfig(file_config=fc_zero)) == 0
    # CLI > file.
    assert _shallow_fanout_threshold(RunConfig(file_config=fc, shallow_fanout_threshold=5)) == 5
