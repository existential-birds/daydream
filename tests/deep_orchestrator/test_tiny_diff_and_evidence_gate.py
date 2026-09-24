"""Tiny Diff And Evidence Gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream import runner
from daydream.atif import validate as atif_validate
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.deep.detection import detect_stacks
from daydream.deep.orchestrator import (
    _collapse_stacks_for_shallow,
    _collapse_stacks_for_tiny_diff,
    _single_stack_agent_count,
    total_agent_count,
)
from daydream.runner import RunConfig, run
from tests.deep_orchestrator.support import (
    _count_merge_prompts,
    _count_review_prompts,
    _root_phase_events,
)
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo
from tests.test_deep_orchestrator import (
    MakeConfig,
    Mute,
    _force_interactive,
    _install_model_capturing_stubs,
    _install_stub_backend,
    _prime_merge_resume,
    _record,
    _run_deep,
    _silence,
)
from tests.test_finite_delegation_parse import _mark_delegated_artifacts


@pytest.mark.parametrize(
    ("files", "mode", "expected_stack"),
    [
        pytest.param(["api.py"], "tiny", "python", id="tiny-one-language"),
        pytest.param(["api.py", "App.tsx"], "tiny", "generic", id="tiny-two-languages"),
        pytest.param(["api.py", "README.md"], "tiny", "python", id="tiny-code-and-docs"),
        pytest.param(["api.py", "App.tsx"], "disabled", None, id="tiny-disabled"),
        pytest.param(["api.py", "README.md"], "shallow", "python", id="shallow-one-language"),
        pytest.param(["api.py", "App.tsx"], "shallow", "generic", id="shallow-two-languages"),
        pytest.param(["api.py", "App.tsx"], "explicit", "python", id="shallow-explicit-stack"),
    ],
)
def test_collapse_stacks_preserves_scope_and_reduces_fanout(
    files: list[str],
    mode: str,
    expected_stack: str | None,
) -> None:
    """Tiny and shallow runs combine language scopes while retaining structure."""

    stacks = detect_stacks(files)
    if mode in {"tiny", "disabled"}:
        collapsed, single = _collapse_stacks_for_tiny_diff(stacks, files, threshold=0 if mode == "disabled" else 2)
    else:
        config = RunConfig(shallow=True, stack="python" if mode == "explicit" else None)
        collapsed, single = _collapse_stacks_for_shallow(stacks, files, config)

    assert single is (mode != "disabled")
    if mode == "disabled":
        assert collapsed == stacks
        return

    assert "structure" in [stack.stack_name for stack in collapsed]
    combined = [stack for stack in collapsed if stack.stack_name != "structure"]
    assert len(combined) == 1
    assert combined[0].stack_name == expected_stack
    assert set(combined[0].files) == set(files)
    assert not hasattr(combined[0], "skill_invocation")
    if mode == "tiny":
        assert _single_stack_agent_count(len(collapsed)) < total_agent_count(len(stacks))


async def test_ac2_tiny_diff_collapses_fanout_and_skips_merge_tiny_host_merge_phase(
    tiny_diff_target: Path,
    multi_stack_target: Path,
    archive_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC2 (real-path): a ≤2-file two-language diff collapses the fan-out."""

    # Run BOTH repos through the identical harness so the count comparison is a
    # paired observation, not an absolute threshold.
    async def _drive(target: Path) -> list[dict[str, Any]]:
        # Fresh shared-call list per run (the factory binds it via closure).
        _silence(monkeypatch)
        shared_calls = _install_model_capturing_stubs(monkeypatch, target)
        # Stub the post-merge side effects so the run terminates cleanly.
        mute_side_effects()
        rc = await run(make_config(target, archive=True, run_eval=False))
        assert rc == 0, f"deep run on {target.name} exited {rc}"
        return list(shared_calls)

    tiny_calls = await _drive(tiny_diff_target)
    multi_calls = await _drive(multi_stack_target)

    # (b) Canonical merged-items.json written for the tiny diff.
    items_file = tiny_diff_target / ".daydream" / "deep" / "merged-items.json"
    assert items_file.is_file(), f"merged-items.json missing at {items_file}"
    items_payload = json.loads(items_file.read_text())
    assert isinstance(items_payload.get("items"), list)

    # (c) Review-prompt count for tiny diff is STRICTLY LESS than multi_stack.
    tiny_reviews = _count_review_prompts(tiny_calls)
    multi_reviews = _count_review_prompts(multi_calls)
    assert tiny_reviews < multi_reviews, (
        f"tiny-diff review fan-out did not collapse: tiny={tiny_reviews}, multi={multi_reviews}"
    )
    # Tiny diff: 2 review agents (combined lang + structure). Multi: 4.
    assert tiny_reviews == 2, f"expected 2 review agents for tiny diff, got {tiny_reviews}"
    assert multi_reviews == 4, f"expected 4 review agents for multi_stack, got {multi_reviews}"

    # The merge agent MUST be skipped on the tiny diff (lever 2).
    assert _count_merge_prompts(tiny_calls) == 0, "merge agent ran on tiny diff"
    # And the multi_stack run still invokes it (regression-guard for AC3).
    assert _count_merge_prompts(multi_calls) == 1, "merge agent missing on multi_stack"
    tiny_merge = _root_phase_events(tiny_diff_target, "merge")
    multi_merge = _root_phase_events(multi_stack_target, "merge")
    assert [event["metadata"] for event in tiny_merge] == [
        {"stage": "single-stack-host"},
        {"stage": "single-stack-host"},
    ]
    assert [event["metadata"] for event in multi_merge] == [
        {"stage": "cross-stack-agent"},
        {"stage": "cross-stack-agent"},
    ]
    assert tiny_merge[-1]["status"] == "succeeded"
    assert multi_merge[-1]["status"] == "succeeded"

    tiny_trajectory = next((tiny_diff_target / ".daydream" / "runs").glob("*/trajectory.json"))
    tiny_session_id = tiny_trajectory.parent.name
    trajectory_documents = [
        tiny_trajectory,
        *sorted((tiny_trajectory.parent / "trajectories").glob("*.json")),
    ]
    merge_invocations: list[dict[str, Any]] = []
    for document in trajectory_documents:
        payload = json.loads(document.read_text(encoding="utf-8"))
        merge_invocations.extend(
            summary
            for summary in (payload.get("extra") or {}).get("subtrajectories", [])
            if isinstance(summary, dict) and "invocation_id" in summary and summary.get("phase") == "merge"
        )
    assert merge_invocations == []

    manifest = json.loads((archive_dir / "runs" / tiny_session_id / "manifest.json").read_text())
    assert manifest["phase_states"]["merge"] == {"ran": True, "status": "succeeded"}


async def test_merge_failure_phase_state_domain_failure_closes_failed_scope(
    multi_stack_target: Path,
    archive_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_emit_str = "no item list"
    stale_deep = multi_stack_target / ".daydream" / "deep"
    stale_deep.mkdir(parents=True, exist_ok=True)
    (stale_deep / "merged-items.json").write_text(
        json.dumps({"items": [{"id": 999, "description": "stale success"}]}),
        encoding="utf-8",
    )

    exit_code = await run(make_config(multi_stack_target, archive=True, run_eval=False))

    assert exit_code == 1
    events = _root_phase_events(multi_stack_target, "merge")
    assert len(events) == 2
    assert events[0]["event"] == "phase_start"
    assert events[1]["event"] == "phase_end"
    assert events[0]["scope_id"] == events[1]["scope_id"]
    assert events[1]["status"] == "failed"
    assert events[1]["reason_code"] == "domain_failure"
    deep = multi_stack_target / ".daydream" / "deep"
    salvage = json.loads((deep / "merged-items.json").read_text(encoding="utf-8"))
    assert isinstance(salvage.get("items"), list)
    failures = json.loads((deep / "per-stack-failures.json").read_text(encoding="utf-8"))
    assert isinstance(failures.get("__merge__"), dict)

    trajectory = next((multi_stack_target / ".daydream" / "runs").glob("*/trajectory.json"))
    manifest = json.loads((archive_dir / "runs" / trajectory.parent.name / "manifest.json").read_text())
    assert manifest["archive_status"] == "complete"
    assert manifest["phase_states"]["merge"] == {"ran": True, "status": "failed"}
    assert manifest["pipeline_status"] == "failed"


async def test_ac5_per_stack_prompt_inlines_diff_hunks(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC5 (real-path): per-stack review prompts contain inlined diff hunks and NO ``Read it directly`` / diff_path
    instruction."""

    _silence(monkeypatch)
    shared_calls = _install_model_capturing_stubs(monkeypatch, tiny_diff_target)
    mute_side_effects()

    rc = await run(make_config(tiny_diff_target))
    assert rc == 0

    # The per-stack review prompt is the one carrying the scope discriminator.
    # (The structural prompt is intentionally NOT inlined — Fix B excludes it.)
    per_stack_review_prompts = [
        c["prompt"]
        for c in shared_calls
        if "you are reviewing the" in c["prompt"].lower() and "stack" in c["prompt"].lower()
    ]
    assert per_stack_review_prompts, "expected at least one per-stack review prompt"
    prompt = per_stack_review_prompts[0]

    # The complete api.py hunk reaches the real per-stack prompt, including
    # its enclosing function context and the exact removed/added return lines.
    expected_api_hunk = "@@ -1,2 +1,2 @@\n def hello():\n-    return 'world'\n+    return 'universe'\n"
    assert expected_api_hunk in prompt, "expected complete api.py diff hunk in per-stack prompt"
    # The Read instruction is absent (the agent is never told to Read diff.patch).
    assert "Read it directly" not in prompt
    # And diff_path is not embedded as an instruction (it remains a required
    # param of the builder but is not surfaced when hunks are inlined).
    diff_path_str = str(tiny_diff_target / ".daydream" / "diff.patch")
    assert diff_path_str not in prompt

    # Discriminating check: the STRUCTURAL prompt still carries the pointer
    # (Fix B does NOT inline the structural / arbiter prompts).
    structural_prompts = [c["prompt"] for c in shared_calls if "you are the structural reviewer" in c["prompt"].lower()]
    assert structural_prompts, "expected a structural review prompt"
    assert "Read it directly" in structural_prompts[0]
    structural_diff_lines = [line for line in structural_prompts[0].splitlines() if line.startswith("- diff: ")]
    assert len(structural_diff_lines) == 1
    structural_diff = Path(structural_diff_lines[0].removeprefix("- diff: "))
    assert structural_diff.is_file()
    assert structural_diff.name == "diff.patch"
    assert diff_path_str not in structural_prompts[0]


async def test_ac6_single_stack_merged_items_carry_structural_lens(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC6: tiny-diff single-stack writer tags structural items ``lens="structural"``."""

    _silence(monkeypatch)
    _install_model_capturing_stubs(monkeypatch, tiny_diff_target)
    mute_side_effects()

    rc = await run(make_config(tiny_diff_target))
    assert rc == 0

    items_file = tiny_diff_target / ".daydream" / "deep" / "merged-items.json"
    assert items_file.is_file()
    items = json.loads(items_file.read_text())["items"]
    lenses = {i.get("lens") for i in items}
    # Structural findings reach the canonical item list tagged correctly.
    assert "structural" in lenses, f"no structural-lens items in {lenses}"
    # And every item carries a fresh contiguous integer id (normalize_items).
    assert all(isinstance(i.get("id"), int) for i in items), "non-integer id in merged items"
    assert [i["id"] for i in items] == list(range(1, len(items) + 1)), "ids not contiguous"


async def test_ac_fix_resume_on_tiny_diff(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Issue #172 risk: ``--start-at fix`` resume on a tiny diff works."""

    # Phase 1: produce merged-items.json via a full tiny-diff run.
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, tiny_diff_target)
    mute_side_effects()

    rc = await run(make_config(tiny_diff_target, assume="no"))
    assert rc == 0
    items_file = tiny_diff_target / ".daydream" / "deep" / "merged-items.json"
    assert items_file.is_file(), "priming run did not produce merged-items.json"

    # Phase 2: resume with --start-at fix and accept the gate; the fix loop
    # must read the canonical JSON and dispatch at least one fix prompt.
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")

    rc = await run(make_config(tiny_diff_target, start_at="fix", assume="yes", non_interactive=False))
    assert rc == 0
    fix_prompts = [c for c in stub.calls if c["prompt"].startswith(("Fix this issue", "Fix these"))]
    assert fix_prompts, "fix loop did not run on --start-at fix resume"


async def test_host_only_merge_resume_publishes_and_archives_system_root(
    tmp_path: Path,
    archive_dir: Path,
    ext_dir: Any,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A registered host-only merge retains its output through real finalization."""

    target = tmp_path / "host-only-merge"
    target.mkdir()
    readme = target / "README.md"
    readme.write_text("# host-only fixture\n", encoding="utf-8")
    _init_repo(target)
    _git(target, "add", ".")
    _commit(target, "initial")
    _git(target, "checkout", "-b", "feature")
    readme.write_text("# host-only fixture\n\nchanged\n", encoding="utf-8")
    _git(target, "add", ".")
    _commit(target, "change")

    host_payload = {"items": [{"id": 1, "description": "deterministic host-only finding"}]}
    expected_items = (json.dumps(host_payload, indent=2) + "\n").encode()
    ext_dir.write_module(
        "from daydream.extensions import FlowStep\n"
        "\n"
        "async def _host_only_merge(ctx):\n"
        "    from daydream.artifact_visibility import artifact_dir_for\n"
        "    from daydream.trajectory import DaydreamPhase, phase_scope\n"
        "    async with phase_scope(DaydreamPhase.MERGE, stage='host-only-fixture'):\n"
        "        deep = artifact_dir_for(ctx.work.repo, session=ctx.artifacts, allow_standalone=False) / 'deep'\n"
        "        deep.mkdir(parents=True, exist_ok=True)\n"
        f"        (deep / 'merged-items.json').write_bytes({expected_items!r})\n"
        "\n"
        "def register(registry):\n"
        "    registry.register_phase(\n"
        "        FlowStep(name='host-only-merge-fixture', run=_host_only_merge)\n"
        "    )\n"
        "    registry.set_flow('host-only-flow', ['host-only-merge-fixture'])\n"
    )

    def fail_backend_construction(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("host-only flow must not construct a backend")

    monkeypatch.setattr(runner, "create_backend", fail_backend_construction)

    rc = await runner.run(make_config(target, flow_name="host-only-flow", archive=True, run_eval=False))

    public_items = target / ".daydream" / "deep" / "merged-items.json"
    public_runs = list((target / ".daydream" / "runs").glob("*"))
    archived_runs = list((archive_dir / "runs").glob("*"))
    assert rc == 0
    assert public_items.is_file()
    assert len(public_runs) == len(archived_runs) == 1, (public_runs, archived_runs)

    public_run = public_runs[0]
    archived_run = archived_runs[0]
    assert public_run.name == archived_run.name
    assert public_items.read_bytes() == expected_items
    assert (archived_run / "deep" / "merged-items.json").read_bytes() == expected_items

    public_trajectory = public_run / "trajectory.json"
    archived_trajectory = archived_run / "trajectory.json"
    assert public_trajectory.read_bytes() == archived_trajectory.read_bytes()
    trajectory = json.loads(public_trajectory.read_bytes())
    assert atif_validate(trajectory, validate_images=False)
    assert trajectory["session_id"] == trajectory["trajectory_id"] == public_run.name
    assert trajectory["steps"] == [
        {
            "step_id": 1,
            "timestamp": trajectory["extra"]["run_ended_at"],
            "source": "system",
            "message": "Daydream host-only run snapshot",
            "extra": {
                "daydream_run_flow": "custom",
                "host_event": "host_only_final_snapshot",
            },
        }
    ]
    assert trajectory["final_metrics"] == {"total_steps": 1}
    assert not trajectory["agent"].get("model_name")
    assert trajectory["extra"].get("subtrajectories", []) == []

    merge_events = [event for event in trajectory["extra"]["phase_events"] if event["phase"] == "merge"]
    assert [event["event"] for event in merge_events] == ["phase_start", "phase_end"]
    assert all(event["session_id"] == public_run.name for event in merge_events)
    assert merge_events[0]["scope_id"] == merge_events[1]["scope_id"]
    assert all(event["metadata"] == {"stage": "host-only-fixture"} for event in merge_events)
    assert merge_events[-1]["status"] == "succeeded"

    manifest = json.loads((archived_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["archive_status"] == "complete"
    assert manifest["phase_states"]["merge"] == {"ran": True, "status": "succeeded"}


@pytest.mark.parametrize("delegated", [False, True])
async def test_ac_merge_resume_on_tiny_diff(
    tiny_diff_target: Path,
    delegated: bool,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Issue #172: ``--start-at merge`` resume on a tiny diff routes to the single-stack merge writer, not the
    multi-stack merge agent."""

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tiny_diff_target)

    # Regression guard: the multi-stack merge agent must NOT run in
    # single_stack_mode. If the merge-resume branch misroutes here, raise.
    async def _fail_merge(*_a: Any, **_k: Any) -> None:
        raise AssertionError("phase_cross_stack_merge must not run in single_stack_mode")

    monkeypatch.setattr("daydream.deep.merge_steps.phase_cross_stack_merge", _fail_merge)
    # Stub the post-merge side effects so the run terminates cleanly.
    mute_side_effects()

    # Prime the deep artifacts the merge-resume branch reads from disk. The
    # tiny-diff collapse yields a ``generic`` (collapsed language) stack plus
    # the ``structure`` meta-stack, so records files must match both.
    _prime_merge_resume(
        tiny_diff_target,
        generic=[_record(id="gen-1", description="generic per-stack issue", evidence="api.py:1")],
        structure=[_record(id="structure-1", description="file-size budget violated", evidence="api.py:1")],
    )

    if delegated:
        _mark_delegated_artifacts(tiny_diff_target / ".daydream/deep", {"generic": ["api.py"]})

    rc = await run(make_config(tiny_diff_target, start_at="merge"))
    assert rc == 0

    items_file = tiny_diff_target / ".daydream" / "deep" / "merged-items.json"
    assert items_file.is_file(), "single-stack merge resume did not write merged-items.json"
    items = json.loads(items_file.read_text())["items"]
    # ``normalize_items`` reassigns ``id`` to a contiguous sequence, so assert on
    # the preserved ``description`` + ``lens`` instead. The generic record is
    # tagged per-stack and the structural record keeps its structural lens (AC6
    # — lens taxonomy survives the host-written merge).
    assert any(i.get("description") == "generic per-stack issue" and i.get("lens") == "per-stack" for i in items), (
        f"generic per-stack item missing or mislabeled: {items}"
    )
    assert any(i.get("description") == "file-size budget violated" and i.get("lens") == "structural" for i in items), (
        f"structural item missing or mislabeled: {items}"
    )

    structural = [item for item in items if item.get("lens") == "structural"]
    assert structural[0]["source_uids"] == ["structure:1"]


async def test_evidence_gate_drops_speculative_finding(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #227 (AC2/AC3/AC6): the structural evidence gate keeps an evidenced finding but drops a speculative
    one before it reaches merged-items.json."""

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [
        {
            "id": 1,
            "lens": "per-stack",
            "file": "api.py",
            "line": 42,
            "severity": "high",
            "description": "Grounded evidenced finding",
            "confidence": "HIGH",
            "rationale": "verified against src/foo.py",
            "evidence": "src/foo.py:42",
        },
        {
            "id": 2,
            "lens": "per-stack",
            "file": "App.tsx",
            "line": 1,
            "severity": "low",
            "description": "Speculative unfounded finding",
            "confidence": "LOW",
            "rationale": "inferred from the diff alone, no exploration evidence",
            "evidence": "",
        },
    ]

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    deep = multi_stack_target / ".daydream" / "deep"
    items = json.loads((deep / "merged-items.json").read_text())["items"]
    descriptions = [i.get("description") for i in items]
    assert "Grounded evidenced finding" in descriptions, f"evidenced finding was dropped: {descriptions}"
    assert "Speculative unfounded finding" not in descriptions, (
        f"speculative finding leaked into merged-items.json: {descriptions}"
    )

    report = (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert "Grounded evidenced finding" in report
    assert "Speculative unfounded finding" not in report, "speculative finding leaked into review-output.md"

    dropped = json.loads((deep / "dropped-speculative.json").read_text())
    assert dropped["dropped_count"] >= 1
    assert "Speculative unfounded finding" in json.dumps(dropped["dropped_items"])
    assert 2 in dropped["dropped_ids"]


async def test_evidence_gate_all_speculative_yields_empty(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Issue #227 (AC5, N=1): a single-stack run whose only findings are all speculative writes an EMPTY
    merged-items.json without crashing and records every drop -- never a silent success."""

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tiny_diff_target)
    mute_side_effects()

    deep = _prime_merge_resume(
        tiny_diff_target,
        generic=[
            _record(
                id="gen-1",
                description="speculative generic finding",
                confidence="MEDIUM",
                rationale="inferred from the diff alone, no exploration evidence",
                evidence="",
            )
        ],
        structure=[
            _record(
                id="structure-1",
                description="speculative structural finding",
                confidence="LOW",
                rationale="hunch",
                evidence="api.py:1",
            )
        ],
    )

    rc = await run(make_config(tiny_diff_target, start_at="merge"))
    assert rc == 0

    items = json.loads((deep / "merged-items.json").read_text())["items"]
    assert items == [], f"speculative findings survived the gate: {items}"

    dropped = json.loads((deep / "dropped-speculative.json").read_text())
    assert dropped["dropped_count"] == 2, dropped
    dropped_desc = json.dumps(dropped["dropped_items"])
    assert "speculative generic finding" in dropped_desc
    assert "speculative structural finding" in dropped_desc


async def test_evidence_gate_keeps_whole_file_structural_finding(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Issue #227 (findings 3/5): a structural (host-tagged, whole-file) finding with ``line: 0`` and colon-free
    evidence SURVIVES the gate -- the structural lens is high-conviction by construction and must not be
    demoted."""

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tiny_diff_target)
    mute_side_effects()

    deep = _prime_merge_resume(
        tiny_diff_target,
        generic=[
            _record(
                id="gen-1",
                description="grounded generic finding",
                confidence="MEDIUM",
                rationale="r",
                evidence="api.py:1",
            )
        ],
        structure=[
            _record(
                id="structure-1",
                description="module exceeds 800 LOC budget",
                file="big.py",
                line=0,
                confidence="HIGH",
                rationale="file-size budget violated",
                evidence="big.py is 800 lines",
            )
        ],
    )

    rc = await run(make_config(tiny_diff_target, start_at="merge"))
    assert rc == 0

    items = json.loads((deep / "merged-items.json").read_text())["items"]
    structural = [
        i for i in items if i.get("lens") == "structural" and i.get("description") == "module exceeds 800 LOC budget"
    ]
    assert structural, f"whole-file structural finding was dropped by the gate: {items}"
    assert structural[0].get("line") == 0


async def test_evidence_gate_clears_stale_dropped_sidecar(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Issue #227 (findings 4/6): a resume that drops 0 findings clears a stale ``dropped-speculative.json`` left
    by a prior run, so the sidecar cannot report phantom drops to eval/benchmark/human auditors."""

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tiny_diff_target)
    mute_side_effects()

    deep = _prime_merge_resume(
        tiny_diff_target,
        generic=[
            _record(
                id="gen-1",
                description="grounded generic finding",
                confidence="MEDIUM",
                rationale="r",
                evidence="api.py:1",
            )
        ],
        structure=[
            _record(
                id="structure-1",
                description="grounded structural finding",
                file="big.py",
                confidence="HIGH",
                rationale="r",
                evidence="big.py:1",
            )
        ],
    )
    # Stale sidecar from a prior run that dropped a finding.
    (deep / "dropped-speculative.json").write_text(
        json.dumps({"dropped_count": 1, "dropped_ids": [99], "dropped_items": [{"id": 99}]})
    )

    rc = await run(make_config(tiny_diff_target, start_at="merge"))
    assert rc == 0

    # The well-evidenced records survive (0 drops), so the sidecar is neither
    # rewritten nor left stale -- it must be gone.
    items = json.loads((deep / "merged-items.json").read_text())["items"]
    assert any(i.get("description") == "grounded generic finding" for i in items), items
    assert not (deep / "dropped-speculative.json").exists(), "stale dropped-speculative.json survived a 0-drop resume"
