"""Deep-mode orchestrator integration tests (plan 05-09).

Covers D-07..D-10, D-17, D-19..D-22, D-24..D-26, D-28, D-30, D-31,
D-34, D-35, D-44.

The tests share a stub backend (``tests.harness.stub_backend.StubBackend``)
that dispatches on prompt content to simulate the full review pipeline without
talking to a real SDK. The stub lives in
``tests.harness.stub_backend``; it is imported and re-aliased below to its
historical ``_``-private names (``_StubBackend``, ``_silence``, ...) so this
module's call sites -- and the sibling modules that
``from tests.test_deep_orchestrator import _StubBackend`` -- keep working now
that the canonical home is the harness, not this file.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import pytest

from daydream.backends import ResultEvent, TextEvent
from daydream.config import SKILL_MAP
from daydream.prompts.authorial_intent import AUTHORITATIVE_INTENT_RULE
from tests.harness.git_helpers import git as _git
from tests.harness.stub_backend import (
    PARTIAL_FIX_MARKER,
    StubBackend,
    force_interactive,
    install_stub_backend,
    silence,
)

if TYPE_CHECKING:
    from daydream.pr_review import PRInfo
    from daydream.runner import RunConfig

# Re-exported under their historical ``_``-private names so this module's call
# sites and the sibling modules that import them keep working unchanged.
_PARTIAL_FIX_MARKER = PARTIAL_FIX_MARKER
_StubBackend = StubBackend
_silence = silence
_force_interactive = force_interactive
_install_stub_backend = install_stub_backend


MakeConfig = Callable[..., "RunConfig"]
Mute = Callable[..., None]


def _install_model_capturing_stubs(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    *,
    parse_severity: str | None = None,
    merge_echo_records: bool = False,
    arbiter_omit_verdicts: bool = False,
    parse_by_stack: dict[str, dict[str, Any]] | None = None,
    suppression_keep: bool = True,
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

    def factory(name: str, model: str | None = None, **kwargs: object) -> _StubBackend:
        stub = _StubBackend(target, model=model or "mock-model", shared_calls=shared_calls)
        stub.parse_severity = parse_severity
        stub.merge_echo_records = merge_echo_records
        stub.arbiter_omit_verdicts = arbiter_omit_verdicts
        stub.parse_by_stack = parse_by_stack
        stub.suppression_keep = suppression_keep
        return stub

    monkeypatch.setattr("daydream.runner.create_backend", factory)
    monkeypatch.setattr("daydream.deep.orchestrator.get_installed_skills", lambda: None)
    monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)
    return shared_calls


async def _run_deep(target: Path, *, start_at: str = "review", precision_mode: bool = False) -> int:
    from daydream.runner import RunConfig, run

    # cleanup=False suppresses the interactive cleanup prompt; deep is the default.
    config = RunConfig(
        target=str(target), start_at=start_at, cleanup=False, precision_mode=precision_mode
    )
    return await run(config)


def _merge_item(item_id: int, file: str, severity: str, *, desc: str | None = None) -> dict[str, Any]:
    """Build a validated merged item (shape copied from the stub default)."""
    return {
        "id": item_id,
        "lens": "per-stack",
        "file": file,
        "line": 1,
        "severity": severity,
        "description": desc if desc is not None else f"{severity} issue in {file}",
        "confidence": "MEDIUM",
        "rationale": "rationale",
        "evidence": f"{file}:1",
    }


def _record(**overrides: Any) -> dict[str, Any]:
    """Build one on-disk per-stack record (the shape a merge resume reads back)."""
    record: dict[str, Any] = {"id": 1, "description": "issue", "file": "api.py", "line": 1}
    record.update(overrides)
    return record


def _write_matching_diff_key(target: Path, deep: Path) -> None:
    """Write ``diff-key`` for *target*'s current diff into *deep*.

    These primed artifacts stand in for a prior run over the same diff, so the
    key must match what ``run_deep``'s preamble computes.
    """
    from daydream import git_ops
    from daydream.deep.artifacts import diff_key, diff_key_path
    from daydream.workspace import _resolve_base

    base = _resolve_base(target, None, None)
    diff = git_ops.diff(target, base)
    diff_key_path(deep).write_text(diff_key(diff or ""), encoding="utf-8")


def _prime_merge_resume(
    target: Path,
    *,
    python: list[dict[str, Any]] | None = None,
    react: list[dict[str, Any]] | None = None,
    generic: list[dict[str, Any]] | None = None,
    structure: list[dict[str, Any]] | None = None,
) -> Path:
    """Prime the deep artifacts a ``--start-at`` resume reads, returning the deep dir.

    ``intent.md`` + ``alternatives.json`` are the TTT artifacts every resume gate
    requires. Any stack passed a record list also gets its
    ``stack-<name>-records.json``; a stack left as ``None`` is deliberately
    absent (the shape that drives the missing-records guard).
    """
    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "intent.md").write_text("primed intent")
    (deep / "alternatives.json").write_text("[]")
    # Freshness key: these artifacts stand in for a prior run over the SAME diff,
    # so the resume gate must see a matching key (a missing one refuses).
    _write_matching_diff_key(target, deep)
    for stack, records in (
        ("python", python),
        ("react", react),
        ("generic", generic),
        ("structure", structure),
    ):
        if records is not None:
            (deep / f"stack-{stack}-records.json").write_text(json.dumps(records))
    return deep


async def _ok(*_a: Any, **_k: Any) -> tuple[bool, int, bool]:
    """Async stand-in for phase_test_and_heal that always passes.

    Kept for the sibling modules that import it (tests/test_archive_data_capture.py);
    tests in this module use the ``mute_side_effects`` fixture instead.
    """
    return (True, 0, True)


async def _noop_commit(*_a: Any, **_k: Any) -> None:
    """Async no-op stand-in for phase_commit_push (see ``_ok`` on why it stays)."""
    return None


def _pin_findings_pr(monkeypatch: pytest.MonkeyPatch, target: Path) -> "PRInfo":
    """Provide the PR metadata required by the findings-out artifact."""
    from daydream import git_ops
    from daydream.pr_review import PRInfo

    head = git_ops.head_sha(target)
    base = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", "rev-parse", "main"],  # noqa: S607 - git is a trusted command
        cwd=target,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    pr = PRInfo(
        number=7,
        head_sha=head,
        base_sha=base,
        base_ref="main",
        owner="o",
        repo="r",
        url="https://example.invalid/pr/7",
    )
    monkeypatch.setattr("daydream.pr_review.find_pr_by_number", lambda target_dir, n: pr)
    return pr


async def test_supervise_rules_drops_deny_globbed_finding(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig
) -> None:
    """Rule supervision rewrites the canonical items before findings-out."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    _pin_findings_pr(monkeypatch, multi_stack_target)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [
        _merge_item(1, "vendor/generated.py", "high", desc="drop this finding"),
        _merge_item(2, "src/app.py", "low", desc="keep this finding"),
    ]
    (multi_stack_target / ".daydream.toml").write_text(
        'supervisor = "rules"\nsupervisor_deny_globs = ["vendor/**"]\n'
    )
    out = multi_stack_target / "findings.json"
    traj = tmp_path / "trajectory.json"

    async def _post_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("findings-out must not post to the PR")

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _post_forbidden)
    rc = await run(
        make_config(
            multi_stack_target,
            pr_number=7,
            findings_out=str(out),
            file_config=load_file_config(multi_stack_target),
            trajectory_path=traj,
        )
    )

    assert rc == 0
    items = json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())
    descriptions = [item["description"] for item in items["items"]]
    assert "drop this finding" not in descriptions
    assert "keep this finding" in descriptions
    findings = json.loads(out.read_text())["findings"]
    finding_descriptions = [finding["title"] for finding in findings]
    assert "drop this finding" not in finding_descriptions
    assert "keep this finding" in finding_descriptions
    events = _scan_phase_events(multi_stack_target / ".daydream", traj, "supervisor_verdict")
    assert any(
        event.get("metadata", {}).get("finding_id") == 1
        and event.get("metadata", {}).get("action") == "drop"
        for event in events
    )


async def test_supervise_hold_excluded_but_rendered(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """Held findings leave the actionable items but remain visible in the report."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [
        _merge_item(1, "vendor/generated.py", "high", desc="hold this finding"),
        _merge_item(2, "src/app.py", "low", desc="keep this finding"),
    ]
    stub.supervise_verdicts = {
        1: {"action": "hold", "reason": "needs human review"},
        2: {"action": "allow", "reason": "confirmed"},
    }
    (multi_stack_target / ".daydream.toml").write_text('supervisor = "llm"\n')

    rc = await run(
        make_config(multi_stack_target, file_config=load_file_config(multi_stack_target))
    )

    assert rc == 0
    payload = json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())
    actionable = [item["description"] for item in payload["items"]]
    held = [item["description"] for item in payload["held"]]
    assert "keep this finding" in actionable
    assert "hold this finding" in held
    report = (multi_stack_target / ".review-output.md").read_text()
    assert "Held Findings" in report
    assert "hold this finding" in report


async def test_supervise_llm_drop_records_step(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig
) -> None:
    """LLM supervision drops by canonical id and records its deep stage."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    _pin_findings_pr(monkeypatch, multi_stack_target)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [
        _merge_item(1, "api.py", "high", desc="drop by llm"),
        _merge_item(2, "App.tsx", "low", desc="keep by llm"),
    ]
    stub.supervise_verdicts = {
        1: {"action": "drop", "reason": "duplicate"},
        2: {"action": "allow", "reason": "confirmed"},
    }
    (multi_stack_target / ".daydream.toml").write_text('supervisor = "llm"\n')
    out = multi_stack_target / "findings.json"
    traj = tmp_path / "trajectory.json"

    async def _post_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("findings-out must not post to the PR")

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _post_forbidden)
    rc = await run(
        make_config(
            multi_stack_target,
            pr_number=7,
            findings_out=str(out),
            file_config=load_file_config(multi_stack_target),
            trajectory_path=traj,
        )
    )

    assert rc == 0
    items = json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())
    descriptions = [item["description"] for item in items["items"]]
    assert "drop by llm" not in descriptions
    assert "keep by llm" in descriptions
    starts = _scan_phase_events(multi_stack_target / ".daydream", traj, "phase_start")
    assert any(event.get("metadata", {}).get("stage") == "supervise" for event in starts)


async def test_supervise_llm_edit_revises_severity(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """LLM edit verdicts revise severity in canonical items and findings-out."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    _pin_findings_pr(monkeypatch, multi_stack_target)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high", desc="downgrade me")]
    stub.supervise_verdicts = {
        1: {"action": "edit", "reason": "less severe", "severity": "low"},
    }
    (multi_stack_target / ".daydream.toml").write_text('supervisor = "llm"\n')
    out = multi_stack_target / "findings.json"

    rc = await run(
        make_config(
            multi_stack_target,
            pr_number=7,
            findings_out=str(out),
            file_config=load_file_config(multi_stack_target),
        )
    )

    assert rc == 0
    payload = json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())
    revised = next(item for item in payload["items"] if item["description"] == "downgrade me")
    assert revised["severity"] == "low"
    findings = json.loads(out.read_text())["findings"]
    assert any(finding["title"] == "downgrade me" and finding["severity"] == "low" for finding in findings)


async def test_supervise_drop_all_writes_empty_artifact_exit_zero(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """All findings may be dropped while findings-out still writes an empty artifact."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    _pin_findings_pr(monkeypatch, multi_stack_target)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high", desc="drop everything")]
    (multi_stack_target / ".daydream.toml").write_text(
        'supervisor = "rules"\nsupervisor_deny_globs = ["**"]\n'
    )
    out = multi_stack_target / "findings.json"

    rc = await run(
        make_config(
            multi_stack_target,
            pr_number=7,
            findings_out=str(out),
            file_config=load_file_config(multi_stack_target),
        )
    )

    assert rc == 0
    assert json.loads(out.read_text())["findings"] == []


async def test_supervise_off_byte_identical(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """No config and explicit off produce the same canonical items bytes."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    _pin_findings_pr(monkeypatch, multi_stack_target)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [
        _merge_item(1, "api.py", "high", desc="first finding"),
        _merge_item(2, "App.tsx", "low", desc="second finding"),
    ]
    out = multi_stack_target / "findings.json"

    empty_config = load_file_config(multi_stack_target)
    first_rc = await run(
        make_config(
            multi_stack_target, pr_number=7, findings_out=str(out), file_config=empty_config
        )
    )
    first_items = (multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_bytes()
    first_findings = json.loads(out.read_text())["findings"]

    (multi_stack_target / ".daydream.toml").write_text('supervisor = "off"\n')
    second_rc = await run(
        make_config(
            multi_stack_target,
            pr_number=7,
            findings_out=str(out),
            file_config=load_file_config(multi_stack_target),
        )
    )
    second_items = (multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_bytes()
    second_findings = json.loads(out.read_text())["findings"]

    assert first_rc == second_rc == 0
    assert first_items == second_items
    assert first_findings == second_findings


async def test_supervise_dropped_finding_never_reaches_fix(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """A dropped finding is absent from the real fix prompt and remains unmodified."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [
        _merge_item(1, "api.py", "high", desc="drop before fix"),
        _merge_item(2, "App.tsx", "low", desc="fix this survivor"),
    ]
    (multi_stack_target / ".daydream.toml").write_text(
        'supervisor = "rules"\nsupervisor_deny_globs = ["api.py"]\n'
    )
    source_before = (multi_stack_target / "api.py").read_bytes()

    rc = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            file_config=load_file_config(multi_stack_target),
        )
    )

    assert rc == 0
    assert (multi_stack_target / "api.py").read_bytes() == source_before
    prompts = "\n".join(_fix_prompts(stub))
    assert "drop before fix" not in prompts


async def test_run_deep_renders_prescan_summary_not_json(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """Real-path: the pre-scan summary renders as a readable panel, not raw JSON.

    Drives ``run`` through ``run_deep`` with EXPLORATION_AVAILABLE left True so
    the real ``pre_scan`` branch executes and the stub answers the specialist
    prompts. Asserts the convention surfaces in the rendered summary and that
    no raw structured-output JSON envelope leaks to the terminal.
    """
    from rich.console import Console

    from daydream.runner import run

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
    monkeypatch.setattr("daydream.deep.orchestrator.console", rec)
    _install_stub_backend(monkeypatch, multi_stack_target, enable_exploration=True)

    exit_code = await run(
        make_config(multi_stack_target, assume="yes", output_mode="loop")
    )
    assert exit_code == 0
    out = rec.export_text()
    assert "OpenAPI First" in out  # convention surfaced by the summary
    assert '{"conventions"' not in out and "pattern-scanner" not in out  # no raw JSON envelope


async def test_parallel_fix_applies_all_disjoint_files(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """AC#3: every disjoint-file group is applied (each per-file sentinel lands).

    A serial loop + single-sentinel stub would fail this -- it asserts EVERY
    per-file ``.fixed-*`` marker, not just the last one written.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    files = ["f1.py", "f2.py", "f3.py", "f4.py"]
    stub.merge_items = [_merge_item(i + 1, f, "high") for i, f in enumerate(files)]
    exit_code = await run(
        make_config(
            multi_stack_target, assume="yes", output_mode="loop", non_interactive=False
        )
    )
    assert exit_code == 0
    for f in files:
        assert (multi_stack_target / f".fixed-{f.replace('.', '_')}").exists()


async def test_parallel_fix_same_file_no_race(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """3 items on ONE file + 1 on another. The 3 same-file findings collapse into
    ONE batched fix turn that addresses every marker in severity order, while the
    other file's group runs concurrently. The read-modify-write append +
    anyio.sleep(0) makes any cross-file race deterministic; per-file partitioning
    keeps shared.py's markers ordered and intact.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    shared = multi_stack_target / "shared.py"
    stub.fix_append_path = shared
    stub.merge_items = [
        _merge_item(1, "shared.py", "high", desc="marker-1"),
        _merge_item(2, "shared.py", "medium", desc="marker-2"),
        _merge_item(3, "shared.py", "low", desc="marker-3"),
        _merge_item(4, "other.py", "high", desc="other"),
    ]
    exit_code = await run(
        make_config(
            multi_stack_target, assume="yes", output_mode="loop", non_interactive=False
        )
    )
    assert exit_code == 0
    assert shared.read_text().split() == ["marker-1", "marker-2", "marker-3"]


async def test_parallel_fix_failure_isolated_returns_nonzero(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """AC#5: a failed fix group is isolated, surfaced, and exits nonzero.

    The bad.py group raises; its sentinel must be absent while the other groups
    apply, a warning naming bad.py must surface (non-silent), commit must be
    skipped, and the run must exit 1 (locked nonzero-on-failure decision).
    """
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects(commit=False)
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
        make_config(
            multi_stack_target, assume="yes", output_mode="loop", non_interactive=False
        )
    )
    assert exit_code == 1  # decision: nonzero on failure
    assert (multi_stack_target / ".fixed-good1_py").exists()  # other groups applied
    assert (multi_stack_target / ".fixed-good2_py").exists()
    assert not (multi_stack_target / ".fixed-bad_py").exists()  # failed group did not apply
    assert any("bad.py" in m for m in warnings)  # non-silent
    assert commit_calls == []  # no commit on failure


async def test_fix_failure_reverts_partial_edit_and_marks_manifest_partial(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real-path: a fix group that raises MaxTurnsError mid-edit is rolled back,
    its partial content saved, and the archived run is marked ``partial``.

    Two fix groups with mixed outcomes (the structural shape that hides bugs):
    ``api.py`` succeeds (its sentinel lands) while ``App.tsx`` writes a broken
    partial edit and then raises ``MaxTurnsError``. Drives ``runner.run`` through
    the deep fix path with a real temp git worktree + real archive dir, mocking
    only the backend. Asserts observable outcomes:

      (a) the archived ``manifest.json`` has ``status == "partial"`` and
          ``fix_failures`` names the failed group;
      (b) the SUCCESSFUL group's edit (its sentinel) survives in the tree;
      (c) the FAILED group's file is reverted to its pre-fix content AND a
          recovery patch was written under ``.daydream/partial-fixes/``.

    Fails if the persistence/revert is removed: without (a) the manifest stays
    ``complete``; without (c) ``App.tsx`` keeps the broken partial edit.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.fix_partial_then_maxturns = "App.tsx"
    stub.merge_items = [
        _merge_item(1, "api.py", "high"),
        _merge_item(2, "App.tsx", "high"),
    ]
    pre_fix_apptsx = (multi_stack_target / "App.tsx").read_text()

    exit_code = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
            archive=True,
        )
    )
    assert exit_code == 1  # dropped fix group => nonzero

    # (b) successful group applied and survives.
    assert (multi_stack_target / ".fixed-api_py").exists()

    # (c) failed group reverted to pre-fix content; broken edit gone.
    apptsx_after = (multi_stack_target / "App.tsx").read_text()
    assert apptsx_after == pre_fix_apptsx
    assert _PARTIAL_FIX_MARKER not in apptsx_after
    patches = list((multi_stack_target / ".daydream" / "partial-fixes").glob("*.patch"))
    assert patches, "expected a recovery patch for the reverted partial fix"
    assert any("App.tsx" in p.read_text() for p in patches), "patch must capture the partial edit"

    # (a) manifest records the failure and is no longer "complete".
    run_dirs = list((archive_dir / "runs").iterdir())
    assert len(run_dirs) == 1, f"expected exactly one archived run, got {run_dirs}"
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["fix_failures"], "manifest must record the dropped fix group"
    assert any("App.tsx" in key for key in manifest["fix_failures"])


async def test_fix_failure_enumerates_leftover_untracked_orphan_in_manifest(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real-path: a stray untracked file a failed group creates -- one that is
    NOT the group's key file -- is enumerated in the manifest and never deleted.

    The failing ``App.tsx`` group writes a stray ``store/uuid.go`` before raising
    ``MaxTurnsError``. Because parallel groups share one tree, that orphan can't
    be attributed to a group, so the orchestrator records it (never deletes it):

      (a) ``manifest.json`` lists ``store/uuid.go`` in ``fix_leftover_untracked``;
      (b) the orphan still EXISTS in the tree (no risk of deleting good work);
      (c) the run is still ``status == "partial"``.

    Fails if the enumeration is removed: without (a) the orphan is invisible in
    the archive -- the exact "half-broken tree presented as clean" gap.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.fix_partial_then_maxturns = "App.tsx"
    stub.fix_orphan_file = "store/uuid.go"
    stub.merge_items = [
        _merge_item(1, "api.py", "high"),
        _merge_item(2, "App.tsx", "high"),
    ]

    exit_code = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
            archive=True,
        )
    )
    assert exit_code == 1

    # (b) the unattributable orphan is preserved, never deleted.
    assert (multi_stack_target / "store" / "uuid.go").exists()

    run_dirs = list((archive_dir / "runs").iterdir())
    assert len(run_dirs) == 1, f"expected exactly one archived run, got {run_dirs}"
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    # (c) partial, and (a) the orphan is enumerated for audit.
    assert manifest["status"] == "partial"
    leftover = manifest["fix_leftover_untracked"]
    assert leftover, "manifest must enumerate untracked files left by the failed fix pass"
    assert "store/uuid.go" in leftover


async def test_parallel_fix_commit_runs_once_after_all(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """AC#6: commit stays serial and runs exactly once, after every parallel fix lands.

    Each fix writes its own ``.fixed-*`` sentinel; the spy commit records whether ALL
    sentinels already exist at commit time. ``seen_at_commit == [True]`` proves a single
    commit that observed every fix -- a regression moving commit inside the fan-out would
    see a partial set (False) or commit more than once.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects(commit=False)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    files = ["f1.py", "f2.py", "f3.py"]
    stub.merge_items = [_merge_item(i + 1, f, "high") for i, f in enumerate(files)]
    seen_at_commit: list[bool] = []

    async def _spy_commit(backend: Any, work: Any) -> None:
        seen_at_commit.append(
            all((multi_stack_target / f".fixed-{f.replace('.', '_')}").exists() for f in files)
        )

    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _spy_commit)
    exit_code = await run(
        make_config(
            multi_stack_target, assume="yes", output_mode="loop", non_interactive=False
        )
    )
    assert exit_code == 0
    assert seen_at_commit == [True]  # exactly one commit, and every fix already landed


INTENT_SENTINEL = "SKIP_IF_NO_QUERY_IS_A_DELIBERATE_GUARD"


def _fix_prompts(stub: _StubBackend) -> list[str]:
    # Same-file findings are batched into one "Fix these N issues" turn; a lone
    # finding still uses the single-finding "Fix this issue" prompt. Match both.
    return [c["prompt"] for c in stub.calls if c["prompt"].startswith(("Fix this issue", "Fix these"))]


async def test_fix_tool_veto_blocks_denied_write(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Built-in rules veto a deferred denied Write and record the abort/event."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high", desc="protected write")]
    stub.deferred_write_pairs = ["api.py"]
    (multi_stack_target / ".daydream.toml").write_text(
        'tool_supervisor = "rules"\nsupervisor_deny_globs = ["api.py"]\n'
    )
    source_before = (multi_stack_target / "api.py").read_bytes()
    traj = tmp_path / "trajectory.json"

    rc = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            file_config=load_file_config(multi_stack_target),
            trajectory_path=traj,
        )
    )

    assert isinstance(rc, int)
    assert (multi_stack_target / "api.py").read_bytes() == source_before
    stop_reasons = _scan_trajectory_extra(multi_stack_target / ".daydream", traj, "stop_reason")
    assert "tool_vetoed:Write" in stop_reasons
    events = _scan_phase_events(multi_stack_target / ".daydream", traj, "tool_veto")
    assert any(event.get("metadata", {}).get("tool_name") == "Write" for event in events)


async def test_fix_tool_veto_allows_unmatched_write(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """Built-in rules allow a Write whose path does not match the deny glob."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "App.tsx", "high", desc="allowed write")]
    stub.deferred_write_pairs = ["App.tsx"]
    (multi_stack_target / ".daydream.toml").write_text(
        'tool_supervisor = "rules"\nsupervisor_deny_globs = ["api.py"]\n'
    )

    rc = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            file_config=load_file_config(multi_stack_target),
        )
    )

    assert isinstance(rc, int)
    assert (multi_stack_target / "App.tsx").read_text() == "backend resumed"
    assert not _scan_trajectory_extra(multi_stack_target / ".daydream", Path("/missing"), "stop_reason")


async def test_fix_tool_veto_stops_subsequent_calls(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """A vetoed first deferred Write prevents the generator's later Write."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high", desc="first"), _merge_item(2, "App.tsx", "low", desc="second")]
    stub.deferred_write_pairs = ["api.py", "App.tsx"]
    (multi_stack_target / ".daydream.toml").write_text(
        'tool_supervisor = "rules"\nsupervisor_deny_globs = ["api.py"]\n'
    )
    api_before = (multi_stack_target / "api.py").read_bytes()
    app_before = (multi_stack_target / "App.tsx").read_bytes()

    rc = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            file_config=load_file_config(multi_stack_target),
        )
    )

    assert isinstance(rc, int)
    assert (multi_stack_target / "api.py").read_bytes() == api_before
    assert (multi_stack_target / "App.tsx").read_bytes() == app_before


async def test_fix_tool_supervisor_off_writes(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """With tool supervision off, the deferred Write resumes and writes."""
    from daydream.config_file import load_file_config
    from daydream.runner import run

    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high", desc="unprotected write")]
    stub.deferred_write_pairs = ["api.py"]
    (multi_stack_target / ".daydream.toml").write_text('tool_supervisor = "off"\n')

    rc = await run(
        make_config(
            multi_stack_target,
            assume="yes",
            output_mode="loop",
            file_config=load_file_config(multi_stack_target),
        )
    )

    assert isinstance(rc, int)
    assert (multi_stack_target / "api.py").read_text() == "backend resumed"


async def test_confirmed_intent_reaches_fix_prompt(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """The confirmed author intent reaches every deep fix prompt so a fixer
    can't undo a deliberate decision.

    Real-path through ``runner.run`` to the fix gate (``assume="yes"``). The PR
    body carries ``INTENT_SENTINEL``; the stub's intent branch echoes it into
    the confirmed-intent file (intent_p), and the fix phase must inline that
    file's text plus the "don't undo deliberate intent" rule into each fix
    prompt. Asserts on observable fix-prompt content, not that a call happened.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    monkeypatch.setattr(
        "daydream.git_ops.gh_pr_view",
        lambda repo, pr=None: {"body": INTENT_SENTINEL},
    )
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(1, "api.py", "high")]

    rc = await run(
        make_config(
            multi_stack_target,
            pr_number=7,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
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
    """Default deep flow preserves stage order, isolation, artifacts, prompts, and report."""
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
    # Wonder now runs CONCURRENTLY with the per-stack fan-out on a multi-stack
    # run, so it no longer strictly precedes it; the guarantee is that it joins
    # before parse consumes alternatives.json.
    assert first["alternatives"] < first["parse"]
    assert first["per-stack"] < first["parse"]
    assert first["parse"] < first["merge"]

    # At minimum: intent + alternatives + 3 per-stack + 3 parse + 1 merge = 9 distinct calls.
    assert len(stub.calls) >= 9
    # Each stage fires a distinct execute call -- prompts must be unique.
    prompts = [c["prompt"] for c in stub.calls]
    assert len(set(prompts)) == len(prompts)

    deep = multi_stack_target / ".daydream" / "deep"
    assert (deep / "intent.md").exists()
    assert (deep / "alternatives.json").exists()
    review_files = list(deep.glob("stack-*-review.md"))
    records_files = list(deep.glob("stack-*-records.json"))
    assert review_files, "expected at least one stack-*-review.md"
    assert records_files, "expected at least one stack-*-records.json"
    assert (deep / "dedup-candidates.json").exists()

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

    # Every execute call must have agents=None per D-38.
    assert all(c["agents"] is None for c in stub.calls)

    assert per_stack_prompts
    for p in per_stack_prompts:
        assert "intent.md" in p
        # Multi-stack: wonder runs alongside this fan-out, so alternatives.json
        # does not exist yet and its pointer is deliberately omitted.
        assert "alternatives.json" not in p

    # The fixture's diff is mixed, so the generic bucket is NOT docs-only (no
    # notice). Contract: a generic-fallback prompt is emitted for README.md.
    fallback_prompts = [
        c["prompt"] for c in stub.calls if "you are reviewing the generic-fallback stack" in c["prompt"].lower()
    ]
    assert fallback_prompts
    assert any("README.md" in p for p in fallback_prompts)

    parse_calls = [
        c for c in stub.calls if "extract only actionable issues" in c["prompt"].lower()
    ]
    per_stack_outputs = list((multi_stack_target / ".daydream" / "deep").glob("stack-*-review.md"))
    assert len(parse_calls) >= len(per_stack_outputs)
    records = list((multi_stack_target / ".daydream" / "deep").glob("stack-*-records.json"))
    assert len(records) == len(per_stack_outputs)

    from daydream.config import REVIEW_OUTPUT_FILE

    assert (multi_stack_target / REVIEW_OUTPUT_FILE).exists()
    text = (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert "## Issues" in text
    assert "## Cross-Stack Issues" in text
    # Numbering continues: 1., 2. in ## Issues then 3. in ## Cross-Stack Issues.
    assert "3." in text.split("## Cross-Stack Issues", 1)[1]
    cross_section = text.split("## Cross-Stack Issues", 1)[1]
    assert "[cross-stack]" in cross_section


PR_SENTINEL = "DELIBERATE_RATIO_PASS_THROUGH_IS_INTENTIONAL"


def _intent_prompt(stub: _StubBackend) -> str:
    """Recover the intent-phase prompt by its stable instruction text."""
    return next(c["prompt"] for c in stub.calls if "understand the intent of these changes" in c["prompt"].lower())


def _review_prompts_by_kind(stub: _StubBackend) -> dict[str, list[str]]:
    """Classify captured prompts for the five finding-producing builders (#279).

    Keys: ``per-stack``, ``generic-fallback``, ``structural``, ``arbiter``,
    ``merge``. Per-stack matches skilled stack reviews; generic-fallback is
    the README / missing-skill path. Structural / arbiter / merge use their
    own stable opening phrases.
    """
    by_kind: dict[str, list[str]] = {
        "per-stack": [],
        "generic-fallback": [],
        "structural": [],
        "arbiter": [],
        "merge": [],
    }
    for c in stub.calls:
        pl = c["prompt"].lower()
        if "you are reviewing the generic-fallback stack" in pl:
            by_kind["generic-fallback"].append(c["prompt"])
        elif "you are reviewing the " in pl:
            by_kind["per-stack"].append(c["prompt"])
        elif "you are the structural reviewer" in pl:
            by_kind["structural"].append(c["prompt"])
        elif "you are the arbiter" in pl:
            by_kind["arbiter"].append(c["prompt"])
        elif "cross-stack merge agent" in pl:
            by_kind["merge"].append(c["prompt"])
    return by_kind


def _assert_authoritative_rule_gated(stub: _StubBackend, *, expect_present: bool) -> None:
    """Assert the precedence rule is present/absent in every finding-producing prompt."""
    by_kind = _review_prompts_by_kind(stub)
    # Arbiter only runs when parse emits high/contested; tests that call this
    # helper set parse_severity="high" so all five kinds are exercised.
    missing = [k for k, prompts in by_kind.items() if not prompts]
    assert not missing, f"expected prompts for all five kinds, missing: {missing}"
    for kind, prompts in by_kind.items():
        if expect_present:
            assert all(AUTHORITATIVE_INTENT_RULE in p for p in prompts), (
                f"{kind}: expected AUTHORITATIVE_INTENT_RULE in every prompt"
            )
        else:
            assert all(AUTHORITATIVE_INTENT_RULE not in p for p in prompts), (
                f"{kind}: expected AUTHORITATIVE_INTENT_RULE absent from every prompt"
            )


async def test_pr_body_reaches_intent_prompt(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig
) -> None:
    """The PR description body is threaded into the initial intent prompt."""
    from daydream.runner import run

    _silence(monkeypatch)
    monkeypatch.setattr(
        "daydream.git_ops.gh_pr_view",
        lambda repo, pr=None: {"number": 7, "body": PR_SENTINEL},
    )
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    # High severity so the scoped arbiter fires and all five builders are covered.
    stub.parse_severity = "high"

    rc = await run(make_config(multi_stack_target, pr_number=7))
    assert rc == 0
    assert PR_SENTINEL in _intent_prompt(stub)
    _assert_authoritative_rule_gated(stub, expect_present=True)


async def test_no_pr_body_degrades_cleanly(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig
) -> None:
    """No PR body -> intent prompt carries no PR-description section.

    Also asserts the diff-is-the-target directives: the intent prompt must
    ground the agent in the diff, tell it the run is not tied to a GitHub pull
    request (so it never hunts for or asks about open PRs), and forbid
    skill/slash-command invocation. This fixture's diff is under
    ``INLINE_DIFF_BUDGET_BYTES``, so the diff arrives inlined with a
    do-not-re-Read clause rather than as a bare pointer.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda repo, pr=None: None)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"

    rc = await run(make_config(multi_stack_target, pr_number=7))
    assert rc == 0
    intent = _intent_prompt(stub)
    assert PR_SENTINEL not in intent
    assert "pull request description" not in intent.lower()
    assert "diff --git" in intent  # inlined, not pointed at
    assert "do NOT re-Read" in intent
    assert ".daydream/diff.patch" in intent
    assert "not tied to a GitHub pull request" in intent
    assert "Do not invoke any skills or slash commands" in intent
    _assert_authoritative_rule_gated(stub, expect_present=False)


async def test_whitespace_only_pr_body_is_not_authoritative(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig
) -> None:
    """Whitespace-only PR bodies must not publish intent_authoritative (#279).

    ``build_intent_prompt`` strips and ignores blank bodies; the orchestrator
    flag must match so downstream prompts never get the precedence rule without
    author-stated intent.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    monkeypatch.setattr(
        "daydream.git_ops.gh_pr_view",
        lambda repo, pr=None: {"number": 7, "body": "   \n\t  "},
    )
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"

    rc = await run(make_config(multi_stack_target, pr_number=7))
    assert rc == 0
    intent = _intent_prompt(stub)
    assert "pull request description" not in intent.lower()
    assert AUTHORITATIVE_INTENT_RULE not in intent
    _assert_authoritative_rule_gated(stub, expect_present=False)


async def test_non_interactive_intent_prompt_carries_pr_body(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
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
    from daydream.runner import run

    _silence_gate_noise(monkeypatch)
    mute_side_effects()
    monkeypatch.setattr(
        "daydream.git_ops.gh_pr_view",
        lambda repo, pr=None: {"number": 7, "body": PR_SENTINEL},
    )
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    def _forbidden_input(*_a: Any, **_kw: Any) -> str:
        raise AssertionError("input() was called in non-interactive mode -- stdin must not be touched")

    monkeypatch.setattr("builtins.input", _forbidden_input)

    reset_state()
    rc = -1
    try:
        assert get_non_interactive() is False
        rc = await run(make_config(multi_stack_target, pr_number=7))
        assert get_non_interactive() is True
    finally:
        reset_state()

    assert rc == 0
    assert PR_SENTINEL in _intent_prompt(stub)


@pytest.mark.asyncio
async def test_non_open_pr_state_suppresses_pr_body(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig
) -> None:
    """When gh_pr_view returns a non-OPEN state (CLOSED or MERGED), the
    orchestrator must NOT thread the PR body into the intent prompt — trusting
    a stale description would be wrong.  Asserts on the observable prompt
    content, not on internal state."""
    from daydream.runner import run

    _silence(monkeypatch)
    for state in ("CLOSED", "MERGED"):
        monkeypatch.setattr(
            "daydream.git_ops.gh_pr_view",
            lambda repo, pr=None, _s=state: {"number": 7, "body": PR_SENTINEL, "state": _s},
        )
        stub = _install_stub_backend(monkeypatch, multi_stack_target)

        rc = await run(make_config(multi_stack_target, pr_number=7))
        assert rc == 0
        intent = _intent_prompt(stub)
        assert PR_SENTINEL not in intent, f"PR body must be suppressed when state={state!r}"
        assert "pull request description" not in intent.lower(), (
            f"PR section header must be absent when state={state!r}"
        )


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


async def test_yes_auto_applies_fix(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig
) -> None:
    """Task 6 real-path: ``--yes`` (assume="yes") auto-applies fixes without prompting.

    Drives ``runner.run`` through the deep orchestrator's fix gate with
    ``assume="yes"``. The gate must NOT call ``prompt_user`` and MUST proceed to
    ``phase_fix`` — the observable consequence is the sentinel file the stub
    writes when it receives a fix prompt.
    """
    from daydream.runner import run

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

    exit_code = await run(make_config(multi_stack_target, assume="yes", output_mode="loop"))

    assert exit_code == 0
    assert not any(
        "apply" in msg.lower() or "fix" in msg.lower() for msg, _ in prompt_calls
    ), f"fix gate prompted under --yes: {prompt_calls}"
    # Observable consequence: the fix landed.
    assert fix_marker.exists(), "phase_fix never ran -> --yes did not auto-apply"


async def test_preflight_notice(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-30: pre-flight notice lists stages, stacks, skill per stack, total agent count."""
    captured: list[dict[str, Any]] = []

    def _capture(console, *, stages, stack_lines, agent_count, exploration_available) -> None:
        captured.append(
            {
                "stages": stages,
                "stack_lines": stack_lines,
                "agent_count": agent_count,
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


async def test_resume_per_stack_reruns_all(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-34: --start-at per-stack re-runs ALL per-stack reviews (after priming TTT artifacts)."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    _prime_merge_resume(multi_stack_target)

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
    deep = _prime_merge_resume(multi_stack_target)
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
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    # Records are primed but NOT the review.md files -- resume must consume
    # records.json. Every detected stack (including the generic bucket the
    # markdown file routes to, and the structure meta-stack) needs records, else
    # the merge-resume validation fails the run.
    _prime_merge_resume(
        multi_stack_target,
        python=[_record(description="py issue")],
        react=[_record(description="tsx issue", file="App.tsx")],
        generic=[_record(description="docs issue", file="README.md")],
        structure=[_record(description="structural issue")],
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


def _registry_text(plugin_names: list[str]) -> str:
    return (
        '{"version": 2, "plugins": {'
        + ", ".join(f'"{name}@marketplace": []' for name in plugin_names)
        + "}}"
    )


def _write_plugin_registry(config_dir: Path, plugin_names: list[str]) -> None:
    registry = config_dir / "plugins" / "installed_plugins.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(_registry_text(plugin_names))


@pytest.mark.parametrize(
    ("registry_text", "expected"),
    [
        pytest.param(
            _registry_text([skill.split(":", 1)[0] for skill in SKILL_MAP.values()]),
            set(SKILL_MAP.keys()),
            id="all-plugins-present-full-coverage",
        ),
        pytest.param(
            _registry_text(["beagle-python", "beagle-react"]),
            {"python", "react"},
            id="missing-beagle-go-excludes-go",
        ),
        pytest.param(None, None, id="missing-registry-signals-unknown"),
        pytest.param("not json {{{", None, id="unparseable-registry-optimistic"),
        # Regression: `data.get("plugins", {})` raised AttributeError when the
        # registry parsed to a non-dict, aborting deep mode instead of returning None.
        pytest.param("[]", None, id="non-dict-root-payload"),
        # Regression: iterating ``data.get("plugins", {})`` raised TypeError when the
        # `plugins` field was e.g. a list, aborting deep mode instead of returning None.
        pytest.param(
            '{"version": 2, "plugins": ["beagle-python@marketplace"]}',
            None,
            id="non-dict-plugins-field",
        ),
    ],
)
def test_get_installed_skills(
    registry_text: str | None, expected: set[str] | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registry shape -> resolved skill availability (``None`` == unknown, fall
    back to optimistic availability)."""
    from daydream.deep.orchestrator import get_installed_skills

    if registry_text is not None:
        registry = tmp_path / "plugins" / "installed_plugins.json"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(registry_text)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    assert get_installed_skills() == expected


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


async def test_resume_merge_errors_on_missing_stack_records(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--start-at merge must fail loudly when a detected stack has no records.

    Regression: previously the merge branch globbed whatever ``stack-*-records.json``
    files happened to exist on disk, so a detected stack with no prior records
    would silently disappear from the merged report.
    """
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    # Records for python only; react and generic are missing.
    _prime_merge_resume(multi_stack_target, python=[_record(description="py issue")])

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
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    # No records for the generic bucket, but it's listed as a prior failure.
    deep = _prime_merge_resume(
        multi_stack_target,
        python=[_record(description="py issue")],
        react=[_record(description="tsx issue", file="App.tsx")],
        structure=[_record(description="structural issue")],
    )
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

    # The structural record carries a sentinel id so we can verify it never lands
    # in the dedup input lists.
    _prime_merge_resume(
        multi_stack_target,
        python=[_record(id="py-1", description="py issue")],
        react=[_record(id="react-1", description="tsx issue", file="App.tsx")],
        generic=[_record(id="generic-1", description="docs issue", file="README.md")],
        structure=[_record(id="structure-1", description="1000-line file budget violated")],
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
    deep = _prime_merge_resume(multi_stack_target)
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
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute
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

    def spy(config, phase, cache=None, *, cwd=None):
        seen_phases.append(phase)
        return original(config, phase, cache, cwd=cwd)

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

    # Stub the outward-facing tail phases (they still trigger their resolver call)
    # plus phase_fix, so the fix loop doesn't mutate the workspace.
    mute_side_effects()

    async def _stub_fix(backend, work, item, idx, total, **kwargs):  # noqa: ARG001
        return None

    monkeypatch.setattr("daydream.phases.phase_fix", _stub_fix)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    expected_phases = {"intent", "wonder", "per_stack_review", "parse", "merge", "fix", "test", "verify"}
    captured = set(seen_phases)
    missing = expected_phases - captured
    assert not missing, (
        f"Deep orchestrator missing per-phase resolver calls for {missing}; "
        f"got {sorted(captured)}"
    )


def test_intent_phase_resolves_to_sonnet_default(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC3: the ``intent`` phase resolves to ``claude-sonnet-5`` by default.

    Intent summarization is a single mid-complexity turn that does not need Opus;
    Sonnet matches the FIX/TEST/EXPLORATION/PER_STACK_REVIEW tier. Captures the
    ``model=`` passed to ``create_backend`` and asserts the default, then that an
    explicit ``RunConfig(model=...)`` override still wins.
    """
    from daydream.runner import RunConfig, _resolve_backend

    captured: dict[str, Any] = {}

    class _B:
        def __init__(self, model: str | None) -> None:
            self.model = model

    def fake_create(name: str, model: str | None = None, **kwargs: object) -> _B:  # noqa: ARG001
        captured["model"] = model
        return _B(model)

    monkeypatch.setattr("daydream.runner.create_backend", fake_create)

    # Default: intent lands on Sonnet (mid tier), not Opus.
    backend = _resolve_backend(RunConfig(), "intent", {})
    assert backend.model == "claude-sonnet-5", (
        f"intent phase default should be claude-sonnet-5, got {backend.model!r}"
    )

    # An explicit global model override still wins over the phase default.
    backend_override = _resolve_backend(RunConfig(model="claude-opus-5"), "intent", {})
    assert backend_override.model == "claude-opus-5", (
        f"RunConfig(model=...) override should win for intent, got {backend_override.model!r}"
    )


async def test_intent_phase_runs_on_sonnet_through_runner_run(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#171 real-path: the intent phase Sonnet downgrade must be observable
    through the runner.run production entrypoint, not only at the unit seam.

    The unit-level assertion (test_intent_phase_resolves_to_sonnet_default) pins
    ``_resolve_backend(RunConfig(), "intent", {})`` directly. This is its
    real-path counterpart: it drives runner.run via _run_deep (which calls
    ``await run(config)``), captures the model on the backend that actually
    executes the intent prompt via _install_model_capturing_stubs, and asserts
    it is claude-sonnet-5 (mid tier). Mirrors the exact pattern of
    test_per_stack_sonnet_merge_opus_and_arbiter_on_high_severity. Regression: a
    revert of the intent default to Opus fails this assertion.
    """
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(
        monkeypatch, multi_stack_target, parse_severity="high", merge_echo_records=True
    )

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # "understand the intent of these changes" is unique to build_intent_prompt
    # (phases.py) -- the established intent-phase prompt discriminator.
    intent_models = [
        c["model"]
        for c in calls
        if "understand the intent of these changes" in c["prompt"].lower()
    ]
    assert intent_models, "intent phase did not execute through runner.run"
    assert set(intent_models) == {"claude-sonnet-5"}, (
        f"intent phase should run on claude-sonnet-5 (mid tier), got {sorted(intent_models)!r}"
    )


async def test_verifier_runs_after_merge_before_fix(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute
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

    # phase_fix stays REAL so verdict propagation is observable.
    stub = _install_accept_gate_pipeline(monkeypatch, multi_stack_target, mute_side_effects)

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
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute
) -> None:
    """When the verifier returns `contradicts` for an issue_id matching a parsed
    feedback item, the orchestrator attaches the verdict and phase_fix inlines
    `Verifier verdict: contradicts` into the fix-agent prompt.
    """
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
    fix_prompts = [
        c["prompt"]
        for c in stub.calls
        if c["prompt"].lower().startswith(("fix this issue:", "fix these"))
    ]
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
    # AC2 regression guard: when the gate accepts, verify runs and the verdicts
    # artifact lands on disk (same path asserted by the ordering test).
    assert (multi_stack_target / ".daydream" / "deep" / "recommendation-verdicts.json").is_file(), (
        "verdicts file missing -- verify did not run when the gate accepted"
    )


async def test_heal_loop_receives_feedback_items_in_fix_prompt(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute
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
    _silence(monkeypatch, prompts=False)
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
        "scope instruction missing from heal fix prompt -- feedback_items not "
        f"honored; prompt was: {heal_prompt!r}"
    )
    # First call failed, second passed: exactly two runs.
    assert stub.test_suite_calls == 2, (
        f"expected 2 test-suite runs (fail then pass), saw {stub.test_suite_calls}"
    )


async def test_structural_finding_reaches_fix_loop(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute
) -> None:
    """The fix gate feeds the canonical merged-items.json (structural included),
    severity-ordered, into phase_fix -- never the LLM re-parse that dropped
    structural findings.

    Observable consequence: every item that reaches phase_fix is captured. The
    structural item (lens="structural") MUST appear (not silently dropped by a
    markdown re-parse), and items MUST arrive severity-ordered (high before low,
    stable within a tier).
    """
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
    async def _capture_fix(backend, work, items, item_nums, total, **kwargs):  # noqa: ARG001
        fixed.extend(items)

    monkeypatch.setattr("daydream.phases.phase_fix_batched", _capture_fix)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    assert any(i.get("lens") == "structural" for i in fixed), (
        "structural finding never reached phase_fix -- it was dropped before the "
        f"fix loop; items fixed: {[(i.get('lens'), i.get('severity')) for i in fixed]!r}"
    )
    sev = [str(i["severity"]) for i in fixed]
    assert sev == sorted(sev, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])


async def test_start_at_fix_recovers_merged_items(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute
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

    _install_accept_gate_pipeline(monkeypatch, multi_stack_target, mute_side_effects)

    fixed: list[dict[str, Any]] = []

    async def _capture_fix(backend, work, item, idx, total, **kwargs):  # noqa: ARG001
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
        "phase_fix received an item that did not originate from the deep-dir "
        f"merged-items.json; got {fixed!r}"
    )
    # AC5 regression guard: a --start-at fix resume that applies fixes (gate
    # accepted) still produces verdicts -- verify runs post-gate-accept on resume.
    assert (multi_stack_target / ".daydream" / "deep" / "recommendation-verdicts.json").is_file(), (
        "verdicts file missing on --start-at fix resume -- verify must run when "
        "the gate accepts and fixes are applied"
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
    _silence(monkeypatch, prompts=False)


async def test_apply_fixes_gate_non_interactive_takes_safe_default(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real-path: non-interactive deep run declines fixes and exits 0 without
    reading stdin.

    Drives ``runner.run`` -> ``run_deep`` (deep is the default dispatch) with a
    mock backend to the real ``prompt_user`` apply-fixes gate. With
    ``config.non_interactive=True`` propagated by ``run``, the gate must
    short-circuit to its "n" default -- never touching stdin -- so the run
    declines fixes and returns 0.

    AC1: because the gate declines, the recommendation verifier must NOT run --
    no ``verify`` value among the run's trajectory ``daydream_phase`` steps and
    no ``recommendation-verdicts.json`` on disk. Verify lives inside the
    fix-accept branch; a declined run skips it (and its cost).
    """
    from daydream.agent import get_non_interactive, reset_state
    from daydream.config import REVIEW_OUTPUT_FILE
    from daydream.runner import run

    _silence_gate_noise(monkeypatch)
    # The PR post runs before the gate; stub the non-idempotent GitHub write.
    mute_side_effects()
    _install_stub_backend(monkeypatch, multi_stack_target)

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

    traj = tmp_path / "trajectory.json"
    reset_state()
    exit_code = -1
    try:
        assert get_non_interactive() is False
        exit_code = await run(make_config(multi_stack_target, trajectory_path=traj))
        assert get_non_interactive() is True
    finally:
        reset_state()

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
    assert "verify" not in phases, (
        f"verify phase ran despite the gate declining; phases: {phases!r}"
    )
    verdicts_file = multi_stack_target / ".daydream" / "deep" / "recommendation-verdicts.json"
    assert not verdicts_file.exists(), (
        f"recommendation-verdicts.json exists despite declined gate -- verify must "
        f"not run when fixes are not applied; found {verdicts_file}"
    )


async def test_apply_fixes_gate_eof_declines_cleanly_no_crash(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
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
    from daydream.runner import run

    _silence_gate_noise(monkeypatch)
    mute_side_effects()
    _install_stub_backend(monkeypatch, multi_stack_target)

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
    exit_code = -1
    try:
        assert get_non_interactive() is False
        # If the gate did not catch EOFError, this await would raise.
        exit_code = await run(make_config(multi_stack_target, non_interactive=False))
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
    assert set(per_stack_models) == {"claude-sonnet-5"}

    # (b) Merge backend created with an Opus model id.
    assert models_where(lambda pl: "cross-stack merge agent" in pl) == ["claude-opus-5"]

    # (c) Opus arbiter created exactly once when a high-severity record exists.
    assert models_where(lambda pl: "you are the arbiter" in pl) == ["claude-opus-5"]

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
    assert per_stack_models == {"claude-sonnet-5"}


# Issue #232: precision-mode suppression pass over borderline uncontested findings.
#
# Every test drives runner.run through run_deep with a mock scripting one HIGH
# finding (python @ api.py:1, an arbiter target) and one borderline low-severity
# finding (react @ App.tsx:1, a suppression target) at NON-colliding locations so
# they stay uncontested. App.tsx is the sole discriminator for the borderline
# finding: its presence/absence in the canonical merged-items.json is the
# observable outcome.
#
# The borderline finding is low-severity but MEDIUM confidence, NOT LOW: the
# always-on evidence gate (#227) drops every LOW-confidence finding at merge
# regardless of precision, so a LOW-confidence finding could never observe the
# suppression pass. Per issue #232, suppression trims by *materiality* (severity),
# not evidence -- a low-severity MEDIUM-confidence finding survives the gate and
# is dropped ONLY when suppression declines it.
_PRECISION_STACKS: dict[str, dict[str, Any]] = {
    "python": {"severity": "high", "confidence": "HIGH", "file": "api.py", "line": 1},
    "react": {"severity": "low", "confidence": "MEDIUM", "file": "App.tsx", "line": 1},
}


def _merged_item_files(target: Path) -> list[str]:
    """Return the ``file`` of every item in the canonical merged-items.json."""
    items_file = target / ".daydream" / "deep" / "merged-items.json"
    items = json.loads(items_file.read_text())["items"]
    return [it.get("file") for it in items]


def _merged_item_descriptions(target: Path) -> list[str]:
    """Return the ``description`` of every item in the canonical merged-items.json."""
    items_file = target / ".daydream" / "deep" / "merged-items.json"
    items = json.loads(items_file.read_text())["items"]
    return [it.get("description", "") for it in items]


# A borderline LOW-severity sibling sharing one (file, line) with a HIGH finding
# from the SAME stack (py_module.py:7). Single stack -> uncontested, so only the
# HIGH one is an arbiter target. Other stacks stay on api.py:1 (no severity) so
# nothing there is high or contested. If suppression exclusion were keyed by
# (file, line), the HIGH sibling's location would wrongly exclude the LOW sibling
# too, silently skipping it (#232 review, Comment 2).
_SUPPRESSION_COLLISION_STACKS: dict[str, dict[str, Any]] = {
    "python": {
        "severity": "high",
        "confidence": "HIGH",
        "file": "py_module.py",
        "line": 7,
        "description": "the HIGH finding",
        "extra": {
            "severity": "low",
            "confidence": "MEDIUM",
            "file": "py_module.py",
            "line": 7,
            "description": "borderline sibling sharing the HIGH location",
        },
    },
}


async def test_precision_suppresses_low_sibling_sharing_high_finding_location(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#232 Comment 2: a borderline LOW finding sharing a (file, line) with an
    arbitrated HIGH finding from the same stack must STILL be suppression-reviewed
    and dropped. A (file, line)-keyed exclusion excluded both siblings, letting the
    LOW one survive unreviewed; the per-record-identity key fixes it."""
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


async def test_precision_on_drops_unconfirmed_low_finding(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#232 Test A (precision ON, no evidence): the borderline LOW finding is
    dropped when the suppression reviewer returns keep=false; the HIGH finding
    survives, and the suppression pass makes exactly one batched Sonnet call."""
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(
        monkeypatch,
        multi_stack_target,
        merge_echo_records=True,
        parse_by_stack=_PRECISION_STACKS,
        suppression_keep=False,
    )

    exit_code = await _run_deep(multi_stack_target, precision_mode=True)
    assert exit_code == 0

    files = _merged_item_files(multi_stack_target)
    # The borderline LOW finding (App.tsx) is suppressed; the HIGH one (api.py) stays.
    assert "App.tsx" not in files, f"unconfirmed LOW finding must be dropped:\n{files}"
    assert "api.py" in files, f"HIGH finding must survive suppression:\n{files}"

    # Exactly one batched suppression call, resolved via the cheap `suppression`
    # phase key (Sonnet), NOT per-finding Opus.
    sup_calls = [c for c in calls if "you are the suppression reviewer" in c["prompt"].lower()]
    assert len(sup_calls) == 1, f"suppression must make exactly one batched call, got {len(sup_calls)}"
    assert sup_calls[0]["model"] == "claude-sonnet-5"

    # The arbiter still ran on the HIGH finding (fail-open, unchanged by #232).
    arbiter_calls = [c for c in calls if "you are the arbiter" in c["prompt"].lower()]
    assert len(arbiter_calls) == 1
    assert arbiter_calls[0]["model"] == "claude-opus-5"


async def test_precision_off_keeps_low_finding(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#232 Test B (precision OFF, product default): identical inputs, but the
    borderline LOW finding survives to merge and NO suppression pass runs.

    Regression guard for the "don't drop possibly-real findings" non-goal."""
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(
        monkeypatch,
        multi_stack_target,
        merge_echo_records=True,
        parse_by_stack=_PRECISION_STACKS,
        suppression_keep=False,  # would drop if it ran -- it must NOT run
    )

    exit_code = await _run_deep(multi_stack_target, precision_mode=False)
    assert exit_code == 0

    files = _merged_item_files(multi_stack_target)
    assert "App.tsx" in files, f"LOW finding must survive when precision is OFF:\n{files}"
    assert "api.py" in files

    sup_calls = [c for c in calls if "you are the suppression reviewer" in c["prompt"].lower()]
    assert sup_calls == [], "suppression must never run when precision is OFF"


async def test_precision_on_keeps_low_finding_with_evidence(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#232 Test C (precision ON, evidence): a borderline LOW finding the
    suppression reviewer CONFIRMS (keep=true with a rationale) is retained."""
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(
        monkeypatch,
        multi_stack_target,
        merge_echo_records=True,
        parse_by_stack=_PRECISION_STACKS,
        suppression_keep=True,
    )

    exit_code = await _run_deep(multi_stack_target, precision_mode=True)
    assert exit_code == 0

    # The suppression pass ran (borderline finding exists) ...
    sup_calls = [c for c in calls if "you are the suppression reviewer" in c["prompt"].lower()]
    assert len(sup_calls) == 1

    # ... but confirmed the finding, so App.tsx is retained.
    files = _merged_item_files(multi_stack_target)
    assert "App.tsx" in files, f"a confirmed borderline finding must be kept:\n{files}"
    assert "api.py" in files


def _prime_merge_resume_records(target: Path, *, python_severity: str | None) -> Path:
    """Write the per-stack records a `--start-at merge` resume needs on disk.

    Every detected stack (python, react, generic, structure) must have a records
    file or be a recorded failure, else the resume guard returns 1. The python
    record optionally carries ``python_severity`` to drive arbiter selection.
    """
    py_record = _record(description="py issue", evidence="api.py:1")
    if python_severity is not None:
        py_record |= {"severity": python_severity, "confidence": "HIGH", "rationale": "stub"}
    return _prime_merge_resume(
        target,
        python=[py_record],
        react=[_record(description="tsx issue", file="App.tsx", evidence="App.tsx:1")],
        generic=[_record(description="docs issue", file="README.md", evidence="README.md:1")],
        structure=[_record(description="structural issue", evidence="api.py:1")],
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
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
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


def _scan_trajectory_extra(run_root: Path, traj: Path, key: str) -> list[str]:
    """Collect ``step["extra"][key]`` across every trajectory JSON written for a run.

    An aborted/forked turn writes sibling trajectory files under the per-run dir, so
    scan all ``*.json`` beneath ``run_root`` plus the top-level ``traj`` path. Non-dict
    or unparseable files are skipped. Returns only truthy values, in discovery order.
    """
    values: list[str] = []
    for tf in list(run_root.rglob("*.json")) + ([traj] if traj.exists() else []):
        try:
            payload = json.loads(tf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        for step in payload.get("steps", []):
            value = (step.get("extra") or {}).get(key)
            if value:
                values.append(value)
    return values


async def test_run_terminates_under_tool_call_budget(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC#6a real-path: a runaway fix turn is capped by the tool-call budget.

    The stub's fix branch yields an unbounded burst of ToolStartEvents and never
    a ResultEvent (the 1.5-5h time-tail #169 targets). Driven through the real
    ``runner.run`` -> deep orchestrator -> ``phase_fix_parallel`` -> ``run_agent``
    path with a small ``tool_call_budget``, the loop must break, ``run`` must
    return an int exit code (no hang/exception), and the aborted turn's ATIF step
    must carry ``extra["stop_reason"]``.

    Discriminating: without the in-loop tool-call budget in ``run_agent`` the
    stub stream never completes, so ``run`` never returns -- the ``fail_after``
    timeout turns that regression into a failure instead of an infinite hang.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    # Patch the binding actually read at the fix call site (Task 5 imported the
    # constant INTO daydream.phases); patching only daydream.config would not take
    # effect because phases already resolved the name at import time.
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_fix = True
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(
            make_config(
                multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"
            )
        )

    assert isinstance(exit_code, int)

    # The aborted fix turn runs inside a sibling (forked) trajectory under the
    # per-run dir, so scan every trajectory JSON written for this run for a step
    # whose extra carries a stop_reason -- the observable proof the turn aborted.
    run_root = multi_stack_target / ".daydream"
    stop_reasons = _scan_trajectory_extra(run_root, traj, "stop_reason")

    assert stop_reasons, "no trajectory step recorded extra['stop_reason']; budget did not trip"
    assert "tool_call_budget_exceeded" in stop_reasons


async def test_run_terminates_under_wall_budget(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """#169 real-path: a runaway fix turn is capped by the wall-clock budget.

    The stub's fix branch yields a slow, unbounded burst of ToolStartEvents (a real
    per-event sleep, never a ResultEvent) -- the 1.5-5h time-tail #169 targets.
    Driven through the real ``runner.run`` -> deep orchestrator -> ``phase_fix_parallel``
    -> ``run_agent`` path with a tiny ``DEFAULT_WALL_BUDGET_S``, the wall scope must
    cancel the turn, ``run`` must return an int exit code, and the aborted turn's ATIF
    step must carry ``extra["stop_reason"] == "wall_budget_exceeded"``.

    Discriminating: the wall budget only engages because ``phase_fix_parallel`` now
    passes ``wall_budget_s=DEFAULT_WALL_BUDGET_S`` at the fix call site. Unwire that
    (the pre-fix state, where the constant was defined but never passed) and the wall
    scope is ``nullcontext()`` in production -- the tool-call budget trips instead, so
    ``stop_reason`` is ``tool_call_budget_exceeded`` and this assertion fails.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    # Patch the binding read at the fix call site (phases imported the constant by
    # name at import time, so patching daydream.config alone would not take effect).
    # 0.3s wall trips after ~6 slow events, well before the 50-call tool-call budget.
    monkeypatch.setattr("daydream.phases.DEFAULT_WALL_BUDGET_S", 0.3)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_fix = True
    stub.runaway_fix_sleep_s = 0.05
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(
            make_config(
                multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"
            )
        )

    assert isinstance(exit_code, int)

    run_root = multi_stack_target / ".daydream"
    stop_reasons = _scan_trajectory_extra(run_root, traj, "stop_reason")

    assert stop_reasons, "no trajectory step recorded extra['stop_reason']; budget did not trip"
    assert "wall_budget_exceeded" in stop_reasons


def _scan_phase_events(run_root: Path, traj: Path, event: str) -> list[dict[str, Any]]:
    """Collect ``extra['phase_events']`` entries of a given ``event`` across all run JSONs.

    The group-budget marker lives in ``Trajectory.extra['phase_events']`` (not
    step ``extra``), and the fix work runs in a forked sibling trajectory, so scan
    every ``*.json`` beneath ``run_root`` plus the top-level ``traj``.
    """
    found: list[dict[str, Any]] = []
    for tf in list(run_root.rglob("*.json")) + ([traj] if traj.exists() else []):
        try:
            payload = json.loads(tf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        for ev in (payload.get("extra") or {}).get("phase_events", []):
            if isinstance(ev, dict) and ev.get("event") == event:
                found.append(ev)
    return found


def _batched_group_size(stub: "_StubBackend", file_basename: str) -> int:
    """Return N from the failed batched ``Fix these N issues in <file>`` fix turn.

    The deep pipeline may inject an extra structural finding into a file group, so
    the group size is derived from the batched prompt rather than hard-coded.
    """
    import re as _re

    for c in stub.calls:
        m = _re.search(r"^Fix these (\d+) issues in (.+):$", c["prompt"], _re.M)
        if m is not None and Path(m.group(2)).name == file_basename:
            return int(m.group(1))
    raise AssertionError(f"no batched fix turn found for {file_basename}")


async def test_run_caps_runaway_file_group_serial_fixes(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """#201 real-path: a runaway file group is capped by the serial-item budget.

    Six findings target api.py; the batched api.py fix turn fails, forcing the
    per-finding fallback loop -- the exact #186 shape where 9 serial fix calls on
    one file silently dominated a 62-min run. Driven through the real
    ``runner.run`` -> deep orchestrator -> ``phase_fix_parallel`` path with the
    group serial-item ceiling lowered to 3, only 3 of the 6 fallback fixes run;
    the remaining 3 are skipped, recorded in ``failures`` (surfaced as the
    fix-failures artifact), and a ``file_group_budget_exceeded`` trajectory event
    is emitted naming the file, reason, and processed/skipped counts.

    Discriminating: without the group budget, the fallback loop fixes all 6
    api.py findings (six "fix this issue" turns) and no budget event exists --
    both assertions fail.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    # Lower the group serial-item ceiling at the binding the orchestrator resolves
    # (it imported the constant by name, so patching daydream.config alone is inert).
    monkeypatch.setattr("daydream.deep.orchestrator.DEFAULT_GROUP_MAX_SERIAL_ITEMS", 3)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(i, "api.py", "high") for i in range(1, 7)] + [
        _merge_item(7, "App.tsx", "high")
    ]
    stub.fail_batched_fix_file = "api.py"  # force the per-finding fallback for api.py
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(
            make_config(
                multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"
            )
        )
    assert isinstance(exit_code, int)

    # The failed batched turn names the api.py group size (the pipeline may add a
    # structural finding, so derive N rather than hard-coding it).
    group_size = _batched_group_size(stub, "api.py")
    assert group_size >= 6

    # Only 3 fallback fixes ran (the ceiling) before the group budget tripped --
    # NOT the full group, which is the runaway #186 behaviour the guard bounds.
    api_singles = [
        c
        for c in stub.calls
        if c["prompt"].lower().startswith("fix this issue") and "api.py" in c["prompt"]
    ]
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
    """#201 real-path: under a high ceiling the group budget is purely additive.

    Same six-finding api.py fallback shape, but the serial-item ceiling stays well
    above the group size: all six fallback fixes run, no budget event is emitted,
    and no budget failure is recorded. Proves the guard changes nothing when a
    group stays within budget (spec AC#5).
    """
    from daydream.runner import run

    _silence(monkeypatch)
    monkeypatch.setattr("daydream.deep.orchestrator.DEFAULT_GROUP_MAX_SERIAL_ITEMS", 20)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(i, "api.py", "high") for i in range(1, 7)] + [
        _merge_item(7, "App.tsx", "high")
    ]
    stub.fail_batched_fix_file = "api.py"
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(
            make_config(
                multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"
            )
        )
    assert isinstance(exit_code, int)

    group_size = _batched_group_size(stub, "api.py")
    api_singles = [
        c
        for c in stub.calls
        if c["prompt"].lower().startswith("fix this issue") and "api.py" in c["prompt"]
    ]
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
    """#201 real-path: a batched turn's OWN wall trip carries into the fallback.

    The batched api.py turn burns real wall until run_agent's per-invocation wall
    scope cancels it and returns a ``budget_reason`` -- so ``phase_fix_batched``
    raises through the *real* budget path (NOT a synchronous stub raise like
    ``fail_batched_fix_file``) and ``phase_fix_parallel`` falls back to per-finding
    fixes. Because the SAME ``FileGroupBudget`` is reused across the batched and
    fallback stages (a spec invariant), the ~1.8s the batched turn already spent
    is on the group's wall clock when the fallback's first ``budget.check()`` runs.
    With the group wall ceiling at 1.0s -- below the batched turn's scaled 1.8s
    per-invocation budget -- that first check trips immediately, so ZERO fallback
    fixes run, the whole group is recorded as a wall budget failure, and the
    trajectory event reports 0 processed / all skipped.

    Discriminating: if the batched turn's wall did NOT carry into the fallback
    (e.g. a fresh per-call budget), the fallback would see a ~0s clock and fix all
    six api.py findings. Asserting zero ``fix this issue`` api.py turns + a
    ``group_wall_budget_exceeded`` reason proves the carryover. The batched turn's
    own ``wall_budget_exceeded`` stop_reason proves it failed via the real budget
    path rather than a stub raise.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    # Tiny per-invocation wall so the batched turn (scaled to N * 0.3s) trips after
    # ~1.8s of real wall; patch the binding read at the fix call site.
    monkeypatch.setattr("daydream.phases.DEFAULT_WALL_BUDGET_S", 0.3)
    # Group wall ceiling below the batched turn's scaled per-invocation budget, so
    # the wall the batched turn already burned guarantees the fallback's first
    # check trips (deterministic: 1.0 < 0.3 * 6).
    monkeypatch.setattr("daydream.deep.orchestrator.DEFAULT_GROUP_MAX_WALL_S", 1.0)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_items = [_merge_item(i, "api.py", "high") for i in range(1, 7)] + [
        _merge_item(7, "App.tsx", "high")
    ]
    stub.runaway_batched_fix_file = "api.py"  # batched api.py turn trips its own wall budget
    stub.runaway_batched_sleep_s = 0.05
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(
            make_config(
                multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"
            )
        )
    assert isinstance(exit_code, int)

    # The batched api.py turn actually ran (and is where the wall was burned).
    group_size = _batched_group_size(stub, "api.py")
    assert group_size >= 6

    # ZERO fallback fixes ran: the batched turn's carried-over wall tripped the
    # group budget on the fallback's very first check -- the #186 runaway is fully
    # bounded, not merely trimmed.
    api_singles = [
        c
        for c in stub.calls
        if c["prompt"].lower().startswith("fix this issue") and "api.py" in c["prompt"]
    ]
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
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """#202 real-path: N findings on ONE file collapse to a single FIX run_agent turn.

    Driven through the real ``runner.run`` -> deep orchestrator ->
    ``phase_fix_parallel`` -> ``phase_fix_batched`` -> ``run_agent`` path. Several
    findings target api.py and one targets App.tsx, so the fix stage must issue
    exactly TWO fix turns (one batched per file-group), never one-per-finding.
    The batched api.py turn carries a single "Fix these N issues" prompt naming
    every same-file finding, and still lands the per-file sentinel.

    Discriminating: a per-finding loop (the pre-#202 state) issues one fix turn
    PER finding, so the fix-prompt count balloons past two and no single batched
    "Fix these N issues" prompt exists -- both assertions fail.
    """
    import re

    from daydream.runner import run

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

    exit_code = await run(
        make_config(
            multi_stack_target, assume="yes", output_mode="loop", non_interactive=False
        )
    )

    assert exit_code == 0

    fix_prompts = [
        c["prompt"]
        for c in stub.calls
        if c["prompt"].lower().startswith(("fix this issue", "fix these"))
    ]
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
    # The batched turn still lands its per-file sentinel (observable apply).
    assert (multi_stack_target / ".fixed-api_py").exists()
    assert (multi_stack_target / ".fixed-App_tsx").exists()


async def test_environmental_failure_aborts_heal_loop(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC#6b real-path: an environmental test failure aborts heal without a fix turn.

    Drives the REAL ``phase_test_and_heal`` through ``runner.run`` -> deep
    orchestrator. The stub's test-suite branch emits a Postgres-unreachable
    signature, so ``detect_test_success`` is False AND ``is_environmental_failure``
    is True. The orchestrator's short-circuit (Task 6) must return failure BEFORE
    re-entering a fix turn -- so the heal-fix sentinel
    ``.daydream-heal-fix-applied`` must NEVER be written.

    ``assume="yes"`` opts into a SINGLE bounded auto fix-and-retry, so this test
    is DISCRIMINATING without hanging: remove the environmental short-circuit and
    the heal loop runs exactly one fix turn (then aborts), writing the sentinel.
    With the short-circuit in place the sentinel is absent, the run returns an int
    failure exit code, and a TEST-phase trajectory step is recorded with no fix.
    """
    _silence(monkeypatch, prompts=False)
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")

    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.environmental_test_failure = True  # every test run reports infra-down

    # phase_test_and_heal stays REAL so the environmental short-circuit runs.
    mute_side_effects(heal=False)

    from daydream.runner import run

    traj = tmp_path / "trajectory.json"
    exit_code = await run(
        make_config(multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop")
    )

    # Environmental failure is not healable -> run reports failure, not success.
    assert isinstance(exit_code, int)
    assert exit_code != 0, "environmental failure must surface as a non-zero exit, not be healed"

    # The observable proof: the heal loop NEVER re-entered a fix turn, so the
    # heal-fix sentinel was never written. (Discriminating: without the
    # short-circuit, choice "2" would write this file.)
    heal_sentinel = multi_stack_target / ".daydream-heal-fix-applied"
    assert not heal_sentinel.exists(), (
        "heal-fix sentinel exists -- the environmental short-circuit did not "
        "abort before re-entering a fix turn"
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


def _install_accept_gate_pipeline(
    monkeypatch: pytest.MonkeyPatch, target: Path, mute: Mute
) -> _StubBackend:
    """Patch the deep pipeline for a fix-gate-ACCEPT run.

    Bundles the setup every accept-the-gate test shares: pin the interactive
    stdin/CI axis so a forced accept is honoured, silence the deep UI noise
    (including the recommendation verification summary), force every
    ``prompt_user`` seam to ``"y"`` (belt-and-suspenders alongside
    ``assume="yes"``, which short-circuits the gate before any prompt runs),
    and stub the non-idempotent PR-post / test / commit steps. ``phase_fix``
    stays REAL. Returns the stub backend.
    """
    _force_interactive(monkeypatch)
    _silence(monkeypatch, prompts=False)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")
    mute()

    return _install_stub_backend(monkeypatch, target)


async def test_alternatives_skipped_for_trivial_diff(
    feature_branch_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC4 real-path (negative): a 1-file diff skips the alternatives (wonder) phase.

    ``select_tier(count_changed_files(diff)) == "skip"`` for <=1 changed file, so
    the alternatives phase must NOT run: no ``alternatives`` value among the run's
    trajectory ``daydream_phase`` steps, no ``wonder`` prompt dispatched through
    the stub, and ``alternatives.json`` written as ``[]`` (downstream per-stack
    and merge consumers still find the file). Intent still runs unconditionally.
    """
    from daydream.runner import run

    target = feature_branch_repo

    # Accept the fix gate so the full pipeline runs; the shared helper pins
    # interactivity, silences deep UI noise, and stubs the non-idempotent
    # PR-post / test / commit steps.
    stub = _install_accept_gate_pipeline(monkeypatch, target, mute_side_effects)

    traj = tmp_path / "trajectory.json"
    exit_code = await run(
        make_config(
            target, trajectory_path=traj, assume="yes", output_mode="loop", non_interactive=False
        )
    )
    assert exit_code == 0

    run_root = target / ".daydream"
    phases = _scan_trajectory_extra(run_root, traj, "daydream_phase")

    # alternatives phase never ran.
    assert "alternatives" not in phases, (
        f"alternatives phase ran for a trivial (1-file) diff; phases: {phases!r}"
    )
    # No wonder/alternatives prompt was dispatched (discriminator per _StubBackend.execute).
    alt_calls = [
        c
        for c in stub.calls
        if "would you have done this differently" in c["prompt"].lower()
        or "evaluate the implementation" in c["prompt"].lower()
    ]
    assert not alt_calls, (
        f"alternatives wonder prompt dispatched for a trivial diff: {len(alt_calls)} call(s)"
    )
    # alternatives.json still written as [] so downstream consumers find the file.
    alts_json = json.loads((target / ".daydream" / "deep" / "alternatives.json").read_text())
    assert alts_json == [], f"expected empty alternatives.json on skip, got {alts_json!r}"
    # intent still ran -- it is never gated by tier.
    assert "intent" in phases, (
        f"intent phase did not run; it must not be gated by diff tier; phases: {phases!r}"
    )


async def test_alternatives_runs_for_multi_file_diff(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC4 real-path (positive): a >=2-file diff runs the alternatives phase.

    ``multi_stack_target`` carries a 3-file diff (api.py, App.tsx, README.md),
    so ``select_tier(count_changed_files(diff)) == "single"`` (not ``"skip"``).
    The alternatives phase MUST run: an ``alternatives`` value among the run's
    trajectory ``daydream_phase`` steps and a ``wonder`` prompt dispatched.
    """
    from daydream.runner import run

    # Accept the fix gate; the shared helper pins interactivity, silences deep
    # UI noise, and stubs the non-idempotent PR-post / test / commit steps.
    stub = _install_accept_gate_pipeline(monkeypatch, multi_stack_target, mute_side_effects)

    traj = tmp_path / "trajectory.json"
    exit_code = await run(
        make_config(
            multi_stack_target,
            trajectory_path=traj,
            assume="yes",
            output_mode="loop",
            non_interactive=False,
        )
    )
    assert exit_code == 0

    run_root = multi_stack_target / ".daydream"
    phases = _scan_trajectory_extra(run_root, traj, "daydream_phase")

    # alternatives phase ran.
    assert "alternatives" in phases, (
        f"alternatives phase did not run for a 3-file diff; phases: {phases!r}"
    )
    # A wonder/alternatives prompt was dispatched.
    alt_calls = [
        c
        for c in stub.calls
        if "would you have done this differently" in c["prompt"].lower()
        or "evaluate the implementation" in c["prompt"].lower()
    ]
    assert alt_calls, "alternatives wonder prompt was not dispatched for a 3-file diff"

# =============================================================================
# Issue #172 — Perf: tiny-diff short-circuit + read-once diff hunks
# =============================================================================


def test_shallow_fanout_threshold_precedence() -> None:
    """AC7: SHALLOW_FANOUT_THRESHOLD honors CLI (RunConfig) > config file > default.

    Mirrors the `_resolve_backend` precedence pattern; uses `is not None` so a
    config value of ``0`` (disable the short-circuit entirely) is honored
    rather than treated as falsy.
    """
    from daydream.config_file import DaydreamFileConfig
    from daydream.deep.orchestrator import (
        DEFAULT_SHALLOW_FANOUT_THRESHOLD,
        _shallow_fanout_threshold,
    )
    from daydream.runner import RunConfig

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
    assert (
        _shallow_fanout_threshold(
            RunConfig(file_config=fc, shallow_fanout_threshold=5)
        )
        == 5
    )


def test_collapse_stacks_for_tiny_diff_single_language() -> None:
    """AC1 (unit): 1-file single-language diff collapses to a strictly smaller agent count.

    Today `detect_stacks(["api.py"])` yields 2 stacks (python + structure) →
    `total_agent_count(2) == 8`. After collapse:
      - lever 1 (collapse language fan-out) is a no-op (only one language stack)
      - lever 2 (skip merge+arbiter) drops the count to ``_single_stack_agent_count``
        which must be strictly less than 8.
    """
    from daydream.deep.detection import detect_stacks
    from daydream.deep.orchestrator import (
        _collapse_stacks_for_tiny_diff,
        _single_stack_agent_count,
        total_agent_count,
    )

    stacks = detect_stacks(["api.py"])
    assert len(stacks) == 2  # python + structure (baseline sanity check)
    baseline_count = total_agent_count(len(stacks))  # 8

    collapsed, single_stack_mode = _collapse_stacks_for_tiny_diff(
        stacks, ["api.py"], threshold=2
    )
    assert single_stack_mode is True
    # Structure stack stays as its own assignment (AC6 lens taxonomy preserved).
    stack_names = [s.stack_name for s in collapsed]
    assert "structure" in stack_names
    # The collapsed run uses the single-stack agent count, which MUST be strictly
    # less than the baseline 8 (lever 2: skip merge+arbiter).
    assert _single_stack_agent_count(len(collapsed)) < baseline_count


def test_collapse_stacks_for_tiny_diff_two_languages() -> None:
    """AC1 (unit): 2-file two-language diff collapses the language fan-out.

    Today `detect_stacks(["api.py", "App.tsx"])` yields 3 stacks (python + react
    + structure) → ``total_agent_count(3) == 12``. After collapse the two
    language stacks merge into one combined (generic-fallback) stack, so the
    surviving stack list is `[combined, structure]` (≤2). The single-stack agent
    count for ≤2 stacks is strictly less than 12.
    """
    from daydream.deep.detection import detect_stacks
    from daydream.deep.orchestrator import (
        _collapse_stacks_for_tiny_diff,
        _single_stack_agent_count,
        total_agent_count,
    )

    stacks = detect_stacks(["api.py", "App.tsx"])
    assert len(stacks) == 3  # python + react + structure (baseline)
    baseline_count = total_agent_count(len(stacks))  # 12

    collapsed, single_stack_mode = _collapse_stacks_for_tiny_diff(
        stacks, ["api.py", "App.tsx"], threshold=2
    )
    assert single_stack_mode is True
    # Collapse merged the two language stacks into one combined assignment.
    non_structural = [s for s in collapsed if s.stack_name != "structure"]
    assert len(non_structural) == 1
    combined = non_structural[0]
    # Combined assignment carries both files and uses the generic-fallback skill
    # (a single agent cannot invoke two per-language Beagle skills).
    assert set(combined.files) == {"api.py", "App.tsx"}
    assert combined.skill_invocation is None
    # Single-stack agent count is strictly less than 12.
    assert _single_stack_agent_count(len(collapsed)) < baseline_count


def test_collapse_stacks_for_tiny_diff_code_plus_docs_preserves_language_skill() -> None:
    """Skill-preservation (unit): code+docs tiny diff keeps the per-language skill.

    A code+docs/config tiny diff routes via ``detect_stacks`` to exactly one real
    language stack plus the ``generic`` bucket (e.g. ``api.py`` + ``README.md`` →
    ``python`` + ``generic``). That is a single-language diff, so the collapse
    must absorb the generic files into the language stack and keep its
    per-language Beagle skill -- NOT downgrade it to the generic fallback
    (the skill-preservation goal stated in ``_collapse_stacks_for_tiny_diff``'s
    docstring). Only ≥2 *real* language stacks fall back to generic.
    """
    from daydream.deep.detection import detect_stacks
    from daydream.deep.orchestrator import (
        _collapse_stacks_for_tiny_diff,
        _single_stack_agent_count,
        total_agent_count,
    )

    files = ["api.py", "README.md"]
    stacks = detect_stacks(files)
    # Baseline sanity: python (real language) + generic + structure.
    assert [s.stack_name for s in stacks] == ["python", "generic", "structure"]
    baseline_count = total_agent_count(len(stacks))  # 12

    collapsed, single_stack_mode = _collapse_stacks_for_tiny_diff(stacks, files, threshold=2)
    assert single_stack_mode is True
    non_structural = [s for s in collapsed if s.stack_name != "structure"]
    assert len(non_structural) == 1
    combined = non_structural[0]
    # The real-language skill survives (NOT downgraded to generic fallback).
    assert combined.stack_name == "python"
    assert combined.skill_invocation == "beagle-python:review-python"
    # The docs file is absorbed into the language stack.
    assert set(combined.files) == {"api.py", "README.md"}
    assert _single_stack_agent_count(len(collapsed)) < baseline_count


def test_collapse_stacks_for_tiny_diff_disabled_at_threshold_zero() -> None:
    """AC7 edge: threshold=0 disables the short-circuit (no collapse happens)."""
    from daydream.deep.detection import detect_stacks
    from daydream.deep.orchestrator import _collapse_stacks_for_tiny_diff

    stacks = detect_stacks(["api.py", "App.tsx"])
    collapsed, single_stack_mode = _collapse_stacks_for_tiny_diff(
        stacks, ["api.py", "App.tsx"], threshold=0
    )
    assert single_stack_mode is False
    assert collapsed == stacks  # unchanged


def _count_review_prompts(calls: list[dict[str, Any]]) -> int:
    """Count per-stack + structural review prompts in a captured call list.

    Discriminators mirror the production prompt builders:
      - per-stack / generic-fallback: ``"you are reviewing the <X> stack"``
      - structural: ``"you are the structural reviewer"``
    """
    n = 0
    for c in calls:
        pl = c["prompt"].lower()
        if "you are reviewing the" in pl and "stack" in pl:
            n += 1
        elif "you are the structural reviewer" in pl:
            n += 1
    return n


def _count_merge_prompts(calls: list[dict[str, Any]]) -> int:
    """Count cross-stack merge-agent prompts (discriminator: 'cross-stack merge agent')."""
    return sum(1 for c in calls if "cross-stack merge agent" in c["prompt"].lower())


async def test_ac2_tiny_diff_collapses_fanout_and_skips_merge(
    tiny_diff_target: Path,
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC2 (real-path): a ≤2-file two-language diff collapses the fan-out.

    Drives ``runner.run`` through the full deep pipeline on a 2-file repo
    (``api.py`` + ``App.tsx``). The tiny-diff short-circuit MUST:
      - collapse the two language stacks (python + react) into one combined
        generic-fallback review (≤2 review agents total, vs 4 for multi_stack);
      - skip the merge agent + arbiter;
      - still write the canonical ``merged-items.json`` (AC6 unchanged schema).

    The proof is directional: the same harness run against ``multi_stack_target``
    yields strictly more review prompts and a recorded merge-agent prompt, while
    the tiny-diff run yields strictly fewer and no merge prompt.
    """
    from daydream.runner import run

    # Run BOTH repos through the identical harness so the count comparison is a
    # paired observation, not an absolute threshold.
    async def _drive(target: Path) -> list[dict[str, Any]]:
        # Fresh shared-call list per run (the factory binds it via closure).
        _silence(monkeypatch)
        shared_calls = _install_model_capturing_stubs(monkeypatch, target)
        # Stub the post-merge side effects so the run terminates cleanly.
        mute_side_effects()
        rc = await run(make_config(target))
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


async def test_ac5_per_stack_prompt_inlines_diff_hunks(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """AC5 (real-path): per-stack review prompts contain inlined diff hunks and
    NO ``Read it directly`` / diff_path instruction.

    Drives ``runner.run`` through the deep pipeline on a 2-file fixture and
    inspects the recorded prompts on the stub's shared call list. The per-stack
    review prompt MUST inline the relevant hunks and MUST NOT instruct the agent
    to ``Read it directly`` — proving ``diff.patch`` is read 0 times
    (transitively: no Read instruction + hunks present inline).

    Grounding note: ``_StubBackend`` does not model tool-call execution (its
    review branch writes the review file directly without emitting Read events),
    so "diff.patch is Read 0 times" is proven by the absence of the Read
    instruction in the recorded prompt, not by counting Read events.
    """
    from daydream.runner import run

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
    expected_api_hunk = (
        "@@ -1,2 +1,2 @@\n"
        " def hello():\n"
        "-    return 'world'\n"
        "+    return 'universe'\n"
    )
    assert expected_api_hunk in prompt, "expected complete api.py diff hunk in per-stack prompt"
    # The Read instruction is absent (the agent is never told to Read diff.patch).
    assert "Read it directly" not in prompt
    # And diff_path is not embedded as an instruction (it remains a required
    # param of the builder but is not surfaced when hunks are inlined).
    diff_path_str = str(tiny_diff_target / ".daydream" / "diff.patch")
    assert diff_path_str not in prompt

    # Discriminating check: the STRUCTURAL prompt still carries the pointer
    # (Fix B does NOT inline the structural / arbiter prompts).
    structural_prompts = [
        c["prompt"] for c in shared_calls if "you are the structural reviewer" in c["prompt"].lower()
    ]
    assert structural_prompts, "expected a structural review prompt"
    assert "Read it directly" in structural_prompts[0]
    assert diff_path_str in structural_prompts[0]


async def test_ac6_single_stack_merged_items_carry_structural_lens(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """AC6: tiny-diff single-stack writer tags structural items ``lens="structural"``.

    The single-stack host writer (``_write_single_stack_merged_items``) must
    replicate ``phase_cross_stack_merge``'s structural tagging exactly so the
    verifier's matched/unmatched accounting (orchestrator.py:968/973) does not
    drift. Real-path: drive the tiny-diff flow, parse merged-items.json, and
    assert at least one item carries ``lens="structural"``.
    """
    from daydream.runner import run

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
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """Issue #172 risk: ``--start-at fix`` resume on a tiny diff works.

    ``single_stack_mode`` is recomputed at the top of ``run_deep`` from
    ``changed_files``, so a resume re-enters the same bypass branch. The merge
    block is skipped (``config.start_at == "fix"``), and the fix gate reads the
    surviving ``merged-items.json`` produced by the single-stack writer. This
    test primes the tiny-diff artifacts with a first run, then resumes with
    ``--start-at fix`` and asserts the fix loop reads the JSON and applies.
    """
    from daydream.runner import run

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
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")

    rc = await run(
        make_config(tiny_diff_target, start_at="fix", assume="yes", non_interactive=False)
    )
    assert rc == 0
    fix_prompts = [c for c in stub.calls if c["prompt"].startswith(("Fix this issue", "Fix these"))]
    assert fix_prompts, "fix loop did not run on --start-at fix resume"


async def test_ac_merge_resume_on_tiny_diff(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """Issue #172: ``--start-at merge`` resume on a tiny diff routes to the
    single-stack merge writer, not the multi-stack merge agent.

    Every other ``start_at="merge"`` test drives ``multi_stack_target``. This
    one drives the tiny-diff resume path the finding flagged as untested:
    ``single_stack_mode`` is recomputed True for the 2-file diff at the top of
    ``run_deep``, so the ``config.start_at == "merge"`` branch (which reloads
    the collapsed-stack records from disk and re-partitions the structural
    meta-stack) must route to ``_write_single_stack_merged_items`` rather than
    ``phase_cross_stack_merge``.

    ``phase_cross_stack_merge`` is patched to raise so a regression that fails
    to recompute ``single_stack_mode`` on resume surfaces as a hard failure
    instead of silently routing through the multi-stack merge agent.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tiny_diff_target)
    # Regression guard: the multi-stack merge agent must NOT run in
    # single_stack_mode. If the merge-resume branch misroutes here, raise.
    async def _fail_merge(*_a: Any, **_k: Any) -> None:
        raise AssertionError("phase_cross_stack_merge must not run in single_stack_mode")

    monkeypatch.setattr("daydream.deep.orchestrator.phase_cross_stack_merge", _fail_merge)
    # Stub the post-merge side effects so the run terminates cleanly.
    mute_side_effects()

    # Prime the deep artifacts the merge-resume branch reads from disk. The
    # tiny-diff collapse yields a ``generic`` (collapsed language) stack plus
    # the ``structure`` meta-stack, so records files must match both.
    _prime_merge_resume(
        tiny_diff_target,
        generic=[
            _record(id="gen-1", description="generic per-stack issue", evidence="api.py:1")
        ],
        structure=[
            _record(id="structure-1", description="file-size budget violated", evidence="api.py:1")
        ],
    )

    rc = await run(make_config(tiny_diff_target, start_at="merge"))
    assert rc == 0

    items_file = tiny_diff_target / ".daydream" / "deep" / "merged-items.json"
    assert items_file.is_file(), "single-stack merge resume did not write merged-items.json"
    items = json.loads(items_file.read_text())["items"]
    # ``normalize_items`` reassigns ``id`` to a contiguous sequence, so assert on
    # the preserved ``description`` + ``lens`` instead. The generic record is
    # tagged per-stack and the structural record keeps its structural lens (AC6
    # — lens taxonomy survives the host-written merge).
    assert any(
        i.get("description") == "generic per-stack issue" and i.get("lens") == "per-stack"
        for i in items
    ), f"generic per-stack item missing or mislabeled: {items}"
    assert any(
        i.get("description") == "file-size budget violated" and i.get("lens") == "structural"
        for i in items
    ), f"structural item missing or mislabeled: {items}"


async def test_evidence_gate_drops_speculative_finding(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #227 (AC2/AC3/AC6): the structural evidence gate keeps an evidenced
    finding but drops a speculative one before it reaches merged-items.json.

    Real path through ``runner.run`` (deep default). The merge agent emits one
    grounded finding (``evidence: "src/foo.py:42"``, HIGH) and one speculative
    finding (blank evidence, ``rationale`` claiming "no exploration evidence",
    LOW). Asserts the speculative finding is ABSENT from both the canonical
    ``merged-items.json`` and the rendered ``review-output.md`` while the
    evidenced one survives, and that ``dropped-speculative.json`` records the
    drop.
    """
    from daydream.config import REVIEW_OUTPUT_FILE

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
    assert "Grounded evidenced finding" in descriptions, (
        f"evidenced finding was dropped: {descriptions}"
    )
    assert "Speculative unfounded finding" not in descriptions, (
        f"speculative finding leaked into merged-items.json: {descriptions}"
    )

    report = (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert "Grounded evidenced finding" in report
    assert "Speculative unfounded finding" not in report, (
        "speculative finding leaked into review-output.md"
    )

    dropped = json.loads((deep / "dropped-speculative.json").read_text())
    assert dropped["dropped_count"] >= 1
    assert "Speculative unfounded finding" in json.dumps(dropped["dropped_items"])
    assert 2 in dropped["dropped_ids"]


async def test_evidence_gate_all_speculative_yields_empty(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """Issue #227 (AC5, N=1): a single-stack run whose only findings are all
    speculative writes an EMPTY merged-items.json without crashing and records
    every drop -- never a silent success.

    Drives the ``_write_single_stack_merged_items`` (tiny-diff bypass) gate path
    via a ``--start-at merge`` resume with two primed speculative records: one
    with blank evidence + a "no exploration evidence" rationale, one with LOW
    confidence. Both must be dropped, leaving ``items == []`` and a
    ``dropped-speculative.json`` recording both.
    """
    from daydream.runner import run

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
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """Issue #227 (findings 3/5): a structural (host-tagged, whole-file) finding
    with ``line: 0`` and colon-free evidence SURVIVES the gate -- the structural
    lens is high-conviction by construction and must not be demoted.

    Real path through ``runner.run`` (``--start-at merge`` tiny-diff bypass).
    Primes a structural record whose evidence has no ``path:line`` token and
    whose ``line`` is 0: without the structural carve-out it would fail both
    ``has_file_line`` and ``has_citation`` and be dropped as speculative.
    """
    from daydream.runner import run

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
        i for i in items
        if i.get("lens") == "structural"
        and i.get("description") == "module exceeds 800 LOC budget"
    ]
    assert structural, (
        f"whole-file structural finding was dropped by the gate: {items}"
    )
    assert structural[0].get("line") == 0


async def test_evidence_gate_clears_stale_dropped_sidecar(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """Issue #227 (findings 4/6): a resume that drops 0 findings clears a stale
    ``dropped-speculative.json`` left by a prior run, so the sidecar cannot
    report phantom drops to eval/benchmark/human auditors.

    Real path through ``runner.run`` (``--start-at merge`` tiny-diff bypass).
    Primes a stale sidecar alongside well-evidenced records (0 drops) and
    asserts the sidecar is gone after the run.
    """
    from daydream.runner import run

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
    assert not (deep / "dropped-speculative.json").exists(), (
        "stale dropped-speculative.json survived a 0-drop resume"
    )


async def test_deep_findings_out_emits_artifact_and_stops(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig
) -> None:
    """Real-path: a deep run with ``--findings-out`` writes the PR-pinned findings
    artifact from the canonical merged items and STOPS -- no PR post, no fix.

    Enters through ``runner.run`` -> ``run_deep`` (deep is the default dispatch)
    with a real temp git repo and the scripted ``_StubBackend`` injected through
    the ``create_backend`` seam. The stub drives the full fan-out to a canonical
    ``merged-items.json``; only the backend and the GitHub PR lookup are mocked.

    Observable outcomes:
      (a) the artifact is written to the configured path and pinned to the PR
          (``pr_number``/``head_sha`` match the run's PR + real head SHA);
      (b) exit code is 0;
      (c) NO PR post -- ``post_review_to_pr_from_report`` is replaced with a
          fail-if-called stub, so any post attempt would raise;
      (d) NO fix -- the tracked working tree is byte-identical to its pre-run
          state (real ``git status --porcelain`` empty vs. baseline) and none of
          the stub's ``.fixed-*`` fix sentinels exist.
    """
    from daydream.runner import run

    _silence_gate_noise(monkeypatch)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    _install_stub_backend(monkeypatch, multi_stack_target)

    async def _post_forbidden(target_dir: Path, report_path: Path, *, console: Any) -> None:
        raise AssertionError("--findings-out must not post to the PR")

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _post_forbidden)

    pr = _pin_findings_pr(monkeypatch, multi_stack_target)

    out = multi_stack_target / "findings.json"
    reviewed_sources = ("api.py", "App.tsx", "README.md")
    source_before = {name: (multi_stack_target / name).read_text() for name in reviewed_sources}

    rc = await run(make_config(multi_stack_target, pr_number=7, findings_out=str(out)))

    # (b) exit 0
    assert rc == 0
    # (a) artifact written + PR-pinned
    data = json.loads(out.read_text())
    assert data["pr_number"] == 7
    assert data["head_sha"] == pr.head_sha
    assert data["repo"] == "o/r"
    assert all(re.fullmatch(r"[0-9a-f]{64}", f["fingerprint"]) for f in data["findings"])
    # (d) no fix applied -- the reviewed source is byte-identical and no fix sentinel
    # exists. This is the real "no fix ran" signal: it ignores daydream's own gitignored
    # artifacts (.daydream/, .review-output.md), which the deep pipeline always writes and
    # which are not fixes. (An earlier git-status tree-diff conflated those artifacts with
    # fixes and only passed where a global gitignore happened to mask them.)
    for name in reviewed_sources:
        assert (multi_stack_target / name).read_text() == source_before[name], f"{name} was modified -- a fix ran"
    assert not list(multi_stack_target.glob(".fixed-*")), "fix sentinel present -- a fix ran"
    assert not (multi_stack_target / ".daydream-fix-applied").exists()


async def test_test_verdict_artifact_written_on_passing_suite(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """Real-path: a run whose suite passes leaves ``test-verdict.json`` on disk.

    Drives ``runner.run`` end to end with the scripted ``_StubBackend`` (the
    only mocked seam) and the REAL ``phase_test_and_heal`` (``heal=False``), so
    the verdict written by ``_step_test`` reflects an actual test-suite turn.
    Observable outcome: exit 0 plus a parseable artifact at
    ``<target>/.daydream/deep/test-verdict.json`` recording ``passed`` True.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")
    _install_stub_backend(monkeypatch, tiny_diff_target)
    mute_side_effects(heal=False)

    rc = await run(make_config(tiny_diff_target, assume="yes", non_interactive=False))
    assert rc == 0

    verdict_file = tiny_diff_target / ".daydream" / "deep" / "test-verdict.json"
    assert verdict_file.is_file(), "passing run did not write test-verdict.json"
    verdict = json.loads(verdict_file.read_text())
    assert verdict["passed"] is True, verdict
    assert verdict["retries"] == 0, "a green suite must not have consumed a heal retry"


async def test_test_verdict_artifact_written_on_failing_suite(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """Real-path: a permanently-red suite STILL leaves ``test-verdict.json``.

    ``_step_test`` returns ``Stop(1)`` on failure; the verdict must be
    persisted before that early-return, otherwise the failing outcome -- the
    one a caller most needs -- would never reach disk. With ``--yes`` the heal
    loop gets exactly ONE bounded auto fix-and-retry, so the red suite runs
    twice and then aborts. Observable outcome: exit 1 AND an artifact recording
    ``passed`` False.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")
    stub = _install_stub_backend(monkeypatch, tiny_diff_target)
    stub.fail_all_test_runs = True  # suite never goes green, even after the heal fix
    mute_side_effects(heal=False)

    rc = await run(make_config(tiny_diff_target, assume="yes", non_interactive=False))
    assert rc == 1, "a permanently-red suite must fail the run"

    verdict_file = tiny_diff_target / ".daydream" / "deep" / "test-verdict.json"
    assert verdict_file.is_file(), "failing run lost test-verdict.json to the early-return"
    verdict = json.loads(verdict_file.read_text())
    assert verdict["passed"] is False, verdict
    assert verdict["retries"] == 1, "--yes grants exactly one bounded auto fix-and-retry"


class _CommittingStubBackend(_StubBackend):
    """Stub that answers the commit prompt with a real, untrailered git commit.

    ``_do_commit`` delegates the commit itself to an agent turn, so a stub that
    only records the prompt leaves the whole implementation -- the HEAD-moved
    check and the trailer amend that ``daydream_commits()`` depends on --
    unexercised. Committing without the trailers the prompt asks for drives that
    repair branch for real.
    """

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ) -> Any:
        if prompt.startswith("Stage all changes and commit"):
            _git(cwd, "add", "--all")
            _git(cwd, "commit", "-m", "fix: align greeting copy")
            yield TextEvent(text="Committed.")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
        ):
            yield event


async def test_test_verdict_records_failure_when_operator_ignores_it(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute
) -> None:
    """Real-path: heal-menu choice "3" continues the run WITHOUT claiming a green suite.

    Drives ``runner.run`` -> deep orchestrator -> the REAL ``phase_test_and_heal``
    (``heal=False``) and the REAL ``phase_commit_push`` (``commit=False``) with the
    scripted stub backend as the only mocked seam. The suite is permanently red and
    the operator answers the interactive heal menu with "3" (ignore and continue).
    Two observable outcomes, both required: the on-disk ``test-verdict.json`` records
    ``passed`` False (an "ignore" is an operator override, never evidence the suite
    went green), AND the fix is really committed -- HEAD advances onto a commit that
    carries the fix and the daydream trailers.
    """
    import daydream
    from daydream.runner import run

    _silence(monkeypatch, prompts=False)
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "y")

    # phases.prompt_user is shared: intent-confirmation needs "y"; the heal menu
    # ("Choice") needs "3" (ignore and continue).
    def _phases_prompt(console: Any, message: str, default: str = "") -> str:  # noqa: ARG001
        return "3" if "Choice" in message else "y"

    monkeypatch.setattr("daydream.phases.prompt_user", _phases_prompt)

    stub = _CommittingStubBackend(tiny_diff_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    monkeypatch.setattr("daydream.deep.orchestrator.get_installed_skills", lambda: None)
    monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)
    stub.fail_all_test_runs = True

    mute_side_effects(heal=False, commit=False)

    head_before = _git(tiny_diff_target, "rev-parse", "HEAD")

    rc = await run(make_config(tiny_diff_target, non_interactive=False))
    assert rc == 0, "choice '3' must continue the run, not abort it"

    verdict_file = tiny_diff_target / ".daydream" / "deep" / "test-verdict.json"
    verdict = json.loads(verdict_file.read_text())
    assert verdict["passed"] is False, (
        f"an ignored failure was persisted as a green suite: {verdict}"
    )
    assert verdict["ignored"] is True, f"the operator override was not recorded: {verdict}"
    assert _git(tiny_diff_target, "rev-parse", "HEAD") != head_before, (
        "choice '3' did not produce a commit"
    )
    committed = _git(tiny_diff_target, "show", "--name-only", "--format=", "HEAD").split()
    assert ".fixed-api_py" in committed, f"the applied fix was not part of the commit: {committed}"
    head_message = _git(tiny_diff_target, "log", "-1", "--format=%B")
    assert "Daydream-Run: " in head_message, f"commit lost the run trailer: {head_message}"
    assert f"Daydream-Version: {daydream.__version__}" in head_message, (
        f"commit lost the version trailer: {head_message}"
    )
    assert stub.test_suite_calls == 1, (
        f"expected one test-suite run before the ignore, saw {stub.test_suite_calls}"
    )


# --- Parse fan-out runs concurrently (D-21 parallel) -------------------------


def _parse_stack_name(prompt: str) -> str:
    """Stack name a parse prompt points at, via its ``stack-<name>-review.md``."""
    m = re.search(r"stack-(\S+?)-review\.md", prompt)
    return m.group(1) if m else ""


# Parse tasks are started in sorted(stack_name) order: generic, python, react,
# structure. The rendezvous makes the FIRST of those wait on the LAST, so the
# serial loop deadlocks (structure never starts) while concurrent execution
# completes.
_FIRST_PARSE_STACK = "generic"
_LAST_PARSE_STACK = "structure"


class _RendezvousParseStub(StubBackend):
    """Parse for the first stack blocks until the last stack's parse starts.

    Only a concurrent implementation completes: the serial loop reaches
    ``generic`` first and would wait forever for ``structure``, which it has
    not started yet.
    """

    def __init__(self, target: Path) -> None:
        super().__init__(target)
        self.other_parse_started = anyio.Event()

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
        if "extract only actionable issues" in prompt.lower():
            name = _parse_stack_name(prompt)
            if name == _LAST_PARSE_STACK:
                self.other_parse_started.set()
            elif name == _FIRST_PARSE_STACK:
                await self.other_parse_started.wait()
        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
        ):
            yield event


async def test_parse_runs_concurrently(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The N per-stack parse calls overlap; every stack's records land on disk."""
    stub = _RendezvousParseStub(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kw: stub)
    monkeypatch.setattr("daydream.deep.orchestrator.get_installed_skills", lambda: None)
    monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)

    with anyio.fail_after(15):
        exit_code = await _run_deep(multi_stack_target)

    assert exit_code == 0
    deep = multi_stack_target / ".daydream" / "deep"
    assert sorted(p.name for p in deep.glob("stack-*-records.json")) == [
        "stack-generic-records.json",
        "stack-python-records.json",
        "stack-react-records.json",
        "stack-structure-records.json",
    ]

    # The first-started stack could only finish after the last one had begun.
    parse_order = [
        _parse_stack_name(c["prompt"])
        for c in stub.calls
        if "extract only actionable issues" in c["prompt"].lower()
    ]
    assert parse_order.index(_LAST_PARSE_STACK) < parse_order.index(
        _FIRST_PARSE_STACK
    ), parse_order


class _FailingParseStub(StubBackend):
    """Raises a distinctive error for one stack's parse; siblings succeed."""

    def __init__(self, target: Path, failing_stack: str) -> None:
        super().__init__(target)
        self._failing_stack = failing_stack

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
        if (
            "extract only actionable issues" in prompt.lower()
            and _parse_stack_name(prompt) == self._failing_stack
        ):
            raise ZeroDivisionError("parse blew up")
        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
        ):
            yield event


async def test_parse_failure_propagates_original_exception_type(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One stack's parse failure fails the run with its own exception type.

    Not an ExceptionGroup — the task group's captured failure is re-raised
    directly, matching the serial loop's semantics.
    """
    stub = _FailingParseStub(multi_stack_target, failing_stack="python")
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kw: stub)
    monkeypatch.setattr("daydream.deep.orchestrator.get_installed_skills", lambda: None)
    monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)

    with pytest.raises(ZeroDivisionError, match="parse blew up"):
        await _run_deep(multi_stack_target)

    # Siblings that finished before the failure surfaced kept their records.
    deep = multi_stack_target / ".daydream" / "deep"
    survivors = sorted(p.name for p in deep.glob("stack-*-records.json"))
    assert "stack-python-records.json" not in survivors
    assert survivors, "sibling parse results must survive one stack's failure"


# --- Budget caps on the deep-review agents -----------------------------------


async def test_root_trajectory_step_ids_survive_concurrent_wonder(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """A wonder turn outliving the per-stack fan-out still writes a valid trajectory.

    Wonder and the fan-out are siblings on ONE recorder: wonder's step ids are
    allocated when its invocation opens but flushed when it closes, while the
    fan-out's ``create_dispatch_step`` appends to the same list in between. A
    handful of tool calls in the wonder turn makes wonder lose that race
    deterministically — the count stays under any tool-call ceiling so this
    holds independently of the default budget.

    Discriminating: without the ``step_id``-ordered merge in ``_extend_steps``
    the root list is [1, 2, 4, 3, ...], ATIF's "sequential from 1" check rejects
    it, and the explicit-path write raises SystemExit(2) — the whole run dies at
    the finish line with no trajectory on disk.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.alternatives_tool_calls = 5
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(60):
        await run(
            make_config(
                multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"
            )
        )

    payload = json.loads(traj.read_text())
    ids = [s["step_id"] for s in payload["steps"]]
    assert ids == list(range(1, len(ids) + 1)), ids
    # The concurrent wonder turn is present in the root trajectory, not dropped.
    phases = [(s.get("extra") or {}).get("daydream_phase") for s in payload["steps"]]
    assert "alternatives" in phases, phases


async def test_budget_truncated_wonder_fails_loudly(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """A budget-truncated wonder pass fails the run instead of degrading to [].

    Discriminating: uncapped, the wonder turn's unbounded burst never ends, so
    ``fail_after`` trips; with the cap but the old degrade path the run would
    exit 0 carrying an empty alternatives lens.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_alternatives = True
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        with pytest.raises(RuntimeError, match="budget"):
            await run(
                make_config(
                    multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"
                )
            )

    run_root = multi_stack_target / ".daydream"
    stop_reasons = _scan_trajectory_extra(run_root, traj, "stop_reason")
    assert any("budget" in str(v) for v in stop_reasons), stop_reasons
    # The run died instead of writing an empty alternatives lens.
    alts = multi_stack_target / ".daydream" / "deep" / "alternatives.json"
    assert not alts.exists() or json.loads(alts.read_text()) != []


async def test_budget_truncated_stack_lands_in_failed_stacks(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """A truncated per-stack review is recorded as a failure, not a success."""
    from daydream.runner import run

    _silence(monkeypatch)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_stack = "python"
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        await run(
            make_config(
                multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"
            )
        )

    failures_path = multi_stack_target / ".daydream" / "deep" / "per-stack-failures.json"
    failures = json.loads(failures_path.read_text())
    assert "python" in failures, failures
    assert "budget" in failures["python"].lower()


async def test_budget_truncated_parse_fails_loudly(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """A budget-truncated parse fails the run instead of dropping a stack's findings."""
    from daydream.runner import run

    _silence(monkeypatch)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_parse = "python"
    mute_side_effects()

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        with pytest.raises(RuntimeError, match="budget"):
            await run(
                make_config(
                    multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"
                )
            )

    run_root = multi_stack_target / ".daydream"
    stop_reasons = _scan_trajectory_extra(run_root, traj, "stop_reason")
    assert any("budget" in str(v) for v in stop_reasons), stop_reasons
    # The truncated stack wrote no records file — the run refused to proceed
    # with a silently-empty bucket.
    assert not (run_root / "deep" / "stack-python-records.json").exists()


async def test_runaway_test_turn_is_bounded_and_reaches_abort(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """A hung test turn is capped, so the run reaches the heal/abort path.

    Discriminating: uncapped, the test-suite stream never completes and ``run``
    never returns — ``fail_after`` turns that regression into a failure rather
    than an infinite hang.
    """
    from daydream.runner import run

    _silence(monkeypatch)
    monkeypatch.setattr("daydream.phases.DEFAULT_TOOL_CALL_BUDGET", 3)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.runaway_test = True
    # heal=False keeps phase_test_and_heal REAL so the runaway turn is actually
    # dispatched through run_agent's budget guard.
    mute_side_effects(heal=False)

    traj = tmp_path / "trajectory.json"
    with anyio.fail_after(30):
        exit_code = await run(
            make_config(
                multi_stack_target, trajectory_path=traj, assume="yes", output_mode="loop"
            )
        )

    # Truncation flows into detect_test_success() False -> heal/abort; the run
    # terminates with an int exit code instead of hanging.
    assert isinstance(exit_code, int)
    test_turns = [
        c for c in stub.calls if "run the project's test suite" in c["prompt"].lower()
    ]
    assert test_turns, "the test phase never ran"
    run_root = multi_stack_target / ".daydream"
    stop_reasons = _scan_trajectory_extra(run_root, traj, "stop_reason")
    assert any("budget" in str(v) for v in stop_reasons), stop_reasons


async def test_deep_run_inlines_small_diff_into_intent_and_wonder(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real path: a small diff is inlined into BOTH the intent and wonder prompts."""
    stub = _install_stub_backend(monkeypatch, tiny_diff_target)

    assert await _run_deep(tiny_diff_target) == 0

    intent_prompt = next(
        c["prompt"] for c in stub.calls
        if "understand the intent of these changes" in c["prompt"].lower()
    )
    wonder_prompt = next(
        c["prompt"] for c in stub.calls
        if "evaluate the implementation" in c["prompt"].lower()
    )

    for name, prompt in (("intent", intent_prompt), ("wonder", wonder_prompt)):
        assert "diff --git" in prompt, f"{name} prompt did not inline the diff"
        assert "do NOT re-Read" in prompt, f"{name} prompt kept the read instruction"
    assert "Read the diff file at" not in intent_prompt


async def test_deep_run_keeps_pointer_when_diff_exceeds_budget(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-budget diff falls back to today's diff.patch pointer in both prompts."""
    from daydream.deep.prompts import INLINE_DIFF_BUDGET_BYTES

    # Push the diff over the byte budget with a large committed file.
    big = "\n".join(f"line {i} of filler content" for i in range(INLINE_DIFF_BUDGET_BYTES // 10))
    (multi_stack_target / "big.py").write_text(big + "\n")
    _git(multi_stack_target, "add", "big.py")
    _git(multi_stack_target, "commit", "-m", "add big file")

    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target) == 0

    intent_prompt = next(
        c["prompt"] for c in stub.calls
        if "understand the intent of these changes" in c["prompt"].lower()
    )
    wonder_prompt = next(
        c["prompt"] for c in stub.calls
        if "evaluate the implementation" in c["prompt"].lower()
    )

    assert "Read the diff file at" in intent_prompt
    assert "diff.patch" in wonder_prompt
    for name, prompt in (("intent", intent_prompt), ("wonder", wonder_prompt)):
        assert "line 500 of filler content" not in prompt, f"{name} inlined an over-budget diff"


# --- Merge resumes the arbiter's session -------------------------------------


def _merge_call(stub: StubBackend) -> dict[str, Any]:
    return next(c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower())


async def test_merge_resumes_arbiter_session(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The merge call resumes the arbiter's session and warns about stale records."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"  # drives arbiter selection
    stub.arbiter_session_id = "arb-sess"

    assert await _run_deep(multi_stack_target) == 0

    merge_call = _merge_call(stub)
    assert merge_call["continuation"] is not None
    assert merge_call["continuation"].data["session_id"] == "arb-sess"
    # The resumed context holds pre-adjudication records, so the prompt must
    # force a re-read from disk.
    assert "re-read" in merge_call["prompt"].lower()


async def test_merge_cold_when_arbiter_mints_no_token(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No arbiter token -> merge runs cold with today's prompt, no addendum."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.arbiter_session_id = None  # backend mints nothing (e.g. persist_session off)

    assert await _run_deep(multi_stack_target) == 0

    merge_call = _merge_call(stub)
    assert merge_call["continuation"] is None
    assert "re-read" not in merge_call["prompt"].lower()


async def test_merge_cold_when_arbiter_skipped_on_resume(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--start-at merge past a completed adjudication runs merge cold."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.arbiter_session_id = "arb-sess"

    # High-severity records WOULD qualify for arbitration, so the marker is the
    # only thing keeping the arbiter from running.
    high = _record(description="py issue", severity="high")
    deep = _prime_merge_resume(
        multi_stack_target,
        python=[high],
        react=[_record(description="tsx issue", severity="high")],
        generic=[_record(description="md issue", severity="high")],
        structure=[],
    )
    (deep / "arbiter-complete.marker").write_text("done")

    assert await _run_deep(multi_stack_target, start_at="merge") == 0

    arbiter_calls = [c for c in stub.calls if "you are the arbiter" in c["prompt"].lower()]
    assert arbiter_calls == []
    merge_call = _merge_call(stub)
    assert merge_call["continuation"] is None
    assert "re-read" not in merge_call["prompt"].lower()


def test_merge_prompt_cold_path_is_byte_identical(tmp_path: Path) -> None:
    """resumed_from_arbiter=False reproduces today's prompt exactly."""
    from daydream.deep.prompts import build_merge_prompt

    kwargs: dict[str, Any] = dict(
        per_stack_records_paths=[tmp_path / "stack-python-records.json"],
        intent_path=tmp_path / "intent.md",
        alternatives_path=tmp_path / "alternatives.json",
        dedup_candidates_path=tmp_path / "dedup.json",
        output_path=tmp_path / "out.md",
    )
    omitted = build_merge_prompt(**kwargs)
    explicit_false = build_merge_prompt(**kwargs, resumed_from_arbiter=False)
    resumed = build_merge_prompt(**kwargs, resumed_from_arbiter=True)

    assert omitted == explicit_false
    assert resumed != omitted
    assert resumed.startswith(omitted)  # purely additive addendum


# --- Task 12b: each TTT step writes its own artifact -------------------------


async def test_intent_artifact_survives_wonder_failure(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """intent.md is on disk even when the wonder step dies.

    Discriminating: both files used to be written together AFTER the wonder
    agent, so a wonder failure discarded the intent artifact too.
    """
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.fail_alternatives = True

    with pytest.raises(RuntimeError, match="alternatives blew up"):
        await _run_deep(multi_stack_target)

    intent_md = multi_stack_target / ".daydream" / "deep" / "intent.md"
    assert intent_md.read_text().strip(), "intent.md must survive the wonder failure"
    # The wonder half never ran, so its artifact is legitimately absent.
    assert not (multi_stack_target / ".daydream" / "deep" / "alternatives.json").exists()


async def test_both_ttt_artifacts_written_on_the_happy_path(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relocating the writer leaves contents and ctx.data pointers unchanged."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    assert await _run_deep(multi_stack_target) == 0

    deep = multi_stack_target / ".daydream" / "deep"
    assert deep / "intent.md"
    assert (deep / "intent.md").read_text().strip()
    assert json.loads((deep / "alternatives.json").read_text())


async def test_skip_tier_writes_empty_alternatives(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both artifacts exist even when the diff is small enough to run wonder."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tiny_diff_target)

    assert await _run_deep(tiny_diff_target) == 0

    deep = tiny_diff_target / ".daydream" / "deep"
    assert (deep / "intent.md").read_text().strip()
    assert isinstance(json.loads((deep / "alternatives.json").read_text()), list)


# --- Task 12c: wonder runs concurrently with the per-stack fan-out -----------


class _WonderRendezvousStub(StubBackend):
    """The wonder turn blocks until a per-stack review has started.

    Serial code deadlocks (wonder runs strictly before the fan-out); concurrent
    code completes.
    """

    def __init__(self, target: Path) -> None:
        super().__init__(target)
        self.per_stack_started = anyio.Event()

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
        pl = prompt.lower()
        if "you are reviewing the" in pl:
            self.per_stack_started.set()
        elif "would you have done this differently" in pl or "evaluate the implementation" in pl:
            await self.per_stack_started.wait()
        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
        ):
            yield event


def _install_raw(monkeypatch: pytest.MonkeyPatch, stub: StubBackend) -> None:
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kw: stub)
    monkeypatch.setattr("daydream.deep.orchestrator.get_installed_skills", lambda: None)
    monkeypatch.setattr("daydream.deep.orchestrator.EXPLORATION_AVAILABLE", False)


async def test_wonder_runs_concurrently_with_per_stack(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wonder and the fan-out overlap; alternatives.json lands before parse reads it."""
    _silence(monkeypatch)
    _install_raw(monkeypatch, _WonderRendezvousStub(multi_stack_target))

    with anyio.fail_after(15):
        assert await _run_deep(multi_stack_target) == 0

    alts = json.loads(
        (multi_stack_target / ".daydream" / "deep" / "alternatives.json").read_text()
    )
    assert alts, "wonder's artifact must be written before parse consumes it"


async def test_concurrent_per_stack_prompts_omit_alternatives_pointer(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-stack: reviewers drop the pointer; adjudication prompts keep it."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"  # ensure the arbiter runs

    assert await _run_deep(multi_stack_target) == 0

    per_stack = [c["prompt"] for c in stub.calls if "you are reviewing the" in c["prompt"].lower()]
    assert per_stack
    for prompt in per_stack:
        assert "alternatives.json" not in prompt
        assert "intent.md" in prompt  # only the alternatives paragraph is dropped

    for phrase in ("you are the arbiter", "cross-stack merge agent"):
        matching = [c["prompt"] for c in stub.calls if phrase in c["prompt"].lower()]
        assert matching, phrase
        assert all("alternatives.json" in p for p in matching), phrase


async def test_single_stack_keeps_serial_order_and_pointer(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-stack mode has no merge agent, so the reviewer pointer must survive."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, tiny_diff_target)

    assert await _run_deep(tiny_diff_target) == 0

    order = [c["prompt"].lower() for c in stub.calls]
    wonder_idx = next(
        i for i, p in enumerate(order)
        if "would you have done this differently" in p or "evaluate the implementation" in p
    )
    per_stack_idx = next(i for i, p in enumerate(order) if "you are reviewing the" in p)
    assert wonder_idx < per_stack_idx, "single-stack mode must stay serial"

    per_stack = [c["prompt"] for c in stub.calls if "you are reviewing the" in c["prompt"].lower()]
    assert per_stack
    assert all("alternatives.json" in p for p in per_stack)


async def test_wonder_failure_fails_run_with_fanout_outputs_on_disk(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held wonder exception is re-raised after the join, original type intact."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.fail_alternatives = True

    with pytest.raises(RuntimeError, match="alternatives blew up"):
        await _run_deep(multi_stack_target)

    deep = multi_stack_target / ".daydream" / "deep"
    reviews = sorted(p.name for p in deep.glob("stack-*-review.md"))
    assert reviews, "fan-out outputs must survive for a --start-at resume"


def test_extension_api_version_is_four_and_alternatives_step_is_gone() -> None:
    from daydream.deep.orchestrator import STEPS
    from daydream.extensions.api import EXTENSION_API_VERSION

    assert EXTENSION_API_VERSION == 4
    names = [s.name for s in STEPS]
    assert "alternatives" not in names
    assert "per-stack-reviews" in names


# --- Task 15: --start-at refuses artifacts produced from a different diff -----


async def test_start_at_merge_refuses_after_the_diff_changes(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh run writes diff-key; changing the diff then blocks the resume."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target) == 0

    deep = multi_stack_target / ".daydream" / "deep"
    key_file = deep / "diff-key"
    original_key = key_file.read_text().strip()
    assert original_key, "a fresh run must record the diff key"

    (multi_stack_target / "api.py").write_text("def hello():\n    return 'galaxy'\n")
    _git(multi_stack_target, "add", "api.py")
    _git(multi_stack_target, "commit", "-m", "change again")

    stub2 = _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target, start_at="merge") == 1
    # The run bailed before any agent turn.
    assert stub2.calls == []
    # A refused resume must NOT rewrite the key it is checked against.
    assert key_file.read_text().strip() == original_key


async def test_start_at_merge_proceeds_when_the_diff_is_unchanged(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The freshness gate is not a blanket refusal: same diff still resumes."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target) == 0

    stub2 = _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target, start_at="merge") == 0
    assert any("cross-stack merge agent" in c["prompt"].lower() for c in stub2.calls)


async def test_pre_upgrade_artifacts_without_a_key_refuse_resume(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An artifact dir from before diff tracking is treated as unverifiable."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target) == 0

    (multi_stack_target / ".daydream" / "deep" / "diff-key").unlink()

    stub2 = _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target, start_at="merge") == 1
    assert stub2.calls == []
