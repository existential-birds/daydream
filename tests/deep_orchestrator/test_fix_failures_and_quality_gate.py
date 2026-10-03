"""Fix Failures And Quality Gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.config_file import DaydreamFileConfig
from daydream.eval import analyzer as analyzer_mod
from daydream.runner import run
from tests.deep_orchestrator.support import (
    _capture_warnings,
    _merged_items,
    _only_archived_run,
)
from tests.harness.git_helpers import git as _git
from tests.test_deep_orchestrator import (
    _FIX_EDIT_ERODED,
    _FIX_EDIT_VERBOSE,
    _PARTIAL_FIX_MARKER,
    MakeConfig,
    Mute,
    _add_to_reviewed_diff,
    _build_gate_target,
    _build_gate_target_no_functions,
    _build_gate_target_with_helper,
    _build_scope_creep_target,
    _ExtraEditBackend,
    _force_interactive,
    _install_stub_backend,
    _merge_item,
    _prime_merge_resume,
    _read_quality_gate,
    _run_quality_gate_fixture,
    _silence,
)


async def test_fix_failure_reverts_partial_edit_and_marks_manifest_partial(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real-path: a fix group that raises MaxTurnsError mid-edit is rolled back, its partial content saved, and the
    archived run is marked ``partial``."""

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.fix_partial_then_maxturns = "App.tsx"
    stub.fix_edit_line = "# retained successful group\n"
    stub.merge_items = [_merge_item(1, "api.py", "high"), _merge_item(2, "App.tsx", "high")]
    pre_fix_apptsx = (multi_stack_target / "App.tsx").read_text()
    exit_code = await run(
        make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False, archive=True,)
    )
    assert exit_code == 1  # dropped fix group => nonzero

    # (b) the successful group's authorized edit survives; its old out-of-scope
    # sentinel does not.
    assert "# retained successful group" in (multi_stack_target / "api.py").read_text()
    assert not (multi_stack_target / ".fixed-api_py").exists()

    # (c) failed group reverted to pre-fix content; broken edit gone.
    apptsx_after = (multi_stack_target / "App.tsx").read_text()
    assert apptsx_after == pre_fix_apptsx
    assert _PARTIAL_FIX_MARKER not in apptsx_after
    patches = list((multi_stack_target / ".daydream" / "partial-fixes").glob("*.patch"))
    assert patches == []

    # (a) manifest records the failure and is no longer "complete".
    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["fix_failures"], "manifest must record the dropped fix group"
    assert any("App.tsx" in key for key in manifest["fix_failures"])

async def test_fix_preflight_unconfined_finding_archives_blocked_item_identities(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Footprint preflight blocks unsafe paths before fixing and records why."""

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects(commit=False)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    # Symlink-escape finding: src/handler.py is a repo-local path whose real
    # file lives outside the repo (same shape as _unconfined_finding_file's
    # "symlink" kind). Commit it into the reviewed diff so the fix gate keeps it.
    src = multi_stack_target / "src"
    src.mkdir(exist_ok=True)
    outside = multi_stack_target.parent / f"{multi_stack_target.name}-outside.py"
    outside.write_text("def outside():\n    pass\n")
    (src / "handler.py").symlink_to(outside)
    _add_to_reviewed_diff(multi_stack_target, ["src/handler.py"])

    stub.merge_items = [_merge_item(1, "src/handler.py", "high")]
    warnings = _capture_warnings(monkeypatch, "daydream.deep.fix_steps.print_warning")
    exit_code = await run(
        make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False, archive=True,)
    )
    assert exit_code == 1  # handled failure => Stop(1), not a raise

    # The admitted evidence session records the blocked findings in archive.
    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    items = _merged_items(multi_stack_target / ".daydream" / "deep")
    assert set(manifest["fix_failures"]) == {item["item_uid"] for item in items}
    assert set(manifest["fix_failures"].values()) == {"fix_preflight_rejected: fix cycle did not start"}
    assert manifest["pipeline_status"] == "partial"
    assert manifest["phase_states"]["test"] == {"ran": False, "status": "absent"}

    # (b) no dangling pre-fix stash / tree restored: the symlink finding's
    #     group never got a fix applied (preflight aborts before dispatch).
    assert not (multi_stack_target / ".fixed-src_handler_py").exists()

    # Unsafe path values are not reflected into diagnostics.
    assert not any("src/handler.py" in warning for warning in warnings)

async def test_fix_failure_confines_orphan_and_restores_protected_file_in_archive(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real runner confines a failed fixer and archives its restore audit."""

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.fix_partial_then_maxturns = "App.tsx"
    stub.fix_orphan_file = "store/uuid.go"
    stub.fix_edit_line = "# retained successful fix\n"
    scratch = multi_stack_target / "owner-scratch.bin"
    scratch.write_bytes(b"\x00owner-original")
    stub.fix_damage_protected_file = "owner-scratch.bin"
    stub.merge_items = [_merge_item(1, "api.py", "high"), _merge_item(2, "App.tsx", "high")]
    exit_code = await run(
        make_config(multi_stack_target, assume="yes", output_mode="loop", non_interactive=False, archive=True,)
    )
    assert exit_code == 1

    # The successful group remains while the failed group's outside-run write
    # is removed and the user's protected untracked file is restored exactly.
    assert "# retained successful fix" in (multi_stack_target / "api.py").read_text()
    assert not (multi_stack_target / "store" / "uuid.go").exists()
    assert scratch.read_bytes() == b"\x00owner-original"

    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # The archive is partial, but there is no surviving leftover.  The durable
    # policy audit records both confinement operations.
    assert manifest["status"] == "partial"
    assert manifest["fix_leftover_untracked"] is None
    run_audit = json.loads((run_dir / "deep" / "fix-footprint.json").read_text())
    restored = {event["path"] for event in run_audit["events"] if event["action"] in {"remove", "restore"}}
    assert {"store/uuid.go", "owner-scratch.bin"} <= restored

@pytest.mark.parametrize("location", ["live-artifact", "archived-manifest"])
async def test_fix_quality_gate_persists_verbosity_regression(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, make_config: MakeConfig,
    mute_side_effects: Mute, location: str,
) -> None:
    assert await _run_quality_gate_fixture(multi_stack_target, monkeypatch, make_config, mute_side_effects) == 0
    assert "def choose(x):" in (multi_stack_target / "api.py").read_text()
    if location == "live-artifact":
        gate = _read_quality_gate(multi_stack_target)
    else:
        manifest = json.loads((_only_archived_run(archive_dir) / "manifest.json").read_text())
        gate = manifest["fix_quality_gate"]
    assert gate is not None
    assert gate["enabled"] is True
    entry = gate["rounds"][0]["per_file"]["api.py"]
    assert entry["verbosity_after"] > entry["verbosity_before"]
    assert entry["flagged"] is True


@pytest.mark.parametrize("threshold,flagged", [
    pytest.param(100.0, False, id="tolerant"), pytest.param(0.0, True, id="strict"),
])
async def test_fix_quality_gate_threshold_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    threshold: float, flagged: bool,
) -> None:
    """The same edit is flagged according to the configured delta thresholds."""
    target = _build_gate_target(tmp_path, "gate_threshold")
    assert await _run_quality_gate_fixture(
        target, monkeypatch, make_config, mute_side_effects,
        file_config=DaydreamFileConfig(quality_gate_erosion_delta=threshold, quality_gate_verbosity_delta=threshold),
    ) == 0
    entry = _read_quality_gate(target)["rounds"][0]["per_file"]["api.py"]
    assert entry["verbosity_after"] > entry["verbosity_before"]
    assert entry["flagged"] is flagged

async def test_fix_quality_gate_fail_open(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """Real-path (#315/#329): an analyzer failure degrades to an auditable unavailable round."""

    real_analyze = analyzer_mod.analyze_quality

    def _boom(daydream_dir: Path, candidate_paths: set[str] | None = None, **kwargs: Any,) -> dict[str, Any]:
        if candidate_paths is not None:
            raise RuntimeError("analyzer down")
        return real_analyze(daydream_dir, candidate_paths, **kwargs)


    monkeypatch.setattr(analyzer_mod, "analyze_quality", _boom)
    exit_code = await _run_quality_gate_fixture(multi_stack_target, monkeypatch, make_config, mute_side_effects,
        file_config=DaydreamFileConfig(quality_gate_erosion_absolute=1.25, quality_gate_verbosity_absolute=2.5,),
    )
    assert exit_code == 0

    gate = _read_quality_gate(multi_stack_target)
    assert gate["enabled"] is True
    assert gate["erosion_absolute_threshold"] == 1.25
    assert gate["verbosity_absolute_threshold"] == 2.5
    unavailable = gate["rounds"][0]["unavailable"]
    assert unavailable["stage"] == "before"
    assert "analyzer down" in unavailable["reason"]

@pytest.mark.parametrize("file_config,thresholds,flagged", [
    pytest.param(None, (0.05, 0.05, 0.05, 0.05), True, id="default-absolute"),
    pytest.param(DaydreamFileConfig(quality_gate_erosion_absolute=100.0,
                 quality_gate_verbosity_absolute=100.0, quality_gate_verbosity_delta=100.0),
                 (100.0, 100.0, 0.05, 100.0), False, id="tolerant-absolute"),
])
async def test_fix_quality_gate_absolute_threshold_controls_undefined_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    file_config: DaydreamFileConfig | None, thresholds: tuple[float, ...], flagged: bool,
) -> None:
    """An existing file without functions uses the absolute threshold, never its undefined delta."""
    target = _build_gate_target_no_functions(tmp_path, "gate_undefined_baseline")
    assert await _run_quality_gate_fixture(
        target, monkeypatch, make_config, mute_side_effects, fix_edit_line=_FIX_EDIT_ERODED,
        file_config=file_config,
    ) == 0
    gate = _read_quality_gate(target)
    assert gate["enabled"] is True
    assert tuple(gate[f"{metric}_{kind}_threshold"] for kind in ("absolute", "delta")
                 for metric in ("erosion", "verbosity")) == thresholds
    entry = gate["rounds"][0]["per_file"]["api.py"]
    assert entry["erosion_before"] is None
    assert entry["erosion_after"] is not None and entry["erosion_after"] > 0.05
    assert entry["erosion_delta"] is None
    assert entry["flagged"] is flagged

async def test_fix_quality_gate_flags_undefined_baseline_verbosity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """Real-path (#329): an empty file uses the verbosity absolute fallback."""

    target = _build_gate_target_no_functions(tmp_path, "gate_undefined_verbosity")
    (target / "api.py").write_text("\n")
    exit_code = await _run_quality_gate_fixture(target, monkeypatch, make_config, mute_side_effects,
        file_config=DaydreamFileConfig(
            quality_gate_erosion_absolute=100.0, quality_gate_erosion_delta=100.0, quality_gate_verbosity_delta=100.0,
            quality_gate_verbosity_absolute=0.0,
        ),
    )
    assert exit_code == 0

    entry = _read_quality_gate(target)["rounds"][0]["per_file"]["api.py"]
    assert entry["verbosity_before"] is None
    assert entry["verbosity_after"] is not None
    assert entry["verbosity_after"] > 0.0
    assert entry["verbosity_delta"] is None
    assert entry["flagged"] is True


async def test_fix_quality_gate_artifact_bound_to_current_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, archive_dir: Path,
    mute_side_effects: Mute,
) -> None:
    alpha = _build_gate_target(tmp_path, "gate_alpha")
    exit_code = await _run_quality_gate_fixture(alpha, monkeypatch, make_config, mute_side_effects)
    assert exit_code == 0
    alpha_gate = _read_quality_gate(alpha)
    assert alpha_gate["session_id"]

    beta = _build_gate_target(tmp_path, "gate_beta")
    exit_code = await _run_quality_gate_fixture(beta, monkeypatch, make_config, mute_side_effects)
    assert exit_code == 0
    beta_gate = _read_quality_gate(beta)
    assert beta_gate["session_id"]
    assert beta_gate["session_id"] != alpha_gate["session_id"]

    manifests = [json.loads((d / "manifest.json").read_text(encoding="utf-8")) for d in (archive_dir / "runs").iterdir()
    ]
    # Per-manifest correspondence: the manifest that carries session A's
    # verdict must BE session A's manifest, and vice versa. manifest["session_id"]
    # comes from the recorder; manifest["fix_quality_gate"]["session_id"] comes
    # from the gate artifact, so a swapped verdict would fail this binding.
    by_session = {m["session_id"]: m for m in manifests}
    assert alpha_gate["session_id"] in by_session
    assert beta_gate["session_id"] in by_session
    assert by_session[alpha_gate["session_id"]]["fix_quality_gate"]["session_id"] == alpha_gate["session_id"]
    assert by_session[beta_gate["session_id"]]["fix_quality_gate"]["session_id"] == beta_gate["session_id"]
    # The two verdicts are distinct artifacts: alpha's manifest never carries beta's verdict.
    assert by_session[alpha_gate["session_id"]]["fix_quality_gate"]["session_id"] != beta_gate["session_id"]

async def test_fix_quality_gate_flags_unparseable_post_fix_file(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """Real-path (#329/Finding 5): a file missing from post-fix analyzer output is flagged."""

    real_analyze = analyzer_mod.analyze_quality

    def _stub(daydream_dir: Any, candidate_paths: set[str] | None = None, **kwargs: Any,) -> dict[str, Any]:
        result = real_analyze(daydream_dir, candidate_paths, **kwargs)
        if "def choose(x):" in (multi_stack_target / "api.py").read_text(encoding="utf-8"):
            result["per_file"] = {rel: entry for rel, entry in result["per_file"].items() if rel != "api.py"}
        return result

    monkeypatch.setattr(analyzer_mod, "analyze_quality", _stub)
    warnings = _capture_warnings(monkeypatch, "daydream.deep.quality_gate.print_warning")
    exit_code = await _run_quality_gate_fixture(multi_stack_target, monkeypatch, make_config, mute_side_effects)
    assert exit_code == 0

    gate = _read_quality_gate(multi_stack_target)
    assert gate["enabled"] is True
    entry = gate["rounds"][0]["per_file"]["api.py"]
    assert entry["unparseable"] is True
    assert entry["flagged"] is True
    assert "post-fix analyzer output" in entry["reason"]
    assert any("api.py" in w for w in warnings), "the unparseable file must be named in a warning"

async def test_fix_quality_gate_malformed_resume_artifact_repairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """Real-path (#329/Finding 6): a malformed resume artifact can't silently disable the gate."""

    target = _build_gate_target(tmp_path, "gate_malformed_resume")
    deep = _prime_merge_resume(target, python=[], structure=[])
    (deep / "merged-items.json").write_text(json.dumps({
        "items": [{**_merge_item(1, "api.py", "high"), "item_uid": "item:1"}],
    }))
    from daydream.deep.artifacts import (
        DeepArtifact,
        persist_review_coverage,
    )
    from daydream.review_result import ReviewCoverage

    coverage = ReviewCoverage.from_dict(json.loads(DeepArtifact.REVIEW_COVERAGE.at(deep).read_text()))
    coverage.record_phase("merge", "complete", noop=True)
    persist_review_coverage(deep, coverage)
    gate_p = deep / "fix-quality-gate.json"
    gate_p.write_text("[]")

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    _install_stub_backend(monkeypatch, target)

    warnings = _capture_warnings(monkeypatch, "daydream.deep.quality_gate.print_warning")
    exit_code = await run(make_config(target, start_at="fix", assume="yes", output_mode="loop", non_interactive=False))
    assert exit_code == 0
    assert any("fix-quality-gate.json" in w and "malformed" in w for w in warnings), (
        "a malformed artifact must surface a warning, never fail silently"
    )
    gate = json.loads(gate_p.read_text(encoding="utf-8"))
    assert gate["enabled"] is True
    assert gate["session_id"]
    assert len(gate["rounds"]) == 1

async def test_fix_quality_gate_second_run_discards_prior_session_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:

    target = _build_gate_target(tmp_path, "gate_session_resume")
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    stub = _install_stub_backend(monkeypatch, target)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    stub.fix_edit_line = _FIX_EDIT_VERBOSE
    warnings = _capture_warnings(monkeypatch, "daydream.deep.quality_gate.print_warning")
    first = await run(make_config(target, assume="yes", output_mode="loop", non_interactive=False))
    assert first == 0
    first_gate = _read_quality_gate(target)
    assert first_gate["session_id"]
    assert len(first_gate["rounds"]) == 1

    # Restore the tree to the clean pre-fix state (run 1's fix was not
    # committed) so the resume's worktree-clean + diff-key gates pass; the
    # daydream artifacts -- including run 1's gate artifact -- survive.
    _git(target, "reset", "--hard", "HEAD")
    for sentinel in target.glob(".fixed-*"):
        sentinel.unlink()
    (target / ".daydream-fix-applied").unlink(missing_ok=True)

    second = await run(make_config(target, start_at="fix", assume="yes", output_mode="loop", non_interactive=False))
    assert second == 0
    second_gate = _read_quality_gate(target)
    assert second_gate["session_id"] != first_gate["session_id"]
    assert len(second_gate["rounds"]) == 1, (
        "a new session must not inherit a prior session's rounds; run 1's round must be discarded"
    )
    assert any("another run's session" in w for w in warnings), (
        "a session-mismatched artifact must surface a warning, never silently merge rounds"
    )

async def test_fix_quality_gate_covers_authorized_secondary_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:

    target = _build_gate_target_with_helper(tmp_path, "gate_secondary_edit")
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    stub = _ExtraEditBackend(target, target / "helper.py", _FIX_EDIT_VERBOSE)
    # Related-file authority belongs to the synthesis contract, rather than
    # the reviewer's stricter per-stack record envelope.
    stub.parse_by_stack = {"python": {"severity": "high", "confidence": "MEDIUM"}}
    stub.merge_items = [{**_merge_item(1, "api.py", "high"), "related_files": ["helper.py"]}]
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)

    exit_code = await run(make_config(target, assume="yes", output_mode="loop", non_interactive=False,
                                     shallow_fanout_threshold=0))
    assert exit_code == 0

    gate = _read_quality_gate(target)
    assert gate["enabled"] is True
    per_file = gate["rounds"][0]["per_file"]
    assert "helper.py" in per_file, "an authorized secondary file must still pass the quality gate"
    helper = per_file["helper.py"]
    assert helper["verbosity_delta"] is not None, "the secondary file's delta must be computed"
    assert helper["verbosity_after"] > helper["verbosity_before"]
    assert helper["flagged"] is True, "a secondary-file regression must be flagged"
    assert "api.py" in per_file, "a finding target must stay covered even when unchanged on disk"
    assert per_file["api.py"]["flagged"] is False

async def test_fix_quality_gate_scopes_analyzer_to_reviewed_python_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:

    target = _build_scope_creep_target(tmp_path, "gate_scoped")
    real_analyze = analyzer_mod.analyze_quality
    calls: list[set[str] | None] = []

    def _stub(daydream_dir: Any, candidate_paths: set[str] | None = None, **kwargs: Any,) -> dict[str, Any]:
        calls.append(candidate_paths)
        return real_analyze(daydream_dir, candidate_paths, **kwargs)

    monkeypatch.setattr(analyzer_mod, "analyze_quality", _stub)

    exit_code = await _run_quality_gate_fixture(target, monkeypatch, make_config, mute_side_effects)
    assert exit_code == 0
    # The two GATE captures come first; the archive step's ``analyze_session``
    # adds a trailing argument-free ``analyze_quality`` call (a pinned
    # invariant — standalone evaluation stays whole-workspace), so pin only
    # the first two calls.
    assert len(calls) >= 2, f"expected pre-fix + post-fix captures, got {calls}"
    assert calls[0] == {"api.py"}, f"pre-fix capture must be scoped to reviewed .py, got {calls[0]}"
    assert calls[1] == {"api.py"}, f"post-fix capture must be scoped to reviewed .py, got {calls[1]}"

async def test_fix_quality_gate_excludes_scrubbed_secondary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
) -> None:
    """An out-of-diff module created by the fixer is scrubbed before the quality gate records candidates."""

    target = _build_scope_creep_target(tmp_path, "gate_missing_baseline")
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    mute_side_effects()
    stub = _ExtraEditBackend(target, target / "new_module.py", "def extra():\n    return 1\n", append=False)
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    _capture_warnings(monkeypatch, "daydream.deep.quality_gate.print_warning")

    exit_code = await run(make_config(target, assume="yes", output_mode="loop", non_interactive=False,
            file_config=DaydreamFileConfig(quality_gate_erosion_absolute=100.0, quality_gate_verbosity_absolute=100.0,
            ),
        )
    )
    assert exit_code == 0

    gate = _read_quality_gate(target)
    assert gate["enabled"] is True
    per_file = gate["rounds"][0]["per_file"]
    assert "new_module.py" not in per_file
    assert not (target / "new_module.py").exists()
    # The reviewed file stays covered and clean.
    assert per_file["api.py"]["flagged"] is False
