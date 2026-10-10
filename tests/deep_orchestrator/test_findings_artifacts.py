"""Findings Artifacts And Sweep."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

import daydream.deep.orchestrator as orch_mod
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.deep.diff import bound_deep_diff
from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES
from daydream.runner import run
from tests.deep_orchestrator.support import (
    _eroded_main_repo,
    _silence_gate_noise,
)
from tests.harness.git_helpers import git as _git
from tests.harness.review_profile import independent_alternatives_profile
from tests.test_deep_orchestrator import (
    MakeConfig,
    Mute,
    _add_bare_remote,
    _force_interactive,
    _install_stub_backend,
    _pin_findings_pr,
    _run_deep,
    _silence,
)


async def _post_forbidden(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("--findings-out must not post to the PR")


def _matching_prompt(calls: list[dict[str, Any]], fragment: str) -> str:
    return cast(str, next(call["prompt"] for call in calls if fragment in call["prompt"].lower()))


def _spy_bound_deep_diff(
    monkeypatch: pytest.MonkeyPatch, bounded_results: list[str], called_with: list[str] | None = None,
) -> None:
    """Patch ``orch_mod.bound_deep_diff`` to record each bounded result."""
    def spy(diff: str, budget: int = INLINE_DIFF_BUDGET_BYTES) -> Any:
        if called_with is not None:
            called_with.append(diff)
        result = bound_deep_diff(diff, budget)
        bounded_results.append(result[0])
        return result
    monkeypatch.setattr(orch_mod, "bound_deep_diff", spy)

async def test_deep_findings_out_emits_artifact_and_stops(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """Real-path: a deep run with ``--findings-out`` writes the PR-pinned findings artifact from the canonical
    merged items and STOPS -- no PR post, no fix."""

    _silence_gate_noise(monkeypatch)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    _install_stub_backend(monkeypatch, multi_stack_target)

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

async def test_cleanup_keeps_report_on_findings_out_run(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:

    _silence_gate_noise(monkeypatch)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    _install_stub_backend(monkeypatch, multi_stack_target)

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _post_forbidden)
    _pin_findings_pr(monkeypatch, multi_stack_target)

    out = multi_stack_target / "findings.json"
    report = multi_stack_target / REVIEW_OUTPUT_FILE
    rc = await run(make_config(multi_stack_target, pr_number=7, findings_out=str(out), cleanup=True))

    assert rc == 0
    assert out.exists(), "--findings-out must still write the findings artifact"
    assert report.exists(), ("--findings-out --cleanup must keep .review-output.md: the run exits 0 but "
        "was asked to emit the report, so cleanup must not delete it"
    )

@pytest.mark.parametrize("passed,ignored,exit_code,retries", [
    pytest.param(True, False, 0, 0, id="passing-suite"),
    pytest.param(False, False, 1, 1, id="permanently-red-bounded-retry"),
    pytest.param(False, True, 0, 0, id="operator-ignore-records-failure"),
])
async def test_test_verdict_persists_actual_suite_outcome_and_operator_override(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    passed: bool, ignored: bool, exit_code: int, retries: int,
) -> None:
    _silence(monkeypatch, prompts=False)
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.run_context._prompt_user",
                        lambda _console, message, *args, **kwargs: "3" if ignored and "Choice" in message else "y")
    host = {"system": "Darwin", "release": "25.1.0", "machine": "arm64",
            "python_implementation": "CPython", "python_version": "3.13.7"}
    for name, value in host.items():
        monkeypatch.setattr(f"daydream.remote_ci.artifacts.platform.{name}", lambda value=value: value)
    stub = _install_stub_backend(monkeypatch, tiny_diff_target)
    stub.fail_all_test_runs = not passed
    if ignored:
        _add_bare_remote(tiny_diff_target)
    mute_side_effects(heal=False, commit=not ignored)
    head_before = _git(tiny_diff_target, "rev-parse", "HEAD")
    config = make_config(tiny_diff_target, assume=None if ignored else "yes", non_interactive=False)
    assert await run(config) == exit_code
    path = tiny_diff_target / ".daydream" / "deep" / "test-verdict.json"
    assert path.is_file(), "suite verdict must survive both successful completion and an early failure return"
    verdict = json.loads(path.read_text())
    assert verdict["passed"] is passed and verdict["retries"] == retries and verdict["ignored"] is ignored
    assert verdict["local_host"] == host
    assert "Linux" not in json.dumps(verdict) and "coverage" not in json.dumps(verdict).lower()
    if ignored:
        assert _git(tiny_diff_target, "rev-parse", "HEAD") == head_before
        assert not (tiny_diff_target / ".fixed-api_py").exists()
        assert stub.test_suite_calls == 1



async def test_deep_run_inlines_small_diff_into_intent_and_wonder(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_stub_backend(monkeypatch, tiny_diff_target)
    assert await _run_deep(tiny_diff_target, review_profile=independent_alternatives_profile()) == 0
    intent_prompt = _matching_prompt(stub.calls, "understand the intent of these changes")
    wonder_prompt = _matching_prompt(stub.calls, "evaluate the implementation")
    for name, prompt in (("intent", intent_prompt), ("wonder", wonder_prompt)):
        assert "diff --git" in prompt, f"{name} prompt did not inline the diff"
        assert "do NOT re-Read" in prompt, f"{name} prompt kept the read instruction"
    assert "Read the diff file at" not in intent_prompt

@pytest.mark.parametrize("big_name,extra_lines,small_file,alternatives,oversize", [
    pytest.param("0big.py", 0, None, True, True, id="leading-oversize-block-retained"),
    pytest.param("zzz.py", 0, "aaa.py", True, False, id="trailing-block-dropped"),
    pytest.param("big.py", 50, None, False, False, id="bounded-memory-full-disk"),
])
async def test_over_budget_diff_preserves_full_disk_evidence_and_uses_safe_prompt_transport(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, big_name: str, extra_lines: int,
    small_file: str | None, alternatives: bool, oversize: bool,
) -> None:
    """Whole-block bounding never lets a prompt silently inline only part of its required files."""
    big = "\n".join(f"line {i} of filler content" for i in range(INLINE_DIFF_BUDGET_BYTES // 10 + extra_lines))
    (multi_stack_target / big_name).write_text(big + "\n")
    files = [big_name]
    if small_file is not None:
        (multi_stack_target / small_file).write_text("SMALL_RETAINED_MARKER = 1\n")
        files.append(small_file)
    _git(multi_stack_target, "add", *files)
    _git(multi_stack_target, "commit", "-m", "add over-budget diff")
    bounded_results: list[str] = []
    called_with: list[str] = []
    _spy_bound_deep_diff(monkeypatch, bounded_results, called_with)
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    profile = independent_alternatives_profile() if alternatives else None
    assert await _run_deep(multi_stack_target, review_profile=profile) == 0
    patch = (multi_stack_target / ".daydream" / "diff.patch").read_text()
    assert called_with == [patch] and len(bounded_results) == 1
    bounded = bounded_results[0]
    assert "line 50 of filler content" in patch
    assert ("line 50 of filler content" in bounded) is oversize
    assert ("line 500 of filler content" in bounded) is oversize
    assert (len(bounded.encode("utf-8")) > INLINE_DIFF_BUDGET_BYTES) is oversize
    if oversize:
        assert patch.startswith("diff --git a/0big.py b/0big.py")
    else:
        assert "# daydream: deep diff truncated:" in bounded
    if alternatives:
        intent = _matching_prompt(stub.calls, "understand the intent of these changes")
        wonder = _matching_prompt(stub.calls, "evaluate the implementation")
        assert "Read the diff file at" in intent and "in the diff at " in wonder and "diff.patch" in wonder
        for prompt in (intent, wonder):
            assert "line 500 of filler content" not in prompt and "SMALL_RETAINED_MARKER" not in prompt
    python_prompt = _matching_prompt(stub.calls, "you are reviewing the python stack")
    assert "Read it directly" in python_prompt
    assert "diff --git" not in python_prompt and "line 50 of filler content" not in python_prompt
    assert "SMALL_RETAINED_MARKER" not in python_prompt
    if not oversize:
        # React's complete retained block remains inline while Python spans dropped evidence.
        assert "diff --git" in _matching_prompt(stub.calls, "you are reviewing the react stack")



async def test_intent_artifact_survives_wonder_failure(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.fail_alternatives = True
    with pytest.raises(RuntimeError, match="alternatives blew up"):
        await _run_deep(multi_stack_target, review_profile=independent_alternatives_profile())
    intent_md = multi_stack_target / ".daydream" / "deep" / "intent.md"
    assert intent_md.read_text().strip(), "intent.md must survive the wonder failure"
    # The wonder half never ran, so its artifact is legitimately absent.
    assert not (multi_stack_target / ".daydream" / "deep" / "alternatives.json").exists()

async def test_skip_tier_writes_empty_alternatives(tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both artifacts exist even when the diff is small enough to run wonder."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tiny_diff_target)
    assert await _run_deep(tiny_diff_target) == 0
    deep = tiny_diff_target / ".daydream" / "deep"
    assert (deep / "intent.md").read_text().strip()
    assert isinstance(json.loads((deep / "alternatives.json").read_text()), list)

@pytest.mark.parametrize("change", ["committed", "worktree", "missing-key"])
async def test_start_at_merge_refuses_stale_or_unverifiable_artifacts(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    """A resume stops before backend calls when its diff identity cannot be trusted."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target) == 0

    key_file = multi_stack_target / ".daydream" / "deep" / "diff-key"
    original_key = key_file.read_text().strip()
    assert original_key
    if change == "missing-key":
        key_file.unlink()
    else:
        (multi_stack_target / "api.py").write_text("def hello():\n    return 'galaxy'\n")
        if change == "committed":
            _git(multi_stack_target, "add", "api.py")
            _git(multi_stack_target, "commit", "-m", "change again")

    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target, start_at="merge") == 1
    assert stub.calls == []
    if change != "missing-key":
        assert key_file.read_text().strip() == original_key

@pytest.mark.parametrize(("stack", "description", "lens"),
    [pytest.param(
            "python", "extract the repeated --flag branch pairs into a focused callable", "per-stack", id="python",
        ),
        pytest.param(
            "structure", "structural: the --flag branch pairs grow main() past the extraction threshold", "structural",
            id="structure",
        ),
    ],
)
async def test_anti_slop_extraction_finding_keeps_medium_severity_through_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stack: str, description: str, lens: str,
) -> None:
    _silence(monkeypatch)
    project = _eroded_main_repo(tmp_path)
    stub = _install_stub_backend(monkeypatch, project)
    stub.parse_by_stack = {stack: {
            "severity": "medium", "confidence": "MEDIUM", "file": "main.py", "line": 3, "description": description,
        }
    }

    assert await _run_deep(project) == 0
    items = json.loads((project / ".daydream" / "deep" / "merged-items.json").read_text())["items"]
    matching = [item for item in items if item.get("description") == description]
    assert len(matching) == 1, items
    assert matching[0]["severity"] == "medium"
    assert matching[0]["lens"] == lens
