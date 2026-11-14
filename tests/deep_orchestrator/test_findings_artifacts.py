"""Findings Artifacts And Sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.config import REVIEW_OUTPUT_FILE
from daydream.runner import run
from tests.harness.execution import host_test_command
from tests.harness.git_helpers import git as _git
from tests.harness.review_profile import independent_alternatives_profile
from tests.test_deep_orchestrator import (
    MakeConfig,
    Mute,
    _add_bare_remote,
    _force_interactive,
    _install_stub_backend,
    _run_deep,
    _silence,
)


def _coverage_metadata_keys(value: object) -> set[str]:
    """Find coverage metadata fields without treating command/path strings as metrics."""
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and "coverage" in key.casefold():
                keys.add(key)
            keys.update(_coverage_metadata_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_coverage_metadata_keys(child))
    return keys


@pytest.mark.parametrize("passed,ignored,exit_code,retries", [
    pytest.param(False, True, 0, 0, id="operator-ignore-records-failure"),
])
async def test_embedded_fixing_is_selected_explicitly_and_persists_the_suite_verdict(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    passed: bool, ignored: bool, exit_code: int, retries: int,
) -> None:
    _silence(monkeypatch, prompts=False)
    _force_interactive(monkeypatch)
    asked: list[str] = []

    def answer(_console: object, message: str, *_args: object, **_kwargs: object) -> str:
        asked.append(message)
        return "3" if ignored and "Choice" in message else "y"

    monkeypatch.setattr("daydream.run_context._prompt_user", answer)
    host = {"system": "Darwin", "release": "25.1.0", "machine": "arm64",
            "python_implementation": "CPython", "python_version": "3.13.7"}
    for name, value in host.items():
        monkeypatch.setattr(f"daydream.remote_ci.artifacts.platform.{name}", lambda value=value: value)
    stub = _install_stub_backend(monkeypatch, tiny_diff_target)
    suite_dir = tiny_diff_target.parent / "Linux-coverage-suite"
    command = host_test_command(suite_dir, mode="pass" if passed else "all")
    if ignored:
        _add_bare_remote(tiny_diff_target)
    mute_side_effects(heal=False, commit=not ignored)
    head_before = _git(tiny_diff_target, "rev-parse", "HEAD")
    # A fixing request that could only end as a review is refused before any work.
    refusals: list[str] = []
    with monkeypatch.context() as patch:
        patch.setattr("daydream.runner.print_error", lambda _console, title, _message: refusals.append(title))
        patch.setattr("daydream.runner.print_phase_hero", lambda *_args: refusals.append("run started"))
        codes = [
            await run(make_config(tiny_diff_target, output_mode="loop", test_command=command, **unfixable))
            for unfixable in ({}, {"assume": "no", "non_interactive": False}, {"assume": "yes", "flow_name": "review"},
                              {"assume": "yes", "findings_out": str(suite_dir / "findings.json")})
        ]
    assert (codes, refusals, stub.calls) == ([1] * 4, ["Fix Not Possible"] * 4, [])
    # The default request ends with findings: this operator would accept a fix gate, and none is offered.
    code = await run(make_config(tiny_diff_target, non_interactive=False, test_command=command))
    assert (
        code, "### 1. Sample issue" in (tiny_diff_target / REVIEW_OUTPUT_FILE).read_text(),
        [question for question in asked if "Apply fixes" in question], (suite_dir / "suite-count.txt").exists(),
        _git(tiny_diff_target, "status", "--porcelain", "--untracked-files=no"),
    ) == (0, True, [], False, "")
    config = make_config(tiny_diff_target, output_mode="loop", assume=None if ignored else "yes",
                         non_interactive=False, test_command=command)
    assert await run(config) == exit_code
    path = tiny_diff_target / ".daydream" / "deep" / "test-verdict.json"
    assert path.is_file(), "suite verdict must survive both successful completion and an early failure return"
    verdict = json.loads(path.read_text())
    assert verdict["passed"] is passed and verdict["retries"] == retries and verdict["ignored"] is ignored
    assert verdict["local_host"] == host
    assert verdict["attempts"][0]["command"][-1] == str(suite_dir / "suite.py")
    assert not _coverage_metadata_keys(verdict), "host verdict must not claim unmeasured coverage"
    # The report is first produced before the fix/test cycle. Its final evidence
    # must include the real host execution without refinalizing review coverage.
    from daydream.redaction import redact_value
    context = json.loads((path.parent / 'review-report-context.json').read_text())
    recorded = context['verification']['value']['test-verdict.json']
    assert context['verification']['status'] == 'available'
    assert recorded['attempts'] == redact_value(verdict['attempts'])
    report = (tiny_diff_target / REVIEW_OUTPUT_FILE).read_text()
    attempt = recorded['attempts'][0]
    assert 'Checks actually executed by the recorded host runner' in report
    assert all(value in report for value in ('suite.py', attempt['identity']['outcome'],
                                            attempt['identity']['head_sha'], attempt['identity']['session_id']))
    assert 'Verification artifact availability: <code>available</code>' in report
    if ignored:
        assert _git(tiny_diff_target, "rev-parse", "HEAD") == head_before
        assert not (tiny_diff_target / ".fixed-api_py").exists()
        assert (suite_dir / "suite-count.txt").read_text() == "1"
        assert stub.test_suite_calls == 0


async def test_intent_artifact_survives_wonder_failure(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.fail_alternatives = True
    with pytest.raises(RuntimeError, match="alternatives blew up"):
        await _run_deep(multi_stack_target, review_profile=independent_alternatives_profile())
    intent_md = multi_stack_target / ".daydream" / "deep" / "intent.md"
    assert intent_md.read_text().strip(), "intent.md must survive the wonder failure"
    # The wonder half never ran, so its artifact is legitimately absent.
    assert not (multi_stack_target / ".daydream" / "deep" / "alternatives.json").exists()


def test_coverage_persistence_keeps_the_primary_failure_and_resume_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secondary evidence-write error annotates the original failure; foreign evidence never resumes.

    Runner resume tests only delete the coverage file; these owner checks bind the revision and
    target-decision gates and the primary-error rule without a review run.
    """
    from daydream.deep import artifacts
    from tests.harness.review_result import review_coverage

    stored = review_coverage(scope_ids=('python',))
    stored.record_scope('python', 'complete')

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError('coverage disk full')

    with monkeypatch.context() as patch:
        patch.setattr(artifacts, 'atomic_write_json', fail_write)
        primary = RuntimeError('original stage failure')
        artifacts.persist_review_coverage(tmp_path, stored, error=primary)
        assert primary.__notes__ == ['Review coverage persistence failed: OSError']
        with pytest.raises(OSError, match='coverage disk full'):
            artifacts.persist_review_coverage(tmp_path, stored)
    artifacts.persist_review_coverage(tmp_path, stored)
    restore = cast(Any, artifacts.restore_review_coverage)
    for current, message in ((review_coverage(scope_ids=('python',), head_sha='f' * 40), 'revision'),
                             (review_coverage(scope_ids=('python',)), 'target decisions are unavailable')):
        with pytest.raises(ValueError, match=message):
            restore(tmp_path, current, backend=None, work=None, stacks=[], registry=None,
                    artifact_session=None, allow_standalone=True)


def test_alternatives_prompt_points_at_the_diff_file_when_no_bounded_copy_is_inlined() -> None:
    """An over-budget diff is read from disk; the prompt never claims an inline copy it lacks.

    The runner journeys inline bounded diffs, so the file-pointer branch is otherwise unreached.
    """
    from daydream.phases.review_prompts import build_alternative_review_prompt

    strategy = 'Intent: {intent_summary}\nRead the diff file at {diff_path} and evaluate it.'
    prompt = build_alternative_review_prompt(strategy=strategy, intent_summary='Fix greeting', diff_path='d.patch')
    assert 'Intent: Fix greeting\nRead the diff file at d.patch and evaluate it.' in prompt
    assert 'inlined below' not in prompt and 'do NOT re-Read' not in prompt
