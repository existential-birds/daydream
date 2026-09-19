"""Findings Artifacts And Sweep."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.runner import run
from tests.deep_orchestrator.support import (
    _eroded_main_repo,
    _install_uncovered_sweep_stub,
    _silence_gate_noise,
    _uncovered_sweep_target,
)
from tests.harness.git_helpers import git as _git
from tests.test_deep_orchestrator import (
    MakeConfig,
    Mute,
    _add_bare_remote,
    _force_interactive,
    _install_stub_backend,
    _pin_findings_pr,
    _run_deep,
    _silence,
    _StubBackend,
)


async def _post_forbidden(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("--findings-out must not post to the PR")


def _matching_prompt(calls: list[dict[str, Any]], fragment: str) -> str:
    return cast(str, next(call["prompt"] for call in calls if fragment in call["prompt"].lower()))


async def test_deep_findings_out_emits_artifact_and_stops(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
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
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """Real-path: ``--findings-out --cleanup`` keeps ``.review-output.md``."""
    from daydream.config import REVIEW_OUTPUT_FILE

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
    assert report.exists(), (
        "--findings-out --cleanup must keep .review-output.md: the run exits 0 but "
        "was asked to emit the report, so cleanup must not delete it"
    )


async def test_test_verdict_artifact_written_on_passing_suite(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real-path: a run whose suite passes leaves ``test-verdict.json`` on disk."""

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")
    monkeypatch.setattr("daydream.remote_ci.platform.system", lambda: "Darwin")
    monkeypatch.setattr("daydream.remote_ci.platform.release", lambda: "25.1.0")
    monkeypatch.setattr("daydream.remote_ci.platform.machine", lambda: "arm64")
    monkeypatch.setattr("daydream.remote_ci.platform.python_implementation", lambda: "CPython")
    monkeypatch.setattr("daydream.remote_ci.platform.python_version", lambda: "3.13.7")
    _install_stub_backend(monkeypatch, tiny_diff_target)
    mute_side_effects(heal=False)

    rc = await run(make_config(tiny_diff_target, assume="yes", non_interactive=False))
    assert rc == 0

    verdict_file = tiny_diff_target / ".daydream" / "deep" / "test-verdict.json"
    assert verdict_file.is_file(), "passing run did not write test-verdict.json"
    verdict = json.loads(verdict_file.read_text())
    assert verdict["passed"] is True, verdict
    assert verdict["retries"] == 0, "a green suite must not have consumed a heal retry"
    assert verdict["local_host"] == {
        "system": "Darwin",
        "release": "25.1.0",
        "machine": "arm64",
        "python_implementation": "CPython",
        "python_version": "3.13.7",
    }
    assert "Linux" not in json.dumps(verdict)
    assert "coverage" not in json.dumps(verdict).lower()


async def test_test_verdict_artifact_written_on_failing_suite(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real-path: a permanently-red suite STILL leaves ``test-verdict.json``."""

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")
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


async def test_test_verdict_records_failure_when_operator_ignores_it(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """Real-path: heal-menu choice "3" continues the run WITHOUT claiming a green suite."""

    _silence(monkeypatch, prompts=False)
    _force_interactive(monkeypatch)

    # The single gateway accepts intent and chooses ignore in the heal menu.
    def _prompt(_console: Any, message: str, _default: str = "") -> str:
        return "3" if "Choice" in message else "y"

    monkeypatch.setattr("daydream.run_context._prompt_user", _prompt)

    stub = _StubBackend(tiny_diff_target)
    _add_bare_remote(tiny_diff_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    stub.fail_all_test_runs = True

    mute_side_effects(heal=False, commit=False)

    head_before = _git(tiny_diff_target, "rev-parse", "HEAD")

    rc = await run(make_config(tiny_diff_target, non_interactive=False))
    assert rc == 0, "choice '3' must continue the run, not abort it"

    verdict_file = tiny_diff_target / ".daydream" / "deep" / "test-verdict.json"
    verdict = json.loads(verdict_file.read_text())
    assert verdict["passed"] is False, f"an ignored failure was persisted as a green suite: {verdict}"
    assert verdict["ignored"] is True, f"the operator override was not recorded: {verdict}"
    assert _git(tiny_diff_target, "rev-parse", "HEAD") == head_before
    assert not (tiny_diff_target / ".fixed-api_py").exists()
    assert stub.test_suite_calls == 1, f"expected one test-suite run before the ignore, saw {stub.test_suite_calls}"


async def test_deep_run_inlines_small_diff_into_intent_and_wonder(
    tiny_diff_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real path: a small diff is inlined into BOTH the intent and wonder prompts."""
    stub = _install_stub_backend(monkeypatch, tiny_diff_target)

    assert await _run_deep(tiny_diff_target) == 0

    intent_prompt = _matching_prompt(stub.calls, "understand the intent of these changes")
    wonder_prompt = _matching_prompt(stub.calls, "evaluate the implementation")

    for name, prompt in (("intent", intent_prompt), ("wonder", wonder_prompt)):
        assert "diff --git" in prompt, f"{name} prompt did not inline the diff"
        assert "do NOT re-Read" in prompt, f"{name} prompt kept the read instruction"
    assert "Read the diff file at" not in intent_prompt


async def test_deep_run_keeps_pointer_when_diff_exceeds_budget(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-budget diff falls back to today's diff.patch pointer in both prompts."""
    import daydream.deep.orchestrator as orch_mod
    from daydream.deep.prompts import bound_deep_diff
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES

    # Push the diff over the byte budget with a large committed file.
    big = "\n".join(f"line {i} of filler content" for i in range(INLINE_DIFF_BUDGET_BYTES // 10))
    (multi_stack_target / "0big.py").write_text(big + "\n")
    _git(multi_stack_target, "add", "0big.py")
    _git(multi_stack_target, "commit", "-m", "add big file")

    bounded_results: list[str] = []
    real = bound_deep_diff

    def spy(diff: str, budget: int = INLINE_DIFF_BUDGET_BYTES) -> Any:
        result = real(diff, budget)
        bounded_results.append(result[0])
        return result

    monkeypatch.setattr(orch_mod, "bound_deep_diff", spy)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target) == 0

    # The pointer fallback only discriminates when 0big.py's block really sorts
    # FIRST in git's byte-ordered diff AND the bound keeps it whole (leading
    # oversize rule): verify both, so the "not inlined" assertions below cannot
    # pass via an inline of small retained blocks instead of the pointer.
    patch = (multi_stack_target / ".daydream" / "diff.patch").read_text()
    assert patch.startswith("diff --git a/0big.py b/0big.py"), (
        "0big.py must be the FIRST block in git's byte-ordered diff, "
        f"got {patch.splitlines()[0] if patch else '<empty patch>'!r}"
    )
    assert len(bounded_results) == 1, "gather must bound the diff exactly once"
    assert "line 500 of filler content" in bounded_results[0], "the bound must keep 0big.py's oversize block whole"
    assert len(bounded_results[0].encode("utf-8")) > INLINE_DIFF_BUDGET_BYTES, (
        "the bounded diff must stay over the inline budget so the pointer fallback is exercised"
    )

    intent_prompt = _matching_prompt(stub.calls, "understand the intent of these changes")
    wonder_prompt = _matching_prompt(stub.calls, "evaluate the implementation")

    assert "Read the diff file at" in intent_prompt
    # The wonder pointer clause is "in the diff at {diff_path}"; the inline
    # clause ("do NOT re-Read {diff_path}") embeds the path too, so the bare
    # "diff.patch" substring is not discriminating.
    assert "in the diff at " in wonder_prompt
    for name, prompt in (("intent", intent_prompt), ("wonder", wonder_prompt)):
        assert "line 500 of filler content" not in prompt, f"{name} inlined an over-budget diff"


async def test_deep_run_keeps_pointer_when_trailing_block_dropped(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-file over-budget diff whose trailing block is dropped keeps the diff.patch pointer in the
    intent/wonder prompts."""
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES

    # aaa.py sorts FIRST in the byte-ordered git diff and its block is
    # retained; zzz.py's block alone exceeds the budget and arrives after a
    # retained block, so it is dropped whole.
    (multi_stack_target / "aaa.py").write_text("SMALL_RETAINED_MARKER = 1\n")
    big = "\n".join(f"line {i} of filler content" for i in range(INLINE_DIFF_BUDGET_BYTES // 10))
    (multi_stack_target / "zzz.py").write_text(big + "\n")
    _git(multi_stack_target, "add", "aaa.py", "zzz.py")
    _git(multi_stack_target, "commit", "-m", "add small and big files")

    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target) == 0

    intent_prompt = _matching_prompt(stub.calls, "understand the intent of these changes")
    wonder_prompt = _matching_prompt(stub.calls, "evaluate the implementation")

    assert "Read the diff file at" in intent_prompt
    assert "diff.patch" in wonder_prompt
    for name, prompt in (("intent", intent_prompt), ("wonder", wonder_prompt)):
        assert "SMALL_RETAINED_MARKER" not in prompt, f"{name} inlined the bounded diff"
        assert "line 500 of filler content" not in prompt, f"{name} inlined an over-budget diff"

    # The python stack owns aaa.py (retained by the bound) AND zzz.py (dropped
    # whole by the bound): the truncation marker names zzz.py as dropped, so
    # ``_diff_blocks_for_files`` refuses a partial inline and the python
    # per-stack prompt falls back to the diff.patch pointer instead of silently
    # inlining aaa.py's hunk with zzz.py's hunks unreachable. The react stack
    # (App.tsx only, fully retained) keeps its inline.
    python_prompt = _matching_prompt(stub.calls, "you are reviewing the python stack")
    assert "Read it directly" in python_prompt, (
        "a stack mixing retained and dropped blocks must fall back to the full diff.patch pointer"
    )
    assert "SMALL_RETAINED_MARKER" not in python_prompt, (
        "must not inline the retained block while the dropped block is missing"
    )
    react_prompt = _matching_prompt(stub.calls, "you are reviewing the react stack")
    assert "diff --git" in react_prompt, "a fully-retained stack must keep its inline hunks"


async def test_deep_run_bounds_in_memory_diff_but_keeps_diff_patch_full(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gather stores the BOUNDED diff in ctx.data; diff.patch on disk stays FULL."""
    import daydream.deep.orchestrator as orch_mod
    from daydream.deep.prompts import bound_deep_diff
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES

    big = "\n".join(f"line {i} of filler content" for i in range((INLINE_DIFF_BUDGET_BYTES // 10) + 50))
    (multi_stack_target / "big.py").write_text(big + "\n")
    _git(multi_stack_target, "add", "big.py")
    _git(multi_stack_target, "commit", "-m", "add big file")

    called_with: list[str] = []
    bounded_results: list[str] = []
    real = bound_deep_diff

    def spy(diff: str, budget: int = INLINE_DIFF_BUDGET_BYTES) -> Any:
        called_with.append(diff)
        result = real(diff, budget)
        bounded_results.append(result[0])
        return result

    monkeypatch.setattr(orch_mod, "bound_deep_diff", spy)
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target) == 0

    # (a) the helper was invoked at gather with the full diff (gather wiring).
    assert len(called_with) == 1 and called_with[0] == (multi_stack_target / ".daydream" / "diff.patch").read_text()
    # (b) diff.patch on disk is FULL: the big committed file's content survives.
    patch = (multi_stack_target / ".daydream" / "diff.patch").read_text()
    assert "line 50 of filler content" in patch  # a line far into big.py is present
    # (c) the helper's BOUNDED result -- not the full diff -- reaches
    # ctx.data['diff'] and the prompt pipeline. The TTT (intent/wonder) phases
    # deliberately re-read the FULL on-disk diff when truncation happened
    # (``_ttt_diff_text``), so the per-stack reviewer is the bounded value's
    # prompt consumer: big.py's block sorts LAST in git's byte-ordered diff,
    # so the bound drops it. The python stack owns api.py AND big.py, so its
    # wanted files span retained and dropped blocks: ``_diff_blocks_for_files``
    # reads the truncation marker's dropped names, refuses the partial inline,
    # and the python per-stack prompt carries the diff_path pointer instead of
    # api.py's hunk alone. A gather bug that invokes the helper and discards
    # the result (storing the full diff) would carry no marker and no dropped
    # names -- the mixing guard could not fire.
    assert len(bounded_results) == 1
    assert "# daydream: deep diff truncated:" in bounded_results[0]
    assert "line 50 of filler content" not in bounded_results[0]
    per_stack_prompt = _matching_prompt(stub.calls, "you are reviewing the python stack")
    assert "Read it directly" in per_stack_prompt, (
        "the python stack mixes retained (api.py) and dropped (big.py) blocks; "
        "it must fall back to the full diff.patch pointer, never inline a "
        "silent partial subset"
    )
    assert "diff --git" not in per_stack_prompt
    assert "line 50 of filler content" not in per_stack_prompt
    react_prompt = _matching_prompt(stub.calls, "you are reviewing the react stack")
    assert "diff --git" in react_prompt, "a fully-retained stack must keep its inline hunks"


async def test_uncovered_sweep_reads_full_diff_for_block_extraction(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep extracts blocks from the FULL diff.patch, not the bounded ctx.data['diff'], so sweep targets
    cannot diverge from the coverage set."""
    import daydream.deep.orchestrator as orch_mod
    from daydream.deep.prompts import bound_deep_diff
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES

    # Push the in-memory diff over the budget with a file that sorts FIRST in
    # git's byte-ordered diff: its oversize block is kept whole by the bound,
    # so every later block -- including the small uncovered file below -- is
    # dropped from ctx.data['diff'] but survives in the on-disk diff.patch.
    big = "\n".join(f"line {i} of filler content" for i in range((INLINE_DIFF_BUDGET_BYTES // 10) + 50))
    (multi_stack_target / "0big.py").write_text(big + "\n")
    (multi_stack_target / "zzuncovered.py").write_text("".join(f"content line {i}\n" for i in range(6)))
    _git(multi_stack_target, "add", "0big.py", "zzuncovered.py")
    _git(multi_stack_target, "commit", "-m", "add big file and an uncovered file")

    bounded_results: list[str] = []
    real = bound_deep_diff

    def spy(diff: str, budget: int = INLINE_DIFF_BUDGET_BYTES) -> Any:
        result = real(diff, budget)
        bounded_results.append(result[0])
        return result

    monkeypatch.setattr(orch_mod, "bound_deep_diff", spy)
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    # Per-stack reviewers read their scope files; zzuncovered.py is deliberately
    # unread so it is the sweep's single uncovered target.
    stub.per_stack_emit_reads = True
    stub.per_stack_unread = frozenset({"zzuncovered.py"})

    assert await _run_deep(multi_stack_target, uncovered_sweep=True) == 0

    # Prove the discriminating setup actually held: the bounded in-memory diff
    # dropped zzuncovered.py's block (truncation marker present, the file's
    # content absent) while the full on-disk diff.patch kept it. A sweep that
    # sourced ctx.data['diff'] would then find no block for the file and route
    # it into skipped_small; the fixed sweep sources diff.patch and sweeps it.
    assert len(bounded_results) == 1
    assert "# daydream: deep diff truncated:" in bounded_results[0]
    assert "content line 3" not in bounded_results[0]
    patch = (multi_stack_target / ".daydream" / "diff.patch").read_text()
    assert "content line 3" in patch

    stats = json.loads((multi_stack_target / ".daydream" / "deep" / "coverage-stats.json").read_text())
    assert "zzuncovered.py" in stats["attempted_files"]
    assert "zzuncovered.py" not in stats["sweep_skipped_small_hunks_files"]


async def test_intent_artifact_survives_wonder_failure(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """intent.md is on disk even when the wonder step dies."""
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
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relocating the writer leaves contents and ctx.data pointers unchanged."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    assert await _run_deep(multi_stack_target) == 0

    deep = multi_stack_target / ".daydream" / "deep"
    assert (deep / "intent.md").read_text().strip()
    assert json.loads((deep / "alternatives.json").read_text())


async def test_skip_tier_writes_empty_alternatives(tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both artifacts exist even when the diff is small enough to run wonder."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tiny_diff_target)

    assert await _run_deep(tiny_diff_target) == 0

    deep = tiny_diff_target / ".daydream" / "deep"
    assert (deep / "intent.md").read_text().strip()
    assert isinstance(json.loads((deep / "alternatives.json").read_text()), list)


def test_deep_flow_has_no_feedback_prefix() -> None:
    """M2: the registered deep flow has no feedback-only prefix."""
    from daydream.deep.orchestrator import STEPS

    names = {step.name for step in STEPS}
    assert not names & {"fetch-feedback", "parse-feedback", "fix-items", "commit-push", "respond-feedback"}


def test_no_feedback_mode_resolver() -> None:
    """M2: `_resolve_mode` cannot return `feedback`; no feedback runner exists."""
    from daydream.deep import orchestrator as deep_orchestrator
    from daydream.runner import RunConfig

    assert not hasattr(deep_orchestrator, "_run_feedback_flow")
    config = RunConfig(target="/tmp", pr_number=7)
    assert deep_orchestrator._resolve_mode(config) != "feedback"


def test_extension_api_version_is_six_and_alternatives_step_is_gone() -> None:
    from daydream.deep.orchestrator import STEPS
    from daydream.extensions.api import EXTENSION_API_VERSION

    assert EXTENSION_API_VERSION == 6
    names = [s.name for s in STEPS]
    assert "alternatives" not in names
    assert "per-stack-reviews" in names


@pytest.mark.parametrize("change", ["committed", "worktree", "missing-key"])
async def test_start_at_merge_refuses_stale_or_unverifiable_artifacts(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
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


async def test_fresh_run_discards_stale_deep_artifacts_before_writing_its_diff_key(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh run cannot certify a new diff key alongside old deep outputs."""
    _silence(monkeypatch)
    deep = multi_stack_target / ".daydream" / "deep"
    deep.mkdir(parents=True)
    stale = deep / "obsolete-artifact.txt"
    stale.write_text("stale")

    _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target) == 0
    assert not stale.exists()
    assert (deep / "diff-key").is_file()


async def test_start_at_merge_proceeds_when_the_diff_is_unchanged(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The freshness gate is not a blanket refusal: same diff still resumes."""
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target) == 0

    stub2 = _install_stub_backend(monkeypatch, multi_stack_target)
    assert await _run_deep(multi_stack_target, start_at="merge") == 0
    assert any("cross-stack merge agent" in c["prompt"].lower() for c in stub2.calls)


@pytest.mark.parametrize(
    ("stack", "description", "lens"),
    [
        pytest.param(
            "python",
            "extract the repeated --flag branch pairs into a focused callable",
            "per-stack",
            id="python",
        ),
        pytest.param(
            "structure",
            "structural: the --flag branch pairs grow main() past the extraction threshold",
            "structural",
            id="structure",
        ),
    ],
)
async def test_anti_slop_extraction_finding_keeps_medium_severity_through_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stack: str,
    description: str,
    lens: str,
) -> None:
    """Python and structural extraction findings keep their reported severity."""
    _silence(monkeypatch)
    project = _eroded_main_repo(tmp_path)
    stub = _install_stub_backend(monkeypatch, project)
    stub.parse_by_stack = {
        stack: {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "main.py",
            "line": 3,
            "description": description,
        }
    }

    assert await _run_deep(project) == 0
    items = json.loads((project / ".daydream" / "deep" / "merged-items.json").read_text())["items"]
    matching = [item for item in items if item.get("description") == description]
    assert len(matching) == 1, items
    assert matching[0]["severity"] == "medium"
    assert matching[0]["lens"] == lens


async def test_run_deep_uncovered_sweep_merges_and_improves_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC (issue #309): the sweep reviews an uncovered file, its finding is an ordinary merged finding, coverage
    stats improve, and the report surfaces coverage."""
    from daydream.eval.analyzer import analyze_coverage, load_trajectories

    target = _uncovered_sweep_target(tmp_path)
    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_uncovered_sweep_stub(monkeypatch, target)
    stub.sweep_file = "notes.txt"
    stub.merge_echo_records = True

    exit_code = await run(make_config(target, assume="yes", output_mode="loop"))
    assert exit_code == 0

    deep = target / ".daydream" / "deep"

    # (a) The sweep records file exists and its finding is a MERGED finding.
    records_file = deep / "stack-uncovered-records.json"
    assert records_file.is_file()
    records = json.loads(records_file.read_text())
    assert any(r.get("file") == "notes.txt" for r in records)
    merged_items = json.loads((deep / "merged-items.json").read_text())
    merged_files = {item.get("file") for item in merged_items["items"]}
    assert "notes.txt" in merged_files

    # (b) coverage-stats records the PRE-sweep state separately from the
    # POST-sweep recompute, and labels the swept files it actually completed.
    stats = json.loads((deep / "coverage-stats.json").read_text())
    pre_sweep = stats["pre_sweep"]
    assert pre_sweep["files_in_diff"] == 4
    assert pre_sweep["files_read_by_reviewers"] == 3  # api.py, App.tsx, README.md read pre-sweep
    assert pre_sweep["uncovered_files"] == ["notes.txt"]
    assert stats["attempted_files"] == ["notes.txt"]
    assert stats["completed_files"] == ["notes.txt"]
    assert stats["covered_files"] == ["notes.txt"]  # sweep fork's read is verified
    assert stats["sweep_attempt_status"] == {"notes.txt": "read"}
    assert stats["sweep_finding_count"] == len(records) >= 1
    assert stats["sweep_skipped_small_hunks"] == 0
    # Finding 10: the skip filename lists are persisted alongside the counts
    # (derived from the lists) so capacity/hunk skips stay auditable.
    assert stats["sweep_skipped_small_hunks_files"] == []
    assert stats["sweep_skipped_capacity"] == 0
    assert stats["sweep_skipped_capacity_files"] == []
    # The POST-sweep ratio reflects the sweep fork's completed read of notes.txt.
    assert stats["post_sweep"]["files_read_by_reviewers"] == 4
    assert stats["post_sweep"]["coverage_ratio"] > pre_sweep["coverage_ratio"]  # 0.75 -> 1.0

    # (c) post-run analyze_coverage sees the sweep fork's read: ratio improves.
    trajectories = load_trajectories(target / ".daydream")
    post = analyze_coverage(trajectories, target / ".daydream")
    assert post["files_read_by_reviewers"] == 4
    assert post["coverage_ratio"] == 1.0
    assert post["coverage_ratio"] == stats["post_sweep"]["coverage_ratio"]  # report shows the achieved ratio

    # (d) the merged report carries the Coverage section with the POST-sweep
    # ratio and the completed swept-file line.
    report = (target / ".review-output.md").read_text()
    assert "## Coverage" in report
    assert "Files in diff: 4" in report
    assert "Files read by reviewers: 4" in report
    assert "Coverage ratio: 1.0" in report
    assert "Second-pass sweep covered: notes.txt" in report
