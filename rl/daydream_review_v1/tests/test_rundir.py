"""Boundary guard: the run-dir collector never forwards golden trajectory payloads."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_rewards import _stage_run
from verifiers.v1.runtimes.subprocess import SubprocessRuntime

from daydream_review_v1.rundir import RUN_DIR_FILES, fetch_run_dir

#: The RUN_DIR_FILES allowlist members the projection must include for the
#: golden run. The deep/stack-*-records.json glob members are collected too,
#: but their names depend on which stacks the router detected in the golden
#: run, so they are checked structurally (glob shape) in the test body instead
#: of pinned here. trajectory.json and any per-fork trajectories/*.json are
#: deliberately NOT in the projection — untrusted, model-directed, test-only
#: data that must never reach model context through the collector.
_REQUIRED_SCORING_FILES: frozenset[str] = frozenset(
    {
        "manifest.json",
        "review-output.md",
        "deep/review-output.md",
        "deep/recommendation-verdicts.json",
        "deep/merged-items.json",
        "deep/test-verdict.json",
    }
)


async def test_fetch_run_dir_excludes_fixture_trajectories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: SubprocessRuntime,
    rundir_golden: Path,
) -> None:
    """The collector's projection excludes trajectory payloads from the golden run.

    Stage the real golden archive, install a guarded read that raises on any
    trajectory path, and prove fetch_run_dir forwards only allowlist files plus
    deep/stack-*-records.json glob members — so captured operational text can
    never be forwarded into model context through the run-dir collector.
    """
    archive = tmp_path / "archive"
    staged = _stage_run(archive, rundir_golden, session_id="session-1")
    assert staged.name == "session-1"

    # The sole remaining untrusted trajectory shape in the fixture is staged.
    assert (staged / "trajectory.json").is_file()

    saved_read = runtime.read

    async def guarded_read(path: str) -> bytes:
        rel = Path(path).relative_to(str(staged)).as_posix()
        if rel == "trajectory.json" or rel.startswith("trajectories/"):
            raise AssertionError(f"trajectory path forwarded to collector: {rel}")
        return await saved_read(path)

    monkeypatch.setattr(runtime, "read", guarded_read)

    selected = await fetch_run_dir(runtime, tmp_path / "selected", archive_root=str(archive))

    assert selected is not None
    projected = {p.relative_to(selected).as_posix() for p in selected.rglob("*") if p.is_file()}
    # Every projected member must be an allowlist file or a stack-record glob
    # member, and all the required allowlist members must be present. Exact set
    # equality is deliberately avoided: the stack-record names depend on which
    # stacks the router detected for this golden run, so a different detection
    # would fail the build for reasons unrelated to the exclusion guard (same
    # convention as tests.test_rewards::test_rundir_golden_user_messages_are_inert).
    for rel in projected:
        assert rel in RUN_DIR_FILES or (
            rel.startswith("deep/stack-") and rel.endswith("-records.json")
        ), rel
    assert _REQUIRED_SCORING_FILES <= projected
    assert not (selected / "trajectory.json").exists()
    assert not (selected / "trajectories").exists()


async def test_verify_seal_fails_closed_when_diff_cannot_be_re_derived(
    tmp_path, runtime, rundir_golden, corpus_mini_dir, fixture_manifest_path,
) -> None:
    """verify_seal must fail closed when the candidate diff cannot be re-derived.

    A git failure at verify time must return False, never hash b"" and verify
    True on a run whose diff was never re-derived (the empty-diff collision).
    This is a focused unit guard on verify_seal, complementing the scoring-level
    test_git_failure_at_verify_time_fails_closed.
    """
    from daydream_review_v1.rundir import RUN_DIR_FILES, verify_seal
    from daydream_review_v1.verifier import seal_artifacts
    from test_rewards import _stage_run, _task

    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)

    present = [
        run_dir / rel for rel in RUN_DIR_FILES if rel != "seal.json" and (run_dir / rel).is_file()
    ] + sorted(run_dir.glob("deep/stack-*-records.json"))
    # A seal produced while git failed at seal time sealed the empty diff.
    seal = seal_artifacts(present, candidate_diff=b"")
    (run_dir / "seal.json").write_text(seal.model_dump_json(), encoding="utf-8")

    # The repo under review is not a git repository: git diff fails at verify
    # time with a non-zero exit, exactly the empty-diff collision.
    ok = await verify_seal(run_dir, runtime, str(tmp_path / "not-a-repo"), task.data.head_sha,
                           seal_expected=True)
    assert ok is False
