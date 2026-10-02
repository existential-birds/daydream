"""Boundary guard: the run-dir collector never forwards golden trajectory payloads."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_rewards import _stage_run, _task
from verifiers.v1.runtimes.subprocess import SubprocessRuntime

from daydream_review.rundir import RUN_DIR_FILES, fetch_run_dir, verify_seal
from daydream_review.verifier import seal_artifacts

# Forward only the exact allowlist and stack-record glob. Trajectories contain untrusted
# model-directed data and must never reach model context through this collector.
_REQUIRED_SCORING_FILES: frozenset[str] = frozenset({
        "manifest.json", "review-output.md", "deep/review-output.md", "deep/recommendation-verdicts.json",
        "deep/merged-items.json", "deep/test-verdict.json",
    }
)


async def test_fetch_run_dir_excludes_fixture_trajectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: SubprocessRuntime, rundir_golden: Path,
) -> None:
    """A guarded read rejects every trajectory path. The real golden archive may forward only allowlisted
    files and stack-record globs into scoring.
    """
    archive = tmp_path / "archive"
    staged = _stage_run(archive, rundir_golden, session_id="session-1")
    assert staged.name == "session-1"

    assert (staged / "trajectory.json").is_file()

    saved_read = runtime.read

    async def guarded_read(path: str) -> bytes:
        rel = Path(path).relative_to(str(staged)).as_posix()
        if rel == "trajectory.json" or rel.startswith("trajectories/"):
            raise AssertionError(f"trajectory path forwarded to collector: {rel}")
        return bytes(await saved_read(path))

    monkeypatch.setattr(runtime, "read", guarded_read)

    selected = await fetch_run_dir(runtime, tmp_path / "selected", archive_root=str(archive))

    assert selected is not None
    projected = {p.relative_to(selected).as_posix() for p in selected.rglob("*") if p.is_file()}
    # Require every allowlist member and reject unrelated files. Stack-record names vary with router
    # detection, so exact set equality would couple this guard to stack discovery.
    for rel in projected:
        assert rel in RUN_DIR_FILES or (rel.startswith("deep/stack-") and rel.endswith("-records.json")
        ), rel
    assert _REQUIRED_SCORING_FILES <= projected
    assert not (selected / "trajectory.json").exists()
    assert not (selected / "trajectories").exists()

async def test_verify_seal_fails_closed_when_diff_cannot_be_re_derived(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """A failed diff re-derivation must not verify by hashing empty bytes; that would collide with an empty
    sealed diff.
    """
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    task = _task(fixture_manifest_path)

    present = [run_dir / rel for rel in RUN_DIR_FILES if rel != "seal.json" and (run_dir / rel).is_file()
    ] + sorted(run_dir.glob("deep/stack-*-records.json"))
    # A seal produced while git failed at seal time sealed the empty diff.
    seal = seal_artifacts(present, candidate_diff=b"")
    (run_dir / "seal.json").write_text(seal.model_dump_json(), encoding="utf-8")

    # The repo under review is not a git repository: git diff fails at verify
    # time with a non-zero exit, exactly the empty-diff collision.
    ok = await verify_seal(run_dir, runtime, str(tmp_path / "not-a-repo"), task.data.head_sha, seal_expected=True)
    assert ok is False
