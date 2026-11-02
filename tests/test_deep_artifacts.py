"""Deep-mode artifact path + check_deep_artifacts tests (D-18, D-36, D-37)."""
import os
from pathlib import Path

import pytest

from daydream.deep import artifacts
from daydream.deep.artifacts import (
    DeepArtifact,
    check_deep_artifacts,
)


@pytest.fixture
def deep_artifacts_dir(tmp_path: Path) -> Path:
    """The ``.daydream/deep`` directory the walk-back gates read."""
    path = tmp_path / ".daydream" / "deep"
    path.mkdir(parents=True)
    return path

def test_check_deep_artifacts_merge_requires_records(deep_artifacts_dir: Path) -> None:
    (deep_artifacts_dir / "intent.md").write_text("x")
    (deep_artifacts_dir / "alternatives.json").write_text("[]")
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("merge", deep_artifacts_dir,
                             record_paths=[deep_artifacts_dir / "stack-python-records.json"])
    assert "stack-*-records.json" in str(excinfo.value)

# --- Diff-freshness gate for --start-at resume -------------------------------

def _seed_merge_stage(deep_dir: Path) -> None:
    """Everything ``check_deep_artifacts("merge", ...)`` requires, minus the key."""
    deep_dir.mkdir(parents=True, exist_ok=True)
    (deep_dir / "intent.md").write_text("x")
    (deep_dir / "alternatives.json").write_text("[]")
    (deep_dir / "stack-python-records.json").write_text("[]")

def _seed_fresh_resume_artifacts(deep_dir: Path) -> int:
    """Seed every resume stage with artifacts exactly as new as its matching key."""
    _seed_merge_stage(deep_dir)
    (deep_dir / "merged-items.json").write_text('{"items": []}')
    key_file = deep_dir / "diff-key"
    key_file.write_text("bbbb")
    key_mtime = 1_700_000_000_000_000_000
    for path in deep_dir.iterdir():
        os.utime(path, ns=(key_mtime, key_mtime))
    return key_mtime


@pytest.mark.parametrize(('stage', 'earlier'), [
    ("fix", "merge"),
])
def test_check_deep_artifacts_added_prerequisite_requires_presence_and_freshness(
    deep_artifacts_dir: Path, monkeypatch: pytest.MonkeyPatch, stage: str, earlier: str,
) -> None:
    """Declaring a predecessor must gate resume without a second freshness inventory."""
    key_mtime = _seed_fresh_resume_artifacts(deep_artifacts_dir)
    monkeypatch.setitem(
        artifacts._DEEP_STAGE_PREREQS, stage, [*artifacts._DEEP_STAGE_PREREQS[stage], DeepArtifact.DIAGRAM],
    )
    prerequisite = deep_artifacts_dir / "diagram.json"
    record_paths = [deep_artifacts_dir / "stack-python-records.json"]
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts(stage, deep_artifacts_dir, record_paths=record_paths, current_diff_sha="bbbb")
    assert str(excinfo.value) == (
        f"Cannot resume at stage '{stage}' -- missing artifacts:\n\n"
        f"  - {prerequisite}\n\n"
        f"Re-run from an earlier stage:\n"
        f"  daydream --start-at {earlier}"
    )

    prerequisite.write_text("{}")
    os.utime(prerequisite, ns=(key_mtime - 1_000_000_000, key_mtime - 1_000_000_000))
    with pytest.raises(FileNotFoundError, match="prerequisite artifacts predate"):
        check_deep_artifacts(stage, deep_artifacts_dir, record_paths=record_paths, current_diff_sha="bbbb")

    os.utime(prerequisite, ns=(key_mtime, key_mtime))
    check_deep_artifacts(stage, deep_artifacts_dir, record_paths=record_paths, current_diff_sha="bbbb")

def test_check_deep_artifacts_without_sha_skips_the_freshness_gate(deep_artifacts_dir: Path,) -> None:
    """Omitting current_diff_sha preserves the presence-only behavior."""
    _seed_merge_stage(deep_artifacts_dir)
    check_deep_artifacts("merge", deep_artifacts_dir,
                             record_paths=[deep_artifacts_dir / "stack-python-records.json"])  # must not raise
