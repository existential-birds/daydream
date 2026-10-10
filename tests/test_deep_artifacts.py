"""Deep-mode artifact path + check_deep_artifacts tests (D-18, D-36, D-37)."""
import os
from pathlib import Path

import pytest

from daydream.deep.artifacts import (
    check_deep_artifacts,
)


@pytest.fixture
def deep_artifacts_dir(tmp_path: Path) -> Path:
    """The ``.daydream/deep`` directory the walk-back gates read."""
    path = tmp_path / ".daydream" / "deep"
    path.mkdir(parents=True)
    return path

def test_check_deep_artifacts_missing(deep_artifacts_dir: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("per-stack", deep_artifacts_dir)
    assert "intent.md" in str(excinfo.value)
    assert "--start-at" in str(excinfo.value)

def test_check_deep_artifacts_merge_requires_records(deep_artifacts_dir: Path) -> None:
    (deep_artifacts_dir / "intent.md").write_text("x")
    (deep_artifacts_dir / "alternatives.json").write_text("[]")
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("merge", deep_artifacts_dir,
                             record_paths=[deep_artifacts_dir / "stack-python-records.json"])
    assert "stack-*-records.json" in str(excinfo.value)

def test_check_deep_artifacts_rejects_directory_shadowing_prereq(deep_artifacts_dir: Path,) -> None:
    # intent.md exists as a directory, not a file.
    (deep_artifacts_dir / "intent.md").mkdir()
    (deep_artifacts_dir / "alternatives.json").write_text("[]")
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("per-stack", deep_artifacts_dir)
    assert "intent.md" in str(excinfo.value)

def test_check_deep_artifacts_fix_rejects_directory_merged_items(deep_artifacts_dir: Path,) -> None:
    """The fix gate requires a canonical merged-items.json file; directories and rendered reports cannot satisfy it."""
    (deep_artifacts_dir / "merged-items.json").mkdir()  # directory, not a file
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("fix", deep_artifacts_dir)
    assert "merged-items.json" in str(excinfo.value)

# --- Diff-freshness gate for --start-at resume -------------------------------

def _seed_merge_stage(deep_dir: Path) -> None:
    """Everything ``check_deep_artifacts("merge", ...)`` requires, minus the key."""
    deep_dir.mkdir(parents=True, exist_ok=True)
    (deep_dir / "intent.md").write_text("x")
    (deep_dir / "alternatives.json").write_text("[]")
    (deep_dir / "stack-python-records.json").write_text("[]")

def test_check_deep_artifacts_rejects_mismatched_diff_key(deep_artifacts_dir: Path) -> None:
    _seed_merge_stage(deep_artifacts_dir)
    (deep_artifacts_dir / "diff-key").write_text("aaaa")
    with pytest.raises(FileNotFoundError, match="different diff"):
        check_deep_artifacts("merge", deep_artifacts_dir,
                             record_paths=[deep_artifacts_dir / "stack-python-records.json"], current_diff_sha="bbbb")

def test_check_deep_artifacts_rejects_prerequisites_older_than_matching_key(deep_artifacts_dir: Path,) -> None:
    """A matching key cannot validate artifacts left from an earlier run."""
    _seed_merge_stage(deep_artifacts_dir)
    key_file = deep_artifacts_dir / "diff-key"
    key_file.write_text("bbbb")
    key_mtime = key_file.stat().st_mtime_ns + 1_000_000_000
    os.utime(key_file, ns=(key_mtime, key_mtime))
    with pytest.raises(FileNotFoundError, match="different diff"):
        check_deep_artifacts("merge", deep_artifacts_dir,
                             record_paths=[deep_artifacts_dir / "stack-python-records.json"], current_diff_sha="bbbb")

def test_check_deep_artifacts_without_sha_skips_the_freshness_gate(deep_artifacts_dir: Path,) -> None:
    """Omitting current_diff_sha preserves the presence-only behavior."""
    _seed_merge_stage(deep_artifacts_dir)
    check_deep_artifacts("merge", deep_artifacts_dir,
                             record_paths=[deep_artifacts_dir / "stack-python-records.json"])  # must not raise
