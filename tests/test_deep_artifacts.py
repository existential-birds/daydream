"""Deep-mode artifact path + check_deep_artifacts tests (D-18, D-36, D-37)."""
import os
from pathlib import Path

import pytest

from daydream.config import REVIEW_OUTPUT_FILE
from daydream.deep import artifacts
from daydream.deep.artifacts import (
    check_deep_artifacts,
    deep_dir,
    diagram_markdown_path,
    diagram_path,
    diff_key,
    per_stack_review_path,
)


@pytest.fixture
def deep_artifacts_dir(tmp_path: Path) -> Path:
    """The ``.daydream/deep`` directory the walk-back gates read."""
    path = tmp_path / ".daydream" / "deep"
    path.mkdir(parents=True)
    return path


def test_deep_dir_uses_active_artifact_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    routed = tmp_path / "private" / ".daydream"
    monkeypatch.setattr(artifacts, "artifact_dir_for", lambda _target, **_kwargs: routed)

    assert artifacts.deep_dir(tmp_path / "model-cwd", allow_standalone=True) == routed / "deep"
    assert (routed / "deep").is_dir()


def test_per_stack_path_scheme(tmp_path: Path) -> None:
    """D-18: per-stack output path is deterministic + unique."""
    p1 = per_stack_review_path(tmp_path, "python")
    p2 = per_stack_review_path(tmp_path, "react")
    assert p1 != p2
    assert p1.name == "stack-python-review.md"
    assert p2.name == "stack-react-review.md"


def test_check_deep_artifacts_missing(deep_artifacts_dir: Path) -> None:
    """D-36: check_deep_artifacts raises FileNotFoundError when predecessor missing."""
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("per-stack", deep_artifacts_dir)
    assert "intent.md" in str(excinfo.value)
    assert "--start-at" in str(excinfo.value)


def test_check_deep_artifacts_merge_requires_records(deep_artifacts_dir: Path) -> None:
    """D-37: --start-at merge needs per-stack records on disk."""
    (deep_artifacts_dir / "intent.md").write_text("x")
    (deep_artifacts_dir / "alternatives.json").write_text("[]")
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("merge", deep_artifacts_dir)
    assert "stack-*-records.json" in str(excinfo.value)


def test_check_deep_artifacts_passes_when_present(deep_artifacts_dir: Path) -> None:
    """D-36: check passes silently when all predecessors exist."""
    (deep_artifacts_dir / "intent.md").write_text("x")
    (deep_artifacts_dir / "alternatives.json").write_text("[]")
    check_deep_artifacts("per-stack", deep_artifacts_dir)


def test_check_deep_artifacts_rejects_directory_shadowing_prereq(
    deep_artifacts_dir: Path,
) -> None:
    """A directory named like a prereq must not satisfy the gate."""
    # intent.md exists as a directory, not a file.
    (deep_artifacts_dir / "intent.md").mkdir()
    (deep_artifacts_dir / "alternatives.json").write_text("[]")
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("per-stack", deep_artifacts_dir)
    assert "intent.md" in str(excinfo.value)


def test_check_deep_artifacts_merge_ignores_directory_records(
    deep_artifacts_dir: Path,
) -> None:
    """A directory matching stack-*-records.json must not satisfy the merge gate."""
    (deep_artifacts_dir / "intent.md").write_text("x")
    (deep_artifacts_dir / "alternatives.json").write_text("[]")
    (deep_artifacts_dir / "stack-bogus-records.json").mkdir()  # directory, not a file
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("merge", deep_artifacts_dir)
    assert "stack-*-records.json" in str(excinfo.value)


def test_check_deep_artifacts_fix_rejects_directory_merged_items(
    deep_artifacts_dir: Path,
) -> None:
    """A directory named merged-items.json must not satisfy the fix gate.

    The fix gate keys on the canonical merged-items.json (the source of truth the
    fix loop reads), not the render-only review-output.md markdown.
    """
    (deep_artifacts_dir / "merged-items.json").mkdir()  # directory, not a file
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("fix", deep_artifacts_dir)
    assert "merged-items.json" in str(excinfo.value)


def test_check_deep_artifacts_fix_passes_with_json_only(deep_artifacts_dir: Path) -> None:
    """--start-at fix proceeds when merged-items.json is present even if the
    render-only review-output.md markdown is absent (canonical JSON is the gate).
    """
    (deep_artifacts_dir / "merged-items.json").write_text('{"items": []}')
    # No review-output.md anywhere -- must not raise.
    check_deep_artifacts("fix", deep_artifacts_dir)


def test_check_deep_artifacts_fix_fails_without_json(
    tmp_path: Path, deep_artifacts_dir: Path
) -> None:
    """--start-at fix fails loudly when no merged-items.json exists, even if the
    markdown report is present (markdown alone is not the source of truth).
    """
    (tmp_path / REVIEW_OUTPUT_FILE).write_text("# Review\n")  # markdown present
    (deep_artifacts_dir / "review-output.md").write_text("# Review\n")  # deep-dir markdown too
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
        check_deep_artifacts("merge", deep_artifacts_dir, current_diff_sha="bbbb")


def test_check_deep_artifacts_rejects_missing_key(deep_artifacts_dir: Path) -> None:
    """A pre-upgrade artifact dir (no key) is unverifiable, so it refuses."""
    _seed_merge_stage(deep_artifacts_dir)

    with pytest.raises(FileNotFoundError, match="produced before diff tracking"):
        check_deep_artifacts("merge", deep_artifacts_dir, current_diff_sha="bbbb")


def test_check_deep_artifacts_accepts_matching_diff_key(deep_artifacts_dir: Path) -> None:
    (deep_artifacts_dir / "diff-key").write_text("bbbb")
    _seed_merge_stage(deep_artifacts_dir)

    check_deep_artifacts("merge", deep_artifacts_dir, current_diff_sha="bbbb")  # must not raise


def test_check_deep_artifacts_rejects_prerequisites_older_than_matching_key(
    deep_artifacts_dir: Path,
) -> None:
    """A matching key cannot validate artifacts left from an earlier run."""
    _seed_merge_stage(deep_artifacts_dir)
    key_file = deep_artifacts_dir / "diff-key"
    key_file.write_text("bbbb")
    key_mtime = key_file.stat().st_mtime_ns + 1_000_000_000
    os.utime(key_file, ns=(key_mtime, key_mtime))

    with pytest.raises(FileNotFoundError, match="different diff"):
        check_deep_artifacts("merge", deep_artifacts_dir, current_diff_sha="bbbb")


def test_check_deep_artifacts_without_sha_skips_the_freshness_gate(
    deep_artifacts_dir: Path,
) -> None:
    """Omitting current_diff_sha preserves the presence-only behavior."""
    _seed_merge_stage(deep_artifacts_dir)

    check_deep_artifacts("merge", deep_artifacts_dir)  # must not raise


def test_missing_artifacts_are_reported_before_staleness(deep_artifacts_dir: Path) -> None:
    """A missing prereq keeps its own actionable message, not the stale one."""
    with pytest.raises(FileNotFoundError, match="missing artifacts"):
        check_deep_artifacts("merge", deep_artifacts_dir, current_diff_sha="bbbb")


def test_diff_key_is_content_addressed() -> None:
    assert diff_key("abc") == diff_key("abc")
    assert diff_key("abc") != diff_key("abd")


def test_diagram_artifact_paths_live_in_the_deep_dir(tmp_path: Path) -> None:
    """#1113: the diagram decision JSON and its rendered markdown sit beside the
    other deep artifacts, under the same deep dir the run already owns."""
    dd = deep_dir(tmp_path, allow_standalone=True)
    assert diagram_path(dd) == dd / "diagram.json"
    assert diagram_markdown_path(dd) == dd / "diagram.md"
    assert diagram_path(dd).parent == diagram_markdown_path(dd).parent == dd
