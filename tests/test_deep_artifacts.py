"""Deep-mode artifact path + check_deep_artifacts tests (D-18, D-36, D-37)."""

from pathlib import Path

import pytest


def test_per_stack_path_scheme(tmp_path: Path) -> None:
    """D-18: per-stack output path is deterministic + unique."""
    from daydream.deep.artifacts import per_stack_review_path

    p1 = per_stack_review_path(tmp_path, "python")
    p2 = per_stack_review_path(tmp_path, "react")
    assert p1 != p2
    assert p1.name == "stack-python-review.md"
    assert p2.name == "stack-react-review.md"


def test_check_deep_artifacts_missing(tmp_path: Path) -> None:
    """D-36: check_deep_artifacts raises FileNotFoundError when predecessor missing."""
    from daydream.deep.artifacts import check_deep_artifacts

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("per-stack", deep_dir)
    assert "intent.md" in str(excinfo.value)
    assert "--start-at" in str(excinfo.value)


def test_check_deep_artifacts_merge_requires_records(tmp_path: Path) -> None:
    """D-37: --start-at merge needs per-stack records on disk."""
    from daydream.deep.artifacts import check_deep_artifacts

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    (deep_dir / "intent.md").write_text("x")
    (deep_dir / "alternatives.json").write_text("[]")
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("merge", deep_dir)
    assert "stack-*-records.json" in str(excinfo.value)


def test_check_deep_artifacts_passes_when_present(tmp_path: Path) -> None:
    """D-36: check passes silently when all predecessors exist."""
    from daydream.deep.artifacts import check_deep_artifacts

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    (deep_dir / "intent.md").write_text("x")
    (deep_dir / "alternatives.json").write_text("[]")
    check_deep_artifacts("per-stack", deep_dir)  # must not raise


def test_check_deep_artifacts_rejects_directory_shadowing_prereq(tmp_path: Path) -> None:
    """A directory named like a prereq must not satisfy the gate."""
    from daydream.deep.artifacts import check_deep_artifacts

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    # intent.md exists as a directory, not a file.
    (deep_dir / "intent.md").mkdir()
    (deep_dir / "alternatives.json").write_text("[]")
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("per-stack", deep_dir)
    assert "intent.md" in str(excinfo.value)


def test_check_deep_artifacts_merge_ignores_directory_records(tmp_path: Path) -> None:
    """A directory matching stack-*-records.json must not satisfy the merge gate."""
    from daydream.deep.artifacts import check_deep_artifacts

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    (deep_dir / "intent.md").write_text("x")
    (deep_dir / "alternatives.json").write_text("[]")
    (deep_dir / "stack-bogus-records.json").mkdir()  # directory, not a file
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("merge", deep_dir)
    assert "stack-*-records.json" in str(excinfo.value)


def test_check_deep_artifacts_fix_rejects_directory_merged_items(tmp_path: Path) -> None:
    """A directory named merged-items.json must not satisfy the fix gate.

    The fix gate keys on the canonical merged-items.json (the source of truth the
    fix loop reads), not the render-only review-output.md markdown.
    """
    from daydream.deep.artifacts import check_deep_artifacts

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    (deep_dir / "merged-items.json").mkdir()  # directory, not a file
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("fix", deep_dir)
    assert "merged-items.json" in str(excinfo.value)


def test_check_deep_artifacts_fix_passes_with_json_only(tmp_path: Path) -> None:
    """--start-at fix proceeds when merged-items.json is present even if the
    render-only review-output.md markdown is absent (canonical JSON is the gate).
    """
    from daydream.deep.artifacts import check_deep_artifacts

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    (deep_dir / "merged-items.json").write_text('{"items": []}')
    # No review-output.md anywhere -- must not raise.
    check_deep_artifacts("fix", deep_dir)


def test_check_deep_artifacts_fix_fails_without_json(tmp_path: Path) -> None:
    """--start-at fix fails loudly when no merged-items.json exists, even if the
    markdown report is present (markdown alone is not the source of truth).
    """
    from daydream.config import REVIEW_OUTPUT_FILE
    from daydream.deep.artifacts import check_deep_artifacts

    target = tmp_path
    deep_dir = target / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    (target / REVIEW_OUTPUT_FILE).write_text("# Review\n")  # markdown present
    (deep_dir / "review-output.md").write_text("# Review\n")  # deep-dir markdown too
    with pytest.raises(FileNotFoundError) as excinfo:
        check_deep_artifacts("fix", deep_dir)
    assert "merged-items.json" in str(excinfo.value)
