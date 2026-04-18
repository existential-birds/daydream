"""Deep-mode prompt builder tests (D-09, D-10, D-19, D-20)."""
from pathlib import Path

from daydream.deep.prompts import (
    DOC_REVIEW_NOTICE,
    build_generic_fallback_prompt,
    build_per_stack_prompt,
)


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "diff_path": tmp_path / ".daydream" / "diff.patch",
        "intent_path": tmp_path / ".daydream" / "deep" / "intent.md",
        "alternatives_path": tmp_path / ".daydream" / "deep" / "alternatives.json",
        "output_path": tmp_path / ".daydream" / "deep" / "stack-python-review.md",
    }


def test_per_stack_prompt_has_intent_pointer(tmp_path: Path) -> None:
    """D-19: prompt references the intent path."""
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert str(p["intent_path"]) in out


def test_per_stack_prompt_has_alternatives_pointer(tmp_path: Path) -> None:
    """D-19: prompt references the alternatives path."""
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert str(p["alternatives_path"]) in out


def test_per_stack_prompt_includes_skill_invocation(tmp_path: Path) -> None:
    """D-19: prompt includes the Beagle skill invocation."""
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert "/beagle-python:review-python" in out


def test_per_stack_prompt_scope_lists_only_stack_files(tmp_path: Path) -> None:
    """D-10: stack scope instruction lists only this stack's files."""
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py", "lib/util.py"],
        **p,
    )
    assert "api.py" in out and "lib/util.py" in out
    assert "Do NOT review files from other stacks" in out


def test_generic_fallback_prompt_has_no_skill(tmp_path: Path) -> None:
    """Generic fallback omits any /beagle-* invocation."""
    p = _paths(tmp_path)
    out = build_generic_fallback_prompt(files=["config.yaml"], **p)
    assert "/beagle-" not in out


def test_generic_fallback_docs_notice(tmp_path: Path) -> None:
    """D-20: is_docs_only=True prepends the doc-review notice."""
    p = _paths(tmp_path)
    out = build_generic_fallback_prompt(files=["README.md"], is_docs_only=True, **p)
    assert DOC_REVIEW_NOTICE in out
    # Notice must appear before other content
    assert out.index(DOC_REVIEW_NOTICE) < out.index("Review these files")


def test_generic_fallback_no_docs_notice_by_default(tmp_path: Path) -> None:
    """Docs notice suppressed when is_docs_only=False."""
    p = _paths(tmp_path)
    out = build_generic_fallback_prompt(files=["config.yaml"], **p)
    assert DOC_REVIEW_NOTICE not in out


def test_prompts_embed_no_full_file_contents(tmp_path: Path) -> None:
    """D-09: prompts reference paths, never embed diffs or file bodies."""
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    # Heuristic: no line longer than 400 chars (an embedded diff would blow this up)
    assert all(len(line) < 400 for line in out.splitlines())


def test_per_stack_prompt_points_at_diff_path(tmp_path: Path) -> None:
    """Prompt references diff_path for agents to read directly."""
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert str(p["diff_path"]) in out


def test_per_stack_prompt_omits_bare_git_diff_command(tmp_path: Path) -> None:
    """Prompt must NOT suggest `git diff -- <files>` without a base ref.

    Without a base ref that command only shows uncommitted workspace changes;
    on a clean PR branch it returns empty and hides every committed change.
    """
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert "git diff --no-color -- api.py" not in out
    assert "git diff -- api.py" not in out


def test_generic_fallback_prompt_omits_bare_git_diff_command(tmp_path: Path) -> None:
    """Generic fallback must not embed the broken git-diff command either."""
    p = _paths(tmp_path)
    out = build_generic_fallback_prompt(files=["config.yaml"], **p)
    assert "git diff --no-color -- config.yaml" not in out
    assert "git diff -- config.yaml" not in out
    # diff_path must still be referenced.
    assert str(p["diff_path"]) in out
