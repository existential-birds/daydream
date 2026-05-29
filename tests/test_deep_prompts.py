"""Deep-mode prompt builder tests (D-09, D-10, D-19, D-20)."""
from pathlib import Path

from daydream.deep.prompts import (
    DOC_REVIEW_NOTICE,
    build_generic_fallback_prompt,
    build_merge_prompt,
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


def _merge_paths(tmp_path: Path) -> dict[str, Path | list[Path] | None]:
    return {
        "per_stack_records_paths": [tmp_path / "python.json", tmp_path / "react.json"],
        "intent_path": tmp_path / "intent.md",
        "alternatives_path": tmp_path / "alternatives.json",
        "dedup_candidates_path": tmp_path / "dedup.json",
        "output_path": tmp_path / "review.md",
        "exploration_dir": None,
        "failed_stacks": None,
    }


def test_per_stack_prompt_includes_prior_commits(tmp_path: Path) -> None:
    """prior_commits block appears in per-stack prompt when provided."""
    p = _paths(tmp_path)
    commits = "abc1234 fix: handle edge case\ndef5678 feat: add retry logic"
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        prior_commits=commits,
        **p,
    )
    assert "Prior automated-review commits on this branch" in out
    assert "abc1234 fix: handle edge case" in out
    assert "def5678 feat: add retry logic" in out


def test_per_stack_prompt_omits_prior_commits_when_none(tmp_path: Path) -> None:
    """prior_commits block absent when prior_commits is None."""
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        prior_commits=None,
        **p,
    )
    assert "Prior automated-review commits" not in out


def test_per_stack_prompt_omits_prior_commits_when_empty(tmp_path: Path) -> None:
    """prior_commits block absent when prior_commits is empty string."""
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        prior_commits="",
        **p,
    )
    assert "Prior automated-review commits" not in out


def test_generic_fallback_prompt_includes_prior_commits(tmp_path: Path) -> None:
    """prior_commits block appears in generic-fallback prompt when provided."""
    p = _paths(tmp_path)
    commits = "abc1234 fix: handle edge case"
    out = build_generic_fallback_prompt(
        files=["config.yaml"],
        prior_commits=commits,
        **p,
    )
    assert "Prior automated-review commits on this branch" in out
    assert "abc1234 fix: handle edge case" in out


def test_generic_fallback_prompt_omits_prior_commits_when_none(tmp_path: Path) -> None:
    """prior_commits block absent from generic-fallback when prior_commits is None."""
    p = _paths(tmp_path)
    out = build_generic_fallback_prompt(
        files=["config.yaml"],
        prior_commits=None,
        **p,
    )
    assert "Prior automated-review commits" not in out


def test_generic_fallback_prompt_omits_prior_commits_when_empty(tmp_path: Path) -> None:
    """prior_commits block absent from generic-fallback when prior_commits is empty."""
    p = _paths(tmp_path)
    out = build_generic_fallback_prompt(
        files=["config.yaml"],
        prior_commits="",
        **p,
    )
    assert "Prior automated-review commits" not in out


def test_merge_prompt_requires_structured_item_fields(tmp_path: Path) -> None:
    """The merge agent emits a structured item list; markdown formatting rules
    (bold-wrapping, head-line layout) no longer apply — Python renders the report.
    """
    out = build_merge_prompt(**_merge_paths(tmp_path))  # type: ignore[arg-type]
    assert '{"items": [' in out
    assert "Item fields (MANDATORY):" in out
    # No markdown write-to-file or bold-wrapping directive survives.
    assert "do NOT wrap it in `**...**`" not in out
    assert "write the complete report to" not in out.lower()


def test_merge_prompt_requires_one_path_per_item(tmp_path: Path) -> None:
    """Multi-file concerns must become multiple items, not a comma list in one file field."""
    out = build_merge_prompt(**_merge_paths(tmp_path))  # type: ignore[arg-type]
    assert "EXACTLY ONE path" in out
    assert "separate item per file" in out


def test_build_structural_prompt_has_no_stack_scope_restriction(tmp_path: Path) -> None:
    """Structural reviewer sees the whole change — no 'Focus ONLY on these files' clause."""
    from daydream.config import STRUCTURE_SKILL
    from daydream.deep.prompts import build_structural_prompt

    prompt = build_structural_prompt(
        files=["api/main.py", "ui/App.tsx"],
        diff_path=tmp_path / "diff.patch",
        intent_path=tmp_path / "intent.md",
        alternatives_path=tmp_path / "alternatives.json",
        output_path=tmp_path / "out.md",
    )
    assert "Focus ONLY on these files" not in prompt
    assert "Do NOT review files from other stacks" not in prompt
    assert STRUCTURE_SKILL in prompt or "/" + STRUCTURE_SKILL in prompt
    assert str(tmp_path / "out.md") in prompt


def test_build_structural_prompt_omits_exploration_pointer(tmp_path: Path) -> None:
    """Per spec: structural reviewer discovers via tool calls, not pre-injected context."""
    from daydream.deep.prompts import build_structural_prompt

    prompt = build_structural_prompt(
        files=["main.py"],
        diff_path=tmp_path / "diff.patch",
        intent_path=tmp_path / "intent.md",
        alternatives_path=tmp_path / "alternatives.json",
        output_path=tmp_path / "out.md",
        exploration_dir=tmp_path / "exploration",
    )
    assert "exploration" not in prompt.lower()


def test_merge_prompt_does_not_request_structural_findings(tmp_path: Path) -> None:
    """Structural findings are appended by the host (phase_cross_stack_merge) in
    Python, NOT requested via prose. The agent is never pointed at the structural
    records file and is told not to emit structural items itself."""
    from daydream.deep.prompts import build_merge_prompt

    structural_path = tmp_path / "stack-structure-records.json"
    prompt = build_merge_prompt(
        per_stack_records_paths=[tmp_path / "stack-python-records.json"],
        intent_path=tmp_path / "intent.md",
        alternatives_path=tmp_path / "alts.json",
        dedup_candidates_path=tmp_path / "dedup.json",
        output_path=tmp_path / "report.md",
        structural_records_path=structural_path,
    )
    assert str(structural_path) not in prompt  # agent not pointed at structural records
    assert "## Structural Review" not in prompt
    assert "do NOT emit them yourself" in prompt


def test_merge_prompt_omits_structural_section_when_path_is_none(tmp_path: Path) -> None:
    """No structural section when the meta-stack did not run (docs-only, empty diff)."""
    from daydream.deep.prompts import build_merge_prompt

    prompt = build_merge_prompt(
        per_stack_records_paths=[tmp_path / "stack-python-records.json"],
        intent_path=tmp_path / "intent.md",
        alternatives_path=tmp_path / "alts.json",
        dedup_candidates_path=tmp_path / "dedup.json",
        output_path=tmp_path / "report.md",
        structural_records_path=None,
    )
    assert "## Structural Review" not in prompt
    assert "Structural-stack parsed records:" not in prompt
    assert "Structural-stack handling:" not in prompt
