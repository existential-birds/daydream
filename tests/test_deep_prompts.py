"""Deep-mode prompt builder tests (D-09, D-10, D-19, D-20)."""
from pathlib import Path
from typing import TypedDict

from daydream.deep.prompts import (
    DOC_REVIEW_NOTICE,
    build_generic_fallback_prompt,
    build_merge_prompt,
    build_per_stack_prompt,
)


class _PromptPaths(TypedDict):
    """The four on-disk path kwargs shared by the per-stack and fallback builders.

    Declaring each key's type explicitly lets mypy reconcile ``**p`` unpacking
    with the builders' per-parameter signatures; a plain ``dict[str, Path]`` would
    spill ``Path`` onto unrelated kwargs like ``prior_commits``/``is_docs_only``.
    """

    diff_path: Path
    intent_path: Path
    alternatives_path: Path
    output_path: Path
    cwd: Path


def _paths(tmp_path: Path) -> _PromptPaths:
    return {
        "diff_path": tmp_path / ".daydream" / "diff.patch",
        "intent_path": tmp_path / ".daydream" / "deep" / "intent.md",
        "alternatives_path": tmp_path / ".daydream" / "deep" / "alternatives.json",
        "output_path": tmp_path / ".daydream" / "deep" / "stack-python-review.md",
        "cwd": tmp_path,
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
    """D-09: prompts reference paths, never embed diffs or file bodies.

    Note (issue #172, Fix B): when ``inline_diff`` is supplied the per-stack
    prompt DOES embed the relevant diff hunks — that is the read-once
    optimization, not a D-09 violation. This heuristic guards the default
    (``inline_diff=None``) path only: no line longer than 400 chars there.
    The inlined path is bounded by ``INLINE_DIFF_BUDGET_BYTES`` separately
    (see ``test_inline_diff_byte_budget``).
    """
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    # Heuristic: no line longer than 400 chars (an embedded diff would blow this
    # up). The cwd-grounding instruction (issue #221) is one fixed long line and
    # is not file content, so exclude it from this check.
    from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION

    grounding = CWD_GROUNDING_INSTRUCTION.format(cwd=tmp_path)
    assert all(len(line) < 400 for line in out.splitlines() if line != grounding)


def test_per_stack_prompt_points_at_diff_path(tmp_path: Path) -> None:
    """Fallback (``inline_diff=None``): prompt references diff_path for agents
    to read directly.

    Issue #172 Fix B: with ``inline_diff`` supplied, the path pointer is
    DROPPED (the hunks are inlined instead). The fallback contract — when the
    byte budget is exceeded or the caller has no diff text — keeps the pointer
    so the agent can still locate the full diff for whole-file context.
    """
    p = _paths(tmp_path)
    # Default (inline_diff=None) → pointer present (fallback contract).
    out_fallback = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert str(p["diff_path"]) in out_fallback
    # inline_diff supplied → pointer absent (hunks inlined instead).
    out_inline = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        inline_diff="diff --git a/api.py b/api.py\n+++ b/api.py\n@@ -1 +1 @@\n-x\n+y\n",
        **p,
    )
    assert str(p["diff_path"]) not in out_inline
    assert "Read it directly" not in out_inline
    assert "-x" in out_inline and "+y" in out_inline  # hunks inlined


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
    """Generic fallback must not embed the broken git-diff command either.

    Issue #172 Fix B: with ``inline_diff`` supplied, the diff_path pointer is
    DROPPED. The fallback (``inline_diff=None``) path still references
    diff_path so the agent can locate the full diff.
    """
    p = _paths(tmp_path)
    # Default (inline_diff=None) → pointer present (fallback contract).
    out_fallback = build_generic_fallback_prompt(files=["config.yaml"], **p)
    assert "git diff --no-color -- config.yaml" not in out_fallback
    assert "git diff -- config.yaml" not in out_fallback
    assert str(p["diff_path"]) in out_fallback
    # inline_diff supplied → pointer absent (hunks inlined instead).
    out_inline = build_generic_fallback_prompt(
        files=["config.yaml"],
        inline_diff="diff --git a/config.yaml b/config.yaml\n+++ b/config.yaml\n@@ -1 +1 @@\n-x\n+y\n",
        **p,
    )
    assert str(p["diff_path"]) not in out_inline
    assert "Read it directly" not in out_inline


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
        skill_invocation=f"/{STRUCTURE_SKILL}",
        files=["api/main.py", "ui/App.tsx"],
        diff_path=tmp_path / "diff.patch",
        intent_path=tmp_path / "intent.md",
        alternatives_path=tmp_path / "alternatives.json",
        output_path=tmp_path / "out.md",
        cwd=tmp_path,
    )
    assert "Focus ONLY on these files" not in prompt
    assert "Do NOT review files from other stacks" not in prompt
    assert STRUCTURE_SKILL in prompt or "/" + STRUCTURE_SKILL in prompt
    assert str(tmp_path / "out.md") in prompt


def test_build_structural_prompt_omits_exploration_pointer(tmp_path: Path) -> None:
    """Per spec: structural reviewer discovers via tool calls, not pre-injected context."""
    from daydream.deep.prompts import build_structural_prompt

    prompt = build_structural_prompt(
        skill_invocation="/beagle-core:review-structure",
        files=["main.py"],
        diff_path=tmp_path / "diff.patch",
        intent_path=tmp_path / "intent.md",
        alternatives_path=tmp_path / "alternatives.json",
        output_path=tmp_path / "out.md",
        exploration_dir=tmp_path / "exploration",
        cwd=tmp_path,
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


# =============================================================================
# Issue #172 — Fix B: read-once inline diff hunks in per-stack / generic prompts
# =============================================================================


_DIFF_TWO_FILES = (
    "diff --git a/api.py b/api.py\n"
    "+++ b/api.py\n"
    "@@ -1 +1 @@\n"
    "-def hello(): return 'world'\n"
    "+def hello(): return 'universe'\n"
    "diff --git a/App.tsx b/App.tsx\n"
    "+++ b/App.tsx\n"
    "@@ -1 +1 @@\n"
    "-export const App = () => <div>hello</div>;\n"
    "+export const App = () => <div>universe</div>;\n"
)


def test_diff_blocks_for_files_selects_relevant_hunks() -> None:
    """AC4 helper: ``_diff_blocks_for_files`` returns only the blocks for the
    requested files (post-state path match), concatenated as-is.
    """
    from daydream.deep.prompts import _diff_blocks_for_files

    out = _diff_blocks_for_files(_DIFF_TWO_FILES, ["api.py"])
    assert out is not None
    assert "diff --git a/api.py b/api.py" in out
    assert "def hello(): return 'universe'" in out
    # App.tsx block is NOT in the filtered output.
    assert "App.tsx" not in out
    # Two files requested → both blocks present.
    both = _diff_blocks_for_files(_DIFF_TWO_FILES, ["api.py", "App.tsx"])
    assert both is not None
    assert "def hello(): return 'universe'" in both
    assert "<div>universe</div>" in both


def test_diff_blocks_for_files_returns_none_above_byte_budget() -> None:
    """AC4 byte-bound fallback: when the relevant blocks exceed
    ``INLINE_DIFF_BUDGET_BYTES``, the helper returns ``None`` so the caller
    keeps the path pointer (the agent is told to Read diff.patch directly).
    """
    from daydream.deep.prompts import INLINE_DIFF_BUDGET_BYTES, _diff_blocks_for_files

    # Synthesize a diff whose single matching block exceeds the budget.
    huge_line = "x" * (INLINE_DIFF_BUDGET_BYTES + 64)
    huge_diff = (
        "diff --git a/api.py b/api.py\n"
        "+++ b/api.py\n"
        "@@ -1 +1 @@\n"
        f"-{huge_line}\n"
        f"+{huge_line}\n"
    )
    assert _diff_blocks_for_files(huge_diff, ["api.py"]) is None


def test_diff_blocks_for_files_returns_none_when_no_blocks_match() -> None:
    """AC4 no-match fallback: files not in the diff → None (caller keeps pointer)."""
    from daydream.deep.prompts import _diff_blocks_for_files

    out = _diff_blocks_for_files(_DIFF_TWO_FILES, ["nonexistent.py"])
    assert out is None


def test_per_stack_prompt_inlines_hunks_and_drops_read_instruction(tmp_path: Path) -> None:
    """AC4 (unit): per-stack prompt with ``inline_diff`` supplied contains the
    inlined hunks, NOT the ``Read it directly`` instruction or diff_path.
    """
    from daydream.deep.prompts import _diff_blocks_for_files

    p = _paths(tmp_path)
    inline = _diff_blocks_for_files(_DIFF_TWO_FILES, ["api.py"])
    assert inline is not None  # sanity: the helper found a block
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        inline_diff=inline,
        **p,
    )
    assert "def hello(): return 'universe'" in out  # hunk inlined
    assert "Read it directly" not in out
    assert str(p["diff_path"]) not in out


def test_generic_fallback_prompt_inlines_hunks_and_drops_read_instruction(
    tmp_path: Path,
) -> None:
    """AC4 (unit): generic-fallback prompt with ``inline_diff`` supplied contains
    the inlined hunks, NOT the ``Read it directly`` instruction or diff_path.
    """
    from daydream.deep.prompts import _diff_blocks_for_files

    p = _paths(tmp_path)
    inline = _diff_blocks_for_files(_DIFF_TWO_FILES, ["App.tsx"])
    assert inline is not None
    out = build_generic_fallback_prompt(
        files=["App.tsx"],
        inline_diff=inline,
        **p,
    )
    assert "<div>universe</div>" in out
    assert "Read it directly" not in out
    assert str(p["diff_path"]) not in out


def test_structural_prompt_keeps_diff_pointer_and_read_freedom(tmp_path: Path) -> None:
    """AC4: structural prompt is NOT inlined — it keeps its diff pointer AND
    its repo-wide Read/Grep/Bash freedom (the structural lens roams beyond
    the diff by design). Fix B does not touch the structural / arbiter prompts.
    """
    from daydream.deep.prompts import build_structural_prompt

    p = _paths(tmp_path)
    out = build_structural_prompt(
        skill_invocation="/beagle-core:review-structure",
        files=["api.py"],
        **p,
    )
    assert "read any file in the codebase" in out
    assert str(p["diff_path"]) in out  # keeps its pointer
    assert "Read it directly" in out   # structural prompt unchanged


# =============================================================================
# Issue #221 — cwd grounding injected into every deep prompt builder
# =============================================================================


def test_per_stack_prompt_contains_cwd_grounding(tmp_path: Path) -> None:
    from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION

    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=tmp_path) in out
    assert str(tmp_path) in out


def test_structural_prompt_contains_cwd_grounding(tmp_path: Path) -> None:
    from daydream.deep.prompts import build_structural_prompt
    from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION

    p = _paths(tmp_path)
    out = build_structural_prompt(
        skill_invocation="/beagle-core:review-structure",
        files=["api.py"],
        **p,
    )
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=tmp_path) in out


def test_arbiter_prompt_contains_cwd_grounding(tmp_path: Path) -> None:
    from daydream.deep.prompts import build_arbiter_prompt
    from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION

    out = build_arbiter_prompt(
        arbiter_input_path=tmp_path / "arbiter-input.json",
        diff_path=tmp_path / "diff.patch",
        intent_path=tmp_path / "intent.md",
        alternatives_path=tmp_path / "alternatives.json",
        cwd=tmp_path,
    )
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=tmp_path) in out


def test_generic_fallback_prompt_contains_cwd_grounding(tmp_path: Path) -> None:
    from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION

    p = _paths(tmp_path)
    out = build_generic_fallback_prompt(files=["config.yaml"], **p)
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=tmp_path) in out


def test_verification_prompt_contains_cwd_grounding(tmp_path: Path) -> None:
    from daydream.deep.prompts import build_verification_prompt
    from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION

    out = build_verification_prompt(
        items=[{"id": 1, "lens": "per-stack", "severity": "high", "file": "api.py",
                "line": 10, "description": "x", "rationale": "y"}],
        cwd=tmp_path,
        output_path=tmp_path / "verdicts.json",
    )
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=tmp_path) in out


def test_build_structural_prompt_includes_verification_protocol(tmp_path: Path) -> None:
    from daydream.deep.prompts import build_structural_prompt

    p = _paths(tmp_path)
    prompt = build_structural_prompt(
        skill_invocation="/beagle-core:review-structure",
        files=["api.py"],
        **p,
    )
    assert "review-verification-protocol" in prompt
    assert "anchor" in prompt
    assert "evidence" in prompt


def test_build_generic_fallback_prompt_includes_verification_protocol(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    out = build_generic_fallback_prompt(files=["config.yaml"], **p)
    assert "review-verification-protocol" in out
    assert "anchor" in out
    assert "evidence" in out


def test_build_verification_prompt_includes_gate_zero_echo(tmp_path: Path) -> None:
    from daydream.deep.prompts import build_verification_prompt

    items = [{"id": "1", "file": "x.py", "line": 10, "description": "Test finding"}]
    out = build_verification_prompt(
        items=items,
        cwd=tmp_path,
        output_path=tmp_path / "verdicts.json",
    )
    assert "Gate-0" in out or "anti-confabulation" in out
    assert "same-turn echo" in out or "file:line" in out


def test_no_format_skill_invocation_for_verification_protocol() -> None:
    """The protocol is user-invocable:false — must not be used with format_skill_invocation."""
    import subprocess

    repo_root = Path(__file__).parent.parent
    result = subprocess.run(
        ["grep", "-rn", "format_skill_invocation", "daydream/"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    for line in result.stdout.splitlines():
        assert "review-verification-protocol" not in line, f"format_skill_invocation references protocol: {line}"
