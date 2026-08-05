"""Deep-mode prompt builder tests (D-09, D-10, D-19, D-20)."""
from pathlib import Path
from typing import TypedDict

import pytest

from daydream.deep.prompts import (
    DOC_REVIEW_NOTICE,
    build_arbiter_prompt,
    build_generic_fallback_prompt,
    build_merge_prompt,
    build_per_stack_prompt,
    build_structural_prompt,
)
from daydream.prompts.authorial_intent import AUTHORITATIVE_INTENT_RULE


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


def test_no_format_skill_invocation_for_verification_protocol(tmp_path: Path) -> None:
    """The protocol gates are embedded inline as instruction text — never routed
    through ``backend.format_skill_invocation`` AND never loaded from a skill file.

    The structural and generic-fallback reviewers run with cwd set to the
    reviewed repo, so a bare ``read review-verification-protocol/SKILL.md`` would
    resolve against that repo and fail. The gates are therefore stated inline
    (see ``VERIFICATION_PROTOCOL_INSTRUCTION``). Build the prompts with a real
    backend formatter and assert the gate discipline is present while neither a
    skill-file read nor the protocol's invocation token leaks into the prompt.
    """
    from daydream.backends import create_backend
    from daydream.deep.prompts import build_structural_prompt

    p = _paths(tmp_path)
    prompts = [
        build_structural_prompt(
            skill_invocation="/beagle-core:review-structure",
            files=["api.py"],
            **p,
        ),
        build_generic_fallback_prompt(files=["config.yaml"], **p),
    ]

    for backend_name in ("claude", "codex", "pi"):
        backend = create_backend(backend_name)
        token = backend.format_skill_invocation("review-verification-protocol")
        for prompt in prompts:
            # Gates are embedded as methodology prose (the constant names the
            # protocol and states gates 0-3 inline), never as an invocation and
            # never as an unresolvable skill-file read.
            assert "review-verification-protocol" in prompt
            assert "SKILL.md" not in prompt
            assert token not in prompt, (
                f"{backend_name} protocol invocation token {token!r} leaked into prompt"
            )


# =============================================================================
# Issue #279 — Authoritative-intent rule gate in the deep prompt builders
# =============================================================================


def _build_gated(name: str, tmp_path: Path, *, intent_authoritative: bool) -> str:
    """Dispatch to the named builder with minimal valid kwargs.

    Note the signature differences: ``arbiter`` has no ``output_path`` and needs
    ``arbiter_input_path``; ``merge`` has no ``cwd``/``diff_path`` and needs
    ``per_stack_records_paths`` and ``dedup_candidates_path``.
    """
    p = _paths(tmp_path)
    if name == "per-stack":
        return build_per_stack_prompt(
            skill_invocation="/beagle-python:review-python",
            stack_name="python",
            files=["api.py"],
            intent_authoritative=intent_authoritative,
            **p,
        )
    if name == "structural":
        return build_structural_prompt(
            skill_invocation="/beagle-core:review-structure",
            files=["api.py"],
            intent_authoritative=intent_authoritative,
            **p,
        )
    if name == "generic-fallback":
        return build_generic_fallback_prompt(
            files=["config.yaml"],
            intent_authoritative=intent_authoritative,
            **p,
        )
    if name == "arbiter":
        return build_arbiter_prompt(
            arbiter_input_path=tmp_path / "arbiter-input.json",
            diff_path=p["diff_path"],
            intent_path=p["intent_path"],
            alternatives_path=p["alternatives_path"],
            cwd=p["cwd"],
            intent_authoritative=intent_authoritative,
        )
    if name == "merge":
        return build_merge_prompt(
            per_stack_records_paths=[tmp_path / "python.json", tmp_path / "react.json"],
            intent_path=tmp_path / "intent.md",
            alternatives_path=tmp_path / "alternatives.json",
            dedup_candidates_path=tmp_path / "dedup.json",
            output_path=tmp_path / "report.md",
            intent_authoritative=intent_authoritative,
        )
    msg = f"unknown builder name: {name!r}"
    raise ValueError(msg)


@pytest.mark.parametrize("name", ["per-stack", "structural", "generic-fallback", "arbiter", "merge"])
def test_authoritative_intent_rule_is_gated(name: str, tmp_path: Path) -> None:
    """#279: the precedence rule appears only when a fresh PR body was ingested."""
    assert AUTHORITATIVE_INTENT_RULE not in _build_gated(name, tmp_path, intent_authoritative=False)
    assert AUTHORITATIVE_INTENT_RULE in _build_gated(name, tmp_path, intent_authoritative=True)


def test_verification_prompt_has_no_schema_dump_or_write_instruction(tmp_path: Path) -> None:
    """The verify prompt carries neither the schema dump nor a write instruction.

    The schema reaches every backend via ``output_schema``, and the host writes
    the verdicts file, so both blocks were pure duplication.
    """
    from daydream.deep.prompts import build_verification_prompt

    items = [
        {"id": 1, "lens": "per-stack", "severity": "high", "file": "api.py",
         "line": 10, "description": "x", "rationale": "y"}
    ]
    prompt = build_verification_prompt(
        items=items, cwd=tmp_path, output_path=tmp_path / "verdicts.json"
    )

    assert "conforming EXACTLY to this schema" not in prompt
    assert "RECOMMENDATION_VERDICTS_SCHEMA" not in prompt
    assert "Write your JSON verdicts" not in prompt
    # The read-only clause no longer dangles an exception for the output path.
    assert "Do NOT write, edit, or move files." in prompt
    assert "except the JSON output" not in prompt
    # output_path is accepted-but-ignored: it must not appear in the prompt.
    assert "verdicts.json" not in prompt
    # The substantive instructions survive.
    assert "recommendation-verifier agent" in prompt
    assert "unverified_assumptions" in prompt


def test_verification_prompt_ignores_output_path(tmp_path: Path) -> None:
    """Two different output_path values produce byte-identical prompts."""
    from daydream.deep.prompts import build_verification_prompt

    items = [
        {"id": 1, "lens": "per-stack", "severity": "high", "file": "api.py",
         "line": 10, "description": "x", "rationale": "y"}
    ]
    a = build_verification_prompt(items=items, cwd=tmp_path, output_path=tmp_path / "a.json")
    b = build_verification_prompt(items=items, cwd=tmp_path, output_path=tmp_path / "b.json")
    assert a == b


# --- Task 12a: the per-stack prompt path can omit the alternatives pointer ----


def test_per_stack_prompt_can_omit_alternatives(tmp_path: Path) -> None:
    from daydream.deep.prompts import build_per_stack_prompt

    p = _paths(tmp_path)
    with_alts = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python", stack_name="python",
        files=["api.py"], **p, include_alternatives=True,
    )
    without = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python", stack_name="python",
        files=["api.py"], **p, include_alternatives=False,
    )
    assert "alternatives.json" in with_alts
    assert "alternatives.json" not in without
    assert "intent.md" in without  # ONLY the alternatives paragraph is dropped
    assert build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python", stack_name="python",
        files=["api.py"], **p,
    ) == with_alts  # default is True


def test_structural_prompt_can_omit_alternatives(tmp_path: Path) -> None:
    from daydream.deep.prompts import build_structural_prompt

    p = _paths(tmp_path)
    with_alts = build_structural_prompt(
        skill_invocation="/beagle-core:review-structure", files=["api.py"], **p,
        include_alternatives=True,
    )
    without = build_structural_prompt(
        skill_invocation="/beagle-core:review-structure", files=["api.py"], **p,
        include_alternatives=False,
    )
    assert "alternatives.json" in with_alts
    assert "alternatives.json" not in without
    assert "intent.md" in without
    assert build_structural_prompt(
        skill_invocation="/beagle-core:review-structure", files=["api.py"], **p,
    ) == with_alts


def test_generic_fallback_prompt_can_omit_alternatives(tmp_path: Path) -> None:
    from daydream.deep.prompts import build_generic_fallback_prompt

    p = _paths(tmp_path)
    with_alts = build_generic_fallback_prompt(
        files=["config.yaml"], **p, include_alternatives=True
    )
    without = build_generic_fallback_prompt(
        files=["config.yaml"], **p, include_alternatives=False
    )
    assert "alternatives.json" in with_alts
    assert "alternatives.json" not in without
    assert "intent.md" in without
    assert build_generic_fallback_prompt(files=["config.yaml"], **p) == with_alts


def test_omitting_alternatives_keeps_authoritative_intent_rule(tmp_path: Path) -> None:
    """The authoritative-intent upgrade survives include_alternatives=False."""
    from daydream.deep.prompts import build_per_stack_prompt

    p = _paths(tmp_path)
    without = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python", stack_name="python",
        files=["api.py"], **p, intent_authoritative=True, include_alternatives=False,
    )
    assert "alternatives.json" not in without
    assert "author's stated intent from the pull-request description" in without
    assert AUTHORITATIVE_INTENT_RULE in without


def test_adjudication_builders_keep_alternatives_unconditionally(tmp_path: Path) -> None:
    """Arbiter/supervise/suppression/merge take no kwarg and always point at alts."""
    import inspect

    from daydream.deep import prompts as dp

    for name in ("build_arbiter_prompt", "build_merge_prompt"):
        sig = inspect.signature(getattr(dp, name))
        assert "include_alternatives" not in sig.parameters, name


# =============================================================================
# Issue #308 — test-quality rubric in the per-stack review prompt
# =============================================================================


def test_per_stack_prompt_includes_test_quality_rubric(tmp_path: Path) -> None:
    """#308: the per-stack review prompt ships the test-quality rubric.

    The rubric targets test hunks in the diff: vacuous assertions,
    internal-field/pointer-identity assertions, nondeterminism, canonical-path
    bypasses, and portability breaks.
    """
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert "test-quality rubric" in out
    assert "vacuous assertions" in out
    assert "observable consequences" in out
    assert "canonical public path" in out
    assert "deterministic" in out
    assert "`#[cfg]`" in out


def test_per_stack_prompt_test_quality_rubric_layering_awareness(tmp_path: Path) -> None:
    """#308: the rubric must not over-apply to legitimate pure-function seams.

    A unit test of a pure ``build_driver_request`` / driver-boundary propagation
    helper is NOT an internal-field assertion; the rubric only flags a seam when
    it bypasses the observable behavior the test claims to cover.
    """
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert "pure-function seams" in out
    assert "`build_driver_request`" in out
    assert "internal-field assertion" in out
    assert "bypasses the observable behavior" in out


def test_per_stack_prompt_test_quality_rubric_sits_after_skill_invocation(tmp_path: Path) -> None:
    """#308: the rubric lands after the skill invocation so the reviewer applies
    it to each test hunk, not ahead of the per-stack review instructions."""
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert out.index("test-quality rubric") > out.index("/beagle-python:review-python")


# =============================================================================
# Issue #314 — anti-slop review rubric (structural erosion + verbosity patterns)
# =============================================================================

_ANTI_SLOP_ANCHORS = (
    "complexity concentration",
    "extraction into focused callables",
    "identity comprehension",
    "empty-list guards",
    "single-use intermediate variables",
    "casts to dodge type checking",
    "trivial wrapper",
    "nested ladders",
    "same hunk structure repeated",
    "medium/low",
    "pre-existing-and-growing",
)


def _assert_anti_slop_anchors(out: str) -> None:
    missing = [anchor for anchor in _ANTI_SLOP_ANCHORS if anchor not in out]
    assert not missing, f"anti-slop rubric is missing pinned anchors: {missing}"


def test_per_stack_prompt_includes_anti_slop_rubric(tmp_path: Path) -> None:
    """#314: the per-stack review prompt ships the anti-slop rubric.

    The rubric targets the SlopCodeBench degradation patterns in the diff
    hunks: complexity concentration into already-large functions, verbosity
    (identity comprehensions, empty-list guards, single-use intermediates,
    casts to dodge type checking, trivial wrappers, nested ladders), and
    copy-pasted duplication.
    """
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert "anti-slop rubric" in out
    _assert_anti_slop_anchors(out)


def test_structural_prompt_includes_anti_slop_rubric(tmp_path: Path) -> None:
    """#314: the structural review prompt ships the anti-slop rubric.

    The structural reviewer is the primary home for the erosion half of the
    rubric (file-size budgets, layering, branching shape), so the same
    self-contained instruction is appended there too.
    """
    p = _paths(tmp_path)
    out = build_structural_prompt(
        skill_invocation="/beagle-core:review-structure",
        files=["api.py"],
        **p,
    )
    assert "anti-slop rubric" in out
    _assert_anti_slop_anchors(out)


def test_per_stack_prompt_anti_slop_rubric_sits_after_skill_invocation(tmp_path: Path) -> None:
    """#314: the rubric lands after the skill invocation so the reviewer applies
    it to the diff hunks it reviews, not ahead of the per-stack instructions."""
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert out.index("anti-slop rubric") > out.index("/beagle-python:review-python")


def test_structural_prompt_anti_slop_rubric_sits_after_verification_protocol(tmp_path: Path) -> None:
    """#314: the rubric lands after the verification-protocol gates in the
    structural prompt, so the reviewer applies the gates first, then the
    anti-slop rubric to the hunks."""
    p = _paths(tmp_path)
    out = build_structural_prompt(
        skill_invocation="/beagle-core:review-structure",
        files=["api.py"],
        **p,
    )
    assert out.index("anti-slop rubric") > out.index("verification-protocol gates")


def test_anti_slop_rubric_severity_layering(tmp_path: Path) -> None:
    """#314: severity is calibrated to medium/low, and pre-existing-and-growing
    erosion is flagged as growth -- guards the over-application failure mode.

    Without the layering awareness, a reviewer would re-flag the whole eroded
    function on every PR instead of isolating the growth the diff introduces.
    """
    p = _paths(tmp_path)
    out = build_per_stack_prompt(
        skill_invocation="/beagle-python:review-python",
        stack_name="python",
        files=["api.py"],
        **p,
    )
    assert "medium/low" in out
    assert "pre-existing-and-growing" in out
    assert "flag the growth, not the whole function" in out

