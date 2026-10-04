"""Deep-mode prompt builder tests (D-09, D-10, D-19, D-20)."""
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream import review_profile as rp, severity
from daydream.deep.diagram_prompts import (
    DIAGRAM_GROUNDING_INSTRUCTION,
    build_diagram_repair_prompt,
    build_flowchart_prompt,
    build_sequence_diagram_prompt,
)
from daydream.deep.diff import _diff_blocks_for_files, bound_deep_diff
from daydream.deep.prompts import (
    ANTI_SLOP_RUBRIC_INSTRUCTION,
    CONFIG_FLOW_TRACE_INSTRUCTION,
    CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION,
    DOC_REVIEW_NOTICE,
    TEST_QUALITY_RUBRIC_INSTRUCTION,
    TRUST_MODEL_INSTRUCTION,
    VERIFICATION_PROTOCOL_INSTRUCTION,
    build_arbiter_prompt,
    build_generic_fallback_prompt,
    build_merge_prompt,
    build_per_stack_prompt,
    build_structural_prompt,
    build_supervise_prompt,
    build_suppression_prompt,
)
from daydream.deep.verification_prompts import build_fix_verify_prompt, build_verification_prompt
from daydream.exploration_runner import count_changed_files
from daydream.extensions import Registry
from daydream.extensions.builtins import _register_builtin_prompts, register_builtins
from daydream.phases import append_extended_facts, build_alternative_review_prompt
from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES
from daydream.prompts.authorial_intent import AUTHORITATIVE_INTENT_RULE, PR_DESCRIPTION_UNTRUSTED_FRAMING
from daydream.prompts.grounding import (
    CWD_GROUNDING_INSTRUCTION,
    UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY,
    render_test_recipe_block,
)
from daydream.test_execution import TestRecipe, resolve_test_recipe
from tests.harness.review_profile import default_strategy as _default_strategy, prompt_paths

_paths = partial(prompt_paths, output_name="stack-python-review.md")

def _arbiter_prompt(tmp_path: Path, **overrides: Any) -> str:
    """Build the arbiter prompt with the shared path set, plus any per-test overrides."""
    p = _paths(tmp_path)
    return build_arbiter_prompt(
        strategy=_default_strategy("arbitration"), arbiter_input_path=tmp_path / "arbiter-input.json",
        diff_path=p["diff_path"], intent_path=p["intent_path"], alternatives_path=p["alternatives_path"], cwd=p["cwd"],
        **overrides,
    )

_REVIEW_BUILDERS: dict[str, Callable[..., str]] = {
    "per_stack": build_per_stack_prompt, "structural": build_structural_prompt,
    "generic_fallback": build_generic_fallback_prompt,
}

def _review_prompt(name: str, tmp_path: Path, **overrides: Any) -> str:
    """Build a real prompt with shared paths and explicit per-builder defaults."""
    kwargs: dict[str, Any] = {**_paths(tmp_path), "strategy": _default_strategy(f"discovery.{name}"),
        "files": ["config.yaml" if name == "generic_fallback" else "api.py"],
    }
    if name == "per_stack":
        kwargs["stack_name"] = "python"
    return _REVIEW_BUILDERS[name](**(kwargs | overrides))

@pytest.mark.parametrize("builder", _REVIEW_BUILDERS)
def test_review_prompt_has_artifact_pointers_and_profile_strategy(builder: str, tmp_path: Path) -> None:
    out = _review_prompt(builder, tmp_path)
    paths = _paths(tmp_path)
    assert str(paths["intent_path"]) in out
    assert str(paths["alternatives_path"]) in out
    assert _default_strategy(f"discovery.{builder}") in out
    assert "/beagle-" not in out and "$review-" not in out

def test_per_stack_prompt_scope_lists_only_stack_files(tmp_path: Path) -> None:
    out = _review_prompt("per_stack", tmp_path, files=["api.py", "lib/util.py"])
    assert "api.py" in out and "lib/util.py" in out
    assert "Do NOT review files from other stacks" in out

@pytest.mark.parametrize("is_docs_only", [False, True])
def test_generic_fallback_docs_notice(tmp_path: Path, is_docs_only: bool) -> None:
    overrides = {"is_docs_only": True, "files": ["README.md"]} if is_docs_only else {}
    out = _review_prompt("generic_fallback", tmp_path, **overrides)
    assert (DOC_REVIEW_NOTICE in out) is is_docs_only
    if is_docs_only:
        assert out.index(DOC_REVIEW_NOTICE) < out.index("Review these files")

@pytest.mark.parametrize("builder", ["per_stack", "generic_fallback"])
def test_prompts_embed_no_full_file_contents(tmp_path: Path, builder: str) -> None:
    """D-09: path-based prompts omit source bodies and retain the diff pointer."""
    p = _paths(tmp_path)
    source_body = "def api():\n    return 'source-only sentinel'\n" + ("x" * 1024)
    (tmp_path / "api.py").write_text(source_body)
    prompt = _review_prompt(builder, tmp_path, files=["api.py"])
    assert f"The full PR diff (base..HEAD) is at {p['diff_path']}." in prompt
    assert source_body not in prompt

@pytest.mark.parametrize("builder,file", [("per_stack", "api.py"), ("generic_fallback", "config.yaml")])
@pytest.mark.parametrize("inline", [False, True])
def test_review_diff_transport(tmp_path: Path, builder: str, file: str, inline: bool) -> None:
    """Inline hunks replace the pointer; bare git diff would hide committed changes."""
    diff = f"diff --git a/{file} b/{file}\n+++ b/{file}\n@@ -1 +1 @@\n-x\n+y\n"
    out = _review_prompt(builder, tmp_path, **({"inline_diff": diff} if inline else {}))
    assert "git diff --no-color -- " + file not in out
    assert "git diff -- " + file not in out
    assert (str(_paths(tmp_path)["diff_path"]) in out) is not inline
    if inline:
        assert "Read it directly" not in out
        assert "-x" in out and "+y" in out

def _merge_paths(tmp_path: Path) -> dict[str, Path | list[Path] | None]:
    return {"per_stack_records_paths": [tmp_path / "python.json", tmp_path / "react.json"],
        "intent_path": tmp_path / "intent.md", "alternatives_path": tmp_path / "alternatives.json",
        "dedup_candidates_path": tmp_path / "dedup.json", "exploration_dir": None, "failed_stacks": None,
    }

@pytest.mark.parametrize("builder", ["per_stack", "generic_fallback"])
@pytest.mark.parametrize("commits",
    [None, "", "abc1234 fix: handle edge case", "abc1234 fix: handle edge case\ndef5678 feat: add retry logic"],
    ids=["none", "empty", "one", "multiple"],
)
def test_review_prompt_prior_commits(tmp_path: Path, builder: str, commits: str | None) -> None:
    out = _review_prompt(builder, tmp_path, prior_commits=commits)
    if commits:
        assert "Prior automated-review commits on this branch" in out
        for commit in commits.splitlines():
            assert commit in out
    else:
        assert "Prior automated-review commits" not in out

def test_merge_prompt_requires_structured_item_fields(tmp_path: Path) -> None:
    """The merge agent emits a structured item list; markdown formatting rules
    (bold-wrapping, head-line layout) no longer apply — Python renders the report.
    """
    out = build_merge_prompt(strategy=_default_strategy("merge"), **_merge_paths(tmp_path))  # type: ignore[arg-type]
    assert '{"items": [' in out
    assert "Item fields (MANDATORY):" in out
    assert "lens" in out
    assert "severity" in out
    # No markdown write-to-file or bold-wrapping directive survives.
    assert "do NOT wrap it in `**...**`" not in out
    assert "write the complete report to" not in out.lower()

def test_merge_prompt_requires_one_path_per_item(tmp_path: Path) -> None:
    """Multi-file concerns must become multiple items, not a comma list in one file field."""
    out = build_merge_prompt(strategy=_default_strategy("merge"), **_merge_paths(tmp_path))  # type: ignore[arg-type]
    assert "EXACTLY ONE path" in out
    assert "separate item per file" in out

def test_build_structural_prompt_has_no_stack_scope_restriction(tmp_path: Path) -> None:
    prompt = _review_prompt("structural", tmp_path, files=["api/main.py", "ui/App.tsx"])
    assert "Focus ONLY on these files" not in prompt
    assert "Do NOT review files from other stacks" not in prompt
    # M12: no skill token may appear in the native structural prompt.
    assert "/beagle-" not in prompt and "beagle" not in prompt.lower()
    assert "Return only the JSON object" in prompt
    assert "Do not write review files" in prompt

def test_build_structural_prompt_references_affected_files(tmp_path: Path) -> None:
    exploration_dir = tmp_path / "exploration"
    prompt = _review_prompt("structural", tmp_path, files=["main.py"], exploration_dir=exploration_dir)
    assert str(exploration_dir / "affected_files.md") in prompt
    prompt_none = _review_prompt("structural", tmp_path, files=["main.py"], exploration_dir=None)
    assert "affected_files.md" not in prompt_none

def test_merge_prompt_does_not_request_structural_findings(tmp_path: Path) -> None:
    """Structural findings are appended by the host (phase_cross_stack_merge) in
    Python, NOT requested via prose; the agent is told not to emit them itself."""
    prompt = build_merge_prompt(strategy=_default_strategy("merge"), **_merge_paths(tmp_path))  # type: ignore[arg-type]
    assert "## Structural Review" not in prompt
    assert "do NOT emit them yourself" in prompt

# Issue #172 — Fix B: read-once inline diff hunks in per-stack / generic prompts

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

# Issue #644 — bound the deep-flow diff at gather time (whole-block retention)

def _blk(path: str, body: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        f"-{body}\n"
        f"+{body.upper()}\n"
    )

def test_bound_deep_diff_under_budget_is_byte_identical() -> None:
    out, info = bound_deep_diff(_DIFF_TWO_FILES)
    assert out == _DIFF_TWO_FILES
    assert not info.truncated
    assert info.marker is None

def test_bound_deep_diff_keeps_whole_blocks_up_to_budget() -> None:
    body = INLINE_DIFF_BUDGET_BYTES // 5  # three whole blocks; exactly two fit under the cap
    diff = _blk("a.py", "x" * body) + _blk("b.py", "y" * body) + _blk("c.py", "z" * body)
    out, info = bound_deep_diff(diff)
    assert info.truncated
    assert "diff --git a/a.py b/a.py" in out
    assert "diff --git a/b.py b/b.py" in out
    # c.py's block is dropped whole (never split mid-stream).
    assert "diff --git a/c.py b/c.py" not in out
    assert "-" + "x" * body in out  # a retained block's hunk body is present, unchanged
    # Every retained block is byte-identical to its source block.
    assert _diff_blocks_for_files(out, ["a.py"]) == _diff_blocks_for_files(diff, ["a.py"])
    assert _diff_blocks_for_files(out, ["b.py"]) == _diff_blocks_for_files(diff, ["b.py"])
    assert info.original_bytes == len(diff.encode("utf-8"))
    assert info.retained_bytes == len(out.encode("utf-8")) - len((info.marker or "").encode("utf-8"))
    assert info.retained_blocks < info.total_blocks
    assert info.dropped_paths == ["c.py"]
    assert "dropped: c.py" in (info.marker or "")

def test_bound_deep_diff_oversize_single_block_kept_whole() -> None:
    """Must-have #5: a single block larger than the cap is kept whole + oversize marker."""
    diff = _blk("huge.py", "y" * (INLINE_DIFF_BUDGET_BYTES + 256))
    out, info = bound_deep_diff(diff)
    assert info.truncated
    assert "diff --git a/huge.py b/huge.py" in out
    assert "huge.py" in info.oversize_paths
    assert info.retained_blocks == 1
    assert info.dropped_paths == []  # nothing follows the kept-whole oversize block

def test_bound_deep_diff_marker_is_parse_safe() -> None:
    # Size from the budget: two of three whole blocks fit, forcing a dropped block.
    body = INLINE_DIFF_BUDGET_BYTES // 5
    diff = _blk("a.py", "x" * body) + _blk("b.py", "y" * body) + _blk("c.py", "z" * body)
    out, info = bound_deep_diff(diff)
    assert info.marker is not None
    assert out.startswith("# daydream: deep diff truncated:")
    # count_changed_files / _diff_changed_files / _diff_blocks_for_files ignore the marker.
    assert count_changed_files(out) == 2
    assert _diff_blocks_for_files(out, ["a.py"]) == _diff_blocks_for_files(diff, ["a.py"])
    assert _diff_blocks_for_files(out, ["c.py"]) is None

def test_diff_blocks_for_files_selects_relevant_hunks() -> None:
    """AC4 helper: ``_diff_blocks_for_files`` returns only the blocks for the
    requested files (post-state path match), concatenated as-is.
    """
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

def test_diff_blocks_for_files_refuses_partial_inline_for_mixed_stack() -> None:
    """Mixed retained/dropped stack blocks require the full diff pointer, avoiding
    an inline review that silently omits that stack's dropped hunks.
    """
    body = INLINE_DIFF_BUDGET_BYTES // 5  # three whole blocks; exactly two fit under the cap
    diff = _blk("a.py", "x" * body) + _blk("b.py", "y" * body) + _blk("c.py", "z" * body)
    bounded, info = bound_deep_diff(diff)
    assert info.truncated
    assert info.dropped_paths == ["c.py"]

    # Fully-retained stack still inlines its complete hunks.
    kept = _diff_blocks_for_files(bounded, ["a.py", "b.py"])
    assert kept is not None
    assert "diff --git a/a.py b/a.py" in kept
    assert "diff --git a/b.py b/b.py" in kept
    # Fully-dropped stack falls back (no blocks present in the bounded text).
    assert _diff_blocks_for_files(bounded, ["c.py"]) is None
    # Mixed stack must NOT get a partial inline of a.py alone.
    assert _diff_blocks_for_files(bounded, ["a.py", "c.py"]) is None
    # A scope file never changed in this PR is not mistaken for a dropped block.
    assert _diff_blocks_for_files(bounded, ["a.py", "never.py"]) is not None

def test_diff_blocks_for_files_returns_none_above_byte_budget() -> None:
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
    out = _diff_blocks_for_files(_DIFF_TWO_FILES, ["nonexistent.py"])
    assert out is None

def test_generic_fallback_prompt_inlines_hunks_and_drops_read_instruction(tmp_path: Path,) -> None:
    """AC4 (unit): generic-fallback prompt with ``inline_diff`` supplied contains
    the inlined hunks, NOT the ``Read it directly`` instruction or diff_path.
    """
    p = _paths(tmp_path)
    inline = _diff_blocks_for_files(_DIFF_TWO_FILES, ["App.tsx"])
    assert inline is not None
    out = build_generic_fallback_prompt(
        strategy=_default_strategy("discovery.generic_fallback"), files=["App.tsx"], inline_diff=inline, **p,
    )
    assert "<div>universe</div>" in out
    assert "Read it directly" not in out
    assert str(p["diff_path"]) not in out

def test_structural_prompt_keeps_diff_pointer_and_read_freedom(tmp_path: Path) -> None:
    """Structural review retains its diff pointer and repo-wide Read/Grep/Bash scope."""
    p = _paths(tmp_path)
    out = _review_prompt("structural", tmp_path, files=["api.py"])
    assert _default_strategy("discovery.structural") in out
    assert str(p["diff_path"]) in out  # keeps its pointer
    assert "Read it directly" in out   # structural prompt unchanged

# Issue #221 — cwd grounding injected into every deep prompt builder

@pytest.mark.parametrize("builder", _REVIEW_BUILDERS)
def test_review_prompt_contains_cwd_grounding(tmp_path: Path, builder: str) -> None:
    out = _review_prompt(builder, tmp_path)
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=tmp_path) in out
    assert str(tmp_path) in out

def test_arbiter_prompt_contains_cwd_grounding(tmp_path: Path) -> None:
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=tmp_path) in _arbiter_prompt(tmp_path)

def test_arbiter_prompt_instructs_collapsing_duplicate_findings(tmp_path: Path) -> None:
    """Duplicate instructions reject one twin and retain the other's severity.

    Selecting both records alone would allow two keep verdicts to preserve
    the duplicate with unchanged severities.
    """
    out = _arbiter_prompt(tmp_path)
    assert "the same defect, keep exactly one" in out
    assert "`keep: false` on the redundant entry" in out
    assert "higher of the two severities" in out
    # The whole-file anchor is called out, because half of these pairs arrive
    # with one side at `line: 0` rather than at the cited line.
    assert "`line: 0`" in out
    # Overlap alone must not be grounds for rejection -- the arbiter must not
    # start pruning neighbouring findings that merely touch the same code.
    assert "never reject a finding merely for overlapping" in out

def _verification_prompt(tmp_path: Path, *, items: list[dict[str, Any]] | None = None, **overrides: Any) -> str:
    """Build the verifier prompt with the shared one-item finding, cwd and output path."""
    return build_verification_prompt(strategy=_default_strategy("verification"),
        items=[{"id": 1, "lens": "per-stack", "severity": "high", "file": "api.py",
                "line": 10, "description": "x", "rationale": "y"}] if items is None else items,
        cwd=tmp_path, output_path=tmp_path / "verdicts.json",
        **overrides,
    )

def test_verification_prompt_contains_cwd_grounding(tmp_path: Path) -> None:
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=tmp_path) in _verification_prompt(tmp_path)

@pytest.mark.parametrize("builder", _REVIEW_BUILDERS)
def test_review_prompt_includes_verification_protocol(tmp_path: Path, builder: str) -> None:
    prompt = _review_prompt(builder, tmp_path)
    assert VERIFICATION_PROTOCOL_INSTRUCTION in prompt
    assert "verification gates" in prompt
    assert "anchor" in prompt
    assert "evidence" in prompt
    assert "SKILL.md" not in prompt
    assert "/skill:" not in prompt
    assert "/review-verification-protocol" not in prompt
    assert "$review-verification-protocol" not in prompt

def test_build_verification_prompt_includes_gate_zero_echo(tmp_path: Path) -> None:
    out = _verification_prompt(tmp_path,
        items=[{"id": "1", "file": "x.py", "line": 10, "description": "Test finding"}],
    )
    assert "Gate-0" in out or "anti-confabulation" in out
    assert "same-turn echo" in out or "file:line" in out

# Issue #279 — Authoritative-intent rule gate in the deep prompt builders

def _build_gated(name: str, tmp_path: Path, *, intent_authoritative: bool) -> str:
    """Call each named builder with its distinct minimal required inputs."""
    if name.replace("-", "_") in _REVIEW_BUILDERS:
        return _review_prompt(name.replace("-", "_"), tmp_path, intent_authoritative=intent_authoritative)
    if name == "arbiter":
        return _arbiter_prompt(tmp_path, intent_authoritative=intent_authoritative)
    if name == "merge":
        return build_merge_prompt(strategy=_default_strategy("merge"),
            per_stack_records_paths=[tmp_path / "python.json", tmp_path / "react.json"],
            intent_path=tmp_path / "intent.md", alternatives_path=tmp_path / "alternatives.json",
            dedup_candidates_path=tmp_path / "dedup.json", intent_authoritative=intent_authoritative,
        )
    msg = f"unknown builder name: {name!r}"
    raise ValueError(msg)

@pytest.mark.parametrize("name", ["per-stack", "structural", "generic-fallback", "arbiter", "merge"])
def test_authoritative_intent_rule_is_gated(name: str, tmp_path: Path) -> None:
    """#279: the precedence rule appears only when a fresh PR body was ingested."""
    assert AUTHORITATIVE_INTENT_RULE not in _build_gated(name, tmp_path, intent_authoritative=False)
    assert AUTHORITATIVE_INTENT_RULE in _build_gated(name, tmp_path, intent_authoritative=True)
    # NEW #579: the untrusted framing rides with the rule — same gating.
    assert PR_DESCRIPTION_UNTRUSTED_FRAMING not in _build_gated(name, tmp_path, intent_authoritative=False)
    assert PR_DESCRIPTION_UNTRUSTED_FRAMING in _build_gated(name, tmp_path, intent_authoritative=True)

def test_verification_prompt_has_no_schema_dump_or_write_instruction(tmp_path: Path) -> None:
    """The backend receives output_schema and the host persists verdicts; prompts
    must not duplicate schema dumps or request agent-written verdict files.
    """
    prompt = _verification_prompt(tmp_path)

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

def test_verification_prompt_advertises_full_read_only_bash_allowlist(tmp_path: Path) -> None:
    """The verifier's advertised Bash commands must be rendered from the enforced
    single source — never a hand-written partial list — and must not instruct a
    shell command the read-only guard denies."""

    prompt = _verification_prompt(tmp_path)

    # Pin literal commands independently of the renderer to catch omissions.
    assert ("`ls`, `cat`, `git status`, `git log`, `git show`, `git blame`, `git diff`" in prompt)
    # The stale partial list is gone, no shell-grep step remains, and the
    # read-only clause an existing test pins survives.
    assert "`git`, `cat`, `ls`" not in prompt
    assert "grep -rn" not in prompt
    assert "Do NOT write, edit, or move files." in prompt

# --- Task 12a: the per-stack prompt path can omit the alternatives pointer ----

@pytest.mark.parametrize("builder", _REVIEW_BUILDERS)
def test_review_prompt_can_omit_alternatives(tmp_path: Path, builder: str) -> None:
    with_alts = _review_prompt(builder, tmp_path, include_alternatives=True)
    without = _review_prompt(builder, tmp_path, include_alternatives=False)
    assert "alternatives.json" in with_alts
    assert "alternatives.json" not in without
    assert "intent.md" in without
    assert _review_prompt(builder, tmp_path) == with_alts

def test_omitting_alternatives_keeps_authoritative_intent_rule(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    without = build_per_stack_prompt(strategy=_default_strategy("discovery.per_stack"), stack_name="python",
        files=["api.py"], **p, intent_authoritative=True, include_alternatives=False,
    )
    assert "alternatives.json" not in without
    assert "author's stated intent from the pull-request description" in without
    assert AUTHORITATIVE_INTENT_RULE in without
    assert PR_DESCRIPTION_UNTRUSTED_FRAMING in without  # NEW #579

def test_adjudication_builders_keep_alternatives_unconditionally(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    for prompt in _adjudication_prompts(tmp_path).values():
        assert str(p["alternatives_path"]) in prompt

# Issue #308 — test-quality rubric in the per-stack review prompt

def test_per_stack_prompt_includes_test_quality_rubric(tmp_path: Path) -> None:
    """#308: test quality includes behavior, determinism, and portability."""
    out = _review_prompt("per_stack", tmp_path)
    assert "test-quality rubric" in out
    assert "vacuous assertions" in out
    assert "observable consequences" in out
    assert "canonical public path" in out
    assert "deterministic" in out
    assert "`#[cfg]`" in out

def test_per_stack_prompt_test_quality_rubric_layering_awareness(tmp_path: Path) -> None:
    """Legitimate pure propagation tests pass; seams that bypass their claimed
    observable behavior trigger the rubric.
    """
    out = _review_prompt("per_stack", tmp_path)
    assert "pure-function seams" in out
    assert "`build_driver_request`" in out
    assert "internal-field assertion" in out
    assert "bypasses the observable behavior" in out

def test_per_stack_prompt_test_quality_rubric_follows_strategy(tmp_path: Path) -> None:
    out = _review_prompt("per_stack", tmp_path)
    assert out.index("test-quality rubric") > out.index(_default_strategy("discovery.per_stack"))

# Issue #314 — anti-slop review rubric (structural erosion + verbosity patterns)

_ANTI_SLOP_ANCHORS = ("concrete maintenance consequence", "established repository convention", "canonical helpers",
    "Size, single-use variables, and wrappers alone do not establish a defect", "medium/low",
    "newly introduced or worsened",
)

@pytest.mark.parametrize("builder", ["per_stack", "structural"])
def test_review_prompt_includes_anti_slop_rubric(tmp_path: Path, builder: str) -> None:
    """#314: both reviewers retain the complete maintainability rubric."""
    out = _review_prompt(builder, tmp_path)
    assert "maintainability rubric" in out
    _assert_anchors(out, _ANTI_SLOP_ANCHORS)

@pytest.mark.parametrize(
    "builder,preceding", [("per_stack", "discovery.per_stack"), ("structural", "verification gates")],
)
def test_anti_slop_rubric_order(tmp_path: Path, builder: str, preceding: str) -> None:
    out = _review_prompt(builder, tmp_path)
    anchor = _default_strategy(preceding) if preceding.startswith("discovery.") else preceding
    assert out.index("maintainability rubric") > out.index(anchor)

@pytest.mark.parametrize("builder", ["per_stack", "structural"])
def test_anti_slop_rubric_severity_and_scope(tmp_path: Path, builder: str) -> None:
    """#314: pre-existing growth changes scope, never the medium/low ceiling."""
    out = _review_prompt(builder, tmp_path)
    assert "never high" in out
    assert "unless the erosion is pre-existing-and-growing" not in out
    assert "medium/low" in out
    assert "newly introduced or worsened" in out
    assert "scoped to this diff's contribution" in out

# =============================================================================
# Issue #310 — cross-file verification instruction (symbol existence,
# config-flow traces, trust-model checks)
# =============================================================================

_CROSS_FILE_ANCHORS = (
    "defined OUTSIDE the diff", "subcommand invoked by a CLI wrapper", "trait method implemented by generated code",
    "`rg`", "downgrade the finding's confidence", "call site alone",
)

_CONFIG_TRACE_ANCHORS = (
    "config struct", "driver config", "request construction", "investigation method, not extra output", "silent drops",
    "double-resolves", "TOCTOU",
)

_TRUST_MODEL_ANCHORS = (
    "cache-control", "trust boundaries", "untrusted party", "honor the boundary", "retain or forward sensitive content",
    "escaping", "credential forwarding",
)

def _assert_anchors(out: str, anchors: tuple[str, ...]) -> None:
    missing = [anchor for anchor in anchors if anchor not in out]
    assert not missing, f"missing pinned anchors: {missing}"

@pytest.mark.parametrize("builder", _REVIEW_BUILDERS)
@pytest.mark.parametrize("heading,anchors,owners",
    [("Cross-file symbol existence check", _CROSS_FILE_ANCHORS, {"structural"}),
        ("Config/env flow trace", _CONFIG_TRACE_ANCHORS, {"per_stack", "generic_fallback"}),
        ("Trust-model check", _TRUST_MODEL_ANCHORS, set(_REVIEW_BUILDERS)),
    ], ids=["symbols", "config", "trust"],
)
def test_review_instruction_ownership(
    tmp_path: Path, builder: str, heading: str, anchors: tuple[str, ...], owners: set[str],
) -> None:
    """#310: shared trust checks coexist with each reviewer's specific duties."""
    out = _review_prompt(builder, tmp_path)
    if builder in owners:
        assert heading in out
        _assert_anchors(out, anchors)
    else:
        assert heading not in out
        leaked = [anchor for anchor in anchors if anchor in out]
        assert not leaked, f"instruction anchors leaked into {builder}: {leaked}"

def test_cross_file_additions_keep_existing_rubrics(tmp_path: Path) -> None:
    structural = _review_prompt("structural", tmp_path)
    per_stack = _review_prompt("per_stack", tmp_path)
    assert TEST_QUALITY_RUBRIC_INSTRUCTION in per_stack
    assert TEST_QUALITY_RUBRIC_INSTRUCTION not in structural
    for out in (structural, per_stack):
        assert ANTI_SLOP_RUBRIC_INSTRUCTION in out
        _assert_anchors(out, _ANTI_SLOP_ANCHORS)

def test_cross_file_instructions_contain_no_banned_words() -> None:
    """#310: the new instruction text avoids banned vocabulary
    (defer/TODO/partial/future/TBD/out of scope/follow-up) and names no external
    review tools."""
    text = CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION
    text += CONFIG_FLOW_TRACE_INSTRUCTION
    text += TRUST_MODEL_INSTRUCTION
    lower = text.lower()
    banned = ("defer", "todo", "partial", "future", "tbd", "out of scope", "follow-up")
    found = [word for word in banned if word in lower]
    assert not found, f"banned words leaked into cross-file instruction text: {found}"

# --- Issue #731: coverage-evidence grounding + frontier-read instruction ---

def test_per_stack_prompt_instructs_frontier_read(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    prompt = build_per_stack_prompt(strategy=_default_strategy("discovery.per_stack"),
        stack_name="python#0",
        files=["a.py"], frontier_files=["shared/iface.py"], **p,
    )
    assert "shared/iface.py" in prompt
    assert "cross-shard" in prompt or "interface file" in prompt

def test_diff_instruction_mandates_read_first(tmp_path: Path) -> None:
    """Reviewers read source context before judging changed behavior."""
    p = _paths(tmp_path)
    per_stack = build_per_stack_prompt(strategy=_default_strategy("discovery.per_stack"), stack_name="python",
        files=["api.py"], inline_diff="@@ -1 +1 @@\n-'x'\n+'y'\n", **p,
    )
    fallback = build_generic_fallback_prompt(strategy=_default_strategy("discovery.generic_fallback"),
        files=["config.yaml"], inline_diff="@@ -1 +1 @@\n-'x'\n+'y'\n", **p,
    )
    for prompt in (per_stack, fallback):
        assert "you MAY Read the source files directly" not in prompt
        assert "Read the source file FIRST" in prompt

def test_exploration_pointer_distinguishes_exploration_from_assigned_sources(tmp_path: Path) -> None:
    """Bounded-context exploration pointers never carry the assigned-source mandate."""
    p = _paths(tmp_path)
    out = build_per_stack_prompt(strategy=_default_strategy("discovery.per_stack"), stack_name="python",
        files=["api.py"], exploration_dir=tmp_path / ".daydream" / "exploration", **p,
    )
    # Exploration artifacts are pointed at as bounded context only: two named
    # files, sibling artifacts explicitly out of scope.
    assert "Read the pre-scan summary at" in out
    assert str(tmp_path / ".daydream" / "exploration" / "summary.md") in out
    assert "Do not infer or enumerate sibling artifact files" in out
    assert "assigned source files" in out
    assert "MUST read in full all assigned source files" not in out
    assert "enclosing symbol or configuration section" in out
    # The exploration read bound must not include the assigned-source mandate.
    bounded = out[out.index("Read the pre-scan summary at"):]
    bounded = bounded[: bounded.index("\n")]
    assert "assigned source files" not in bounded

def test_per_stack_prompt_uses_profile_strategy_and_no_skill() -> None:
    strategy = rp.build_default_profile().strategies["discovery.per_stack"].content
    p = build_per_stack_prompt(strategy=strategy, stack_name="python", files=["a.py"],
        diff_path=Path("/d"), intent_path=Path("/i"), alternatives_path=Path("/a"),
        output_path=Path("/o"), cwd=Path("/c"),
    )
    assert "Review the changed behavior assigned to this stack" in p
    assert "python" in p and "a.py" in p and "/d" in p and "/c" in p
    assert "/beagle-" not in p and "$review-" not in p and "/skill:" not in p
    assert "Apply this specialist skill" not in p

def test_structural_prompt_uses_profile_strategy_and_no_skill() -> None:
    strategy = rp.build_default_profile().strategies["discovery.structural"].content
    p = build_structural_prompt(strategy=strategy, files=["a.py", "b.ts"], diff_path=Path("/d"),
        intent_path=Path("/i"), alternatives_path=Path("/a"), output_path=Path("/o"), cwd=Path("/c"),
    )
    assert "Review the repository-wide interactions" in p
    assert "a.py, b.ts" in p and "/d" in p and "/c" in p
    assert "/beagle-" not in p and "Apply this specialist skill" not in p

# Issue #972 R1 — host-owned severity rubric reaches every assigning prompt

def _rubric_assigning_prompts(tmp_path: Path) -> list[str]:

    p = _paths(tmp_path)
    per_stack = build_per_stack_prompt(
        strategy=_default_strategy("discovery.per_stack"), stack_name="python", files=["api.py"], **p,
    )
    structural = build_structural_prompt(strategy=_default_strategy("discovery.structural"), files=["api.py"], **p,)
    generic = build_generic_fallback_prompt(
        strategy=_default_strategy("discovery.generic_fallback"), files=["api.py"], **p,
    )
    alternative = build_alternative_review_prompt(
        strategy=_default_strategy("alternatives"), intent_summary="intent", diff_path=str(p["diff_path"]),
    )
    return [per_stack, structural, generic, alternative]

def test_every_assigning_prompt_carries_severity_rubric(tmp_path: Path) -> None:
    for prompt in _rubric_assigning_prompts(tmp_path):
        assert severity.SEVERITY_RUBRIC in prompt

def test_rubric_is_high_definition_not_a_prohibition() -> None:
    """R1.2: the rubric defines all three levels in checkable terms, high first."""
    rubric = severity.SEVERITY_RUBRIC
    assert "high" in rubric and "medium" in rubric and "low" in rubric
    assert rubric.index("high") < rubric.index("medium") < rubric.index("low")
    assert "Informational" not in rubric

def test_rubric_after_strategy_text(tmp_path: Path) -> None:
    for prompt, strategy_stage in (
        (0, "discovery.per_stack"), (1, "discovery.structural"), (2, "discovery.generic_fallback"),
    ):
        prompts = _rubric_assigning_prompts(tmp_path)
        assert prompts[prompt].index(severity.SEVERITY_RUBRIC) > prompts[prompt].index(_default_strategy(strategy_stage)
        )

# Issue #972 R1.3 — adjudication restatements cite the severity rubric

def _adjudication_prompts(tmp_path: Path) -> dict[str, str]:
    p = _paths(tmp_path)
    return {"build_arbiter_prompt": build_arbiter_prompt(
            strategy=_default_strategy("arbitration"), arbiter_input_path=tmp_path / "arbiter-input.json",
            diff_path=p["diff_path"], intent_path=p["intent_path"], alternatives_path=p["alternatives_path"],
            cwd=p["cwd"],
        ),
        "build_suppression_prompt": build_suppression_prompt(
            strategy=_default_strategy("suppression"), suppression_input_path=tmp_path / "suppression-input.json",
            diff_path=p["diff_path"], intent_path=p["intent_path"], alternatives_path=p["alternatives_path"],
            cwd=p["cwd"],
        ),
        "build_merge_prompt": build_merge_prompt(strategy=_default_strategy("merge"),
            **(_merge_paths(tmp_path) | {"alternatives_path": p["alternatives_path"]}),  # type: ignore[arg-type]
        ),
        "build_supervise_prompt": build_supervise_prompt(
            strategy=_default_strategy("supervision"), supervise_input_path=tmp_path / "supervise-input.json",
            diff_path=p["diff_path"], intent_path=p["intent_path"], alternatives_path=p["alternatives_path"],
            cwd=p["cwd"],
        ),
    }

@pytest.mark.parametrize(
    "builder", ["build_arbiter_prompt", "build_suppression_prompt", "build_merge_prompt", "build_supervise_prompt"],
)
def test_adjudication_prompts_reference_severity_rubric(builder: str, tmp_path: Path) -> None:
    prompt = _adjudication_prompts(tmp_path)[builder]
    assert severity.SEVERITY_RUBRIC in prompt or "severity rubric" in prompt

@pytest.mark.parametrize(
    "builder", ["build_arbiter_prompt", "build_suppression_prompt", "build_merge_prompt", "build_supervise_prompt"],
)
def test_adjudication_prompts_no_divergent_severity_fragment(builder: str, tmp_path: Path) -> None:
    assert "high | medium" not in _adjudication_prompts(tmp_path)[builder]

# Issue #1113 — grounded diagram prompts

def _diagram_schema() -> dict[str, object]:
    """A stand-in spec schema: the builders take the schema as a kwarg."""
    return {"type": "object", "properties": {"participants": {"type": "array"}}, "required": ["participants"],
        "additionalProperties": False,
    }

def _sequence_prompt(tmp_path: Path, **overrides: object) -> str:
    kwargs: dict[str, object] = {"diff_path": tmp_path / ".daydream" / "diff.patch", "inline_diff": None,
        "files_by_module": {"api": ["api/handler.py"], "core": ["core/resolve.py", "core/db.py"]}, "cwd": tmp_path,
        "exploration_dir": tmp_path / ".daydream" / "exploration", "schema": _diagram_schema(),
    }
    kwargs.update(overrides)
    return build_sequence_diagram_prompt(**kwargs)  # type: ignore[arg-type]

def _flowchart_prompt(tmp_path: Path, **overrides: object) -> str:
    kwargs: dict[str, object] = {"diff_path": tmp_path / ".daydream" / "diff.patch", "inline_diff": None,
        "candidate_roots": [
            {"file": "core/resolve.py", "name": "resolve_identity", "line": 22, "end_line": 71, "branch_points": 4},
            {"file": "api/handler.py", "name": "handle", "line": 8, "end_line": 30, "branch_points": 3},
        ], "forced": False, "cwd": tmp_path, "exploration_dir": tmp_path / ".daydream" / "exploration",
        "schema": _diagram_schema(),
    }
    kwargs.update(overrides)
    return build_flowchart_prompt(**kwargs)  # type: ignore[arg-type]

_FAILURES: list[dict[str, Any]] = [{"element": "message", "ref": "2", "reason": "FILE_MISSING", "grounded": False},
    {"element": "participant", "ref": "Identity Resolver", "reason": "PARTICIPANT_FILE_MISSING"},
    {"element": "root", "ref": "core/resolve.py:resolve_identity", "reason": "ROOT_NOT_CANDIDATE"},
]

def _repair_prompt(kind: str, **overrides: Any) -> str:
    """Build the repair prompt with the shared failure list and schema, no candidate roots."""
    kwargs: dict[str, Any] = {"kind": kind, "failures": _FAILURES, "candidate_roots": None,
        "schema": _diagram_schema()}
    return build_diagram_repair_prompt(**(kwargs | overrides))

def _diagram_prompts(tmp_path: Path) -> dict[str, str]:
    """The three diagram prompt families keyed by the parametrised builder name."""
    return {"sequence": _sequence_prompt(tmp_path), "flowchart": _flowchart_prompt(tmp_path),
        "repair": _repair_prompt("flowchart")}

def test_sequence_prompt_clone_mode_inlines_exploration_under_boundary(tmp_path: Path) -> None:
    """Issue #1123: clone mode renders inline exploration + dependency content
    inside the untrusted-content boundary and never a .daydream pointer."""
    prompt = _sequence_prompt(tmp_path,
        inline_exploration="## Summary\n| 3 files indexed |",
        inline_dependencies="api/handler -> core/resolve",
        exploration_dir=None,  # clone mode suppresses pointers
        clone_mode=True,
    )
    assert "## Summary" in prompt
    assert "api/handler -> core/resolve" in prompt
    assert UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY in prompt
    assert ".daydream/exploration" not in prompt
    assert "affected_files.md" not in prompt  # no pointer names
    assert "diff.patch" not in prompt  # inline_diff=None + clone_mode → no pointer degrade

def test_diagram_prompts_clone_mode_truncates_oversized_diff_with_marker(tmp_path: Path) -> None:
    """Clone mode has no on-disk diff fallback: an over-budget diff is inlined
    truncated with the explicit marker, never a .daydream/diff.patch pointer."""
    big = "x" * (INLINE_DIFF_BUDGET_BYTES + 1)
    for prompt in (_sequence_prompt(tmp_path, inline_diff=big, clone_mode=True),
        _flowchart_prompt(tmp_path, inline_diff=big, clone_mode=True),
    ):
        # Task 8 moved the marker inside the shared budget, so the retained
        # diff prefix is the budget minus the banner and marker bytes.
        assert big[: INLINE_DIFF_BUDGET_BYTES // 2] in prompt
        assert "[diff truncated to fit the prompt budget]" in prompt
        assert "diff.patch" not in prompt

def test_diagram_prompts_clone_mode_truncation_is_byte_accurate(tmp_path: Path) -> None:
    """The clone-mode truncation slices UTF-8 bytes, not characters, so a
    multibyte over-budget diff cannot exceed INLINE_DIFF_BUDGET_BYTES."""
    big = "é" * (INLINE_DIFF_BUDGET_BYTES // 2 + 100)  # 2 bytes per char
    for prompt in (_sequence_prompt(tmp_path, inline_diff=big, clone_mode=True),
        _flowchart_prompt(tmp_path, inline_diff=big, clone_mode=True),
    ):
        assert "[diff truncated to fit the prompt budget]" in prompt
        assert "diff.patch" not in prompt
        body = prompt.split("is inlined below:\n\n", 1)[1].split("\n[diff truncated", 1)[0]
        assert len(body.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES

def test_diagram_prompts_non_clone_keeps_pointer_behavior_byte_for_byte(tmp_path: Path) -> None:
    """Non-disposable backends keep host-path pointers (never inline): default
    kwargs (clone_mode=False) name the exploration files and the diff pointer,
    including the over-budget degrade."""
    prompt = _sequence_prompt(tmp_path, inline_diff="x" * (INLINE_DIFF_BUDGET_BYTES + 1))
    assert "diff.patch" in prompt  # pointer degrade intact
    assert "[diff truncated" not in prompt
    assert "Read the pre-scan summary at" in prompt  # host-path exploration pointer
    assert str(tmp_path / ".daydream" / "exploration" / "summary.md") in prompt
    assert "dependencies.md" in prompt  # dependency-edge pointer intact

def test_diagram_prompts_missing_inline_exploration_omits_block(tmp_path: Path) -> None:
    """clone_mode with no readable exploration content omits the block entirely —
    no placeholder, no marker, boundary still present."""
    for prompt in (_sequence_prompt(tmp_path, clone_mode=True), _flowchart_prompt(tmp_path, clone_mode=True),):
        assert UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY in prompt
        assert "truncated" not in prompt
        assert ".daydream" not in prompt

def test_diagram_prompts_open_with_their_role_sentence(tmp_path: Path) -> None:
    """D17: the role sentence is the prompt's opening text (the stub dispatches on it)."""
    assert _sequence_prompt(tmp_path).startswith("You are the sequence-diagram author for this pull request.")
    assert _flowchart_prompt(tmp_path).startswith("You are the flowchart author for this pull request.")
    assert _repair_prompt("flowchart").startswith("Diagram repair turn (flowchart):")

def test_diagram_grounding_instruction_states_the_three_host_guarantees() -> None:
    """The contract describes source validation, pruning, and host rendering."""
    text = DIAGRAM_GROUNDING_INSTRUCTION
    assert "Inspect the source for every element" in text
    assert "successful structured Read" not in text
    assert "trajectory" not in text
    assert "verifies every file:line you emit deterministically" in text
    assert "dropped from the rendered diagram" in text
    assert "never write mermaid" in text

@pytest.mark.parametrize("builder", ["sequence", "flowchart", "repair"])
def test_diagram_prompts_carry_the_grounding_instruction(builder: str, tmp_path: Path) -> None:
    assert DIAGRAM_GROUNDING_INSTRUCTION in _diagram_prompts(tmp_path)[builder]

def test_diagram_grounding_instruction_not_delivered_to_review_prompts(tmp_path: Path) -> None:
    per_stack = build_per_stack_prompt(
        strategy=_default_strategy("discovery.per_stack"), stack_name="python", files=["api/handler.py"],
        **_paths(tmp_path),
    )
    assert DIAGRAM_GROUNDING_INSTRUCTION not in per_stack

@pytest.mark.parametrize("builder", ["sequence", "flowchart"])
def test_diagram_prompts_inline_a_small_diff(builder: str, tmp_path: Path) -> None:
    diff = "diff --git a/api/handler.py b/api/handler.py\n@@ -1 +1 @@\n-old\n+new\n"
    build = _sequence_prompt if builder == "sequence" else _flowchart_prompt
    prompt = build(tmp_path, inline_diff=diff)
    assert "+new" in prompt
    assert "do NOT re-Read diff.patch" in prompt
    assert str(tmp_path / ".daydream" / "diff.patch") not in prompt

@pytest.mark.parametrize("builder", ["sequence", "flowchart"])
def test_diagram_prompts_fall_back_to_the_diff_pointer(builder: str, tmp_path: Path) -> None:
    oversized = "x" * (INLINE_DIFF_BUDGET_BYTES + 1)
    build = _sequence_prompt if builder == "sequence" else _flowchart_prompt
    for inline in (None, oversized):
        prompt = build(tmp_path, inline_diff=inline)
        assert str(tmp_path / ".daydream" / "diff.patch") in prompt
        assert "The full PR diff (base..HEAD) is at" in prompt
        assert oversized not in prompt

def test_sequence_prompt_groups_changed_files_by_module(tmp_path: Path) -> None:
    """Participants must align with real boundaries, so the modules are enumerated."""
    prompt = _sequence_prompt(tmp_path)
    assert "grouped by module/service" in prompt
    assert "- api\n    - api/handler.py" in prompt
    assert "- core\n    - core/resolve.py\n    - core/db.py" in prompt

def test_sequence_prompt_states_the_spec_rules(tmp_path: Path) -> None:
    """Spec section 2: participant/message/block rules and the render floor."""
    prompt = _sequence_prompt(tmp_path)
    assert "3 to 10 entries" in prompt
    assert "`internal` | `external`" in prompt
    assert "≤ 80" in prompt
    assert "`call` | `reply` | `self`" in prompt
    assert "place it immediately after the reversed `call`" in prompt
    assert "the FIRST message" in prompt
    assert "`alt` (2 or more branches)" in prompt
    assert "0-based indices" in prompt
    assert "at least 3 grounded messages" in prompt

def test_flowchart_prompt_lists_candidate_roots_with_ranges_and_counts(tmp_path: Path) -> None:
    """The model may only pick a root from this list, so ranges and counts are shown."""
    prompt = _flowchart_prompt(tmp_path)
    assert "- `resolve_identity` in core/resolve.py, lines 22-71, 4 changed branch point(s)" in prompt
    assert "- `handle` in api/handler.py, lines 8-30, 3 changed branch point(s)" in prompt
    assert "MUST be one of these" in prompt
    assert "explicitly requested" not in prompt

def test_flowchart_prompt_forced_names_the_widened_candidate_list(tmp_path: Path) -> None:
    """forced=True: the list is every changed function, not only threshold-meeting ones."""
    prompt = _flowchart_prompt(tmp_path, forced=True)
    assert "explicitly requested" in prompt
    assert "every changed function rather than only those meeting the branch-point threshold" in prompt
    assert "rather than inventing branches" in prompt

def test_flowchart_prompt_states_the_spec_rules(tmp_path: Path) -> None:
    """Spec section 2: node kinds, per-kind evidence, edge labels, and the floor."""
    prompt = _flowchart_prompt(tmp_path)
    assert "4 to 25 entries" in prompt
    assert "`start` | `end` | `process` | `decision` | `subroutine` | `io`" in prompt
    assert "≤ 60" in prompt
    assert "cites the CALL SITE" in prompt
    assert "at least 2 outgoing edges with distinct labels" in prompt
    assert "Exactly one `start` node; at least one `end` node" in prompt
    assert "at least 4 grounded nodes" in prompt

def test_flowchart_prompt_has_no_candidate_roots(tmp_path: Path) -> None:
    """An empty candidate list renders as ``(none)`` rather than an empty section."""
    prompt = _flowchart_prompt(tmp_path, candidate_roots=[], forced=True)
    assert "- (none)" in prompt

def test_repair_prompt_lists_every_failure_with_its_reason_code() -> None:
    prompt = _repair_prompt("sequence")
    assert "- message `2`: FILE_MISSING" in prompt
    assert "- participant `Identity Resolver`: PARTICIPANT_FILE_MISSING" in prompt
    assert "- root `core/resolve.py:resolve_identity`: ROOT_NOT_CANDIDATE" in prompt

def test_repair_prompt_states_the_repair_contract() -> None:
    """Correct or remove, return the full spec, exactly one repair turn."""
    prompt = _repair_prompt("sequence")
    assert "Correct its evidence" in prompt
    assert "Remove the element from the spec entirely" in prompt
    assert "Return the FULL corrected spec in the same JSON shape" in prompt
    assert "ONLY repair turn" in prompt
    assert "ROOT_NOT_CANDIDATE` verdict" not in prompt

def test_repair_prompt_repeats_the_candidate_list_for_a_root_repick() -> None:
    prompt = _repair_prompt("flowchart", candidate_roots=[
            {"file": "core/resolve.py", "name": "resolve_identity", "line": 22, "end_line": 71, "branch_points": 4}
        ])
    assert "`ROOT_NOT_CANDIDATE` verdict means the root you chose is not in the candidate list" in prompt
    assert "Re-pick a root from the list below" in prompt
    assert "- `resolve_identity` in core/resolve.py, lines 22-71, 4 changed branch point(s)" in prompt
    assert "explicitly requested" not in prompt

@pytest.mark.parametrize("builder", ["sequence", "flowchart", "repair"])
def test_diagram_prompts_embed_the_output_schema(builder: str, tmp_path: Path) -> None:
    prompt = _diagram_prompts(tmp_path)[builder]
    assert "Return ONLY a JSON object matching this schema:" in prompt
    assert json.dumps(_diagram_schema(), indent=2) in prompt

@pytest.mark.parametrize("builder", ["sequence", "flowchart"])
def test_diagram_prompts_ground_cwd_and_fence_untrusted_content(builder: str, tmp_path: Path) -> None:
    """Shared scaffold: cwd grounding, the untrusted-content boundary, exploration pointers."""
    build = _sequence_prompt if builder == "sequence" else _flowchart_prompt
    prompt = build(tmp_path)
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=tmp_path) in prompt
    assert UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY in prompt
    assert str(tmp_path / ".daydream" / "exploration" / "affected_files.md") in prompt
    assert str(tmp_path / ".daydream" / "exploration") + "/dependencies.md" in prompt

@pytest.mark.parametrize("builder", ["sequence", "flowchart"])
def test_diagram_prompts_without_exploration_keep_the_content_boundary(builder: str, tmp_path: Path) -> None:
    """No exploration directory still fences repository content as untrusted data."""
    build = _sequence_prompt if builder == "sequence" else _flowchart_prompt
    prompt = build(tmp_path, exploration_dir=None)
    assert UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY in prompt
    assert "Pre-scan exploration" not in prompt

def test_diagram_prompt_names_are_registered() -> None:
    reg = Registry()
    register_builtins(reg)
    assert "diagram_sequence" in reg.prompt_names()
    assert "diagram_flowchart" in reg.prompt_names()
    assert reg.prompt("diagram_sequence") is build_sequence_diagram_prompt
    assert reg.prompt("diagram_flowchart") is build_flowchart_prompt

def test_registry_diagram_prompt_override_accepts_inline_kwargs(tmp_path: Path) -> None:
    """Issue #1123 planning spike: the builtins override of diagram_sequence must
    pass new inline kwargs through the registry indirection unchanged."""
    reg = Registry()
    _register_builtin_prompts(reg)
    prompt = reg.prompt("diagram_sequence")(
        diff_path=tmp_path / "d.patch", inline_diff=None, inline_exploration=None, inline_dependencies=None,
        clone_mode=True, files_by_module={}, cwd=tmp_path, exploration_dir=None,
        schema={"type": "object", "properties": {}},
    )
    assert isinstance(prompt, str)

def test_fix_verify_prompt_audits_complete_retained_patch_and_all_findings(tmp_path: Path) -> None:
    """The compatible prompt contract is stage-neutral and final-tree aware."""
    prompt = build_fix_verify_prompt(items=[{"id": 1, "description": "fix it", "file": "a.py", "line": 1}],
        changed_hunks="diff --git a/a.py b/a.py\n",
        cwd=tmp_path, round_number=2,
    )
    assert "complete current retained patch" in prompt
    assert "all canonical findings" in prompt
    assert "round's changed hunks ONLY" not in prompt
    assert "findings the round dispatched" not in prompt

def _resolved_recipe(tmp_path: Path, *, cli: str | None = None) -> TestRecipe:
    api = tmp_path / "services" / "api"
    api.mkdir(parents=True, exist_ok=True)
    (api / "pyproject.toml").write_text("[project]\nname = 'api'\n")
    (api / "uv.lock").write_text("version = 1\n")
    return resolve_test_recipe(
        SimpleNamespace(test_command=None), SimpleNamespace(test_command=cli), repo_root=tmp_path, cwd=api,
    )

def test_every_review_and_fix_prompt_states_the_resolved_test_recipe(tmp_path: Path) -> None:
    recipe = _resolved_recipe(tmp_path, cli="uv run pytest")
    block = render_test_recipe_block(recipe)
    assert "uv run pytest" in block and "services/api" in block
    prompt = build_per_stack_prompt(strategy="s", stack_name="python", files=["a.py"], diff_path=tmp_path / "d.patch",
        intent_path=tmp_path / "intent.md", alternatives_path=tmp_path / "alts.json",
        output_path=tmp_path / "out.json", cwd=tmp_path,
    )
    assert block in append_extended_facts(prompt, recipe)
    assert append_extended_facts(prompt, None) == prompt
