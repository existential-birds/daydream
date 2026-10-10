"""Deep-mode prompt builder tests (D-09, D-10, D-19, D-20)."""
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.deep.diagram_prompts import (
    build_flowchart_prompt,
    build_sequence_diagram_prompt,
)
from daydream.deep.diff import _diff_blocks_for_files
from daydream.deep.prompts import (
    build_arbiter_prompt,
    build_generic_fallback_prompt,
    build_merge_prompt,
    build_per_stack_prompt,
    build_structural_prompt,
)
from daydream.phases import append_extended_facts
from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES
from daydream.prompts.authorial_intent import AUTHORITATIVE_INTENT_RULE, PR_DESCRIPTION_UNTRUSTED_FRAMING
from daydream.prompts.grounding import (
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

def test_diff_blocks_for_files_returns_none_when_no_blocks_match() -> None:
    out = _diff_blocks_for_files(_DIFF_TWO_FILES, ["nonexistent.py"])
    assert out is None

# Issue #221 — cwd grounding injected into every deep prompt builder

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

# --- Task 12a: the per-stack prompt path can omit the alternatives pointer ----

def test_omitting_alternatives_keeps_authoritative_intent_rule(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    without = build_per_stack_prompt(strategy=_default_strategy("discovery.per_stack"), stack_name="python",
        files=["api.py"], **p, intent_authoritative=True, include_alternatives=False,
    )
    assert "alternatives.json" not in without
    assert "author's stated intent from the pull-request description" in without
    assert AUTHORITATIVE_INTENT_RULE in without
    assert PR_DESCRIPTION_UNTRUSTED_FRAMING in without  # NEW #579

# Issue #308 — test-quality rubric in the per-stack review prompt

# Issue #314 — anti-slop review rubric (structural erosion + verbosity patterns)

# =============================================================================
# Issue #310 — cross-file verification instruction (symbol existence,
# config-flow traces, trust-model checks)
# =============================================================================

# --- Issue #731: coverage-evidence grounding + frontier-read instruction ---


@pytest.mark.parametrize('name', ['per_stack', 'generic_fallback'])
def test_diff_instruction_allows_useful_source_investigation(tmp_path: Path, name: str) -> None:
    """Supplied hunks come first and reviewers can inspect more context when useful."""
    prompt = _review_prompt(name, tmp_path, inline_diff="@@ -1 +1 @@\n-'x'\n+'y'\n")
    assert "supplied hunks and context" in prompt or "supplied changed hunks" in prompt
    assert "when more context" in prompt or "when useful" in prompt

def test_exploration_pointer_keeps_artifacts_bounded_and_source_work_optional(tmp_path: Path) -> None:
    """Exploration pointers stay bounded while useful source inspection remains optional."""
    out = _review_prompt("per_stack", tmp_path, exploration_dir=tmp_path / ".daydream" / "exploration")
    # Exploration artifacts are pointed at as bounded context only: two named
    # files, sibling artifacts explicitly out of scope.
    assert "Read the pre-scan summary at" in out
    assert str(tmp_path / ".daydream" / "exploration" / "summary.md") in out
    assert "Do not infer or enumerate sibling artifact files" in out
    assert "Inspect enclosing symbols or other sections when useful" in out
    assert "MUST read in full all assigned source files" not in out


# Issue #972 R1 — host-owned severity rubric reaches every assigning prompt

# Issue #972 R1.3 — adjudication restatements cite the severity rubric

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
