"""Deep-mode prompt builder tests (D-09, D-10, D-19, D-20)."""
from pathlib import Path

from daydream.deep.diagram_prompts import (
    build_flowchart_prompt,
    build_sequence_diagram_prompt,
)
from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES
from daydream.prompts.grounding import (
    UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY,
)

# Issue #172 — Fix B: read-once inline diff hunks in per-stack / generic prompts

# Issue #644 — bound the deep-flow diff at gather time (whole-block retention)

# Issue #221 — cwd grounding injected into every deep prompt builder

# Issue #279 — Authoritative-intent rule gate in the deep prompt builders


# --- Task 12a: the per-stack prompt path can omit the alternatives pointer ----


# Issue #308 — test-quality rubric in the per-stack review prompt

# Issue #314 — anti-slop review rubric (structural erosion + verbosity patterns)

# =============================================================================
# Issue #310 — cross-file verification instruction (symbol existence,
# config-flow traces, trust-model checks)
# =============================================================================

# --- Issue #731: coverage-evidence grounding + frontier-read instruction ---


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
