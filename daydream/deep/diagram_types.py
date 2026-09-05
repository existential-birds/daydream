"""Shared value types for the grounded-diagram pipeline (issue #1113).

The diagram pipeline is four modules deep — eligibility (``diagram_trigger``),
grounding (``diagram_grounding``), rendering (``diagram_render``) and the
orchestrator step — and three of them need the same two dataclasses and the
same result alias. They live here rather than in any one of those modules so
none of them has to import another: ``diagram_trigger`` produces
``CandidateRoot`` values that ``diagram_grounding`` consumes, and a direct
import between the two would make the grounding pass depend on the eligibility
pass it is meant to be independent of. ``diagram_trigger`` re-exports both
dataclasses so ``from daydream.deep.diagram_trigger import CandidateRoot``
reads naturally at the eligibility call sites.

Pure data: no I/O, no imports outside the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

# One kind's diagram outcome, as written to ``diagram.json``,
# ``ctx.data["diagrams"]`` and the Phase A findings artifact::
#
#     {"status": "rendered" | "omitted" | "skipped" | "failed",
#      "reason": str | None,          # why it was skipped or failed
#      "spec_proposed": dict | None,  # the model's last proposed spec
#      "spec_final": dict | None,     # pruned + capped; schema-valid
#      "grounding": {"elements": [...], "summary": {...},
#                    "capped": {...}, "root_range": [int, int] | None} | None,
#      "omit_reasons": list[str],
#      "mermaid": str | None}         # dropped in the findings artifact
#
# Deliberately a plain ``dict[str, Any]`` rather than a ``TypedDict``: four
# modules produce and consume it, several keys are meaningful only for some
# statuses, and ``total=False`` would erase exactly the strictness a TypedDict
# is for while adding friction at every construction site. The renderer
# documents the precise subset of keys it may read.
DiagramResult: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class DiagramThresholds:
    """Deterministic eligibility thresholds for one run.

    Resolved once from ``[tool.daydream.diagram]`` over the ``config.py``
    defaults and then passed by value, so the eligibility decision is a pure
    function of its arguments and is reproducible from ``diagram.json``.

    Attributes:
        min_code_files: Minimum changed non-test code files for the sequence
            diagram's cross-module rule.
        min_modules: Minimum distinct modules those files must span for the
            cross-module rule.
        min_branch_points: Minimum changed branch points inside a single
            function for the flowchart rule.
    """

    min_code_files: int = 3
    min_modules: int = 2
    min_branch_points: int = 3


@dataclass(frozen=True)
class CandidateRoot:
    """One changed function the flowchart may be rooted at.

    The model picks its flowchart root only from the run's candidate list, so a
    root always has a tree-sitter-derived range the grounding pass can check
    every node against. ``end_line`` comes from the definition node, which
    spans the function body — that range is what makes ``NODE_OUTSIDE_ROOT``
    decidable.

    Attributes:
        file: Repository-relative POSIX path of the defining file.
        name: Function/method name as tree-sitter reported it.
        line: 1-based first line of the definition.
        end_line: 1-based last line of the definition (inclusive).
        branch_points: Count of branch statements inside the range that also
            fall inside a head-side changed hunk. This is the number the
            flowchart threshold is compared against, and the sort key that
            orders candidates.
    """

    file: str
    name: str
    line: int
    end_line: int
    branch_points: int


__all__ = ["CandidateRoot", "DiagramResult", "DiagramThresholds"]
