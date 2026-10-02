"""Grounded diagram authoring and repair prompts.

Clone backends receive bounded inline diff and exploration inputs; worktree
backends may read the full artifact paths. Projection bounds affect the prompt,
while grounding always checks the complete host-side eligibility artifact.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from daydream.deep.diff import _full_diff_pointer, _hunk_index_authority
from daydream.phases.review_prompts import _exploration_pointer
from daydream.prompt_budget import (
    INLINE_DIFF_BUDGET_BYTES,
    fits_inline_diff_budget,
    truncate_utf8_to_budget,
)
from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION, UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY
from daydream.prompts.schema_block import schema_block

# Role sentences the diagram builders open with. They are the phase's stable
# discriminator: the test stub backend dispatches on them, so a reword here is
# a wire change, not a copy edit.
SEQUENCE_DIAGRAM_ROLE = "You are the sequence-diagram author for this pull request."
FLOWCHART_ROLE = "You are the flowchart author for this pull request."

# Keep the source-validation contract inline: diagram agents run in the reviewed
# repository, where a skill-file pointer would refer to the target's files.
DIAGRAM_GROUNDING_INSTRUCTION = (
    "Grounding contract for this diagram (stated inline here — no skill file "
    "read is required):\n"
    "  Inspect the source for every element and cite its exact file:line and "
    "code. The source is the only truth; never infer an interaction, a branch, "
    "a call, or a component from the branch name, cwd, or memory.\n"
    "  The host verifies every file:line you emit deterministically, with no "
    "second model in the loop: the path must exist at HEAD and resolve inside "
    "this repository, the line must be within the file, the cited `symbol` must "
    "appear on that line (a ±3-line snap is attempted first), and a line cited "
    "as a branch, a terminal statement, or a call site must really be one for "
    "that file's language. Anything unverifiable is dropped from the rendered "
    "diagram, and a diagram left with too little to say is omitted entirely.\n"
    "  You never write mermaid. Return ONLY the JSON spec; a deterministic "
    "renderer draws the diagram from the elements that survive verification, so "
    "no drawing syntax, HTML, or markdown belongs in any label.\n"
    "  Prefer fewer, fully grounded elements over a complete-looking diagram "
    "resting on invented evidence: a small verified diagram ships, a large "
    "unverifiable one does not."
)


def _diagram_diff_block(diff_path: Path, inline_diff: str | None, *, clone_mode: bool = False) -> str:
    """Inline a diff that fits; otherwise use its artifact pointer.

    Clones cannot read host paths: missing input is omitted and oversized input
    is truncated with a marker. Banner, content, and marker share the byte budget.
    """
    if inline_diff and fits_inline_diff_budget(inline_diff):
        head = (
            "The PR diff (base..HEAD) is inlined below; do NOT re-Read "
            "diff.patch for it:\n\n"
            if not clone_mode
            else "The PR diff (base..HEAD) is inlined below:\n\n"
        )
        if clone_mode:
            return f"{head}{inline_diff.rstrip()}\n\n"
        return f"{head}{inline_diff.rstrip()}\n\n{_hunk_index_authority(diff_path)}"
    if clone_mode:
        if not inline_diff:
            return ""
        head = "The PR diff (base..HEAD) is inlined below:\n\n"
        marker = "\n[diff truncated to fit the prompt budget]\n\n"
        truncated = truncate_utf8_to_budget(
            inline_diff.rstrip(), INLINE_DIFF_BUDGET_BYTES - len(head.encode("utf-8")), marker
        )
        return f"{head}{truncated}"
    return f"{_full_diff_pointer(diff_path)}\n{_hunk_index_authority(diff_path)}"


def _diagram_exploration_block(
    exploration_dir: Path | None,
    inline_exploration: str | None = None,
    inline_dependencies: str | None = None,
    *,
    clone_mode: bool = False,
) -> str:
    """Fence repository content, using inline context for disposable clones.

    Worktree backends receive exploration pointers. Missing exploration still
    emits the untrusted-content boundary; missing inline content is omitted.
    """
    if clone_mode:
        blocks = [UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY]
        if inline_exploration:
            blocks.append(inline_exploration.rstrip())
        if inline_dependencies:
            blocks.append(
                "Deterministic import edges between changed files (use to place component "
                "and module boundaries, never as a substitute for reading the source):\n"
                f"{inline_dependencies.rstrip()}"
            )
        return "\n\n".join(blocks)
    pointer = _exploration_pointer(exploration_dir)
    if not pointer:
        return UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY
    return (
        f"{pointer}\n"
        f"{exploration_dir}/dependencies.md lists the deterministic import edges between changed files — "
        "use it to place component and module boundaries, never as a substitute for reading the source."
    )


def _bounded_projection(
    full_text: str,
    lines: list[str],
    render: Callable[[list[str], str], str],
    count_entry: Callable[[str], bool],
    noun: str,
    budget_bytes: int,
) -> str:
    """Retain an ordered prefix within the byte budget and count omitted entries.

    Return fitting input unchanged. Stop at the first overflowing line; count_entry
    excludes structural headers from the omission count. If even the empty body
    and notice exceed the budget, return a truncated notice.
    """
    if len(full_text.encode("utf-8")) <= budget_bytes:
        return full_text
    total = sum(1 for line in lines if count_entry(line))
    kept_entries = 0
    best_text: str | None = None
    for keep in range(len(lines) + 1):
        if keep > 0 and count_entry(lines[keep - 1]):
            kept_entries += 1
        dropped = total - kept_entries
        notice = f"- ({dropped} more {noun} omitted to fit the prompt budget)" if dropped else ""
        text = render(lines[:keep], notice)
        if len(text.encode("utf-8")) <= budget_bytes:
            best_text = text
        else:
            break
    if best_text is not None:
        return best_text
    # Even an empty body cannot carry the header plus the notice: keep the
    # notice truncated so the caller still learns why the projection is empty.
    return truncate_utf8_to_budget(
        f"- ({total} more {noun} omitted to fit the prompt budget)", budget_bytes
    )


def _files_by_module_block(
    files_by_module: dict[str, list[str]],
    *,
    budget_bytes: int = INLINE_DIFF_BUDGET_BYTES,
) -> str:
    """Render bounded changed-file groups with an exact omitted-file count."""
    lines: list[str] = []
    for module in sorted(files_by_module):
        lines.append(f"- {module}")
        for path in files_by_module[module]:
            lines.append(f"    - {path}")
    body = "\n".join(lines) or "- (no changed code files)"
    header = (
        "Changed code files grouped by module/service. Participants must align "
        "with these real boundaries: every `internal` participant owns at least "
        "one of these paths, and no participant may be invented for a component "
        "that owns none of them.\n"
    )
    return _bounded_projection(
        f"{header}{body}",
        lines,
        lambda kept, notice: header + "\n".join(kept) + (f"\n{notice}" if notice else ""),
        lambda line: line.startswith("    - "),
        "changed files",
        budget_bytes,
    )


def _candidate_roots_block(
    candidate_roots: list[dict[str, Any]],
    *,
    forced: bool,
    budget_bytes: int = INLINE_DIFF_BUDGET_BYTES,
) -> str:
    """Render the bounded root list with ranges and branch counts.

    The model may choose only a displayed root; grounding checks the full host list.
    """
    lines = [
        f"- `{root.get('name')}` in {root.get('file')}, lines {root.get('line')}-{root.get('end_line')}, "
        f"{root.get('branch_points')} changed branch point(s)"
        for root in candidate_roots
    ]
    body = "\n".join(lines) or "- (none)"
    header = (
        "Candidate root functions. The `root` you return MUST be one of these, "
        "with `file`, `name` and `line` copied verbatim from the entry you pick "
        "— a root outside this list is rejected outright. Every node's evidence "
        "line must fall inside the chosen root's line range shown here."
    )
    if forced:
        header += (
            " This flowchart was explicitly requested, so the list is every "
            "changed function rather than only those meeting the branch-point "
            "threshold; pick the one whose control flow is most worth reading, "
            "and if none of them has a real decision point, return the nodes you "
            "can actually ground rather than inventing branches to fill the shape."
        )
    return _bounded_projection(
        f"{header}\n{body}",
        lines,
        lambda kept, notice: f"{header}\n" + "\n".join(kept) + (f"\n{notice}" if notice else ""),
        lambda line: True,
        "candidate roots",
        budget_bytes,
    )


_SEQUENCE_SPEC_RULES = (
    "Sequence spec rules:\n"
    "  - `participants`: 3 to 10 entries. Each has `name` (the component as a "
    "human reader would name it), `kind` (`internal` | `external`), `files` "
    "(repo-relative paths that exist at HEAD — at least one for `internal`, "
    "empty for `external`), and `service` (the owning service/app name, or null "
    "when the repo has no service boundaries).\n"
    "  - `messages`: the interaction in order. Each has `from` and `to` "
    "(participant names, exactly as spelled in `participants`), `label` (≤ 80 "
    "characters, what actually happens), `kind` (`call` | `reply` | `self`), "
    "`changed` (true when this diff adds or modifies the interaction), and "
    "`evidence` = {`file`, `line`, `symbol`}.\n"
    "  - Evidence per message kind: for `call` and `self`, cite the call-site "
    "line in one of the `from` participant's files and set `symbol` to the "
    "callee name on that line. For `reply`, cite a `return` line in one of the "
    "`from` participant's files, set `symbol` to the enclosing function name, and "
    "place it immediately after the reversed `call`. For a call to an `external` "
    "participant, cite the in-repo line "
    "that makes the outbound call and set `symbol` to the client method token on "
    "that line. For an internal target, `symbol` must be defined in one of the "
    "`to` participant's files.\n"
    "  - An `external` participant may only be the source of the FIRST message "
    "(the entrypoint) or the target of a reply. The entrypoint message's "
    "evidence is the handler definition line in a `to` participant file.\n"
    "  - `blocks` (may be `[]`): `alt` (2 or more branches), `opt` (exactly 1), "
    "`loop` (exactly 1). Each branch has `condition` text, `evidence` = "
    "{`file`, `line`} pointing at the branch or loop statement itself, and "
    "`messages`, the 0-based indices into `messages` that the branch contains. "
    "An index must appear in at most one branch.\n"
    "  - Floor: the diagram renders only with at least 3 grounded messages, at "
    "least 2 participants, and at least 1 message whose evidence line falls "
    "inside a changed hunk. Anchor the interaction on the diff, not on "
    "untouched surrounding plumbing."
)

_FLOWCHART_SPEC_RULES = (
    "Flowchart spec rules:\n"
    "  - `root` = {`file`, `name`, `line`}, copied verbatim from one candidate "
    "root entry.\n"
    "  - `nodes`: 4 to 25 entries. Each has `id` (unique within the spec), "
    "`kind` (`start` | `end` | `process` | `decision` | `subroutine` | `io`), "
    "`label` (≤ 60 characters) and `evidence` = {`file`, `line`, `symbol`} "
    "(`symbol` may be null except on a `subroutine`). Every evidence line must "
    "be inside the root's line range.\n"
    "  - Evidence per node kind: `start` cites the root's definition line. `end` "
    "cites a `return`/`raise`/`throw`/`panic`/exit statement inside the root "
    "range. `process` and `io` cite a statement inside the root range. "
    "`decision` cites an actual branch or loop statement (`if`/`elif`/`else`/"
    "`match`/`case`/`switch`/`for`/`while`/`try`/`except`/`catch`) inside the "
    "root range. `subroutine` cites the CALL SITE inside the root range and sets "
    "`symbol` to the called function, which must be defined somewhere in this "
    "repository and must appear on the cited line.\n"
    "  - `edges`: each has `from` and `to` (node ids) and `label` (null when "
    "unlabeled). Every edge leaving a `decision` node must carry a label, and a "
    "`decision` must have at least 2 outgoing edges with distinct labels — "
    "otherwise it is not a decision and the host demotes it to a plain step.\n"
    "  - Exactly one `start` node; at least one `end` node. Nodes unreachable "
    "from `start` are dropped.\n"
    "  - Floor: the diagram renders only with at least 4 grounded nodes "
    "including the `start`, at least 1 `end`, and at least 1 grounded "
    "`decision`. Show the control flow the diff actually changed, not the "
    "function's every statement."
)


def build_sequence_diagram_prompt(
    *,
    diff_path: Path,
    inline_diff: str | None,
    inline_exploration: str | None = None,
    inline_dependencies: str | None = None,
    clone_mode: bool = False,
    files_by_module: dict[str, list[str]],
    cwd: Path,
    exploration_dir: Path | None,
    schema: dict[str, Any],
) -> str:
    """Describe a grounded interaction using participants from real module boundaries."""
    parts: list[str] = [
        f"{SEQUENCE_DIAGRAM_ROLE} Propose a sequence diagram of the interaction "
        "this change is about, as a structured JSON spec in which every element "
        "carries file:line evidence.",
        _diagram_exploration_block(
            exploration_dir, inline_exploration, inline_dependencies, clone_mode=clone_mode
        ),
        CWD_GROUNDING_INSTRUCTION.format(cwd=cwd),
        _diagram_diff_block(diff_path, inline_diff, clone_mode=clone_mode),
        _files_by_module_block(files_by_module),
        _SEQUENCE_SPEC_RULES,
        DIAGRAM_GROUNDING_INSTRUCTION,
        schema_block(schema),
    ]
    return "\n\n".join(parts)


def build_flowchart_prompt(
    *,
    diff_path: Path,
    inline_diff: str | None,
    inline_exploration: str | None = None,
    inline_dependencies: str | None = None,
    clone_mode: bool = False,
    candidate_roots: list[dict[str, Any]],
    forced: bool,
    cwd: Path,
    exploration_dir: Path | None,
    schema: dict[str, Any],
) -> str:
    """Describe control flow inside one eligible root function.

    forced means the user requested this kind, so candidate roots include changed
    functions below the ordinary branch-point threshold.
    """
    parts: list[str] = [
        f"{FLOWCHART_ROLE} Propose a flowchart of the control flow inside ONE "
        "changed function, as a structured JSON spec in which every element "
        "carries file:line evidence.",
        _diagram_exploration_block(
            exploration_dir, inline_exploration, inline_dependencies, clone_mode=clone_mode
        ),
        CWD_GROUNDING_INSTRUCTION.format(cwd=cwd),
        _diagram_diff_block(diff_path, inline_diff, clone_mode=clone_mode),
        _candidate_roots_block(candidate_roots, forced=forced),
        _FLOWCHART_SPEC_RULES,
        DIAGRAM_GROUNDING_INSTRUCTION,
        schema_block(schema),
    ]
    return "\n\n".join(parts)


def _diagram_failure_lines(failures: list[dict[str, Any]]) -> str:
    """Render one line per rejected element: element, ref, and reason code."""
    lines = [
        f"- {failure.get('element')} `{failure.get('ref')}`: {failure.get('reason')}"
        for failure in failures
    ]
    return "\n".join(lines) or "- (none)"


def build_diagram_repair_prompt(
    *,
    kind: str,
    failures: list[dict[str, Any]],
    candidate_roots: list[dict[str, Any]] | None,
    schema: dict[str, Any],
) -> str:
    """Continue the kind's existing session with rejected elements and repair rules.

    Prior diff and exploration context remain available. Repeat candidate roots
    for root-reselection failures; require a complete replacement spec.
    """
    parts: list[str] = [
        f"Diagram repair turn ({kind}): the deterministic grounding pass "
        "rejected the element(s) below. Nothing about the repository changed "
        "between the two turns, so re-sending the same evidence cannot pass.",
        "Rejected elements (element, reference, reason code):\n" + _diagram_failure_lines(failures),
        "For EACH rejected element do exactly one of two things:\n"
        "  1. Correct its evidence — read the file in THIS turn, then cite a "
        "real file:line that satisfies the reason code (right file, right line, "
        "the cited symbol actually on that line, the right kind of statement).\n"
        "  2. Remove the element from the spec entirely, along with anything "
        "that only existed to support it (a participant left with no messages, "
        "an edge whose node is gone, a branch left with no messages).\n"
        "Removing an element you cannot ground is the correct answer, not a "
        "failure. Do not substitute a different invented element for it.",
    ]
    if candidate_roots is not None:
        parts.append(
            "A `ROOT_NOT_CANDIDATE` verdict means the root you chose is not in "
            "the candidate list. Re-pick a root from the list below, copying "
            "`file`, `name` and `line` verbatim, and re-anchor every node inside "
            "the new root's line range."
        )
        parts.append(_candidate_roots_block(candidate_roots, forced=False))
    parts.append(
        "Return the FULL corrected spec in the same JSON shape — not a patch, "
        "not only the elements you changed. Elements you are not repairing must "
        "be repeated verbatim, with their indices/ids kept consistent. This is "
        "the ONLY repair turn: after it, every still-ungrounded element is "
        "dropped, and the diagram is omitted if too little survives."
    )
    parts.append(DIAGRAM_GROUNDING_INSTRUCTION)
    parts.append(schema_block(schema))
    return "\n\n".join(parts)
