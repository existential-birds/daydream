"""Deterministic grounded-diagram eligibility (issue #1113).

Which diagram kinds a run may attempt is decided here, before any model is
asked anything: a sequence diagram when the change crosses a service or module
boundary, a flowchart when a single changed function gained or reworked enough
branch points to be worth drawing. The decision is a pure function of its
arguments -- it reads no ``FlowContext``, no config file and no environment,
and touches the filesystem only to parse the changed files it was handed -- so
the whole of it round-trips into ``diagram.json`` via
:meth:`Eligibility.to_dict` and a reviewer can re-derive it later.

Two properties are worth stating because they are easy to break:

* **Stacks must come from a fresh** ``detect_stacks(sorted(changed_files))``.
  The orchestrator's published ``ctx.data["stacks"]`` is post-collapse and
  post-shard: on a diff of two files in two languages the tiny-diff collapse
  folds everything into one ``generic`` assignment, which would classify an
  entirely real code change as "no code files" and make every diagram
  permanently ineligible. ``decide_eligibility`` therefore takes ``stacks`` by
  argument and the caller owns building them un-collapsed.
* **Everything about tree-sitter here is fail-open.** A missing grammar, an
  unparseable file or a known-bad ``tree-sitter`` install yields *no* branch
  points, never an exception. Failing open can only ever suppress a diagram,
  and a review must not die over a picture.

:class:`CandidateRoot` and :class:`DiagramThresholds` are defined in
``daydream.deep.diagram_types`` (so the grounding pass can consume them
without importing this module) and re-exported here, where the eligibility
call sites read them most naturally.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from daydream._tree_sitter_safety import TreeSitterBadVersionError
from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.detection import GENERIC_STACK
from daydream.deep.diagram_types import CandidateRoot, DiagramThresholds
from daydream.repository_paths import is_test_path
from daydream.tree_sitter_index import (
    branch_statement_lines,
    definitions_in_file,
    language_for_path,
)

if TYPE_CHECKING:
    from pathlib import Path

    from daydream.deep.detection import StackAssignment
    from daydream.services import Service

# Wall budget for the whole branch-point sweep, mirroring
# ``dependency._GRAPH_BUILD_WALL_BUDGET_S``: a pathological diff (hundreds of
# changed files, each re-parsed) must not stall the review. On expiry the
# partial counts built so far are used, which is the fail-open direction.
_BRANCH_COUNT_WALL_BUDGET_S = 5.0

# ``force`` values that name specific kinds. A kind not named by a forcing mode
# is skipped, not auto-evaluated: ``--diagram sequence`` means "the sequence
# diagram", which is why ``both`` exists as its own mode (spec section 7).
_FORCED_KINDS: dict[str, tuple[str, ...]] = {
    "sequence": ("sequence",),
    "flowchart": ("flowchart",),
    "both": ("sequence", "flowchart"),
}

_OFF_REASON = "Diagram mode is off."


@dataclass(frozen=True)
class KindDecision:
    """Whether one diagram kind may be attempted, and why.

    Attributes:
        eligible: Whether the kind may be attempted this run.
        rule: The rule that decided it -- ``"cross-service"``,
            ``"cross-module"``, ``"branch-points"``, ``"forced"``, or ``None``
            when the kind is not eligible.
        reason: A human sentence, always set, for ``diagram.json`` and the
            omission notice. Set for the eligible case too, so the artifact
            records why a diagram was drawn and not only why it was not.
    """

    eligible: bool
    rule: str | None
    reason: str


@dataclass
class Eligibility:
    """Every signal behind one run's diagram decision.

    Attributes:
        code_files: Sorted changed non-test files assigned to a real language
            stack (the ``generic`` and ``structure`` buckets are not code).
        modules: ``{code file: module}``, where a module is the owning service
            root when one owns the file and its top-level directory otherwise
            (``"."`` for a repository-root file).
        services: ``{code file: service name}`` for the code files an
            enumerated service owns. Files outside every service root are
            absent rather than mapped to a placeholder.
        cross_module_edges: Count of directed import edges between two code
            files in different modules.
        function_branch_counts: Every changed function with at least one
            changed branch point, in candidate order.
        candidate_roots: The roots the flowchart may be rooted at -- the
            functions meeting ``thresholds.min_branch_points``, or every
            changed function when the flowchart is forced and none meet it.
            Sorted by branch points descending, then file, then line.
        sequence: The sequence diagram's decision.
        flowchart: The flowchart's decision.
        thresholds: The thresholds the decision was taken against.
        force: The resolved diagram mode this decision was taken under.
    """

    code_files: list[str]
    modules: dict[str, str]
    services: dict[str, str]
    cross_module_edges: int
    function_branch_counts: list[CandidateRoot]
    candidate_roots: list[CandidateRoot]
    sequence: KindDecision
    flowchart: KindDecision
    thresholds: DiagramThresholds
    force: str

    def eligible_kinds(self) -> list[str]:
        """Return the eligible kinds in render order (sequence first)."""
        kinds: list[str] = []
        if self.sequence.eligible:
            kinds.append("sequence")
        if self.flowchart.eligible:
            kinds.append("flowchart")
        return kinds

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form written to ``diagram.json``."""
        return {
            "code_files": list(self.code_files),
            "modules": dict(self.modules),
            "services": dict(self.services),
            "cross_module_edges": self.cross_module_edges,
            "function_branch_counts": [asdict(root) for root in self.function_branch_counts],
            "candidate_roots": [asdict(root) for root in self.candidate_roots],
            "sequence": asdict(self.sequence),
            "flowchart": asdict(self.flowchart),
            "thresholds": asdict(self.thresholds),
            "force": self.force,
        }


def _code_files(stacks: list[StackAssignment], changed_files: list[str]) -> list[str]:
    """Return the sorted changed non-test files that belong to a language stack.

    A file is code when ``detect_stacks`` routed it to a real language stack:
    the ``generic`` bucket holds docs and config (``.md`` is pinned there) and
    the ``structure`` meta-stack holds a copy of *every* changed file, so both
    are skipped. Shard suffixes (``python#2``) are stripped so a sharded
    assignment list classifies identically to an unsharded one.
    """
    changed = set(changed_files)
    selected: set[str] = set()
    for assignment in stacks:
        base_stack = assignment.stack_name.split("#", 1)[0]
        if base_stack in (GENERIC_STACK, STRUCTURE_STACK_NAME):
            continue
        for path in assignment.files:
            if path in changed and not is_test_path(path):
                selected.add(path)
    return sorted(selected)


def _owning_service(path: str, services: list[Service]) -> Service | None:
    """Return the deepest enumerated service whose root contains ``path``."""
    best: Service | None = None
    best_depth = -1
    for service in services:
        root = service.root.as_posix()
        if root in ("", "."):
            continue
        if path != root and not path.startswith(f"{root}/"):
            continue
        depth = len(PurePosixPath(root).parts)
        if depth > best_depth:
            best = service
            best_depth = depth
    return best


def _module_of(path: str, services: list[Service]) -> str:
    """Return ``path``'s module: its service root, else its top-level directory."""
    service = _owning_service(path, services)
    if service is not None:
        return service.root.as_posix()
    parts = PurePosixPath(path).parts
    return parts[0] if len(parts) > 1 else "."


def _count_cross_module_edges(
    import_graph: dict[str, set[str]], modules: dict[str, str]
) -> int:
    """Count directed import edges whose endpoints are code files in different modules.

    Directed, so a mutual import between two modules counts twice -- the number
    is a strength signal for the cross-module rule (which needs >= 1), never an
    undirected edge count.
    """
    total = 0
    for source, targets in import_graph.items():
        source_module = modules.get(source)
        if source_module is None:
            continue
        for target in targets:
            target_module = modules.get(target)
            if target_module is not None and target_module != source_module:
                total += 1
    return total


def count_function_branch_points(
    repo_root: Path, file: str, ranges: list[tuple[int, int]]
) -> list[CandidateRoot]:
    """Return every changed function in ``file`` with its changed branch-point count.

    "Changed function" means a tree-sitter function definition whose
    ``line..end_line`` body range overlaps at least one of ``ranges`` (the
    file's head-side changed hunks). A branch point counts for a function when
    its line is inside both that function's range and a changed hunk, and it is
    attributed to the *innermost* enclosing definition, so a branch inside a
    nested closure belongs to the closure and not to its host.

    Functions with zero changed branch points are still returned -- they are
    the candidate pool a forced flowchart falls back to. Callers filter by
    ``branch_points``.

    Returns ``[]`` for a language with no grammar, an unreadable file, or any
    tree-sitter failure (fail-open: no branch points can only suppress a
    diagram, never fabricate one).
    """
    if not ranges:
        return []
    language_id = language_for_path(file)
    if language_id is None:
        return []
    try:
        source = (repo_root / file).read_bytes()
    except OSError:
        return []
    try:
        definitions = definitions_in_file(repo_root, file)
        branch_lines = branch_statement_lines(language_id, source)
    except TreeSitterBadVersionError:
        # A known-bad tree-sitter install must not fail a review over a
        # picture; ``get_parser`` raises this before any native parsing.
        return []
    except Exception:
        # Both callees document their own fail-open behavior; this is the
        # belt-and-braces clause for anything they let through.
        return []

    functions: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for record in definitions:
        name = record.get("name")
        line = record.get("line")
        end_line = record.get("end_line")
        if record.get("kind") != "function":
            continue
        if not isinstance(name, str) or not isinstance(line, int) or not isinstance(end_line, int):
            continue
        key = (name, line, end_line)
        if key in seen:
            continue
        seen.add(key)
        if any(line <= end and start <= end_line for start, end in ranges):
            functions.append(key)

    counts: dict[tuple[str, int, int], int] = {key: 0 for key in functions}
    for branch_line in branch_lines:
        if not any(start <= branch_line <= end for start, end in ranges):
            continue
        owner = _innermost_owner(branch_line, functions)
        if owner is not None:
            counts[owner] += 1

    roots: list[CandidateRoot] = []
    for name, line, end_line in sorted(functions, key=lambda item: (item[1], item[2], item[0])):
        roots.append(
            CandidateRoot(
                file=file,
                name=name,
                line=line,
                end_line=end_line,
                branch_points=counts[(name, line, end_line)],
            )
        )
    return roots


def _innermost_owner(
    line: int, functions: list[tuple[str, int, int]]
) -> tuple[str, int, int] | None:
    """Return the narrowest function range containing ``line``, or None."""
    best: tuple[str, int, int] | None = None
    for candidate in functions:
        _, start, end = candidate
        if not start <= line <= end:
            continue
        if best is None or (end - start) < (best[2] - best[1]):
            best = candidate
    return best


def _candidate_order(roots: list[CandidateRoot]) -> list[CandidateRoot]:
    """Sort candidate roots by branch points descending, then file, then line."""
    return sorted(roots, key=lambda root: (-root.branch_points, root.file, root.line))


def _changed_functions(
    repo_root: Path, code_files: list[str], hunk_ranges: dict[str, list[tuple[int, int]]]
) -> list[CandidateRoot]:
    """Return every changed function across ``code_files``, in candidate order."""
    deadline = time.monotonic() + _BRANCH_COUNT_WALL_BUDGET_S
    found: list[CandidateRoot] = []
    for file in code_files:
        ranges = hunk_ranges.get(file) or []
        if not ranges:
            continue
        if time.monotonic() > deadline:
            break
        found.extend(count_function_branch_points(repo_root, file, ranges))
    return _candidate_order(found)


def _decide_sequence(
    *,
    code_files: list[str],
    modules: dict[str, str],
    services: dict[str, str],
    cross_module_edges: int,
    thresholds: DiagramThresholds,
) -> KindDecision:
    """Apply the cross-service and cross-module rules to the sequence diagram."""
    service_names = sorted(set(services.values()))
    module_names = sorted(set(modules.values()))
    if len(service_names) >= 2:
        return KindDecision(
            eligible=True,
            rule="cross-service",
            reason=(
                f"{len(code_files)} changed code file(s) span {len(service_names)} services "
                f"({', '.join(service_names)}), so the change crosses a service boundary."
            ),
        )
    if (
        len(code_files) >= thresholds.min_code_files
        and len(module_names) >= thresholds.min_modules
        and cross_module_edges >= 1
    ):
        return KindDecision(
            eligible=True,
            rule="cross-module",
            reason=(
                f"{len(code_files)} changed code file(s) span {len(module_names)} modules "
                f"({', '.join(module_names)}) with {cross_module_edges} cross-module import edge(s)."
            ),
        )
    return KindDecision(
        eligible=False,
        rule=None,
        reason=(
            f"Needs 2 services, or >= {thresholds.min_code_files} code files across "
            f">= {thresholds.min_modules} modules with >= 1 cross-module import edge; saw "
            f"{len(code_files)} code file(s), {len(service_names)} service(s), "
            f"{len(module_names)} module(s), {cross_module_edges} cross-module edge(s)."
        ),
    )


def _decide_flowchart(
    *, qualifying: list[CandidateRoot], changed_functions: list[CandidateRoot], minimum: int
) -> KindDecision:
    """Apply the branch-point rule to the flowchart."""
    if qualifying:
        top = qualifying[0]
        return KindDecision(
            eligible=True,
            rule="branch-points",
            reason=(
                f"{len(qualifying)} changed function(s) have >= {minimum} changed branch points "
                f"(most: {top.branch_points} in {top.name} at {top.file}:{top.line})."
            ),
        )
    best = max((root.branch_points for root in changed_functions), default=0)
    return KindDecision(
        eligible=False,
        rule=None,
        reason=(
            f"No changed function has >= {minimum} changed branch points across "
            f"{len(changed_functions)} changed function(s) (most: {best})."
        ),
    )


def _forced_decision(kind: str, force: str, *, note: str = "") -> KindDecision:
    """Return the decision for ``kind`` under a kind-naming mode (never ``off``)."""
    if kind in _FORCED_KINDS[force]:
        reason = f"Forced eligible: diagram mode {force!r} names this kind."
        return KindDecision(eligible=True, rule="forced", reason=reason + note)
    return KindDecision(
        eligible=False,
        rule=None,
        reason=f"Not requested: diagram mode {force!r} does not name this kind.",
    )


def decide_eligibility(
    *,
    repo_root: Path,
    changed_files: list[str],
    hunk_ranges: dict[str, list[tuple[int, int]]],
    stacks: list[StackAssignment],
    services: list[Service],
    import_graph: dict[str, set[str]],
    thresholds: DiagramThresholds,
    force: str,
) -> Eligibility:
    """Decide which diagram kinds this run may attempt, and record why.

    Args:
        repo_root: Absolute path the changed files are read under.
        changed_files: Repository-relative POSIX paths in the diff.
        hunk_ranges: ``{file: [(new_start, new_end), ...]}`` head-side changed
            ranges, from ``hunk_index.head_side_ranges_by_file``.
        stacks: ``detect_stacks(sorted(changed_files))`` -- un-collapsed and
            un-sharded (see the module docstring).
        services: ``services.enumerate_services`` output for the repository.
        import_graph: ``dependency.build_import_graph`` output over the changed
            files. An empty graph simply denies the cross-module rule.
        thresholds: Resolved ``[tool.daydream.diagram]`` thresholds.
        force: The resolved diagram mode -- ``"auto"``, ``"sequence"``,
            ``"flowchart"``, ``"both"`` or ``"off"``. An unrecognized value is
            treated as ``"auto"`` (this function never raises; the CLI and
            config-file parsers are the vocabulary gate).

    Returns:
        The :class:`Eligibility` record, whose ``to_dict`` is written to
        ``diagram.json`` so the decision is auditable after the fact.
    """
    code_files = _code_files(stacks, changed_files)
    modules = {path: _module_of(path, services) for path in code_files}
    service_names = {
        path: owner.name
        for path, owner in ((path, _owning_service(path, services)) for path in code_files)
        if owner is not None
    }
    cross_module_edges = _count_cross_module_edges(import_graph, modules)

    # An "off" run still records the signals that cost no parsing, so the
    # artifact shows what the run would have been eligible for; the branch
    # sweep is the one input expensive enough to be worth skipping.
    changed_functions = (
        [] if force == "off" else _changed_functions(repo_root, code_files, hunk_ranges)
    )
    qualifying = [
        root for root in changed_functions if root.branch_points >= thresholds.min_branch_points
    ]

    if force == "off":
        sequence = flowchart = KindDecision(eligible=False, rule=None, reason=_OFF_REASON)
    elif force in _FORCED_KINDS:
        note = (
            ""
            if qualifying or "flowchart" not in _FORCED_KINDS[force]
            else (
                " No changed function meets the branch-point threshold, so every changed"
                " function is offered as a candidate root."
            )
        )
        sequence = _forced_decision("sequence", force)
        flowchart = _forced_decision("flowchart", force, note=note)
    else:
        sequence = _decide_sequence(
            code_files=code_files,
            modules=modules,
            services=service_names,
            cross_module_edges=cross_module_edges,
            thresholds=thresholds,
        )
        flowchart = _decide_flowchart(
            qualifying=qualifying,
            changed_functions=changed_functions,
            minimum=thresholds.min_branch_points,
        )

    candidate_roots = qualifying
    if flowchart.eligible and flowchart.rule == "forced" and not qualifying:
        candidate_roots = changed_functions

    return Eligibility(
        code_files=code_files,
        modules=modules,
        services=service_names,
        cross_module_edges=cross_module_edges,
        function_branch_counts=[root for root in changed_functions if root.branch_points >= 1],
        candidate_roots=list(candidate_roots),
        sequence=sequence,
        flowchart=flowchart,
        thresholds=thresholds,
        force=force,
    )


__all__ = [
    "CandidateRoot",
    "DiagramThresholds",
    "Eligibility",
    "KindDecision",
    "count_function_branch_points",
    "decide_eligibility",
]
