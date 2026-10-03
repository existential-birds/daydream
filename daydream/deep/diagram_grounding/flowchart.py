"""Ground, prune, cap and floor-test flowchart proposals."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from daydream.config import (
    DIAGRAM_MAX_EDGES,
    DIAGRAM_MAX_NODES,
)
from daydream.deep.diagram_grounding import evidence
from daydream.deep.diagram_grounding.models import ElementCheck, GroundingReport
from daydream.deep.diagram_types import (
    CandidateRoot,
    FlowchartSpec,
    as_int as _norm_line,
)
from daydream.repository_paths import strip_dot_slash

_MIN_NODES = 4


def _match_candidate(
    root: dict[str, Any], candidate_roots: list[CandidateRoot]
) -> CandidateRoot | None:
    """Match the proposed (file, name, line) to a root with a verified range."""
    file = strip_dot_slash(evidence.norm_str(root.get("file")))
    name = evidence.norm_str(root.get("name"))
    line = _norm_line(root.get("line"))
    for candidate in candidate_roots:
        if (
            strip_dot_slash(candidate.file) == file
            and candidate.name == name
            and candidate.line == line
        ):
            return candidate
    return None


def _ground_node(
    repo_root: Path,
    sources: evidence.SourceCache,
    symbols: evidence.RepoSymbols,
    hunk_ranges: dict[str, list[tuple[int, int]]],
    record: dict[str, Any],
    root_file: str,
    root_range: tuple[int, int],
) -> ElementCheck:
    """Check one flowchart node and rewrite its evidence line on a snap."""
    check = ElementCheck("node", record["id"], True)
    citation = record["evidence"]
    file, line, reason = evidence.check_location(
        repo_root, sources, citation["file"], citation["line"]
    )
    citation["file"], citation["line"] = file, line
    if reason is not None:
        check.grounded, check.reason = False, reason
        return check
    if file != root_file or not (root_range[0] <= line <= root_range[1]):
        check.grounded, check.reason = False, "NODE_OUTSIDE_ROOT"
        return check

    symbol = citation["symbol"]
    if record["kind"] == "subroutine":
        if not symbol:
            check.grounded, check.reason = False, "MALFORMED_ELEMENT"
            return check
        if not evidence.token_on_line(sources.line_text(file, line), symbol):
            snapped = evidence.snap_symbol(sources, file, line, symbol, within=root_range)
            if snapped is None:
                check.grounded, check.reason = False, "SUBROUTINE_NOT_CALLED_HERE"
                return check
            citation["line"] = line = snapped
            check.snapped_line = snapped
        strength, defined_at = evidence.resolve_anywhere(repo_root, symbols, symbol)
        if strength is None:
            check.grounded, check.reason = False, "SUBROUTINE_NOT_DEFINED"
            check.in_changed_hunk = evidence.in_ranges(hunk_ranges.get(file, []), line)
            return check
        check.strength, check.defined_at = strength, defined_at
    elif symbol and not evidence.token_on_line(sources.line_text(file, line), symbol):
        snapped = evidence.snap_symbol(sources, file, line, symbol, within=root_range)
        if snapped is None:
            check.grounded, check.reason = False, "SYMBOL_NOT_ON_LINE"
            check.in_changed_hunk = evidence.in_ranges(hunk_ranges.get(file, []), line)
            return check
        citation["line"] = line = snapped
        check.snapped_line = snapped
        check.strength = "token"
    elif symbol:
        check.strength = "token"

    if record["kind"] == "end" and not evidence.terminal_line(sources, file, line):
        check.grounded, check.reason = False, "NOT_A_TERMINAL_STATEMENT"
    elif record["kind"] == "decision" and not evidence.branch_line(sources, file, line):
        check.grounded, check.reason = False, "NOT_A_BRANCH_STATEMENT"
    elif record["kind"] not in {"end", "decision"} and not evidence.executable_line(
        sources, file, line
    ):
        check.grounded, check.reason = False, "NOT_AN_EXECUTABLE_STATEMENT"
    check.in_changed_hunk = evidence.in_ranges(hunk_ranges.get(file, []), line)
    return check


def _decision_edge_reason(
    edge: dict[str, Any], nodes: dict[str, dict[str, Any]], seen_labels: set[str]
) -> str | None:
    """Require distinct, nonempty labels on decision edges.

    Thin decisions are later demoted to process nodes by structural pruning.
    """
    if nodes.get(edge["from"], {}).get("kind") != "decision":
        return None
    label = edge["label"]
    if not label or label in seen_labels:
        return "DECISION_EDGES_INVALID"
    seen_labels.add(label)
    return None


def _edges_with_live_endpoints(
    edges: list[tuple[str, dict[str, Any]]], alive: set[str]
) -> list[tuple[str, dict[str, Any]]]:
    """Drop edges whose ``from`` or ``to`` endpoint has been pruned."""
    return [item for item in edges if item[1]["from"] in alive and item[1]["to"] in alive]


def _structural_pass(
    node_ids: list[str],
    nodes: dict[str, dict[str, Any]],
    edges: list[tuple[str, dict[str, Any]]],
    start_id: str | None,
) -> tuple[list[str], list[tuple[str, dict[str, Any]]], set[str]]:
    """Keep nodes reachable from start and demote decisions with fewer than two labels.

    Demotion changes only rendering, so it cannot change reachability. Preserve
    demotions for unreachable nodes too; callers may use the full diagnostic set.
    """
    alive_edges = _edges_with_live_endpoints(edges, set(node_ids))
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for _, edge in alive_edges:
        outgoing.setdefault(edge["from"], []).append(edge)
    demoted = {
        node_id for node_id in node_ids
        if nodes[node_id]["kind"] == "decision" and len({
            edge["label"] for edge in outgoing.get(node_id, []) if edge["label"]
        }) < 2
    }
    reachable: set[str] = set()
    frontier = [start_id] if start_id is not None and start_id in node_ids else []
    while frontier:
        current = frontier.pop()
        if current in reachable:
            continue
        reachable.add(current)
        frontier.extend(edge["to"] for edge in outgoing.get(current, []))
    return (
        [node_id for node_id in node_ids if node_id in reachable],
        _edges_with_live_endpoints(alive_edges, reachable),
        demoted,
    )


def _rejected_root(root: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the schema-shaped root, or None when it is not even well-formed."""
    if root is None:
        return None
    file = strip_dot_slash(evidence.norm_str(root.get("file")))
    name = evidence.norm_str(root.get("name"))
    line = _norm_line(root.get("line"))
    if not file or not name or line < 1:
        return None
    return {"file": file, "name": name, "line": line}


def ground_flowchart(
    spec: FlowchartSpec,
    *,
    repo_root: Path,
    hunk_ranges: dict[str, list[tuple[int, int]]],
    candidate_roots: list[CandidateRoot],
    symbols: evidence.RepoSymbols,
) -> GroundingReport[FlowchartSpec]:
    """Ground, prune, cap and floor-test a proposal rooted in an eligible changed function.

    Consume a proposal admitted by coerce_flowchart_spec. Scalar shape and node-ID
    admission belong to that boundary. The root must match candidate_roots and
    still overlap a changed hunk, including
    on repair turns. A rejected root yields no nodes or edges.
    """
    spec = copy.deepcopy(spec)
    sources = evidence.SourceCache(repo_root)
    root = spec["root"]
    candidate = _match_candidate(root, candidate_roots) if root is not None else None
    root_name = evidence.norm_str(root.get("name")) if root is not None else ""
    root_ref = root_name or "<no-root>"
    if candidate is None or not _overlaps_hunks(candidate, hunk_ranges):
        check = ElementCheck("root", root_ref, False, "ROOT_NOT_CANDIDATE")
        return GroundingReport(
            elements=[check],
            spec_final={"root": _rejected_root(root), "nodes": [], "edges": []},
            omit_reasons=["TOO_FEW_NODES"],
            rejected="ROOT_NOT_CANDIDATE",
        )
    root_range = (candidate.line, candidate.end_line)
    root_file = strip_dot_slash(candidate.file)
    root_check = ElementCheck(
        "root", root_ref, True, in_changed_hunk=True
    )
    root_final = {"file": root_file, "name": candidate.name, "line": candidate.line}

    # --- node checks ---------------------------------------------------------
    node_checks: list[ElementCheck] = []
    nodes: dict[str, dict[str, Any]] = {}
    node_order: list[str] = []
    start_seen = False
    for record in spec["nodes"]:
        node_id = record["id"]
        check = _ground_node(
            repo_root,
            sources,
            symbols,
            hunk_ranges,
            record,
            root_file,
            root_range,
        )
        if check.grounded and record["kind"] == "start":
            if start_seen:
                check.grounded, check.reason = False, "MULTIPLE_START"
            else:
                start_seen = True
        node_checks.append(check)
        if check.grounded:
            nodes[node_id] = record
            node_order.append(node_id)

    # --- edge checks ---------------------------------------------------------
    edge_checks: list[ElementCheck] = []
    edges: list[tuple[str, dict[str, Any]]] = []
    seen_refs: set[str] = set()
    decision_labels: dict[str, set[str]] = {}
    for record in spec["edges"]:
        ref = f"{record['from']}->{record['to']}"
        if ref in seen_refs:
            edge_checks.append(
                ElementCheck(
                    "edge", ref,
                    False,
                    "MALFORMED_ELEMENT",
                )
            )
            continue
        seen_refs.add(ref)
        if record["from"] not in nodes or record["to"] not in nodes:
            edge_checks.append(
                ElementCheck("edge", ref, False, "EDGE_ENDPOINT_UNGROUNDED")
            )
            continue
        reason = _decision_edge_reason(
            record, nodes, decision_labels.setdefault(record["from"], set())
        )
        edge_checks.append(ElementCheck("edge", ref, reason is None, reason))
        if reason is None:
            edges.append((ref, record))

    start_id = next(
        (node_id for node_id in node_order if nodes[node_id]["kind"] == "start"), None
    )

    # --- prune ---------------------------------------------------------------
    kept_ids, kept_edges, demoted = _structural_pass(node_order, nodes, edges, start_id)
    # Pruned endpoints need repair; cap-trimmed edges remain grounded.
    survived_prune = {ref for ref, _ in kept_edges}
    for check in edge_checks:
        if check.grounded and check.ref not in survived_prune:
            check.grounded, check.reason = False, "EDGE_ENDPOINT_UNGROUNDED"

    # --- cap -----------------------------------------------------------------
    if len(kept_ids) > DIAGRAM_MAX_NODES:
        head = kept_ids[:DIAGRAM_MAX_NODES]
        if start_id is not None and start_id not in head:
            # Keep the start node within the cap; otherwise nothing is reachable.
            head = [start_id, *head[:-1]]
        kept_ids = head
    kept_edges = _edges_with_live_endpoints(kept_edges, set(kept_ids))
    kept_edges = kept_edges[:DIAGRAM_MAX_EDGES]
    kept_ids, kept_edges, demoted = _structural_pass(
        kept_ids, nodes, kept_edges, start_id
    )

    # --- assemble ------------------------------------------------------------
    spec_final: FlowchartSpec = {
        "root": root_final,
        "nodes": [
            {
                **nodes[node_id],
                "kind": "process" if node_id in demoted else nodes[node_id]["kind"],
            }
            for node_id in kept_ids
        ],
        "edges": [dict(record) for _, record in kept_edges],
    }

    # --- floor ---------------------------------------------------------------
    kinds = [node["kind"] for node in spec_final["nodes"]]
    omit_reasons: list[str] = []
    if len(kinds) < _MIN_NODES or "start" not in kinds:
        omit_reasons.append("TOO_FEW_NODES")
    if "end" not in kinds:
        omit_reasons.append("NO_END")
    if "decision" not in kinds:
        omit_reasons.append("NO_DECISION")

    elements = [root_check, *node_checks, *edge_checks]
    return GroundingReport(
        elements=elements,
        spec_final=spec_final,
        omit_reasons=omit_reasons,
        rejected=None,
    )


def _overlaps_hunks(
    candidate: CandidateRoot, hunk_ranges: dict[str, list[tuple[int, int]]]
) -> bool:
    """Whether the candidate's range overlaps a head-side changed hunk."""
    ranges = hunk_ranges.get(strip_dot_slash(candidate.file), [])
    return any(
        start <= candidate.end_line and candidate.line <= end for start, end in ranges
    )
