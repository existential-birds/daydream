"""Ground, prune, cap and floor-test sequence-diagram proposals."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.config import (
    DIAGRAM_MAX_BLOCKS,
    DIAGRAM_MAX_MESSAGES,
    DIAGRAM_MAX_PARTICIPANTS,
)
from daydream.deep.diagram_grounding import evidence
from daydream.deep.diagram_grounding.models import ElementCheck, GroundingReport
from daydream.deep.diagram_types import (
    SequenceSpec,
)

_MIN_MESSAGES = 3
_MIN_PARTICIPANTS = 2


def _ground_participants(
    repo_root: Path,
    sources: evidence.SourceCache,
    participants: list[dict[str, Any]],
    source_indices: dict[str, list[int]],
) -> tuple[list[ElementCheck], dict[str, dict[str, Any]]]:
    """Validate participants; external actors may source only the entrypoint message.

    External participants cannot declare repository files. Internal participants
    must have confined, readable files.
    """
    checks: list[ElementCheck] = []
    accepted: dict[str, dict[str, Any]] = {}
    for record in participants:
        name = record["name"]
        reason: str | None = None
        if record["kind"] == "external":
            later = [i for i in source_indices.get(name, []) if i != 0]
            if record["files"] or later:
                reason = "EXTERNAL_MISUSED"
        elif not record["files"]:
            reason = "PARTICIPANT_NO_FILES"
        else:
            for file in record["files"]:
                normalized, path_reason = evidence.check_path(repo_root, file)
                if path_reason is not None:
                    reason = path_reason
                    break
                if sources.read(normalized) is None:
                    reason = "PARTICIPANT_FILE_MISSING"
                    break
        checks.append(ElementCheck("participant", name, reason is None, reason))
        if reason is None:
            accepted[name] = record
    return checks, accepted


def _message_source_files(
    index: int,
    record: dict[str, Any],
    accepted: dict[str, dict[str, Any]],
) -> list[str]:
    """Return the sender's files, except an external entrypoint cites its receiver.

    That entrypoint has no repository source and must cite the handler definition.
    """
    sender = accepted.get(record["from"], {})
    if index == 0 and sender.get("kind") == "external":
        return list(accepted.get(record["to"], {}).get("files", []))
    return list(sender.get("files", []))


def _ground_message(
    repo_root: Path,
    sources: evidence.SourceCache,
    symbols: evidence.RepoSymbols,
    hunk_ranges: dict[str, list[tuple[int, int]]],
    index: int,
    record: dict[str, Any],
    previous: dict[str, Any] | None,
    previous_check: ElementCheck | None,
    accepted: dict[str, dict[str, Any]],
) -> ElementCheck:
    """Check one message and rewrite its evidence line on a successful snap."""
    check = ElementCheck("message", str(index), True)
    # An unknown or ungrounded endpoint is not a separate reason code: the
    # message fails on the endpoint that cannot hold its evidence (source) or
    # cannot own its callee (target), which is what the repair turn must fix.
    if record["from"] not in accepted:
        check.grounded, check.reason = False, "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"
        return check
    if record["to"] not in accepted:
        check.grounded, check.reason = False, "CALLEE_NOT_DEFINED_IN_TARGET"
        return check

    citation = record["evidence"]
    file, line, reason = evidence.check_location(
        repo_root, sources, citation["file"], citation["line"]
    )
    citation["file"], citation["line"] = file, line
    if reason is not None:
        check.grounded, check.reason = False, reason
        return check

    source_files = _message_source_files(index, record, accepted)
    if file not in source_files:
        check.grounded, check.reason = False, "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"
        return check

    symbol = citation["symbol"]
    if record["kind"] == "reply":
        if (
            previous is None
            or previous_check is None
            or not previous_check.grounded
            or previous["kind"] != "call"
            or previous["from"] != record["to"]
            or previous["to"] != record["from"]
        ):
            check.grounded, check.reason = False, "REPLY_NOT_PRECEDED_BY_CALL"
            return check
        if not evidence.reply_line(sources, file, line):
            check.grounded, check.reason = False, "NOT_A_REPLY_STATEMENT"
            return check
        definition = evidence.reply_definition(symbols, symbol, file, line)
        if definition is None:
            check.grounded, check.reason = False, "REPLY_NOT_IN_ENCLOSING_FUNCTION"
            return check
        check.strength = "definition"
        check.defined_at = evidence.definition_location(definition)
        check.in_changed_hunk = evidence.in_ranges(hunk_ranges.get(file, []), line)
        return check

    if not symbol or not evidence.token_on_line(sources.line_text(file, line), symbol):
        snapped = evidence.snap_symbol(sources, file, line, symbol)
        if snapped is None:
            check.grounded, check.reason = False, "SYMBOL_NOT_ON_LINE"
            check.in_changed_hunk = evidence.in_ranges(hunk_ranges.get(file, []), line)
            return check
        citation["line"] = line = snapped
        check.snapped_line = snapped
    check.in_changed_hunk = evidence.in_ranges(hunk_ranges.get(file, []), line)

    target = accepted[record["to"]]
    if record["kind"] in ("call", "self") and target["kind"] == "internal":
        strength, defined_at = evidence.resolve_in_files(
            repo_root, symbols, symbol, target["files"]
        )
        if strength is None:
            check.grounded, check.reason = False, "CALLEE_NOT_DEFINED_IN_TARGET"
            return check
        check.strength, check.defined_at = strength, defined_at
    else:
        # An outbound call names a client method rather than a repository callee,
        # so the word-boundary match on the cited line is the whole proof.
        check.strength = "token"
    return check


def _ground_branch(
    repo_root: Path,
    sources: evidence.SourceCache,
    hunk_ranges: dict[str, list[tuple[int, int]]],
    ref: str,
    record: dict[str, Any],
) -> tuple[ElementCheck, dict[str, Any]]:
    """Check one block branch, returning its check and its ``spec_final`` payload."""
    citation = record["evidence"]
    condition = record["condition"]
    file, line, reason = evidence.check_location(
        repo_root, sources, citation.get("file"), citation.get("line")
    )
    if reason is None and (not condition or not evidence.branch_line(sources, file, line)):
        reason = "NOT_A_BRANCH_STATEMENT"
    check = ElementCheck("branch", ref, reason is None, reason)
    check.in_changed_hunk = evidence.in_ranges(hunk_ranges.get(file, []), line)
    payload = {
        "condition": condition,
        "evidence": {"file": file, "line": line},
        "messages": record["messages"],
    }
    return check, payload


@dataclass
class _KeptBlock:
    """One surviving block's kind and branches."""

    kind: str
    branches: list[dict[str, Any]]


def _assemble_blocks(
    kept: list[_KeptBlock], final_positions: dict[int, int]
) -> list[_KeptBlock]:
    """Remap surviving message indices, dropping empty branches and blocks.

    Messages from dropped blocks render flat. alt needs two branches; opt/loop
    keep only the first.
    """
    result: list[_KeptBlock] = []
    for block in kept:
        branches: list[dict[str, Any]] = []
        for payload in block.branches:
            remapped = [
                final_positions[index]
                for index in payload["messages"]
                if index in final_positions
            ]
            if not remapped:
                continue
            branches.append({**payload, "messages": remapped})
        if block.kind == "alt":
            if len(branches) < 2:
                continue
        else:
            branches = branches[:1]
        if not branches:
            continue
        result.append(_KeptBlock(block.kind, branches))
    return result


def ground_sequence(
    spec: SequenceSpec,
    *,
    repo_root: Path,
    hunk_ranges: dict[str, list[tuple[int, int]]],
    symbols: evidence.RepoSymbols,
) -> GroundingReport[SequenceSpec]:
    """Ground against head-tree citations and changed hunks, then prune, cap and floor-test.

    Consume an admitted proposal from coerce_sequence_spec. Shape and message-index
    admission belongs to that boundary; source grounding owns the checks here.
    Render spec_final only when omit_reasons is empty. Share symbols across repairs.
    """
    spec = copy.deepcopy(spec)
    sources = evidence.SourceCache(repo_root)
    participants = spec["participants"]
    messages = spec["messages"]
    blocks = spec["blocks"]

    # Participant checks need to know which messages each name sources, so the
    # external-actor rule is decidable before any message is adjudicated.
    source_indices: dict[str, list[int]] = {}
    for index, record in enumerate(messages):
        source_indices.setdefault(record["from"], []).append(index)

    participant_checks, accepted = _ground_participants(
        repo_root,
        sources,
        participants,
        source_indices,
    )

    message_checks: list[ElementCheck] = []
    for index, record in enumerate(messages):
        message_checks.append(
            _ground_message(
                repo_root,
                sources,
                symbols,
                hunk_ranges,
                index,
                record,
                messages[index - 1] if index else None,
                message_checks[index - 1] if index else None,
                accepted,
            )
        )

    block_checks: list[ElementCheck] = []
    branch_checks: list[ElementCheck] = []
    kept_blocks: list[_KeptBlock] = []
    for block_index, record in enumerate(blocks):
        block_ref = f"b{block_index}"
        kind = record["kind"]
        raw_branches = record["branches"]
        surviving: list[dict[str, Any]] = []
        for branch_index, raw_branch in enumerate(raw_branches):
            check, payload = _ground_branch(
                repo_root,
                sources,
                hunk_ranges,
                f"{block_ref}.{branch_index}",
                raw_branch,
            )
            branch_checks.append(check)
            if check.grounded:
                surviving.append(payload)
        block_checks.append(
            ElementCheck(
                "block",
                block_ref,
                bool(surviving),
                None if surviving else "NOT_A_BRANCH_STATEMENT",
            )
        )
        if surviving:
            kept_blocks.append(_KeptBlock(kind, surviving))

    # --- prune ---------------------------------------------------------------
    kept_messages = [
        index for index, check in enumerate(message_checks) if check.grounded
    ]
    used = {
        name
        for index in kept_messages
        for name in (
            messages[index]["from"],
            messages[index]["to"],
        )
    }
    kept_participants = [name for name in accepted if name in used]
    # --- cap -----------------------------------------------------------------
    kept_participants = kept_participants[:DIAGRAM_MAX_PARTICIPANTS]
    participant_set = set(kept_participants)
    kept_messages = [
        index
        for index in kept_messages
        if messages[index]["from"] in participant_set
        and messages[index]["to"] in participant_set
    ][:DIAGRAM_MAX_MESSAGES]
    final_positions = {index: pos for pos, index in enumerate(kept_messages)}
    final_blocks = _assemble_blocks(kept_blocks[:DIAGRAM_MAX_BLOCKS], final_positions)

    # --- assemble ------------------------------------------------------------
    spec_final: SequenceSpec = {
        "participants": [accepted[name] for name in kept_participants],
        "messages": [messages[index] for index in kept_messages],
        "blocks": [
            {
                "kind": block.kind,
                "branches": block.branches,
            }
            for block in final_blocks
        ],
    }

    # --- floor ---------------------------------------------------------------
    omit_reasons: list[str] = []
    if len(spec_final["messages"]) < _MIN_MESSAGES:
        omit_reasons.append("TOO_FEW_MESSAGES")
    if len(spec_final["participants"]) < _MIN_PARTICIPANTS:
        omit_reasons.append("TOO_FEW_PARTICIPANTS")
    if not any(
        message_checks[index].in_changed_hunk for index in kept_messages
    ):
        omit_reasons.append("NO_CHANGED_INTERACTION")

    elements = participant_checks + message_checks + block_checks + branch_checks
    return GroundingReport(
        elements=elements,
        spec_final=spec_final,
        omit_reasons=omit_reasons,
        rejected=None,
    )
