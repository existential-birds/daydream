"""Findings for review and fix phases."""

import json
import re
from pathlib import Path
from typing import Any

from daydream import agent, ui
from daydream.artifact_visibility import (
    ArtifactSession,
    review_output_path_for,
)
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.dedup import (
    FOLD_SIM_THRESHOLD,
    bigrams,
    descriptions_match,
    jaccard,
    normalize_title,
)
from daydream.deep.location_validator import validate_records
from daydream.deep.records import (
    RECORD_SOURCE_UIDS_KEY,
    item_source_uids,
    record_issues,
    record_issues_or_empty,
    record_uid,
    stamp_item_uids,
    union_source_uids,
)
from daydream.deep.render import render_report
from daydream.hunk_index import load_hunk_index
from daydream.severity import SEVERITY_RANK, normalize_severity, stronger_severity


class CrossStackMergeError(ValueError):
    """Unparseable merge output carrying response_shape and input-order stack_context.

    The orchestrator salvages completed records when the current items envelope
    cannot supply a valid merged list.
    """

    def __init__(
        self,
        response_shape: str,
        stack_context: list[str],
        *,
        message: str | None = None,
        budget_reason: str | None = None,
    ) -> None:
        self.budget_reason = budget_reason
        self.response_shape = response_shape
        self.stack_context = stack_context
        super().__init__(
            message
            or (
                f"Cross-stack merge returned no item list (got {response_shape}); "
                f"stacks: {stack_context}"
            )
        )


def normalize_items(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy items, assign dense display IDs, and preserve or mint durable item_uids.

    Order and other fields survive. Host-owned item_uids are separate from record
    UIDs/source attribution and survive reordering. Non-list input raises ValueError.
    """
    if not isinstance(raw, list):
        raise ValueError(f"normalize_items expected a list, got {type(raw).__name__}")
    normalized: list[dict[str, Any]] = [
        {**item, "id": new_id} for new_id, item in enumerate(raw, start=1)
    ]
    # Reserve existing item_uids across the list before minting: preserving
    # item:2 at position 1 must not mint another item:2 at position 2.
    stamp_item_uids(normalized)
    return normalized


# Placeholder evidence strings that are structurally present but carry no
# grounding -- treated as "no evidence" by the gate (issue #227).
_PLACEHOLDER_EVIDENCE: frozenset[str] = frozenset({"n/a", "none", "-"})


def _is_evidenced(item: dict[str, Any]) -> bool:
    """Require concrete evidence and a grounded citation except for host-tagged structural items."""
    # Authored LOW confidence is speculative; location demotion runs after this gate.
    if str(item.get("confidence", "")).upper() == "LOW":
        return False

    evidence = str(item.get("evidence", "")).strip()
    if not evidence or evidence.lower() in _PLACEHOLDER_EVIDENCE:
        return False

    rationale = str(item.get("rationale", ""))
    if "no exploration evidence" in rationale.lower():
        return False

    file_val = str(item.get("file", ""))
    line_val = item.get("line", 0)
    has_file_line = bool(file_val) and isinstance(line_val, int) and line_val > 0
    # Require a path component in path:line evidence, excluding port:8080 and
    # bare symbols such as helper:42; those need explicit file/line grounding.
    has_citation = bool(re.search(r"\S*[./]\S*:\d+", evidence))

    # Host-tagged structural findings may be whole-file (line:0) with colon-free
    # evidence. Nonblank evidence suffices for this distinct trust class.
    if item.get("lens") == "structural":
        return True

    return has_file_line or has_citation


def _evidence_gate_then_validate(
    raw_items: list[dict[str, Any]],
    items_path: Path,
) -> list[dict[str, Any]]:
    """Drop speculative findings before location validation can demote retained findings.

    Both use LOW confidence with different meanings, so order is mandatory. Persist
    aligned dropped IDs, UIDs, and source UIDs, then snap/demote the survivors.
    """

    evidenced: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for item in raw_items:
        (evidenced if _is_evidenced(item) else dropped).append(item)

    if dropped:
        sidecar_path = items_path.parent / "dropped-speculative.json"
        # Reviewer display IDs restart per stack; durable UIDs retain the source identity.
        dropped_ids = [d.get("id") for d in dropped]
        # Only merge-bypass items retain a birth UID. Merge-agent items identify
        # sources separately; use "" for missing object UIDs so all three dropped
        # arrays remain positionally aligned.
        dropped_uids = [record_uid(d) for d in dropped]
        # Preserve per-item derivation groups, including merge-agent items without
        # their own UID. item_source_uids falls back to birth UID on bypass paths.
        dropped_source_uids = [item_source_uids(d) for d in dropped]
        sidecar_path.write_text(
            json.dumps(
                {
                    "dropped_count": len(dropped),
                    "dropped_ids": dropped_ids,
                    "dropped_uids": dropped_uids,
                    "dropped_source_uids": dropped_source_uids,
                    "dropped_items": dropped,
                },
                indent=2,
            )
        )
        ui.print_info(
            agent.console,
            f"Evidence gate: dropped {len(dropped)} speculative finding(s) "
            f"(ids: {dropped_ids}), wrote {sidecar_path}",
        )

    index = load_hunk_index(items_path.parent.parent)
    return validate_records(index, evidenced)


def severity_sorted(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable-sort canonical items by severity (high < medium < low)."""
    return sorted(items, key=lambda it: SEVERITY_RANK.get(it.get("severity") or "", 1))


def _fold_structural_duplicates(
    base_items: list[dict[str, Any]],
    structural_items: list[dict[str, Any]],
    items_path: Path,
) -> list[dict[str, Any]]:
    """Fold same-file structural twins, keeping grounded evidence and both provenances.

    Choose the best description match, then the strongest normalized severity,
    then the earliest input position. Record similarity ties for auditability.
    """
    surviving: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for item in structural_items:
        file_key = str(item.get("file", ""))
        description = str(item.get("description", ""))
        candidates: list[tuple[int, dict[str, Any], float]] = []
        if file_key and description:
            item_bigrams = bigrams(normalize_title(description))
            for index, base in enumerate(base_items):
                base_description = str(base.get("description", ""))
                if str(base.get("file", "")) == file_key and descriptions_match(
                    description, base_description, threshold=FOLD_SIM_THRESHOLD
                ):
                    similarity = jaccard(item_bigrams, bigrams(normalize_title(base_description)))
                    candidates.append((index, base, similarity))
        if not candidates:
            surviving.append(item)
            continue
        similarity = max(score for _, _, score in candidates)
        tied = [(index, base) for index, base, score in candidates if score == similarity]
        twin_index, twin = min(
            tied,
            key=lambda candidate: SEVERITY_RANK.get(
                normalize_severity(candidate[1].get("severity")) or "", len(SEVERITY_RANK)
            ),
        )
        # The evidence gate must not discard both twins. An evidenced structural
        # item survives when its base twin would be dropped as speculative.
        survivor_is_structural = not _is_evidenced(twin) and _is_evidenced(item)
        survivor, absorbed = (item, twin) if survivor_is_structural else (twin, item)
        source_uids = union_source_uids(item_source_uids(survivor), item_source_uids(absorbed))
        survivor[RECORD_SOURCE_UIDS_KEY] = source_uids
        severity = stronger_severity(twin.get("severity"), item.get("severity"))
        if severity is not None:
            survivor["severity"] = severity
        if survivor_is_structural:
            surviving.append(item)
        fold_records.append({
            "structural_description": description,
            "structural_file": file_key,
            "base_description": str(twin.get("description", "")),
            "similarity": similarity,
            # Merge-agent items usually have no record uid; their position still
            # identifies which base item received the in-place severity update.
            "base_uid": record_uid(twin),
            "base_index": twin_index,
            "tied_candidates": len(tied),
            "survivor": "structural" if survivor_is_structural else "base",
            "source_uids": source_uids,
        })
    if fold_records:
        sidecar_path = items_path.parent / "folded-structural.json"
        sidecar_path.write_text(json.dumps({"folded_count": len(fold_records), "folded": fold_records}, indent=2))
        ui.print_info(
            agent.console,
            f"Folded {len(fold_records)} structural finding(s) into the language-stack "
            f"finding reporting the same defect, wrote {sidecar_path}",
        )
    return surviving


def _record_item(record: dict[str, Any], lens: str) -> dict[str, Any]:
    """Host-authored items retain their record identity and explicit provenance."""
    return {**record, "lens": lens, RECORD_SOURCE_UIDS_KEY: union_source_uids([record_uid(record)])}


def _load_structural_items(path: Path | None) -> list[dict[str, Any]]:
    """Load current structural records; requested artifacts must remain readable."""
    if path is None:
        return []
    records = record_issues(json.loads(path.read_text()))
    if records is None or any(not isinstance(record, dict) or not record_uid(record) for record in records):
        raise ValueError('Structural records require the current host identity envelope')
    return [_record_item(record, 'structural') for record in records]


def _append_structural_and_write_merged(
    base_items: list[dict[str, Any]],
    structural_records_path: Path | None,
    items_path: Path,
    report_path: Path,
    canonical_path: Path,
) -> None:
    """Fold structural twins, gate evidence, validate locations, then publish both views."""
    structural = _fold_structural_duplicates(base_items, _load_structural_items(structural_records_path), items_path)
    items = normalize_items(_evidence_gate_then_validate(base_items + structural, items_path))
    items_path.write_text(json.dumps({"items": items}, indent=2))
    ui.print_info(agent.console, f"Merged into {len(items)} items")
    report = render_report(items)
    report_path.write_text(report)
    canonical_path.write_text(report)


def _reset_merged_outputs(canonical_path: Path, report_path: Path, items_path: Path) -> None:
    """Clear prior outputs and drop/fold sidecars so fresh results cannot retain stale audits."""
    canonical_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)
    items_path.unlink(missing_ok=True)
    (items_path.parent / "dropped-speculative.json").unlink(missing_ok=True)
    (items_path.parent / "folded-structural.json").unlink(missing_ok=True)


def _write_single_stack_merged_items(
    repo: Path,
    deep_dir_path: Path,
    all_records: list[dict[str, Any]],
    structural_records_path: Path | None,
    *,
    failed_stacks: dict[str, str] | None = None,
    artifact_session: ArtifactSession | None = None,
    allow_standalone: bool = False,
) -> None:
    """Normalize single-stack records and structural findings into the merged artifact."""
    canonical_path = review_output_path_for(
        repo,
        session=artifact_session,
        allow_standalone=allow_standalone,
    )
    report_path = DeepArtifact.MERGED_REPORT.at(deep_dir_path)
    items_path = DeepArtifact.MERGED_ITEMS.at(deep_dir_path)

    # The bypass must expose failed reviewers just as the merge prompt does.
    if failed_stacks:
        ui.print_warning(
            agent.console,
            f"Single-stack run completed with {len(failed_stacks)} failed stack(s): "
            + ", ".join(sorted(failed_stacks)),
        )

    # Clear stale outputs (mirrors phase_cross_stack_merge).
    _reset_merged_outputs(canonical_path, report_path, items_path)
    _append_structural_and_write_merged(
        [_record_item(record, "per-stack") for record in all_records],
        structural_records_path, items_path, report_path, canonical_path,
    )


#: Cap on how many unknown uids the aggregate provenance warning names inline.
#: The warning exists to be read, and a model that hallucinates provenance can
#: hallucinate a lot of it; naming the first few and counting the rest keeps the
#: message actionable instead of turning it into the per-item flood the
#: aggregation was there to avoid.
_MAX_REPORTED_UNKNOWN_UIDS = 10


def _validate_agent_source_uids(
    agent_items: list[dict[str, Any]],
    per_stack_records_paths: list[Path],
    structural_records_path: Path | None,
) -> None:
    """Clamp model source_uids to the run's real language/structural record pool in place.

    Unknown claims are discarded without dropping findings. Every dict item receives
    a source_uids list, possibly empty to honestly represent absent provenance.
    """
    # Unreadable records cannot establish model-owned provenance.
    pool: set[str] = set()
    records_paths = list(per_stack_records_paths)
    if structural_records_path is not None:
        records_paths.append(structural_records_path)
    for records_path in records_paths:
        if not records_path.is_file():
            continue
        try:
            loaded = json.loads(records_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            ui.print_warning(
                agent.console,
                f"Skipping {records_path.name} for source_uids validation: {type(exc).__name__}: {exc}",
            )
            continue
        issues = record_issues_or_empty(loaded)
        if not isinstance(issues, list):
            continue
        for record in issues:
            if isinstance(record, dict):
                uid = record_uid(record)
                if uid:
                    pool.add(uid)

    unknown: dict[str, None] = {}
    unattributed = 0
    for item in agent_items:
        if not isinstance(item, dict):
            continue
        raw = item.get(RECORD_SOURCE_UIDS_KEY)
        # Missing, null, and wrong-typed values claim no source UIDs.
        claimed = raw if isinstance(raw, list) else []
        for uid in claimed:
            if isinstance(uid, str) and uid and uid not in pool:
                unknown.setdefault(uid, None)
        item[RECORD_SOURCE_UIDS_KEY] = union_source_uids(
            uid for uid in claimed if isinstance(uid, str) and uid in pool
        )
        if not item[RECORD_SOURCE_UIDS_KEY]:
            unattributed += 1

    if unknown:
        # Aggregate unknown UIDs so repeated bad claims cannot flood the output.
        named = sorted(unknown)
        shown = ", ".join(named[:_MAX_REPORTED_UNKNOWN_UIDS])
        if len(named) > _MAX_REPORTED_UNKNOWN_UIDS:
            shown += f", +{len(named) - _MAX_REPORTED_UNKNOWN_UIDS} more"
        ui.print_warning(
            agent.console,
            f"Merge agent cited {len(named)} source_uid(s) that match no record in this run "
            f"({shown}); dropped them from item provenance. "
            f"{unattributed} of {len(agent_items)} merged item(s) now carry no record attribution.",
        )
