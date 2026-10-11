"""Project per-finding training examples from immutable record snapshots, offline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, cast, overload

from daydream.dataset import LocalRecordStore
from daydream.json_utils import atomic_write_bytes, canonical_json
from daydream.training.admission import REASON_CODE_C5_EXCLUDED_REPO, REASON_CODE_REPO_IDENTITY_MISSING
from daydream.training.corpus import _is_posterior_leak, _trajectory_set_hash
from daydream.training.corpus_projection.provenance import extract_provenance
from daydream.training.corpus_projection.segments import segment
from daydream.training.corpus_projection.selection import (
    Record,
    _apply_share_caps,
    _share_caps_report,
    count_by,
    retain_group_limits,
)
from daydream.training.corpus_projection.splits import SPLIT_FILENAMES, assign_split
from daydream.training.corpus_projection.tiers import classify_tier
from daydream.training.exclusion import EXCLUSION_PATH, load_exclusion_list
from daydream.training.labeler_versions import LABELER_POLICY_VERSION, REPLY_CLASSIFIER_VERSION, RUBRIC_SCHEMA_VERSION
from daydream.training.record_evidence import sessions_from_snapshot
from daydream.training.record_identity import normalize_repo_slug, record_finding_id

SOURCE_IDENTITY_VERSION = "record-snapshot-v2"


def _dump_jsonl(records: list[Record]) -> str:
    return "".join(canonical_json(record) + "\n" for record in records)


def _write_artifact(path: Path, data: bytes) -> None:
    atomic_write_bytes(path, data, fsync=False, dir_fsync=False, mode=None)


@dataclass(frozen=True)
class BuildFrozenCorpusConfig:
    """Pinned input membership and projection settings; no network or repository checkout required."""

    out_dir: Path
    store_dir: Path
    snapshot_id: str
    as_of: str | None = None
    holdout_rate: float = 0.1
    val_rate: float = 0.1
    salt: str = "daydream-projection-salt"
    caps: dict[str, int] = field(default_factory=dict)
    max_stack_share: float | None = None
    max_repo_share: float | None = None
    max_profile_share: float | None = None
    labeler_policy_version: str = LABELER_POLICY_VERSION
    reply_classifier_version: str = REPLY_CLASSIFIER_VERSION
    rubric_schema_version: str = RUBRIC_SCHEMA_VERSION
    emit_process_traces: bool = False

    def __post_init__(self) -> None:
        for name in ("out_dir", "store_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        output_root = self.out_dir.expanduser().resolve()
        store_root = self.store_dir.expanduser().resolve()
        if output_root.is_relative_to(store_root) or store_root.is_relative_to(output_root):
            raise ValueError("corpus output overlaps the record store namespace")
        for name in ("max_stack_share", "max_repo_share", "max_profile_share"):
            share = getattr(self, name)
            if share is not None and not 0 < share <= 1:
                raise ValueError(f"{name} must be in (0.0, 1.0], got {share}")
        if self.holdout_rate < 0 or self.val_rate < 0 or self.holdout_rate + self.val_rate > 1:
            raise ValueError("split rates must be nonnegative and sum to at most 1")
        if self.as_of is not None:
            value = datetime.fromisoformat(self.as_of)
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise ValueError("as_of must be a UTC timestamp")
            object.__setattr__(self, "as_of", value.astimezone(timezone.utc).isoformat())


def _refuse_posterior_evidence(
    session_id: str, fingerprint: str, evidence: list[Record], as_of: str | None,
) -> None:
    for item in evidence:
        if isinstance(item, Mapping) and _is_posterior_leak(dict(item), as_of):
            raise ValueError(f"session {session_id!r} finding {fingerprint!r}: refusing posterior outcome evidence")


def _provenance_for(resolution: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    prov = extract_provenance(resolution)
    captured = extract_provenance(run)
    if not any(prov["profile"].values()):
        prov["profile"] = captured["profile"]
    if prov["stack"] is None:
        prov["stack"] = captured["stack"]
    return prov


def _max_valid_at(evidence: list[Record], base: str | None) -> str | None:
    result = base
    for item in evidence:
        if isinstance(item, Mapping) and item.get("valid_at"):
            candidate = str(item["valid_at"])
            if result is None or datetime.fromisoformat(candidate) > datetime.fromisoformat(result):
                result = candidate
    return result


@overload
def project_findings(session: Mapping[str, object], *, return_adjudication: Literal[False] = False) -> list[Record]: ...


@overload
def project_findings(
    session: Mapping[str, object], *, return_adjudication: Literal[True]
) -> tuple[list[Record], list[Record]]: ...


def project_findings(
    session: Mapping[str, object], *, return_adjudication: bool = False
) -> list[Record] | tuple[list[Record], list[Record]]:
    """Project one segmented session's per-finding resolutions into records.

    Returns the record list, or ``(records, adjudication_entries)`` when
    ``return_adjudication`` is true. Raises ``ValueError`` naming the
    session and the offending key on a malformed resolution.
    """
    session_id = session.get("session_id")
    trajectory_id = session.get("trajectory_id")
    segment_id = session.get("segment_id")
    resolutions = session.get("resolutions")
    for name, value in (
        ("session_id", session_id),
        ("trajectory_id", trajectory_id),
        ("segment_id", segment_id),
        ("resolutions", resolutions),
    ):
        if not value:
            raise ValueError(f"project_findings: session missing required key {name!r}")
    if not isinstance(resolutions, list):
        raise ValueError(
            f"project_findings: session {session_id!r} key 'resolutions' "
            f"must be a list, got {type(resolutions).__name__}"
        )

    records: list[Record] = []
    adjudication: list[Record] = []
    for index, resolution in enumerate(resolutions):
        if not isinstance(resolution, Mapping):
            raise ValueError(
                f"project_findings: session {session_id!r} resolutions[{index}] "
                f"is not a mapping (got {type(resolution).__name__})"
            )
        fingerprint = resolution.get("fingerprint")
        if not fingerprint:
            raise ValueError(
                f"project_findings: session {session_id!r} resolutions[{index}] "
                "missing required key 'fingerprint'"
            )
        tier = classify_tier(resolution)
        if tier == "gold" and resolution.get("gold_eligible") is False:
            tier = "silver"
        disposition = resolution.get("disposition")
        evidence = list(resolution.get("evidence") or [])
        provenance = extract_provenance(resolution)
        item_uid = resolution.get("item_uid")
        if not isinstance(item_uid, str) or not item_uid:
            raise ValueError(f"project_findings: session {session_id!r} resolution missing host item_uid")
        identity = record_finding_id(str(session_id), str(trajectory_id), str(segment_id), item_uid)
        record = {
            "record_id": identity,
            "record_type": "outcome-finding",
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "task_segment": segment_id,
            "finding_fingerprint": fingerprint,
            "tier": tier,
            "disposition": disposition,
            "outcome_label": disposition if tier == "gold" else None,
            "evidence": evidence,
            "reply_captures": resolution.get("reply_captures", []),
            "correction": resolution.get("correction"),
            "profile": provenance["profile"],
            "stack": provenance["stack"],
        }
        record["item_uid"] = item_uid
        records.append(record)
        if tier == "task-only":
            adjudication.append(
                {
                    "fingerprint": fingerprint,
                    "item_uid": item_uid,
                    "disposition": disposition,
                    "evidence": evidence,
                    "reply_captures": resolution.get("reply_captures", []),
                    "correction": resolution.get("correction"),
                    "exclusion_reason": (
                        f"non-decisive disposition {disposition!r} — missing decisive "
                        "human verdict (evidence carried for the adjudication pass)"
                    ),
                }
            )

    if return_adjudication:
        return records, adjudication
    return records


def build_frozen_corpus(config: BuildFrozenCorpusConfig) -> dict[str, Any]:
    """Project verified snapshot evidence; gate all admission before writing any output.

    Findings belong to the first trajectory segment in captured fork order. IDs
    use stable host UID identities; source lineage pins captured run/trajectory
    identity and immutable snapshot membership.
    """
    store = LocalRecordStore(config.store_dir)
    frozen = store.read_snapshot(config.snapshot_id)
    pin = frozen.snapshot["valid_before"]
    if config.as_of is not None and (
        pin is None or datetime.fromisoformat(pin) != datetime.fromisoformat(config.as_of)
    ):
        raise ValueError("as_of must match the frozen snapshot valid_before pin")
    as_of = config.as_of or pin
    source = frozen.snapshot.get("source") or {}
    hub_commit = source.get("revision")
    sessions = {row["session_id"]: row for row in sessions_from_snapshot(frozen)}
    runs = {run["run_id"]: run for run in frozen.runs}
    repo_slugs: dict[str, str] = {}
    identity_refusals: list[tuple[str, str]] = []
    excluded = {slug.casefold() for slug in load_exclusion_list()}
    for run_id in sessions:
        original_task = runs[run_id]["original_task"]
        task = original_task["value"] if original_task["status"] == "available" else {}
        raw_slug = task.get("repository", {}).get("repo_slug")
        slug = normalize_repo_slug(raw_slug) if isinstance(raw_slug, str) else ""
        if not slug:
            identity_refusals.append((run_id, REASON_CODE_REPO_IDENTITY_MISSING))
        elif slug.casefold() in excluded:
            identity_refusals.append((run_id, REASON_CODE_C5_EXCLUDED_REPO))
        else:
            repo_slugs[run_id] = slug
    if identity_refusals:
        details = "\n".join(f"{reason}: {run_id}" for run_id, reason in sorted(identity_refusals))
        raise ValueError(f"repository identity gate: refusing to project; no output written\n{details}")

    records: list[Record] = []
    adjudication: list[Record] = []
    exclusions_by_reason: dict[str, int] = {}
    run_digests = {member["identity"]: member["record_digest"] for member in frozen.snapshot["runs"]}
    for run_id, row in sessions.items():
        run = runs[run_id]
        trajectory_section = run["trajectories"]
        if trajectory_section["status"] != "available":
            raise ValueError(f"run {run_id}: retained trajectories are required for corpus projection")
        documents = trajectory_section["value"]["documents"]
        root = next(document for document in documents if document["trajectory_id"] == run_id)
        segments = segment(root)
        if not segments:
            raise ValueError(f"run {run_id}: no trajectory segments")
        first = segments[0]
        session_view = {"session_id": run_id, "trajectory_id": first.trajectory_id,
                        "segment_id": first.segment_id, "resolutions": row["resolutions"]}
        projected, report = project_findings(session_view, return_adjudication=True)
        adjudication.extend({"session_id": run_id, **entry} for entry in report)
        resolution_by_uid = {str(item["item_uid"]): item for item in row["resolutions"]}
        task = run["original_task"]["value"] if run["original_task"]["status"] == "available" else {}
        items = run["findings"]["value"]["items"] if run["findings"]["status"] == "available" else []
        findings_by_uid = {item["item_uid"]: item for item in items}
        for projected_record in projected:
            fingerprint = str(projected_record["finding_fingerprint"])
            item_uid = str(projected_record["item_uid"])
            resolution = resolution_by_uid[item_uid]
            if projected_record["tier"] == "task-only":
                if not config.emit_process_traces:
                    continue
                emit_records = []
                for record_type in ("process-trace", "task-only"):
                    derived = dict(projected_record)
                    derived.update(record_type=record_type, tier=classify_tier(resolution, record_type=record_type),
                                   outcome_label=None, record_id=record_finding_id(run_id, first.trajectory_id,
                                   first.segment_id, f"{record_type}:{item_uid}"))
                    emit_records.append(derived)
            else:
                emit_records = [projected_record]
            for rec in emit_records:
                evidence = cast(list[Record], rec["evidence"])
                _refuse_posterior_evidence(run_id, fingerprint, evidence, as_of)
                prov = _provenance_for(resolution, run["provenance"])
                rec["profile"], rec["stack"] = prov["profile"], prov["stack"]
                rec["lineage"] = {
                    "hub_commit": hub_commit, "snapshot_id": config.snapshot_id,
                    "source_identity_version": SOURCE_IDENTITY_VERSION,
                    "content_digests": [run_digests[run_id], *sorted(
                        member["record_digest"] for member in frozen.snapshot["observations"]
                        if any(obs["observation_id"] == member["identity"] and obs["run_id"] == run_id
                               for obs in frozen.eligible_observations))],
                    "labeler_policy_version": config.labeler_policy_version,
                    "reply_classifier_version": config.reply_classifier_version,
                    "rubric_schema_version": config.rubric_schema_version,
                    "as_of": as_of, "valid_at": _max_valid_at(evidence, as_of),
                    "split": assign_split(str(rec["record_id"]), holdout_rate=config.holdout_rate,
                                          val_rate=config.val_rate, salt=config.salt),
                    "exclusion_reason": None, "repo_slug": repo_slugs[run_id],
                }
                finding = findings_by_uid.get(item_uid, {})
                text = finding.get("body")
                if isinstance(text, str):
                    rec["finding_text"] = text
                    rec["finding_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                revision = task.get("analyzed_revision", {})
                task_identity: dict[str, Any] = {"repo_slug": repo_slugs[run_id]}
                for name, value in (("base_sha", revision.get("pr_base_sha") or revision.get("merge_base_sha")),
                                    ("head_sha", revision.get("head_sha"))):
                    if value:
                        task_identity[name] = value
                if task.get("diff"):
                    rec["diff"] = task["diff"]
                    diff_digest = task["diff_sha256"]
                    diff_ref = {"run_id": run_id, "record_digest": run_digests[run_id], "section": "original_task.diff"}
                    task_identity.update(diff_digest=diff_digest, diff_ref=diff_ref)
                    cast(dict[str, Any], rec["lineage"]).update(diff_digest=diff_digest, diff_ref=diff_ref)
                rec["task_identity"] = task_identity
                rec["schema_version"] = "3"
                scoring = run["scoring"]
                if scoring["status"] == "available":
                    rec["reward_version"] = scoring["value"]["persisted_breakdown"]["reward_version"]
                    rec["intrinsic_reward"] = scoring["value"]["persisted_breakdown"]
                    rec["verification"] = run["verification"]
                    rec["scoring"] = scoring
                rec["has_posterior"] = bool(evidence)
                rec["decisive_only"] = rec["disposition"] in {"accepted", "rejected"}
                rec["decisive_mix"] = False
                if row.get("annotation") is not None:
                    rec["annotation"] = row["annotation"]
                records.append(rec)

    records.sort(key=lambda r: str(r["record_id"]))
    adjudication.sort(key=lambda r: str(r["fingerprint"]))

    # Cap the deduplicated population after split assignment; account for every exclusion.
    exclusions_by_reason["non-decisive-adjudication"] = 0 if config.emit_process_traces else len(adjudication)
    if config.caps:
        records, tier_exclusions = retain_group_limits(records, lambda record: str(record["tier"]), config.caps)
        exclusions_by_reason.update({f"tier-cap:{tier}": count for tier, count in tier_exclusions.items()})

    share_caps_report: dict[str, Any] | None = None
    if (
        config.max_stack_share is not None
        or config.max_repo_share is not None
        or config.max_profile_share is not None
    ):
        records, share_exclusions = _apply_share_caps(
            records,
            max_stack_share=config.max_stack_share,
            max_repo_share=config.max_repo_share,
            max_profile_share=config.max_profile_share,
        )
        for excl_key, excl_count in share_exclusions.items():
            exclusions_by_reason[f"share-cap:{excl_key}"] = (
                exclusions_by_reason.get(f"share-cap:{excl_key}", 0) + excl_count
            )
        share_caps_report = _share_caps_report(
            records,
            share_exclusions,
            max_stack_share=config.max_stack_share,
            max_repo_share=config.max_repo_share,
            max_profile_share=config.max_profile_share,
        )

    # Aggregate only emitted evidence; capped-out records must not advance lineage valid_at.
    valid_at = _max_valid_at([cast(Record, rec["lineage"]) for rec in records], as_of)

    canonical = _dump_jsonl(records)
    _write_artifact(config.out_dir / "corpus.jsonl", canonical.encode("utf-8"))
    split_counts: dict[str, int] = {}
    for split_name, filename in SPLIT_FILENAMES.items():
        split_records = [r for r in records if cast(dict[str, Any], r["lineage"])["split"] == split_name]
        _write_artifact(config.out_dir / filename, _dump_jsonl(split_records).encode("utf-8"))
        split_counts[split_name] = len(split_records)
    _write_artifact(
        config.out_dir / "adjudication-report.json",
        (json.dumps(adjudication, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    schema_src = Path(__file__).parent.parent / "schema" / "record-schema.json"
    _write_artifact(config.out_dir / "schema.json", schema_src.read_text(encoding="utf-8").encode("utf-8"))

    content_digests = {
        f"{kind}/{member['identity']}": member["record_digest"]
        for kind in ("runs", "observations") for member in frozen.snapshot[kind]
    }
    caps_report = {"configured": dict(sorted(config.caps.items())),
                   "applied": count_by(records, lambda r: str(r["tier"])) if config.caps else {}}
    share_caps_entry = {"share_caps": share_caps_report} if share_caps_report is not None else {}
    lineage = {
        "schema_version": "lineage",
        "source_identity_version": SOURCE_IDENTITY_VERSION,
        "snapshot_id": config.snapshot_id,
        "snapshot": {key: value for key, value in frozen.snapshot.items() if key != "diagnostics"},
        "hub_commit": hub_commit,
        "source_hub_commit": hub_commit,
        "source_repository": source.get("repository"),
        "content_digests": content_digests,
        "labeler_policy_version": config.labeler_policy_version,
        "reply_classifier_version": config.reply_classifier_version,
        "rubric_schema_version": config.rubric_schema_version,
        "as_of": as_of,
        "valid_at": valid_at,
        "salt": config.salt,
        "holdout_rate": config.holdout_rate,
        "val_rate": config.val_rate,
        "trajectory_set_hash": _trajectory_set_hash(
            sorted({str(r["session_id"]) for r in records})
        ),
        "split_counts": split_counts,
        "exclusions_by_reason": dict(sorted(exclusions_by_reason.items())),
        "caps": caps_report,
        **share_caps_entry,
        "adjudication_count": len(adjudication),
        "exclusion_list_digest": hashlib.sha256(EXCLUSION_PATH.read_bytes()).hexdigest(),
    }
    _write_artifact(
        config.out_dir / "lineage.json",
        (json.dumps(lineage, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    output_names = ("corpus.jsonl", *SPLIT_FILENAMES.values(), "adjudication-report.json", "schema.json",
                    "lineage.json")
    sums = "".join(f"{hashlib.sha256((config.out_dir / name).read_bytes()).hexdigest()}  {name}\n"
                   for name in sorted(output_names))
    _write_artifact(config.out_dir / "SHA256SUMS", sums.encode("utf-8"))

    # Consumers require _SUCCESS; write it only after every artifact succeeds.
    _write_artifact(config.out_dir / "_SUCCESS", b"ok\n")

    return {
        "total": len(records),
        "emitted": len(records),
        "adjudication": len(adjudication),
        **{f"split_{name}": count for name, count in split_counts.items()},
        "records_by_type": count_by(records, lambda r: str(r["record_type"])),
        "records_by_tier": count_by(records, lambda r: str(r["tier"])),
        "records_by_split": dict(split_counts),
        "caps": caps_report,
        **share_caps_entry,
        "exclusions_by_reason": dict(sorted(exclusions_by_reason.items())),
    }
