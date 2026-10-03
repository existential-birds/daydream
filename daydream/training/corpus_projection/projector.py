"""Project each finding independently from frozen curation and annotation bundles.

Mixed outcomes retain distinct finding labels. Non-decisive findings feed the
adjudication report and, when requested, separate process-trace/task-only
records. Segment, project, assign content-derived splits, enforce evidence
gates, and write pinned manifests without Git or network access.
"""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, cast, overload

from daydream.archive.index import normalize_as_of
from daydream.archive.sanitize import _derivative_digest
from daydream.json_utils import atomic_write_bytes, canonical_json
from daydream.training.corpus import _is_posterior_leak, _trajectory_set_hash
from daydream.training.corpus_projection.bundle import (
    CuratedBundle,
    _verify_sha256sums,
    load_curated_bundle,
)
from daydream.training.corpus_projection.identity import record_id
from daydream.training.corpus_projection.license import load_license_policy, resolve_repo_decision
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
from daydream.training.corpus_projection.tiers import Tier, classify_tier
from daydream.training.exclusion import EXCLUSION_PATH

__all__ = [
    "BatchArtifacts",
    "BuildFrozenCorpusConfig",
    "project_findings",
    "read_batch_artifacts",
    "build_frozen_corpus",
]

@dataclass(frozen=True)
class BatchArtifacts:
    """Optional batch review artifacts, joined to resolutions by fingerprint.

    Findings supply body text; diff.patch supplies task context. Manifest git
    supplies head_sha, while code_context supplies base_sha and a fallback head."""

    findings_by_fingerprint: dict[str, str]
    diff: str
    manifest_git: dict[str, Any]
    manifest_code_context: dict[str, Any]


def read_batch_artifacts(bundle_dir: Path, session_id: str) -> BatchArtifacts:
    """Read the finding text / diff / git shas from a batch directory.

    Pure filesystem read; every file is optional (an absent artifact yields an
    empty value — the caller decides what is mandatory). Findings without both
    a ``fingerprint`` and a ``body`` are skipped.
    """
    batch_dir = Path(bundle_dir) / "batches" / session_id
    findings_by_fingerprint: dict[str, str] = {}
    try:
        data = json.loads((batch_dir / "findings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    for finding in (data.get("findings") if isinstance(data, dict) else None) or []:
        if not isinstance(finding, dict):
            continue
        fingerprint, body = finding.get("fingerprint"), finding.get("body")
        if isinstance(fingerprint, str) and isinstance(body, str):
            findings_by_fingerprint[fingerprint] = body
    try:
        diff = (batch_dir / "diff.patch").read_text(encoding="utf-8")
    except (OSError, ValueError):
        diff = ""
    try:
        manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    manifest_git = manifest.get("git") if isinstance(manifest, dict) else None
    manifest_code_context = (
        manifest.get("code_context") if isinstance(manifest, dict) else None
    )
    return BatchArtifacts(
        findings_by_fingerprint=findings_by_fingerprint,
        diff=diff,
        manifest_git=manifest_git if isinstance(manifest_git, dict) else {},
        manifest_code_context=(
            manifest_code_context if isinstance(manifest_code_context, dict) else {}
        ),
    )


def _dump_jsonl(records: list[Record]) -> str:
    """Canonical JSONL: sorted keys, compact separators, record order fixed
    by the caller — byte-for-byte stable across re-runs."""
    return "".join(
        canonical_json(r) + "\n"
        for r in records
    )


def _write_artifact(path: Path, data: bytes) -> None:
    """Write one projection artifact, skipping fsync and mode.

    Projection outputs are re-derivable from the bundle, so the durability
    knobs ``atomic_write_bytes`` offers by default are deliberately omitted.
    """
    atomic_write_bytes(path, data, fsync=False, dir_fsync=False, mode=None)


@dataclass(frozen=True)
class BuildFrozenCorpusConfig:
    """Configuration for the frozen projection (pure — no git, no network)."""

    out_dir: Path
    bundle_dir: Path
    # None produces a field-specific ValueError in __post_init__, including omitted args.
    annotation_bundle_dir: Path | None = None
    as_of: str | None = None
    holdout_rate: float = 0.1
    val_rate: float = 0.1
    salt: str = "daydream-projection-salt"
    caps: dict[str, int] = field(default_factory=dict)
    max_stack_share: float | None = None
    max_repo_share: float | None = None
    max_profile_share: float | None = None
    labeler_policy_version: str = "1"
    reply_classifier_version: str = "1"
    rubric_schema_version: str = "per-finding-resolutions-v1"
    license_policy_path: Path | None = None
    allow_copyleft: frozenset[str] = frozenset()
    # Non-decisive findings always enter the adjudication report. Opt in to additional
    # silver process-trace and task-only records with no outcome label, replacing their
    # non-decisive-adjudication exclusion count.
    emit_process_traces: bool = False

    def __post_init__(self) -> None:
        for name, label in (("annotation_bundle_dir", "annotation bundle"), ("license_policy_path", "license policy")):
            value = getattr(self, name)
            if value is None:
                raise ValueError(
                    f"{name} is required: a projection build without a "
                    f"pinned {label} is a configuration error"
                )
            object.__setattr__(self, name, Path(value))
        for field_name in ("max_stack_share", "max_repo_share", "max_profile_share"):
            share = getattr(self, field_name)
            if share is not None and not (0.0 < share <= 1.0):
                raise ValueError(f"{field_name} must be in (0.0, 1.0], got {share}")
        if self.as_of is not None:
            object.__setattr__(self, "as_of", normalize_as_of(self.as_of))


def _load_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    """Index canonical annotations.jsonl by record_id and fingerprint.

    Require fingerprints and reject duplicate values of either key.
    """
    rows: dict[str, dict[str, Any]] = {}
    seen_fingerprints: dict[str, int] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"annotations snapshot {path}: line {line_no} is not valid JSON: {exc}"
            ) from exc
        record_id_value = row.get("record_id")
        if not record_id_value:
            raise ValueError(
                f"annotations snapshot {path}: line {line_no} missing 'record_id'"
            )
        fingerprint = row.get("fingerprint")
        if not fingerprint:
            raise ValueError(f"annotations snapshot {path}: line {line_no} missing 'fingerprint'")
        record_id_value = str(record_id_value)
        if record_id_value in rows:
            raise ValueError(
                f"annotations snapshot {path}: duplicate record_id {record_id_value!r} "
                f"(lines {line_no} and earlier) — snapshot must be keyed by record_id"
            )
        fp = str(fingerprint)
        if fp in seen_fingerprints:
            raise ValueError(
                f"annotations snapshot {path}: duplicate fingerprint {fp!r} "
                f"(lines {seen_fingerprints[fp]} and {line_no}) — snapshot must be "
                "keyed by fingerprint"
            )
        seen_fingerprints[fp] = line_no
        rows[record_id_value] = row
    return rows


def _refuse_posterior_evidence(
    session_id: str, fingerprint: str, evidence: list[Record], as_of: str | None
) -> None:
    """Abort the entire build if any evidence is chronologically after as_of."""
    if as_of is None:
        return
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        if _is_posterior_leak(dict(item), as_of):
            raise ValueError(
                f"session {session_id!r} finding {fingerprint!r}: evidence "
                f"valid_at {item.get('valid_at')!r} is after as_of {as_of!r} "
                "— refusing posterior outcome evidence"
            )


def _provenance_for(
    resolution_row: Mapping[str, Any], manifest_row: Mapping[str, Any]
) -> dict[str, Any]:
    """Prefer resolution provenance (flat or nested); fall back to the batch manifest."""
    prov = extract_provenance(resolution_row)
    if (
        not any(prov["profile"].values())
        and prov.get("skill") is None
        and prov["stack"] is None
    ):
        return extract_provenance(manifest_row)
    return prov


def _max_valid_at(evidence: list[Record], base: str | None) -> str | None:
    """Latest evidence time, bounded below by base; parse ISO offsets chronologically."""
    result = base
    for item in evidence:
        if isinstance(item, Mapping) and item.get("valid_at"):
            candidate = str(item["valid_at"])
            if result is None or datetime.fromisoformat(candidate) > datetime.fromisoformat(result):
                result = candidate
    return result


_ANNOTATION_SCHEMA_PREFIX = "annotation-snapshot/"


def _verify_annotation_bundle(
    annotation_bundle_dir: Path, bundle: CuratedBundle, bundle_dir: Path
) -> dict[str, Any]:
    """Verify the annotation bundle before linking it to the curation bundle.

    Require _SUCCESS, validate its own SHA256SUMS, then match curation_id,
    sanitized_hub_commit, and the actual batch/file-set digest. Require compatible
    snapshot schema and labeler/rubric/classifier versions. Empty as_of is an
    allowed unpinned edge reported in lineage. Failures raise ValueError (including
    BundleError) naming the mismatched fields/values.
    """
    root = Path(annotation_bundle_dir)
    if not (root / "_SUCCESS").is_file():
        raise ValueError(f"annotation bundle {root}: missing _SUCCESS marker")
    _verify_sha256sums(root, "")
    lineage_path = root / "lineage.json"
    if not lineage_path.is_file():
        raise ValueError(f"annotation bundle {root}: missing lineage.json")
    try:
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"annotation bundle {root}: lineage.json is not valid JSON: {exc}") from exc
    if not isinstance(lineage, dict):
        raise ValueError(f"annotation bundle {root}: lineage.json is not a JSON object")

    def _gate(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"annotation bundle {root}: {message}")

    recorded_curation = lineage.get("curation_id")
    _gate(
        recorded_curation == bundle.curation_id,
        f"curation_id mismatch: annotation bundle records {recorded_curation!r} "
        f"but the curation bundle is {bundle.curation_id!r}",
    )
    recorded_commit = lineage.get("sanitized_hub_commit")
    _gate(
        recorded_commit == bundle.source_hub_commit,
        f"sanitized_hub_commit mismatch: annotation bundle records {recorded_commit!r} "
        f"but the curation bundle is {bundle.source_hub_commit!r}",
    )
    schema_version = lineage.get("schema_version")
    _gate(
        isinstance(schema_version, str)
        and schema_version.startswith(_ANNOTATION_SCHEMA_PREFIX),
        f"incompatible schema_version {schema_version!r} — expected an "
        f"{_ANNOTATION_SCHEMA_PREFIX!r}* annotation snapshot",
    )
    batch_fileset_digest = lineage.get("batch_fileset_digest")
    _gate(
        isinstance(batch_fileset_digest, str) and bool(batch_fileset_digest),
        "missing 'batch_fileset_digest' — the annotation bundle must record "
        "the curation bundle's batch/file-set digest it was harvested against",
    )
    actual_fileset_digest = _derivative_digest(bundle_dir)
    _gate(
        batch_fileset_digest == actual_fileset_digest,
        f"stale batch fileset: the annotation bundle records "
        f"batch_fileset_digest {batch_fileset_digest} but the curation bundle "
        f"now hashes to {actual_fileset_digest} — re-harvest the annotation "
        "snapshot against this curation bundle before building",
    )
    for version_field in ("labeler_version", "rubric_version", "classifier_version"):
        value = lineage.get(version_field)
        _gate(
            isinstance(value, str) and bool(value),
            f"missing or empty {version_field!r} — labeler/rubric/classifier "
            "versions must be recorded",
        )
    _gate(
        "as_of" in lineage,
        "missing 'as_of' — the evidence pin must be recorded (empty/null "
        "allowed for the unpinned edge)",
    )
    return lineage


def _read_trajectory_documents(bundle_dir: Path, artifact_relpath: str) -> list[dict[str, Any]]:
    """Read a directory batch's trajectory.json object, or a file artifact as JSONL."""
    artifact = bundle_dir / artifact_relpath
    if artifact.is_dir():
        artifact = artifact / "trajectory.json"
        if not artifact.is_file():
            raise ValueError(
                f"bundle {bundle_dir}: {artifact_relpath} contains no trajectory.json"
            )
    raw = artifact.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        documents: list[dict[str, Any]] = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"bundle {bundle_dir}: {artifact_relpath} line {line_no} "
                    f"is not valid JSON: {exc}"
                ) from exc
        return documents
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [doc for doc in parsed if isinstance(doc, dict)]
    raise ValueError(f"bundle {bundle_dir}: {artifact_relpath} is not a trajectory object")


def finding_resolutions(
    session: Mapping[str, object],
) -> Iterator[tuple[Mapping[str, Any], Tier]]:
    """Enumerate and validate source findings once for projection and adjudication."""
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
        yield resolution, classify_tier(resolution)


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

    records: list[Record] = []
    adjudication: list[Record] = []
    for resolution, tier in finding_resolutions(session):
        fingerprint = resolution["fingerprint"]
        disposition = resolution.get("disposition")
        evidence = list(resolution.get("evidence") or [])
        provenance = extract_provenance(resolution)
        record = {
            "record_id": record_id(
                str(session_id), str(trajectory_id), str(segment_id), str(fingerprint)
            ),
            "record_type": "outcome-finding",
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "task_segment": segment_id,
            "finding_fingerprint": fingerprint,
            "tier": tier,
            "disposition": disposition,
            "outcome_label": disposition if tier == "gold" else None,
            "evidence": evidence,
            "profile": provenance["profile"],
            "stack": provenance["stack"],
        }
        records.append(record)
        if tier == "task-only":
            adjudication.append(
                {
                    "fingerprint": fingerprint,
                    "disposition": disposition,
                    "evidence": evidence,
                    "exclusion_reason": (
                        f"non-decisive disposition {disposition!r} — missing decisive "
                        "human verdict (evidence carried for the adjudication pass)"
                    ),
                }
            )

    if return_adjudication:
        return records, adjudication
    return records


def _license_decision_distribution(
    decisions: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Admitted/rejected decision counts across admitted batches, keyed by
    ``admitted`` or the rejection reason code — deterministic order."""
    return count_by(
        list(decisions.values()),
        lambda d: "admitted" if d["status"] == "admitted" else str(d["reason_code"]),
    )


def build_frozen_corpus(config: BuildFrozenCorpusConfig) -> dict[str, Any]:
    """Project verified curation and annotation bundles without Git/network access.

    Recheck license decisions before any output. Validate every trajectory
    segment; each session's findings belong to its first segment in fork order.
    Non-decisive findings go to the adjudication report, optionally also to
    process-trace/task-only records with no outcome label. Reject evidence after
    as_of; assign content-derived splits, then apply tier and output-share caps.

    Write corpus, splits, schema, reports, and lineage atomically per file, with
    _SUCCESS last. All bytes and returned counts derive from frozen inputs and
    the final emitted population; no wall-clock timestamp enters lineage."""
    bundle = load_curated_bundle(config.bundle_dir)
    assert config.annotation_bundle_dir is not None  # __post_init__ guarantees
    # Recheck each admitted batch against the pinned license policy before any write.
    # __post_init__ guarantees the policy path; assert narrows it for mypy.
    assert config.license_policy_path is not None
    policy, policy_digest = load_license_policy(config.license_policy_path)
    decisions: dict[str, dict[str, Any]] = {}
    license_refusals: list[tuple[str, str]] = []
    for batch in bundle.admitted:
        repo_decision = resolve_repo_decision(
            batch.repo_slug or "", batch.license_evidence, policy, config.allow_copyleft
        )
        decisions[batch.session_id] = asdict(repo_decision)
        # Any rejection aborts the whole projection, including unopted copyleft or
        # missing identity/evidence; partial admission must not produce training data.
        if repo_decision.status != "admitted":
            license_refusals.append((batch.session_id, repo_decision.reason_code or ""))
    if license_refusals:
        raise ValueError(
            "license gate: refusing to project — non-admitted repo license "
            "decisions reached the projection boundary as (session_id, "
            f"reason_code) pairs {sorted(license_refusals)}; no output written"
        )
    annotation_lineage = _verify_annotation_bundle(
        config.annotation_bundle_dir, bundle, config.bundle_dir
    )
    snapshot_path = Path(config.annotation_bundle_dir) / "annotations.jsonl"
    snapshot_rows = _load_snapshot(snapshot_path)
    snapshot_digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

    records: list[Record] = []
    adjudication: list[Record] = []
    # A session's resolutions belong to its first segment, preserving fork order.
    projected_sessions: set[str] = set()
    exclusions_by_reason: dict[str, int] = {}
    for batch in bundle.admitted:
        if batch.manifest_relpath is not None:
            manifest_path = config.bundle_dir / batch.manifest_relpath
            try:
                manifest_row_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"bundle {config.bundle_dir}: {batch.manifest_relpath} "
                    f"is not valid JSON: {exc}"
                ) from exc
            batch_manifest_row = manifest_row_raw if isinstance(manifest_row_raw, dict) else {}
        else:
            batch_manifest_row = {}
        digest_parts = [d for d in (batch.content_digest, snapshot_digest) if d]
        batch_artifacts = read_batch_artifacts(config.bundle_dir, batch.session_id)
        trajectory_documents = _read_trajectory_documents(config.bundle_dir, batch.artifact_relpath)
        for trajectory in trajectory_documents:
            segs = segment(trajectory)
            for seg in segs:
                resolutions = [
                    row
                    for row in snapshot_rows.values()
                    if row.get("session_id") == seg.session_id
                ]
                if not resolutions:
                    continue
                session_view: dict[str, Any] = {
                    "session_id": seg.session_id,
                    "trajectory_id": seg.trajectory_id,
                    "segment_id": seg.segment_id,
                    "resolutions": resolutions,
                }
                seg_records, seg_adjudication = project_findings(
                    session_view, return_adjudication=True
                )
                # Still validate every segment, including duplicate session views.
                if seg.session_id in projected_sessions:
                    continue
                projected_sessions.add(seg.session_id)
                resolution_by_fp = {
                    str(row.get("fingerprint")): row for row in resolutions
                }
                for rec in seg_records:
                    fingerprint = str(rec["finding_fingerprint"])
                    decision = decisions.get(seg.session_id)
                    if decision is None:
                        raise ValueError(
                            f"session {seg.session_id}: no recorded license decision "
                            "found at projection; refusing to project"
                        )
                    record_decision: dict[str, Any] = decision
                    # Non-decisive findings enter training only as opt-in, schema-distinct
                    # process-trace/task-only records. Their adjudication report stays intact.
                    if str(rec["tier"]) == "task-only":
                        if not config.emit_process_traces:
                            continue
                        resolution = resolution_by_fp.get(fingerprint, {})
                        emit_records: list[Record] = []
                        for derived_type in ("process-trace", "task-only"):
                            derived = dict(rec)
                            derived["record_type"] = derived_type
                            derived["tier"] = classify_tier(
                                resolution, record_type=derived_type
                            )
                            derived["outcome_label"] = None
                            derived["record_id"] = record_id(
                                seg.session_id,
                                seg.trajectory_id,
                                seg.segment_id,
                                f"{derived_type}:{fingerprint}",
                            )
                            emit_records.append(derived)
                    else:
                        emit_records = [rec]
                    for rec in emit_records:
                        _refuse_posterior_evidence(
                            seg.session_id,
                            fingerprint,
                            cast(list[Record], rec["evidence"]),
                            config.as_of,
                        )
                        split = assign_split(
                            str(rec["record_id"]),
                            holdout_rate=config.holdout_rate,
                            val_rate=config.val_rate,
                            salt=config.salt,
                        )
                        prov = _provenance_for(
                            resolution_by_fp.get(fingerprint, {}), batch_manifest_row
                        )
                        rec_valid_at = _max_valid_at(
                            cast(list[Record], rec["evidence"]), config.as_of
                        )
                        rec["profile"] = prov["profile"]
                        rec["stack"] = prov["stack"]
                        rec["lineage"] = {
                            "hub_commit": bundle.source_hub_commit,
                            "curation_id": bundle.curation_id,
                            "content_digests": digest_parts,
                            "labeler_policy_version": config.labeler_policy_version,
                            "reply_classifier_version": config.reply_classifier_version,
                            "rubric_schema_version": config.rubric_schema_version,
                            "as_of": config.as_of,
                            "valid_at": rec_valid_at,
                            "split": split,
                            "exclusion_reason": None,
                            "repo_slug": record_decision["repo_slug"],
                            "license_decision": record_decision,
                        }
                        # Enrich only present artifacts; consumers enforce required inputs.
                        # Finding identity, tier, and outcome label remain unchanged.
                        finding_text = batch_artifacts.findings_by_fingerprint.get(fingerprint)
                        if finding_text is not None:
                            rec["finding_text"] = finding_text
                            rec["finding_text_sha256"] = hashlib.sha256(
                                finding_text.encode("utf-8")
                            ).hexdigest()
                        task_identity: dict[str, Any] = {
                            "repo_slug": record_decision["repo_slug"],
                        }
                        manifest_git = batch_artifacts.manifest_git
                        manifest_code_context = batch_artifacts.manifest_code_context
                        # Base lives in code_context; git supplies the preferred head.
                        for sha_name, sha_value in (
                            ("base_sha", manifest_code_context.get("base_sha")),
                            (
                                "head_sha",
                                manifest_git.get("head_sha")
                                or manifest_code_context.get("head_sha"),
                            ),
                        ):
                            if isinstance(sha_value, str) and bool(sha_value):
                                task_identity[sha_name] = sha_value
                        if batch_artifacts.diff:
                            diff_digest = hashlib.sha256(
                                batch_artifacts.diff.encode("utf-8")
                            ).hexdigest()
                            diff_ref: dict[str, Any] = {
                                "batch": batch.content_digest,
                                "relpath": f"batches/{batch.session_id}/diff.patch",
                            }
                            task_identity["diff_digest"] = diff_digest
                            task_identity["diff_ref"] = diff_ref
                            # Freeze the diff into the record for Stage-2 RFT.
                            rec["diff"] = batch_artifacts.diff
                        rec["task_identity"] = task_identity
                        if "diff_digest" in task_identity:
                            lineage_fields = cast(dict[str, Any], rec["lineage"])
                            lineage_fields["diff_digest"] = task_identity["diff_digest"]
                            lineage_fields["diff_ref"] = task_identity["diff_ref"]
                        rec["schema_version"] = "2"
                        records.append(rec)
                adjudication.extend(seg_adjudication)

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
    valid_at = _max_valid_at([cast(Record, rec["lineage"]) for rec in records], config.as_of)

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

    content_digests: dict[str, str] = {
        batch.session_id: batch.content_digest for batch in bundle.admitted if batch.content_digest
    }
    content_digests["annotations.jsonl"] = snapshot_digest
    annotation_as_of = annotation_lineage.get("as_of")
    license_distribution = _license_decision_distribution(decisions)
    caps_report = {"configured": dict(sorted(config.caps.items())),
                   "applied": count_by(records, lambda r: str(r["tier"])) if config.caps else {}}
    share_caps_entry = {"share_caps": share_caps_report} if share_caps_report is not None else {}
    copyleft_opt_ins = sorted(config.allow_copyleft)
    lineage = {
        "schema_version": "lineage",
        "curation_id": bundle.curation_id,
        "hub_commit": bundle.source_hub_commit,
        "source_hub_commit": bundle.source_hub_commit,
        "content_digests": content_digests,
        # Pin the annotation bundle without modifying the finalized curation bundle.
        "annotation_bundle": {
            "snapshot_id": annotation_lineage.get("snapshot_id")
            or config.annotation_bundle_dir.name,
            "curation_id": annotation_lineage.get("curation_id"),
            "sanitized_hub_commit": annotation_lineage.get("sanitized_hub_commit"),
            "annotations_digest": snapshot_digest,
            "as_of": annotation_as_of,
            "unpinned_as_of": annotation_as_of in (None, ""),
        },
        "labeler_policy_version": config.labeler_policy_version,
        "reply_classifier_version": config.reply_classifier_version,
        "rubric_schema_version": config.rubric_schema_version,
        "as_of": config.as_of,
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
        # Pin all license inputs and decisions for byte-identical reconstruction.
        "license_policy": {
            "path_digest": policy_digest,
            "policy_version": policy.policy_version,
        },
        "exclusion_list_digest": hashlib.sha256(EXCLUSION_PATH.read_bytes()).hexdigest(),
        "copyleft_opt_ins": copyleft_opt_ins,
        "license_decisions": {
            str(session_id): decision for session_id, decision in decisions.items()
        },
        "license_decision_distribution": license_distribution,
    }
    _write_artifact(
        config.out_dir / "lineage.json",
        (json.dumps(lineage, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    # The human report shares lineage's license evidence and precedes _SUCCESS.
    license_report = {
        "policy": {"policy_version": policy.policy_version, "digest": policy_digest},
        "exclusion_list_digest": lineage["exclusion_list_digest"],
        "copyleft_opt_ins": copyleft_opt_ins,
        "decisions": dict(sorted(
            (str(session_id), decision) for session_id, decision in decisions.items()
        )),
        "distribution": license_distribution,
    }
    _write_artifact(
        config.out_dir / "license-report.json",
        (json.dumps(license_report, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )

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
        "license_distribution": license_distribution,
    }
