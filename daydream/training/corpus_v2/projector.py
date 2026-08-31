"""Per-finding projection for corpus v2 (Req 6, Req 18, D8).

Each ``PerFindingResolution`` becomes its own record with a per-record
``outcome_label`` — a mixed session (some findings accepted, some rejected)
never collapses into a run-level aggregate like v1's ``contested``
``outcome_label``. Non-decisive dispositions route to an adjudication
report (report output, not a pipeline stage); the projector stays pure and
deterministic.

``run_build_corpus_v2()`` is the top-level pure projection (no git, no
network): load bundle → segment → project findings → assign frozen
content-derived splits → refuse posterior evidence → write split manifests
atomically with a lineage pin.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, cast, overload

from daydream.archive.index import normalize_as_of
from daydream.training.corpus import _is_posterior_leak, _trajectory_set_hash
from daydream.training.corpus_v2.bundle import (
    CuratedBundle,
    _verify_sha256sums,
    load_curated_bundle,
)
from daydream.training.corpus_v2.identity import record_id
from daydream.training.corpus_v2.provenance import extract_provenance
from daydream.training.corpus_v2.segments import segment
from daydream.training.corpus_v2.splits import assign_split
from daydream.training.corpus_v2.tiers import classify_tier

__all__ = ["BuildCorpusV2Config", "project_findings", "run_build_corpus_v2"]

Record = dict[str, object]

_SPLIT_FILENAMES: dict[str, str] = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "holdout": "holdout.jsonl",
}


def _dump_jsonl(records: list[Record]) -> str:
    """Canonical JSONL: sorted keys, compact separators, record order fixed
    by the caller — byte-for-byte stable across re-runs."""
    return "".join(
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for r in records
    )


def _atomic_write(path: Path, content: str) -> None:
    """Tempfile-in-same-dir + ``Path.replace`` (mirrors ``corpus.py``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        Path(tmp_name).replace(path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class BuildCorpusV2Config:
    """Configuration for the corpus v2 projection (pure — no git, no network)."""

    out_dir: Path
    bundle_dir: Path
    # Required: a build without a pinned annotation bundle is a config error
    # (raised as ValueError naming the field, not a TypeError from the
    # dataclass) — declared optional so that misconfiguration, not a missing
    # kwarg, is what callers see.
    annotation_bundle_dir: Path | None = None
    as_of: str | None = None
    holdout_rate: float = 0.1
    val_rate: float = 0.1
    salt: str = "daydream-corpus-v2"
    caps: dict[str, int] = field(default_factory=dict)
    labeler_policy_version: str = "1"
    reply_classifier_version: str = "1"
    rubric_schema_version: str = "per-finding-resolutions-v1"
    license: str | None = None

    def __post_init__(self) -> None:
        if self.annotation_bundle_dir is None:
            raise ValueError(
                "annotation_bundle_dir is required: a corpus v2 build without a "
                "pinned annotation bundle is a configuration error"
            )
        object.__setattr__(self, "annotation_bundle_dir", Path(self.annotation_bundle_dir))
        if self.as_of is not None:
            object.__setattr__(self, "as_of", normalize_as_of(self.as_of))


def _load_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    """Parse the annotation bundle's ``annotations.jsonl`` (canonical
    per-finding record shape): one JSON object per line, keyed by
    ``record_id``. ``fingerprint`` is required on every row and duplicates
    of either key fail closed — the canonical record is identified by
    ``record_id`` and located by ``fingerprint``, so neither may be ambiguous."""
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
    """Refusal, not drop: any evidence item whose ``valid_at`` lands after
    the ``as_of`` pin aborts the whole build (spec: "refused"). Reuses v1's
    chronological ``_is_posterior_leak`` comparison (Pattern Q)."""
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


def _merge_nested_profile(prov: dict[str, Any], row: Mapping[str, Any]) -> None:
    """Surface the canonical record's nested ``profile`` block (Req 8).

    ``adjudication.snapshot.build_canonical_record`` emits the four
    review-profile fields nested under top-level ``profile`` (with ``stack``
    — not ``profile_*`` — at top level), so a top-level read comes back
    empty and the nested block must supply the values instead — they are
    never dropped at the projection boundary.
    """
    if any(prov["profile"].values()):
        return
    nested = row.get("profile")
    if isinstance(nested, Mapping):
        for field in prov["profile"]:
            value = nested.get(field)
            if value is not None:
                prov["profile"][field] = value


def _provenance_for(
    resolution_row: Mapping[str, Any], manifest_row: Mapping[str, Any]
) -> dict[str, Any]:
    """Profile/stack provenance for one record (Req 8).

    The resolution row's native review-profile fields win when present;
    otherwise the batch manifest row supplies them — the values carried by
    either row type must not be dropped at the projection boundary. Canonical
    annotation records nest the profile block, so both shapes are read.
    """
    prov = extract_provenance(resolution_row)
    _merge_nested_profile(prov, resolution_row)
    if (
        not any(prov["profile"].values())
        and prov.get("skill") is None
        and prov["stack"] is None
    ):
        return extract_provenance(manifest_row)
    return prov


def _max_valid_at(evidence: list[Record], base: str | None) -> str | None:
    """Max ``valid_at`` over evidence items (chronological ISO-8601 compare,
    mirroring ``_is_posterior_leak``'s parse), never lower than the ``as_of``
    pin. Parsing both sides with :func:`datetime.fromisoformat` means spelling
    differences — ``Z`` vs ``+00:00`` — can never mis-order the max."""
    result = base
    for item in evidence:
        if isinstance(item, Mapping) and item.get("valid_at"):
            candidate = str(item["valid_at"])
            if result is None or datetime.fromisoformat(candidate) > datetime.fromisoformat(result):
                result = candidate
    return result


_ANNOTATION_SCHEMA_PREFIX = "annotation-snapshot/"


def _verify_annotation_bundle(
    annotation_bundle_dir: Path, bundle: CuratedBundle
) -> dict[str, Any]:
    """Two-bundle contract: the annotation bundle is verified as its own
    published artifact and then linked to the curation bundle exactly.

    Gates, fail-closed in order:

    1. ``_SUCCESS`` completeness marker (a mid-write bundle is refused).
    2. Every file against the bundle's own ``SHA256SUMS`` (self-verification
       via ``bundle._verify_sha256sums`` — raises ``BundleError``, a
       ``ValueError``).
    3. Cross-bundle linkage against the curation bundle: ``curation_id``
       equal to ``bundle.curation_id``, ``sanitized_hub_commit`` equal to
       ``bundle.source_hub_commit``, a recorded batch/file-set digest, a
       compatible annotation-snapshot schema version, and
       labeler/rubric/classifier versions plus ``as_of`` present. An empty
       ``as_of`` is allowed (the unpinned edge) and reported through the
       build lineage, not refused.

    Any mismatch raises ``ValueError`` naming both sides of the mismatch.
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
    """Read a batch's ATIF trajectory document(s).

    The producer writes each batch as a directory (``batches/<sid>/``) whose
    root trajectory lives at ``trajectory.json`` inside it; a single-object
    JSON read yields that one trajectory. A file artifact keeps the JSONL
    shape — one trajectory object per line.
    """
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
        disposition = resolution.get("disposition")
        evidence = list(resolution.get("evidence") or [])
        provenance = extract_provenance(resolution)
        _merge_nested_profile(provenance, resolution)
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


def _count_by(records: list[Record], key: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        k = key(r)
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


def _caps_applied(records: list[Record], caps: dict[str, int]) -> dict[str, int]:
    """Final per-tier population (what the caps resolved to), deterministic."""
    return _count_by(records, lambda r: str(r["tier"])) if caps else {}


def run_build_corpus_v2(config: BuildCorpusV2Config) -> dict[str, Any]:
    """Top-level corpus v2 projection (mirrors ``corpus.py:985``'s pipeline
    contract). Pure — no git, no network; ``base_sha``/hub commit come from
    the curation manifest only.

    Pipeline: load the curated bundle (fail-closed) → segment each admitted
    batch's trajectory (fork-order per-agent) → project per-finding records
    → assign frozen content-derived splits → refuse any record whose
    annotation evidence carries ``valid_at > as_of`` (raise ``ValueError``
    naming the session and both timestamps — refusal, not drop) → write
    ``corpus.jsonl`` plus ``train.jsonl``/``validation.jsonl``/``holdout.jsonl``
    atomically, copy ``schema/v2.json`` alongside, and write ``lineage.json``
    pinning the annotation bundle's provenance (no wall-clock timestamps — every
    manifest byte is a function of the immutable inputs, so re-runs are
    byte-for-byte identical), finishing with a ``_SUCCESS`` completeness
    marker.

    Each finding is projected exactly once (the annotation resolutions are
    session-scoped, so sibling segments never fabricate per-segment copies);
    non-decisive findings are excluded here and routed to the adjudication
    report only.

    Returns a summary dict with ``total``/``emitted``, per-type/per-tier/
    per-split counts, the ``caps`` block (configured vs applied), and
    ``exclusions_by_reason`` — all derived from the final population in
    deterministic order. Non-decisive findings land in the human
    ``adjudication-report.json`` with their evidence (D8: report output, not
    a pipeline stage). ``corpus.jsonl`` is also published as the versioned
    twin ``corpus-v2.jsonl`` from the same in-memory bytes via the same
    atomic write, before ``_SUCCESS`` — covered by the identical fail-closed
    completeness gate, so the twin can never diverge from the canonical file.
    """
    bundle = load_curated_bundle(config.bundle_dir)
    assert config.annotation_bundle_dir is not None  # __post_init__ guarantees
    annotation_lineage = _verify_annotation_bundle(config.annotation_bundle_dir, bundle)
    snapshot_path = Path(config.annotation_bundle_dir) / "annotations.jsonl"
    snapshot_rows = _load_snapshot(snapshot_path)
    snapshot_digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

    records: list[Record] = []
    adjudication: list[Record] = []
    # The annotation resolutions are session-scoped (keyed by session_id +
    # fingerprint, not by trajectory/segment), so each finding is projected
    # exactly once — sibling segments never fabricate per-segment copies that
    # could hash into different splits (D5 disjointness is per finding).
    seen_findings: set[tuple[str, str]] = set()
    adjudicated_findings: set[tuple[str, str]] = set()
    # lineage valid_at is pinned over the emitted records' evidence only —
    # annotation rows for sessions never admitted/emitted must not drift it.
    valid_at = config.as_of
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
                resolution_by_fp = {
                    str(row.get("fingerprint")): row for row in resolutions
                }
                for rec in seg_records:
                    fingerprint = str(rec["finding_fingerprint"])
                    key = (seg.session_id, fingerprint)
                    if key in seen_findings:
                        continue
                    seen_findings.add(key)
                    # Non-decisive findings are report output only (D8): they
                    # never become training records, so corpus.jsonl and the
                    # split manifests exclude them by construction while
                    # lineage/summary count them under exclusions_by_reason.
                    if str(rec["tier"]) == "task-only":
                        continue
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
                    if rec_valid_at is not None and (
                        valid_at is None
                        or datetime.fromisoformat(rec_valid_at) > datetime.fromisoformat(valid_at)
                    ):
                        valid_at = rec_valid_at
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
                        "license_decision": config.license,
                    }
                    # v2 schema stamp: every projected record carries the
                    # training-record schema version it was emitted under.
                    rec["schema_version"] = "2"
                    records.append(rec)
                for entry in seg_adjudication:
                    key = (seg.session_id, str(entry["fingerprint"]))
                    if key not in adjudicated_findings:
                        adjudicated_findings.add(key)
                        adjudication.append(entry)

    records.sort(key=lambda r: str(r["record_id"]))
    adjudication.sort(key=lambda r: str(r["fingerprint"]))

    # Post-segmentation caps (D6): caps bind after segmentation and split
    # assignment, over the deduplicated final population. Excess records of a
    # capped tier are excluded (never silently dropped) and named in the
    # summary, lineage, and exclusion reasons.
    exclusions_by_reason: dict[str, int] = {"non-decisive-adjudication": len(adjudication)}
    if config.caps:
        kept: list[Record] = []
        per_tier: dict[str, int] = {}
        for rec in records:
            tier = str(rec["tier"])
            limit = config.caps.get(tier)
            if limit is not None and per_tier.get(tier, 0) >= limit:
                exclusions_by_reason[f"tier-cap:{tier}"] = (
                    exclusions_by_reason.get(f"tier-cap:{tier}", 0) + 1
                )
                continue
            per_tier[tier] = per_tier.get(tier, 0) + 1
            kept.append(rec)
        records = kept

    canonical = _dump_jsonl(records)
    _atomic_write(config.out_dir / "corpus.jsonl", canonical)
    # Versioned twin of the canonical corpus (same bytes, atomic, pre-_SUCCESS).
    _atomic_write(config.out_dir / "corpus-v2.jsonl", canonical)
    for split_name, filename in _SPLIT_FILENAMES.items():
        split_records = [r for r in records if cast(dict[str, Any], r["lineage"])["split"] == split_name]
        _atomic_write(config.out_dir / filename, _dump_jsonl(split_records))
    _atomic_write(
        config.out_dir / "adjudication-report.json",
        json.dumps(adjudication, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    )

    schema_src = Path(__file__).parent.parent / "schema" / "v2.json"
    _atomic_write(config.out_dir / "schema.json", schema_src.read_text(encoding="utf-8"))

    split_counts = {name: 0 for name in _SPLIT_FILENAMES}
    for r in records:
        lineage_field = cast(dict[str, Any], r["lineage"])
        split_counts[str(lineage_field["split"])] += 1
    # Every Req-11 field: hub commit, curation id, content digests, policy/
    # classifier/rubric versions, as_of + valid_at pins, split assignment,
    # exclusion reasons, and the C5/license decision. Nothing silently dropped.
    content_digests: dict[str, str] = {
        batch.session_id: batch.content_digest for batch in bundle.admitted if batch.content_digest
    }
    content_digests["annotations.jsonl"] = snapshot_digest
    annotation_as_of = annotation_lineage.get("as_of")
    lineage = {
        "schema_version": "corpus-v2",
        "curation_id": bundle.curation_id,
        "hub_commit": bundle.source_hub_commit,
        "source_hub_commit": bundle.source_hub_commit,
        "content_digests": content_digests,
        # Two-bundle contract (K3): the annotation bundle's snapshot_id/commit
        # are pinned into the projection lineage; the finalized curation
        # bundle itself is never touched.
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
        "license": config.license,
        "salt": config.salt,
        "holdout_rate": config.holdout_rate,
        "val_rate": config.val_rate,
        "trajectory_set_hash": _trajectory_set_hash(
            sorted({str(r["session_id"]) for r in records})
        ),
        "split_assignment": split_counts,
        "split_counts": split_counts,
        "exclusions_by_reason": dict(sorted(exclusions_by_reason.items())),
        "caps": {"configured": dict(sorted(config.caps.items())),
                 "applied": _caps_applied(records, config.caps)},
        "adjudication_count": len(adjudication),
    }
    _atomic_write(
        config.out_dir / "lineage.json",
        json.dumps(lineage, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    )

    # Completeness marker (mirrors the bundle's own ``_SUCCESS`` gate): the
    # projection file set is only consumable once every member is in place,
    # so a mid-write failure never leaves a partial projection behind.
    _atomic_write(config.out_dir / "_SUCCESS", "ok\n")

    return {
        "total": len(records),
        "emitted": len(records),
        "adjudication": len(adjudication),
        **{f"split_{name}": count for name, count in split_counts.items()},
        "records_by_type": _count_by(records, lambda r: str(r["record_type"])),
        "records_by_tier": _count_by(records, lambda r: str(r["tier"])),
        "records_by_split": dict(split_counts),
        "caps": {"configured": dict(sorted(config.caps.items())),
                 "applied": _caps_applied(records, config.caps)},
        "exclusions_by_reason": dict(sorted(exclusions_by_reason.items())),
    }
