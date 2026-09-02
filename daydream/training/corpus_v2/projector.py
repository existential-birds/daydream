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
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, NoReturn, cast, overload

from daydream.archive.index import normalize_as_of
from daydream.archive.sanitize import _derivative_digest
from daydream.training.corpus import _is_posterior_leak, _trajectory_set_hash
from daydream.training.corpus_v2.bundle import (
    CuratedBundle,
    _verify_sha256sums,
    load_curated_bundle,
)
from daydream.training.corpus_v2.identity import record_id
from daydream.training.corpus_v2.license import load_license_policy, resolve_repo_decision
from daydream.training.corpus_v2.provenance import extract_provenance
from daydream.training.corpus_v2.segments import segment
from daydream.training.corpus_v2.splits import assign_split
from daydream.training.corpus_v2.tiers import classify_tier
from daydream.training.exclusion import EXCLUSION_PATH

__all__ = [
    "BatchArtifacts",
    "BuildCorpusV2Config",
    "project_findings",
    "read_batch_artifacts",
    "run_build_corpus_v2",
]

Record = dict[str, object]

_SPLIT_FILENAMES: dict[str, str] = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "holdout": "holdout.jsonl",
}


@dataclass(frozen=True)
class BatchArtifacts:
    """Per-batch review artifacts read from a curated bundle's batch directory
    (``batches/<session_id>/``). Producer-realistic shapes (confirmed by the
    task-0 spike probe, tests/test_corpus_v2_spike_probe.py):

    - ``findings.json`` — ``{"findings": [{"fingerprint": <64-hex>, "body":
      <str>, ...}]}`` (the same artifact ``daydream/archive/__init__.py``
      copies into the run bundle; fingerprints join 1:1 with the annotation
      snapshot resolution rows).
    - ``diff.patch`` — the run's diff text.
    - ``manifest.json`` — ``git.head_sha`` plus ``code_context.{base_sha,
      head_sha}`` (archive/manifest.py serialization).
    """

    findings_by_fingerprint: dict[str, str]
    diff: str
    manifest_git: dict[str, Any]


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
    return BatchArtifacts(
        findings_by_fingerprint=findings_by_fingerprint,
        diff=diff,
        manifest_git=manifest_git if isinstance(manifest_git, dict) else {},
    )


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
    # Output-share caps (issue #1079): true share of the final emitted
    # population, enforced per dimension over the post-tier-cap population.
    max_stack_share: float | None = None
    max_repo_share: float | None = None
    max_profile_share: float | None = None
    labeler_policy_version: str = "1"
    reply_classifier_version: str = "1"
    rubric_schema_version: str = "per-finding-resolutions-v1"
    # Required: a build without a pinned license policy is a config error
    # (raised as ValueError naming the field, mirroring annotation_bundle_dir)
    # — declared optional so that misconfiguration, not a missing kwarg, is
    # what callers see. The per-repo license decision is resolved from this
    # policy against each batch's recorded identity + evidence; there is no
    # global license string to stamp (C5/C8, issue #1080).
    license_policy_path: Path | None = None
    allow_copyleft: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.annotation_bundle_dir is None:
            raise ValueError(
                "annotation_bundle_dir is required: a corpus v2 build without a "
                "pinned annotation bundle is a configuration error"
            )
        object.__setattr__(self, "annotation_bundle_dir", Path(self.annotation_bundle_dir))
        if self.license_policy_path is None:
            raise ValueError(
                "license_policy_path is required: a corpus v2 build without a "
                "pinned license policy is a configuration error"
            )
        object.__setattr__(self, "license_policy_path", Path(self.license_policy_path))
        for field_name in ("max_stack_share", "max_repo_share", "max_profile_share"):
            share = getattr(self, field_name)
            if share is not None and not (0.0 < share <= 1.0):
                raise ValueError(f"{field_name} must be in (0.0, 1.0], got {share}")
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
    annotation_bundle_dir: Path, bundle: CuratedBundle, bundle_dir: Path
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
       ``bundle.source_hub_commit``, a recorded ``batch_fileset_digest`` that
       matches the curation bundle's actual batch/file-set digest
       (``_derivative_digest`` of the bundle root — the same canonical
       file-set vocabulary each batch's ``content_digest`` uses, so a stale
       annotation bundle harvested against an older file set is refused
       rather than passing on curation_id + commit alone), a compatible
       annotation-snapshot schema version, and labeler/rubric/classifier
       versions plus ``as_of`` present. An empty ``as_of`` is allowed (the
       unpinned edge) and reported through the build lineage, not refused.

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


def _license_decision_distribution(
    decisions: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Admitted/rejected decision counts across admitted batches, keyed by
    ``admitted`` or the rejection reason code — deterministic order."""
    distribution: dict[str, int] = {}
    for decision in decisions.values():
        key = (
            "admitted"
            if decision["status"] == "admitted"
            else str(decision["reason_code"])
        )
        distribution[key] = distribution.get(key, 0) + 1
    return dict(sorted(distribution.items()))


_SHARE_DIMENSIONS: tuple[tuple[str, str, Callable[[Record], Any]], ...] = (
    # One dimension table consumed by both the share-cap selection stage and
    # the report builder, so the three vocabulary uses (configured keys,
    # applied keys, exclusion-key prefixes) can never drift apart: every
    # consumer spells the repository dimension ``repo``, matching the
    # ``max_repo_share`` flag.
    ("stack", "max_stack_share", lambda r: r.get("stack")),
    (
        "repo",
        "max_repo_share",
        lambda r: cast(dict[str, Any], r.get("lineage") or {}).get("repo_slug"),
    ),
    (
        "profile",
        "max_profile_share",
        lambda r: cast(dict[str, Any], r.get("profile") or {}).get("profile_name"),
    ),
)


def _max_keep_count(total: int, limit: float) -> int:
    """Largest ``allowed`` with ``allowed / total <= limit`` (float-safety
    guards around the floor; the same arithmetic runs on every pass, so the
    result is deterministic and order-invariant)."""
    allowed = math.floor(limit * total)
    while (allowed + 1) / total <= limit:
        allowed += 1
    while allowed > 0 and allowed / total > limit:
        allowed -= 1
    return allowed


def _raise_share_caps_fail_closed(
    *,
    flag: str,
    limit: float,
    dimension: str,
    population: int | None = None,
    value: Any | None = None,
    values: list[Any] | None = None,
) -> NoReturn:
    """Single home for the fail-closed ``ValueError`` terminal state of the
    share-cap stage (the contract documented in ``_apply_share_caps``): every
    total-population-zero path — the entry degeneracy pre-check, a sole
    remaining value that can no longer be trimmed without emptying the
    population, and a trim round that would empty the whole population —
    funnels through this one raise so the message shape cannot drift between
    sites. ``value`` names the single over-share value (first two paths);
    ``values`` spells out the over-share value list (third path)."""
    if values is not None:
        raise ValueError(
            f"{flag}={limit} would reduce the total population to zero across "
            f"{dimension} values {sorted(str(v) for v in values)} — fail-closed"
        )
    raise ValueError(
        f"{flag}={limit} for {dimension}={value!r} would reduce the total population "
        f"to zero (population {population}) — fail-closed"
    )


def _apply_share_caps(
    records: list[Record],
    *,
    max_stack_share: float | None,
    max_repo_share: float | None,
    max_profile_share: float | None,
) -> tuple[list[Record], dict[str, int]]:
    """Pure share-cap selection over the post-tier-cap population.

    Within a dimension, iterates until every value's share of the current
    population is within its configured limit (strict output-share semantics:
    ``count / total <= limit`` over the final emitted population), keeping the
    lowest ``record_id`` records of any over-share group. The three dimensions
    (stack → repo → profile) are applied sequentially and the whole sequence
    is re-run to a fixed point: a later dimension's exclusions re-shape the
    population and can push an earlier dimension's value back over its limit
    (sequential drift on correlated dimensions — e.g. multi-stack repos,
    per-repo profiles), so every dimension must be re-enforced until a
    complete pass excludes nothing and every configured share is within its
    limit of the final population. Returns the kept list (sorted by
    ``record_id``) and exclusion counts keyed ``"<dimension>:<value>"``
    (``None`` values form their own bucket spelled ``(none)``).

    Fails closed with ``ValueError`` if enforcing a cap would leave a total
    population of zero (the cap cannot keep a single record of the population
    it is asked to cap). An already-empty population (no decisive findings, or
    tier caps that trimmed everything) is returned unchanged — configured caps
    exclude nothing from an empty corpus. A dimension value dropping out
    entirely is fine — only total-population-zero is fatal; a sole remaining
    value that can no longer be trimmed without emptying the population hits
    the same fail-closed terminal state as the entry degeneracy pre-check and
    raises rather than emitting a lone value at 100% share over its cap.
    Never samples; input order does not affect the result.
    """
    limits = {
        "stack": max_stack_share,
        "repo": max_repo_share,
        "profile": max_profile_share,
    }
    dimensions: list[tuple[str, str, Callable[[Record], Any], float]] = []
    for name, flag, getter in _SHARE_DIMENSIONS:
        limit = limits[name]
        if limit is not None:
            dimensions.append((name, flag, getter, limit))
    exclusions: dict[str, int] = {}
    population = sorted(records, key=lambda r: str(r["record_id"]))
    if not population:
        # Empty population (no decisive findings, or tier caps trimmed
        # everything): configured caps exclude nothing — a previously
        # completing zero-record build stays completing.
        return population, exclusions
    original = list(population)
    # Fail-closed degeneracy pre-check against the original input population:
    # a cap that cannot keep a single record of an over-share group covering
    # the whole input would collapse it to zero — refuse before doing any work.
    # (Populations reduced by earlier dimensions or by a fixed-point re-pass
    # are handled by the fail-closed sole-value check in the trim loop below,
    # so this check is invariant and runs once per configured dimension on the
    # entry population.)
    entry_total = len(original)
    for dimension, flag, getter, limit in dimensions:
        entry_counts: dict[Any, int] = {}
        for rec in original:
            value = getter(rec)
            entry_counts[value] = entry_counts.get(value, 0) + 1
        for value, count in entry_counts.items():
            if count / entry_total > limit and count == entry_total:
                if _max_keep_count(entry_total, limit) == 0:
                    _raise_share_caps_fail_closed(
                        flag=flag,
                        limit=limit,
                        dimension=dimension,
                        population=entry_total,
                        value=value,
                    )
    # Fixed-point loop (issue #1079, F1): a later dimension's trimming can
    # remove records of one value and push an earlier dimension's value back
    # over its limit, so a single sequential pass does not guarantee the M4
    # contract. Re-run the full dimension sequence until a complete pass
    # excludes nothing — each pass that excludes reduces the population, so
    # the loop provably terminates, and every configured dimension is then
    # within its limit of the final emitted population.
    while True:
        pass_exclusions = 0
        for dimension, flag, getter, limit in dimensions:
            while True:
                total = len(population)
                counts: dict[Any, int] = {}
                for rec in population:
                    value = getter(rec)
                    counts[value] = counts.get(value, 0) + 1
                over: dict[Any, int] = {}
                for value, count in counts.items():
                    if count / total > limit:
                        # Largest keep-count whose share of the current
                        # population is still <= limit.
                        allowed = _max_keep_count(total, limit)
                        if allowed == 0 and count == total:
                            # Sole remaining value after earlier reductions or
                            # exclusions: trimming it further would empty the
                            # population entirely — the same terminal state as
                            # the entry degeneracy pre-check above, so fail
                            # closed (never emit a lone value above its cap).
                            _raise_share_caps_fail_closed(
                                flag=flag,
                                limit=limit,
                                dimension=dimension,
                                population=total,
                                value=value,
                            )
                        over[value] = allowed
                if not over:
                    break
                kept: list[Record] = []
                taken: dict[Any, int] = {}
                round_exclusions = 0
                for rec in population:
                    value = getter(rec)
                    if value in over and taken.get(value, 0) >= over[value]:
                        key = f"{dimension}:{str(value) if value is not None else '(none)'}"
                        exclusions[key] = exclusions.get(key, 0) + 1
                        round_exclusions += 1
                        continue
                    taken[value] = taken.get(value, 0) + 1
                    kept.append(rec)
                if not kept:
                    _raise_share_caps_fail_closed(
                        flag=flag,
                        limit=limit,
                        dimension=dimension,
                        values=list(over),
                    )
                population = kept
                pass_exclusions += round_exclusions
        if pass_exclusions == 0:
            break
    return population, exclusions


def _share_caps_report(
    records: list[Record],
    exclusions: dict[str, int],
    *,
    max_stack_share: float | None,
    max_repo_share: float | None,
    max_profile_share: float | None,
) -> dict[str, Any]:
    """Shared summary/lineage ``share_caps`` block (one builder, two consumers
    — they cannot drift). ``configured`` lists the non-None share limits keyed
    stack/repo/profile; ``applied`` reports final per-value counts + shares
    against the final emitted population per dimension; ``exclusions`` lists
    the merged ``share-cap:<dimension>:<value>`` exclusion counts. Both draw
    dimension names and value getters from the single ``_SHARE_DIMENSIONS``
    table, so configured/applied/exclusion vocabulary stays one spelling."""
    limits = {
        "stack": max_stack_share,
        "repo": max_repo_share,
        "profile": max_profile_share,
    }
    configured = {
        name: limits[name]
        for name, _flag, _getter in _SHARE_DIMENSIONS
        if limits[name] is not None
    }
    applied: dict[str, dict[str, dict[str, float]]] = {}
    total = len(records)
    for name, _flag, getter in _SHARE_DIMENSIONS:
        counts: dict[Any, int] = {}
        for rec in records:
            value = getter(rec)
            counts[value] = counts.get(value, 0) + 1
        applied[name] = {
            str(value) if value is not None else "(none)": {
                "count": count,
                "share": (count / total) if total else 0.0,
            }
            for value, count in sorted(
                counts.items(), key=lambda kv: (kv[0] is None, str(kv[0]))
            )
        }
    return {
        "version": 1,
        "configured": configured,
        "applied": applied,
        "exclusions": dict(sorted(exclusions.items())),
    }


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
    per-split counts, the ``caps`` block (configured vs applied), the
    ``share_caps`` block (only when at least one output-share cap is
    configured — shared with lineage.json so the two cannot drift), and
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
    # Per-repo license decisions (issue #1080): the projector re-runs the C5/C8
    # gate over every admitted batch (defence in depth — admission should have
    # caught these) as the pure function of the batch's recorded repo identity
    # + evidence under the pinned policy, so the re-evaluation is
    # replay-identical to admission. Any non-admitted decision refuses the
    # build outright before any file write (C5-excluded, unopted copyleft,
    # missing identity/evidence alike — always-enforced, fail-closed), so no
    # benchmark or unopted-copyleft repo content can reach training data.
    # mypy narrowing: __post_init__ guarantees a non-None policy path
    # (ValueError otherwise), but mypy can't see through object.__setattr__,
    # so assert it at the use site.
    assert config.license_policy_path is not None
    policy, policy_digest = load_license_policy(config.license_policy_path)
    decisions: dict[str, dict[str, Any]] = {}
    license_refusals: list[tuple[str, str]] = []
    for batch in bundle.admitted:
        repo_decision = resolve_repo_decision(
            batch.repo_slug or "", batch.license_evidence, policy, config.allow_copyleft
        )
        decisions[batch.session_id] = {
            "status": repo_decision.status,
            "reason_code": repo_decision.reason_code,
            "spdx_id": repo_decision.spdx_id,
            "policy_version": repo_decision.policy_version,
            "evidence_ref": repo_decision.evidence_ref,
            "repo_slug": repo_decision.repo_slug,
        }
        # Any license rejection (not just the always-enforced C5 refusal)
        # refuses the whole projection before any file write: an unopted
        # copyleft (or identity/evidence-missing) batch's records must never
        # be emitted, and the fail-closed ordering is compute rejections →
        # raise if any → else write (M9; AC6; covered end-to-end by
        # test_end_to_end_mixed_repo_publication_gated).
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
    # The annotation resolutions are session-scoped (keyed by session_id +
    # fingerprint, not by trajectory/segment), so each finding is projected
    # exactly once — sibling segments never fabricate per-segment copies that
    # could hash into different splits (D5 disjointness is per finding).
    seen_findings: set[tuple[str, str]] = set()
    adjudicated_findings: set[tuple[str, str]] = set()
    exclusions_by_reason: dict[str, int] = {}
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
                resolution_by_fp = {
                    str(row.get("fingerprint")): row for row in resolutions
                }
                for rec in seg_records:
                    fingerprint = str(rec["finding_fingerprint"])
                    key = (seg.session_id, fingerprint)
                    if key in seen_findings:
                        continue
                    seen_findings.add(key)
                    decision = decisions.get(seg.session_id)
                    if decision is None:
                        raise ValueError(
                            f"session {seg.session_id}: no recorded license decision "
                            "found at projection; refusing to project"
                        )
                    record_decision: dict[str, Any] = decision
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
                        "repo_slug": record_decision["repo_slug"],
                        "license_decision": record_decision,
                    }
                    # Additive enrichment from the batch's review artifacts
                    # (findings.json / diff.patch / manifest.json): localized
                    # finding text and a task-identity block. Every field is
                    # opt-in on artifact presence — a missing artifact leaves
                    # the field absent (the consumer fails closed, not the
                    # projector), and record_id/tier/outcome_label are never
                    # touched.
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
                    for sha_key in ("base_sha", "head_sha"):
                        sha_value = manifest_git.get(sha_key)
                        if isinstance(sha_value, str) and bool(sha_value):
                            task_identity[sha_key] = sha_value
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
                    rec["task_identity"] = task_identity
                    if "diff_digest" in task_identity:
                        lineage_fields = cast(dict[str, Any], rec["lineage"])
                        lineage_fields["diff_digest"] = task_identity["diff_digest"]
                        lineage_fields["diff_ref"] = task_identity["diff_ref"]
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
    exclusions_by_reason["non-decisive-adjudication"] = len(adjudication)
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

    # Output-share caps (issue #1079): true share of the final emitted
    # population, enforced per dimension (stack → repo → profile) over
    # the post-tier-cap population. The dimension sequence is re-run to a
    # fixed point so a later pass can never leave an earlier dimension back
    # over its cap (F1: a single sequential pass drifts on correlated
    # dimensions). Only runs when at least one share is configured, so
    # tier-cap-only builds stay byte-identical and neither the summary nor
    # lineage gains a ``share_caps`` block.
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

    # Re-pin lineage valid_at over the final emitted population (post tier-cap
    # and share-cap exclusions): the in-loop pin above is computed before
    # either caps stage drops records, so it could name a cap-excluded record's
    # evidence — contradicting the "emitted records' evidence only" contract
    # above. Recomputing over the post-cap records keeps the pin reachable;
    # uncapped builds re-pin to the identical value (max is order-independent).
    valid_at = config.as_of
    for rec in records:
        rec_valid_at = cast(dict[str, Any], rec["lineage"]).get("valid_at")
        if rec_valid_at is not None and (
            valid_at is None
            or datetime.fromisoformat(str(rec_valid_at)) > datetime.fromisoformat(valid_at)
        ):
            valid_at = str(rec_valid_at)

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
    # exclusion reasons, and each record's per-repo license decision. Nothing
    # silently dropped.
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
        **(
            {"share_caps": share_caps_report}
            if share_caps_report is not None
            else {}
        ),
        "adjudication_count": len(adjudication),
        # License identity pin (M7/AC4, issue #1080): the digest-pinned policy
        # the re-evaluation consumed, the C5 exclusion list the C5 gate
        # consulted, the copyleft opt-ins, every recorded per-repo decision,
        # and the admitted/rejected decision distribution — all pure functions
        # of the bundle + policy, so re-runs stay byte-identical.
        "license_policy": {
            "path_digest": policy_digest,
            "policy_version": policy.policy_version,
        },
        "exclusion_list_digest": hashlib.sha256(EXCLUSION_PATH.read_bytes()).hexdigest(),
        "copyleft_opt_ins": sorted(config.allow_copyleft),
        "license_decisions": {
            str(session_id): decision for session_id, decision in decisions.items()
        },
        "license_decision_distribution": _license_decision_distribution(decisions),
    }
    _atomic_write(
        config.out_dir / "lineage.json",
        json.dumps(lineage, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    )

    # Human license report (issue #1080 M7): the digest-pinned policy, the C5
    # exclusion-list digest, the copyleft opt-ins, every per-repo decision, and
    # the decision distribution — a pure function of the bundle + policy +
    # exclusion.txt bytes, so re-runs are byte-identical. Written atomically
    # before ``_SUCCESS`` so the completeness gate covers it; the license-gate
    # refusal above means this file only ever exists on clean builds.
    license_report = {
        "policy": {"policy_version": policy.policy_version, "digest": policy_digest},
        "exclusion_list_digest": lineage["exclusion_list_digest"],
        "copyleft_opt_ins": sorted(config.allow_copyleft),
        "decisions": dict(sorted(
            (str(session_id), decision) for session_id, decision in decisions.items()
        )),
        "distribution": _license_decision_distribution(decisions),
    }
    _atomic_write(
        config.out_dir / "license-report.json",
        json.dumps(license_report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
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
        **(
            {"share_caps": share_caps_report}
            if share_caps_report is not None
            else {}
        ),
        "exclusions_by_reason": dict(sorted(exclusions_by_reason.items())),
        "license_distribution": _license_decision_distribution(decisions),
    }
