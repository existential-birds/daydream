"""Staging final-bundle constructor (issue #1078, M4 core).

Assembles the seven-file semantic annotation staging bundle, including exact
preview-manifest and verified v2 policy-binding bytes, from pipeline state (the
materialization dir, the hydrated index, and the archive). Pure construction:
no Hub I/O, no publishing; :func:`publish_final_annotation_bundle` consumes the
directory this function produces, so ``--dry-run`` and the real publish share
100% of the construction/validation code (M6).

Determinism: every file is written as canonical JSON (sorted keys, compact
separators), so identical pipeline state produces byte-identical bundles
across re-runs, and atomically via the shared ``atomic_write_bytes``
primitive, so a torn write can never leave a truncated contract file in the
staged ``out_dir`` (the deterministic-atomic-writes convention). Every missing
or invalid input raises ``ValueError`` / ``FileNotFoundError`` naming the
artifact — no fallback defaults, no silent skips (the lineage file must never
contain a fabricated field).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from daydream.archive.hydrate_rules import derive_curation_id
from daydream.archive.index import label_observation_history
from daydream.archive.sanitize import _derivative_digest
from daydream.json_utils import atomic_write_bytes, canonical_json as _canonical, umask_derived_mode
from daydream.training.adjudication.canonical import (
    _evidence_after_as_of,
    _load_materialized_records as _load_sessions_records,
    _read_manifest,
    read_jsonl,
)
from daydream.training.adjudication.materialize import (
    _ANNOTATIONS_FILENAME,
    _MANIFEST_FILENAME,
    _SESSIONS_OUT_FILENAME,
    index_sessions,
)
from daydream.training.adjudication.observations import (
    group_observations_by_record,
    load_observations,
)
from daydream.training.adjudication.precedence import effective_adjudication
from daydream.training.adjudication.queue import build_queue
from daydream.training.adjudication.report import build_report
from daydream.training.corpus_projection.bundle import load_curated_bundle
from daydream.training.corpus_projection.tiers import classify_tier
from daydream.training.dispositions import (
    DECISIVE_DISPOSITIONS,
    NON_DECISIVE_DISPOSITIONS,
)
from daydream.training.labeler_versions import ANNOTATION_SNAPSHOT_SCHEMA_VERSION

__all__ = ["FINAL_IDENTITY_FILES", "build_final_bundle", "final_snapshot_id"]

_OBSERVATIONS_FILENAME = "label-observations.jsonl"
_REPORT_FILENAME = "coverage-report.json"
_LINEAGE_FILENAME = "lineage.json"
_POLICY_BINDING_FILENAME = "policy-binding.json"

_BUNDLE_FILES = (
    _ANNOTATIONS_FILENAME,
    _SESSIONS_OUT_FILENAME,
    _OBSERVATIONS_FILENAME,
    _REPORT_FILENAME,
    _LINEAGE_FILENAME,
    _MANIFEST_FILENAME,
    _POLICY_BINDING_FILENAME,
)
FINAL_IDENTITY_FILES = _BUNDLE_FILES

# Older publishers left this scratch directory in the bundle root. Continue
# tolerating it during deterministic reconstruction; current publication uses
# unique sibling temporary directories and leaves no in-bundle scratch state.
_PUBLISH_STAGE_DIRNAME = ".publish-stage"

# The lineage fields that must be present and non-empty in the pin/manifest —
# a missing field is a hard error naming the field, never a fallback default.
_LINEAGE_PIN_FIELDS = (
    "curation_id",
    "sanitized_hub_commit",
    "snapshot_id",
    "labeler_version",
    "rubric_version",
    "classifier_version",
)


def _bundle_input_names(root: Path) -> set[str]:
    """Ignore only a real legacy scratch directory, without reading its contents."""
    return {
        path.name
        for path in root.iterdir()
        if not (
            path.name == _PUBLISH_STAGE_DIRNAME
            and not path.is_symlink()
            and path.is_dir()
        )
    }


def _write_bundle_file(out_dir: Path, name: str, data: bytes) -> None:
    """Atomically stage one bundle file under the deterministic-write convention."""
    atomic_write_bytes(out_dir / name, data, fsync=False, dir_fsync=False, mode=umask_derived_mode())


def final_snapshot_id(bundle_dir: Path) -> tuple[str, dict[str, str]]:
    """Hash the exact seven-file semantic annotation bundle contract."""
    root = Path(bundle_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("final bundle must be a real directory")
    allowed_envelope = {"publication-manifest.json", "SHA256SUMS", "_SUCCESS"}
    names = _bundle_input_names(root)
    foreign = sorted(names - set(FINAL_IDENTITY_FILES) - allowed_envelope)
    missing = sorted(set(FINAL_IDENTITY_FILES) - names)
    if foreign or missing:
        raise ValueError(
            f"final bundle identity file set mismatch: missing={missing}, foreign={foreign}"
        )
    digests: dict[str, str] = {}
    for name in FINAL_IDENTITY_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"final bundle identity input {name!r} must be a regular file")
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = _canonical(digests).encode("utf-8") + b"\n"
    return hashlib.sha256(identity).hexdigest(), dict(sorted(digests.items()))


def _require_regular_input(root: Path, name: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"input root for {name} must be a real directory")
    path = root / name
    if path.is_symlink():
        raise ValueError(f"input {name} must be a regular non-symlink file")
    if not path.is_file():
        raise FileNotFoundError(f"input {name} is missing")
    return path


def _load_materialized_records(materialize_dir: Path) -> list[dict[str, Any]]:
    annotations_path = materialize_dir / _ANNOTATIONS_FILENAME
    return read_jsonl(
        annotations_path,
        missing=(
            f"materialized annotations not found (run `corpus adjudicate materialize` and "
            f"`corpus adjudicate harvest-snapshot` first): {annotations_path}"
        ),
        invalid="unreadable materialized annotations",
    )


def _load_manifest(materialize_dir: Path) -> dict[str, Any]:
    manifest_path = materialize_dir / _MANIFEST_FILENAME
    manifest = _read_manifest(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"preview manifest at {manifest_path} is not a JSON object")
    return manifest


def _lineage_field(manifest: Mapping[str, Any], field: str, manifest_path: Path) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"preview manifest at {manifest_path} is missing required lineage field {field!r}"
        )
    return value


def _validate_policy_binding(
    raw: bytes,
    *,
    label: str,
    curation_id: str,
    source_hub_commit: str,
) -> None:
    """Validate producer-canonical v2 policy-binding bytes and rederive identity.

    Shared by the staging constructor and the publication boundary so the
    publication path cannot drift from the construction rules.
    """
    try:
        binding = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: unreadable policy binding ({exc})") from None
    required = {
        "schema_version",
        "policy_digest",
        "policy_version",
        "allow_copyleft",
        "exclusions_digest",
        "resolved_decisions_digest",
        "distribution_digest",
    }
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError(f"{label}: must contain the exact v2 field set")
    if binding["schema_version"] != "2":
        raise ValueError(f"{label}: unsupported schema_version")
    for name in (
        "policy_digest",
        "exclusions_digest",
        "resolved_decisions_digest",
        "distribution_digest",
    ):
        value = binding[name]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{label}: invalid {name}")
    policy_version = binding["policy_version"]
    if not isinstance(policy_version, str) or not policy_version:
        raise ValueError(f"{label}: invalid policy_version")
    allow_copyleft = binding["allow_copyleft"]
    if (
        not isinstance(allow_copyleft, list)
        or any(
            not isinstance(slug, str)
            or not slug
            or slug != slug.casefold()
            for slug in allow_copyleft
        )
        or allow_copyleft != sorted(set(allow_copyleft))
    ):
        raise ValueError(f"{label}: invalid allow_copyleft")
    canonical = (json.dumps(binding, sort_keys=True) + "\n").encode("utf-8")
    if raw != canonical:
        raise ValueError(f"{label}: not canonically encoded")
    derived = derive_curation_id(
        source_hub_commit,
        binding["policy_digest"],
        policy_version,
        frozenset(allow_copyleft),
        binding["exclusions_digest"],
        binding["resolved_decisions_digest"],
        binding["distribution_digest"],
    )
    if derived != curation_id:
        raise ValueError(f"{label}: derives curation_id {derived!r}, not {curation_id!r}")


def _validated_policy_binding(
    curation_bundle_dir: Path,
    *,
    curation_id: str,
    source_hub_commit: str,
) -> bytes:
    """Load the producer-canonical v2 binding and rederive its identity."""
    path = curation_bundle_dir / _POLICY_BINDING_FILENAME
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"curation policy binding not found as a regular file: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"unreadable policy binding at {path}: {exc}") from None
    _validate_policy_binding(
        raw,
        label=f"policy binding at {path}",
        curation_id=curation_id,
        source_hub_commit=source_hub_commit,
    )
    return raw


def _enrich_report_items(
    items: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Enrich queue items into report items — the single shared implementation
    behind both the CLI report path (``cli._report_items``) and the final
    bundle's coverage report, so the published 80% admission gate sees exactly
    the same human-decision state as ``corpus adjudicate report``.

    Attaches each record's observations, applies three-tier effective
    adjudication (a human decisive judgment whose evidence digest matches the
    fresh item overrides the disposition), stamps the temporal axis first, and
    classifies ``tier``/``posterior_eligible`` with the projection authority.
    Gold eligibility requires a human decision made against the item's fresh
    evidence digest, so automatic decisive records without one — and decisive
    records whose only human observation was made against older evidence, a
    judgment the canonical merge refuses to reuse (digest mismatch) — are
    demoted to task-only and never count as adjudicated. An observation
    referencing a record_id absent from ``items`` raises ``ValueError`` naming
    it (fail-closed, mirroring the CLI twin).
    """
    queue_ids = {str(item["record_id"]) for item in items}
    grouped = group_observations_by_record(observations, queue_ids, "report")

    enriched: list[dict[str, Any]] = []
    for item in items:
        enriched_item = dict(item)
        record_obs = grouped.get(str(item["record_id"]), [])
        enriched_item["observations"] = record_obs
        gold_eligible = False
        if record_obs:
            resolved = effective_adjudication(record_obs)
            # A human judgment made against different evidence is never
            # silently reused (the queue's digest-drift rule); gold
            # eligibility therefore holds only when the effective
            # observation's evidence_digest equals the item's fresh digest —
            # mirroring the disposition override two lines below and the
            # canonical merge's digest-match guard (canonical.py).
            fresh_match = resolved["evidence_digest"] == str(item["evidence_digest"])
            if fresh_match:
                gold_eligible = resolved["gold_eligible"]
            if (
                fresh_match
                and resolved["role"] in ("rater", "adjudicator")
                and resolved["disposition"] in DECISIVE_DISPOSITIONS
            ):
                enriched_item["disposition"] = resolved["disposition"]
        # Stamp the temporal axis FIRST so classify_tier sees it, matching the
        # canonical serializer and the corpus projection (tiers.py C5/M9): an
        # evidence-after-as_of record must classify "silver" here exactly as it
        # does on the canonical record — never gold/posterior_eligible.
        enriched_item["evidence_after_as_of"] = _evidence_after_as_of(enriched_item, as_of)
        # The gold gate has one implementation (classify_tier); gold-eligibility
        # comes from the human-observation resolution (conflict/review-required
        # decisive judgments stay out of the gold tier). A classifier failure
        # fail-closes naming the record, never a silent skip.
        try:
            tier = classify_tier(enriched_item)
        except Exception as exc:
            raise ValueError(
                f"cannot classify tier for record_id {str(item['record_id'])!r}: {exc}"
            ) from exc
        if tier == "gold" and not gold_eligible:
            tier = "task-only"
        enriched_item["tier"] = tier
        enriched_item["posterior_eligible"] = tier == "gold" and str(
            enriched_item["profile"]
        ) == "pr_review"
        enriched.append(enriched_item)
    return enriched


def build_final_bundle(
    *,
    index_root: Path,
    materialize_dir: Path,
    archive_dir: Path,
    out_dir: Path,
    curation_bundle_dir: Path | None = None,
    observations_path: Path | None = None,
) -> dict[str, Any]:
    """Construct the final annotation staging bundle into a fresh ``out_dir``.

    Writes exactly the seven semantic contract files (``_BUNDLE_FILES``) and never
    publishes: the caller feeds the directory to
    :func:`daydream.training.adjudication.publish.publish_final_annotation_bundle`.

    - ``annotations.jsonl`` / ``sessions.jsonl``: both copied byte-for-byte from the
      materialization dir (missing file raises ``FileNotFoundError`` naming it).
    - ``label-observations.jsonl``: the archive's immutable per-session
      observation history (``archive.index.label_observation_history``) for
      every session in the snapshot, flattened and sorted chronologically by
      ``observed_at``. The projector's bundle verification treats unknown
      bundle files as part of the SHA256SUMS file set, so this extra file is
      additive-safe.
    - ``coverage-report.json``: ``report.build_report`` over a fresh complete
      queue (``build_queue(..., include_decisive=True)``) enriched by the
      shared :func:`_enrich_report_items` — the same observations/
      effective-adjudication/gold-eligibility enrichment the CLI report twin
      (``cli._report_items``) applies — so the published 80% admission gate
      counts human-adjudicated outcome-bearing records only, never every
      automatic decisive record. ``observations_path`` (the ``--state-dir``
      observations store; a missing file is an empty store) supplies the
      human observations; empty when not given.
    - ``lineage.json``: generated from the preview manifest's pin fields —
      never hand-authored, never defaulted. ``batch_fileset_digest`` is
      ``_derivative_digest`` over ``curation_bundle_dir`` (default: the index
      root when it *is* the curation bundle root), validated to exist.

    ``out_dir`` must not exist, must be empty, or may contain only the
    seven contract files from a prior (deterministic) construction — e.g. a
    ``--dry-run`` validation immediately followed by a real publish over the
    same state. Construction is deterministic, so re-writing those files is
    byte-identical; any foreign file (a previous run's ``_SUCCESS``, editor
    droppings, a partial publish) raises ``ValueError`` naming the directory,
    because a stale or published staging dir must never be silently mixed
    with fresh content. A real, non-symlink ``.publish-stage`` directory left
    by an older publisher is preserved without reading its contents; it is
    excluded from construction, identity, and publication.

    Returns a summary dict with ``disposition_counts`` covering all five
    dispositions (``accepted``/``rejected``/``ambiguous``/``unanswered``/
    ``missing``) plus ``record_count`` and the written file names.
    """
    bundle_root = curation_bundle_dir if curation_bundle_dir is not None else index_root
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise ValueError("curation bundle root must be a real directory")
    curated = load_curated_bundle(bundle_root)
    if out_dir.exists():
        foreign = sorted(_bundle_input_names(out_dir) - set(_BUNDLE_FILES))
        if foreign:
            raise ValueError(
                f"final-bundle staging dir {out_dir} contains foreign content "
                f"({', '.join(foreign)}); remove it or pass a fresh path"
            )
    _require_regular_input(materialize_dir, _ANNOTATIONS_FILENAME)
    _require_regular_input(materialize_dir, _SESSIONS_OUT_FILENAME)
    _require_regular_input(materialize_dir, _MANIFEST_FILENAME)
    records = _load_materialized_records(materialize_dir)
    _load_sessions_records(materialize_dir)
    manifest_path = materialize_dir / _MANIFEST_FILENAME
    manifest = _load_manifest(materialize_dir)
    preview_curation = _lineage_field(manifest, "curation_id", manifest_path)
    preview_source = _lineage_field(manifest, "source_hub_commit", manifest_path)
    preview_sanitized = _lineage_field(manifest, "sanitized_hub_commit", manifest_path)
    if preview_curation != curated.curation_id:
        raise ValueError(
            f"preview manifest curation_id {preview_curation!r} does not match "
            f"curated bundle {curated.curation_id!r}"
        )
    if preview_source != curated.source_hub_commit or preview_sanitized != curated.source_hub_commit:
        raise ValueError("preview manifest source/materialization pins do not match the curated bundle")
    policy_binding = _validated_policy_binding(
        bundle_root,
        curation_id=curated.curation_id,
        source_hub_commit=curated.source_hub_commit,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Both consumer views use the canonical merged records. Copying preview
    #    sessions here would discard imported/human decisions for projection.
    _write_bundle_file(
        out_dir, _ANNOTATIONS_FILENAME, (materialize_dir / _ANNOTATIONS_FILENAME).read_bytes()
    )
    _write_bundle_file(
        out_dir, _SESSIONS_OUT_FILENAME, (materialize_dir / _ANNOTATIONS_FILENAME).read_bytes()
    )
    _write_bundle_file(out_dir, _MANIFEST_FILENAME, manifest_path.read_bytes())
    _write_bundle_file(out_dir, _POLICY_BINDING_FILENAME, policy_binding)

    # 2. label-observations.jsonl: the archive's per-session observation
    #    history, chronological by ``observed_at`` (per-session rows are
    #    already ordered; the cross-session flatten re-sorts for determinism).
    snapshot_session_ids = sorted({str(r["session_id"]) for r in records})
    history_rows: list[dict[str, Any]] = []
    for session_id in snapshot_session_ids:
        history_rows.extend(label_observation_history(archive_dir, session_id))
    history_rows.sort(key=lambda row: (str(row.get("observed_at")), str(row.get("session_id"))))
    _write_bundle_file(
        out_dir,
        _OBSERVATIONS_FILENAME,
        "".join(_canonical(row) + "\n" for row in history_rows).encode("utf-8"),
    )

    # 3. coverage-report.json over the fresh complete queue, enriched exactly
    #    like the CLI report twin (``cli._report_items`` -> shared
    #    ``_enrich_report_items``): observations attached per record, three-tier
    #    effective adjudication applied, and gold records without a human
    #    decision demoted to task-only, so the published 80% admission gate
    #    sees real human adjudication state instead of counting every automatic
    #    decisive record as adjudicated. ``as_of`` comes from the preview
    #    manifest (empty when unpinned).
    observations = load_observations(observations_path) if observations_path is not None else []
    report_items = _enrich_report_items(
        build_queue(index_sessions(index_root)[0], include_decisive=True),
        observations,
        as_of=manifest.get("as_of"),
    )
    report = build_report(report_items)
    # ``strata`` keys are (stack, profile) tuples in memory; the on-disk report
    # is JSON, so flatten them to ``"stack/profile"`` (deterministic order).
    report["strata"] = {
        f"{stack}/{profile}": count for (stack, profile), count in report["strata"].items()
    }
    _write_bundle_file(out_dir, _REPORT_FILENAME, (_canonical(report) + "\n").encode("utf-8"))

    # 4. lineage.json: generated from the pin — every field must be present.
    lineage: dict[str, Any] = {
        field: _lineage_field(manifest, field, manifest_path) for field in _LINEAGE_PIN_FIELDS
    }
    lineage["schema_version"] = f"annotation-snapshot/{ANNOTATION_SNAPSHOT_SCHEMA_VERSION}"
    lineage["batch_fileset_digest"] = _derivative_digest(bundle_root)
    # ``as_of`` is pin-required but legitimately empty when unpinned (mirroring
    # ``snapshot.snapshot_id``'s as-of edge) — a *missing* key is an error, and
    # a null/empty value is the unpinned edge, never the fabricated "None".
    if "as_of" not in manifest:
        raise ValueError(
            f"preview manifest at {manifest_path} is missing required lineage field 'as_of'"
        )
    as_of = manifest["as_of"]
    lineage["as_of"] = "" if as_of is None else str(as_of)
    _write_bundle_file(out_dir, _LINEAGE_FILENAME, (_canonical(lineage) + "\n").encode("utf-8"))

    written = sorted(path.name for path in out_dir.iterdir() if path.is_file())
    missing = [name for name in _BUNDLE_FILES if name not in written]
    if missing:
        raise ValueError(f"final bundle at {out_dir} is missing contract files: {missing}")

    counts = Counter(str(r["disposition"]) for r in records)
    disposition_counts = {
        **{d: counts[d] for d in sorted(DECISIVE_DISPOSITIONS | NON_DECISIVE_DISPOSITIONS)},
        **{k: v for k, v in sorted(counts.items())
           if k not in DECISIVE_DISPOSITIONS and k not in NON_DECISIVE_DISPOSITIONS},
    }
    return {
        "disposition_counts": disposition_counts,
        "record_count": len(records),
        "observation_history_rows": len(history_rows),
        "files": written,
        "out_dir": str(out_dir),
    }
