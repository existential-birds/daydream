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

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, cast, overload

from daydream.archive.index import normalize_as_of
from daydream.training.corpus import _is_posterior_leak, _trajectory_set_hash
from daydream.training.corpus_v2.bundle import load_curated_bundle
from daydream.training.corpus_v2.identity import record_id
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
    annotations_snapshot: Path
    as_of: str | None = None
    holdout_rate: float = 0.1
    val_rate: float = 0.1
    salt: str = "daydream-corpus-v2"
    caps: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.as_of is not None:
            object.__setattr__(self, "as_of", normalize_as_of(self.as_of))


def _load_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    """Parse the side-car annotations snapshot (Task 0A shape): one JSON
    object per line, keyed by ``fingerprint``. Fail-closed on malformed
    lines or duplicate fingerprints."""
    rows: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"annotations snapshot {path}: line {line_no} is not valid JSON: {exc}"
            ) from exc
        fingerprint = row.get("fingerprint")
        if not fingerprint:
            raise ValueError(f"annotations snapshot {path}: line {line_no} missing 'fingerprint'")
        if fingerprint in rows:
            raise ValueError(
                f"annotations snapshot {path}: duplicate fingerprint {fingerprint!r} "
                f"(lines {line_no} and earlier) — snapshot must be keyed by fingerprint"
            )
        rows[str(fingerprint)] = row
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
        record = {
            "record_id": record_id(
                str(session_id), str(trajectory_id), str(segment_id), str(fingerprint)
            ),
            "record_type": "outcome-finding",
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "segment_id": segment_id,
            "finding_fingerprint": fingerprint,
            "tier": tier,
            "disposition": disposition,
            "outcome_label": disposition if tier == "gold" else None,
            "evidence": evidence,
        }
        records.append(record)
        if tier == "task-only":
            adjudication.append(
                {
                    "fingerprint": fingerprint,
                    "disposition": disposition,
                    "evidence": evidence,
                    "reason": f"non-decisive disposition {disposition!r}",
                }
            )

    if return_adjudication:
        return records, adjudication
    return records


def run_build_corpus_v2(config: BuildCorpusV2Config) -> dict[str, int]:
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
    pinning the snapshot's provenance (no wall-clock timestamps — every
    manifest byte is a function of the immutable inputs, so re-runs are
    byte-for-byte identical).

    Returns a summary dict with ``total``, ``emitted``, per-split counts and
    ``adjudication`` (non-decisive findings routed to the human pass).
    """
    bundle = load_curated_bundle(config.bundle_dir)
    snapshot_rows = _load_snapshot(config.annotations_snapshot)

    records: list[Record] = []
    adjudication: list[Record] = []
    included_sessions: list[str] = []
    for batch in bundle.admitted:
        artifact = config.bundle_dir / batch.artifact_relpath
        for line_no, line in enumerate(artifact.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                trajectory = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"bundle {config.bundle_dir}: {batch.artifact_relpath} line {line_no} "
                    f"is not valid JSON: {exc}"
                ) from exc
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
                for rec in seg_records:
                    fingerprint = str(rec["finding_fingerprint"])
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
                    rec["lineage"] = {"split": split}
                records.extend(seg_records)
                adjudication.extend(seg_adjudication)
                included_sessions.append(seg.session_id)

    records.sort(key=lambda r: str(r["record_id"]))
    adjudication.sort(key=lambda r: str(r["fingerprint"]))

    canonical = _dump_jsonl(records)
    _atomic_write(config.out_dir / "corpus.jsonl", canonical)
    for split_name, filename in _SPLIT_FILENAMES.items():
        split_records = [r for r in records if cast(dict[str, Any], r["lineage"])["split"] == split_name]
        _atomic_write(config.out_dir / filename, _dump_jsonl(split_records))
    _atomic_write(
        config.out_dir / "adjudication.json",
        json.dumps(adjudication, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    )

    schema_src = Path(__file__).parent.parent / "schema" / "v2.json"
    schema_dst = config.out_dir / "schema.json"
    schema_dst.parent.mkdir(parents=True, exist_ok=True)
    schema_dst.write_bytes(schema_src.read_bytes())

    split_counts = {name: 0 for name in _SPLIT_FILENAMES}
    for r in records:
        lineage_field = cast(dict[str, Any], r["lineage"])
        split_counts[str(lineage_field["split"])] += 1
    lineage = {
        "schema_version": "corpus-v2",
        "curation_id": bundle.curation_id,
        "source_hub_commit": bundle.source_hub_commit,
        "annotations_snapshot": config.annotations_snapshot.name,
        "as_of": config.as_of,
        "salt": config.salt,
        "holdout_rate": config.holdout_rate,
        "val_rate": config.val_rate,
        "trajectory_set_hash": _trajectory_set_hash(sorted(set(included_sessions))),
        "split_counts": split_counts,
        "adjudication_count": len(adjudication),
    }
    _atomic_write(
        config.out_dir / "lineage.json",
        json.dumps(lineage, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    )

    return {
        "total": len(records),
        "emitted": len(records),
        "adjudication": len(adjudication),
        **{f"split_{name}": count for name, count in split_counts.items()},
    }
