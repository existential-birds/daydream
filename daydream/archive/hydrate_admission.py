"""License gates, content deduplication, policy identity, and admission accounting."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from daydream.archive import hydrate_rules, sanitize
from daydream.archive.hydrate_discovery import (
    _is_bare_segment,
)
from daydream.archive.hydrate_rules import (
    REASON_CODE_BUNDLE_UNREADABLE,
    REASON_CODE_C5_EXCLUDED_REPO,
    REASON_CODE_C8_COPYLEFT_UNOPTED,
    REASON_CODE_IDENTITY_COLLISION,
    REASON_CODE_LICENSE_EVIDENCE_MISSING,
    REASON_CODE_REPO_COMMIT_UNRESOLVED,
    REASON_CODE_REPO_IDENTITY_MISSING,
)
from daydream.archive.hydrate_stage import (
    _admitted_session_id,
    _bundle_dirs,
    _derivative_manifests,
    _discovered_session_ids,
    _download_discovery_block,
    _manifest_remote_fields,
    _move_dir,
    _read_manifest_dict,
    _read_manifest_field,
)
from daydream.archive.hydrate_types import (
    DedupeResult,
    HydrationError,
)
from daydream.archive.index import manifest_index_fields, query_runs, upsert_run
from daydream.json_utils import atomic_write_json
from daydream.redaction import redact_text
from daydream.timeutil import now_iso_utc
from daydream.training.exclusion import EXCLUSION_PATH
from daydream.trajectory import RUNS_DIRNAME

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


_LICENSE_BUCKET_BY_CODE: dict[str, str] = {
    REASON_CODE_C5_EXCLUDED_REPO: "c5_excluded",
    REASON_CODE_C8_COPYLEFT_UNOPTED: "c8_copyleft_unopted",
    REASON_CODE_LICENSE_EVIDENCE_MISSING: "license_evidence_missing",
    REASON_CODE_REPO_IDENTITY_MISSING: "license_evidence_missing",
    REASON_CODE_REPO_COMMIT_UNRESOLVED: "license_evidence_missing",
}


_LICENSE_REASON_CODES = frozenset(_LICENSE_BUCKET_BY_CODE)


_LICENSE_BUCKETS = ("admitted", "c5_excluded", "c8_copyleft_unopted", "license_evidence_missing")


def _manifest_repo_slug(data: dict[str, Any]) -> str | None:
    """Repository identity from a session manifest (nested ``git.repo_slug`` or flat)."""
    raw = _read_manifest_field(data, "repo_slug")
    return raw if isinstance(raw, str) and raw.strip() else None


def _manifest_license_evidence(data: dict[str, Any]) -> dict[str, str] | None:
    """Declared license evidence (``spdx_id`` + ``source``) from a session manifest."""
    raw = data.get("license_evidence")
    if not isinstance(raw, dict):
        return None
    spdx_id = raw.get("spdx_id")
    if not isinstance(spdx_id, str) or not spdx_id.strip():
        return None
    evidence = {"spdx_id": spdx_id.strip()}
    source = raw.get("source")
    if isinstance(source, str):
        evidence["source"] = source
    return evidence


def _repo_license_decision(
    data: dict[str, Any], policy: Any, allow_copyleft: frozenset[str] | set[str]
) -> Any:
    """Resolve one manifest's repo decision from its slug/evidence + gate policy."""
    from daydream.training.corpus_projection.license import (  # noqa: PLC0415  # local: avoid import cycle at module load
        resolve_repo_decision,
    )

    return resolve_repo_decision(
        _manifest_repo_slug(data) or "",
        _manifest_license_evidence(data),
        policy,
        allow_copyleft,
    )


def _session_identity(stage: Path, sid: str, revision: str, *, root: str, collision: bool) -> \
        tuple[str | None, dict[str, str] | None]:
    """Read repo/license identity in derivative precedence, or (None, None) if every manifest is unreadable."""
    segment = sid if _is_bare_segment(sid) else hashlib.sha256(sid.encode()).hexdigest()
    candidates = [stage / RUNS_DIRNAME / sid]
    if collision:
        candidates.append(stage / "quarantine" / f"{segment}.conflict")
    candidates += [stage / root / segment,
                   stage / "downloads" / str(revision) / "bundles" / segment]
    for candidate in candidates:
        data = _read_manifest_dict(candidate)
        if data is not None:
            return _manifest_repo_slug(data), _manifest_license_evidence(data)
    return None, None


def _iter_enrichment_cache(stage: Path) -> Iterator[dict[str, Any]]:
    """Yield the dict rows of the enrichment evidence cache, skipping malformed lines."""
    from daydream.archive.license_enrich import (  # noqa: PLC0415  # local: avoid import cycle at module load
        _ENRICH_CACHE_NAME,
        _ENRICH_DIR,
    )

    path = stage / _ENRICH_DIR / _ENRICH_CACHE_NAME
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            yield entry


def _repo_commit_unresolved_sessions(stage: Path) -> set[str]:
    """Session ids whose enrichment could identify a repo but could not pin its commit."""
    unresolved: set[str] = set()
    for entry in _iter_enrichment_cache(stage):
        if entry.get("status") != REASON_CODE_REPO_COMMIT_UNRESOLVED:
            continue
        sid = entry.get("session_id")
        if isinstance(sid, str) and sid:
            unresolved.add(sid)
    return unresolved


def apply_license_gate(
    stage: Path,
    *,
    revision: str,
    license_policy_path: str | Path | None,
    allow_copyleft: frozenset[str] | set[str],
) -> list[tuple[str, str]]:
    """Apply the required license policy after ingest, dedupe, and enrichment."""
    if not license_policy_path:
        raise ValueError(
            "license admission gate requires license_policy_path (fail-closed): "
            "no license policy file was provided"
        )
    from daydream.training.corpus_projection.license import (  # noqa: PLC0415  # local: avoid import cycle at module load
        load_license_policy,
    )

    policy, _digest = load_license_policy(license_policy_path)
    curated = _pre_identity_dir(stage, str(revision))
    ledger_path = _dedupe_dir(stage, curated.name) / "dedupe.jsonl"
    rejected: list[tuple[str, str]] = []
    unresolved = _repo_commit_unresolved_sessions(stage)
    for derivative, data in _derivative_manifests(stage):
        sid = _admitted_session_id(derivative, data)
        decision = _repo_license_decision(data, policy, allow_copyleft)
        if decision.status != "rejected" or decision.reason_code is None:
            continue
        reason_code = decision.reason_code
        if reason_code == REASON_CODE_LICENSE_EVIDENCE_MISSING and sid in unresolved:
            # Retain enrichment's specific unresolved-commit reason in the ledger.
            reason_code = REASON_CODE_REPO_COMMIT_UNRESOLVED
        _move_dir(derivative, stage / "excluded" / sid)
        _append_dedupe_entry(ledger_path, sid, str(revision), status="excluded", reason_code=reason_code)
        rejected.append((sid, reason_code))
    if rejected:
        rebuild_index(stage)  # excluded derivatives leave the staging index immediately
    return rejected


def _staging_local_source_path(raw: Any, stage: Path) -> str | None:
    """Rewrite an embedded ``source_path`` to a staging-local value or ``None``."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw)
    if not p.is_absolute():
        return raw
    try:
        p.relative_to(stage)
    except ValueError:
        return None
    return raw


def rebuild_index(stage: Path) -> None:
    """Index admitted derivatives with credential-free URLs and staging-local paths."""
    for derivative, data in _derivative_manifests(stage):
        kwargs = manifest_index_fields(data)
        # A hydrated bundle never attests the local executable that is rebuilding it.
        kwargs.pop("daydream", None)
        _has_url, slug, canonical = _manifest_remote_fields(data)
        kwargs["repo_slug"], kwargs["remote_url"] = slug, canonical
        kwargs["source_path"] = _staging_local_source_path(_read_manifest_field(data, "source_path"), stage)
        kwargs["archive_path"] = str(derivative)
        upsert_run(stage, kwargs)


def build_resolution_map(
    stage: Path, *, repo_commits: Mapping[str, str],
) -> dict[str, Any]:
    """Group admitted sessions by trusted repo slug and resolved full repository SHA."""
    cmap: dict[str, Any] = {}
    unavailable: list[str] = []
    indexed: set[str] = set()
    for row in query_runs(stage):
        indexed.add(str(row["session_id"]))
        slug = row.get("repo_slug")
        if not slug:
            unavailable.append(str(row["session_id"]))
            continue
        commit = repo_commits.get(str(slug))
        if not isinstance(commit, str) or not _FULL_SHA_RE.fullmatch(commit):
            # No resolved Git repository commit for this slug: reported, never
            # a fabricated revision (the Hub revision is not a repo commit).
            unavailable.append(str(row["session_id"]))
            continue
        entry = cmap.setdefault(
            str(slug), {"repo_slug": str(slug), "pinned_sha": commit, "session_ids": []}
        )
        entry["session_ids"].append(str(row["session_id"]))
    # Bundles rejected at admission for a non-allowlisted host never reached the
    # index; they are still reported under "unavailable" (no raw-URL fallback).
    for manifest_path in sorted(stage.glob("downloads/*/bundles/*/manifest.json")):
        data = _read_manifest_dict(manifest_path.parent)
        if data is None:
            continue
        session_id = str(data.get("session_id") or manifest_path.parent.name)
        if session_id in indexed:
            continue
        _has_url, slug, _canonical = _manifest_remote_fields(data)
        if slug is None and session_id not in unavailable:
            unavailable.append(session_id)
    if unavailable:
        cmap["unavailable"] = sorted(unavailable)
    return cmap


def _curated_dir(stage: Path, source_commit: str, binding: dict[str, Any] | None = None) -> Path:
    """Curated output prefix derived from the source and optional post-gate binding."""
    if binding is not None:
        cid = hydrate_rules.derive_curation_id(
            source_commit,
            str(binding["policy_digest"]),
            str(binding["policy_version"]),
            binding["allow_copyleft"],
            str(binding["exclusions_digest"]),
            str(binding["decisions_digest"]),
            str(binding["distribution_digest"]),
        )
    else:
        cid = hydrate_rules.derive_pre_identity_curation_id(
            source_commit,
            hydrate_rules.SANITIZER_VERSION,
            hydrate_rules.HYDRATION_INDEX_SCHEMA_VERSION,
            hydrate_rules.ADMISSION_POLICY_VERSION,
        )
    return stage / "curated" / cid


def _pre_identity_dir(stage: Path, source_commit: str) -> Path:
    """Source-scoped staging key for dedupe, restamping, and gate rejections."""
    return _curated_dir(stage, source_commit)


def _exclusions_digest() -> str:
    """sha256 over the sorted stable exclusion-codes string of the pinned C5
    exclusion list (``training/schema/exclusion.txt`` bytes) — a bound identity
    input, so editing the list changes the curation id."""
    codes = sorted(
        line.strip() for line in
        EXCLUSION_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    canonical = "".join(f"{code}\n" for code in codes)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _policy_binding(
    stage: Path,
    source_commit: str,
    policy: Any,
    policy_digest: str,
    allow_copyleft: frozenset[str] | set[str],
) -> dict[str, Any]:
    """Bind post-gate decisions and pinned policy inputs to canonical digests."""
    decisions: dict[str, tuple[str, str | None, str | None]] = {}
    for derivative, data in _derivative_manifests(stage):
        decision = _repo_license_decision(data, policy, allow_copyleft)
        decisions[str(decision.repo_slug)] = (
            str(decision.status), decision.reason_code, decision.spdx_id,
        )
    # Bind rejected repos too: changed gate reasons must change identity.
    # Preserve the recorded reason, which may be more specific than a fresh
    # policy resolution. Fixture/ingest exclusions were never license decisions.
    recorded_excluded = _DedupeLedger.load(
        _dedupe_dir(stage, _pre_identity_dir(stage, str(source_commit)).name) / "dedupe.jsonl"
    ).latest
    for derivative, data in _derivative_manifests(stage, "excluded"):
        sid = str(data.get("session_id") or derivative.name)
        entry = recorded_excluded.get(sid) or {}
        code = entry.get("reason_code")
        if code not in _LICENSE_REASON_CODES:
            continue  # never a license-gate decision, never in the digest
        decision = _repo_license_decision(data, policy, allow_copyleft)
        decisions[str(decision.repo_slug)] = (
            "rejected", str(code), decision.spdx_id,
        )
    decision_lines = sorted(
        f"{slug}\t{status}\t{reason_code or ''}\t{spdx_id or ''}\n"
        for slug, (status, reason_code, spdx_id) in decisions.items()
    )
    decisions_digest = hashlib.sha256("".join(decision_lines).encode()).hexdigest()
    distribution = Counter(spdx for (_, __, spdx) in decisions.values() if spdx)
    distribution_lines = sorted(
        f"{spdx}\t{count}\n" for spdx, count in distribution.items()
    )
    distribution_digest = hashlib.sha256("".join(distribution_lines).encode()).hexdigest()
    return {
        "policy_digest": str(policy_digest),
        "policy_version": str(policy.policy_version),
        "allow_copyleft": set(allow_copyleft),
        "exclusions_digest": _exclusions_digest(),
        "decisions_digest": decisions_digest,
        "distribution_digest": distribution_digest,
    }


def resolve_curation_identity(
    stage: Path,
    *,
    source_commit: str,
    license_policy_path: str | Path | None,
    allow_copyleft: frozenset[str] | set[str],
) -> dict[str, Any]:
    """Derive the v2 curation id after the license gate from its policy-bound evidence."""
    if not license_policy_path:
        raise HydrationError(
            "curation identity requires license_policy_path (fail-closed): "
            "no license policy file was provided"
        )
    from daydream.training.corpus_projection.license import (  # noqa: PLC0415  # local: avoid import cycle
        load_license_policy,
    )

    policy, policy_digest = load_license_policy(license_policy_path)
    binding = _policy_binding(stage, str(source_commit), policy, policy_digest, allow_copyleft)
    binding["curation_id"] = _curated_dir(stage, str(source_commit), binding=binding).name
    return binding


def _dedupe_dir(stage: Path, curation_id: str) -> Path:
    """Private ledger and admitted baselines, outside every curated upload prefix."""
    return stage / "_dedupe" / curation_id


def _append_dedupe_entry(
    path: Path, sid: str, revision: str, *, status: str = "admitted",
    reason_code: str | None = None, digest: str | None = None,
) -> None:
    """Append one JSONL record to the dedupe ledger (same shape as sanitize progress)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"session_id": sid, "status": status, "reason_code": reason_code,
             "content_digest": digest, "revision": str(revision), "at": now_iso_utc()}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


@dataclass
class _DedupeLedger:
    """Three views of the append-only dedupe ledger."""

    latest: dict[str, dict[str, Any]] = field(default_factory=dict)
    admitted: dict[str, str | None] = field(default_factory=dict)
    ever_admitted: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> _DedupeLedger:
        ledger = cls()
        if not path.is_file():
            return ledger
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if not isinstance(entry, dict) or not entry.get("session_id"):
                    continue
                sid = str(entry["session_id"])
                ledger.latest[sid] = entry
                if entry.get("status") == "admitted":
                    digest = entry.get("content_digest")
                    ledger.admitted[sid] = digest
                    if isinstance(digest, str) and digest:
                        ledger.ever_admitted.setdefault(sid, set()).add(digest)
        except (OSError, ValueError):
            ledger.admitted.clear()
            ledger.ever_admitted.clear()
        return ledger


def restamp_admitted_digests(stage: Path, *, revision: str) -> None:
    """Refresh changed admitted baselines and append their enriched content digests."""
    curated = _pre_identity_dir(stage, revision)
    dedupe_dir = _dedupe_dir(stage, curated.name)
    baseline_root = dedupe_dir / "admitted"
    ledger_path = dedupe_dir / "dedupe.jsonl"
    for derivative in _bundle_dirs(stage / RUNS_DIRNAME):
        sid = derivative.name
        digest = sanitize._derivative_digest(derivative)
        baseline = baseline_root / sid
        if baseline.is_dir() and sanitize._derivative_digest(baseline) == digest:
            continue  # unchanged by enrichment; ledger digest stays authoritative
        if baseline.is_dir():
            shutil.rmtree(baseline)
        shutil.copytree(derivative, baseline)
        _append_dedupe_entry(ledger_path, sid, str(revision), digest=digest)


def _deep_artifact_evidence(derivative: Path, data: dict[str, Any]) -> dict[str, object] | None:
    """Read legacy-status evidence from deep sidecars and derived manifest fields."""
    evidence: dict[str, object] = {}
    deep_dir = derivative / "deep"
    if deep_dir.is_dir():
        for path in sorted(deep_dir.glob("*.json")):
            try:
                evidence[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    for key, value in (
        ("fix_failures", data.get("fix_failures")),
        ("phase_states", data.get("phase_states")),
        ("archive_status", data.get("archive_status")),
    ):
        if key not in evidence and isinstance(value, (dict, str)):
            evidence[key] = value
    return evidence or None


def dedupe_admitted(stage: Path, *, revision: str) -> DedupeResult:
    """Exclude fixtures, revalidate legacy status, then dedupe by session and digest."""
    revision = str(revision)
    curated = _pre_identity_dir(stage, revision)
    dedupe_dir = _dedupe_dir(stage, curated.name)
    ledger_path = dedupe_dir / "dedupe.jsonl"
    baseline_root = dedupe_dir / "admitted"
    # Collision records never replace the last admitted digest or its history.
    ledger = _DedupeLedger.load(ledger_path)
    result = DedupeResult()
    for derivative, data in _derivative_manifests(stage):
        sid = _admitted_session_id(derivative, data)
        try:
            codes = hydrate_rules.fixture_exclusion_codes(derivative)
        except (OSError, ValueError):
            codes = [REASON_CODE_BUNDLE_UNREADABLE]
        if not codes and "pipeline_status" in data:
            verdict = hydrate_rules.legacy_pipeline_status(
                data.get("pipeline_status"), _deep_artifact_evidence(derivative, data),
            )
            if isinstance(verdict, tuple):
                codes = [verdict[1]]
        if codes:
            code = codes[0]
            _move_dir(derivative, stage / "excluded" / sid)
            _append_dedupe_entry(ledger_path, sid, revision, status="excluded", reason_code=code)
            result.excluded.append((sid, code))
            continue

        digest = sanitize._derivative_digest(derivative)
        baseline = baseline_root / sid
        status, reason_code = "admitted", None
        if ledger.admitted.get(sid) not in (None, digest):
            # Pristine pre-enrichment content is an idempotent re-ingest;
            # every other changed digest is a collision. Both restore the
            # published baseline, whose existence is checked before moving bytes.
            reingest = digest in ledger.ever_admitted.get(sid, set())
            if not baseline.is_dir():
                message = (f"cannot restore admitted baseline for {sid}: no admitted baseline exists"
                           if reingest else f"identity collision for {sid} but no admitted baseline exists")
                raise HydrationError(redact_text(message))
            if reingest:
                shutil.rmtree(derivative)
                digest = sanitize._derivative_digest(baseline)
            else:
                _move_dir(derivative, stage / "quarantine" / f"{sid}.conflict")
                status, reason_code = "collision", REASON_CODE_IDENTITY_COLLISION
            shutil.copytree(baseline, stage / RUNS_DIRNAME / sid)
        elif not baseline.exists():
            baseline.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(derivative, baseline)
        _append_dedupe_entry(ledger_path, sid, revision, status=status, reason_code=reason_code, digest=digest)
        if status == "collision":
            result.collisions += 1
            result.collision_ids.append(sid)
        else:
            result.admitted += 1
    rebuild_index(stage)
    return result


def _license_bucket(code: str | None) -> str:
    """Map a license-gate decision to its human admission bucket."""
    if code is None:
        return "admitted"
    try:
        return _LICENSE_BUCKET_BY_CODE[code]
    except KeyError:
        raise ValueError(
            f"license admission summary: {code!r} is not a license-gate "
            "reason code — the summary buckets only partition license decisions"
        ) from None


def admission_summary_buckets(
    entries: Iterable[tuple[str, str | None]],
) -> dict[str, int]:
    """Count license decisions, rejecting codes outside the license gate."""
    buckets: dict[str, int] = dict.fromkeys(_LICENSE_BUCKETS, 0)
    for _sid, code in entries:
        buckets[_license_bucket(code)] += 1
    return buckets


def _license_admission_entries(ledger: Mapping[str, Any]) -> list[tuple[str, str | None]]:
    """Extract every license decision before callers read session manifests."""
    entries: list[tuple[str, str | None]] = [
        (str(item["session_id"]), None) for item in ledger.get("imported", [])
    ]
    for item in ledger.get("rejections", []):
        code = item.get("reason_code")
        if code in _LICENSE_REASON_CODES:
            entries.append((str(item["session_id"]), str(code)))
    return entries


def license_admission_summary(ledger: Mapping[str, Any]) -> dict[str, int]:
    """Count imports and license-gate rejections, excluding ingest/fixture failures that the gate never adjudicated."""
    return admission_summary_buckets(_license_admission_entries(ledger))


def license_admission_by_repo(
    stage: Path, ledger: Mapping[str, Any]
) -> dict[str, dict[str, int]]:
    """Group the license summary population by its manifest repo slug."""
    revision = str(ledger["pinned_revision"])
    entries = _license_admission_entries(ledger)
    by_repo: dict[str, dict[str, int]] = {}
    for sid, code in entries:
        # Imported sessions still live under stage/runs/<sid> (checked first);
        # license-gate rejections were moved to stage/excluded/<sid>.
        slug, _evidence = _session_identity(
            stage, sid, revision, root="excluded", collision=False
        )
        buckets = by_repo.setdefault(slug or "unresolved", dict.fromkeys(_LICENSE_BUCKETS, 0))
        buckets[_license_bucket(code)] += 1
    return by_repo


def build_import_ledger(
    stage: Path,
    *,
    revision: str,
    source_commit: str,
    binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically persist admission accounting under the policy-bound curated prefix."""
    revision = str(revision)
    source_commit = str(source_commit)
    curated = _curated_dir(stage, source_commit, binding=binding)
    dedupe_ledger = _dedupe_dir(stage, _pre_identity_dir(stage, source_commit).name) / "dedupe.jsonl"
    dedupe_state = _DedupeLedger.load(dedupe_ledger)

    ingest_results: list[dict[str, Any]] = []
    ingest_path = stage / "downloads" / revision / "_ingest_results.json"
    if ingest_path.is_file():
        try:
            loaded = json.loads(ingest_path.read_text(encoding="utf-8"))
            ingest_results = list(loaded.get("results", []))
        except (OSError, ValueError, AttributeError):
            ingest_results = []

    candidate_ids = _discovered_session_ids(stage, revision)
    if candidate_ids is None:
        candidate_ids = [str(e["session_id"]) for e in ingest_results]
    if len(ingest_results) != len(candidate_ids):
        raise HydrationError(
            redact_text(
                f"discovery accounting mismatch for revision {revision!r}: "
                f"discovered {len(candidate_ids)} candidate(s), "
                f"ingest produced {len(ingest_results)} result(s)"
            )
        )

    imported: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_session_ids: set[str] = set()
    for raw_result in ingest_results:
        sid = str(raw_result["session_id"])
        if sid in seen_session_ids:
            raise HydrationError(redact_text(f"duplicate ingest result for candidate session {sid!r}"))
        seen_session_ids.add(sid)
        entry = dedupe_state.latest.get(sid) or {}
        reason_code = raw_result.get("reason_code")
        status = raw_result.get("status")
        if status == "admitted":
            # Dedupe exclusions and collisions override the ingest admission.
            destination = {"excluded": excluded, "collision": quarantined}.get(cast(str, entry.get("status")))
            if destination is None:
                imported.append({
                    "session_id": sid,
                    "content_digest": dedupe_state.admitted.get(sid, entry.get("content_digest")),
                })
                continue
            reason_code = entry.get("reason_code")
        elif status == "quarantined":
            destination = quarantined
        else:
            raise HydrationError(redact_text(f"unknown ingest status for candidate {sid!r}"))
        destination.append({"session_id": sid, "reason_code": reason_code})

    rejections = [
        {
            "session_id": str(entry["session_id"]),
            "reason_code": entry.get("reason_code"),
            "content_digest": (dedupe_state.latest.get(str(entry["session_id"]), {}) or {}).get("content_digest"),
        }
        for entry in sorted(quarantined + excluded, key=lambda item: str(item["session_id"]))
    ]
    discovery_block = _download_discovery_block(stage, revision)
    accounted = len(imported) + len(rejections)
    if accounted != len(candidate_ids):
        raise HydrationError(
            redact_text(
                f"discovery accounting mismatch for revision {revision!r}: "
                f"discovered {len(candidate_ids)} candidate(s), admitted {len(imported)}, "
                f"rejected {len(rejections)}"
            )
        )

    ledger: dict[str, Any] = {
        "schema_version": hydrate_rules.HYDRATION_INDEX_SCHEMA_VERSION,
        "pinned_revision": revision,
        "source_commit": source_commit,
        "curation_id": curated.name,
        "generated_at": now_iso_utc(),
        "imported": imported,
        "quarantined": sorted(quarantined, key=lambda x: x["session_id"]),
        "excluded": sorted(excluded, key=lambda x: x["session_id"]),
        "rejections": rejections,
        "tallies": {
            "discovered": len(candidate_ids),
            "run_shaped_manifests": int(
                discovery_block.get("run_shaped_manifests", len(candidate_ids))
            ),
            "incomplete_manifests": [
                str(item) for item in discovery_block.get("incomplete_manifests", [])
            ],
            "imported": len(imported),
            "quarantined": len(quarantined),
            "excluded": len(excluded),
            "rejections": len(rejections),
            "accounted": accounted,
        },
    }
    atomic_write_json(curated / "import-ledger.json", ledger)
    return ledger
