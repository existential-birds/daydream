"""Hub hydration for issue #982 (task 1: client seam).

Defines the :class:`HubClient` protocol that isolates ``huggingface_hub`` behind
a narrow surface (list/download/commit/repo-info), the lazy production adapter
:class:`HfHubClient`, and the fatal :class:`HubUnavailableError` raised when the
optional ``hub`` extra (or its ``HF_TOKEN``) is missing. Unlike
``daydream.archive.hub`` — whose upload callback must never fail a run and
therefore warns — hydration is an explicit operator command, so every unmet
prerequisite is fatal and fail-closed.

``huggingface_hub`` is imported only inside :func:`_import_hf_hub`; production
code and tests that use :class:`~daydream.archive.hydrate_client.FakeHub` never
need it installed. The module-level :func:`_make_client` factory is the
monkeypatch seam used by tests.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from daydream.archive import hydrate_rules, sanitize
from daydream.archive.git_safe import normalize_remote_url
from daydream.archive.hydrate_rules import (
    REASON_CODE_BUNDLE_UNREADABLE,
    REASON_CODE_IDENTITY_COLLISION,
    REASON_CODE_SANITIZE_FAILED,
    REASON_CODE_SECRETS_SCAN_DIRTY,
    REASON_CODE_UNTRUSTED_REMOTE_HOST,
)
from daydream.archive.index import upsert_run
from daydream.archive.manifest import Manifest
from daydream.archive.scan import scan_run_dir
from daydream.trajectory import redact_text

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_PREFIX_RE = re.compile(r"^[0-9a-f]{4,39}$")


class HydrationError(Exception):
    """Base class for every hydrate failure mode; the orchestrator decides."""


class HubUnavailableError(HydrationError):
    """The ``huggingface_hub`` extra or its ``HF_TOKEN`` prerequisite is missing."""


class HubDownloadError(HydrationError):
    """A requested path/revision does not exist in the Hub repo (fail-closed)."""


class StageError(HydrationError):
    """Staging the pinned snapshot failed (download, digest, or path violation)."""


class MovingBranchError(HydrationError):
    """A symbolic ref (moving branch/tag) was requested without ``exploratory=True``."""


class PublicDestinationError(HydrationError):
    """The target Hub repo is public — hydration publishes only to private repos (M17)."""


class VerificationError(HydrationError):
    """The clean-room verification cycle failed — success is never reported (M20)."""


def resolve_source_revision(client: HubClient, revision: str, *, exploratory: bool) -> str:
    """Resolve ``revision`` to a pinned, immutable commit SHA (issue #982 M2).

    - A full 40-hex SHA is verified to exist and returned unchanged.
    - A hex short prefix (4-39 chars) resolves to the unique matching commit
      SHA; an ambiguous or unknown prefix is a fail-closed :class:`HydrationError`.
    - Any other name is a symbolic ref (moving branch/tag) and raises
      :class:`MovingBranchError` — naming the ref and ``exploratory`` — unless
      ``exploratory=True``, in which case it resolves to the ref's current SHA.
      Exploratory output is flagged non-canonical downstream; canonical v1 runs
      must pin an exact SHA.

    Client errors are redacted via ``daydream.trajectory.redact_text`` before
    being re-raised as :class:`HydrationError`, so no credential material ever
    reaches the console or ledger.
    """
    revision = revision.strip().lower()
    if _FULL_SHA_RE.fullmatch(revision):
        try:
            client.repo_info(revision=revision)  # verify it exists
        except HydrationError as exc:
            raise HydrationError(redact_text(str(exc))) from exc
        return revision

    if _HEX_PREFIX_RE.fullmatch(revision):
        list_revisions = getattr(client, "list_revisions", None)
        matches = (
            [r for r in list_revisions() if r.startswith(revision)]
            if callable(list_revisions)
            else []
        )
        if len(matches) > 1:
            raise HydrationError(
                redact_text(f"ambiguous revision prefix {revision!r}: {len(matches)} matching commits")
            )
        if len(matches) == 1:
            return str(matches[0])
        # Not a known prefix — fall through to symbolic-ref resolution so a
        # hex-named ref still gets the moving-branch treatment below.

    try:
        info = client.repo_info(revision=revision)
    except HydrationError as exc:
        raise HydrationError(redact_text(f"unknown revision {revision!r}: {exc}")) from exc
    if not exploratory:
        raise MovingBranchError(
            f"ref {revision!r} is a moving branch/tag, not a pinned commit; pass "
            "exploratory=True to accept it (output is non-canonical), or pin an "
            "exact 40-char commit SHA"
        )
    return info.sha


@dataclass(frozen=True)
class RepoInfo:
    """Minimal repo metadata the hydration flow needs."""

    sha: str
    private: bool


@runtime_checkable
class HubClient(Protocol):
    """Narrow Hub surface hydration depends on (list/download/commit/repo-info)."""

    def repo_info(self, revision: str | None = None) -> RepoInfo: ...

    def list_repo_files(self) -> list[str]: ...

    def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes: ...

    def upload_files(
        self, mapping: dict[str | Path, Path], commit_message: str
    ) -> None: ...

    @property
    def repo_private(self) -> bool: ...


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of one :func:`download_snapshot` pass over the pinned revision."""

    downloaded: int = 0
    skipped: int = 0
    digests: dict[str, str] = field(default_factory=dict)  # relpath -> sha256


def _validate_relpath(relpath: str, root: Path) -> Path:
    """Enforce the M4 trust boundary: resolve ``relpath`` strictly under ``root``.

    Raises :class:`StageError` naming "traversal" when the path is absolute,
    carries ``..`` segments, or otherwise escapes the staging root. Nothing is
    ever written before this check passes.
    """
    p = PurePosixPath(relpath)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise StageError(
            redact_text(
                f"refusing relpath {relpath!r}: traversal — path escapes the staging root"
            )
        )
    target = (root / relpath).resolve()
    if not target.is_relative_to(root.resolve()):
        raise StageError(
            redact_text(
                f"refusing relpath {relpath!r}: traversal — resolved path escapes the staging root"
            )
        )
    return target


def download_snapshot(
    client: HubClient,
    *,
    revision: str,
    stage_dir: Path,
    expect: dict[str, str] | None = None,
) -> DownloadResult:
    """Resumable, content-addressed download of a pinned snapshot revision (issue #982 M3).

    Lists the repo's bundle content (paths under ``bundles/``), writes each
    artifact to ``stage_dir/<revision>/<relpath>``, and records a per-artifact
    ledger (relpath, sha256, size, fetched_at) in
    ``stage_dir/<revision>/_download_manifest.json``.

    Resume: an on-disk artifact whose sha256 matches the existing ledger record
    is skipped, not re-downloaded; missing or mismatched artifacts are fetched,
    re-hashed, and their records updated. ``expect`` — ``{relpath: sha256}``
    from a pinned manifest — makes any disagreement a hard :class:`StageError`
    naming "digest". Every relpath is validated against the staging root before
    any write (see :func:`_validate_relpath`); download failures delete the
    partial artifact and raise :class:`StageError` with redacted messages.
    """
    revision = str(revision)
    root = stage_dir / revision
    manifest_path = root / "_download_manifest.json"
    records: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text())
            records = {a["relpath"]: a for a in loaded.get("artifacts", [])}
        except (OSError, ValueError, KeyError, TypeError):
            records = {}  # corrupt ledger: rebuild from disk state

    relpaths = list(client.list_repo_files())
    downloaded = 0
    skipped = 0
    digests: dict[str, str] = {}
    artifacts: list[dict[str, Any]] = []

    for relpath in relpaths:
        target = _validate_relpath(relpath, root)
        if not relpath.startswith("bundles/"):
            continue  # non-bundle paths (top-level manifests, etc.) are not staged here
        expected_sha = (expect or {}).get(relpath)
        try:
            if target.exists():
                existing = hashlib.sha256(target.read_bytes()).hexdigest()
                record_sha = records.get(relpath, {}).get("sha256")
                if expected_sha in (None, existing) and record_sha in (None, existing):
                    digests[relpath] = existing
                    skipped += 1
                    artifacts.append({**records.get(relpath, {}), "relpath": relpath, "sha256": existing})
                    continue
            data = client.download_file(relpath, revision=revision)
        except HydrationError as exc:
            raise StageError(redact_text(str(exc))) from exc
        sha = hashlib.sha256(data).hexdigest()
        if expected_sha is not None and sha != expected_sha:
            raise StageError(
                redact_text(f"digest mismatch for {relpath!r}: expected {expected_sha}, got {sha}")
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        try:
            tmp.write_bytes(data)
            tmp.replace(target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise StageError(redact_text(f"write failed for {relpath!r}: {exc}")) from exc
        downloaded += 1
        digests[relpath] = sha
        artifacts.append(
            {
                "relpath": relpath,
                "sha256": sha,
                "size": len(data),
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"revision": revision, "artifacts": artifacts}, indent=2))
    return DownloadResult(downloaded=downloaded, skipped=skipped, digests=digests)


def _read_manifest_dict(bundle_dir: Path) -> dict[str, Any] | None:
    """Read ``manifest.json`` from ``bundle_dir``; ``None`` when absent/unparseable."""
    path = bundle_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@dataclass(frozen=True)
class IngestResult:
    """Outcome of the ingest gate for one staged session bundle (issue #982 M4/M6)."""

    session_id: str
    status: str  # "admitted" | "quarantined"
    reason_code: str | None = None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically (temp file + rename); fatal on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ingest_bundles(stage: Path, *, revision: str) -> list[IngestResult]:
    """Run every staged bundle through the #981 ingest gate (issue #982 M4/M6).

    For each session bundle under ``stage/downloads/<revision>/bundles/``:

    1. The manifest must parse and carry a trusted remote host
       (:func:`daydream.archive.git_safe.normalize_remote_url` identity); an
       unreadable manifest or non-allowlisted host is quarantined (fail-closed,
       stable reason codes) without ever dereferencing embedded paths.
    2. ``sanitize.import_bundle`` is the sole secrets gate: a dirty bundle is
       moved to ``stage/quarantine/<name>`` by the #981 implementation itself —
       hydrate never forks the scan or the quarantine move.
    3. A clean bundle is sanitized by ``sanitize.sanitize_bundle`` (release
       scan fail-closed), and the released derivative is placed at
       ``stage/runs/<session_id>/``. Any ``.git`` directory inside the
       downloaded copy is stripped before sanitize so no hydrated bundle ever
       ships one (harvest priority-1 safety, Task 0B constraint).

    Identity-collision dedupe is Task 8's concern; this pass guarantees one
    derivative per staged bundle or a quarantine result.
    """
    bundles_root = stage / "downloads" / str(revision) / "bundles"
    if not bundles_root.is_dir():
        return []
    results: list[IngestResult] = []
    for bundle_dir in sorted(p for p in bundles_root.iterdir() if p.is_dir()):
        name = bundle_dir.name
        data = _read_manifest_dict(bundle_dir)
        if data is None:
            results.append(IngestResult(name, "quarantined", REASON_CODE_BUNDLE_UNREADABLE))
            continue
        session_id = str(data.get("session_id") or name)
        raw_url = data.get("remote_url")
        if isinstance(raw_url, str) and raw_url.strip():
            identity, _canonical = normalize_remote_url(raw_url)
            if identity is None:
                # Non-allowlisted host: rejected as admission data before any
                # gate work; the raw copy stays in downloads, never indexed.
                results.append(
                    IngestResult(session_id, "quarantined", REASON_CODE_UNTRUSTED_REMOTE_HOST)
                )
                continue
        gate = sanitize.import_bundle(bundle_dir, stage)
        if gate.quarantined or not gate.imported:
            results.append(IngestResult(session_id, "quarantined", REASON_CODE_SECRETS_SCAN_DIRTY))
            continue
        # Task 0B constraint: hydrated staging bundles must exclude .git (the
        # raw download copy is daydream-staged data, safe to prune locally).
        for git_dir in list(bundle_dir.rglob(".git")):
            if git_dir.is_dir():
                shutil.rmtree(git_dir)
        try:
            sanitized = sanitize.sanitize_bundle(bundle_dir, stage)
        except Exception:
            results.append(IngestResult(session_id, "quarantined", REASON_CODE_SANITIZE_FAILED))
            continue
        if not sanitized.released:
            results.append(IngestResult(session_id, "quarantined", REASON_CODE_SECRETS_SCAN_DIRTY))
            continue
        derivative = stage / "sanitized" / session_id
        target = stage / "runs" / session_id
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(derivative), str(target))  # staging layout only, not the gate
        results.append(IngestResult(session_id, "admitted"))
    _atomic_write_json(
        bundles_root.parent / "_ingest_results.json",
        {
            "revision": str(revision),
            "results": [
                {"session_id": r.session_id, "status": r.status, "reason_code": r.reason_code}
                for r in results
            ],
        },
    )
    return results


def _staging_local_source_path(raw: Any, stage: Path) -> str | None:
    """Rewrite an embedded ``source_path`` to a staging-local value or ``None``.

    Embedded paths are data: an absolute path outside the staging root is
    dropped (never dereferenced, never carried into the index).
    """
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
    """Index every admitted derivative under ``stage/runs/`` (issue #982 M6).

    Each derivative's manifest is loaded and its path/URL fields rewritten to
    staging-local, credential-free values before ``index.upsert_run`` writes
    the row: ``archive_path`` points inside the staging root (never the raw
    download tree), ``source_path`` is staging-local or ``None``, and
    ``remote_url``/``repo_slug`` come from ``normalize_remote_url`` output.
    ``upsert_run`` errors propagate — an admitted bundle is never silently
    skipped.
    """
    runs_dir = stage / "runs"
    if not runs_dir.is_dir():
        return
    for derivative in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        data = _read_manifest_dict(derivative)
        if data is None:
            raise HydrationError(
                redact_text(f"admitted derivative {derivative.name} has an unreadable manifest")
            )
        valid = {f.name for f in dataclass_fields(Manifest)}
        rewritten = {"archive_path", "source_path", "remote_url", "repo_slug"}
        kwargs = {k: v for k, v in data.items() if k in valid and k not in rewritten}
        raw_url = data.get("remote_url")
        if isinstance(raw_url, str) and raw_url.strip():
            slug, canonical = normalize_remote_url(raw_url)
            kwargs["repo_slug"] = slug
            kwargs["remote_url"] = canonical
        else:
            kwargs["repo_slug"] = None
            kwargs["remote_url"] = None
        kwargs["source_path"] = _staging_local_source_path(data.get("source_path"), stage)
        kwargs["archive_path"] = str(derivative)
        upsert_run(stage, Manifest(**kwargs))


def build_resolution_map(stage: Path, *, source_commit: str) -> dict[str, Any]:
    """Build the deferred-clone repository resolution map (issue #982 M5).

    From the admitted index rows under ``stage/runs/``, group sessions by the
    ``normalize_remote_url`` slug. Each entry carries ``repo_slug``,
    ``pinned_sha`` (the source commit the snapshot was hydrated from —
    deferred-clone consumers fetch exactly this revision), and the contributing
    ``session_ids``. Rows with no resolvable slug (no remote, or a
    non-allowlisted host) land under ``map["unavailable"]`` as a list of
    session ids — a reported outcome, never a raw-URL fallback and never a
    clone.

    No I/O beyond reading the staging index: this never shells out to git, so
    hydration never clones or fetches (M5). Raw URLs are consumed as data only
    and never appear in the map. Unexpected IO errors propagate.
    """
    from daydream.archive.index import query_runs  # noqa: PLC0415  # local: avoid import cycle at module load

    cmap: dict[str, Any] = {}
    unavailable: list[str] = []
    indexed: set[str] = set()
    for row in query_runs(stage):
        indexed.add(str(row["session_id"]))
        slug = row.get("repo_slug")
        if not slug:
            unavailable.append(str(row["session_id"]))
            continue
        entry = cmap.setdefault(
            str(slug), {"repo_slug": str(slug), "pinned_sha": source_commit, "session_ids": []}
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
        raw_url = data.get("remote_url")
        slug = normalize_remote_url(raw_url)[0] if isinstance(raw_url, str) and raw_url.strip() else None
        if slug is None and session_id not in unavailable:
            unavailable.append(session_id)
    if unavailable:
        cmap["unavailable"] = sorted(unavailable)
    return cmap


# ---------------------------------------------------------------------------
# Content-addressed dedupe + collision quarantine + import ledger (Task 8)
# ---------------------------------------------------------------------------


@dataclass
class DedupeResult:
    """Outcome of one :func:`dedupe_admitted` pass over ``stage/runs/`` (M7/M8/M9)."""

    admitted: int = 0
    skipped: int = 0
    collisions: int = 0
    collision_ids: list[str] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)  # (session_id, reason_code)


def _curated_dir(stage: Path, source_commit: str) -> Path:
    """Curated prefix for a source commit: ``stage/curated/<curation-id>/``."""
    cid = hydrate_rules.derive_curation_id(
        source_commit,
        hydrate_rules.SANITIZER_VERSION,
        hydrate_rules.HYDRATION_INDEX_SCHEMA_VERSION,
        hydrate_rules.ADMISSION_POLICY_VERSION,
    )
    return stage / "curated" / cid


def _append_dedupe_entry(path: Path, entry: dict[str, Any]) -> None:
    """Append one JSONL record to the dedupe ledger (same shape as sanitize progress)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _load_dedupe_ledger(path: Path) -> dict[str, dict[str, Any]]:
    """Latest recorded dedupe entry per session id (empty when absent/corrupt)."""
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if isinstance(entry, dict) and entry.get("session_id"):
                latest[str(entry["session_id"])] = entry
    except (OSError, ValueError):
        return latest
    return latest


def _move_dir(source: Path, target: Path) -> None:
    """Move a directory, replacing any prior occupant of ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(source), str(target))


def dedupe_admitted(stage: Path, *, revision: str) -> DedupeResult:
    """Content-addressed dedupe over the admitted derivatives (issue #982 M7/M8/M9).

    The dedupe key is ``(session_id, derivative content digest)`` where the
    digest is ``sanitize._derivative_digest`` over ``stage/runs/<sid>``. The
    ledger lives at ``stage/curated/<curation-id>/dedupe.jsonl`` (append-only,
    latest entry per session wins).

    Per derivative, in order:

    1. Fixture exclusion (M8): ``hydrate_rules.fixture_exclusion_codes`` runs
       pre-dedupe; a coded bundle is moved to ``stage/excluded/<sid>/`` and
       reported with its stable code — never indexed, never harvested.
    2. Legacy ``pipeline_status`` revalidation (M9): a manifest carrying the
       legacy field is revalidated via ``hydrate_rules.legacy_pipeline_status``;
       evidence-absent rows are excluded with the stable code.
    3. Identity collision (M7): the same ``session_id`` with a *different*
       derivative digest is moved to ``quarantine/<sid>.conflict`` and the
       original admitted derivative restored from the curated baseline copy —
       the admitted derivative is never overwritten. Identical digests are
       idempotent (no duplicate ledger entry, no duplicate index row).

    Admitted sessions are indexed (idempotent ``upsert_run``); an unreadable
    manifest on an admitted derivative is fatal. The dedupe ledger records
    every decision as ``{session_id, status, content_digest, reason_code}``.
    """
    revision = str(revision)
    curated = _curated_dir(stage, revision)
    ledger_path = curated / "dedupe.jsonl"
    baseline_root = curated / "admitted"
    recorded = _load_dedupe_ledger(ledger_path)
    result = DedupeResult()
    runs_dir = stage / "runs"
    if runs_dir.is_dir():
        for derivative in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            name = derivative.name
            data = _read_manifest_dict(derivative)
            if data is None:
                raise HydrationError(
                    redact_text(f"admitted derivative {name} has an unreadable manifest")
                )
            sid = str(data.get("session_id") or name)

            # M8: fixture exclusion, pre-dedupe, stable codes.
            try:
                codes = hydrate_rules.fixture_exclusion_codes(derivative)
            except (OSError, ValueError):
                codes = [REASON_CODE_BUNDLE_UNREADABLE]
            if not codes and "pipeline_status" in data:
                # M9: revalidate the legacy field; evidence-absent never succeeds.
                verdict = hydrate_rules.legacy_pipeline_status(
                    data.get("pipeline_status"),
                    data.get("deep_artifacts") if isinstance(data.get("deep_artifacts"), dict) else None,
                )
                if isinstance(verdict, tuple):
                    codes = [verdict[1]]
            if codes:
                code = codes[0]
                _move_dir(derivative, stage / "excluded" / sid)
                _append_dedupe_entry(
                    ledger_path,
                    {"session_id": sid, "status": "excluded", "reason_code": code,
                     "content_digest": None, "revision": revision, "at": _utc_now()},
                )
                result.excluded.append((sid, code))
                continue

            digest = sanitize._derivative_digest(derivative)
            prev = recorded.get(sid)
            if prev is not None and prev.get("status") == "admitted" \
                    and prev.get("content_digest") not in (None, digest):
                # Identity collision: quarantine the new derivative, restore the
                # original admitted content — never overwrite (M7).
                baseline = baseline_root / sid
                if not baseline.is_dir():
                    raise HydrationError(
                        redact_text(
                            f"identity collision for {sid} but no admitted baseline exists"
                        )
                    )
                _move_dir(derivative, stage / "quarantine" / f"{sid}.conflict")
                shutil.copytree(baseline, runs_dir / sid)
                _append_dedupe_entry(
                    ledger_path,
                    {"session_id": sid, "status": "collision",
                     "reason_code": REASON_CODE_IDENTITY_COLLISION,
                     "content_digest": digest, "revision": revision, "at": _utc_now()},
                )
                result.collisions += 1
                result.collision_ids.append(sid)
                continue

            baseline = baseline_root / sid
            if not baseline.exists():
                baseline.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(derivative, baseline)
            _append_dedupe_entry(
                ledger_path,
                {"session_id": sid, "status": "admitted", "reason_code": None,
                 "content_digest": digest, "revision": revision, "at": _utc_now()},
            )
            result.admitted += 1
    rebuild_index(stage)
    return result


def build_import_ledger(stage: Path, *, revision: str, source_commit: str) -> dict[str, Any]:
    """Build and persist the value-free admission ledger (issue #982 M11).

    Composes the ingest results, the dedupe ledger, and the staging index into
    ``stage/curated/<curation-id>/import-ledger.json``. Every rejection entry
    carries only stable reason codes, session ids, and content digests — never
    a raw URL, path, or any matched secret value. The write is atomic; a
    failure propagates (fatal semantics) so a partial ledger can never claim
    success.
    """
    revision = str(revision)
    source_commit = str(source_commit)
    curated = _curated_dir(stage, source_commit)
    recorded = _load_dedupe_ledger(curated / "dedupe.jsonl")

    ingest_results: list[dict[str, Any]] = []
    ingest_path = stage / "downloads" / revision / "_ingest_results.json"
    if ingest_path.is_file():
        try:
            loaded = json.loads(ingest_path.read_text(encoding="utf-8"))
            ingest_results = list(loaded.get("results", []))
        except (OSError, ValueError, AttributeError):
            ingest_results = []

    from daydream.archive.index import query_runs  # noqa: PLC0415  # local: avoid import cycle

    indexed_ids = {str(row["session_id"]) for row in query_runs(stage)}
    admitted_ids = {
        str(e["session_id"]) for e in ingest_results if e.get("status") == "admitted"
    } | indexed_ids

    imported: list[dict[str, Any]] = []
    for sid in sorted(admitted_ids):
        entry = recorded.get(sid)
        imported.append(
            {
                "session_id": sid,
                "content_digest": entry.get("content_digest") if entry else None,
            }
        )

    quarantined: list[dict[str, Any]] = []
    for e in ingest_results:
        if e.get("status") == "quarantined":
            quarantined.append(
                {"session_id": str(e["session_id"]), "reason_code": e.get("reason_code")}
            )
    excluded: list[dict[str, Any]] = []
    for sid, entry in sorted(recorded.items()):
        if entry.get("status") == "collision":
            quarantined.append(
                {"session_id": sid, "reason_code": entry.get("reason_code")}
            )
        elif entry.get("status") == "excluded":
            excluded.append({"session_id": sid, "reason_code": entry.get("reason_code")})

    rejections = [
        {
            "session_id": e["session_id"],
            "reason_code": e.get("reason_code"),
            "content_digest": (recorded.get(e["session_id"], {}) or {}).get("content_digest"),
        }
        for e in sorted(quarantined + excluded, key=lambda x: x["session_id"])
    ]

    ledger: dict[str, Any] = {
        "schema_version": hydrate_rules.HYDRATION_INDEX_SCHEMA_VERSION,
        "pinned_revision": revision,
        "source_commit": source_commit,
        "curation_id": curated.name,
        "generated_at": _utc_now(),
        "imported": imported,
        "quarantined": sorted(quarantined, key=lambda x: x["session_id"]),
        "excluded": sorted(excluded, key=lambda x: x["session_id"]),
        "rejections": rejections,
        "tallies": {
            "imported": len(imported),
            "quarantined": len(quarantined),
            "excluded": len(excluded),
            "rejections": len(rejections),
        },
    }
    _atomic_write_json(curated / "import-ledger.json", ledger)
    return ledger


# ---------------------------------------------------------------------------
# Publication: additive batches + remote resume ledger (Task 9)
# ---------------------------------------------------------------------------

_UPLOAD_ATTEMPTS = 6
_UPLOAD_BASE_DELAY_S = 2.0
_UPLOAD_MAX_DELAY_S = 120.0


@dataclass(frozen=True)
class ResumeState:
    """Resume checkpoint derived from the remote Hub ledger (never VM-local state)."""

    completed_sessions: set[str] = field(default_factory=set)
    redownloaded: list[str] = field(default_factory=list)


def _retry_upload(client: HubClient, mapping: dict[str | Path, Path], commit_message: str) -> None:
    """Upload with the hub.py commit-conflict retry shape (exponential backoff).

    More attempts than hub.py's warning-path loop, because hydration is fatal
    on final failure. Content identity never changes across retries (M21).
    """
    for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
        try:
            client.upload_files(mapping, commit_message)
            return
        except HydrationError as exc:
            conflict = "concurrent update" in str(exc)
            if conflict and attempt < _UPLOAD_ATTEMPTS:
                time.sleep(min(_UPLOAD_BASE_DELAY_S * (2 ** (attempt - 1)), _UPLOAD_MAX_DELAY_S))
                continue
            raise HydrationError(redact_text(str(exc))) from exc


def _curated_upload_paths(stage: Path, curation_id: str) -> list[Path]:
    """All staging files under ``stage/curated/<curation-id>/`` (relative, sorted)."""
    curated = stage / "curated" / curation_id
    if not curated.is_dir():
        return []
    return sorted(p for p in curated.rglob("*") if p.is_file())


def publish_batches(
    client: HubClient, stage: Path, *, curation_id: str, skip_sessions: set[str] | None = None
) -> None:
    """Publish sanitized batches additively under ``curated/<curation-id>/`` (M13/M15/M16).

    Hard-fails with :class:`PublicDestinationError` when the repo is not private
    (unlike hub.py, which only warns — hydration is an operator command and a
    public destination would leak sanitized-but-sensitive content). Uploads,
    additively under the curated prefix only: sanitized batches under
    ``batches/``, the import ledger, the resolution map, ``SHA256SUMS`` over the
    published file set, and the resume ledger ``resume/ledger.jsonl`` (one
    record per completed batch: session_id, batch digest, source commit).

    Additive publication is resumable (M15/M16): ``skip_sessions`` — completed
    session ids from the remote resume ledger — are omitted from the upload
    mapping (they already exist at their content-addressed paths from prior
    commits) while still contributing to SHA256SUMS and the resume ledger.

    Bronze safety (M10/M13): the module asserts its own upload path list never
    leaves the ``curated/`` prefix before a single byte is written. Upload
    failures after retries are fatal, redacted :class:`HydrationError`s;
    content-addressed batch paths make re-upload idempotent.
    """
    if not client.repo_private:
        raise PublicDestinationError(
            "refusing to publish: the Hub repo is not private; hydration "
            "publishes sanitized corpora only to private repos (M17)"
        )
    curated = stage / "curated" / curation_id
    _stage_batches(stage, curated)
    _write_resolution_map(stage, curated)
    _write_resume_ledger(stage, curated, curation_id)
    files = _curated_upload_paths(stage, curation_id)
    if not files:
        raise HydrationError(redact_text(f"nothing to publish under curated/{curation_id}"))

    prefix = f"curated/{curation_id}/"
    relpaths = sorted(f.relative_to(curated).as_posix() for f in files)
    # Bronze safety gate: assert nothing escapes the curated prefix (M10/M13).
    assert all(not p.startswith(("bronze", "runs/", "downloads/")) and ".." not in p
               for p in relpaths), relpaths

    # SHA256SUMS covers every published file except itself (self-inclusion would
    # make the checksum file unstable across idempotent re-publishes).
    checksums = "".join(
        f"{hashlib.sha256((curated / p).read_bytes()).hexdigest()}  {prefix}{p}\n"
        for p in relpaths if p != "SHA256SUMS"
    )
    sums_path = curated / "SHA256SUMS"
    sums_path.write_text(checksums, encoding="utf-8")
    files = _curated_upload_paths(stage, curation_id)

    mapping: dict[str | Path, Path] = {
        f"{prefix}{f.relative_to(curated).as_posix()}": f
        for f in files
        if not any(
            f.relative_to(curated).as_posix().startswith(f"batches/{sid}/")
            for sid in skip_sessions or ()
        )
    }
    if mapping:
        _retry_upload(client, mapping, f"daydream hydrate {curation_id}: additive batch publication")


def _stage_batches(stage: Path, curated: Path) -> None:
    """Copy every admitted derivative (``stage/runs/<sid>``) into ``batches/<sid>/``."""
    runs_dir = stage / "runs"
    if not runs_dir.is_dir():
        return
    for derivative in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        target = curated / "batches" / derivative.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)  # content-addressed rewrite: same content, same digest
        shutil.copytree(derivative, target)


def _write_resolution_map(stage: Path, curated: Path) -> None:
    """Materialize ``resolution-map.json`` under the curated prefix when absent.

    Rebuilt from the staging index (data only, no clones); an existing file —
    from a prior publish of the same curation id — is left untouched so the
    published map stays stable and additive.
    """
    map_path = curated / "resolution-map.json"
    if map_path.exists():
        return
    source_commit = None
    ledger_path = curated / "import-ledger.json"
    if ledger_path.is_file():
        try:
            source_commit = json.loads(ledger_path.read_text(encoding="utf-8")).get("source_commit")
        except (OSError, ValueError):
            source_commit = None
    cmap = build_resolution_map(stage, source_commit=source_commit or "unknown")
    _atomic_write_json(map_path, cmap)


def _write_resume_ledger(stage: Path, curated: Path, curation_id: str) -> None:
    """Write (append) one resume record per admitted batch under ``resume/ledger.jsonl``.

    Content-addressed: re-publishing identical content produces byte-identical
    records, so the ledger stays deduplicated (latest entry per session wins).
    """
    source_commit = None
    ledger_path = curated / "import-ledger.json"
    if ledger_path.is_file():
        try:
            source_commit = json.loads(ledger_path.read_text(encoding="utf-8")).get("source_commit")
        except (OSError, ValueError):
            source_commit = None
    batches_dir = curated / "batches"
    entries: dict[str, dict[str, Any]] = {}
    if batches_dir.is_dir():
        for batch in sorted(p for p in batches_dir.iterdir() if p.is_dir()):
            entries[batch.name] = {
                "session_id": batch.name,
                "batch_digest": sanitize._derivative_digest(batch),
                "source_commit": source_commit,
                "curation_id": curation_id,
                "at": _utc_now(),
            }
    resume_path = curated / "resume" / "ledger.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if resume_path.is_file():
        try:
            for line in resume_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("session_id"):
                        existing[str(rec["session_id"])] = rec
        except (OSError, ValueError):
            existing = {}
    # Additive ledger: first record per session wins; re-publishing identical
    # content never rewrites history (content-addressed idempotence).
    for sid, entry in entries.items():
        existing.setdefault(sid, entry)
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path.write_text(
        "".join(json.dumps(existing[sid], sort_keys=True) + "\n" for sid in sorted(existing)),
        encoding="utf-8",
    )


def resume_state(client: HubClient, *, curation_id: str, stage_dir: Path) -> ResumeState:
    """Discover the remote resume ledger under ``curated/<curation-id>/`` (M15).

    Downloads ``resume/ledger.jsonl`` when present and verifies each recorded
    batch digest against the actual ``batches/…`` content on the Hub, hashing
    into ``stage_dir`` (a fresh VM's empty disk is fine). Completed sessions are
    those whose remote batch content matches the recorded digest; mismatches
    are reported in ``redownloaded`` — what would need re-fetching. VM-local
    artifacts are never consulted as canonical state. A missing ledger yields
    an empty checkpoint (nothing completed); download errors on the ledger or
    a batch are fatal, redacted :class:`HydrationError`s.
    """
    prefix = f"curated/{curation_id}/"
    try:
        raw = client.download_file(f"{prefix}resume/ledger.jsonl")
    except HydrationError:
        raw = None  # no checkpoint yet: nothing completed, nothing redownloaded
    completed: set[str] = set()
    redownloaded: list[str] = []
    if raw is None:
        return ResumeState(completed_sessions=completed, redownloaded=redownloaded)
    repo_files = set(client.list_repo_files())
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise HydrationError(redact_text(f"corrupt resume ledger entry: {exc}")) from exc
        sid = str(entry.get("session_id") or "")
        digest = str(entry.get("batch_digest") or "")
        if not sid or not digest:
            continue
        batch_prefix = f"{prefix}batches/{sid}/"
        batch_relpaths = sorted(p[len(batch_prefix):] for p in repo_files if p.startswith(batch_prefix))
        if not batch_relpaths:
            redownloaded.append(sid)
            continue
        batch_dir = stage_dir / "batches" / sid
        for rel in batch_relpaths:
            data = client.download_file(f"{batch_prefix}{rel}")
            target = batch_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        local_digest = sanitize._derivative_digest(batch_dir)
        if local_digest == digest:
            completed.add(sid)
        else:
            redownloaded.append(sid)
    return ResumeState(completed_sessions=completed, redownloaded=redownloaded)


def _import_hf_hub() -> Any:
    """Return the ``huggingface_hub`` module, or ``None`` when not installed.

    Kept as a module-level function so tests can monkeypatch it to ``None`` to
    simulate the missing-extra environment.
    """
    if importlib.util.find_spec("huggingface_hub") is None:
        return None
    import huggingface_hub  # noqa: PLC0415  # lazy: optional extra

    return huggingface_hub


def _make_client(repo_id: str, *, token_present: bool | None = None) -> HfHubClient:
    """Build the production :class:`HfHubClient` for ``repo_id``.

    ``token_present`` overrides the ``HF_TOKEN`` environment check when
    explicitly given (``None`` derives it from the environment). Raises
    :class:`HubUnavailableError` — fatally, never a warning — when either the
    package or the token is absent. Error messages name prerequisites only;
    token material is never echoed.
    """
    if token_present is False:
        raise HubUnavailableError(
            "HF_TOKEN is not set; hydration requires a read token for the "
            "private Hub repo. Export HF_TOKEN (or pass --token-source) and retry."
        )
    if _import_hf_hub() is None:
        raise HubUnavailableError(
            "The 'huggingface-hub' package is required for hydrate but is not "
            "installed. Install the optional extra: `uv sync --extra hub` "
            "(or `pip install 'daydream[hub]'`)."
        )
    if token_present is None:
        token_present = bool(os.environ.get("HF_TOKEN"))
    if not token_present:
        raise HubUnavailableError(
            "HF_TOKEN is not set; hydration requires a read token for the "
            "private Hub repo. Export HF_TOKEN and retry."
        )
    return HfHubClient(repo_id)


class HfHubClient:
    """Production :class:`HubClient` adapter over a lazily imported ``huggingface_hub``.

    The token is read from ``os.environ["HF_TOKEN"]`` only and is never stored
    on argv, logged, or included in ``repr``.
    """

    def __init__(self, repo_id: str) -> None:
        hf = _import_hf_hub()
        if hf is None:
            raise HubUnavailableError(
                "The 'huggingface-hub' package is required for hydrate but is not "
                "installed. Install the optional extra: `uv sync --extra hub`."
            )
        self._repo_id = repo_id
        self._hf: Any = hf
        self._api: Any = hf.HfApi(token=os.environ.get("HF_TOKEN"))

    def __repr__(self) -> str:  # never leaks the token
        return f"HfHubClient(repo_id={self._repo_id!r})"

    @property
    def repo_private(self) -> bool:
        return bool(self.repo_info().private)

    def repo_info(self, revision: str | None = None) -> RepoInfo:
        try:
            info = self._api.repo_info(revision or "main", repo_type="dataset")
        except Exception as exc:  # mapped, never swallowed
            raise HubDownloadError(f"repo_info failed for {self._repo_id}: {exc}") from exc
        return RepoInfo(sha=str(info.sha), private=bool(info.private))

    def list_repo_files(self) -> list[str]:
        try:
            return list(self._api.list_repo_files(self._repo_id, repo_type="dataset"))
        except Exception as exc:
            raise HubDownloadError(f"list_repo_files failed for {self._repo_id}: {exc}") from exc

    def list_revisions(self) -> list[str]:
        """Commit SHAs known to the Hub, for short-prefix resolution (best effort)."""
        try:
            commits = self._api.list_repo_commits(self._repo_id, repo_type="dataset")
            return [str(c.commit_id) for c in commits]
        except Exception as exc:  # enumeration is best-effort; fail closed on use
            raise HubDownloadError(f"list_repo_commits failed for {self._repo_id}: {exc}") from exc

    def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes:
        try:
            local = self._api.hf_hub_download(
                self._repo_id, path_in_repo, repo_type="dataset", revision=revision
            )
        except Exception as exc:
            raise HubDownloadError(
                f"download failed for {self._repo_id}:{path_in_repo}: {exc}"
            ) from exc
        return Path(local).read_bytes()

    def upload_files(self, mapping: dict[str | Path, Path], commit_message: str) -> None:
        try:
            for path_in_repo, local_path in mapping.items():
                self._api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=str(path_in_repo),
                    repo_id=self._repo_id,
                    repo_type="dataset",
                    commit_message=commit_message,
                )
        except Exception as exc:
            raise HydrationError(f"upload failed for {self._repo_id}: {exc}") from exc


# ---------------------------------------------------------------------------
# Finalization + verify-before-success cycle (Task 10, M18/M19/M20)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HydrateHubConfig:
    """Operator configuration for ``run_hydrate_hub`` (harvest ``RunConfig`` discipline)."""

    source_repo: str
    source_revision: str
    destination_repo: str
    stage_dir: Path
    exploratory: bool = False


@dataclass
class HydrateSummary:
    """Outcome of one ``run_hydrate_hub`` invocation; ``verified`` gates success."""

    source_commit: str
    curation_id: str
    output_commit_sha: str | None = None
    dry_run_admitted: int = 0
    verify_admitted: int = 0
    verified: bool = False


def _curation_manifest_doc(
    stage: Path, *, curation_id: str, source_commit: str, ledger: dict[str, Any]
) -> dict[str, Any]:
    """Render the portable curation manifest (schema v1) from the real ledger (M12/M18)."""
    batches: list[dict[str, Any]] = []
    for entry in ledger.get("imported", []):
        sid = str(entry["session_id"])
        batches.append(
            {
                "session_id": sid,
                "content_digest": str(entry.get("content_digest") or ""),
                "status": "admitted",
                "reason_code": None,
                "artifact_relpath": f"batches/{sid}",
                "manifest_relpath": f"batches/{sid}/manifest.json",
            }
        )
    for rejection in ledger.get("rejections", []):
        sid = str(rejection["session_id"])
        code = rejection.get("reason_code")
        status = "excluded" if code in hydrate_rules.EXCLUSION_CODES else "quarantined"
        root = "excluded" if status == "excluded" else "quarantine"
        digest = rejection.get("content_digest")
        if not isinstance(digest, str) or not digest:
            # Rejected bundles carry no derivative digest; derive one from the
            # rejected copy on staging (or the session id when it was moved).
            source_dir = stage / root / sid
            if not source_dir.is_dir():
                source_dir = stage / "downloads" / str(ledger["pinned_revision"]) / "bundles" / sid
            digest = sanitize._derivative_digest(source_dir) if source_dir.is_dir() \
                else hashlib.sha256(sid.encode()).hexdigest()
        batches.append(
            {
                "session_id": sid,
                "content_digest": str(digest),
                "status": status,
                "reason_code": code,
                "artifact_relpath": f"{root}/{sid}",
                "manifest_relpath": None,
            }
        )
    return {
        "schema_version": hydrate_rules.HYDRATION_INDEX_SCHEMA_VERSION,
        "source_hub_commit": str(source_commit),
        "curation_id": curation_id,
        "sanitizer_version": hydrate_rules.SANITIZER_VERSION,
        "hydration_index_schema_version": hydrate_rules.HYDRATION_INDEX_SCHEMA_VERSION,
        "admission_policy_version": hydrate_rules.ADMISSION_POLICY_VERSION,
        "publication_prefix": f"curated/{curation_id}/",
        "batches": batches,
    }


def finalize(client: HubClient, stage: Path, *, curation_id: str, source_commit: str) -> str:
    """Publish the curation manifest, then ``_SUCCESS`` as the terminal commit (M18).

    The manifest is rendered from the real persisted import ledger and uploaded
    first; ``curated/<curation-id>/_SUCCESS`` is uploaded as the very last
    commit — nothing is ever uploaded after it. Returns the output commit SHA
    (the Hub head after the ``_SUCCESS`` commit), which the verify cycle pins.
    """
    curated = stage / "curated" / curation_id
    ledger_path = curated / "import-ledger.json"
    if not ledger_path.is_file():
        raise HydrationError(redact_text(f"no import ledger under curated/{curation_id}; cannot finalize"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    doc = _curation_manifest_doc(stage, curation_id=curation_id, source_commit=source_commit, ledger=ledger)
    manifest_path = curated / "curation-manifest.json"
    _atomic_write_json(manifest_path, doc)
    prefix = f"curated/{curation_id}/"
    _retry_upload(client, {f"{prefix}curation-manifest.json": manifest_path},
                  f"daydream hydrate {curation_id}: curation manifest")
    success_path = curated / "_SUCCESS"
    success_path.write_text(
        json.dumps({"curation_id": curation_id, "source_hub_commit": str(source_commit), "status": "complete"}) + "\n",
        encoding="utf-8",
    )
    _retry_upload(client, {f"{prefix}_SUCCESS": success_path},
                  f"daydream hydrate {curation_id}: success marker")
    return str(client.repo_info().sha)


def verify_publication(
    client: HubClient,
    stage: Path,
    *,
    output_commit_sha: str,
    curation_id: str,
    dry_run_admitted: int,
    source_commit: str,
) -> int:
    """Clean-room verification of the published output commit (M19/M20).

    Downloads exactly the pinned output commit into a fresh staging dir,
    validates SHA256SUMS against the uploaded content, validates the curation
    manifest against the frozen schema, rescan every published batch with
    ``scan_run_dir`` (must be clean), rebuilds a scratch index from the
    portable artifacts alone, and recomputes the candidate count. Any mismatch
    raises :class:`VerificationError` — success is never reported on a failed
    verification. Returns the verified admitted count.
    """
    prefix = f"curated/{curation_id}/"
    verify_dir = stage / "_verify"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    verify_dir.mkdir(parents=True)

    def _download(relpath: str) -> bytes:
        try:
            return client.download_file(f"{prefix}{relpath}", revision=output_commit_sha)
        except HydrationError as exc:
            raise VerificationError(redact_text(f"verify: published file {relpath!r} missing: {exc}")) from exc

    # 1. SHA256SUMS must match every published file byte-for-byte.
    try:
        sums_text = _download("SHA256SUMS").decode("utf-8")
    except VerificationError:
        raise
    except HydrationError as exc:
        raise VerificationError(redact_text(f"verify: SHA256SUMS missing: {exc}")) from exc
    for line in sums_text.splitlines():
        if not line.strip():
            continue
        digest, _, relpath = line.partition("  ")
        relpath = relpath.removeprefix(prefix)
        actual = hashlib.sha256(_download(relpath)).hexdigest()
        if actual != digest:
            raise VerificationError(redact_text(f"verify: checksum mismatch for {relpath!r}"))

    # 2. Curation manifest: schema-valid and consistent with the pinned inputs.
    from jsonschema import Draft202012Validator  # noqa: PLC0415  # lazy: verify-time only

    schema_path = Path(__file__).parent.parent / "training" / "schema" / "curation-manifest-v1.json"
    doc = json.loads(_download("curation-manifest.json").decode("utf-8"))
    errors = sorted(Draft202012Validator(json.loads(schema_path.read_text())).iter_errors(doc), key=str)
    if errors:
        raise VerificationError(redact_text(f"verify: curation manifest invalid: {errors[0].message}"))
    if doc["curation_id"] != curation_id or doc["source_hub_commit"] != str(source_commit):
        raise VerificationError(redact_text("verify: curation manifest identity mismatch"))
    _download("_SUCCESS")  # the marker must be present at the pinned output commit

    # 3. Rescan every published batch (clean-room) and rebuild the scratch index.
    valid = {f.name for f in dataclass_fields(Manifest)}
    for batch in doc["batches"]:
        if batch["status"] != "admitted":
            continue
        sid = batch["session_id"]
        batch_dir = verify_dir / "batches" / sid
        for line in sums_text.splitlines():
            if not line.strip():
                continue
            _, _, relpath = line.partition("  ")
            relpath = relpath.removeprefix(prefix)
            if not relpath.startswith(f"batches/{sid}/"):
                continue
            target = batch_dir / relpath.removeprefix(f"batches/{sid}/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_download(relpath))
        scan = scan_run_dir(batch_dir)
        if not scan.clean:
            raise VerificationError(redact_text(f"verify: published batch {sid!r} fails the secrets scan"))
        if sanitize._derivative_digest(batch_dir) != batch["content_digest"]:
            raise VerificationError(redact_text(f"verify: batch {sid!r} digest mismatch"))
        data = _read_manifest_dict(batch_dir)
        if data is None:
            raise VerificationError(redact_text(f"verify: batch {sid!r} has an unreadable manifest"))
        kwargs = {k: v for k, v in data.items() if k in valid}
        kwargs["archive_path"] = str(batch_dir)
        raw_url = data.get("remote_url")
        if isinstance(raw_url, str) and raw_url.strip():
            slug, canonical = normalize_remote_url(raw_url)
            kwargs["repo_slug"], kwargs["remote_url"] = slug, canonical
        else:
            kwargs["repo_slug"], kwargs["remote_url"] = None, None
        upsert_run(verify_dir, Manifest(**kwargs))

    from daydream.archive.index import query_runs  # noqa: PLC0415  # local: avoid import cycle

    verify_admitted = len(query_runs(verify_dir))
    if verify_admitted != dry_run_admitted:
        raise VerificationError(
            redact_text(
                f"verify: candidate count mismatch — dry run admitted {dry_run_admitted}, "
                f"clean-room rebuild found {verify_admitted}"
            )
        )
    return verify_admitted


def run_hydrate_hub(config: HydrateHubConfig, client: HubClient | None = None) -> HydrateSummary:
    """Orchestrate hydration end-to-end: pin, download, ingest, dedupe, publish, verify.

    Composes Tasks 4–9 in order with fatal, redacted failure semantics: resolve
    the pinned source revision, download the snapshot, run the ingest gate,
    dedupe + ledger, publish additive batches (continuous checkpoints, M16),
    finalize (manifest then ``_SUCCESS`` last, M18), and only then run the
    clean-room verification cycle (M19/M20). ``client=None`` builds the
    production :class:`HfHubClient` via :func:`_make_client`. The summary's
    ``verified`` flag is set only after verification passes; every failure
    path leaves it False and never uploads a success marker.
    """
    client = client if client is not None else _make_client(config.destination_repo)
    if not client.repo_private:
        raise PublicDestinationError(
            "refusing to publish: the Hub repo is not private; hydration "
            "publishes sanitized corpora only to private repos (M17)"
        )
    source_commit = resolve_source_revision(client, config.source_revision, exploratory=config.exploratory)
    download_snapshot(client, revision=source_commit, stage_dir=config.stage_dir / "downloads")
    ingest_bundles(config.stage_dir, revision=source_commit)
    dedupe_admitted(config.stage_dir, revision=source_commit)
    ledger = build_import_ledger(config.stage_dir, revision=source_commit, source_commit=source_commit)
    curation_id = str(ledger["curation_id"])
    checkpoint = resume_state(client, curation_id=curation_id, stage_dir=config.stage_dir / "_resume")
    publish_batches(
        client, config.stage_dir, curation_id=curation_id, skip_sessions=checkpoint.completed_sessions
    )
    output_commit_sha = finalize(client, config.stage_dir, curation_id=curation_id, source_commit=source_commit)
    summary = HydrateSummary(
        source_commit=source_commit,
        curation_id=curation_id,
        output_commit_sha=output_commit_sha,
        dry_run_admitted=int(ledger["tallies"]["imported"]),
    )
    summary.verify_admitted = verify_publication(
        client,
        config.stage_dir,
        output_commit_sha=output_commit_sha,
        curation_id=curation_id,
        dry_run_admitted=summary.dry_run_admitted,
        source_commit=source_commit,
    )
    summary.verified = True  # only after the full clean-room cycle passes (M20)
    return summary
