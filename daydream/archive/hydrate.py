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
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from daydream.archive import hydrate_rules, sanitize
from daydream.archive.git_safe import normalize_remote_url
from daydream.archive.hydrate_rules import (
    REASON_CODE_BUNDLE_UNREADABLE,
    REASON_CODE_C5_EXCLUDED_REPO,
    REASON_CODE_C8_COPYLEFT_UNOPTED,
    REASON_CODE_IDENTITY_COLLISION,
    REASON_CODE_LICENSE_EVIDENCE_MISSING,
    REASON_CODE_PATH_TRAVERSAL,
    REASON_CODE_REPO_COMMIT_UNRESOLVED,
    REASON_CODE_REPO_IDENTITY_MISSING,
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


class NoSessionCandidatesError(HydrationError):
    """The snapshot looks like a run archive but contains no complete sessions."""


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
    revision = revision.strip()
    if _FULL_SHA_RE.fullmatch(revision.lower()):
        # Hex validation is case-insensitive and the Hub's canonical SHAs are
        # lowercase, so fold only inside the hex branches — a symbolic ref
        # (branch/tag name) is resolved case-sensitively below (M2 contract:
        # no silent case-folding of symbolic refs).
        revision = revision.lower()
        try:
            client.repo_info(revision=revision)  # verify it exists
        except HydrationError as exc:
            raise HydrationError(redact_text(str(exc))) from exc
        return revision

    if _HEX_PREFIX_RE.fullmatch(revision.lower()):
        prefix = revision.lower()
        list_revisions = getattr(client, "list_revisions", None)
        matches = (
            [r for r in list_revisions() if r.startswith(prefix)]
            if callable(list_revisions)
            else []
        )
        if len(matches) > 1:
            raise HydrationError(
                redact_text(f"ambiguous revision prefix {prefix!r}: {len(matches)} matching commits")
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

    def list_repo_files(self, revision: str | None = None) -> list[str]: ...

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
    discovered: int = 0
    run_shaped_manifests: int = 0
    incomplete_manifests: tuple[str, ...] = ()


_REQUIRED_SESSION_ARTIFACTS = frozenset(("manifest.json", "trajectory.json"))
_DERIVED_ARCHIVE_ROOTS = frozenset(("annotations", "curated"))
# Root names that are never valid session ids at depth 1, in either layout.
# ``bronze`` is the immutable raw-ingest tree (M10): hydration must never
# discover or stage anything under it.
_RESERVED_CANONICAL_ROOTS = frozenset(("annotations", "bronze", "bundle", "bundles", "curated"))


@dataclass(frozen=True)
class _DiscoveredSession:
    """One source-layout session identified by its manifest path."""

    session_id: str
    source_root: str
    layout: str


@dataclass(frozen=True)
class _Discovery:
    """Validated snapshot discovery and its normalized staging file map."""

    sessions: tuple[_DiscoveredSession, ...]
    normalized_paths: tuple[tuple[str, str], ...]  # (normalized, source)
    run_shaped_manifests: int
    canonical_manifests: int
    legacy_manifests: int
    incomplete_manifests: tuple[str, ...]


def _source_root_for_path(relpath: str) -> str | None:
    """Return the possible session root for an archive path, if any.

    The returned value is only a discovery hint. A path becomes a candidate
    source file only after a matching root-level or legacy manifest has been
    found and the required artifact set has been validated.
    """
    parts = PurePosixPath(relpath).parts
    if not parts or parts[0] in _DERIVED_ARCHIVE_ROOTS:
        return None
    if parts[0] == "bundles":
        if (
            len(parts) >= 3
            and _is_bare_segment(parts[1])
            and parts[1] not in _RESERVED_CANONICAL_ROOTS
        ):
            return f"bundles/{parts[1]}"
        return None
    if len(parts) >= 2 and _is_bare_segment(parts[0]) and parts[0] not in _RESERVED_CANONICAL_ROOTS:
        return parts[0]
    return None


def _manifest_source_session(relpath: str) -> _DiscoveredSession | None:
    """Recognize the two supported manifest layouts without reading content."""
    parts = PurePosixPath(relpath).parts
    if len(parts) == 2 and parts[1] == "manifest.json":
        session_id = parts[0]
        if _is_bare_segment(session_id) and session_id not in _RESERVED_CANONICAL_ROOTS:
            return _DiscoveredSession(session_id, session_id, "canonical")
    if (
        len(parts) == 3
        and parts[0] == "bundles"
        and parts[2] == "manifest.json"
        and _is_bare_segment(parts[1])
        and parts[1] not in _RESERVED_CANONICAL_ROOTS
    ):
        return _DiscoveredSession(parts[1], f"bundles/{parts[1]}", "legacy")
    return None


def _discover_snapshot(relpaths: list[str]) -> _Discovery:
    """Discover complete source sessions and map them to ``bundles/<id>/``.

    Discovery is based on manifest shape plus the required trajectory artifact,
    not on a broad directory prefix. Both source layouts normalize to the same
    internal staging tree; duplicate normalized paths are fatal rather than
    silently overwritten.
    """
    manifest_sessions: dict[str, _DiscoveredSession] = {}
    paths_by_root: dict[str, set[str]] = {}
    canonical_manifests = 0
    legacy_manifests = 0
    for relpath in relpaths:
        session = _manifest_source_session(relpath)
        if session is not None:
            if session.source_root in manifest_sessions:
                raise StageError(
                    redact_text(f"duplicate run-shaped manifest path for {session.source_root!r}")
                )
            manifest_sessions[session.source_root] = session
            if session.layout == "canonical":
                canonical_manifests += 1
            else:
                legacy_manifests += 1
        source_root = _source_root_for_path(relpath)
        if source_root is not None:
            paths_by_root.setdefault(source_root, set()).add(relpath)

    complete: list[_DiscoveredSession] = []
    incomplete: list[str] = []
    for source_root, session in sorted(manifest_sessions.items()):
        available = {
            PurePosixPath(path).relative_to(PurePosixPath(source_root)).as_posix()
            for path in paths_by_root.get(source_root, set())
        }
        missing = sorted(_REQUIRED_SESSION_ARTIFACTS - available)
        if missing:
            incomplete.append(f"{source_root} (missing {', '.join(missing)})")
        else:
            complete.append(session)

    session_id_counts = Counter(session.session_id for session in complete)
    duplicates = sorted(
        session_id for session_id, count in session_id_counts.items() if count > 1
    )
    if duplicates:
        raise StageError(
            redact_text(
                "multiple source layouts normalize to the same session id(s): "
                + ", ".join(repr(value) for value in duplicates)
            )
        )

    normalized_paths: dict[str, str] = {}
    for session in complete:
        source_root = session.source_root
        for source_path in sorted(paths_by_root[source_root]):
            suffix = PurePosixPath(source_path).relative_to(PurePosixPath(source_root)).as_posix()
            normalized = f"bundles/{session.session_id}/{suffix}"
            prior = normalized_paths.get(normalized)
            if prior is not None and prior != source_path:
                raise StageError(
                    redact_text(
                        f"source paths {prior!r} and {source_path!r} collide after normalization "
                        f"at {normalized!r}"
                    )
                )
            normalized_paths[normalized] = source_path

    return _Discovery(
        sessions=tuple(sorted(complete, key=lambda item: item.session_id)),
        normalized_paths=tuple(sorted(normalized_paths.items())),
        run_shaped_manifests=canonical_manifests + legacy_manifests,
        canonical_manifests=canonical_manifests,
        legacy_manifests=legacy_manifests,
        incomplete_manifests=tuple(incomplete),
    )


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


def _is_bare_segment(value: str) -> bool:
    """True when ``value`` is one safe path segment (no separators, no ``..``).

    Enforces the M4 trust boundary on Hub-derived session ids before they are
    joined into a filesystem path — the same boundary :func:`_validate_relpath`
    applies to download relpaths. An absolute, empty, ``.``/``..``, or
    separator-bearing value can never be a bare segment.
    """
    return bool(value) and value not in (".", "..") and "/" not in value and "\\" not in value


def download_snapshot(
    client: HubClient,
    *,
    revision: str,
    stage_dir: Path,
    expect: dict[str, str] | None = None,
) -> DownloadResult:
    """Resumable, content-addressed download of a pinned snapshot revision (issue #982 M3).

    Discovers complete sessions in the producer's canonical
    ``<session-id>/manifest.json`` + ``trajectory.json`` layout and the tested
    legacy ``bundles/<session-id>/...`` layout. Both are normalized to
    ``stage_dir/<revision>/bundles/<session-id>/...``. Derived ``curated/`` and
    ``annotations/`` trees and unrelated top-level files are excluded. Each
    staged artifact is recorded in a per-artifact ledger (normalized relpath,
    source relpath, sha256, size, fetched_at) in
    ``stage_dir/<revision>/_download_manifest.json``.

    Resume: an on-disk artifact whose sha256 matches the existing ledger record
    is skipped, not re-downloaded; missing or mismatched artifacts are fetched,
    re-hashed, and their records updated. ``expect`` —
    ``{source-or-normalized-relpath: sha256}``
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

    relpaths = list(client.list_repo_files(revision=revision))
    for relpath in relpaths:
        # Validate every Hub path, including ignored metadata and derived
        # output, before applying discovery filters. A malicious ignored path
        # must not become an escape hatch around the staging trust boundary.
        _validate_relpath(relpath, root)
    discovery = _discover_snapshot(relpaths)
    if not discovery.sessions:
        details = "; ".join(discovery.incomplete_manifests) or "required artifacts were not found"
        raise NoSessionCandidatesError(
            redact_text(
                f"source revision {revision!r} contains {discovery.run_shaped_manifests} "
                "run-shaped manifest(s) but discovery produced zero candidates "
                f"(canonical={discovery.canonical_manifests}, legacy={discovery.legacy_manifests}); "
                "expected <session-id>/manifest.json plus trajectory.json or "
                f"bundles/<session-id>/...; {details}"
            )
        )
    downloaded = 0
    skipped = 0
    digests: dict[str, str] = {}
    artifacts: list[dict[str, Any]] = []

    for normalized, source_relpath in discovery.normalized_paths:
        target = _validate_relpath(normalized, root)
        expected_sha = (expect or {}).get(source_relpath) or (expect or {}).get(normalized)
        try:
            if target.exists():
                existing = hashlib.sha256(target.read_bytes()).hexdigest()
                record_sha = records.get(normalized, {}).get("sha256")
                if expected_sha in (None, existing) and record_sha in (None, existing):
                    digests[normalized] = existing
                    skipped += 1
                    artifacts.append(
                        {
                            **records.get(normalized, {}),
                            "relpath": normalized,
                            "source_relpath": source_relpath,
                            "sha256": existing,
                        }
                    )
                    continue
            data = client.download_file(source_relpath, revision=revision)
        except HydrationError as exc:
            raise StageError(redact_text(str(exc))) from exc
        sha = hashlib.sha256(data).hexdigest()
        if expected_sha is not None and sha != expected_sha:
            raise StageError(
                redact_text(f"digest mismatch for {source_relpath!r}: expected {expected_sha}, got {sha}")
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        try:
            tmp.write_bytes(data)
            tmp.replace(target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise StageError(redact_text(f"write failed for {normalized!r}: {exc}")) from exc
        downloaded += 1
        digests[normalized] = sha
        artifacts.append(
            {
                "relpath": normalized,
                "source_relpath": source_relpath,
                "sha256": sha,
                "size": len(data),
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "revision": revision,
                "artifacts": artifacts,
                "candidate_sessions": [session.session_id for session in discovery.sessions],
                "discovery": {
                    "run_shaped_manifests": discovery.run_shaped_manifests,
                    "canonical_manifests": discovery.canonical_manifests,
                    "legacy_manifests": discovery.legacy_manifests,
                    "incomplete_manifests": list(discovery.incomplete_manifests),
                },
            },
            indent=2,
        )
    )
    return DownloadResult(
        downloaded=downloaded,
        skipped=skipped,
        digests=digests,
        discovered=len(discovery.sessions),
        run_shaped_manifests=discovery.run_shaped_manifests,
        incomplete_manifests=discovery.incomplete_manifests,
    )


def _read_manifest_dict(bundle_dir: Path) -> dict[str, Any] | None:
    """Read ``manifest.json`` from ``bundle_dir``; ``None`` when absent/unparseable."""
    path = bundle_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_manifest_field(data: dict[str, Any], key: str) -> Any:
    """Read a provenance field from a produced manifest, with flat fallback.

    The canonical producer nests ``git.remote_url`` / ``git.source_path`` /
    ``git.repo_slug`` under ``git.*`` (``Manifest.to_dict``); hand-built
    (test-stage / legacy flat) manifests carry the same keys top-level. Read
    the nested spelling first, then the flat fallback.
    """
    git = data.get("git")
    if isinstance(git, dict) and key in git:
        return git[key]
    return data.get(key)


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


def _download_discovery_block(stage: Path, revision: str) -> dict[str, Any]:
    """Read the discovery diagnostics block from the download manifest.

    Returns an empty dict when the manifest predates the discovery ledger
    (a manually staged legacy tree). Invalid JSON raises ``HydrationError``
    fail-closed, matching :func:`_discovered_session_ids`.
    """
    path = stage / "downloads" / str(revision) / "_download_manifest.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HydrationError(redact_text(f"invalid download discovery ledger {path}: {exc}")) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("discovery", {}), dict):
        raise HydrationError(redact_text(f"invalid download discovery ledger {path}"))
    discovery: dict[str, Any] = payload["discovery"]
    return discovery


def _discovered_session_ids(stage: Path, revision: str) -> list[str] | None:
    """Read the normalized candidate list written by :func:`download_snapshot`.

    ``None`` preserves compatibility with a manually staged legacy tree that
    predates the discovery ledger. A present list is authoritative, including
    an empty list, so stale normalized directories cannot become candidates on
    a later run.
    """
    path = stage / "downloads" / str(revision) / "_download_manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HydrationError(redact_text(f"invalid download discovery ledger {path}: {exc}")) from exc
    if not isinstance(payload, dict):
        raise HydrationError(redact_text(f"invalid download discovery ledger {path}"))
    if "candidate_sessions" not in payload:
        return None
    candidates = payload["candidate_sessions"]
    if not isinstance(candidates, list) or not all(
        isinstance(session_id, str) and _is_bare_segment(session_id) for session_id in candidates
    ):
        raise HydrationError(redact_text(f"invalid candidate session ids in {path}"))
    if len(set(candidates)) != len(candidates):
        raise HydrationError(redact_text(f"duplicate candidate session ids in {path}"))
    return list(candidates)


def ingest_bundles(stage: Path, *, revision: str) -> list[IngestResult]:
    """Run every staged bundle through the #981 ingest gate (issue #982 M4/M6).

    For each discovered session normalized under
    ``stage/downloads/<revision>/bundles/``:

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
    discovered_ids = _discovered_session_ids(stage, revision)
    if discovered_ids is None:
        bundle_items = (
            [(bundle_dir.name, bundle_dir) for bundle_dir in sorted(bundles_root.iterdir()) if bundle_dir.is_dir()]
            if bundles_root.is_dir()
            else []
        )
    else:
        bundle_items = [(session_id, bundles_root / session_id) for session_id in discovered_ids]
    results: list[IngestResult] = []
    for name, bundle_dir in bundle_items:
        data = _read_manifest_dict(bundle_dir)
        if data is None:
            results.append(IngestResult(name, "quarantined", REASON_CODE_BUNDLE_UNREADABLE))
            continue
        session_id = str(data.get("session_id") or name)
        if not _is_bare_segment(session_id):
            # M4: the manifest's session id is Hub-provided path data; reject
            # traversal/absolute ids before any write — the sanitize gate
            # (which mkdirs from the session id) never sees them.
            results.append(IngestResult(session_id, "quarantined", REASON_CODE_PATH_TRAVERSAL))
            continue
        raw_url = _read_manifest_field(data, "remote_url")
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


def _manifest_repo_slug(data: dict[str, Any]) -> str | None:
    """Repository identity from a session manifest (nested ``git.repo_slug`` or flat).

    A missing/blank slug stays ``None`` in the row — identity refusal is the
    admission gate's job (fail-closed there), never a substituted fallback.
    """
    raw = _read_manifest_field(data, "repo_slug")
    return raw if isinstance(raw, str) and raw.strip() else None


def _manifest_license_evidence(data: dict[str, Any]) -> dict[str, str] | None:
    """Declared license evidence (``spdx_id`` + ``source``) from a session manifest.

    Schema-shaped: only the two string fields the frozen curation-manifest-v1
    schema allows are carried; anything else (missing, blank, non-dict) is ``None``.
    """
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


def _session_identity(stage: Path, sid: str, revision: str, *, root: str, collision: bool) -> \
        tuple[str | None, dict[str, str] | None]:
    """Read ``repo_slug`` + ``license_evidence`` for a session from its manifest.

    Tries the derivative locations in the same precedence the digest derivation
    uses (runs/, quarantine conflict copy, moved rejected root, raw download
    bundle); ``(None, None)`` when no manifest is readable anywhere.
    """
    segment = sid if _is_bare_segment(sid) else hashlib.sha256(sid.encode()).hexdigest()
    candidates = [stage / "runs" / sid]
    if collision:
        candidates.append(stage / "quarantine" / f"{segment}.conflict")
    candidates += [stage / root / segment,
                   stage / "downloads" / str(revision) / "bundles" / segment]
    for candidate in candidates:
        data = _read_manifest_dict(candidate)
        if data is not None:
            return _manifest_repo_slug(data), _manifest_license_evidence(data)
    return None, None


def _repo_commit_unresolved_sessions(stage: Path) -> set[str]:
    """Session ids whose enrichment recorded an unresolvable repo commit.

    Reads the ``repo_commit_unresolved`` rows of ``stage/_enrich/evidence.jsonl``
    (written by the enrichment stage, which always precedes the license gate):
    the repo slug was identified but no resolved Git repository commit could be
    pinned, so the gate records such evidence-missing rejections under the
    specific stable code instead of the generic one.
    """
    from daydream.archive.license_enrich import (  # noqa: PLC0415  # local: avoid import cycle at module load
        _ENRICH_CACHE_NAME,
        _ENRICH_DIR,
    )

    path = stage / _ENRICH_DIR / _ENRICH_CACHE_NAME
    unresolved: set[str] = set()
    if not path.is_file():
        return unresolved
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("status") != REASON_CODE_REPO_COMMIT_UNRESOLVED:
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
    """Per-repo license admission gate over the admitted derivatives (issue #1080).

    Runs after the existing gates (ingest -> dedupe -> fixture exclusion): each
    admitted session's ``repo_slug`` + declared license evidence are resolved
    into an immutable per-repo decision via
    :func:`daydream.training.corpus_v2.license.resolve_repo_decision` (C5
    exclusion list first, then policy + opt-in). A ``rejected`` decision moves
    the derivative to ``stage/excluded/<sid>/`` exactly like the fixture
    exclusion path and records a stable-code exclusion in the dedupe ledger, so
    the import-ledger accounting invariant (admitted + rejected = input) holds
    by construction — every session still lands in exactly one bucket.

    Fail-closed: a missing ``license_policy_path`` raises ``ValueError`` before
    any gate work — never downgraded to a warning. Apart from the excluded-
    directory move the gate is pure w.r.t. its inputs, so decisions are
    replay-identical. When enrichment identified the repo but could not pin a
    resolved Git repository commit (its cache recorded
    ``repo_commit_unresolved``), the evidence-missing rejection is recorded
    under that more specific stable code — the same rejection, bucketed
    identically — so the ledger surfaces the commit-unresolved path. Returns
    the rejected ``(session_id, reason_code)`` pairs.
    """
    if not license_policy_path:
        raise ValueError(
            "license admission gate requires license_policy_path (fail-closed): "
            "no license policy file was provided"
        )
    from daydream.training.corpus_v2.license import (  # noqa: PLC0415  # local: avoid import cycle at module load
        load_license_policy,
        resolve_repo_decision,
    )

    policy, _digest = load_license_policy(license_policy_path)
    curated = _pre_identity_dir(stage, str(revision))
    ledger_path = _dedupe_dir(stage, curated.name) / "dedupe.jsonl"
    rejected: list[tuple[str, str]] = []
    unresolved = _repo_commit_unresolved_sessions(stage)
    runs_dir = stage / "runs"
    if runs_dir.is_dir():
        for derivative in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            data = _read_manifest_dict(derivative)
            if data is None:
                raise HydrationError(
                    redact_text(f"admitted derivative {derivative.name} has an unreadable manifest")
                )
            sid = str(data.get("session_id") or derivative.name)
            if not _is_bare_segment(sid):
                raise HydrationError(
                    redact_text(f"admitted derivative {derivative.name} has an unsafe session id {sid!r}")
                )
            decision = resolve_repo_decision(
                _manifest_repo_slug(data) or "",
                _manifest_license_evidence(data),
                policy,
                allow_copyleft,
            )
            if decision.status != "rejected" or decision.reason_code is None:
                continue
            reason_code = decision.reason_code
            if reason_code == REASON_CODE_LICENSE_EVIDENCE_MISSING and sid in unresolved:
                # Enrichment identified the repo but could not pin a resolved
                # Git commit; the ledger records the specific stable code so
                # the repo_commit_unresolved rejection path is exercisable
                # (it folds into the evidence-missing bucket).
                reason_code = REASON_CODE_REPO_COMMIT_UNRESOLVED
            _move_dir(derivative, stage / "excluded" / sid)
            _append_dedupe_entry(
                ledger_path,
                {"session_id": sid, "status": "excluded", "reason_code": reason_code,
                 "content_digest": None, "revision": str(revision), "at": _utc_now()},
            )
            rejected.append((sid, reason_code))
    if rejected:
        rebuild_index(stage)  # excluded derivatives leave the staging index immediately
    return rejected


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
        # ``daydream`` provenance is a nested dict in produced manifests; the
        # index expects the executable-provenance object, so it is dropped from
        # the hydrated rebuild (never coerced into a Manifest field).
        kwargs = {
            k: v for k, v in data.items()
            if k in valid and k not in rewritten and k != "daydream"
        }
        raw_url = _read_manifest_field(data, "remote_url")
        if isinstance(raw_url, str) and raw_url.strip():
            slug, canonical = normalize_remote_url(raw_url)
            kwargs["repo_slug"] = slug
            kwargs["remote_url"] = canonical
        else:
            kwargs["repo_slug"] = None
            kwargs["remote_url"] = None
        kwargs["source_path"] = _staging_local_source_path(_read_manifest_field(data, "source_path"), stage)
        kwargs["archive_path"] = str(derivative)
        upsert_run(stage, Manifest(**kwargs))


def build_resolution_map(
    stage: Path, *, source_commit: str, repo_commits: Mapping[str, str],
) -> dict[str, Any]:
    """Build the deferred-clone repository resolution map (issue #982 M5).

    From the admitted index rows under ``stage/runs/``, group sessions by the
    ``normalize_remote_url`` slug. Each entry carries ``repo_slug``,
    ``pinned_sha`` (the resolved Git repository commit for that repo, from
    ``repo_commits`` — the enrichment cache's per-slug resolution; the Hub
    dataset revision is never recorded as a repository commit), and the
    contributing ``session_ids``. Rows with no resolvable slug (no remote, or a
    non-allowlisted host) land under ``map["unavailable"]`` as a list of
    session ids, as do rows whose slug carries no full 40-hex resolved commit —
    a reported outcome, never a raw-URL fallback, never a fabricated revision,
    and never a clone.

    ``source_commit`` (the Hub dataset revision the snapshot was hydrated
    from) is accepted for ledger bookkeeping at the call site but is
    deliberately never recorded in any map entry.

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
        raw_url = _read_manifest_field(data, "remote_url")
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


_EXCLUSION_LIST_PATH = (
    Path(__file__).resolve().parents[1] / "training" / "schema" / "exclusion.txt"
)


def _curated_dir(stage: Path, source_commit: str, binding: dict[str, Any] | None = None) -> Path:
    """Curated prefix for a source commit: ``stage/curated/<curation-id>/``.

    Issue #1094: with a post-gate ``binding`` (from
    :func:`resolve_curation_identity`) the curation id is the v2 derivation,
    which binds the policy digest/version, exact copyleft opt-ins, the
    exclusions digest, the resolved per-repo decisions digest, and the license
    distribution digest. Without a binding the historical v1 derivation (four
    inputs) is kept for pre-identity staging and historical prefixes only —
    publications are always keyed by the v2 id.
    """
    if binding is not None:
        cid = hydrate_rules.derive_curation_id_v2(
            source_commit,
            str(binding["policy_digest"]),
            str(binding["policy_version"]),
            binding["allow_copyleft"],
            str(binding["exclusions_digest"]),
            str(binding["decisions_digest"]),
            str(binding["distribution_digest"]),
        )
    else:
        cid = hydrate_rules.derive_curation_id(
            source_commit,
            hydrate_rules.SANITIZER_VERSION,
            hydrate_rules.HYDRATION_INDEX_SCHEMA_VERSION,
            hydrate_rules.ADMISSION_POLICY_VERSION,
        )
    return stage / "curated" / cid


def _pre_identity_dir(stage: Path, source_commit: str) -> Path:
    """Pre-identity staging location for the dedupe ledger key (issue #1094).

    The v2 curation id cannot exist until after the license gate (it binds the
    resolved decisions), so every pre-gate ledger writer (dedupe, enrichment
    restamp, gate rejections) keys the VM-local dedupe state by a
    source-commit-scoped v1-shaped staging id (Assumption A4). Staging-
    internal only — never published.
    """
    return _curated_dir(stage, source_commit)


def _exclusions_digest() -> str:
    """sha256 over the sorted stable exclusion-codes string of the pinned C5
    exclusion list (``training/schema/exclusion.txt`` bytes) — a bound identity
    input, so editing the list changes the curation id."""
    codes = sorted(
        line.strip() for line in
        _EXCLUSION_LIST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
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
    """Post-gate identity binding (issue #1094): pure function of the gate
    outputs plus the pinned inputs, so replay over identical evidence is
    byte-identical.

    Computes, over every resolved per-repo decision of the post-gate admitted
    runs plus the gate-rejected derivatives under ``stage/excluded/<sid>``
    (the gate is a pure function of policy + evidence, so re-resolving the
    survivors reproduces the gate's decisions exactly):

    - ``exclusions_digest``: sha256 over the sorted stable exclusion-codes
      string of the pinned C5 list (see :func:`_exclusions_digest`);
    - ``decisions_digest``: sha256 over the canonical sorted
      ``repo_slug\\tstatus\\treason_code\\tspdx_id\\n`` lines, deduped per
      repo slug;
    - ``distribution_digest``: sha256 over the sorted ``spdx_id\\tcount\\n``
      license-distribution lines.

    The binding carries the policy digest/version and the exact copyleft
    opt-ins so ``derive_curation_id_v2`` can bind all of them.
    """
    from daydream.training.corpus_v2.license import (  # noqa: PLC0415  # local: avoid import cycle
        resolve_repo_decision,
    )

    decisions: dict[str, tuple[str, str | None, str | None]] = {}
    runs_dir = stage / "runs"
    if runs_dir.is_dir():
        for derivative in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            data = _read_manifest_dict(derivative)
            if data is None:
                raise HydrationError(
                    redact_text(f"admitted derivative {derivative.name} has an unreadable manifest")
                )
            decision = resolve_repo_decision(
                _manifest_repo_slug(data) or "",
                _manifest_license_evidence(data),
                policy,
                allow_copyleft,
            )
            decisions[str(decision.repo_slug)] = (
                str(decision.status), decision.reason_code, decision.spdx_id,
            )
    # Gate-rejected derivatives were moved out of runs/ (apply_license_gate ->
    # stage/excluded/<sid>); the binding must cover them too, or a rejected
    # repo's adjudication drift (e.g. repo_commit_unresolved ->
    # c8_copyleft_unopted) would change the republished ledger/excluded bytes
    # under a byte-identical curation id. Re-resolve the excluded session's
    # decision and bind it under the *recorded* gate reason code (the gate
    # records a more specific stable code, like repo_commit_unresolved, than
    # plain re-resolution over absent evidence would produce). Non-license
    # exclusions (ingest/fixture) were never adjudicated by the license gate
    # and stay out of the license decisions digest.
    recorded_excluded = _load_dedupe_ledger(
        _dedupe_dir(stage, _pre_identity_dir(stage, str(source_commit)).name) / "dedupe.jsonl"
    )
    license_codes = {
        REASON_CODE_C5_EXCLUDED_REPO,
        REASON_CODE_C8_COPYLEFT_UNOPTED,
        REASON_CODE_LICENSE_EVIDENCE_MISSING,
        REASON_CODE_REPO_IDENTITY_MISSING,
        REASON_CODE_REPO_COMMIT_UNRESOLVED,
    }
    excluded_dir = stage / "excluded"
    if excluded_dir.is_dir():
        for derivative in sorted(p for p in excluded_dir.iterdir() if p.is_dir()):
            data = _read_manifest_dict(derivative)
            if data is None:
                raise HydrationError(
                    redact_text(
                        f"excluded derivative {derivative.name} has an unreadable manifest"
                    )
                )
            sid = str(data.get("session_id") or derivative.name)
            entry = recorded_excluded.get(sid) or {}
            code = entry.get("reason_code")
            if code not in license_codes:
                continue  # never a license-gate decision, never in the digest
            decision = resolve_repo_decision(
                _manifest_repo_slug(data) or "",
                _manifest_license_evidence(data),
                policy,
                allow_copyleft,
            )
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
    """Derive the v2 curation id from the post-gate policy binding (issue #1094).

    Must be called only after :func:`apply_license_gate`: the binding is a
    pure function of the gate outputs (resolved per-repo decisions + license
    distribution) plus the pinned inputs (policy digest/version, opt-ins, C5
    exclusions digest). Returns the binding plus the derived ``curation_id``.
    Fail-closed: a missing ``license_policy_path`` raises before any
    derivation — identity is never computed without a pinned policy.
    """
    if not license_policy_path:
        raise HydrationError(
            "curation identity requires license_policy_path (fail-closed): "
            "no license policy file was provided"
        )
    from daydream.training.corpus_v2.license import (  # noqa: PLC0415  # local: avoid import cycle
        load_license_policy,
    )

    policy, policy_digest = load_license_policy(license_policy_path)
    binding = _policy_binding(stage, str(source_commit), policy, policy_digest, allow_copyleft)
    binding["curation_id"] = _curated_dir(stage, str(source_commit), binding=binding).name
    return binding


def _dedupe_dir(stage: Path, curation_id: str) -> Path:
    """Internal dedupe state (ledger + admitted baselines) for a curation.

    Deliberately outside ``stage/curated/<curation-id>/``: the append-only
    dedupe ledger and the admitted-baseline copies are VM-local bookkeeping and
    are never part of the published file set (M13 publication list), so they
    are never re-uploaded on additive runs.
    """
    return stage / "_dedupe" / curation_id


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


def restamp_admitted_digests(stage: Path, *, revision: str) -> None:
    """Refresh admitted-baseline content and dedupe-ledger digests after
    enrichment (issue #1094).

    Enrichment rewrites admitted manifests under ``stage/runs/`` *after*
    ``dedupe_admitted`` recorded their ``content_digest``; the curation
    manifest pins that digest and the clean-room verify recomputes
    ``_derivative_digest`` over the published bytes — so without a restamp
    every enriched session fails verification. For each admitted session
    whose derivative digest changed, refresh the admitted baseline copy and
    append a new ``admitted`` ledger entry carrying the enriched digest
    (latest-entry-wins, same convention as the dedupe pass itself)."""
    from daydream.archive import sanitize  # noqa: PLC0415  # local: avoid import cycle

    runs_dir = stage / "runs"
    if not runs_dir.is_dir():
        return
    curated = _pre_identity_dir(stage, revision)
    dedupe_dir = _dedupe_dir(stage, curated.name)
    baseline_root = dedupe_dir / "admitted"
    ledger_path = dedupe_dir / "dedupe.jsonl"
    for derivative in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        sid = derivative.name
        digest = sanitize._derivative_digest(derivative)
        baseline = baseline_root / sid
        if baseline.is_dir() and sanitize._derivative_digest(baseline) == digest:
            continue  # unchanged by enrichment; ledger digest stays authoritative
        if baseline.is_dir():
            shutil.rmtree(baseline)
        shutil.copytree(derivative, baseline)
        _append_dedupe_entry(
            ledger_path,
            {"session_id": sid, "status": "admitted", "reason_code": None,
             "content_digest": digest, "revision": str(revision), "at": _utc_now()},
        )


def _move_dir(source: Path, target: Path) -> None:
    """Move a directory, replacing any prior occupant of ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(source), str(target))


def _deep_artifact_evidence(derivative: Path, data: dict[str, Any]) -> dict[str, object] | None:
    """Synthesize M9 revalidation evidence from a real bundle's deep artifacts.

    Produced manifests never carry a top-level ``deep_artifacts`` key; the
    evidence lives in the bundle's ``deep/*.json`` sidecars (copied from
    ``.daydream/deep``, keyed by stem) plus the manifest's own derived
    ``phase_states`` / ``fix_failures`` / ``archive_status`` fields. ``None``
    when no evidence exists — the M9 gate then excludes the bundle.
    """
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
       idempotent (no duplicate ledger entry, no duplicate index row). A
       derivative matching a *previously* recorded ``admitted`` digest (the
       pristine pre-enrichment content of an already-enriched session) is an
       idempotent same-stage-dir re-ingest, not a collision: the published
       baseline is restored and the admission re-recorded, so an interrupted-
       resume or an idempotent republish of the same revision never downgrades
       a published batch to quarantined.

    Admitted sessions are indexed (idempotent ``upsert_run``); an unreadable
    manifest on an admitted derivative is fatal. The dedupe ledger records
    every decision as ``{session_id, status, content_digest, reason_code}``.
    """
    revision = str(revision)
    curated = _pre_identity_dir(stage, revision)
    dedupe_dir = _dedupe_dir(stage, curated.name)
    ledger_path = dedupe_dir / "dedupe.jsonl"
    baseline_root = dedupe_dir / "admitted"
    # The latest *admitted* digest is the durable collision key: a later
    # collision entry must not let a re-run re-admit the mutated derivative
    # over the published baseline (M7 durability across re-runs).
    admitted_digests = _latest_admitted_digests(ledger_path)
    ever_admitted = _ever_admitted_digests(ledger_path)
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
            if not _is_bare_segment(sid):
                # M4: the manifest's session id must never be joined into a
                # staging path (excluded/, quarantine/, baseline restore).
                raise HydrationError(
                    redact_text(f"admitted derivative {name} has an unsafe session id {sid!r}")
                )

            # M8: fixture exclusion, pre-dedupe, stable codes.
            try:
                codes = hydrate_rules.fixture_exclusion_codes(derivative)
            except (OSError, ValueError):
                codes = [REASON_CODE_BUNDLE_UNREADABLE]
            if not codes and "pipeline_status" in data:
                # M9: revalidate the legacy field against the bundle's real
                # evidence; evidence-absent never succeeds (produced manifests
                # never carry a top-level ``deep_artifacts`` key).
                verdict = hydrate_rules.legacy_pipeline_status(
                    data.get("pipeline_status"),
                    _deep_artifact_evidence(derivative, data),
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
            if admitted_digests.get(sid) not in (None, digest):
                if digest in ever_admitted.get(sid, set()):
                    # Idempotent re-ingest of the same source content: the
                    # derivative differs from the *latest* admitted digest only
                    # because this pipeline's own enrichment rewrote its
                    # manifest after admission (restamp refreshed the baseline
                    # + ledger to the enriched digest). Restore the published
                    # baseline and re-record the admission — never a collision,
                    # so an idempotent same-stage-dir re-run (interrupted-
                    # resume or a fixed-path republish of the same revision)
                    # does not downgrade a published batch to quarantined.
                    baseline = baseline_root / sid
                    if not baseline.is_dir():
                        raise HydrationError(
                            redact_text(
                                f"cannot restore admitted baseline for {sid}: "
                                "no admitted baseline exists"
                            )
                        )
                    shutil.rmtree(derivative)
                    shutil.copytree(baseline, runs_dir / sid)
                    _append_dedupe_entry(
                        ledger_path,
                        {"session_id": sid, "status": "admitted", "reason_code": None,
                         "content_digest": sanitize._derivative_digest(baseline),
                         "revision": revision, "at": _utc_now()},
                    )
                    result.admitted += 1
                    continue
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


def _latest_admitted_digests(path: Path) -> dict[str, str | None]:
    """Latest *admitted* dedupe entry per session id (collision entries never win).

    The admitted derivative is never overwritten (M7), so a later collision
    entry must not replace the admitted content digest in the import ledger —
    the published batch content is still the baseline.
    """
    latest: dict[str, str | None] = {}
    if not path.is_file():
        return latest
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if isinstance(entry, dict) and entry.get("session_id") \
                    and entry.get("status") == "admitted":
                latest[str(entry["session_id"])] = entry.get("content_digest")
    except (OSError, ValueError):
        return {}
    return latest


def _ever_admitted_digests(path: Path) -> dict[str, set[str]]:
    """Every *admitted* content digest ever recorded per session id.

    Richer than :func:`_latest_admitted_digests`: an idempotent same-stage-dir
    re-run re-ingests the *pristine* pre-enrichment content, whose digest was
    safely recorded (as ``admitted``) before enrichment rewrote the manifest
    and :func:`restamp_admitted_digests` refreshed the baseline to the enriched
    digest. That pristine digest is a known-good published baseline — never a
    collision — so it must survive in the re-ingest set even after a later
    restamp moved the latest-admitted digest on. Collision-entry digests are
    deliberately excluded: a mutated derivative stays a collision on every
    later run (M7 durability).
    """
    ever: dict[str, set[str]] = {}
    if not path.is_file():
        return ever
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if isinstance(entry, dict) and entry.get("session_id") \
                    and entry.get("status") == "admitted":
                digest = entry.get("content_digest")
                if isinstance(digest, str) and digest:
                    ever.setdefault(str(entry["session_id"]), set()).add(digest)
    except (OSError, ValueError):
        return {}
    return ever


def admission_summary_buckets(
    entries: Iterable[tuple[str, str | None]],
) -> dict[str, int]:
    """Pure four-bucket license summary over ``(session_id, reason_code)``
    admission decisions (issue #1080 S2).

    ``None`` counts as admitted; the stable rejection codes map to the
    human buckets (``repo_identity_missing`` and ``repo_commit_unresolved``
    fold into ``license_evidence_missing`` — missing identity or an
    unresolvable repo commit is missing evidence for the license gate). Any
    other code is not a license-gate decision and
    raises: the bucket sum equals the license-gate session count by
    construction (M8).
    """
    code_map = {
        REASON_CODE_C5_EXCLUDED_REPO: "c5_excluded",
        REASON_CODE_C8_COPYLEFT_UNOPTED: "c8_copyleft_unopted",
        REASON_CODE_LICENSE_EVIDENCE_MISSING: "license_evidence_missing",
        REASON_CODE_REPO_IDENTITY_MISSING: "license_evidence_missing",
        REASON_CODE_REPO_COMMIT_UNRESOLVED: "license_evidence_missing",
    }
    buckets: dict[str, int] = {
        "admitted": 0,
        "c5_excluded": 0,
        "c8_copyleft_unopted": 0,
        "license_evidence_missing": 0,
    }
    for _sid, code in entries:
        if code is None:
            buckets["admitted"] += 1
        elif code in code_map:
            buckets[code_map[code]] += 1
        else:
            raise ValueError(
                f"license admission summary: {code!r} is not a license-gate "
                "reason code — the summary buckets only partition license decisions"
            )
    return buckets


def license_admission_summary(ledger: Mapping[str, Any]) -> dict[str, int]:
    """Human admission summary derived from the built import ledger.

    Imported sessions count as admitted; rejections count only when they
    carry a license-gate reason code (ingest/fixture rejections were never
    adjudicated by the license gate and are skipped).
    """
    entries: list[tuple[str, str | None]] = [
        (str(item["session_id"]), None) for item in ledger.get("imported", [])
    ]
    license_codes = {
        REASON_CODE_C5_EXCLUDED_REPO,
        REASON_CODE_C8_COPYLEFT_UNOPTED,
        REASON_CODE_LICENSE_EVIDENCE_MISSING,
        REASON_CODE_REPO_IDENTITY_MISSING,
        REASON_CODE_REPO_COMMIT_UNRESOLVED,
    }
    for item in ledger.get("rejections", []):
        code = item.get("reason_code")
        if code in license_codes:
            entries.append((str(item["session_id"]), str(code)))
    return admission_summary_buckets(entries)


def license_admission_by_repo(
    stage: Path, ledger: Mapping[str, Any]
) -> dict[str, dict[str, int]]:
    """Per-repo license admission counts derived from the built import ledger
    (issue #1094 Task 8).

    Same adjudicated population as :func:`license_admission_summary` (imported
    sessions plus license-gate rejections; ingest/fixture rejections were
    never adjudicated by the license gate and are skipped), grouped by the
    repo slug read from the session's manifest via ``_session_identity``
    (enrichment wrote the resolved evidence into that same manifest, so this
    is the slug the gate decided on). Sessions without a readable slug bucket
    under ``"unresolved"``. Unknown reason codes raise exactly as
    :func:`admission_summary_buckets` does — the buckets are value-free slugs
    and counts, never URLs or paths.
    """
    revision = str(ledger["pinned_revision"])
    license_codes = {
        REASON_CODE_C5_EXCLUDED_REPO,
        REASON_CODE_C8_COPYLEFT_UNOPTED,
        REASON_CODE_LICENSE_EVIDENCE_MISSING,
        REASON_CODE_REPO_IDENTITY_MISSING,
        REASON_CODE_REPO_COMMIT_UNRESOLVED,
    }
    entries: list[tuple[str, str | None]] = [
        (str(item["session_id"]), None) for item in ledger.get("imported", [])
    ]
    for item in ledger.get("rejections", []):
        code = item.get("reason_code")
        if code in license_codes:
            entries.append((str(item["session_id"]), str(code)))
    by_repo: dict[str, dict[str, int]] = {}
    for sid, code in entries:
        # Imported sessions still live under stage/runs/<sid> (checked first);
        # license-gate rejections were moved to stage/excluded/<sid>.
        slug, _evidence = _session_identity(
            stage, sid, revision, root="excluded", collision=False
        )
        buckets = by_repo.setdefault(slug or "unresolved", {
            "admitted": 0,
            "c5_excluded": 0,
            "c8_copyleft_unopted": 0,
            "license_evidence_missing": 0,
        })
        for bucket, count in admission_summary_buckets([(sid, code)]).items():
            buckets[bucket] += count
    return by_repo


def build_import_ledger(
    stage: Path,
    *,
    revision: str,
    source_commit: str,
    binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and persist the value-free admission ledger (issue #982 M11).

    Composes the ingest results, the dedupe ledger, and the staging index into
    ``stage/curated/<curation-id>/import-ledger.json``. Every rejection entry
    carries only stable reason codes, session ids, and content digests — never
    a raw URL, path, or any matched secret value. The write is atomic; a
    failure propagates (fatal semantics) so a partial ledger can never claim
    success.

    Issue #1094: pass the post-gate ``binding`` (from
    :func:`resolve_curation_identity`) so the ledger — and everything
    published from it onward — is keyed by the v2 curation id. The dedupe
    ledger is still read from the pre-identity staging location (Assumption
    A4); only the curated output prefix moves to the v2 id.
    """
    revision = str(revision)
    source_commit = str(source_commit)
    curated = _curated_dir(stage, source_commit, binding=binding)
    dedupe_ledger = _dedupe_dir(stage, _pre_identity_dir(stage, source_commit).name) / "dedupe.jsonl"
    recorded = _load_dedupe_ledger(dedupe_ledger)
    admitted_digests = _latest_admitted_digests(dedupe_ledger)

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
        entry = recorded.get(sid) or {}
        reason_code = raw_result.get("reason_code")
        if raw_result.get("status") == "admitted":
            # A current ingest admission can be turned into an exclusion or
            # collision by the dedupe pass; that decision is the authoritative
            # current outcome and must not be listed as imported as well.
            if entry.get("status") == "excluded":
                excluded.append({"session_id": sid, "reason_code": entry.get("reason_code")})
            elif entry.get("status") == "collision":
                quarantined.append({"session_id": sid, "reason_code": entry.get("reason_code")})
            else:
                imported.append(
                    {
                        "session_id": sid,
                        "content_digest": admitted_digests.get(
                            sid, entry.get("content_digest")
                        ),
                    }
                )
        elif raw_result.get("status") == "quarantined":
            quarantined.append({"session_id": sid, "reason_code": reason_code})
        else:
            raise HydrationError(redact_text(f"unknown ingest status for candidate {sid!r}"))

    rejections = [
        {
            "session_id": str(entry["session_id"]),
            "reason_code": entry.get("reason_code"),
            "content_digest": (recorded.get(str(entry["session_id"]), {}) or {}).get("content_digest"),
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
        "generated_at": _utc_now(),
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


def _policy_binding_record(binding: dict[str, Any]) -> str:
    """Canonical ``policy-binding.json`` record text (issue #1094 task 7).

    Byte-canonical: sorted keys, compact canonical JSON, trailing newline —
    comparison against the remote record is bytewise, so the record must never
    depend on dict ordering or whitespace. Carries the full binding: policy
    digest/version, sorted casefolded copyleft opt-ins (matching
    :func:`hydrate_rules.derive_curation_id_v2`'s canonicalization, so a
    differently-cased spelling of the same logical opt-in yields byte-identical
    records), exclusions digest, resolved per-repo decisions digest, license
    distribution digest.
    """
    record = {
        "policy_digest": str(binding["policy_digest"]),
        "policy_version": str(binding["policy_version"]),
        "allow_copyleft": sorted(str(slug).casefold() for slug in binding["allow_copyleft"]),
        "exclusions_digest": str(binding["exclusions_digest"]),
        "resolved_decisions_digest": str(binding["decisions_digest"]),
        "distribution_digest": str(binding["distribution_digest"]),
        "schema_version": "2",
    }
    return json.dumps(record, sort_keys=True) + "\n"


def check_prefix_binding(
    client: HubClient, *, curation_id: str, binding: dict[str, Any],
    allow_unbound_resume: bool = False,
) -> None:
    """Fail closed when a curated prefix was published under a different
    policy binding (issue #1094 task 7).

    Downloads ``curated/<curation-id>/policy-binding.json`` from the
    destination repo and compares it bytewise against the current run's
    canonical binding record. A differing record raises
    :class:`HydrationError` ("conflicting policy binding"); an absent record
    on a prefix that already has published batches but no resume ledger is a
    pre-v2 legacy prefix and also fails closed with a distinct legacy message
    (legacy prefixes are never republished under the new scheme). An absent
    record on a fresh prefix — or on a prefix whose resume ledger shows an
    interrupted v2 run that died between ``publish_batches`` and ``finalize``
    (``allow_unbound_resume``, used by :func:`run_hydrate_hub`'s pre-publish
    check and by :func:`finalize`) — proceeds.

    Runs strictly before any upload, so a conflict uploads zero bytes. Errors
    name the prefix and digest fields only — never policy file contents or
    credentials.
    """
    prefix = f"curated/{curation_id}/"
    current = _policy_binding_record(binding).encode("utf-8")
    try:
        remote = client.download_file(f"{prefix}policy-binding.json")
    except HubDownloadError:
        remote = None  # no binding record yet (fresh or legacy prefix)
    except HydrationError as exc:
        raise HydrationError(
            redact_text(f"cannot read the published policy binding under {prefix}: {exc}")
        ) from exc
    if remote is not None:
        if remote == current:
            return
        detail = ""
        try:
            remote_digest = json.loads(remote.decode("utf-8")).get("policy_digest", "?")
            detail = f" (remote policy_digest={remote_digest}, current policy_digest={binding['policy_digest']})"
        except (ValueError, UnicodeDecodeError):
            detail = " (remote record is not a readable binding record)"
        raise HydrationError(
            redact_text(
                f"conflicting policy binding under {prefix}: the prefix was "
                "published under a different policy; refuse to publish "
                f"(fail-closed){detail}"
            )
        )
    # Absent record: fresh prefix, interrupted v2 run, or pre-v2 legacy prefix.
    repo_files = client.list_repo_files()
    has_batches = any(p.startswith(f"{prefix}batches/") for p in repo_files)
    if not has_batches:
        return  # fresh prefix: nothing published yet
    has_ledger = f"{prefix}resume/ledger.jsonl" in set(repo_files)
    if allow_unbound_resume and has_ledger:
        return  # interrupted v2 run; finalize is about to publish the record
    raise HydrationError(
        redact_text(
            f"conflicting policy binding under {prefix}: the prefix has "
            "published batches but no policy-binding record (pre-v2 legacy "
            "prefix); refuse to publish (fail-closed)"
        )
    )


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
    Resumability is bounded by the policy binding (issue #1094): the caller
    must run :func:`check_prefix_binding` first — a prefix whose published
    ``policy-binding.json`` differs from the current binding is never
    republished, additively or otherwise.

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


def _repo_commits_from_enrichment_cache(stage: Path) -> dict[str, str]:
    """Per-slug resolved Git repository commits from the enrichment cache.

    Groups the ``resolved`` rows of ``stage/_enrich/evidence.jsonl`` (written
    by the enrichment stage) by provenance slug, latest row per slug winning.
    Only resolver-produced full 40-hex commits qualify — the Hub dataset
    revision is never consulted and never recorded as a repository commit
    (issue #1094).
    """
    from daydream.archive.license_enrich import (  # noqa: PLC0415  # local: avoid import cycle at module load
        _ENRICH_CACHE_NAME,
        _ENRICH_DIR,
    )

    path = stage / _ENRICH_DIR / _ENRICH_CACHE_NAME
    commits: dict[str, str] = {}
    if not path.is_file():
        return commits
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("status") != "resolved":
            continue
        slug = entry.get("repo_slug")
        commit = entry.get("repo_commit")
        if isinstance(slug, str) and isinstance(commit, str) and _FULL_SHA_RE.fullmatch(commit):
            commits[slug] = commit
    return commits


def _write_resolution_map(stage: Path, curated: Path) -> None:
    """Materialize ``resolution-map.json`` under the curated prefix when absent.

    Rebuilt from the staging index (data only, no clones); an existing file —
    from a prior publish of the same curation id — is left untouched so the
    published map stays stable and additive. Per-slug ``pinned_sha`` values
    come from the enrichment cache's resolved Git repository commits.
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
    cmap = build_resolution_map(
        stage,
        source_commit=source_commit or "unknown",
        repo_commits=_repo_commits_from_enrichment_cache(stage),
    )
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
        if not _is_bare_segment(sid):
            raise StageError(redact_text(f"refusing resume session id {sid!r}: traversal"))
        batch_dir = stage_dir / "batches" / sid
        for rel in batch_relpaths:
            data = client.download_file(f"{batch_prefix}{rel}")
            target = _validate_relpath(rel, batch_dir)
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
            info = self._api.repo_info(self._repo_id, revision=revision, repo_type="dataset")
        except Exception as exc:  # mapped, never swallowed
            raise HubDownloadError(f"repo_info failed for {self._repo_id}: {exc}") from exc
        return RepoInfo(sha=str(info.sha), private=bool(info.private))

    def list_repo_files(self, revision: str | None = None) -> list[str]:
        try:
            return list(self._api.list_repo_files(self._repo_id, revision=revision, repo_type="dataset"))
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
    # License admission gate (issue #1080): the policy file is the digest-pinned
    # versioned artifact; allow_copyleft is the exact-slug C8 opt-in set.
    license_policy_path: str | None = None
    allow_copyleft: frozenset[str] = frozenset()


@dataclass
class HydrateSummary:
    """Outcome of one ``run_hydrate_hub`` invocation; ``verified`` gates success."""

    source_commit: str
    curation_id: str
    output_commit_sha: str | None = None
    dry_run_discovered: int = 0
    dry_run_admitted: int = 0
    dry_run_rejected: int = 0
    dry_run_incomplete_manifests: tuple[str, ...] = ()
    verify_admitted: int = 0
    verified: bool = False
    # Issue #1080 S2: four-bucket license admission summary (admitted /
    # c5-excluded / copyleft-unopted / evidence-missing, with
    # repo_identity_missing and repo_commit_unresolved folded into the
    # evidence-missing bucket) over the import ledger; empty when no license
    # policy was configured.
    license_admission: dict[str, int] = field(default_factory=dict)


def _curation_manifest_doc(
    stage: Path, *, curation_id: str, source_commit: str, ledger: dict[str, Any]
) -> dict[str, Any]:
    """Render the portable curation manifest (schema v1) from the real ledger (M12/M18)."""
    batches: list[dict[str, Any]] = []
    for entry in ledger.get("imported", []):
        sid = str(entry["session_id"])
        repo_slug, license_evidence = _session_identity(
            stage, sid, str(ledger["pinned_revision"]), root="excluded", collision=False
        )
        batches.append(
            {
                "session_id": sid,
                "content_digest": str(entry.get("content_digest") or ""),
                "status": "admitted",
                "reason_code": None,
                "artifact_relpath": f"batches/{sid}",
                "manifest_relpath": f"batches/{sid}/manifest.json",
                "repo_slug": repo_slug,
                "license_evidence": license_evidence,
            }
        )
    for rejection in ledger.get("rejections", []):
        sid = str(rejection["session_id"])
        code = rejection.get("reason_code")
        status = "excluded" if code in hydrate_rules.EXCLUSION_CODES else "quarantined"
        root = "excluded" if status == "excluded" else "quarantine"
        # Rejected bundles are never published, so the relpath names the actual
        # staging tree (portable/relative): dedupe moves identity collisions to
        # ``quarantine/<sid>.conflict``, everything else to ``<root>/<sid>``.
        # A non-bare session id is a traversal attempt — reference a derived
        # hash segment instead of the raw value and never touch the filesystem.
        segment = sid if _is_bare_segment(sid) else hashlib.sha256(sid.encode()).hexdigest()
        collision = code == hydrate_rules.REASON_CODE_IDENTITY_COLLISION
        relpath = f"{root}/{segment}.conflict" if collision else f"{root}/{segment}"
        digest = rejection.get("content_digest")
        if not isinstance(digest, str) or not digest:
            # Rejected bundles carry no derivative digest; derive one from the
            # rejected copy on staging (or the session id when it was moved).
            candidates: list[Path] = []
            if collision:
                candidates.append(stage / "quarantine" / f"{segment}.conflict")
            candidates.append(stage / root / segment)
            candidates.append(stage / "downloads" / str(ledger["pinned_revision"]) / "bundles" / segment)
            source_dir = next((c for c in candidates if c.is_dir()), None)
            digest = sanitize._derivative_digest(source_dir) if source_dir is not None \
                else hashlib.sha256(sid.encode()).hexdigest()
        repo_slug, license_evidence = _session_identity(
            stage, sid, str(ledger["pinned_revision"]), root=root, collision=collision
        )
        batches.append(
            {
                "session_id": sid,
                "content_digest": str(digest),
                "status": status,
                "reason_code": code,
                "artifact_relpath": relpath,
                "manifest_relpath": None,
                "repo_slug": repo_slug,
                "license_evidence": license_evidence,
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


def finalize(client: HubClient, stage: Path, *, curation_id: str, source_commit: str,
             binding: dict[str, Any]) -> str:
    """Publish the curation manifest and refreshed checksums (M18).

    The manifest is rendered from the real persisted import ledger and uploaded
    with ``SHA256SUMS`` over the final published file set. The canonical
    ``policy-binding.json`` record (issue #1094) is written locally and
    re-checked against the remote prefix immediately before the manifest
    upload, so a conflicting republication can never add bytes. The ``_SUCCESS``
    marker is deliberately NOT uploaded here: it is published only after the
    clean-room verification cycle passes (verify-before-success), so a
    verification failure can never leave a published "complete" marker.
    Returns the output commit SHA (the Hub head after the manifest commit),
    which the verify cycle pins.
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
    # Issue #1094: pin the policy binding into the published prefix and fail
    # closed on any conflicting remote record before the manifest commit.
    binding_path = curated / "policy-binding.json"
    binding_path.write_text(_policy_binding_record(binding), encoding="utf-8")
    check_prefix_binding(client, curation_id=curation_id, binding=binding, allow_unbound_resume=True)
    # SHA256SUMS must cover the *final* published file set — including the
    # curation manifest rendered just above, which can change between runs
    # (e.g. a newly recorded identity collision). Refresh it here so the
    # checksums always match the bytes the verify cycle will pin. ``_SUCCESS``
    # is never covered: the marker does not exist at the verify commit (it is
    # published only after verification), and a stale local marker must never
    # leak into the pinned checksums on a re-run.
    final_relpaths = [
        p.relative_to(curated).as_posix()
        for p in curated.rglob("*") if p.is_file() and p.name != "_SUCCESS"
    ]
    checksums = "".join(
        f"{hashlib.sha256((curated / p).read_bytes()).hexdigest()}  {prefix}{p}\n"
        for p in final_relpaths if p != "SHA256SUMS"
    )
    (curated / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    _retry_upload(
        client,
        {
            f"{prefix}curation-manifest.json": manifest_path,
            f"{prefix}SHA256SUMS": curated / "SHA256SUMS",
            f"{prefix}policy-binding.json": binding_path,
        },
        f"daydream hydrate {curation_id}: curation manifest + checksums",
    )
    return str(client.repo_info().sha)


def _publish_success_marker(
    client: HubClient, stage: Path, *, curation_id: str, source_commit: str
) -> None:
    """Publish ``curated/<curation-id>/_SUCCESS`` as the terminal commit (M18).

    Called only after :func:`verify_publication` passes — every failure path
    leaves it unpublished, so a published ``_SUCCESS`` always means the
    clean-room cycle verified the output commit. Returns nothing; the caller
    re-reads the hub head as the final output commit SHA.
    """
    curated = stage / "curated" / curation_id
    prefix = f"curated/{curation_id}/"
    success_path = curated / "_SUCCESS"
    success_path.write_text(
        json.dumps({"curation_id": curation_id, "source_hub_commit": str(source_commit), "status": "complete"}) + "\n",
        encoding="utf-8",
    )
    _retry_upload(client, {f"{prefix}_SUCCESS": success_path},
                  f"daydream hydrate {curation_id}: success marker")


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
    # The _SUCCESS marker is *not* expected at this commit: it is published by
    # run_hydrate_hub only after this cycle passes (verify-before-success), so
    # a failed verification can never leave a published "complete" marker.

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
        kwargs = {k: v for k, v in data.items() if k in valid and k != "daydream"}
        kwargs["archive_path"] = str(batch_dir)
        raw_url = _read_manifest_field(data, "remote_url")
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
    the pinned source revision from the source repo, download the snapshot,
    run the ingest gate, dedupe + ledger, fail closed on any conflicting
    published policy binding, publish additive batches (continuous
    checkpoints, M16), finalize (policy binding + curation manifest +
    refreshed checksums, M18), run the clean-room verification cycle against
    that pinned commit (M19/M20), and only after it passes publish the
    ``_SUCCESS`` marker as the very last
    commit — a verification failure never leaves a published success marker.
    ``client=None`` builds two production :class:`HfHubClient` instances via
    :func:`_make_client` (one for the source snapshot repo, one for the
    destination publication repo). The summary's ``verified`` flag is set only
    after verification passes; every failure path leaves it False and never
    uploads a success marker.
    """
    # Issue #1094 defense-in-depth: the CLI refuses a non-dry publication
    # without --license-policy; the orchestrator re-checks so a future caller
    # bypassing the CLI still fails closed before any publication.
    if config.license_policy_path is None:
        raise HydrationError(
            "run_hydrate_hub requires license_policy_path for any publication "
            "path (fail-closed)"
        )
    # Two clients: the source repo guards the pinned snapshot, the destination
    # repo receives the published output. Tests inject one FakeHub for both.
    source_client = client if client is not None else _make_client(config.source_repo)
    dest_client = client if client is not None else _make_client(config.destination_repo)
    if not dest_client.repo_private:
        raise PublicDestinationError(
            "refusing to publish: the Hub repo is not private; hydration "
            "publishes sanitized corpora only to private repos (M17)"
        )
    source_commit = resolve_source_revision(
        source_client, config.source_revision, exploratory=config.exploratory
    )
    download_snapshot(source_client, revision=source_commit, stage_dir=config.stage_dir / "downloads")
    ingest_bundles(config.stage_dir, revision=source_commit)
    dedupe_admitted(config.stage_dir, revision=source_commit)
    # With the policy now required on every non-dry publication path, the
    # gate and its admission summary always run (issue #1094).
    # Issue #1094: enrichment fills legacy records' missing license_evidence
    # from an authorized immutable source before the gate. The staging cache
    # makes same-VM re-runs decision-identical; the published copy under the
    # curated prefix is the pinned evidence record of this curation (audit +
    # replay harnesses). Fresh-VM production runs re-enrich from the live
    # resolver, so upstream drift surfaces as a new v2 curation id, never a
    # silent rewrite of a previous curation.
    from daydream.archive.license_enrich import (  # noqa: PLC0415  # local: avoid import cycle at module load
        _make_license_resolver,
        enrich_license_evidence,
        publish_enrichment_cache,
    )

    enrich_license_evidence(
        config.stage_dir, revision=source_commit, resolver=_make_license_resolver(),
    )
    # Enrichment may have rewritten admitted manifests; refresh the dedupe
    # baselines/ledger digests so the published content identity matches the
    # enriched content (the clean-room verify recomputes the digest).
    restamp_admitted_digests(config.stage_dir, revision=source_commit)
    # Issue #1080: the per-repo license gate runs after the existing gates;
    # apply_license_gate itself refuses (ValueError, fail-closed) on a
    # missing policy input (unreachable-but-documented now that the
    # orchestrator pre-checks).
    apply_license_gate(
        config.stage_dir,
        revision=source_commit,
        license_policy_path=config.license_policy_path,
        allow_copyleft=config.allow_copyleft,
    )
    # Issue #1094: the v2 curation id is derived post-gate from the resolved
    # policy binding (policy digest/version, opt-ins, exclusions digest,
    # resolved per-repo decisions, license distribution) — everything from
    # the ledger onward is keyed by it. The digest captured from
    # load_license_policy is the binding's policy_digest; policy_version is
    # the policy file's own version.
    binding = resolve_curation_identity(
        config.stage_dir,
        source_commit=source_commit,
        license_policy_path=config.license_policy_path,
        allow_copyleft=config.allow_copyleft,
    )
    curation_id = str(binding["curation_id"])
    # The enrichment cache is copied into the *v2* curated prefix as the
    # pinned evidence record of this curation (audit + replay harnesses).
    publish_enrichment_cache(
        config.stage_dir, revision=source_commit,
        curated_dir=config.stage_dir / "curated" / curation_id,
    )
    ledger = build_import_ledger(
        config.stage_dir, revision=source_commit, source_commit=source_commit, binding=binding,
    )
    license_admission = license_admission_summary(ledger)
    checkpoint = resume_state(dest_client, curation_id=curation_id, stage_dir=config.stage_dir / "_resume")
    # Issue #1094 task 7: fail closed on a conflicting published policy
    # binding before a single byte is uploaded (additive republication under
    # a prefix bound to a different policy is refused here). The resumed run
    # died between publish_batches and finalize: its prefix has published
    # batches and the resume ledger but no policy-binding record yet —
    # allow_unbound_resume lets that exact window proceed (the v2 id is
    # deterministic, so a reachable prefix always carries this run's binding;
    # pre-v2 legacy prefixes have no resume ledger and still fail closed).
    check_prefix_binding(
        dest_client, curation_id=curation_id, binding=binding,
        allow_unbound_resume=True,
    )
    publish_batches(
        dest_client, config.stage_dir, curation_id=curation_id, skip_sessions=checkpoint.completed_sessions
    )
    # finalize pins the verify commit (manifest + checksums); _SUCCESS is
    # uploaded only after the clean-room cycle passes (M18/M20).
    output_commit_sha = finalize(
        dest_client, config.stage_dir, curation_id=curation_id, source_commit=source_commit,
        binding=binding,
    )
    summary = HydrateSummary(
        source_commit=source_commit,
        curation_id=curation_id,
        output_commit_sha=output_commit_sha,
        dry_run_discovered=int(ledger["tallies"]["discovered"]),
        dry_run_admitted=int(ledger["tallies"]["imported"]),
        dry_run_rejected=int(ledger["tallies"]["rejections"]),
        dry_run_incomplete_manifests=tuple(ledger["tallies"]["incomplete_manifests"]),
        license_admission=license_admission,
    )
    summary.verify_admitted = verify_publication(
        dest_client,
        config.stage_dir,
        output_commit_sha=output_commit_sha,
        curation_id=curation_id,
        dry_run_admitted=summary.dry_run_admitted,
        source_commit=source_commit,
    )
    _publish_success_marker(
        dest_client, config.stage_dir, curation_id=curation_id, source_commit=source_commit
    )
    summary.output_commit_sha = str(dest_client.repo_info().sha)  # head after the _SUCCESS commit
    summary.verified = True  # only after the full clean-room cycle passes (M20)
    return summary
